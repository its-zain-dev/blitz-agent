//! Minimal LLM clients: OpenAI-compatible, Google Gemini, Anthropic, plus a scripted mock.

use std::collections::VecDeque;
use std::sync::Mutex;
use std::time::Duration;

use anyhow::{anyhow, bail, Context, Result};
use serde_json::{json, Value};

use crate::util::truncate_chars;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    User,
    Assistant,
}

#[derive(Debug, Clone)]
pub struct Message {
    pub role: Role,
    pub content: String,
}

impl Message {
    pub fn user(s: impl Into<String>) -> Self {
        Self {
            role: Role::User,
            content: s.into(),
        }
    }

    pub fn assistant(s: impl Into<String>) -> Self {
        Self {
            role: Role::Assistant,
            content: s.into(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum ProviderKind {
    Openai,
    Gemini,
    Anthropic,
    Mock,
}

struct ProviderDefaults {
    key_envs: &'static [&'static str],
    model: &'static str,
    base_url: &'static str,
}

fn defaults(kind: ProviderKind) -> ProviderDefaults {
    match kind {
        ProviderKind::Openai => ProviderDefaults {
            key_envs: &["OPENAI_API_KEY"],
            model: "gpt-4.1-mini",
            base_url: "https://api.openai.com/v1",
        },
        ProviderKind::Gemini => ProviderDefaults {
            key_envs: &["GEMINI_API_KEY", "GOOGLE_API_KEY"],
            model: "gemini-2.5-flash",
            base_url: "https://generativelanguage.googleapis.com",
        },
        ProviderKind::Anthropic => ProviderDefaults {
            key_envs: &["ANTHROPIC_API_KEY"],
            model: "claude-sonnet-4-5",
            base_url: "https://api.anthropic.com",
        },
        ProviderKind::Mock => ProviderDefaults {
            key_envs: &[],
            model: "mock",
            base_url: "",
        },
    }
}

pub struct LlmClient {
    kind: ProviderKind,
    model: String,
    api_key: String,
    base_url: String,
    json_mode: bool,
    http: reqwest::Client,
    mock: Mutex<VecDeque<String>>,
}

impl LlmClient {
    /// Build a real provider client. API keys are read from the environment.
    pub fn new(
        kind: ProviderKind,
        model: Option<String>,
        base_url: Option<String>,
        json_mode: bool,
    ) -> Result<Self> {
        let d = defaults(kind);
        let api_key = if kind == ProviderKind::Mock {
            String::new()
        } else {
            d.key_envs
                .iter()
                .find_map(|k| std::env::var(k).ok().filter(|v| !v.trim().is_empty()))
                .ok_or_else(|| anyhow!("missing API key: set {}", d.key_envs.join(" or ")))?
        };
        let base_url = base_url
            .or_else(|| {
                if kind == ProviderKind::Openai {
                    std::env::var("OPENAI_BASE_URL").ok()
                } else {
                    None
                }
            })
            .unwrap_or_else(|| d.base_url.to_string());
        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(300))
            .build()
            .context("building HTTP client")?;
        Ok(Self {
            kind,
            model: model.unwrap_or_else(|| d.model.to_string()),
            api_key,
            base_url: base_url.trim_end_matches('/').to_string(),
            json_mode,
            http,
            mock: Mutex::new(VecDeque::new()),
        })
    }

    /// A scripted client that returns `responses` in order (for tests and demos).
    pub fn mock(responses: Vec<String>) -> Self {
        Self {
            kind: ProviderKind::Mock,
            model: "mock".into(),
            api_key: String::new(),
            base_url: String::new(),
            json_mode: false,
            http: reqwest::Client::new(),
            mock: Mutex::new(responses.into()),
        }
    }

    pub fn kind(&self) -> ProviderKind {
        self.kind
    }

    pub fn model(&self) -> &str {
        &self.model
    }

    pub async fn complete(&self, system: &str, msgs: &[Message]) -> Result<String> {
        match self.kind {
            ProviderKind::Openai => self.openai(system, msgs).await,
            ProviderKind::Gemini => self.gemini(system, msgs).await,
            ProviderKind::Anthropic => self.anthropic(system, msgs).await,
            ProviderKind::Mock => self.next_mock(),
        }
    }

    fn next_mock(&self) -> Result<String> {
        let next = {
            let mut q = self.mock.lock().unwrap();
            q.pop_front()
        };
        next.ok_or_else(|| anyhow!("mock provider: no more scripted responses"))
    }

    async fn openai(&self, system: &str, msgs: &[Message]) -> Result<String> {
        let mut messages = vec![json!({"role": "system", "content": system})];
        for m in msgs {
            let role = if m.role == Role::User { "user" } else { "assistant" };
            messages.push(json!({"role": role, "content": m.content}));
        }
        let mut body = json!({"model": self.model, "messages": messages});
        if self.json_mode {
            body["response_format"] = json!({"type": "json_object"});
        }
        let url = format!("{}/chat/completions", self.base_url);
        let auth = format!("Bearer {}", self.api_key);
        let v = self.post(&url, &body, &[("Authorization", auth)]).await?;
        v["choices"][0]["message"]["content"]
            .as_str()
            .map(|s| s.to_string())
            .ok_or_else(|| {
                anyhow!(
                    "unexpected OpenAI response: {}",
                    truncate_chars(&v.to_string(), 500)
                )
            })
    }

    async fn gemini(&self, system: &str, msgs: &[Message]) -> Result<String> {
        let contents: Vec<Value> = msgs
            .iter()
            .map(|m| {
                let role = if m.role == Role::User { "user" } else { "model" };
                json!({"role": role, "parts": [{"text": m.content}]})
            })
            .collect();
        let mut body = json!({
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": contents,
        });
        if self.json_mode {
            body["generationConfig"] = json!({"responseMimeType": "application/json"});
        }
        let url = format!(
            "{}/v1beta/models/{}:generateContent",
            self.base_url, self.model
        );
        let v = self
            .post(&url, &body, &[("x-goog-api-key", self.api_key.clone())])
            .await?;
        let parts = v["candidates"][0]["content"]["parts"]
            .as_array()
            .ok_or_else(|| {
                anyhow!(
                    "unexpected Gemini response: {}",
                    truncate_chars(&v.to_string(), 500)
                )
            })?;
        let text: String = parts
            .iter()
            .filter(|p| !p["thought"].as_bool().unwrap_or(false))
            .filter_map(|p| p["text"].as_str())
            .collect::<Vec<_>>()
            .join("");
        if text.trim().is_empty() {
            bail!("Gemini returned an empty response");
        }
        Ok(text)
    }

    async fn anthropic(&self, system: &str, msgs: &[Message]) -> Result<String> {
        let messages: Vec<Value> = msgs
            .iter()
            .map(|m| {
                let role = if m.role == Role::User { "user" } else { "assistant" };
                json!({"role": role, "content": m.content})
            })
            .collect();
        let body = json!({
            "model": self.model,
            "max_tokens": 8192,
            "system": system,
            "messages": messages,
        });
        let url = format!("{}/v1/messages", self.base_url);
        let v = self
            .post(
                &url,
                &body,
                &[
                    ("x-api-key", self.api_key.clone()),
                    ("anthropic-version", "2023-06-01".to_string()),
                ],
            )
            .await?;
        let blocks = v["content"].as_array().ok_or_else(|| {
            anyhow!(
                "unexpected Anthropic response: {}",
                truncate_chars(&v.to_string(), 500)
            )
        })?;
        let text: String = blocks
            .iter()
            .filter(|b| b["type"].as_str() == Some("text"))
            .filter_map(|b| b["text"].as_str())
            .collect::<Vec<_>>()
            .join("");
        if text.trim().is_empty() {
            bail!("Anthropic returned an empty response");
        }
        Ok(text)
    }

    /// POST JSON with retries on 429 / 5xx / network errors (exponential backoff).
    async fn post(&self, url: &str, body: &Value, headers: &[(&str, String)]) -> Result<Value> {
        const ATTEMPTS: u32 = 4;
        let mut last_err = anyhow!("request not sent");
        for attempt in 0..ATTEMPTS {
            let mut req = self.http.post(url).json(body);
            for (k, v) in headers {
                req = req.header(*k, v.as_str());
            }
            match req.send().await {
                Ok(resp) => {
                    let status = resp.status();
                    let text = resp.text().await.unwrap_or_default();
                    if status.is_success() {
                        return serde_json::from_str(&text).with_context(|| {
                            format!("invalid JSON from provider: {}", truncate_chars(&text, 300))
                        });
                    }
                    last_err = anyhow!(
                        "HTTP {} from provider: {}",
                        status,
                        truncate_chars(&text, 800)
                    );
                    let retryable = status.as_u16() == 429 || status.is_server_error();
                    if !retryable {
                        break;
                    }
                }
                Err(e) => last_err = anyhow!("request failed: {e}"),
            }
            if attempt + 1 < ATTEMPTS {
                tokio::time::sleep(Duration::from_millis(1000 * 2u64.pow(attempt))).await;
            }
        }
        Err(last_err)
    }
}
