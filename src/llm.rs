//! Minimal LLM clients: Antigravity CLI (`agy`), Google Gemini, OpenAI-compatible, Anthropic, plus mock.

use std::collections::VecDeque;
use std::fmt;
use std::path::PathBuf;
use std::process::Stdio;
use std::sync::Mutex;
use std::time::Duration;

use anyhow::{anyhow, bail, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tokio::process::Command;

use crate::config::AppConfig;
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

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, clap::ValueEnum, Default,
)]
#[serde(rename_all = "lowercase")]
pub enum ProviderKind {
    #[default]
    Agy,
    Gemini,
    Openai,
    Anthropic,
    Mock,
}

impl ProviderKind {
    pub fn next(self) -> Self {
        match self {
            Self::Agy => Self::Gemini,
            Self::Gemini => Self::Openai,
            Self::Openai => Self::Anthropic,
            Self::Anthropic => Self::Agy,
            Self::Mock => Self::Agy,
        }
    }

    pub fn prev(self) -> Self {
        match self {
            Self::Agy => Self::Anthropic,
            Self::Gemini => Self::Agy,
            Self::Openai => Self::Gemini,
            Self::Anthropic => Self::Openai,
            Self::Mock => Self::Agy,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Agy => "Antigravity (agy account)",
            Self::Gemini => "Google Gemini",
            Self::Openai => "OpenAI / Compatible",
            Self::Anthropic => "Anthropic Claude",
            Self::Mock => "Mock (Scripted)",
        }
    }

    pub fn short(self) -> &'static str {
        match self {
            Self::Agy => "AGY",
            Self::Gemini => "GEMINI",
            Self::Openai => "OPENAI",
            Self::Anthropic => "ANTHROPIC",
            Self::Mock => "MOCK",
        }
    }
}

impl fmt::Display for ProviderKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.short())
    }
}

struct ProviderDefaults {
    key_envs: &'static [&'static str],
    model: &'static str,
    base_url: &'static str,
}

fn defaults(kind: ProviderKind) -> ProviderDefaults {
    match kind {
        ProviderKind::Agy => ProviderDefaults {
            key_envs: &[],
            model: "gemini-4-argon-high",
            base_url: "",
        },
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
    agy_conv_id: Mutex<Option<String>>,
}

impl LlmClient {
    /// Build a client from `AppConfig`.
    pub fn from_config(cfg: &AppConfig) -> Result<Self> {
        Self::with_key(
            cfg.provider,
            Some(cfg.active_model().to_string()),
            Some(cfg.active_api_key().to_string()),
            Some(cfg.active_base_url().to_string()),
            cfg.json_mode,
        )
    }

    /// Build a real provider client. API keys are read from the environment or `AppConfig`.
    pub fn new(
        kind: ProviderKind,
        model: Option<String>,
        base_url: Option<String>,
        json_mode: bool,
    ) -> Result<Self> {
        Self::with_key(kind, model, None, base_url, json_mode)
    }

    pub fn with_key(
        kind: ProviderKind,
        model: Option<String>,
        explicit_key: Option<String>,
        base_url: Option<String>,
        json_mode: bool,
    ) -> Result<Self> {
        let d = defaults(kind);
        let saved = AppConfig::load();
        let api_key = if matches!(kind, ProviderKind::Mock | ProviderKind::Agy) {
            String::new()
        } else {
            explicit_key
                .filter(|v| !v.trim().is_empty())
                .or_else(|| {
                    d.key_envs
                        .iter()
                        .find_map(|k| std::env::var(k).ok().filter(|v| !v.trim().is_empty()))
                })
                .or_else(|| {
                    let k = match kind {
                        ProviderKind::Gemini => &saved.gemini_api_key,
                        ProviderKind::Openai => &saved.openai_api_key,
                        ProviderKind::Anthropic => &saved.anthropic_api_key,
                        ProviderKind::Agy | ProviderKind::Mock => "",
                    };
                    (!k.trim().is_empty()).then(|| k.to_string())
                })
                .ok_or_else(|| {
                    anyhow!(
                        "missing API key for {}: set {} or configure it in Settings (F2 / /settings)",
                        kind.label(),
                        d.key_envs.join(" or ")
                    )
                })?
        };
        let base_url = base_url
            .filter(|u| !u.trim().is_empty())
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
            model: model
                .filter(|m| !m.trim().is_empty())
                .unwrap_or_else(|| d.model.to_string()),
            api_key,
            base_url: base_url.trim_end_matches('/').to_string(),
            json_mode,
            http,
            mock: Mutex::new(VecDeque::new()),
            agy_conv_id: Mutex::new(None),
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
            agy_conv_id: Mutex::new(None),
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
            ProviderKind::Agy => self.agy(system, msgs).await,
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

    /// Delegate inference to the user's locally authenticated `agy` CLI in print mode.
    async fn agy(&self, system: &str, msgs: &[Message]) -> Result<String> {
        let bin = find_agy_bin();
        let existing_conv = self.agy_conv_id.lock().unwrap().clone();

        let prompt = if existing_conv.is_some() {
            let latest = msgs.last().map(|m| m.content.as_str()).unwrap_or("");
            format!(
                "REMINDER: Do NOT invoke any of your own built-in tools. \
                 Respond with ONLY ONE valid Blitz JSON plan object.\n\n{latest}"
            )
        } else {
            let mut convo = String::new();
            for m in msgs {
                let tag = if m.role == Role::User {
                    "USER"
                } else {
                    "ASSISTANT"
                };
                convo.push_str(&format!("[{tag}]\n{}\n\n", m.content));
            }
            format!(
                "CRITICAL SESSION RULE:\n\
                 You are acting strictly as the JSON planner brain for Blitz.\n\
                 DO NOT call ANY of your own built-in tools (no run_command, no read_file, \
                 no edit_file, no write_file, no search, no web search).\n\
                 Output ONLY ONE valid JSON object following the Blitz schema below.\n\n\
                 === BLITZ SYSTEM PROMPT ===\n{system}\n\n\
                 === CONVERSATION ===\n{convo}"
            )
        };

        let mut cmd = Command::new(&bin);
        cmd.arg("--disable-slash-commands")
            .arg("--mode")
            .arg("plan")
            .arg("--model")
            .arg(&self.model)
            .arg("--output-format")
            .arg("json");
        if let Some(cid) = &existing_conv {
            cmd.arg("--conversation").arg(cid);
        }
        cmd.arg(format!("-p={prompt}"))
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);

        let output = cmd.output().await.with_context(|| {
            format!(
                "failed to launch '{}'. Make sure Antigravity CLI (`agy`) is installed and logged in",
                bin.display()
            )
        })?;

        let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();

        if let Ok(v) = serde_json::from_str::<Value>(&stdout) {
            if let Some(cid) = v["conversation_id"]
                .as_str()
                .filter(|s| !s.trim().is_empty())
            {
                *self.agy_conv_id.lock().unwrap() = Some(cid.to_string());
            }
            if let Some(err) = v["error"].as_str().filter(|s| !s.trim().is_empty()) {
                bail!("agy error: {err}");
            }
            if let Some(resp) = v["response"].as_str() {
                if !resp.trim().is_empty() {
                    return Ok(resp.to_string());
                }
            }
        }

        if !output.status.success() {
            let detail = if !stderr.is_empty() { &stderr } else { &stdout };
            bail!(
                "agy exited with {}: {}",
                output.status,
                truncate_chars(detail, 500)
            );
        }
        if stdout.is_empty() {
            bail!(
                "agy returned empty output (stderr: {})",
                truncate_chars(&stderr, 300)
            );
        }
        Ok(stdout)
    }

    async fn openai(&self, system: &str, msgs: &[Message]) -> Result<String> {
        let mut messages = vec![json!({"role": "system", "content": system})];
        for m in msgs {
            let role = if m.role == Role::User {
                "user"
            } else {
                "assistant"
            };
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
                let role = if m.role == Role::User {
                    "user"
                } else {
                    "model"
                };
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
                let role = if m.role == Role::User {
                    "user"
                } else {
                    "assistant"
                };
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

fn find_agy_bin() -> PathBuf {
    if let Ok(home) = std::env::var("HOME") {
        let local = PathBuf::from(home).join(".local").join("bin").join("agy");
        if local.exists() {
            return local;
        }
    }
    PathBuf::from("agy")
}
