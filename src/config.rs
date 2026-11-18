//! Persistent user configuration (`~/.config/blitz/config.toml`) and runtime settings.

use std::fmt;
use std::path::PathBuf;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::llm::ProviderKind;
use crate::plan::Limits;

/// Controls how aggressively the agent predicts and chains future tool calls.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, clap::ValueEnum, Default)]
#[serde(rename_all = "lowercase")]
pub enum GuessMode {
    /// Always predict 1-3 outcomes on every step, chain multi-step trees, use `{{stdout:id}}`
    /// placeholders, and finish optimistically inside prediction branches.
    #[default]
    Aggressive,
    /// Predict high-confidence outcomes and use optimistic finish when confident.
    Balanced,
    /// Only predict read-only / verification follow-ups; never chain file writes or mutations.
    Conservative,
    /// Run tools in parallel (width only), without speculative prediction trees.
    Off,
}

impl GuessMode {
    pub fn next(self) -> Self {
        match self {
            Self::Aggressive => Self::Balanced,
            Self::Balanced => Self::Conservative,
            Self::Conservative => Self::Off,
            Self::Off => Self::Aggressive,
        }
    }

    pub fn prev(self) -> Self {
        match self {
            Self::Aggressive => Self::Off,
            Self::Balanced => Self::Aggressive,
            Self::Conservative => Self::Balanced,
            Self::Off => Self::Conservative,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Aggressive => "Aggressive (max auto-chaining)",
            Self::Balanced => "Balanced (high-confidence chains)",
            Self::Conservative => "Conservative (read/verify chains only)",
            Self::Off => "Off (parallel tools only, no guessing)",
        }
    }

    pub fn short(self) -> &'static str {
        match self {
            Self::Aggressive => "AGGRESSIVE",
            Self::Balanced => "BALANCED",
            Self::Conservative => "CONSERVATIVE",
            Self::Off => "OFF",
        }
    }
}

impl fmt::Display for GuessMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.short())
    }
}

/// Full persisted configuration for Blitz.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct AppConfig {
    pub provider: ProviderKind,
    pub agy_model: String,
    pub gemini_model: String,
    pub openai_model: String,
    pub anthropic_model: String,
    pub gemini_api_key: String,
    pub openai_api_key: String,
    pub anthropic_api_key: String,
    pub openai_base_url: String,
    pub gemini_base_url: String,
    pub anthropic_base_url: String,
    pub guess_mode: GuessMode,
    pub max_depth: usize,
    pub max_branches: usize,
    pub max_nodes: usize,
    pub optimistic_finish: bool,
    pub concurrency: usize,
    pub max_turns: usize,
    pub timeout_secs: u64,
    pub max_output: usize,
    pub json_mode: bool,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            provider: ProviderKind::Agy,
            agy_model: "gemini-4-argon-high".into(),
            gemini_model: "gemini-2.5-flash".into(),
            openai_model: "gpt-4.1-mini".into(),
            anthropic_model: "claude-sonnet-4-5".into(),
            gemini_api_key: String::new(),
            openai_api_key: String::new(),
            anthropic_api_key: String::new(),
            openai_base_url: "https://api.openai.com/v1".into(),
            gemini_base_url: "https://generativelanguage.googleapis.com".into(),
            anthropic_base_url: "https://api.anthropic.com".into(),
            guess_mode: GuessMode::Aggressive,
            max_depth: 4,
            max_branches: 3,
            max_nodes: 64,
            optimistic_finish: true,
            concurrency: 16,
            max_turns: 30,
            timeout_secs: 120,
            max_output: 8000,
            json_mode: true,
        }
    }
}

impl AppConfig {
    /// Path to `~/.config/blitz/config.toml`.
    pub fn path() -> PathBuf {
        if let Ok(xdg) = std::env::var("XDG_CONFIG_HOME") {
            if !xdg.trim().is_empty() {
                return PathBuf::from(xdg).join("blitz").join("config.toml");
            }
        }
        if let Ok(home) = std::env::var("HOME") {
            return PathBuf::from(home)
                .join(".config")
                .join("blitz")
                .join("config.toml");
        }
        PathBuf::from(".blitz-config.toml")
    }

    /// Load config from disk (or default) and hydrate missing keys from environment variables.
    pub fn load() -> Self {
        let path = Self::path();
        let mut cfg: Self = std::fs::read_to_string(&path)
            .ok()
            .and_then(|s| toml::from_str(&s).ok())
            .unwrap_or_default();
        cfg.hydrate_from_env();
        cfg
    }

    /// Fill in empty API keys or base URLs from environment variables.
    pub fn hydrate_from_env(&mut self) {
        if self.gemini_api_key.trim().is_empty() {
            if let Some(k) = std::env::var("GEMINI_API_KEY")
                .ok()
                .or_else(|| std::env::var("GOOGLE_API_KEY").ok())
                .filter(|v| !v.trim().is_empty())
            {
                self.gemini_api_key = k;
            }
        }
        if self.openai_api_key.trim().is_empty() {
            if let Some(k) = std::env::var("OPENAI_API_KEY")
                .ok()
                .filter(|v| !v.trim().is_empty())
            {
                self.openai_api_key = k;
            }
        }
        if self.anthropic_api_key.trim().is_empty() {
            if let Some(k) = std::env::var("ANTHROPIC_API_KEY")
                .ok()
                .filter(|v| !v.trim().is_empty())
            {
                self.anthropic_api_key = k;
            }
        }
        if let Ok(url) = std::env::var("OPENAI_BASE_URL") {
            if !url.trim().is_empty() {
                self.openai_base_url = url;
            }
        }
    }

    /// Save config to `~/.config/blitz/config.toml`.
    pub fn save(&self) -> Result<()> {
        let path = Self::path();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("creating {}", parent.display()))?;
        }
        let text = toml::to_string_pretty(self).context("serializing config")?;
        std::fs::write(&path, text).with_context(|| format!("writing {}", path.display()))?;
        Ok(())
    }

    /// Whether the active provider requires a user-supplied API key.
    pub fn requires_api_key(&self) -> bool {
        !matches!(self.provider, ProviderKind::Agy | ProviderKind::Mock)
    }

    /// Currently active model name for the selected provider.
    pub fn active_model(&self) -> &str {
        match self.provider {
            ProviderKind::Agy => &self.agy_model,
            ProviderKind::Gemini => &self.gemini_model,
            ProviderKind::Openai => &self.openai_model,
            ProviderKind::Anthropic => &self.anthropic_model,
            ProviderKind::Mock => "mock",
        }
    }

    pub fn set_active_model(&mut self, model: String) {
        if model.trim().is_empty() {
            return;
        }
        let m = model.trim().to_string();
        match self.provider {
            ProviderKind::Agy => self.agy_model = m,
            ProviderKind::Gemini => self.gemini_model = m,
            ProviderKind::Openai => self.openai_model = m,
            ProviderKind::Anthropic => self.anthropic_model = m,
            ProviderKind::Mock => {}
        }
    }

    /// Cycle through popular model presets for the current provider.
    pub fn cycle_model(&mut self, forward: bool) {
        let presets = Self::model_presets(self.provider);
        if presets.is_empty() {
            return;
        }
        let cur = self.active_model();
        let idx = presets.iter().position(|m| *m == cur).unwrap_or(0);
        let next = if forward {
            (idx + 1) % presets.len()
        } else {
            (idx + presets.len() - 1) % presets.len()
        };
        self.set_active_model(presets[next].to_string());
    }

    pub fn model_presets(provider: ProviderKind) -> &'static [&'static str] {
        match provider {
            ProviderKind::Agy => &[
                "gemini-4-argon-high",
                "gemini-4-argon-medium",
                "gemini-4-argon-low",
                "gemini-4-neon-xhigh",
                "gemini-4-neon-high",
                "gemini-4-neon-medium",
                "gemini-4-neon-low",
                "gemini-4-hydrogen-high",
                "gemini-4-hydrogen-low",
                "gemini-3.8-flash-high",
                "gemini-3.8-flash-medium",
                "gemini-3.8-flash-low",
                "claude-opus-5-5-high",
                "claude-opus-5-5-medium",
                "claude-sonnet-5-5-high",
                "claude-sonnet-5-5-medium",
                "gpt-oss-120b-medium",
            ],
            ProviderKind::Gemini => &[
                "gemini-2.5-flash",
                "gemini-2.5-pro",
                "gemini-2.5-flash-lite",
                "gemini-2.0-flash",
            ],
            ProviderKind::Openai => &[
                "gpt-4.1-mini",
                "gpt-4.1",
                "gpt-4.1-nano",
                "o4-mini",
                "gpt-4o",
            ],
            ProviderKind::Anthropic => &[
                "claude-sonnet-4-5",
                "claude-opus-4-5",
                "claude-3-7-sonnet-latest",
                "claude-3-5-haiku-latest",
            ],
            ProviderKind::Mock => &["mock"],
        }
    }

    pub fn active_api_key(&self) -> &str {
        match self.provider {
            ProviderKind::Agy | ProviderKind::Mock => "",
            ProviderKind::Gemini => &self.gemini_api_key,
            ProviderKind::Openai => &self.openai_api_key,
            ProviderKind::Anthropic => &self.anthropic_api_key,
        }
    }

    pub fn set_active_api_key(&mut self, key: String) {
        let k = key.trim().to_string();
        match self.provider {
            ProviderKind::Gemini => self.gemini_api_key = k,
            ProviderKind::Openai => self.openai_api_key = k,
            ProviderKind::Anthropic => self.anthropic_api_key = k,
            ProviderKind::Agy | ProviderKind::Mock => {}
        }
    }

    pub fn active_base_url(&self) -> &str {
        match self.provider {
            ProviderKind::Agy => "(local agy CLI)",
            ProviderKind::Gemini => &self.gemini_base_url,
            ProviderKind::Openai => &self.openai_base_url,
            ProviderKind::Anthropic => &self.anthropic_base_url,
            ProviderKind::Mock => "",
        }
    }

    pub fn set_active_base_url(&mut self, url: String) {
        let u = url.trim().to_string();
        if u.is_empty() {
            return;
        }
        match self.provider {
            ProviderKind::Gemini => self.gemini_base_url = u,
            ProviderKind::Openai => self.openai_base_url = u,
            ProviderKind::Anthropic => self.anthropic_base_url = u,
            ProviderKind::Agy | ProviderKind::Mock => {}
        }
    }

    /// Effective plan limits based on `guess_mode`.
    pub fn limits(&self) -> Limits {
        let effective_depth = if self.guess_mode == GuessMode::Off {
            1
        } else {
            self.max_depth.max(1)
        };
        Limits {
            max_depth: effective_depth,
            max_nodes: self.max_nodes.max(1),
            max_branches: self.max_branches.max(1),
        }
    }

    pub fn mask_key(key: &str) -> String {
        let k = key.trim();
        if k.is_empty() {
            return "(not set)".into();
        }
        let chars: Vec<char> = k.chars().collect();
        if chars.len() <= 8 {
            return "••••••••".into();
        }
        let head: String = chars[..4].iter().collect();
        let tail: String = chars[chars.len() - 4..].iter().collect();
        format!("{head}••••••••{tail}")
    }
}
