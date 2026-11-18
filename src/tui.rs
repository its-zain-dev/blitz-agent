//! Clean, organized widget-style TUI for Blitz (`ratatui` + `crossterm`).

use std::io;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use anyhow::Result;
use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
};
use crossterm::ExecutableCommand;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Clear, List, ListItem, Paragraph};
use ratatui::Terminal;
use serde_json::Value;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;

use crate::agent::{Agent, AgentConfig, AgentEvent, RunResult, Stats};
use crate::config::{AppConfig, GuessMode};
use crate::exec::{ExecEvent, Executor, NodeRecord};
use crate::llm::{LlmClient, Message, ProviderKind};
use crate::tools::ToolBox;
use crate::util::truncate_chars;

const BRAILLE_SPINNER: &[&str] = &["⣾", "⣽", "⣻", "⢿", "⡿", "⣟", "⣯", "⣷"];

#[derive(Debug, Clone)]
enum ChatEntry {
    Hero,
    SlashCmd {
        cmd: String,
        result: String,
    },
    User(String),
    Thought {
        text: String,
        llm_ms: u64,
        has_tools: bool,
    },
    ToolRunning {
        id: String,
        depth: usize,
        verb: String,
        detail: String,
    },
    ToolDone {
        record: NodeRecord,
        verb: String,
        detail: String,
        is_last_in_turn: bool,
    },
    Assistant(String),
    Error(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SettingItem {
    Provider,
    Model,
    GeminiKey,
    OpenaiKey,
    AnthropicKey,
    BaseUrl,
    GuessMode,
    MaxDepth,
    MaxBranches,
    MaxNodes,
    OptimisticFinish,
    Concurrency,
    MaxTurns,
    TimeoutSecs,
    JsonMode,
}

const ALL_SETTINGS: &[SettingItem] = &[
    SettingItem::Provider,
    SettingItem::Model,
    SettingItem::GeminiKey,
    SettingItem::OpenaiKey,
    SettingItem::AnthropicKey,
    SettingItem::BaseUrl,
    SettingItem::GuessMode,
    SettingItem::MaxDepth,
    SettingItem::MaxBranches,
    SettingItem::MaxNodes,
    SettingItem::OptimisticFinish,
    SettingItem::Concurrency,
    SettingItem::MaxTurns,
    SettingItem::TimeoutSecs,
    SettingItem::JsonMode,
];

enum WorkerMsg {
    Agent(AgentEvent),
    Done {
        result: Result<RunResult, String>,
        history: Vec<Message>,
        elapsed_ms: u64,
    },
}

/// Format a tool call into a clean `(verb, detail)` pair like `("Ran", "which agy && agy --version")`
/// or `("Explored 1 search", "TODO")` or `("Edited 1 file", "src/main.rs")`.
fn describe_tool(tool: &str, args: &Value) -> (String, String) {
    let raw = match tool {
        "bash" => args
            .get("cmd")
            .or_else(|| args.get("command"))
            .and_then(|v| v.as_str())
            .unwrap_or(""),
        "read_file" | "write_file" | "edit_file" | "list_dir" => {
            args.get("path").and_then(|v| v.as_str()).unwrap_or(".")
        }
        "search" => args.get("pattern").and_then(|v| v.as_str()).unwrap_or(""),
        _ => "",
    };
    let single = raw.lines().next().unwrap_or("").trim();
    let detail = if single.is_empty() {
        truncate_chars(&args.to_string(), 56)
    } else {
        truncate_chars(single, 56)
    };
    let verb = match tool {
        "bash" => "Ran".to_string(),
        "read_file" => "Read 1 file".to_string(),
        "write_file" => "Wrote 1 file".to_string(),
        "edit_file" => "Edited 1 file".to_string(),
        "list_dir" => "Explored 1 directory".to_string(),
        "search" => "Explored 1 search".to_string(),
        other => other.to_string(),
    };
    (verb, detail)
}

/// Word/character wrap a string to `max_cols` characters so 1 `Line` == 1 visual terminal row.
fn wrap_str(s: &str, max_cols: usize) -> Vec<String> {
    let width = max_cols.max(8);
    if s.is_empty() {
        return vec![String::new()];
    }
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut cur_len = 0usize;

    for word in s.split_inclusive(' ') {
        let wlen = word.chars().count();
        if cur_len + wlen <= width {
            cur.push_str(word);
            cur_len += wlen;
        } else if wlen > width {
            for ch in word.chars() {
                if cur_len >= width {
                    out.push(std::mem::take(&mut cur));
                    cur_len = 0;
                }
                cur.push(ch);
                cur_len += 1;
            }
        } else {
            if !cur.is_empty() {
                out.push(std::mem::take(&mut cur));
            }
            cur.push_str(word);
            cur_len = wlen;
        }
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    if out.is_empty() {
        out.push(String::new());
    }
    out
}

/// Parse inline `**bold**` markers into styled `Span`s.
fn parse_inline_markdown(prefix: &str, text: &str, base_style: Style) -> Line<'static> {
    let mut spans: Vec<Span<'static>> = Vec::new();
    if !prefix.is_empty() {
        spans.push(Span::styled(prefix.to_string(), base_style));
    }
    let mut rest = text;
    let bold_style = base_style.fg(Color::White).add_modifier(Modifier::BOLD);
    while let Some(start) = rest.find("**") {
        if start > 0 {
            spans.push(Span::styled(rest[..start].to_string(), base_style));
        }
        let after_start = &rest[start + 2..];
        if let Some(end) = after_start.find("**") {
            spans.push(Span::styled(after_start[..end].to_string(), bold_style));
            rest = &after_start[end + 2..];
        } else {
            spans.push(Span::styled(rest[start..].to_string(), base_style));
            rest = "";
            break;
        }
    }
    if !rest.is_empty() {
        spans.push(Span::styled(rest.to_string(), base_style));
    }
    Line::from(spans)
}

fn short_home_path(p: &std::path::Path) -> String {
    if let Ok(home) = std::env::var("HOME") {
        let hp = std::path::Path::new(&home);
        if p == hp {
            return "~".into();
        }
        if let Ok(rel) = p.strip_prefix(hp) {
            return format!("~/{}", rel.display());
        }
    }
    p.display().to_string()
}

pub struct TuiApp {
    cfg: AppConfig,
    workdir: PathBuf,
    entries: Vec<ChatEntry>,
    history: Vec<Message>,
    stats: Stats,
    input: String,
    cursor: usize,
    input_history: Vec<String>,
    history_idx: Option<usize>,
    scroll: u16,
    follow_bottom: bool,
    show_tool_output: bool,
    show_settings: bool,
    setting_idx: usize,
    editing_setting: Option<String>,
    running: Option<JoinHandle<()>>,
    running_label: String,
    last_thought: Option<String>,
    worker_rx: Option<mpsc::UnboundedReceiver<WorkerMsg>>,
    tick: usize,
    should_quit: bool,
}

impl TuiApp {
    pub fn new(cfg: AppConfig, workdir: PathBuf) -> Self {
        let mut entries = vec![ChatEntry::Hero];
        if cfg.requires_api_key() && cfg.active_api_key().trim().is_empty() {
            entries.push(ChatEntry::SlashCmd {
                cmd: "/status".into(),
                result: format!(
                    "No API key set for {}. Type /key YOUR_KEY, /provider agy, or press F2 (/settings).",
                    cfg.provider.label()
                ),
            });
        }
        Self {
            cfg,
            workdir,
            entries,
            history: Vec::new(),
            stats: Stats::default(),
            input: String::new(),
            cursor: 0,
            input_history: Vec::new(),
            history_idx: None,
            scroll: 0,
            follow_bottom: true,
            show_tool_output: false,
            show_settings: false,
            setting_idx: 0,
            editing_setting: None,
            running: None,
            running_label: String::new(),
            last_thought: None,
            worker_rx: None,
            tick: 0,
            should_quit: false,
        }
    }

    fn push_entry(&mut self, e: ChatEntry) {
        self.entries.push(e);
        self.follow_bottom = true;
    }

    fn save_cfg(&mut self) {
        if let Err(e) = self.cfg.save() {
            self.push_entry(ChatEntry::Error(format!("Failed to save config: {e}")));
        }
    }

    fn start_task(&mut self, task: String) {
        if self.running.is_some() {
            self.push_entry(ChatEntry::Error(
                "A task is already running. Press Esc or type /cancel to stop it.".into(),
            ));
            return;
        }
        let llm = match LlmClient::from_config(&self.cfg) {
            Ok(c) => c,
            Err(e) => {
                self.push_entry(ChatEntry::Error(format!("{e}")));
                self.show_settings = true;
                return;
            }
        };

        self.last_thought = None;
        self.push_entry(ChatEntry::User(task.clone()));
        let (tx, rx) = mpsc::unbounded_channel();
        let (agent_tx, mut agent_rx) = mpsc::unbounded_channel();
        let tx_events = tx.clone();
        tokio::spawn(async move {
            while let Some(ev) = agent_rx.recv().await {
                let _ = tx_events.send(WorkerMsg::Agent(ev));
            }
        });

        let exec = Executor::new(
            ToolBox::new(
                self.workdir.clone(),
                Duration::from_secs(self.cfg.timeout_secs),
            ),
            self.cfg.concurrency,
        );
        let agent_cfg = AgentConfig {
            max_turns: self.cfg.max_turns,
            max_output: self.cfg.max_output,
            limits: self.cfg.limits(),
            guess_mode: self.cfg.guess_mode,
            optimistic_finish: self.cfg.optimistic_finish,
            verbose: false,
            events: Some(agent_tx),
        };
        let mut history = self.history.clone();
        self.running_label = format!("Thinking with {}...", self.cfg.active_model());
        self.worker_rx = Some(rx);
        self.running = Some(tokio::spawn(async move {
            let t0 = Instant::now();
            let agent = Agent::new(llm, exec, agent_cfg);
            let result = agent
                .run_with_history(&task, &mut history)
                .await
                .map_err(|e| format!("{e:#}"));
            let _ = tx.send(WorkerMsg::Done {
                result,
                history,
                elapsed_ms: t0.elapsed().as_millis() as u64,
            });
        }));
    }

    fn cancel_running(&mut self) {
        if let Some(h) = self.running.take() {
            h.abort();
            self.worker_rx = None;
            self.running_label.clear();
            self.push_entry(ChatEntry::SlashCmd {
                cmd: "cancel".into(),
                result: "Cancelled current task.".into(),
            });
        }
    }

    fn drain_worker(&mut self) {
        let mut msgs = Vec::new();
        if let Some(rx) = &mut self.worker_rx {
            while let Ok(m) = rx.try_recv() {
                msgs.push(m);
            }
        }
        for m in msgs {
            match m {
                WorkerMsg::Agent(ev) => self.on_agent_event(ev),
                WorkerMsg::Done {
                    result,
                    history,
                    elapsed_ms,
                } => {
                    self.running = None;
                    self.worker_rx = None;
                    self.running_label.clear();
                    self.history = history;
                    match result {
                        Ok(res) => {
                            self.stats.merge(&res.stats);
                            let ans = res
                                .answer
                                .filter(|a| !a.trim().is_empty())
                                .or_else(|| self.last_thought.take());
                            match ans {
                                Some(text) => self.push_entry(ChatEntry::Assistant(text)),
                                None => self.push_entry(ChatEntry::Error(format!(
                                    "Stopped after reaching max turns ({}) in {:.1}s.",
                                    self.cfg.max_turns,
                                    elapsed_ms as f64 / 1000.0
                                ))),
                            }
                        }
                        Err(e) => self.push_entry(ChatEntry::Error(e)),
                    }
                }
            }
        }
    }

    fn on_agent_event(&mut self, ev: AgentEvent) {
        match ev {
            AgentEvent::TurnStarted { turn, model, .. } => {
                self.running_label = if turn == 1 {
                    format!("Thinking ({model})...")
                } else {
                    format!("Turn {turn} · thinking ({model})...")
                };
            }
            AgentEvent::Thought { text, llm_ms } => {
                let one_line = text.split_whitespace().collect::<Vec<_>>().join(" ");
                self.running_label = truncate_chars(&one_line, 70);
                self.last_thought = Some(text.clone());
                self.push_entry(ChatEntry::Thought {
                    text,
                    llm_ms,
                    has_tools: false,
                });
            }
            AgentEvent::PlanReady { .. } => {
                // Mark the preceding Thought as having tools so it renders with `○` + summary
                if let Some(ChatEntry::Thought { has_tools, .. }) = self
                    .entries
                    .iter_mut()
                    .rev()
                    .find(|e| matches!(e, ChatEntry::Thought { .. }))
                {
                    *has_tools = true;
                }
            }
            AgentEvent::Exec(ExecEvent::NodeStarted {
                id,
                depth,
                tool,
                args,
                ..
            }) => {
                let (verb, detail) = describe_tool(&tool, &args);
                self.running_label = format!("{verb} ({detail})");
                self.push_entry(ChatEntry::ToolRunning {
                    id,
                    depth,
                    verb,
                    detail,
                });
            }
            AgentEvent::Exec(ExecEvent::NodeFinished(rec)) => {
                let (verb, detail) = describe_tool(&rec.tool, &rec.args);
                let done = ChatEntry::ToolDone {
                    record: rec.clone(),
                    verb,
                    detail,
                    is_last_in_turn: false,
                };
                if let Some(pos) = self
                    .entries
                    .iter()
                    .rposition(|e| matches!(e, ChatEntry::ToolRunning { id, .. } if *id == rec.id))
                {
                    self.entries[pos] = done;
                } else {
                    self.push_entry(done);
                }
                self.follow_bottom = true;
            }
            AgentEvent::TurnReport { .. } => {
                // Clear previous `is_last_in_turn` flags and mark the latest ToolDone
                for e in &mut self.entries {
                    if let ChatEntry::ToolDone {
                        is_last_in_turn, ..
                    } = e
                    {
                        *is_last_in_turn = false;
                    }
                }
                if let Some(ChatEntry::ToolDone {
                    is_last_in_turn, ..
                }) = self
                    .entries
                    .iter_mut()
                    .rev()
                    .find(|e| matches!(e, ChatEntry::ToolDone { .. }))
                {
                    *is_last_in_turn = true;
                }
            }
            AgentEvent::OptimisticFinish => {}
            AgentEvent::ParseError(e) => {
                self.push_entry(ChatEntry::Error(format!("Retrying plan: {e}")));
            }
        }
    }

    fn submit_input(&mut self) {
        let line = self.input.trim().to_string();
        self.input.clear();
        self.cursor = 0;
        self.history_idx = None;
        if line.is_empty() {
            return;
        }
        self.input_history.push(line.clone());
        if line.starts_with('/') {
            self.handle_slash_command(&line);
        } else {
            self.start_task(line);
        }
    }

    fn handle_slash_command(&mut self, line: &str) {
        let mut parts = line.splitn(2, char::is_whitespace);
        let cmd = parts.next().unwrap_or("");
        let arg = parts.next().unwrap_or("").trim();

        match cmd {
            "/settings" | "/config" | "/s" => {
                self.show_settings = true;
                self.editing_setting = None;
            }
            "/provider" | "/p" => {
                if arg.is_empty() {
                    self.cfg.provider = self.cfg.provider.next();
                } else {
                    match arg.to_lowercase().as_str() {
                        "agy" | "antigravity" => self.cfg.provider = ProviderKind::Agy,
                        "gemini" | "google" | "g" => self.cfg.provider = ProviderKind::Gemini,
                        "openai" | "o" => self.cfg.provider = ProviderKind::Openai,
                        "anthropic" | "claude" | "a" => self.cfg.provider = ProviderKind::Anthropic,
                        _ => {
                            self.push_entry(ChatEntry::SlashCmd {
                                cmd: line.to_string(),
                                result: "Usage: /provider <agy|gemini|openai|anthropic>".into(),
                            });
                            return;
                        }
                    }
                }
                self.save_cfg();
                self.push_entry(ChatEntry::SlashCmd {
                    cmd: line.to_string(),
                    result: format!(
                        "Provider set to {} ({})",
                        self.cfg.provider.label(),
                        self.cfg.active_model()
                    ),
                });
            }
            "/model" | "/m" => {
                if arg.is_empty() {
                    self.cfg.cycle_model(true);
                } else {
                    self.cfg.set_active_model(arg.to_string());
                }
                self.save_cfg();
                self.push_entry(ChatEntry::SlashCmd {
                    cmd: line.to_string(),
                    result: format!(
                        "Model set to {} ({})",
                        self.cfg.active_model(),
                        self.cfg.guess_mode.label()
                    ),
                });
            }
            "/key" | "/k" => {
                if !self.cfg.requires_api_key() {
                    self.push_entry(ChatEntry::SlashCmd {
                        cmd: "/key".into(),
                        result: format!(
                            "{} uses your logged-in `agy` CLI account (no API key required).",
                            self.cfg.provider.label()
                        ),
                    });
                } else if arg.is_empty() {
                    self.show_settings = true;
                    self.setting_idx = match self.cfg.provider {
                        ProviderKind::Gemini | ProviderKind::Agy | ProviderKind::Mock => 2,
                        ProviderKind::Openai => 3,
                        ProviderKind::Anthropic => 4,
                    };
                } else {
                    self.cfg.set_active_api_key(arg.to_string());
                    self.save_cfg();
                    self.push_entry(ChatEntry::SlashCmd {
                        cmd: "/key ********".into(),
                        result: format!(
                            "API key saved for {} ({})",
                            self.cfg.provider.label(),
                            AppConfig::mask_key(self.cfg.active_api_key())
                        ),
                    });
                }
            }
            "/guess" | "/g" => {
                if arg.is_empty() {
                    self.cfg.guess_mode = self.cfg.guess_mode.next();
                } else {
                    match arg.to_lowercase().as_str() {
                        "aggressive" | "max" | "a" => self.cfg.guess_mode = GuessMode::Aggressive,
                        "balanced" | "b" => self.cfg.guess_mode = GuessMode::Balanced,
                        "conservative" | "safe" | "c" => {
                            self.cfg.guess_mode = GuessMode::Conservative
                        }
                        "off" | "none" | "o" => self.cfg.guess_mode = GuessMode::Off,
                        _ => {
                            self.push_entry(ChatEntry::SlashCmd {
                                cmd: line.to_string(),
                                result: "Usage: /guess <aggressive|balanced|conservative|off>"
                                    .into(),
                            });
                            return;
                        }
                    }
                }
                self.save_cfg();
                self.push_entry(ChatEntry::SlashCmd {
                    cmd: line.to_string(),
                    result: format!("Guessing mode set to {}", self.cfg.guess_mode.label()),
                });
            }
            "/depth" => {
                if let Ok(d) = arg.parse::<usize>() {
                    self.cfg.max_depth = d.clamp(1, 8);
                    self.save_cfg();
                    self.push_entry(ChatEntry::SlashCmd {
                        cmd: line.to_string(),
                        result: format!("Max prediction depth set to {}", self.cfg.max_depth),
                    });
                } else {
                    self.push_entry(ChatEntry::SlashCmd {
                        cmd: line.to_string(),
                        result: "Usage: /depth <1..8>".into(),
                    });
                }
            }
            "/output" | "/o" => {
                self.show_tool_output = !self.show_tool_output;
                self.push_entry(ChatEntry::SlashCmd {
                    cmd: line.to_string(),
                    result: format!(
                        "Tool output expansion {}",
                        if self.show_tool_output {
                            "enabled"
                        } else {
                            "collapsed"
                        }
                    ),
                });
            }
            "/cancel" | "/stop" => self.cancel_running(),
            "/clear" | "/reset" => {
                self.entries.clear();
                self.history.clear();
                self.entries.push(ChatEntry::Hero);
                self.scroll = 0;
                self.follow_bottom = true;
            }
            "/help" | "/h" | "/?" => {
                self.push_entry(ChatEntry::SlashCmd {
                    cmd: "/help".into(),
                    result:
                        "/settings (F2) · /model [name] · /provider [name] · /key <key> · /guess [mode] · /output (ctrl+o) · /clear · /quit"
                            .into(),
                });
            }
            "/quit" | "/exit" | "/q" => {
                self.should_quit = true;
            }
            _ => {
                self.push_entry(ChatEntry::SlashCmd {
                    cmd: line.to_string(),
                    result: "Unknown command. Type /help or /settings (F2).".into(),
                });
            }
        }
    }

    fn on_key(&mut self, key: KeyEvent) {
        if key.kind != KeyEventKind::Press {
            return;
        }
        if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('c') {
            if self.running.is_some() {
                self.cancel_running();
            } else {
                self.should_quit = true;
            }
            return;
        }
        if key.code == KeyCode::F(2)
            || (key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('s'))
        {
            self.show_settings = !self.show_settings;
            self.editing_setting = None;
            return;
        }
        if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('o') {
            self.show_tool_output = !self.show_tool_output;
            return;
        }

        if self.show_settings {
            self.on_settings_key(key);
            return;
        }

        match key.code {
            KeyCode::Esc => {
                if self.running.is_some() {
                    self.cancel_running();
                }
            }
            KeyCode::Enter => self.submit_input(),
            KeyCode::Char(c) => {
                self.input.insert(self.cursor, c);
                self.cursor += c.len_utf8();
            }
            KeyCode::Backspace => {
                if self.cursor > 0 {
                    let prev = self.input[..self.cursor]
                        .char_indices()
                        .last()
                        .map(|(i, _)| i)
                        .unwrap_or(0);
                    self.input.replace_range(prev..self.cursor, "");
                    self.cursor = prev;
                }
            }
            KeyCode::Delete => {
                if self.cursor < self.input.len() {
                    let next = self.input[self.cursor..]
                        .chars()
                        .next()
                        .map(|c| self.cursor + c.len_utf8())
                        .unwrap_or(self.input.len());
                    self.input.replace_range(self.cursor..next, "");
                }
            }
            KeyCode::Left => {
                if self.cursor > 0 {
                    self.cursor = self.input[..self.cursor]
                        .char_indices()
                        .last()
                        .map(|(i, _)| i)
                        .unwrap_or(0);
                }
            }
            KeyCode::Right => {
                if self.cursor < self.input.len() {
                    self.cursor = self.input[self.cursor..]
                        .chars()
                        .next()
                        .map(|c| self.cursor + c.len_utf8())
                        .unwrap_or(self.input.len());
                }
            }
            KeyCode::Home => self.cursor = 0,
            KeyCode::End => self.cursor = self.input.len(),
            KeyCode::PageUp => {
                self.follow_bottom = false;
                self.scroll = self.scroll.saturating_sub(10);
            }
            KeyCode::PageDown => {
                self.scroll = self.scroll.saturating_add(10);
            }
            KeyCode::Up => {
                if self.input.is_empty() && !self.input_history.is_empty() {
                    let next = match self.history_idx {
                        Some(i) => i.saturating_sub(1),
                        None => self.input_history.len() - 1,
                    };
                    self.history_idx = Some(next);
                    self.input = self.input_history[next].clone();
                    self.cursor = self.input.len();
                } else {
                    self.follow_bottom = false;
                    self.scroll = self.scroll.saturating_sub(3);
                }
            }
            KeyCode::Down => {
                if let Some(i) = self.history_idx {
                    if i + 1 < self.input_history.len() {
                        self.history_idx = Some(i + 1);
                        self.input = self.input_history[i + 1].clone();
                        self.cursor = self.input.len();
                    } else {
                        self.history_idx = None;
                        self.input.clear();
                        self.cursor = 0;
                    }
                } else {
                    self.scroll = self.scroll.saturating_add(3);
                }
            }
            _ => {}
        }
    }

    fn on_settings_key(&mut self, key: KeyEvent) {
        if let Some(buf) = &mut self.editing_setting {
            match key.code {
                KeyCode::Esc => {
                    self.editing_setting = None;
                }
                KeyCode::Enter => {
                    let val = buf.trim().to_string();
                    self.editing_setting = None;
                    self.apply_text_setting(ALL_SETTINGS[self.setting_idx], val);
                    self.save_cfg();
                }
                KeyCode::Backspace => {
                    buf.pop();
                }
                KeyCode::Char(c) => {
                    buf.push(c);
                }
                _ => {}
            }
            return;
        }

        let item = ALL_SETTINGS[self.setting_idx];
        match key.code {
            KeyCode::Esc | KeyCode::Char('q') => {
                self.show_settings = false;
                self.save_cfg();
            }
            KeyCode::Up | KeyCode::Char('k') => {
                self.setting_idx = (self.setting_idx + ALL_SETTINGS.len() - 1) % ALL_SETTINGS.len();
            }
            KeyCode::Down | KeyCode::Char('j') => {
                self.setting_idx = (self.setting_idx + 1) % ALL_SETTINGS.len();
            }
            KeyCode::Left | KeyCode::Char('h') => {
                self.adjust_setting(item, false);
                self.save_cfg();
            }
            KeyCode::Right | KeyCode::Char('l') => {
                self.adjust_setting(item, true);
                self.save_cfg();
            }
            KeyCode::Enter => match item {
                SettingItem::Model => {
                    self.editing_setting = Some(self.cfg.active_model().to_string());
                }
                SettingItem::GeminiKey => {
                    self.editing_setting = Some(self.cfg.gemini_api_key.clone());
                }
                SettingItem::OpenaiKey => {
                    self.editing_setting = Some(self.cfg.openai_api_key.clone());
                }
                SettingItem::AnthropicKey => {
                    self.editing_setting = Some(self.cfg.anthropic_api_key.clone());
                }
                SettingItem::BaseUrl => {
                    self.editing_setting = Some(self.cfg.active_base_url().to_string());
                }
                other => {
                    self.adjust_setting(other, true);
                    self.save_cfg();
                }
            },
            _ => {}
        }
    }

    fn apply_text_setting(&mut self, item: SettingItem, val: String) {
        match item {
            SettingItem::Model => self.cfg.set_active_model(val),
            SettingItem::GeminiKey => self.cfg.gemini_api_key = val,
            SettingItem::OpenaiKey => self.cfg.openai_api_key = val,
            SettingItem::AnthropicKey => self.cfg.anthropic_api_key = val,
            SettingItem::BaseUrl => self.cfg.set_active_base_url(val),
            _ => {}
        }
    }

    fn adjust_setting(&mut self, item: SettingItem, forward: bool) {
        match item {
            SettingItem::Provider => {
                self.cfg.provider = if forward {
                    self.cfg.provider.next()
                } else {
                    self.cfg.provider.prev()
                };
            }
            SettingItem::Model => {
                self.cfg.cycle_model(forward);
            }
            SettingItem::GuessMode => {
                self.cfg.guess_mode = if forward {
                    self.cfg.guess_mode.next()
                } else {
                    self.cfg.guess_mode.prev()
                };
            }
            SettingItem::MaxDepth => {
                self.cfg.max_depth = if forward {
                    (self.cfg.max_depth + 1).min(8)
                } else {
                    self.cfg.max_depth.saturating_sub(1).max(1)
                };
            }
            SettingItem::MaxBranches => {
                self.cfg.max_branches = if forward {
                    (self.cfg.max_branches + 1).min(6)
                } else {
                    self.cfg.max_branches.saturating_sub(1).max(1)
                };
            }
            SettingItem::MaxNodes => {
                self.cfg.max_nodes = if forward {
                    (self.cfg.max_nodes + 8).min(128)
                } else {
                    self.cfg.max_nodes.saturating_sub(8).max(8)
                };
            }
            SettingItem::OptimisticFinish => {
                self.cfg.optimistic_finish = !self.cfg.optimistic_finish;
            }
            SettingItem::Concurrency => {
                self.cfg.concurrency = if forward {
                    (self.cfg.concurrency + 4).min(64)
                } else {
                    self.cfg.concurrency.saturating_sub(4).max(1)
                };
            }
            SettingItem::MaxTurns => {
                self.cfg.max_turns = if forward {
                    (self.cfg.max_turns + 5).min(100)
                } else {
                    self.cfg.max_turns.saturating_sub(5).max(1)
                };
            }
            SettingItem::TimeoutSecs => {
                self.cfg.timeout_secs = if forward {
                    (self.cfg.timeout_secs + 15).min(600)
                } else {
                    self.cfg.timeout_secs.saturating_sub(15).max(15)
                };
            }
            SettingItem::JsonMode => {
                self.cfg.json_mode = !self.cfg.json_mode;
            }
            _ => {}
        }
    }

    fn draw(&mut self, f: &mut ratatui::Frame) {
        let bottom_height = if self.running.is_some() { 5 } else { 4 };
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Min(4), Constraint::Length(bottom_height)])
            .split(f.area());

        self.draw_feed(f, chunks[0]);
        self.draw_bottom(f, chunks[1]);

        if self.show_settings {
            self.draw_settings_modal(f, f.area());
        }
    }

    fn draw_feed(&mut self, f: &mut ratatui::Frame, area: Rect) {
        let width = area.width as usize;
        let mut lines: Vec<Line> = Vec::new();
        let braille = BRAILLE_SPINNER[self.tick % BRAILLE_SPINNER.len()];

        for entry in &self.entries {
            match entry {
                ChatEntry::Hero => {
                    let key_status = if !self.cfg.requires_api_key() {
                        "CLI Account Ready"
                    } else if self.cfg.active_api_key().trim().is_empty() {
                        "No API Key · /settings"
                    } else {
                        "API Ready"
                    };
                    let dir_str = short_home_path(&self.workdir);
                    lines.push(Line::from(""));
                    lines.push(Line::from(vec![
                        Span::styled("      ▄▀▀▄        ", Style::default().fg(Color::Cyan)),
                        Span::styled(
                            "Blitz CLI 0.3.1",
                            Style::default()
                                .fg(Color::White)
                                .add_modifier(Modifier::BOLD),
                        ),
                    ]));
                    lines.push(Line::from(vec![
                        Span::styled("     ▀▀▀▀▀▀       ", Style::default().fg(Color::Cyan)),
                        Span::styled(
                            format!("{} ({key_status})", self.cfg.provider.label()),
                            Style::default().fg(Color::Gray),
                        ),
                    ]));
                    lines.push(Line::from(vec![
                        Span::styled("    ▀▀▀▀▀▀▀▀      ", Style::default().fg(Color::Cyan)),
                        Span::styled(
                            format!(
                                "{} ({})",
                                self.cfg.active_model(),
                                self.cfg.guess_mode.short().to_lowercase()
                            ),
                            Style::default().fg(Color::Gray),
                        ),
                    ]));
                    lines.push(Line::from(vec![
                        Span::styled("   ▄▀▀    ▀▀▄     ", Style::default().fg(Color::Cyan)),
                        Span::styled(dir_str, Style::default().fg(Color::DarkGray)),
                    ]));
                    lines.push(Line::from(vec![Span::styled(
                        "  ▄▀▀      ▀▀▄",
                        Style::default().fg(Color::Cyan),
                    )]));
                    lines.push(Line::from(""));
                }
                ChatEntry::SlashCmd { cmd, result } => {
                    lines.push(Line::from(vec![
                        Span::styled("> ", Style::default().fg(Color::DarkGray)),
                        Span::styled(cmd.clone(), Style::default().fg(Color::White)),
                    ]));
                    for part in wrap_str(result, width.saturating_sub(6)) {
                        lines.push(Line::from(vec![
                            Span::styled("  ⎿  ", Style::default().fg(Color::DarkGray)),
                            Span::styled(part, Style::default().fg(Color::Gray)),
                        ]));
                    }
                    lines.push(Line::from(""));
                }
                ChatEntry::User(text) => {
                    let rule = "─".repeat(width.min(60));
                    lines.push(Line::from(Span::styled(
                        rule,
                        Style::default().fg(Color::DarkGray),
                    )));
                    for (idx, raw) in text.lines().enumerate() {
                        for (widx, part) in wrap_str(raw, width.saturating_sub(2))
                            .into_iter()
                            .enumerate()
                        {
                            let prefix = if idx == 0 && widx == 0 { "> " } else { "  " };
                            lines.push(Line::from(vec![
                                Span::styled(
                                    prefix,
                                    Style::default()
                                        .fg(Color::White)
                                        .add_modifier(Modifier::BOLD),
                                ),
                                Span::styled(
                                    part,
                                    Style::default()
                                        .fg(Color::White)
                                        .add_modifier(Modifier::BOLD),
                                ),
                            ]));
                        }
                    }
                    lines.push(Line::from(""));
                }
                ChatEntry::Thought {
                    text,
                    llm_ms,
                    has_tools,
                } => {
                    let secs = *llm_ms as f64 / 1000.0;
                    if *has_tools {
                        let one_line = text.split_whitespace().collect::<Vec<_>>().join(" ");
                        let prefix = format!("○ Thought for {secs:.1}s (");
                        let max_preview = width.saturating_sub(prefix.len() + 2);
                        let preview = truncate_chars(&one_line, max_preview);
                        lines.push(Line::from(vec![
                            Span::styled("○ ", Style::default().fg(Color::DarkGray)),
                            Span::styled(
                                format!("Thought for {secs:.1}s "),
                                Style::default().fg(Color::DarkGray),
                            ),
                            Span::styled(
                                format!("({preview})"),
                                Style::default().fg(Color::DarkGray),
                            ),
                        ]));
                    } else {
                        lines.push(Line::from(vec![
                            Span::styled("▸ ", Style::default().fg(Color::DarkGray)),
                            Span::styled(
                                format!("Thought for {secs:.1}s"),
                                Style::default().fg(Color::DarkGray),
                            ),
                        ]));
                    }
                }
                ChatEntry::ToolRunning {
                    depth,
                    verb,
                    detail,
                    ..
                } => {
                    let prefix = if *depth > 0 { "  ⎿  " } else { "" };
                    lines.push(Line::from(vec![
                        Span::styled(prefix, Style::default().fg(Color::DarkGray)),
                        Span::styled(format!("{braille} "), Style::default().fg(Color::Cyan)),
                        Span::styled(
                            format!("{verb} "),
                            Style::default()
                                .fg(Color::White)
                                .add_modifier(Modifier::BOLD),
                        ),
                        Span::styled(format!("({detail})"), Style::default().fg(Color::DarkGray)),
                    ]));
                }
                ChatEntry::ToolDone {
                    record,
                    verb,
                    detail,
                    is_last_in_turn,
                } => {
                    let ok = record.output.success();
                    let dot_color = if ok { Color::Green } else { Color::Red };
                    let mut spans = Vec::new();
                    if record.depth > 0 {
                        spans.push(Span::styled("  ⎿  ", Style::default().fg(Color::DarkGray)));
                        spans.push(Span::styled("● ", Style::default().fg(dot_color)));
                    } else {
                        spans.push(Span::styled("● ", Style::default().fg(dot_color)));
                    }
                    spans.push(Span::styled(
                        format!("{verb} "),
                        Style::default()
                            .fg(Color::White)
                            .add_modifier(Modifier::BOLD),
                    ));
                    spans.push(Span::styled(
                        format!("({detail})"),
                        Style::default().fg(Color::Gray),
                    ));
                    if !ok {
                        spans.push(Span::styled(" failed", Style::default().fg(Color::Red)));
                    }
                    if *is_last_in_turn {
                        spans.push(Span::styled(
                            " (ctrl+o to expand)",
                            Style::default().fg(Color::DarkGray),
                        ));
                    }
                    lines.push(Line::from(spans));

                    if self.show_tool_output {
                        let out = record.output.display();
                        let trimmed = out.trim();
                        if !trimmed.is_empty() {
                            for l in trimmed.lines().take(6) {
                                lines.push(Line::from(vec![
                                    Span::styled("  ⎿  ", Style::default().fg(Color::DarkGray)),
                                    Span::styled(
                                        truncate_chars(l, width.saturating_sub(6)),
                                        Style::default().fg(Color::DarkGray),
                                    ),
                                ]));
                            }
                        }
                    }
                }
                ChatEntry::Assistant(ans) => {
                    lines.push(Line::from(""));
                    self.render_markdown_response(ans, width, &mut lines);
                    lines.push(Line::from(""));
                }
                ChatEntry::Error(err) => {
                    lines.push(Line::from(""));
                    for raw in err.lines() {
                        for part in wrap_str(raw, width.saturating_sub(4)) {
                            lines.push(Line::from(vec![
                                Span::styled("⚠ ", Style::default().fg(Color::Yellow)),
                                Span::styled(part, Style::default().fg(Color::LightRed)),
                            ]));
                        }
                    }
                    lines.push(Line::from(""));
                }
            }
        }

        let view_height = area.height as usize;
        let total = lines.len();
        let max_scroll = total.saturating_sub(view_height) as u16;
        if self.follow_bottom || self.scroll >= max_scroll {
            self.scroll = max_scroll;
            self.follow_bottom = true;
        }

        let paragraph = Paragraph::new(lines).scroll((self.scroll, 0));
        f.render_widget(paragraph, area);
    }

    fn render_markdown_response(&self, text: &str, width: usize, lines: &mut Vec<Line<'static>>) {
        let content_width = width.saturating_sub(4).max(12);
        let mut in_code_block = false;

        for raw in text.lines() {
            let trimmed = raw.trim();
            if trimmed.starts_with("```") {
                in_code_block = !in_code_block;
                continue;
            }
            if in_code_block {
                for part in wrap_str(raw, content_width.saturating_sub(2)) {
                    lines.push(Line::from(vec![
                        Span::raw("    "),
                        Span::styled(part, Style::default().fg(Color::Cyan)),
                    ]));
                }
                continue;
            }
            if trimmed == "---" || trimmed == "***" {
                lines.push(Line::from(Span::styled(
                    "  ──────",
                    Style::default().fg(Color::DarkGray),
                )));
                continue;
            }
            if trimmed.is_empty() {
                lines.push(Line::from(""));
                continue;
            }
            if trimmed.starts_with('#') {
                let heading = trimmed.trim_start_matches('#').trim();
                for part in wrap_str(heading, content_width) {
                    lines.push(Line::from(vec![
                        Span::raw("  "),
                        Span::styled(
                            part,
                            Style::default()
                                .fg(Color::White)
                                .add_modifier(Modifier::BOLD),
                        ),
                    ]));
                }
                continue;
            }
            if let Some(rest) = trimmed
                .strip_prefix("- ")
                .or_else(|| trimmed.strip_prefix("* "))
                .or_else(|| trimmed.strip_prefix("• "))
            {
                let wrapped = wrap_str(rest, content_width.saturating_sub(2));
                for (i, part) in wrapped.into_iter().enumerate() {
                    let prefix = if i == 0 { "  • " } else { "    " };
                    lines.push(parse_inline_markdown(
                        prefix,
                        &part,
                        Style::default().fg(Color::Gray),
                    ));
                }
                continue;
            }

            for part in wrap_str(raw, content_width) {
                lines.push(parse_inline_markdown(
                    "  ",
                    &part,
                    Style::default().fg(Color::White),
                ));
            }
        }
    }

    fn draw_bottom(&self, f: &mut ratatui::Frame, area: Rect) {
        let width = area.width as usize;
        let full_rule = "─".repeat(width);
        let braille = BRAILLE_SPINNER[self.tick % BRAILLE_SPINNER.len()];

        let mut y_offset = 0u16;
        if self.running.is_some() {
            let spinner_rect = Rect::new(area.x, area.y, area.width, 1);
            let label = truncate_chars(&self.running_label, width.saturating_sub(4));
            let spinner_line = Line::from(vec![
                Span::styled(format!("{braille}  "), Style::default().fg(Color::Cyan)),
                Span::styled(label, Style::default().fg(Color::Gray)),
            ]);
            f.render_widget(Paragraph::new(spinner_line), spinner_rect);
            y_offset = 1;
        }

        let top_rule_rect = Rect::new(area.x, area.y + y_offset, area.width, 1);
        let input_rect = Rect::new(area.x, area.y + y_offset + 1, area.width, 1);
        let bot_rule_rect = Rect::new(area.x, area.y + y_offset + 2, area.width, 1);
        let footer_rect = Rect::new(area.x, area.y + y_offset + 3, area.width, 1);

        f.render_widget(
            Paragraph::new(Line::from(Span::styled(
                full_rule.clone(),
                Style::default().fg(Color::DarkGray),
            ))),
            top_rule_rect,
        );

        let prompt_line = Line::from(vec![
            Span::styled(
                "> ",
                Style::default()
                    .fg(Color::White)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(self.input.clone(), Style::default().fg(Color::White)),
        ]);
        f.render_widget(Paragraph::new(prompt_line), input_rect);

        f.render_widget(
            Paragraph::new(Line::from(Span::styled(
                full_rule,
                Style::default().fg(Color::DarkGray),
            ))),
            bot_rule_rect,
        );

        let left = if self.running.is_some() {
            "esc to cancel"
        } else {
            "/settings (f2) · /model · /guess"
        };
        let right = format!(
            "{} · {}",
            self.cfg.active_model(),
            self.cfg.guess_mode.short().to_lowercase()
        );
        let pad = width.saturating_sub(left.chars().count() + right.chars().count());
        let footer_line = Line::from(vec![
            Span::styled(left, Style::default().fg(Color::DarkGray)),
            Span::raw(" ".repeat(pad)),
            Span::styled(right, Style::default().fg(Color::DarkGray)),
        ]);
        f.render_widget(Paragraph::new(footer_line), footer_rect);

        if !self.show_settings {
            let x = (input_rect.x + 2 + self.input[..self.cursor].chars().count() as u16)
                .min(input_rect.x + input_rect.width.saturating_sub(1));
            let y = input_rect.y;
            f.set_cursor_position((x, y));
        }
    }

    fn draw_settings_modal(&self, f: &mut ratatui::Frame, area: Rect) {
        let w = area.width.saturating_sub(2).min(74);
        let h = area.height.saturating_sub(2).min(20);
        let x = area.x + (area.width.saturating_sub(w)) / 2;
        let y = area.y + (area.height.saturating_sub(h)) / 2;
        let modal = Rect::new(x, y, w, h);

        f.render_widget(Clear, modal);

        let rows: Vec<(&str, String)> = vec![
            ("Provider", format!("◀ {} ▶", self.cfg.provider.label())),
            ("Model", format!("◀ {} ▶", self.cfg.active_model())),
            ("Gemini Key", AppConfig::mask_key(&self.cfg.gemini_api_key)),
            ("OpenAI Key", AppConfig::mask_key(&self.cfg.openai_api_key)),
            (
                "Anthropic Key",
                AppConfig::mask_key(&self.cfg.anthropic_api_key),
            ),
            ("Base URL", self.cfg.active_base_url().to_string()),
            ("Guess Mode", format!("◀ {} ▶", self.cfg.guess_mode.short())),
            ("Guess Depth", format!("◀ {} ▶", self.cfg.max_depth)),
            ("Max Branches", format!("◀ {} ▶", self.cfg.max_branches)),
            ("Max Nodes", format!("◀ {} ▶", self.cfg.max_nodes)),
            (
                "Fast Finish",
                format!(
                    "◀ {} ▶",
                    if self.cfg.optimistic_finish {
                        "ON"
                    } else {
                        "OFF"
                    }
                ),
            ),
            ("Concurrency", format!("◀ {}x ▶", self.cfg.concurrency)),
            ("Max Turns", format!("◀ {} ▶", self.cfg.max_turns)),
            ("Timeout", format!("◀ {}s ▶", self.cfg.timeout_secs)),
            (
                "JSON Mode",
                format!("◀ {} ▶", if self.cfg.json_mode { "ON" } else { "OFF" }),
            ),
        ];

        let items: Vec<ListItem> = rows
            .iter()
            .enumerate()
            .map(|(i, (label, val))| {
                let selected = i == self.setting_idx;
                let shown_val = if selected {
                    if let Some(edit_buf) = &self.editing_setting {
                        format!("✎ {edit_buf}█")
                    } else {
                        val.clone()
                    }
                } else {
                    val.clone()
                };
                let marker = if selected { "❯ " } else { "  " };
                let style = if selected {
                    Style::default()
                        .fg(Color::Black)
                        .bg(Color::Cyan)
                        .add_modifier(Modifier::BOLD)
                } else {
                    Style::default().fg(Color::White)
                };
                ListItem::new(Line::from(vec![
                    Span::styled(format!("{marker}{label:<15} "), style),
                    Span::styled(shown_val, style),
                ]))
            })
            .collect();

        let title = if self.editing_setting.is_some() {
            " Settings • Type value & press Enter (Esc cancel) "
        } else {
            " Settings • ↑/↓ Select • ←/→ Change • Enter Edit • Esc Close "
        };
        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::Cyan))
            .title(Span::styled(
                title,
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            ));

        f.render_widget(List::new(items).block(block), modal);
    }
}

/// Launch the interactive TUI.
pub async fn run_tui(cfg: AppConfig, workdir: PathBuf) -> Result<()> {
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    stdout.execute(EnterAlternateScreen)?;
    let backend = ratatui::backend::CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let mut app = TuiApp::new(cfg, workdir);
    let res = tui_loop(&mut terminal, &mut app).await;

    disable_raw_mode()?;
    terminal.backend_mut().execute(LeaveAlternateScreen)?;
    terminal.show_cursor()?;
    res
}

async fn tui_loop(
    terminal: &mut Terminal<ratatui::backend::CrosstermBackend<io::Stdout>>,
    app: &mut TuiApp,
) -> Result<()> {
    loop {
        app.drain_worker();
        terminal.draw(|f| app.draw(f))?;
        if app.should_quit {
            app.cancel_running();
            break;
        }
        if event::poll(Duration::from_millis(60))? {
            if let Event::Key(key) = event::read()? {
                app.on_key(key);
            }
        }
        app.tick = app.tick.wrapping_add(1);
    }
    Ok(())
}
