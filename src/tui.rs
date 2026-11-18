//! Full interactive Terminal UI (TUI) for Blitz (`ratatui` + `crossterm`).

use std::io;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use anyhow::Result;
use crossterm::event::{
    self, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers,
};
use crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
};
use crossterm::ExecutableCommand;
use ratatui::layout::{Alignment, Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{
    Block, Borders, Clear, List, ListItem, Paragraph, Scrollbar, ScrollbarOrientation,
    ScrollbarState, Wrap,
};
use ratatui::Terminal;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;

use crate::agent::{Agent, AgentConfig, AgentEvent, RunResult, Stats};
use crate::config::{AppConfig, GuessMode};
use crate::exec::{ExecEvent, Executor, NodeRecord, Outcome};
use crate::llm::{LlmClient, Message, ProviderKind};
use crate::tools::ToolBox;
use crate::util::truncate_chars;

const SPINNER: &[&str] = &["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];

#[derive(Debug, Clone)]
enum ChatEntry {
    Banner,
    User(String),
    TurnHeader {
        turn: usize,
        max_turns: usize,
        model: String,
    },
    Thought(String),
    PlanInfo {
        steps: usize,
        total_nodes: usize,
    },
    ToolRunning {
        id: String,
        parent: Option<String>,
        depth: usize,
        tool: String,
        args: String,
    },
    ToolDone(NodeRecord),
    OptimisticFinish,
    TurnSummary {
        turn: usize,
        tools: usize,
        chained: usize,
        hits: usize,
        misses: usize,
        wall_ms: u64,
        seq_ms: u64,
    },
    Assistant(String),
    Notice(String),
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
    worker_rx: Option<mpsc::UnboundedReceiver<WorkerMsg>>,
    tick: usize,
    should_quit: bool,
}

impl TuiApp {
    pub fn new(cfg: AppConfig, workdir: PathBuf) -> Self {
        let mut entries = vec![ChatEntry::Banner];
        if cfg.active_api_key().trim().is_empty() {
            entries.push(ChatEntry::Notice(format!(
                "No API key set for {}. Press F2 (or type /settings or /key YOUR_KEY) to configure it.",
                cfg.provider.label()
            )));
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
            show_tool_output: true,
            show_settings: false,
            setting_idx: 0,
            editing_setting: None,
            running: None,
            running_label: String::new(),
            worker_rx: None,
            tick: 0,
            should_quit: false,
        }
    }

    fn push_entry(&mut self, e: ChatEntry) {
        self.entries.push(e);
        if self.follow_bottom {
            self.scroll = u16::MAX;
        }
    }

    fn save_cfg(&mut self) {
        if let Err(e) = self.cfg.save() {
            self.push_entry(ChatEntry::Error(format!("Failed to save config: {e}")));
        }
    }

    fn start_task(&mut self, task: String) {
        if self.running.is_some() {
            self.push_entry(ChatEntry::Notice(
                "A task is already running. Press Esc or type /cancel to abort it first.".into(),
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
        self.running_label = format!("Starting ({}) ...", self.cfg.active_model());
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
            self.push_entry(ChatEntry::Notice("⏹ Task cancelled by user.".into()));
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
                            match res.answer {
                                Some(ans) => self.push_entry(ChatEntry::Assistant(ans)),
                                None => self.push_entry(ChatEntry::Notice(format!(
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
            AgentEvent::TurnStarted {
                turn,
                max_turns,
                model,
            } => {
                self.running_label = format!("Turn {turn}/{max_turns} • thinking ({model})...");
                self.push_entry(ChatEntry::TurnHeader {
                    turn,
                    max_turns,
                    model,
                });
            }
            AgentEvent::Thought(t) => {
                self.push_entry(ChatEntry::Thought(t));
            }
            AgentEvent::PlanReady { steps, total_nodes } => {
                self.running_label =
                    format!("Executing {steps} parallel step(s) ({total_nodes} nodes)...");
                self.push_entry(ChatEntry::PlanInfo { steps, total_nodes });
            }
            AgentEvent::Exec(ExecEvent::NodeStarted {
                id,
                parent,
                depth,
                tool,
                args,
            }) => {
                self.running_label = format!("Running [{id}] {tool}...");
                self.push_entry(ChatEntry::ToolRunning {
                    id,
                    parent,
                    depth,
                    tool,
                    args: truncate_chars(&args.to_string(), 110),
                });
            }
            AgentEvent::Exec(ExecEvent::NodeFinished(rec)) => {
                if let Some(pos) = self.entries.iter().rposition(|e| {
                    matches!(e, ChatEntry::ToolRunning { id, .. } if *id == rec.id)
                }) {
                    self.entries[pos] = ChatEntry::ToolDone(rec);
                } else {
                    self.push_entry(ChatEntry::ToolDone(rec));
                }
                if self.follow_bottom {
                    self.scroll = u16::MAX;
                }
            }
            AgentEvent::TurnReport {
                turn,
                report,
                stats: _,
            } => {
                self.push_entry(ChatEntry::TurnSummary {
                    turn,
                    tools: report.tool_calls(),
                    chained: report.chained(),
                    hits: report.hits(),
                    misses: report.misses(),
                    wall_ms: report.wall_ms,
                    seq_ms: report.tool_ms(),
                });
            }
            AgentEvent::ParseError(e) => {
                self.push_entry(ChatEntry::Error(format!("Plan validation warning: {e}")));
            }
            AgentEvent::OptimisticFinish => {
                self.push_entry(ChatEntry::OptimisticFinish);
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
                        "gemini" | "google" | "g" => self.cfg.provider = ProviderKind::Gemini,
                        "openai" | "o" => self.cfg.provider = ProviderKind::Openai,
                        "anthropic" | "claude" | "a" => self.cfg.provider = ProviderKind::Anthropic,
                        _ => {
                            self.push_entry(ChatEntry::Error(
                                "Usage: /provider <gemini|openai|anthropic>".into(),
                            ));
                            return;
                        }
                    }
                }
                self.save_cfg();
                self.push_entry(ChatEntry::Notice(format!(
                    "Provider switched to {} (model: {})",
                    self.cfg.provider.label(),
                    self.cfg.active_model()
                )));
            }
            "/model" | "/m" => {
                if arg.is_empty() {
                    self.cfg.cycle_model(true);
                } else {
                    self.cfg.set_active_model(arg.to_string());
                }
                self.save_cfg();
                self.push_entry(ChatEntry::Notice(format!(
                    "Active model for {} set to '{}'",
                    self.cfg.provider.label(),
                    self.cfg.active_model()
                )));
            }
            "/key" | "/k" => {
                if arg.is_empty() {
                    self.show_settings = true;
                    self.setting_idx = match self.cfg.provider {
                        ProviderKind::Gemini | ProviderKind::Mock => 2,
                        ProviderKind::Openai => 3,
                        ProviderKind::Anthropic => 4,
                    };
                } else {
                    self.cfg.set_active_api_key(arg.to_string());
                    self.save_cfg();
                    self.push_entry(ChatEntry::Notice(format!(
                        "Saved API key for {} ({})",
                        self.cfg.provider.label(),
                        AppConfig::mask_key(self.cfg.active_api_key())
                    )));
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
                            self.push_entry(ChatEntry::Error(
                                "Usage: /guess <aggressive|balanced|conservative|off>".into(),
                            ));
                            return;
                        }
                    }
                }
                self.save_cfg();
                self.push_entry(ChatEntry::Notice(format!(
                    "Guessing mode: {}",
                    self.cfg.guess_mode.label()
                )));
            }
            "/depth" => {
                if let Ok(d) = arg.parse::<usize>() {
                    self.cfg.max_depth = d.clamp(1, 8);
                    self.save_cfg();
                    self.push_entry(ChatEntry::Notice(format!(
                        "Max prediction depth set to {}",
                        self.cfg.max_depth
                    )));
                } else {
                    self.push_entry(ChatEntry::Error("Usage: /depth <1..8>".into()));
                }
            }
            "/output" | "/o" => {
                self.show_tool_output = !self.show_tool_output;
                self.push_entry(ChatEntry::Notice(format!(
                    "Tool output preview in chat: {}",
                    if self.show_tool_output { "ON" } else { "OFF" }
                )));
            }
            "/cancel" | "/stop" => self.cancel_running(),
            "/clear" | "/reset" => {
                self.entries.clear();
                self.history.clear();
                self.entries.push(ChatEntry::Banner);
                self.scroll = 0;
                self.follow_bottom = true;
            }
            "/help" | "/h" | "/?" => {
                self.push_entry(ChatEntry::Notice(
                    "Commands:\n\
                     • /settings (or F2)      Open full interactive Settings menu\n\
                     • /provider [name]       Cycle or set provider (gemini, openai, anthropic)\n\
                     • /model [name]          Cycle presets or set custom model name\n\
                     • /key <api_key>         Save API key for active provider\n\
                     • /guess [mode]          Cycle or set guessing (aggressive, balanced, conservative, off)\n\
                     • /depth <1-8>           Set max prediction tree depth\n\
                     • /output                Toggle tool stdout/stderr previews in chat\n\
                     • /cancel (or Esc)       Stop the currently running task\n\
                     • /clear                 Clear chat & conversation memory\n\
                     • /quit (or Ctrl+C)      Exit Blitz"
                        .into(),
                ));
            }
            "/quit" | "/exit" | "/q" => {
                self.should_quit = true;
            }
            other => {
                self.push_entry(ChatEntry::Error(format!(
                    "Unknown command '{other}'. Type /help for commands or /settings to open Settings."
                )));
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
            KeyCode::Enter => {
                match item {
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
                }
            }
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
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3),
                Constraint::Min(6),
                Constraint::Length(3),
                Constraint::Length(1),
            ])
            .split(f.area());

        self.draw_header(f, chunks[0]);
        self.draw_chat(f, chunks[1]);
        self.draw_input(f, chunks[2]);
        self.draw_footer(f, chunks[3]);

        if self.show_settings {
            self.draw_settings_modal(f, f.area());
        }
    }

    fn draw_header(&self, f: &mut ratatui::Frame, area: Rect) {
        let spinner = SPINNER[self.tick % SPINNER.len()];
        let status_span = if self.running.is_some() {
            Span::styled(
                format!(" {spinner} {} ", self.running_label),
                Style::default()
                    .fg(Color::Black)
                    .bg(Color::Yellow)
                    .add_modifier(Modifier::BOLD),
            )
        } else {
            Span::styled(
                " ● READY ",
                Style::default()
                    .fg(Color::Black)
                    .bg(Color::Green)
                    .add_modifier(Modifier::BOLD),
            )
        };

        let line = Line::from(vec![
            Span::styled(
                " ⚡ BLITZ ",
                Style::default()
                    .fg(Color::Black)
                    .bg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::raw(" "),
            Span::styled(
                format!(" {} • {} ", self.cfg.provider.short(), self.cfg.active_model()),
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::raw("│ "),
            Span::styled(
                format!(
                    "GUESS: {} (depth {}, {} br) ",
                    self.cfg.guess_mode.short(),
                    self.cfg.max_depth,
                    self.cfg.max_branches
                ),
                Style::default()
                    .fg(Color::Magenta)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::raw("│ "),
            Span::styled(
                format!(
                    "turns:{} tools:{} chained:{} hits:{}/{} ({:.1}x) ",
                    self.stats.llm_turns,
                    self.stats.tool_calls,
                    self.stats.chained,
                    self.stats.hits,
                    self.stats.hits + self.stats.misses,
                    self.stats.parallel_speedup()
                ),
                Style::default().fg(Color::Gray),
            ),
            status_span,
        ]);

        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::DarkGray))
            .title(Span::styled(
                format!(" {} ", self.workdir.display()),
                Style::default().fg(Color::DarkGray),
            ));
        f.render_widget(Paragraph::new(line).block(block), area);
    }

    fn draw_chat(&mut self, f: &mut ratatui::Frame, area: Rect) {
        let mut lines: Vec<Line> = Vec::new();
        let spinner = SPINNER[self.tick % SPINNER.len()];

        for entry in &self.entries {
            match entry {
                ChatEntry::Banner => {
                    lines.push(Line::from(vec![
                        Span::styled(
                            "⚡ Blitz Autonomous Speculative Agent",
                            Style::default()
                                .fg(Color::Cyan)
                                .add_modifier(Modifier::BOLD),
                        ),
                        Span::styled(
                            " — Parallel Tool Execution + Automatic Branch Prediction",
                            Style::default().fg(Color::Gray),
                        ),
                    ]));
                    lines.push(Line::from(vec![
                        Span::styled("   • Press ", Style::default().fg(Color::DarkGray)),
                        Span::styled("F2", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
                        Span::styled(" or type ", Style::default().fg(Color::DarkGray)),
                        Span::styled("/settings", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
                        Span::styled(
                            " to configure Provider, Model, API Keys, and Guessing Mode.",
                            Style::default().fg(Color::DarkGray),
                        ),
                    ]));
                    lines.push(Line::from(""));
                }
                ChatEntry::User(text) => {
                    lines.push(Line::from(""));
                    for (i, l) in text.lines().enumerate() {
                        let prefix = if i == 0 { "❯ You: " } else { "       " };
                        lines.push(Line::from(vec![
                            Span::styled(
                                prefix,
                                Style::default()
                                    .fg(Color::Cyan)
                                    .add_modifier(Modifier::BOLD),
                            ),
                            Span::styled(
                                l.to_string(),
                                Style::default()
                                    .fg(Color::White)
                                    .add_modifier(Modifier::BOLD),
                            ),
                        ]));
                    }
                }
                ChatEntry::TurnHeader {
                    turn,
                    max_turns,
                    model,
                } => {
                    lines.push(Line::from(vec![Span::styled(
                        format!("  ━━ Turn {turn}/{max_turns} • {model} ━━"),
                        Style::default().fg(Color::DarkGray),
                    )]));
                }
                ChatEntry::Thought(t) => {
                    for (i, l) in t.lines().enumerate() {
                        let prefix = if i == 0 { "  💭 " } else { "     " };
                        lines.push(Line::from(vec![
                            Span::styled(prefix, Style::default().fg(Color::Yellow)),
                            Span::styled(
                                l.to_string(),
                                Style::default()
                                    .fg(Color::Yellow)
                                    .add_modifier(Modifier::ITALIC),
                            ),
                        ]));
                    }
                }
                ChatEntry::PlanInfo { steps, total_nodes } => {
                    let extra = total_nodes.saturating_sub(*steps);
                    lines.push(Line::from(vec![
                        Span::styled("  ⚡ ", Style::default().fg(Color::Magenta)),
                        Span::styled(
                            format!(
                                "Plan: {steps} parallel step(s) + {extra} predicted branch node(s)"
                            ),
                            Style::default().fg(Color::Magenta),
                        ),
                    ]));
                }
                ChatEntry::ToolRunning {
                    id,
                    parent,
                    depth,
                    tool,
                    args,
                } => {
                    let indent = "    ".repeat(*depth);
                    let via = parent
                        .as_ref()
                        .map(|p| format!("↳ (chained from {p}) "))
                        .unwrap_or_default();
                    lines.push(Line::from(vec![
                        Span::raw(format!("  {indent}")),
                        Span::styled(
                            format!("{spinner} "),
                            Style::default().fg(Color::Yellow),
                        ),
                        Span::styled(via, Style::default().fg(Color::Magenta)),
                        Span::styled(
                            format!("[{id}] {tool} "),
                            Style::default()
                                .fg(Color::Cyan)
                                .add_modifier(Modifier::BOLD),
                        ),
                        Span::styled(args.clone(), Style::default().fg(Color::DarkGray)),
                    ]));
                }
                ChatEntry::ToolDone(r) => {
                    let indent = "    ".repeat(r.depth);
                    let tree = if r.depth > 0 { "↳ ⚡ " } else { "• " };
                    let (status_badge, status_color) = if r.output.success() {
                        (format!("✓ exit 0 ({}ms)", r.output.duration_ms), Color::Green)
                    } else {
                        (
                            format!("✗ exit {} ({}ms)", r.output.exit_code, r.output.duration_ms),
                            Color::Red,
                        )
                    };
                    let outcome_span = match &r.outcome {
                        Outcome::NoPrediction => Span::raw(""),
                        Outcome::Hit { branch, spawned } => {
                            if spawned.is_empty() {
                                Span::styled(
                                    format!("  ✔ GUESS HIT #{branch}"),
                                    Style::default()
                                        .fg(Color::Green)
                                        .add_modifier(Modifier::BOLD),
                                )
                            } else {
                                Span::styled(
                                    format!(
                                        "  ✔ GUESS HIT #{branch} → auto-ran [{}]",
                                        spawned.join(", ")
                                    ),
                                    Style::default()
                                        .fg(Color::LightGreen)
                                        .add_modifier(Modifier::BOLD),
                                )
                            }
                        }
                        Outcome::Miss { predictions } => Span::styled(
                            format!("  ✘ GUESS MISS (0/{predictions})"),
                            Style::default().fg(Color::Yellow),
                        ),
                    };

                    lines.push(Line::from(vec![
                        Span::raw(format!("  {indent}")),
                        Span::styled(tree, Style::default().fg(Color::Magenta)),
                        Span::styled(
                            format!("[{}] {} ", r.id, r.tool),
                            Style::default()
                                .fg(Color::Cyan)
                                .add_modifier(Modifier::BOLD),
                        ),
                        Span::styled(
                            format!("{} ", truncate_chars(&r.args.to_string(), 75)),
                            Style::default().fg(Color::Gray),
                        ),
                        Span::styled(status_badge, Style::default().fg(status_color)),
                        outcome_span,
                    ]));

                    if self.show_tool_output {
                        let out = r.output.display();
                        let trimmed = out.trim();
                        if !trimmed.is_empty() {
                            for l in trimmed.lines().take(4) {
                                lines.push(Line::from(vec![
                                    Span::raw(format!("    {indent}│ ")),
                                    Span::styled(
                                        truncate_chars(l, 120),
                                        Style::default().fg(Color::DarkGray),
                                    ),
                                ]));
                            }
                            let total_lines = trimmed.lines().count();
                            if total_lines > 4 {
                                lines.push(Line::from(vec![
                                    Span::raw(format!("    {indent}│ ")),
                                    Span::styled(
                                        format!("... (+{} more lines)", total_lines - 4),
                                        Style::default().fg(Color::DarkGray),
                                    ),
                                ]));
                            }
                        }
                    }
                }
                ChatEntry::OptimisticFinish => {
                    lines.push(Line::from(vec![Span::styled(
                        "  🏁 Predictions held — finished immediately without another LLM round trip!",
                        Style::default()
                            .fg(Color::LightGreen)
                            .add_modifier(Modifier::BOLD),
                    )]));
                }
                ChatEntry::TurnSummary {
                    turn,
                    tools,
                    chained,
                    hits,
                    misses,
                    wall_ms,
                    seq_ms,
                } => {
                    let speedup = if *wall_ms > 0 {
                        (*seq_ms as f64 / *wall_ms as f64).max(1.0)
                    } else {
                        1.0
                    };
                    lines.push(Line::from(vec![Span::styled(
                        format!(
                            "  📊 Turn {turn}: {tools} tool(s), {chained} auto-chained, {hits} hit / {misses} miss • {wall_ms}ms wall vs {seq_ms}ms seq ({speedup:.1}x)"
                        ),
                        Style::default().fg(Color::DarkGray),
                    )]));
                }
                ChatEntry::Assistant(ans) => {
                    lines.push(Line::from(""));
                    lines.push(Line::from(vec![Span::styled(
                        "⚡ Blitz:",
                        Style::default()
                            .fg(Color::Green)
                            .add_modifier(Modifier::BOLD),
                    )]));
                    for l in ans.lines() {
                        lines.push(Line::from(vec![
                            Span::raw("  "),
                            Span::styled(l.to_string(), Style::default().fg(Color::White)),
                        ]));
                    }
                    lines.push(Line::from(""));
                }
                ChatEntry::Notice(msg) => {
                    for l in msg.lines() {
                        lines.push(Line::from(vec![
                            Span::styled("  ℹ ", Style::default().fg(Color::Cyan)),
                            Span::styled(l.to_string(), Style::default().fg(Color::Cyan)),
                        ]));
                    }
                }
                ChatEntry::Error(err) => {
                    for l in err.lines() {
                        lines.push(Line::from(vec![
                            Span::styled("  ⚠ ", Style::default().fg(Color::Red)),
                            Span::styled(l.to_string(), Style::default().fg(Color::LightRed)),
                        ]));
                    }
                }
            }
        }

        let inner_height = area.height.saturating_sub(2) as usize;
        let total = lines.len();
        let max_scroll = total.saturating_sub(inner_height) as u16;
        if self.follow_bottom || self.scroll >= max_scroll {
            self.scroll = max_scroll;
            self.follow_bottom = true;
        }

        let chat_block = Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::DarkGray))
            .title(Span::styled(
                " Conversation & Speculative Execution Tree ",
                Style::default().fg(Color::Gray),
            ));

        let paragraph = Paragraph::new(lines)
            .block(chat_block)
            .wrap(Wrap { trim: false })
            .scroll((self.scroll, 0));
        f.render_widget(paragraph, area);

        let mut scrollbar_state = ScrollbarState::new(total).position(self.scroll as usize);
        f.render_stateful_widget(
            Scrollbar::new(ScrollbarOrientation::VerticalRight),
            area,
            &mut scrollbar_state,
        );
    }

    fn draw_input(&self, f: &mut ratatui::Frame, area: Rect) {
        let title = if self.running.is_some() {
            " Running... (Press Esc or /cancel to stop) "
        } else {
            " Prompt (Enter to run • /settings or F2 for Settings • /help for commands) "
        };
        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(if self.show_settings {
                Color::DarkGray
            } else {
                Color::Cyan
            }))
            .title(Span::styled(title, Style::default().fg(Color::Cyan)));
        let p = Paragraph::new(format!("❯ {}", self.input)).block(block);
        f.render_widget(p, area);

        if !self.show_settings {
            let x = (area.x + 3 + self.input[..self.cursor].chars().count() as u16)
                .min(area.x + area.width.saturating_sub(2));
            let y = area.y + 1;
            f.set_cursor_position((x, y));
        }
    }

    fn draw_footer(&self, f: &mut ratatui::Frame, area: Rect) {
        let line = Line::from(vec![
            Span::styled(" F2", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
            Span::styled(":Settings  ", Style::default().fg(Color::DarkGray)),
            Span::styled("/provider", Style::default().fg(Color::Yellow)),
            Span::styled(":Switch API  ", Style::default().fg(Color::DarkGray)),
            Span::styled("/model", Style::default().fg(Color::Yellow)),
            Span::styled(":Switch Model  ", Style::default().fg(Color::DarkGray)),
            Span::styled("/guess", Style::default().fg(Color::Yellow)),
            Span::styled(":Guess Mode  ", Style::default().fg(Color::DarkGray)),
            Span::styled("Ctrl+O", Style::default().fg(Color::Yellow)),
            Span::styled(":Toggle Logs  ", Style::default().fg(Color::DarkGray)),
            Span::styled("PgUp/PgDn", Style::default().fg(Color::Yellow)),
            Span::styled(":Scroll  ", Style::default().fg(Color::DarkGray)),
            Span::styled("Ctrl+C", Style::default().fg(Color::Yellow)),
            Span::styled(":Quit", Style::default().fg(Color::DarkGray)),
        ]);
        f.render_widget(Paragraph::new(line), area);
    }

    fn draw_settings_modal(&self, f: &mut ratatui::Frame, area: Rect) {
        let w = area.width.saturating_sub(6).min(82);
        let h = area.height.saturating_sub(2).min(23);
        let x = area.x + (area.width.saturating_sub(w)) / 2;
        let y = area.y + (area.height.saturating_sub(h)) / 2;
        let modal = Rect::new(x, y, w, h);

        f.render_widget(Clear, modal);

        let rows: Vec<(&str, String, &str)> = vec![
            (
                "Provider",
                format!("◀ {} ▶", self.cfg.provider.label()),
                "←/→ cycle (Gemini, OpenAI, Anthropic)",
            ),
            (
                "Active Model",
                format!("◀ {} ▶", self.cfg.active_model()),
                "←/→ presets • Enter to type custom model",
            ),
            (
                "Gemini API Key",
                AppConfig::mask_key(&self.cfg.gemini_api_key),
                "Enter to paste/edit GEMINI_API_KEY",
            ),
            (
                "OpenAI API Key",
                AppConfig::mask_key(&self.cfg.openai_api_key),
                "Enter to paste/edit OPENAI_API_KEY",
            ),
            (
                "Anthropic API Key",
                AppConfig::mask_key(&self.cfg.anthropic_api_key),
                "Enter to paste/edit ANTHROPIC_API_KEY",
            ),
            (
                "Provider Base URL",
                self.cfg.active_base_url().to_string(),
                "Enter to edit endpoint (OpenRouter, Groq, etc.)",
            ),
            (
                "Guessing Mode",
                format!("◀ {} ▶", self.cfg.guess_mode.label()),
                "←/→ cycle automatic prediction aggressiveness",
            ),
            (
                "Max Guess Depth",
                format!("◀ {} levels ▶", self.cfg.max_depth),
                "←/→ max nested prediction steps per turn (1..8)",
            ),
            (
                "Max Branches / Tool",
                format!("◀ {} branches ▶", self.cfg.max_branches),
                "←/→ max predicted outcomes per tool call (1..6)",
            ),
            (
                "Max Nodes / Plan",
                format!("◀ {} nodes ▶", self.cfg.max_nodes),
                "←/→ max total nodes in one turn's execution tree",
            ),
            (
                "Optimistic Finish",
                format!(
                    "◀ {} ▶",
                    if self.cfg.optimistic_finish { "ON" } else { "OFF" }
                ),
                "Finish inside a prediction branch without an extra turn",
            ),
            (
                "Parallel Concurrency",
                format!("◀ {} tools ▶", self.cfg.concurrency),
                "←/→ max tools running at the same instant",
            ),
            (
                "Max Turns / Task",
                format!("◀ {} turns ▶", self.cfg.max_turns),
                "←/→ safety cap on LLM turns per task",
            ),
            (
                "Bash Timeout",
                format!("◀ {} secs ▶", self.cfg.timeout_secs),
                "←/→ default timeout per shell command",
            ),
            (
                "Strict JSON Mode",
                format!("◀ {} ▶", if self.cfg.json_mode { "ON" } else { "OFF" }),
                "←/→ provider response_format JSON enforcement",
            ),
        ];

        let items: Vec<ListItem> = rows
            .iter()
            .enumerate()
            .map(|(i, (label, val, _hint))| {
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
                    Span::styled(format!("{marker}{label:<21} "), style),
                    Span::styled(format!("{shown_val:<48}"), style),
                ]))
            })
            .collect();

        let inner = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Min(10), Constraint::Length(3)])
            .split(modal);

        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::Cyan))
            .title(Span::styled(
                " ⚙ Blitz Settings (Auto-saved to ~/.config/blitz/config.toml) ",
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            ));

        f.render_widget(List::new(items).block(block), inner[0]);

        let hint = if self.editing_setting.is_some() {
            "Typing new value... Press Enter to save or Esc to cancel."
        } else {
            rows.get(self.setting_idx)
                .map(|(_, _, h)| *h)
                .unwrap_or("")
        };
        let hint_block = Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::DarkGray))
            .title(" ↑/↓ Select • ←/→ Adjust • Enter Edit • Esc Close ");
        f.render_widget(
            Paragraph::new(hint)
                .style(Style::default().fg(Color::Yellow))
                .alignment(Alignment::Center)
                .block(hint_block),
            inner[1],
        );
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
