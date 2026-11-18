//! The agent loop: ask the model for a plan, execute it speculatively, feed results back.

use std::fmt;
use std::time::Instant;

use anyhow::{bail, Result};
use tokio::sync::mpsc;

use crate::config::GuessMode;
use crate::exec::{ExecEvent, ExecReport, Executor};
use crate::llm::{LlmClient, Message};
use crate::plan::{Limits, Plan};
use crate::prompt::system_prompt_with;
use crate::tools::TOOL_NAMES;
use crate::util::truncate_chars;

/// Live events emitted by the agent loop (consumed by the TUI).
#[derive(Debug, Clone)]
pub enum AgentEvent {
    TurnStarted {
        turn: usize,
        max_turns: usize,
        model: String,
    },
    Thought(String),
    PlanReady {
        steps: usize,
        total_nodes: usize,
    },
    Exec(ExecEvent),
    TurnReport {
        turn: usize,
        report: ExecReport,
        stats: Stats,
    },
    ParseError(String),
    OptimisticFinish,
}

#[derive(Debug, Clone)]
pub struct AgentConfig {
    pub max_turns: usize,
    /// Max bytes of each tool's output that is shown to the model.
    pub max_output: usize,
    pub limits: Limits,
    pub guess_mode: GuessMode,
    pub optimistic_finish: bool,
    /// Print progress to stderr.
    pub verbose: bool,
    pub events: Option<mpsc::UnboundedSender<AgentEvent>>,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            max_turns: 30,
            max_output: 8000,
            limits: Limits::default(),
            guess_mode: GuessMode::Aggressive,
            optimistic_finish: true,
            verbose: false,
            events: None,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct Stats {
    pub llm_turns: usize,
    pub tool_calls: usize,
    /// Tool calls that ran automatically after a prediction hit.
    pub chained: usize,
    pub hits: usize,
    pub misses: usize,
    pub parse_errors: usize,
    pub llm_ms: u64,
    /// Wall-clock time spent executing tools.
    pub exec_wall_ms: u64,
    /// Sum of individual tool durations (what a sequential agent would wait).
    pub tool_ms: u64,
}

impl Stats {
    pub fn merge(&mut self, other: &Stats) {
        self.llm_turns += other.llm_turns;
        self.tool_calls += other.tool_calls;
        self.chained += other.chained;
        self.hits += other.hits;
        self.misses += other.misses;
        self.parse_errors += other.parse_errors;
        self.llm_ms += other.llm_ms;
        self.exec_wall_ms += other.exec_wall_ms;
        self.tool_ms += other.tool_ms;
    }

    pub fn parallel_speedup(&self) -> f64 {
        if self.exec_wall_ms > 0 {
            (self.tool_ms as f64 / self.exec_wall_ms as f64).max(1.0)
        } else {
            1.0
        }
    }
}

impl fmt::Display for Stats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "turns: {} | tool calls: {} | auto-chained: {} | predictions: {} hit / {} miss | \
             parse errors: {} | llm: {:.1}s | tools: {:.1}s wall vs {:.1}s sequential ({:.1}x)",
            self.llm_turns,
            self.tool_calls,
            self.chained,
            self.hits,
            self.misses,
            self.parse_errors,
            self.llm_ms as f64 / 1000.0,
            self.exec_wall_ms as f64 / 1000.0,
            self.tool_ms as f64 / 1000.0,
            self.parallel_speedup()
        )
    }
}

#[derive(Debug, Clone)]
pub struct RunResult {
    /// `None` if the agent hit `max_turns` without finishing.
    pub answer: Option<String>,
    pub stats: Stats,
}

pub struct Agent {
    llm: LlmClient,
    exec: Executor,
    cfg: AgentConfig,
}

impl Agent {
    pub fn new(llm: LlmClient, exec: Executor, cfg: AgentConfig) -> Self {
        Self { llm, exec, cfg }
    }

    fn log(&self, msg: impl AsRef<str>) {
        if self.cfg.verbose {
            eprintln!("{}", msg.as_ref());
        }
    }

    fn emit(&self, ev: AgentEvent) {
        if let Some(tx) = &self.cfg.events {
            let _ = tx.send(ev);
        }
    }

    pub async fn run(&self, task: &str) -> Result<RunResult> {
        let mut msgs = Vec::new();
        self.run_with_history(task, &mut msgs).await
    }

    pub async fn run_with_history(&self, task: &str, msgs: &mut Vec<Message>) -> Result<RunResult> {
        let system = system_prompt_with(
            self.cfg.guess_mode,
            self.cfg.limits,
            self.cfg.optimistic_finish,
        );
        msgs.push(Message::user(format!(
            "TASK:\n{task}\n\nWorking directory: {}\nOS: {}\nRespond with your JSON plan.",
            self.exec.tools.workdir.display(),
            std::env::consts::OS
        )));
        let mut stats = Stats::default();
        let mut consecutive_errors = 0usize;

        for turn in 1..=self.cfg.max_turns {
            self.log(format!(
                "\n━━ turn {turn} ━━ asking {} ...",
                self.llm.model()
            ));
            self.emit(AgentEvent::TurnStarted {
                turn,
                max_turns: self.cfg.max_turns,
                model: self.llm.model().to_string(),
            });
            let t0 = Instant::now();
            let reply = self.llm.complete(&system, msgs).await?;
            stats.llm_turns += 1;
            stats.llm_ms += t0.elapsed().as_millis() as u64;
            msgs.push(Message::assistant(reply.clone()));

            let parsed = Plan::parse(&reply).and_then(|mut p| {
                p.validate(TOOL_NAMES, &self.cfg.limits)?;
                Ok(p)
            });
            let plan = match parsed {
                Ok(p) => {
                    consecutive_errors = 0;
                    p
                }
                Err(e) => {
                    stats.parse_errors += 1;
                    consecutive_errors += 1;
                    self.log(format!("⚠ invalid plan: {e}"));
                    self.emit(AgentEvent::ParseError(e.clone()));
                    if consecutive_errors >= 3 {
                        bail!("model produced an invalid plan 3 times in a row: {e}");
                    }
                    msgs.push(Message::user(format!(
                        "ERROR: {e}\nRespond again with ONLY one valid JSON plan object."
                    )));
                    continue;
                }
            };

            if let Some(t) = &plan.thought {
                self.log(format!("💭 {}", truncate_chars(t, 300)));
                self.emit(AgentEvent::Thought(t.clone()));
            }
            if plan.steps.is_empty() {
                if let Some(f) = plan.finish {
                    return Ok(RunResult {
                        answer: Some(f),
                        stats,
                    });
                }
            }

            let total_nodes = plan.node_count();
            self.log(format!(
                "⚡ executing {} step(s) in parallel ({} node(s) incl. predictions)",
                plan.steps.len(),
                total_nodes
            ));
            self.emit(AgentEvent::PlanReady {
                steps: plan.steps.len(),
                total_nodes,
            });

            let exec = if let Some(agent_tx) = &self.cfg.events {
                let (exec_tx, mut exec_rx) = mpsc::unbounded_channel();
                let forward_tx = agent_tx.clone();
                tokio::spawn(async move {
                    while let Some(ev) = exec_rx.recv().await {
                        let _ = forward_tx.send(AgentEvent::Exec(ev));
                    }
                });
                self.exec.clone().with_events(exec_tx)
            } else {
                self.exec.clone()
            };

            let report = exec
                .run_with_after_all(&plan.steps, &plan.after_all)
                .await;
            stats.tool_calls += report.tool_calls();
            stats.chained += report.chained();
            stats.hits += report.hits();
            stats.misses += report.misses();
            stats.exec_wall_ms += report.wall_ms;
            stats.tool_ms += report.tool_ms();
            for r in &report.records {
                self.log(format!("  {}", ExecReport::summary_line(r)));
            }
            self.emit(AgentEvent::TurnReport {
                turn,
                report: report.clone(),
                stats: stats.clone(),
            });

            let clean = report.clean();
            let finish = if self.cfg.optimistic_finish {
                report.finish.clone().or_else(|| plan.finish.clone())
            } else {
                None
            };
            if clean {
                if let Some(f) = finish {
                    self.log("🏁 predictions held – finishing without another round trip");
                    self.emit(AgentEvent::OptimisticFinish);
                    return Ok(RunResult {
                        answer: Some(f),
                        stats,
                    });
                }
            }

            let mut feedback = report.render(self.cfg.max_output);
            if (report.finish.is_some() || plan.finish.is_some()) && !clean {
                feedback.push_str(
                    "\nNOTE: a finish was requested but NOT accepted because a prediction missed \
                     or an un-predicted step failed. Review the results above.\n",
                );
            }
            feedback.push_str(&format!(
                "\nTurn {turn}/{} complete. Respond with the next JSON plan.",
                self.cfg.max_turns
            ));
            msgs.push(Message::user(feedback));
        }

        Ok(RunResult {
            answer: None,
            stats,
        })
    }
}
