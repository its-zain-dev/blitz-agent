//! The agent loop: ask the model for a plan, execute it speculatively, feed results back.

use std::fmt;
use std::time::Instant;

use anyhow::{bail, Result};

use crate::exec::{ExecReport, Executor};
use crate::llm::{LlmClient, Message};
use crate::plan::{Limits, Plan};
use crate::prompt::system_prompt;
use crate::tools::TOOL_NAMES;
use crate::util::truncate_chars;

#[derive(Debug, Clone)]
pub struct AgentConfig {
    pub max_turns: usize,
    /// Max bytes of each tool's output that is shown to the model.
    pub max_output: usize,
    pub limits: Limits,
    /// Print progress to stderr.
    pub verbose: bool,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            max_turns: 30,
            max_output: 8000,
            limits: Limits::default(),
            verbose: false,
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

impl fmt::Display for Stats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let parallel_gain = if self.exec_wall_ms > 0 {
            self.tool_ms as f64 / self.exec_wall_ms as f64
        } else {
            1.0
        };
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
            parallel_gain
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

    pub async fn run(&self, task: &str) -> Result<RunResult> {
        let system = system_prompt(self.cfg.limits.max_depth);
        let mut msgs = vec![Message::user(format!(
            "TASK:\n{task}\n\nWorking directory: {}\nOS: {}\nRespond with your first JSON plan.",
            self.exec.tools.workdir.display(),
            std::env::consts::OS
        ))];
        let mut stats = Stats::default();
        let mut consecutive_errors = 0usize;

        for turn in 1..=self.cfg.max_turns {
            self.log(format!(
                "\n━━ turn {turn} ━━ asking {} ...",
                self.llm.model()
            ));
            let t0 = Instant::now();
            let reply = self.llm.complete(&system, &msgs).await?;
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
            }
            if plan.steps.is_empty() {
                if let Some(f) = plan.finish {
                    return Ok(RunResult {
                        answer: Some(f),
                        stats,
                    });
                }
            }

            self.log(format!(
                "⚡ executing {} step(s) in parallel ({} node(s) incl. predictions)",
                plan.steps.len(),
                plan.node_count()
            ));
            let report = self.exec.run(&plan.steps).await;
            stats.tool_calls += report.tool_calls();
            stats.chained += report.chained();
            stats.hits += report.hits();
            stats.misses += report.misses();
            stats.exec_wall_ms += report.wall_ms;
            stats.tool_ms += report.tool_ms();
            for r in &report.records {
                self.log(format!("  {}", ExecReport::summary_line(r)));
            }

            let clean = report.clean();
            let finish = report.finish.clone().or_else(|| plan.finish.clone());
            if clean {
                if let Some(f) = finish {
                    self.log("🏁 predictions held – finishing without another round trip");
                    return Ok(RunResult {
                        answer: Some(f),
                        stats,
                    });
                }
            }

            let mut feedback = report.render(self.cfg.max_output);
            if finish.is_some() && !clean {
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
