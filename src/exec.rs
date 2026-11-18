//! The speculative, parallel executor.
//!
//! * All top-level steps run concurrently (bounded by a semaphore).
//! * When a step finishes, its predicted branches are checked in order; the first
//!   match spawns that branch's `then` nodes concurrently – no LLM round trip.

use std::sync::{Arc, Mutex};
use std::time::Instant;

use futures::future::{join_all, BoxFuture, FutureExt};
use serde_json::Value;
use tokio::sync::Semaphore;

use crate::plan::Node;
use crate::tools::{ToolBox, ToolOutput};
use crate::util::{truncate_chars, truncate_middle};

#[derive(Debug, Clone, PartialEq)]
pub enum Outcome {
    /// The node had no predictions.
    NoPrediction,
    /// Prediction `branch` matched; `spawned` are the ids it auto-ran.
    Hit { branch: usize, spawned: Vec<String> },
    /// None of the `predictions` matched – the model must decide.
    Miss { predictions: usize },
}

#[derive(Debug, Clone)]
pub struct NodeRecord {
    pub id: String,
    pub parent: Option<String>,
    pub depth: usize,
    pub tool: String,
    pub args: Value,
    pub output: ToolOutput,
    pub outcome: Outcome,
}

#[derive(Debug, Clone, Default)]
pub struct ExecReport {
    /// Records in completion order.
    pub records: Vec<NodeRecord>,
    /// Set if a matched branch carried a `finish`.
    pub finish: Option<String>,
    pub wall_ms: u64,
}

#[derive(Default)]
struct Shared {
    records: Vec<NodeRecord>,
    finish: Option<String>,
}

#[derive(Clone)]
pub struct Executor {
    pub tools: Arc<ToolBox>,
    sem: Arc<Semaphore>,
}

impl Executor {
    pub fn new(tools: ToolBox, concurrency: usize) -> Self {
        Self {
            tools: Arc::new(tools),
            sem: Arc::new(Semaphore::new(concurrency.max(1))),
        }
    }

    /// Execute a set of top-level steps (and every predicted branch that hits).
    pub async fn run(&self, steps: &[Node]) -> ExecReport {
        let start = Instant::now();
        let shared = Arc::new(Mutex::new(Shared::default()));
        let futs: Vec<_> = steps
            .iter()
            .cloned()
            .map(|n| self.clone().run_node(n, None, 0, Arc::clone(&shared)))
            .collect();
        join_all(futs).await;
        let s = std::mem::take(&mut *shared.lock().unwrap());
        ExecReport {
            records: s.records,
            finish: s.finish,
            wall_ms: start.elapsed().as_millis() as u64,
        }
    }

    fn run_node(
        self,
        node: Node,
        parent: Option<String>,
        depth: usize,
        shared: Arc<Mutex<Shared>>,
    ) -> BoxFuture<'static, ()> {
        async move {
            let output = {
                let _permit = self.sem.acquire().await.expect("semaphore closed");
                self.tools.call(&node.tool, &node.args).await
            };

            let mut outcome = if node.predict.is_empty() {
                Outcome::NoPrediction
            } else {
                Outcome::Miss {
                    predictions: node.predict.len(),
                }
            };
            let mut children: Vec<Node> = Vec::new();
            for (i, b) in node.predict.iter().enumerate() {
                if b.when.eval(&output) {
                    outcome = Outcome::Hit {
                        branch: i,
                        spawned: b.then.iter().map(|c| c.id.clone()).collect(),
                    };
                    if let Some(f) = &b.finish {
                        let mut s = shared.lock().unwrap();
                        if s.finish.is_none() {
                            s.finish = Some(f.clone());
                        }
                    }
                    children = b.then.clone();
                    break;
                }
            }

            let record = NodeRecord {
                id: node.id.clone(),
                parent,
                depth,
                tool: node.tool.clone(),
                args: node.args.clone(),
                output,
                outcome,
            };
            shared.lock().unwrap().records.push(record);

            let parent_id = node.id.clone();
            let futs: Vec<_> = children
                .into_iter()
                .map(|c| {
                    self.clone().run_node(
                        c,
                        Some(parent_id.clone()),
                        depth + 1,
                        Arc::clone(&shared),
                    )
                })
                .collect();
            join_all(futs).await;
        }
        .boxed()
    }
}

impl ExecReport {
    pub fn tool_calls(&self) -> usize {
        self.records.len()
    }

    /// Nodes that ran automatically because a prediction hit (≈ round trips saved).
    pub fn chained(&self) -> usize {
        self.records.iter().filter(|r| r.depth > 0).count()
    }

    pub fn hits(&self) -> usize {
        self.records
            .iter()
            .filter(|r| matches!(r.outcome, Outcome::Hit { .. }))
            .count()
    }

    pub fn misses(&self) -> usize {
        self.records
            .iter()
            .filter(|r| matches!(r.outcome, Outcome::Miss { .. }))
            .count()
    }

    /// Sum of individual tool durations (compare with `wall_ms` to see parallel gain).
    pub fn tool_ms(&self) -> u64 {
        self.records.iter().map(|r| r.output.duration_ms).sum()
    }

    pub fn record(&self, id: &str) -> Option<&NodeRecord> {
        self.records.iter().find(|r| r.id == id)
    }

    /// No prediction missed and no un-predicted step failed.
    /// Finishing (optimistic or branch) is only accepted when this holds.
    pub fn clean(&self) -> bool {
        self.records.iter().all(|r| match r.outcome {
            Outcome::Miss { .. } => false,
            Outcome::NoPrediction => r.output.success(),
            Outcome::Hit { .. } => true,
        })
    }

    /// One-line summary of a record (for logs).
    pub fn summary_line(r: &NodeRecord) -> String {
        let indent = "  ".repeat(r.depth);
        let outcome = match &r.outcome {
            Outcome::NoPrediction => String::new(),
            Outcome::Hit { branch, spawned } => {
                if spawned.is_empty() {
                    format!("  ✔ HIT #{branch}")
                } else {
                    format!("  ✔ HIT #{branch} → auto-run [{}]", spawned.join(", "))
                }
            }
            Outcome::Miss { predictions } => format!("  ✘ MISS (0/{predictions})"),
        };
        format!(
            "{indent}[{}] {} {} → exit {} ({} ms){}",
            r.id,
            r.tool,
            truncate_chars(&r.args.to_string(), 100),
            r.output.exit_code,
            r.output.duration_ms,
            outcome
        )
    }

    /// Full report sent back to the model.
    pub fn render(&self, max_output: usize) -> String {
        let mut s = format!(
            "EXECUTION REPORT: {} tool calls, {} ms wall time ({} ms of tool time), \
             {} auto-chained via predictions; predictions: {} hit, {} miss.\n",
            self.tool_calls(),
            self.wall_ms,
            self.tool_ms(),
            self.chained(),
            self.hits(),
            self.misses()
        );
        for r in &self.records {
            let via = r
                .parent
                .as_ref()
                .map(|p| format!(" (auto-chained after [{p}])"))
                .unwrap_or_default();
            let outcome = match &r.outcome {
                Outcome::NoPrediction => String::new(),
                Outcome::Hit { branch, .. } => format!(" | prediction HIT #{branch}"),
                Outcome::Miss { predictions } => format!(
                    " | prediction MISS (none of {predictions} matched) -> needs your decision"
                ),
            };
            s.push_str(&format!(
                "\n[{}] {} {}{} -> exit {} ({} ms){}\n",
                r.id,
                r.tool,
                truncate_chars(&r.args.to_string(), 200),
                via,
                r.output.exit_code,
                r.output.duration_ms,
                outcome
            ));
            let body = r.output.display();
            let body = body.trim_end();
            if !body.is_empty() {
                s.push_str(&truncate_middle(body, max_output));
                s.push('\n');
            }
        }
        if let Some(f) = &self.finish {
            s.push_str(&format!("\nA prediction branch requested finish: {f}\n"));
        }
        s
    }
}
