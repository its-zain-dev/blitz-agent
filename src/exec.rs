//! The speculative, parallel executor.
//!
//! * All top-level steps run concurrently (bounded by a semaphore).
//! * When a step finishes, its predicted branches are checked in order; the first
//!   match spawns that branch's `then` nodes concurrently – no LLM round trip.
//! * Once all top-level steps finish, optional `after_all` join branches can spawn
//!   follow-up steps that combine outputs from any of the parallel steps.
//! * Predicted child nodes can reference parent/earlier outputs via `{{stdout:id}}`,
//!   `{{stderr:id}}`, and `{{exit:id}}` placeholders in any string argument.

use std::sync::{Arc, Mutex};
use std::time::Instant;

use futures::future::{join_all, BoxFuture, FutureExt};
use serde_json::Value;
use tokio::sync::{mpsc, Semaphore};

use crate::plan::{Branch, Node};
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

/// Live events emitted during plan execution (used by the TUI).
#[derive(Debug, Clone)]
pub enum ExecEvent {
    NodeStarted {
        id: String,
        parent: Option<String>,
        depth: usize,
        tool: String,
        args: Value,
    },
    NodeFinished(NodeRecord),
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

/// Replace `{{stdout:id}}`, `{{stderr:id}}`, `{{exit:id}}` in string values of `val`
/// using completed `records`.
fn interpolate_args(val: &Value, records: &[NodeRecord]) -> Value {
    match val {
        Value::String(s) => {
            if !s.contains("{{") {
                return Value::String(s.clone());
            }
            let mut out = s.clone();
            for r in records {
                let stdout_tag = format!("{{{{stdout:{}}}}}", r.id);
                if out.contains(&stdout_tag) {
                    out = out.replace(&stdout_tag, r.output.stdout.trim_end());
                }
                let stderr_tag = format!("{{{{stderr:{}}}}}", r.id);
                if out.contains(&stderr_tag) {
                    out = out.replace(&stderr_tag, r.output.stderr.trim_end());
                }
                let exit_tag = format!("{{{{exit:{}}}}}", r.id);
                if out.contains(&exit_tag) {
                    out = out.replace(&exit_tag, &r.output.exit_code.to_string());
                }
            }
            Value::String(out)
        }
        Value::Array(arr) => Value::Array(
            arr.iter()
                .map(|v| interpolate_args(v, records))
                .collect(),
        ),
        Value::Object(map) => Value::Object(
            map.iter()
                .map(|(k, v)| (k.clone(), interpolate_args(v, records)))
                .collect(),
        ),
        other => other.clone(),
    }
}

#[derive(Clone)]
pub struct Executor {
    pub tools: Arc<ToolBox>,
    sem: Arc<Semaphore>,
    events: Option<mpsc::UnboundedSender<ExecEvent>>,
}

impl Executor {
    pub fn new(tools: ToolBox, concurrency: usize) -> Self {
        Self {
            tools: Arc::new(tools),
            sem: Arc::new(Semaphore::new(concurrency.max(1))),
            events: None,
        }
    }

    pub fn with_events(mut self, tx: mpsc::UnboundedSender<ExecEvent>) -> Self {
        self.events = Some(tx);
        self
    }

    /// Execute a set of top-level steps (and every predicted branch that hits).
    pub async fn run(&self, steps: &[Node]) -> ExecReport {
        self.run_with_after_all(steps, &[]).await
    }

    /// Execute top-level steps in parallel, then evaluate `after_all` join branches.
    pub async fn run_with_after_all(&self, steps: &[Node], after_all: &[Branch]) -> ExecReport {
        let start = Instant::now();
        let shared = Arc::new(Mutex::new(Shared::default()));
        let futs: Vec<_> = steps
            .iter()
            .cloned()
            .map(|n| self.clone().run_node(n, None, 0, Arc::clone(&shared)))
            .collect();
        join_all(futs).await;

        if !after_all.is_empty() {
            let synthetic = {
                let s = shared.lock().unwrap();
                let all_ok = s.records.iter().all(|r| r.output.success());
                let combined_stdout = s
                    .records
                    .iter()
                    .map(|r| r.output.stdout.as_str())
                    .collect::<Vec<_>>()
                    .join("\n");
                let combined_stderr = s
                    .records
                    .iter()
                    .map(|r| r.output.stderr.as_str())
                    .collect::<Vec<_>>()
                    .join("\n");
                ToolOutput {
                    exit_code: if all_ok { 0 } else { 1 },
                    stdout: combined_stdout,
                    stderr: combined_stderr,
                    duration_ms: 0,
                }
            };

            let mut join_children: Vec<Node> = Vec::new();
            for b in after_all {
                if b.when.eval(&synthetic) {
                    if let Some(f) = &b.finish {
                        let mut s = shared.lock().unwrap();
                        if s.finish.is_none() {
                            let interpolated =
                                match interpolate_args(&Value::String(f.clone()), &s.records) {
                                    Value::String(st) => st,
                                    _ => f.clone(),
                                };
                            s.finish = Some(interpolated);
                        }
                    }
                    join_children = b.then.clone();
                    break;
                }
            }
            if !join_children.is_empty() {
                let jfuts: Vec<_> = join_children
                    .into_iter()
                    .map(|c| {
                        self.clone()
                            .run_node(c, Some("all".into()), 1, Arc::clone(&shared))
                    })
                    .collect();
                join_all(jfuts).await;
            }
        }

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
            let resolved_args = {
                let s = shared.lock().unwrap();
                interpolate_args(&node.args, &s.records)
            };

            if let Some(tx) = &self.events {
                let _ = tx.send(ExecEvent::NodeStarted {
                    id: node.id.clone(),
                    parent: parent.clone(),
                    depth,
                    tool: node.tool.clone(),
                    args: resolved_args.clone(),
                });
            }

            let output = {
                let _permit = self.sem.acquire().await.expect("semaphore closed");
                self.tools.call(&node.tool, &resolved_args).await
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
                            let interpolated_finish =
                                match interpolate_args(&Value::String(f.clone()), &s.records) {
                                    Value::String(st) => st,
                                    _ => f.clone(),
                                };
                            let stdout_self = format!("{{{{stdout:{}}}}}", node.id);
                            let final_f = interpolated_finish
                                .replace(&stdout_self, output.stdout.trim_end());
                            s.finish = Some(final_f);
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
                args: resolved_args,
                output,
                outcome,
            };
            if let Some(tx) = &self.events {
                let _ = tx.send(ExecEvent::NodeFinished(record.clone()));
            }
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
