//! The plan format the model emits each turn.

use std::collections::HashSet;

use serde::Deserialize;
use serde_json::Value;

use crate::condition::Condition;

/// One model turn: parallel `steps`, each with optional predicted branches.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct Plan {
    pub thought: Option<String>,
    pub steps: Vec<Node>,
    /// Final answer. With empty `steps` the task ends now; with steps it is an
    /// optimistic finish accepted only if every prediction hits.
    pub finish: Option<String>,
}

/// A single tool call plus predicted follow-ups.
#[derive(Debug, Clone, Deserialize)]
pub struct Node {
    #[serde(default)]
    pub id: String,
    pub tool: String,
    #[serde(default)]
    pub args: Value,
    #[serde(default)]
    pub predict: Vec<Branch>,
}

/// "If the result matches `when`, run `then` (in parallel) and optionally finish."
#[derive(Debug, Clone, Deserialize)]
pub struct Branch {
    #[serde(default)]
    pub when: Condition,
    #[serde(default)]
    pub then: Vec<Node>,
    #[serde(default)]
    pub finish: Option<String>,
}

#[derive(Debug, Clone, Copy)]
pub struct Limits {
    /// Maximum nesting depth of predicted branches (top-level steps are depth 0).
    pub max_depth: usize,
    /// Maximum total nodes in one plan.
    pub max_nodes: usize,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            max_depth: 4,
            max_nodes: 64,
        }
    }
}

impl Plan {
    /// Parse a plan from raw model output (tolerates code fences / surrounding prose).
    pub fn parse(text: &str) -> Result<Plan, String> {
        let json = crate::util::extract_json(text).ok_or("no JSON object found in response")?;
        serde_json::from_str(json).map_err(|e| format!("invalid plan JSON: {e}"))
    }

    /// Validate tools, ids, regexes and limits. Assigns ids to nodes that lack one.
    pub fn validate(&mut self, known_tools: &[&str], limits: &Limits) -> Result<(), String> {
        if self.steps.is_empty() && self.finish.is_none() {
            return Err("plan has no steps and no finish".into());
        }
        let mut ids = HashSet::new();
        let mut counter = 0usize;
        let mut total = 0usize;
        for n in self.steps.iter_mut() {
            validate_node(n, 0, known_tools, limits, &mut ids, &mut counter, &mut total)?;
        }
        Ok(())
    }

    /// Total number of nodes in the plan (all branches).
    pub fn node_count(&self) -> usize {
        fn count(n: &Node) -> usize {
            1 + n
                .predict
                .iter()
                .flat_map(|b| b.then.iter())
                .map(count)
                .sum::<usize>()
        }
        self.steps.iter().map(count).sum()
    }
}

fn validate_node(
    n: &mut Node,
    depth: usize,
    tools: &[&str],
    limits: &Limits,
    ids: &mut HashSet<String>,
    counter: &mut usize,
    total: &mut usize,
) -> Result<(), String> {
    *total += 1;
    if *total > limits.max_nodes {
        return Err(format!(
            "plan too large: more than {} nodes",
            limits.max_nodes
        ));
    }
    if depth >= limits.max_depth {
        return Err(format!(
            "prediction tree too deep (max depth {})",
            limits.max_depth
        ));
    }
    if n.id.trim().is_empty() {
        loop {
            *counter += 1;
            let candidate = format!("_{}", counter);
            if !ids.contains(&candidate) {
                n.id = candidate;
                break;
            }
        }
    }
    if !ids.insert(n.id.clone()) {
        return Err(format!("duplicate node id '{}'", n.id));
    }
    if !tools.contains(&n.tool.as_str()) {
        return Err(format!(
            "node '{}': unknown tool '{}'. Available tools: {}",
            n.id,
            n.tool,
            tools.join(", ")
        ));
    }
    let id = n.id.clone();
    for (i, b) in n.predict.iter_mut().enumerate() {
        b.when
            .validate()
            .map_err(|e| format!("node '{}' predict #{}: {}", id, i, e))?;
        for c in b.then.iter_mut() {
            validate_node(c, depth + 1, tools, limits, ids, counter, total)?;
        }
    }
    Ok(())
}
