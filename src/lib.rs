//! Blitz: a speculative, parallel AI agent engine with an interactive TUI.
//!
//! Two ideas make it fast:
//! * **Width** – every independent tool call in a plan runs in parallel.
//! * **Depth** – the model *predicts* likely outcomes of each call and attaches the
//!   follow-up steps for each prediction. The Rust runtime checks the real result
//!   against the predictions and, on a hit, runs the next steps immediately –
//!   no LLM round trip needed.

pub mod agent;
pub mod condition;
pub mod config;
pub mod exec;
pub mod llm;
pub mod plan;
pub mod prompt;
pub mod tools;
pub mod tui;
pub mod util;
