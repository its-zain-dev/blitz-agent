//! End-to-end agent loop tests using the scripted mock provider.

use std::time::Duration;

use blitz::agent::{Agent, AgentConfig};
use blitz::exec::Executor;
use blitz::llm::LlmClient;
use blitz::tools::ToolBox;

fn agent(dir: &std::path::Path, responses: &[&str]) -> Agent {
    let llm = LlmClient::mock(responses.iter().map(|s| s.to_string()).collect());
    let exec = Executor::new(ToolBox::new(dir.to_path_buf(), Duration::from_secs(30)), 8);
    Agent::new(llm, exec, AgentConfig::default())
}

/// A whole "create, run, verify" task finishes in ONE model turn when predictions hold.
#[tokio::test]
async fn predicted_task_finishes_in_one_turn() {
    let dir = tempfile::tempdir().unwrap();
    let plan = r#"{
      "thought": "write script, run it, verify output, finish",
      "steps": [
        {"id": "w", "tool": "write_file", "args": {"path": "hello.sh", "content": "echo hello blitz\n"},
         "predict": [{"when": {"success": true}, "then": [
            {"id": "run", "tool": "bash", "args": {"cmd": "sh hello.sh"},
             "predict": [{"when": {"contains": "hello blitz"}, "then": [], "finish": "hello.sh works"}]}
         ]}]},
        {"id": "ls", "tool": "list_dir", "args": {"path": "."}}
      ]
    }"#;
    let res = agent(dir.path(), &[plan])
        .run("make hello.sh")
        .await
        .unwrap();
    assert_eq!(res.answer.as_deref(), Some("hello.sh works"));
    assert_eq!(res.stats.llm_turns, 1);
    assert_eq!(res.stats.tool_calls, 3);
    assert_eq!(res.stats.chained, 1);
    assert_eq!(res.stats.misses, 0);
}

/// A missed prediction blocks the optimistic finish and hands control back to the model.
#[tokio::test]
async fn missed_prediction_returns_to_model() {
    let dir = tempfile::tempdir().unwrap();
    let first = r#"{"steps": [{"id": "t", "tool": "bash", "args": {"cmd": "echo boom >&2; exit 3"},
        "predict": [{"when": {"exit_code": 0}, "then": [], "finish": "too optimistic"}]}]}"#;
    let second = r#"{"thought": "saw the failure", "steps": [], "finish": "handled failure"}"#;
    let res = agent(dir.path(), &[first, second]).run("x").await.unwrap();
    assert_eq!(res.answer.as_deref(), Some("handled failure"));
    assert_eq!(res.stats.llm_turns, 2);
    assert_eq!(res.stats.misses, 1);
}

/// A plan-level optimistic finish is rejected if an un-predicted step fails.
#[tokio::test]
async fn optimistic_plan_finish_requires_clean_run() {
    let dir = tempfile::tempdir().unwrap();
    let first = r#"{"steps": [{"tool": "bash", "args": {"cmd": "false"}}], "finish": "premature"}"#;
    let second = r#"{"steps": [], "finish": "final"}"#;
    let res = agent(dir.path(), &[first, second]).run("x").await.unwrap();
    assert_eq!(res.answer.as_deref(), Some("final"));
    assert_eq!(res.stats.llm_turns, 2);

    let ok = r#"{"steps": [{"tool": "bash", "args": {"cmd": "true"}}], "finish": "accepted"}"#;
    let res = agent(dir.path(), &[ok]).run("x").await.unwrap();
    assert_eq!(res.answer.as_deref(), Some("accepted"));
    assert_eq!(res.stats.llm_turns, 1);
}

/// Invalid output (non-JSON, unknown tools) is reported back and the agent recovers.
#[tokio::test]
async fn recovers_from_invalid_plans() {
    let dir = tempfile::tempdir().unwrap();
    let res = agent(
        dir.path(),
        &[
            "I think I will run some commands",
            r#"{"steps": [{"tool": "hack_mainframe"}]}"#,
            r#"{"steps": [], "finish": "ok"}"#,
        ],
    )
    .run("x")
    .await
    .unwrap();
    assert_eq!(res.answer.as_deref(), Some("ok"));
    assert_eq!(res.stats.parse_errors, 2);
}

#[tokio::test]
async fn gives_up_after_three_invalid_plans() {
    let dir = tempfile::tempdir().unwrap();
    let err = agent(dir.path(), &["nope", "nope", "nope"])
        .run("x")
        .await
        .unwrap_err();
    assert!(err.to_string().contains("3 times"));
}

#[tokio::test]
async fn stops_at_max_turns() {
    let dir = tempfile::tempdir().unwrap();
    let step = r#"{"steps": [{"tool": "bash", "args": {"cmd": "true"}}]}"#;
    let llm = LlmClient::mock(vec![step.to_string(); 3]);
    let exec = Executor::new(
        ToolBox::new(dir.path().to_path_buf(), Duration::from_secs(30)),
        8,
    );
    let cfg = AgentConfig {
        max_turns: 3,
        ..AgentConfig::default()
    };
    let res = Agent::new(llm, exec, cfg).run("x").await.unwrap();
    assert!(res.answer.is_none());
    assert_eq!(res.stats.llm_turns, 3);
}

/// Same task, two strategies: a classic one-tool-per-turn agent vs Blitz.
#[tokio::test]
async fn speculative_agent_uses_fewer_turns_than_sequential() {
    let sequential = [
        r#"{"steps": [{"tool": "bash", "args": {"cmd": "mkdir -p app"}}]}"#,
        r#"{"steps": [{"tool": "write_file", "args": {"path": "app/a.txt", "content": "a"}}]}"#,
        r#"{"steps": [{"tool": "write_file", "args": {"path": "app/b.txt", "content": "b"}}]}"#,
        r#"{"steps": [{"tool": "bash", "args": {"cmd": "cat app/a.txt app/b.txt"}}]}"#,
        r#"{"steps": [], "finish": "done"}"#,
    ];
    let speculative = [r#"{"steps": [
        {"tool": "bash", "args": {"cmd": "mkdir -p app"}, "predict": [{"when": {"success": true}, "then": [
            {"id": "a", "tool": "write_file", "args": {"path": "app/a.txt", "content": "a"}},
            {"id": "b", "tool": "write_file", "args": {"path": "app/b.txt", "content": "b"},
             "predict": [{"when": {"success": true}, "then": [
                {"tool": "bash", "args": {"cmd": "sleep 0.2; cat app/a.txt app/b.txt"},
                 "predict": [{"when": {"contains": "ab"}, "then": [], "finish": "done"}]}
             ]}]}
        ]}]}
    ]}"#];

    let d1 = tempfile::tempdir().unwrap();
    let seq = agent(d1.path(), &sequential).run("x").await.unwrap();
    let d2 = tempfile::tempdir().unwrap();
    let spec = agent(d2.path(), &speculative).run("x").await.unwrap();

    assert_eq!(seq.answer.as_deref(), Some("done"));
    assert_eq!(spec.answer.as_deref(), Some("done"));
    assert_eq!(seq.stats.llm_turns, 5);
    assert_eq!(spec.stats.llm_turns, 1);
    assert_eq!(seq.stats.tool_calls, spec.stats.tool_calls);
}
