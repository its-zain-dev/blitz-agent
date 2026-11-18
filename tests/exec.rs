use std::time::Duration;

use blitz::exec::{Executor, Outcome};
use blitz::plan::{Limits, Plan};
use blitz::tools::{ToolBox, TOOL_NAMES};

fn executor(dir: &std::path::Path, concurrency: usize) -> Executor {
    Executor::new(
        ToolBox::new(dir.to_path_buf(), Duration::from_secs(30)),
        concurrency,
    )
}

fn plan(json: &str) -> Plan {
    let mut p = Plan::parse(json).expect("parse");
    p.validate(TOOL_NAMES, &Limits::default())
        .expect("validate");
    p
}

#[tokio::test]
async fn parallel_steps_run_concurrently() {
    let dir = tempfile::tempdir().unwrap();
    let p = plan(
        r#"{"steps": [
            {"id": "a", "tool": "bash", "args": {"cmd": "sleep 1"}},
            {"id": "b", "tool": "bash", "args": {"cmd": "sleep 1"}},
            {"id": "c", "tool": "bash", "args": {"cmd": "sleep 1"}},
            {"id": "d", "tool": "bash", "args": {"cmd": "sleep 1"}},
            {"id": "e", "tool": "bash", "args": {"cmd": "sleep 1"}}
        ]}"#,
    );
    let r = executor(dir.path(), 16).run(&p.steps).await;
    assert_eq!(r.tool_calls(), 5);
    assert!(r.tool_ms() >= 4900, "sequential time {}", r.tool_ms());
    assert!(r.wall_ms < 3000, "5 x sleep 1 took {} ms wall", r.wall_ms);
    assert!(r.clean());
}

#[tokio::test]
async fn concurrency_limit_is_respected() {
    let dir = tempfile::tempdir().unwrap();
    let p = plan(
        r#"{"steps": [
            {"tool": "bash", "args": {"cmd": "sleep 0.3"}},
            {"tool": "bash", "args": {"cmd": "sleep 0.3"}},
            {"tool": "bash", "args": {"cmd": "sleep 0.3"}}
        ]}"#,
    );
    let r = executor(dir.path(), 1).run(&p.steps).await;
    assert!(
        r.wall_ms >= 850,
        "should be serialized, took {} ms",
        r.wall_ms
    );
}

#[tokio::test]
async fn prediction_hits_auto_chain_three_levels() {
    let dir = tempfile::tempdir().unwrap();
    let p = plan(
        r#"{"steps": [
            {"id": "mk", "tool": "bash", "args": {"cmd": "mkdir -p out && echo made"},
             "predict": [{"when": {"exit_code": 0, "contains": "made"}, "then": [
                {"id": "write", "tool": "write_file", "args": {"path": "out/x.txt", "content": "speed\n"},
                 "predict": [{"when": {"success": true}, "then": [
                    {"id": "check", "tool": "bash", "args": {"cmd": "cat out/x.txt"},
                     "predict": [{"when": {"stdout_regex": "^speed"}, "then": [], "finish": "chain complete"}]}
                 ]}]}
             ]}]}
        ]}"#,
    );
    let r = executor(dir.path(), 8).run(&p.steps).await;
    assert_eq!(r.tool_calls(), 3);
    assert_eq!(r.chained(), 2);
    assert_eq!(r.hits(), 3);
    assert_eq!(r.misses(), 0);
    assert!(r.clean());
    assert_eq!(r.finish.as_deref(), Some("chain complete"));
    assert_eq!(r.record("check").unwrap().depth, 2);
    assert_eq!(r.record("check").unwrap().parent.as_deref(), Some("write"));
    assert_eq!(
        std::fs::read_to_string(dir.path().join("out/x.txt")).unwrap(),
        "speed\n"
    );
}

#[tokio::test]
async fn first_matching_branch_wins_and_else_catches_rest() {
    let dir = tempfile::tempdir().unwrap();
    let p = plan(
        r#"{"steps": [
            {"id": "s", "tool": "bash", "args": {"cmd": "echo 'error: ModuleNotFoundError' >&2; exit 1"},
             "predict": [
                {"when": {"exit_code": 0}, "then": [{"id": "ok", "tool": "bash", "args": {"cmd": "echo ok"}}]},
                {"when": {"contains": "ModuleNotFoundError"}, "then": [{"id": "fix", "tool": "bash", "args": {"cmd": "echo fixing"}}]},
                {"when": {}, "then": [{"id": "other", "tool": "bash", "args": {"cmd": "echo other"}}]}
             ]}
        ]}"#,
    );
    let r = executor(dir.path(), 8).run(&p.steps).await;
    assert!(r.record("ok").is_none());
    assert!(r.record("other").is_none());
    assert_eq!(r.record("fix").unwrap().output.stdout.trim(), "fixing");
    match &r.record("s").unwrap().outcome {
        Outcome::Hit { branch, spawned } => {
            assert_eq!(*branch, 1);
            assert_eq!(spawned, &vec!["fix".to_string()]);
        }
        o => panic!("expected hit, got {o:?}"),
    }
    assert!(r.clean());
}

#[tokio::test]
async fn miss_does_not_run_children_and_is_not_clean() {
    let dir = tempfile::tempdir().unwrap();
    let p = plan(
        r#"{"steps": [
            {"id": "s", "tool": "bash", "args": {"cmd": "exit 3"},
             "predict": [{"when": {"exit_code": 0}, "then": [
                {"id": "never", "tool": "write_file", "args": {"path": "never.txt", "content": "x"}}
             ], "finish": "should not finish"}]}
        ]}"#,
    );
    let r = executor(dir.path(), 8).run(&p.steps).await;
    assert_eq!(r.tool_calls(), 1);
    assert_eq!(r.misses(), 1);
    assert!(!r.clean());
    assert!(r.finish.is_none());
    assert!(!dir.path().join("never.txt").exists());
    assert!(r.render(1000).contains("MISS"));
}

#[tokio::test]
async fn unpredicted_failure_is_not_clean() {
    let dir = tempfile::tempdir().unwrap();
    let p = plan(r#"{"steps": [{"tool": "bash", "args": {"cmd": "false"}}]}"#);
    let r = executor(dir.path(), 8).run(&p.steps).await;
    assert!(!r.clean());
}

#[tokio::test]
async fn render_truncates_long_output() {
    let dir = tempfile::tempdir().unwrap();
    let p = plan(r#"{"steps": [{"id": "big", "tool": "bash", "args": {"cmd": "seq 1 100000"}}]}"#);
    let r = executor(dir.path(), 8).run(&p.steps).await;
    let text = r.render(2000);
    assert!(text.contains("bytes omitted"));
    assert!(text.len() < 4000);
}

#[tokio::test]
async fn interpolates_stdout_in_predicted_children() {
    let dir = tempfile::tempdir().unwrap();
    let p = plan(
        r#"{"steps": [
            {"id": "sys", "tool": "bash", "args": {"cmd": "echo KERNEL_42"},
             "predict": [{"when": {"success": true}, "then": [
                {"id": "save", "tool": "write_file",
                 "args": {"path": "summary.md", "content": "Info: {{stdout:sys}}\n"},
                 "predict": [{"when": {"success": true}, "then": [], "finish": "Saved {{stdout:sys}}"}]}
             ]}]}
        ]}"#,
    );
    let r = executor(dir.path(), 8).run(&p.steps).await;
    assert!(r.clean());
    assert_eq!(r.finish.as_deref(), Some("Saved KERNEL_42"));
    assert_eq!(
        std::fs::read_to_string(dir.path().join("summary.md")).unwrap(),
        "Info: KERNEL_42\n"
    );
}

#[tokio::test]
async fn after_all_joins_parallel_steps() {
    let dir = tempfile::tempdir().unwrap();
    let p = plan(
        r#"{
            "steps": [
                {"id": "a", "tool": "bash", "args": {"cmd": "echo ALPHA"}},
                {"id": "b", "tool": "bash", "args": {"cmd": "echo BETA"}}
            ],
            "after_all": [
                {"when": {"success": true}, "then": [
                    {"id": "join", "tool": "write_file",
                     "args": {"path": "joined.txt", "content": "{{stdout:a}}+{{stdout:b}}"},
                     "predict": [{"when": {"success": true}, "then": [], "finish": "joined={{stdout:a}}+{{stdout:b}}"}]}
                ]}
            ]
        }"#,
    );
    let r = executor(dir.path(), 8)
        .run_with_after_all(&p.steps, &p.after_all)
        .await;
    assert!(r.clean());
    assert_eq!(r.tool_calls(), 3);
    assert_eq!(r.finish.as_deref(), Some("joined=ALPHA+BETA"));
    assert_eq!(
        std::fs::read_to_string(dir.path().join("joined.txt")).unwrap(),
        "ALPHA+BETA"
    );
}
