use std::time::Duration;

use blitz::tools::ToolBox;
use serde_json::json;

fn toolbox(dir: &std::path::Path) -> ToolBox {
    ToolBox::new(dir.to_path_buf(), Duration::from_secs(30))
}

#[tokio::test]
async fn bash_captures_stdout_stderr_and_exit_code() {
    let dir = tempfile::tempdir().unwrap();
    let t = toolbox(dir.path());
    let o = t
        .call("bash", &json!({"cmd": "echo out; echo err >&2; exit 7"}))
        .await;
    assert_eq!(o.exit_code, 7);
    assert_eq!(o.stdout.trim(), "out");
    assert_eq!(o.stderr.trim(), "err");
    assert!(o.combined().contains("out") && o.combined().contains("err"));
}

#[tokio::test]
async fn bash_runs_in_workdir() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("marker.txt"), "x").unwrap();
    let o = toolbox(dir.path())
        .call("bash", &json!({"cmd": "ls"}))
        .await;
    assert!(o.stdout.contains("marker.txt"));
}

#[tokio::test]
async fn bash_timeout_returns_124() {
    let dir = tempfile::tempdir().unwrap();
    let o = toolbox(dir.path())
        .call("bash", &json!({"cmd": "sleep 5", "timeout_secs": 1}))
        .await;
    assert_eq!(o.exit_code, 124);
    assert!(o.stderr.contains("timed out"));
    assert!(o.duration_ms < 4000);
}

#[tokio::test]
async fn write_then_read_with_line_range() {
    let dir = tempfile::tempdir().unwrap();
    let t = toolbox(dir.path());
    let w = t
        .call(
            "write_file",
            &json!({"path": "a/b/c.txt", "content": "one\ntwo\nthree\nfour\n"}),
        )
        .await;
    assert!(w.success(), "{w:?}");
    let all = t.call("read_file", &json!({"path": "a/b/c.txt"})).await;
    assert_eq!(all.stdout, "one\ntwo\nthree\nfour\n");
    let part = t
        .call(
            "read_file",
            &json!({"path": "a/b/c.txt", "start_line": 2, "end_line": 3, "numbered": true}),
        )
        .await;
    assert!(part.stdout.contains("2| two"));
    assert!(part.stdout.contains("3| three"));
    assert!(!part.stdout.contains("one"));
    let missing = t.call("read_file", &json!({"path": "nope.txt"})).await;
    assert_eq!(missing.exit_code, 1);
}

#[tokio::test]
async fn edit_file_requires_unique_match() {
    let dir = tempfile::tempdir().unwrap();
    let t = toolbox(dir.path());
    std::fs::write(
        dir.path().join("f.rs"),
        "let a = 1;\nlet a = 1;\nlet b = 2;\n",
    )
    .unwrap();

    let dup = t
        .call(
            "edit_file",
            &json!({"path": "f.rs", "old": "let a = 1;", "new": "let a = 9;"}),
        )
        .await;
    assert_eq!(dup.exit_code, 1);
    assert!(dup.stderr.contains("2 times"));

    let one = t
        .call(
            "edit_file",
            &json!({"path": "f.rs", "old": "let b = 2;", "new": "let b = 3;"}),
        )
        .await;
    assert!(one.success());

    let all = t
        .call(
            "edit_file",
            &json!({"path": "f.rs", "old": "let a = 1;", "new": "let a = 9;", "replace_all": true}),
        )
        .await;
    assert!(all.success());
    assert_eq!(
        std::fs::read_to_string(dir.path().join("f.rs")).unwrap(),
        "let a = 9;\nlet a = 9;\nlet b = 3;\n"
    );

    let none = t
        .call(
            "edit_file",
            &json!({"path": "f.rs", "old": "zzz", "new": "y"}),
        )
        .await;
    assert_eq!(none.exit_code, 1);
}

#[tokio::test]
async fn list_dir_marks_directories() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::create_dir(dir.path().join("sub")).unwrap();
    std::fs::write(dir.path().join("file.txt"), "").unwrap();
    let o = toolbox(dir.path()).call("list_dir", &json!({})).await;
    assert_eq!(o.stdout, "file.txt\nsub/");
}

#[tokio::test]
async fn search_finds_matches_and_skips_build_dirs() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(dir.path().join("src")).unwrap();
    std::fs::create_dir_all(dir.path().join("target")).unwrap();
    std::fs::write(
        dir.path().join("src/main.rs"),
        "fn main() {}\n// TODO: speed\n",
    )
    .unwrap();
    std::fs::write(dir.path().join("target/gen.rs"), "// TODO: ignored\n").unwrap();
    let t = toolbox(dir.path());
    let o = t.call("search", &json!({"pattern": "TODO"})).await;
    assert!(o.success());
    assert!(o.stdout.contains("src/main.rs:2:"), "{}", o.stdout);
    assert!(!o.stdout.contains("ignored"));
    let none = t
        .call("search", &json!({"pattern": "nothing_here_123"}))
        .await;
    assert_eq!(none.exit_code, 1);
    let bad = t.call("search", &json!({"pattern": "("})).await;
    assert_eq!(bad.exit_code, 1);
}

#[tokio::test]
async fn unknown_tool_and_missing_args_fail_gracefully() {
    let dir = tempfile::tempdir().unwrap();
    let t = toolbox(dir.path());
    assert_eq!(t.call("teleport", &json!({})).await.exit_code, 1);
    assert_eq!(t.call("bash", &json!({})).await.exit_code, 1);
    assert_eq!(t.call("write_file", &json!({})).await.exit_code, 1);
}
