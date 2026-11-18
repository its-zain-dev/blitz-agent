use blitz::plan::{Limits, Plan};
use blitz::tools::TOOL_NAMES;

fn validate(json: &str, limits: Limits) -> Result<Plan, String> {
    let mut p = Plan::parse(json)?;
    p.validate(TOOL_NAMES, &limits)?;
    Ok(p)
}

#[test]
fn parses_fenced_json_with_prose() {
    let text = "Here is my plan:\n```json\n{\"thought\": \"t\", \"steps\": [{\"tool\": \"list_dir\"}]}\n```";
    let p = validate(text, Limits::default()).unwrap();
    assert_eq!(p.steps.len(), 1);
    assert_eq!(p.thought.as_deref(), Some("t"));
}

#[test]
fn assigns_ids_to_unnamed_nodes() {
    let p = validate(
        r#"{"steps": [{"tool": "list_dir"}, {"id": "_1", "tool": "list_dir"}, {"tool": "list_dir"}]}"#,
        Limits::default(),
    )
    .unwrap();
    let ids: Vec<_> = p.steps.iter().map(|n| n.id.clone()).collect();
    let unique: std::collections::HashSet<_> = ids.iter().collect();
    assert_eq!(unique.len(), 3, "{ids:?}");
}

#[test]
fn rejects_unknown_tool() {
    let e = validate(r#"{"steps": [{"tool": "rm_rf"}]}"#, Limits::default()).unwrap_err();
    assert!(e.contains("unknown tool"));
}

#[test]
fn rejects_duplicate_ids_across_branches() {
    let e = validate(
        r#"{"steps": [{"id": "x", "tool": "list_dir", "predict": [
            {"when": {}, "then": [{"id": "x", "tool": "list_dir"}]}
        ]}]}"#,
        Limits::default(),
    )
    .unwrap_err();
    assert!(e.contains("duplicate"));
}

#[test]
fn enforces_depth_and_node_limits() {
    let deep = r#"{"steps": [{"tool": "list_dir", "predict": [{"when": {}, "then": [
        {"tool": "list_dir", "predict": [{"when": {}, "then": [{"tool": "list_dir"}]}]}
    ]}]}]}"#;
    assert!(validate(
        deep,
        Limits {
            max_depth: 3,
            max_nodes: 64,
            max_branches: 5,
        }
    )
    .is_ok());
    let e = validate(
        deep,
        Limits {
            max_depth: 2,
            max_nodes: 64,
            max_branches: 5,
        },
    )
    .unwrap_err();
    assert!(e.contains("too deep"));
    let e = validate(
        deep,
        Limits {
            max_depth: 9,
            max_nodes: 2,
            max_branches: 5,
        },
    )
    .unwrap_err();
    assert!(e.contains("too large"));
}

#[test]
fn rejects_bad_regex_and_empty_plan() {
    let e = validate(
        r#"{"steps": [{"tool": "list_dir", "predict": [{"when": {"regex": "(["}}]}]}"#,
        Limits::default(),
    )
    .unwrap_err();
    assert!(e.contains("invalid regex"));
    assert!(validate(r#"{"steps": []}"#, Limits::default()).is_err());
    assert!(validate(r#"{"steps": [], "finish": "done"}"#, Limits::default()).is_ok());
    assert!(Plan::parse("no json here").is_err());
}

#[test]
fn counts_nodes_in_all_branches() {
    let p = validate(
        r#"{"steps": [
            {"tool": "list_dir", "predict": [
                {"when": {"success": true}, "then": [{"tool": "list_dir"}, {"tool": "list_dir"}]},
                {"when": {}, "then": [{"tool": "list_dir"}]}
            ]},
            {"tool": "list_dir"}
        ]}"#,
        Limits::default(),
    )
    .unwrap();
    assert_eq!(p.node_count(), 5);
}
