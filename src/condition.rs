//! Prediction conditions: structured checks evaluated against a tool result.
//!
//! All fields set in one condition must hold (logical AND).
//! An empty condition `{}` always matches and works as an "else" branch.

use regex::Regex;
use serde::Deserialize;

use crate::tools::ToolOutput;

#[derive(Debug, Clone, Default, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct Condition {
    /// Exact exit code.
    pub exit_code: Option<i32>,
    /// Exit code must differ from this.
    pub exit_code_not: Option<i32>,
    /// `true` => exit code 0, `false` => non-zero.
    pub success: Option<bool>,
    /// stdout+stderr contains this substring.
    pub contains: Option<String>,
    /// stdout+stderr does NOT contain this substring.
    pub not_contains: Option<String>,
    /// Regex matched against stdout+stderr.
    pub regex: Option<String>,
    /// Regex matched against stdout only.
    pub stdout_regex: Option<String>,
    /// stdout (trimmed) is empty / non-empty.
    pub stdout_empty: Option<bool>,
    pub any: Option<Vec<Condition>>,
    pub all: Option<Vec<Condition>>,
    pub not: Option<Box<Condition>>,
}

impl Condition {
    /// True if this condition has no checks (matches everything).
    pub fn is_always(&self) -> bool {
        *self == Condition::default()
    }

    pub fn eval(&self, out: &ToolOutput) -> bool {
        let text = out.combined();
        self.eval_with(out, &text)
    }

    fn eval_with(&self, out: &ToolOutput, text: &str) -> bool {
        if let Some(c) = self.exit_code {
            if out.exit_code != c {
                return false;
            }
        }
        if let Some(c) = self.exit_code_not {
            if out.exit_code == c {
                return false;
            }
        }
        if let Some(s) = self.success {
            if out.success() != s {
                return false;
            }
        }
        if let Some(p) = &self.contains {
            if !text.contains(p.as_str()) {
                return false;
            }
        }
        if let Some(p) = &self.not_contains {
            if text.contains(p.as_str()) {
                return false;
            }
        }
        if let Some(p) = &self.regex {
            match Regex::new(p) {
                Ok(re) => {
                    if !re.is_match(text) {
                        return false;
                    }
                }
                Err(_) => return false,
            }
        }
        if let Some(p) = &self.stdout_regex {
            match Regex::new(p) {
                Ok(re) => {
                    if !re.is_match(&out.stdout) {
                        return false;
                    }
                }
                Err(_) => return false,
            }
        }
        if let Some(e) = self.stdout_empty {
            if out.stdout.trim().is_empty() != e {
                return false;
            }
        }
        if let Some(list) = &self.any {
            if !list.iter().any(|c| c.eval_with(out, text)) {
                return false;
            }
        }
        if let Some(list) = &self.all {
            if !list.iter().all(|c| c.eval_with(out, text)) {
                return false;
            }
        }
        if let Some(c) = &self.not {
            if c.eval_with(out, text) {
                return false;
            }
        }
        true
    }

    /// Check that all regexes compile, recursively.
    pub fn validate(&self) -> Result<(), String> {
        for p in [&self.regex, &self.stdout_regex].into_iter().flatten() {
            Regex::new(p).map_err(|e| format!("invalid regex '{p}': {e}"))?;
        }
        for list in [&self.any, &self.all].into_iter().flatten() {
            for c in list {
                c.validate()?;
            }
        }
        if let Some(c) = &self.not {
            c.validate()?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn out(code: i32, stdout: &str, stderr: &str) -> ToolOutput {
        ToolOutput {
            exit_code: code,
            stdout: stdout.into(),
            stderr: stderr.into(),
            duration_ms: 0,
        }
    }

    fn cond(json: &str) -> Condition {
        serde_json::from_str(json).unwrap()
    }

    #[test]
    fn empty_condition_always_matches() {
        let c = cond("{}");
        assert!(c.is_always());
        assert!(c.eval(&out(1, "", "boom")));
    }

    #[test]
    fn exit_code_and_success() {
        assert!(cond(r#"{"exit_code": 0}"#).eval(&out(0, "", "")));
        assert!(!cond(r#"{"exit_code": 0}"#).eval(&out(2, "", "")));
        assert!(cond(r#"{"success": false}"#).eval(&out(2, "", "")));
        assert!(cond(r#"{"exit_code_not": 0}"#).eval(&out(5, "", "")));
    }

    #[test]
    fn text_checks_include_stderr() {
        let o = out(1, "ok", "ModuleNotFoundError: x");
        assert!(cond(r#"{"contains": "ModuleNotFound"}"#).eval(&o));
        assert!(!cond(r#"{"not_contains": "ModuleNotFound"}"#).eval(&o));
        assert!(cond(r#"{"regex": "Module\\w+Error"}"#).eval(&o));
        assert!(!cond(r#"{"stdout_regex": "Module"}"#).eval(&o));
    }

    #[test]
    fn combinators() {
        let o = out(0, "3 passed", "");
        assert!(cond(r#"{"any": [{"contains": "failed"}, {"contains": "passed"}]}"#).eval(&o));
        assert!(!cond(r#"{"all": [{"contains": "failed"}, {"contains": "passed"}]}"#).eval(&o));
        assert!(cond(r#"{"not": {"contains": "failed"}}"#).eval(&o));
        assert!(cond(r#"{"exit_code": 0, "stdout_empty": false}"#).eval(&o));
    }

    #[test]
    fn unknown_fields_are_rejected() {
        assert!(serde_json::from_str::<Condition>(r#"{"exitcode": 0}"#).is_err());
    }

    #[test]
    fn validate_catches_bad_regex() {
        assert!(cond(r#"{"any": [{"regex": "("}]}"#).validate().is_err());
        assert!(cond(r#"{"regex": "a+"}"#).validate().is_ok());
    }
}
