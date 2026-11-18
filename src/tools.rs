//! Built-in tools. Every tool returns a uniform [`ToolOutput`].

use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::{Duration, Instant};

use regex::Regex;
use serde_json::Value;
use tokio::process::Command;

pub const TOOL_NAMES: &[&str] = &[
    "bash",
    "read_file",
    "write_file",
    "edit_file",
    "list_dir",
    "search",
];

const SKIP_DIRS: &[&str] = &[".git", "target", "node_modules", ".venv", "__pycache__"];

#[derive(Debug, Clone, Default, PartialEq)]
pub struct ToolOutput {
    pub exit_code: i32,
    pub stdout: String,
    pub stderr: String,
    pub duration_ms: u64,
}

impl ToolOutput {
    pub fn ok(stdout: impl Into<String>) -> Self {
        Self {
            exit_code: 0,
            stdout: stdout.into(),
            ..Default::default()
        }
    }

    pub fn err(stderr: impl Into<String>) -> Self {
        Self {
            exit_code: 1,
            stderr: stderr.into(),
            ..Default::default()
        }
    }

    pub fn success(&self) -> bool {
        self.exit_code == 0
    }

    /// stdout + stderr, used for condition matching.
    pub fn combined(&self) -> String {
        if self.stderr.is_empty() {
            self.stdout.clone()
        } else if self.stdout.is_empty() {
            self.stderr.clone()
        } else {
            format!("{}\n{}", self.stdout, self.stderr)
        }
    }

    /// Human/model-readable rendering with stderr labelled.
    pub fn display(&self) -> String {
        if self.stderr.trim().is_empty() {
            self.stdout.clone()
        } else if self.stdout.trim().is_empty() {
            format!("[stderr]\n{}", self.stderr)
        } else {
            format!("{}\n[stderr]\n{}", self.stdout, self.stderr)
        }
    }
}

#[derive(Debug, Clone)]
pub struct ToolBox {
    pub workdir: PathBuf,
    pub default_timeout: Duration,
}

fn arg_str<'a>(args: &'a Value, key: &str) -> Option<&'a str> {
    args.get(key).and_then(|v| v.as_str())
}

fn arg_u64(args: &Value, key: &str) -> Option<u64> {
    args.get(key).and_then(|v| v.as_u64())
}

fn arg_bool(args: &Value, key: &str) -> Option<bool> {
    args.get(key).and_then(|v| v.as_bool())
}

impl ToolBox {
    pub fn new(workdir: PathBuf, default_timeout: Duration) -> Self {
        Self {
            workdir,
            default_timeout,
        }
    }

    pub fn resolve(&self, p: &str) -> PathBuf {
        let path = Path::new(p);
        if path.is_absolute() {
            path.to_path_buf()
        } else {
            self.workdir.join(path)
        }
    }

    pub async fn call(&self, name: &str, args: &Value) -> ToolOutput {
        let start = Instant::now();
        let mut out = match name {
            "bash" => self.bash(args).await,
            "read_file" => self.read_file(args).await,
            "write_file" => self.write_file(args).await,
            "edit_file" => self.edit_file(args).await,
            "list_dir" => self.list_dir(args).await,
            "search" => self.search(args).await,
            other => ToolOutput::err(format!("unknown tool '{other}'")),
        };
        out.duration_ms = start.elapsed().as_millis() as u64;
        out
    }

    async fn bash(&self, args: &Value) -> ToolOutput {
        let Some(cmd) = arg_str(args, "cmd").or_else(|| arg_str(args, "command")) else {
            return ToolOutput::err("bash: missing 'cmd' argument");
        };
        let timeout = arg_u64(args, "timeout_secs")
            .map(Duration::from_secs)
            .unwrap_or(self.default_timeout);
        let child = Command::new("sh")
            .arg("-c")
            .arg(cmd)
            .current_dir(&self.workdir)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true)
            .spawn();
        let child = match child {
            Ok(c) => c,
            Err(e) => return ToolOutput::err(format!("bash: failed to spawn shell: {e}")),
        };
        match tokio::time::timeout(timeout, child.wait_with_output()).await {
            Ok(Ok(o)) => ToolOutput {
                exit_code: o.status.code().unwrap_or(-1),
                stdout: String::from_utf8_lossy(&o.stdout).into_owned(),
                stderr: String::from_utf8_lossy(&o.stderr).into_owned(),
                duration_ms: 0,
            },
            Ok(Err(e)) => ToolOutput::err(format!("bash: {e}")),
            Err(_) => ToolOutput {
                exit_code: 124,
                stdout: String::new(),
                stderr: format!("bash: timed out after {}s", timeout.as_secs()),
                duration_ms: 0,
            },
        }
    }

    async fn read_file(&self, args: &Value) -> ToolOutput {
        let Some(p) = arg_str(args, "path") else {
            return ToolOutput::err("read_file: missing 'path' argument");
        };
        let content = match tokio::fs::read_to_string(self.resolve(p)).await {
            Ok(s) => s,
            Err(e) => return ToolOutput::err(format!("read_file {p}: {e}")),
        };
        let start = arg_u64(args, "start_line").unwrap_or(1).max(1) as usize;
        let end = arg_u64(args, "end_line")
            .map(|e| e as usize)
            .unwrap_or(usize::MAX);
        let numbered = arg_bool(args, "numbered").unwrap_or(false);
        if start == 1 && end == usize::MAX && !numbered {
            return ToolOutput::ok(content);
        }
        let mut out = String::new();
        for (i, line) in content.lines().enumerate() {
            let n = i + 1;
            if n < start {
                continue;
            }
            if n > end {
                break;
            }
            if numbered {
                out.push_str(&format!("{n:>5}| {line}\n"));
            } else {
                out.push_str(line);
                out.push('\n');
            }
        }
        ToolOutput::ok(out)
    }

    async fn write_file(&self, args: &Value) -> ToolOutput {
        let Some(p) = arg_str(args, "path") else {
            return ToolOutput::err("write_file: missing 'path' argument");
        };
        let content = arg_str(args, "content").unwrap_or("");
        let path = self.resolve(p);
        if let Some(parent) = path.parent() {
            if let Err(e) = tokio::fs::create_dir_all(parent).await {
                return ToolOutput::err(format!("write_file {p}: {e}"));
            }
        }
        match tokio::fs::write(&path, content).await {
            Ok(()) => ToolOutput::ok(format!("wrote {} bytes to {p}", content.len())),
            Err(e) => ToolOutput::err(format!("write_file {p}: {e}")),
        }
    }

    async fn edit_file(&self, args: &Value) -> ToolOutput {
        let (Some(p), Some(old)) = (arg_str(args, "path"), arg_str(args, "old")) else {
            return ToolOutput::err("edit_file: requires 'path' and 'old' arguments");
        };
        let new = arg_str(args, "new").unwrap_or("");
        let replace_all = arg_bool(args, "replace_all").unwrap_or(false);
        if old.is_empty() {
            return ToolOutput::err("edit_file: 'old' must not be empty");
        }
        let path = self.resolve(p);
        let content = match tokio::fs::read_to_string(&path).await {
            Ok(s) => s,
            Err(e) => return ToolOutput::err(format!("edit_file {p}: {e}")),
        };
        let count = content.matches(old).count();
        if count == 0 {
            return ToolOutput::err(format!("edit_file {p}: 'old' text not found"));
        }
        if count > 1 && !replace_all {
            return ToolOutput::err(format!(
                "edit_file {p}: 'old' text found {count} times; make it unique or set replace_all"
            ));
        }
        let updated = if replace_all {
            content.replace(old, new)
        } else {
            content.replacen(old, new, 1)
        };
        match tokio::fs::write(&path, updated).await {
            Ok(()) => ToolOutput::ok(format!("edited {p}: replaced {count} occurrence(s)")),
            Err(e) => ToolOutput::err(format!("edit_file {p}: {e}")),
        }
    }

    async fn list_dir(&self, args: &Value) -> ToolOutput {
        let p = arg_str(args, "path").unwrap_or(".");
        let mut rd = match tokio::fs::read_dir(self.resolve(p)).await {
            Ok(rd) => rd,
            Err(e) => return ToolOutput::err(format!("list_dir {p}: {e}")),
        };
        let mut names = Vec::new();
        loop {
            match rd.next_entry().await {
                Ok(Some(entry)) => {
                    let is_dir = entry.file_type().await.map(|t| t.is_dir()).unwrap_or(false);
                    let mut name = entry.file_name().to_string_lossy().into_owned();
                    if is_dir {
                        name.push('/');
                    }
                    names.push(name);
                }
                Ok(None) => break,
                Err(e) => return ToolOutput::err(format!("list_dir {p}: {e}")),
            }
        }
        names.sort();
        ToolOutput::ok(names.join("\n"))
    }

    async fn search(&self, args: &Value) -> ToolOutput {
        let Some(pattern) = arg_str(args, "pattern") else {
            return ToolOutput::err("search: missing 'pattern' argument");
        };
        let re = match Regex::new(pattern) {
            Ok(re) => re,
            Err(e) => return ToolOutput::err(format!("search: invalid regex: {e}")),
        };
        let root = self.resolve(arg_str(args, "path").unwrap_or("."));
        let max = arg_u64(args, "max_results").unwrap_or(200) as usize;
        let workdir = self.workdir.clone();
        let res =
            tokio::task::spawn_blocking(move || search_blocking(&re, &root, &workdir, max)).await;
        match res {
            Ok(lines) if lines.is_empty() => ToolOutput {
                exit_code: 1,
                stdout: "no matches".into(),
                ..Default::default()
            },
            Ok(lines) => ToolOutput::ok(lines.join("\n")),
            Err(e) => ToolOutput::err(format!("search: {e}")),
        }
    }
}

fn search_blocking(re: &Regex, root: &Path, workdir: &Path, max: usize) -> Vec<String> {
    let mut results = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(p) = stack.pop() {
        if results.len() >= max {
            break;
        }
        let meta = match std::fs::symlink_metadata(&p) {
            Ok(m) => m,
            Err(_) => continue,
        };
        if meta.is_dir() {
            let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
            if p != root && SKIP_DIRS.contains(&name) {
                continue;
            }
            if let Ok(rd) = std::fs::read_dir(&p) {
                let mut children: Vec<PathBuf> =
                    rd.filter_map(|e| e.ok().map(|e| e.path())).collect();
                children.sort();
                children.reverse();
                stack.extend(children);
            }
        } else if meta.is_file() {
            if meta.len() > 2_000_000 {
                continue;
            }
            let Ok(content) = std::fs::read_to_string(&p) else {
                continue;
            };
            let shown = p.strip_prefix(workdir).unwrap_or(&p).display().to_string();
            for (i, line) in content.lines().enumerate() {
                if re.is_match(line) {
                    results.push(format!(
                        "{}:{}: {}",
                        shown,
                        i + 1,
                        crate::util::truncate_chars(line.trim_end(), 300)
                    ));
                    if results.len() >= max {
                        break;
                    }
                }
            }
        }
    }
    results
}
