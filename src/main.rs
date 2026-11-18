use std::path::PathBuf;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use clap::{Args, Parser, Subcommand};

use blitz::agent::{Agent, AgentConfig};
use blitz::exec::{ExecReport, Executor};
use blitz::llm::{LlmClient, ProviderKind};
use blitz::plan::{Limits, Plan};
use blitz::tools::{ToolBox, TOOL_NAMES};

#[derive(Parser)]
#[command(
    name = "blitz",
    version,
    about = "Speculative, parallel AI agent: many tools per turn + predicted branches that auto-chain."
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Run the agent on a task.
    Run(RunArgs),
    /// Execute a plan JSON file directly (no LLM) and print the report.
    Exec(ExecArgs),
    /// Print the system prompt sent to the model.
    Prompt {
        #[arg(long, default_value_t = 4)]
        max_depth: usize,
    },
}

#[derive(Args)]
struct RunArgs {
    /// The task description.
    task: String,
    #[arg(short, long, value_enum, default_value = "gemini", env = "BLITZ_PROVIDER")]
    provider: ProviderKind,
    /// Model name (defaults depend on provider).
    #[arg(short, long, env = "BLITZ_MODEL")]
    model: Option<String>,
    /// Override the API base URL (e.g. OpenRouter, Groq, local llama.cpp).
    #[arg(long, env = "BLITZ_BASE_URL")]
    base_url: Option<String>,
    /// Working directory for tools.
    #[arg(short = 'C', long, default_value = ".")]
    workdir: PathBuf,
    #[arg(long, default_value_t = 30)]
    max_turns: usize,
    /// Max tools running at the same time.
    #[arg(long, default_value_t = 16)]
    concurrency: usize,
    /// Default bash timeout in seconds.
    #[arg(long, default_value_t = 120)]
    timeout_secs: u64,
    /// Max nesting depth of predicted branches.
    #[arg(long, default_value_t = 4)]
    max_depth: usize,
    /// Max nodes in one plan.
    #[arg(long, default_value_t = 64)]
    max_nodes: usize,
    /// Max bytes of each tool output shown to the model.
    #[arg(long, default_value_t = 8000)]
    max_output: usize,
    /// JSON array of scripted responses for `--provider mock`.
    #[arg(long)]
    mock_file: Option<PathBuf>,
    /// Disable provider JSON mode (for endpoints that don't support it).
    #[arg(long)]
    no_json_mode: bool,
    /// Only print the final answer.
    #[arg(short, long)]
    quiet: bool,
}

#[derive(Args)]
struct ExecArgs {
    /// Path to a plan JSON file.
    plan: PathBuf,
    #[arg(short = 'C', long, default_value = ".")]
    workdir: PathBuf,
    #[arg(long, default_value_t = 16)]
    concurrency: usize,
    #[arg(long, default_value_t = 120)]
    timeout_secs: u64,
    #[arg(long, default_value_t = 4000)]
    max_output: usize,
}

#[tokio::main]
async fn main() {
    let cli = Cli::parse();
    let code = match real_main(cli).await {
        Ok(c) => c,
        Err(e) => {
            eprintln!("error: {e:#}");
            1
        }
    };
    std::process::exit(code);
}

async fn real_main(cli: Cli) -> Result<i32> {
    match cli.cmd {
        Cmd::Run(a) => run(a).await,
        Cmd::Exec(a) => exec(a).await,
        Cmd::Prompt { max_depth } => {
            println!("{}", blitz::prompt::system_prompt(max_depth));
            Ok(0)
        }
    }
}

async fn run(a: RunArgs) -> Result<i32> {
    let workdir = std::fs::canonicalize(&a.workdir)
        .with_context(|| format!("invalid workdir {}", a.workdir.display()))?;
    let llm = if a.provider == ProviderKind::Mock {
        let path = a
            .mock_file
            .as_ref()
            .context("--mock-file is required with --provider mock")?;
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("reading {}", path.display()))?;
        let items: Vec<serde_json::Value> =
            serde_json::from_str(&text).context("mock file must be a JSON array")?;
        LlmClient::mock(
            items
                .into_iter()
                .map(|v| match v {
                    serde_json::Value::String(s) => s,
                    other => other.to_string(),
                })
                .collect(),
        )
    } else {
        LlmClient::new(a.provider, a.model.clone(), a.base_url.clone(), !a.no_json_mode)?
    };

    let exec = Executor::new(
        ToolBox::new(workdir, Duration::from_secs(a.timeout_secs)),
        a.concurrency,
    );
    let cfg = AgentConfig {
        max_turns: a.max_turns,
        max_output: a.max_output,
        limits: Limits {
            max_depth: a.max_depth,
            max_nodes: a.max_nodes,
        },
        verbose: !a.quiet,
    };

    let t = Instant::now();
    let res = Agent::new(llm, exec, cfg).run(&a.task).await?;
    if !a.quiet {
        eprintln!(
            "\n📊 {} | total: {:.1}s",
            res.stats,
            t.elapsed().as_secs_f64()
        );
    }
    match res.answer {
        Some(ans) => {
            if !a.quiet {
                eprintln!("\n✅ ANSWER:");
            }
            println!("{ans}");
            Ok(0)
        }
        None => {
            eprintln!("stopped: reached max turns without finishing");
            Ok(2)
        }
    }
}

async fn exec(a: ExecArgs) -> Result<i32> {
    let workdir = std::fs::canonicalize(&a.workdir)
        .with_context(|| format!("invalid workdir {}", a.workdir.display()))?;
    let text = std::fs::read_to_string(&a.plan)
        .with_context(|| format!("reading {}", a.plan.display()))?;
    let mut plan = Plan::parse(&text).map_err(anyhow::Error::msg)?;
    plan.validate(TOOL_NAMES, &Limits::default())
        .map_err(anyhow::Error::msg)?;
    let exec = Executor::new(
        ToolBox::new(workdir, Duration::from_secs(a.timeout_secs)),
        a.concurrency,
    );
    let report = exec.run(&plan.steps).await;
    for r in &report.records {
        eprintln!("{}", ExecReport::summary_line(r));
    }
    println!("{}", report.render(a.max_output));
    eprintln!(
        "clean: {} | wall {} ms vs {} ms sequential",
        report.clean(),
        report.wall_ms,
        report.tool_ms()
    );
    Ok(if report.clean() { 0 } else { 3 })
}
