use std::path::PathBuf;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use clap::{Args, Parser, Subcommand};

use blitz::agent::{Agent, AgentConfig};
use blitz::config::{AppConfig, GuessMode};
use blitz::exec::{ExecReport, Executor};
use blitz::llm::{LlmClient, ProviderKind};
use blitz::plan::{Limits, Plan};
use blitz::tools::{ToolBox, TOOL_NAMES};
use blitz::tui::run_tui;

#[derive(Parser)]
#[command(
    name = "blitz",
    version,
    about = "Speculative, parallel AI agent with interactive TUI: many tools per turn + predicted branches that auto-chain."
)]
struct Cli {
    /// Working directory for tools when launching the default TUI.
    #[arg(short = 'C', long, default_value = ".")]
    workdir: PathBuf,

    #[command(subcommand)]
    cmd: Option<Cmd>,
}

#[derive(Subcommand)]
enum Cmd {
    /// Launch the full interactive Terminal UI (default when no subcommand is given).
    Tui {
        #[arg(short = 'C', long, default_value = ".")]
        workdir: PathBuf,
    },
    /// Run the agent on a task (or launch TUI if task is omitted).
    Run(RunArgs),
    /// Execute a plan JSON file directly (no LLM) and print the report.
    Exec(ExecArgs),
    /// Print the system prompt sent to the model.
    Prompt {
        #[arg(long, default_value_t = 4)]
        max_depth: usize,
        #[arg(long, value_enum, default_value = "aggressive")]
        guess_mode: GuessMode,
    },
}

#[derive(Args)]
struct RunArgs {
    /// The task description (omit to open the interactive TUI).
    task: Option<String>,
    #[arg(short, long, value_enum, env = "BLITZ_PROVIDER")]
    provider: Option<ProviderKind>,
    /// Model name (defaults to saved config or provider default).
    #[arg(short, long, env = "BLITZ_MODEL")]
    model: Option<String>,
    /// Override the API base URL (e.g. OpenRouter, Groq, local llama.cpp).
    #[arg(long, env = "BLITZ_BASE_URL")]
    base_url: Option<String>,
    /// Automatic guessing mode.
    #[arg(short, long, value_enum, env = "BLITZ_GUESS_MODE")]
    guess_mode: Option<GuessMode>,
    /// Working directory for tools.
    #[arg(short = 'C', long, default_value = ".")]
    workdir: PathBuf,
    #[arg(long)]
    max_turns: Option<usize>,
    /// Max tools running at the same time.
    #[arg(long)]
    concurrency: Option<usize>,
    /// Default bash timeout in seconds.
    #[arg(long)]
    timeout_secs: Option<u64>,
    /// Max nesting depth of predicted branches.
    #[arg(long)]
    max_depth: Option<usize>,
    /// Max predicted branches per node.
    #[arg(long)]
    max_branches: Option<usize>,
    /// Max nodes in one plan.
    #[arg(long)]
    max_nodes: Option<usize>,
    /// Max bytes of each tool output shown to the model.
    #[arg(long)]
    max_output: Option<usize>,
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
        None => {
            let workdir = std::fs::canonicalize(&cli.workdir)
                .with_context(|| format!("invalid workdir {}", cli.workdir.display()))?;
            let cfg = AppConfig::load();
            run_tui(cfg, workdir).await?;
            Ok(0)
        }
        Some(Cmd::Tui { workdir }) => {
            let workdir = std::fs::canonicalize(&workdir)
                .with_context(|| format!("invalid workdir {}", workdir.display()))?;
            let cfg = AppConfig::load();
            run_tui(cfg, workdir).await?;
            Ok(0)
        }
        Some(Cmd::Run(a)) => run(a).await,
        Some(Cmd::Exec(a)) => exec(a).await,
        Some(Cmd::Prompt {
            max_depth,
            guess_mode,
        }) => {
            let limits = Limits {
                max_depth,
                ..Limits::default()
            };
            println!(
                "{}",
                blitz::prompt::system_prompt_with(guess_mode, limits, true)
            );
            Ok(0)
        }
    }
}

async fn run(a: RunArgs) -> Result<i32> {
    let workdir = std::fs::canonicalize(&a.workdir)
        .with_context(|| format!("invalid workdir {}", a.workdir.display()))?;
    let mut saved = AppConfig::load();
    if let Some(p) = a.provider {
        saved.provider = p;
    }
    if let Some(m) = &a.model {
        saved.set_active_model(m.clone());
    }
    if let Some(u) = &a.base_url {
        saved.set_active_base_url(u.clone());
    }
    if let Some(g) = a.guess_mode {
        saved.guess_mode = g;
    }
    if let Some(d) = a.max_depth {
        saved.max_depth = d;
    }
    if let Some(b) = a.max_branches {
        saved.max_branches = b;
    }
    if let Some(n) = a.max_nodes {
        saved.max_nodes = n;
    }
    if let Some(c) = a.concurrency {
        saved.concurrency = c;
    }
    if let Some(t) = a.max_turns {
        saved.max_turns = t;
    }
    if let Some(ts) = a.timeout_secs {
        saved.timeout_secs = ts;
    }
    if let Some(mo) = a.max_output {
        saved.max_output = mo;
    }
    if a.no_json_mode {
        saved.json_mode = false;
    }

    let Some(task) = a.task else {
        run_tui(saved, workdir).await?;
        return Ok(0);
    };

    let llm = if saved.provider == ProviderKind::Mock {
        let path = a
            .mock_file
            .as_ref()
            .context("--mock-file is required with --provider mock")?;
        let text =
            std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
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
        LlmClient::from_config(&saved)?
    };

    let exec = Executor::new(
        ToolBox::new(workdir, Duration::from_secs(saved.timeout_secs)),
        saved.concurrency,
    );
    let cfg = AgentConfig {
        max_turns: saved.max_turns,
        max_output: saved.max_output,
        limits: saved.limits(),
        guess_mode: saved.guess_mode,
        optimistic_finish: saved.optimistic_finish,
        verbose: !a.quiet,
        events: None,
    };

    let t = Instant::now();
    let res = Agent::new(llm, exec, cfg).run(&task).await?;
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
    let report = exec
        .run_with_after_all(&plan.steps, &plan.after_all)
        .await;
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
