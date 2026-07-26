//! Binary entry point for `codex-nim-proxy`.

use clap::Parser;
use codex_nim_proxy::ProxyConfig;
use codex_nim_proxy::config::DEFAULT_BACKEND_MODEL;
use codex_nim_proxy::config::DEFAULT_PROXY_PORT;
use codex_nim_proxy::rate_limit::DEFAULT_RPM;
use codex_nim_proxy::serve;

#[derive(Debug, Parser)]
#[command(
    name = "codex-nim-proxy",
    version,
    about = "Transparent override bridge: codex CLI → NVIDIA NIM (any model)"
)]
pub struct Args {
    /// Port to listen on. Defaults to 8765.
    #[arg(long, default_value_t = DEFAULT_PROXY_PORT)]
    port: u16,

    /// Upstream base URL. Defaults to the public NVIDIA NIM endpoint.
    #[arg(long)]
    upstream_base_url: Option<String>,

    /// NVIDIA API key. If omitted, read from the `NVIDIA_API_KEY` env var.
    #[arg(long)]
    api_key: Option<String>,

    /// The NIM model slug that ALL codex requests are silently rerouted to.
    /// Codex's UI will show whatever model codex picked (e.g. "gpt-5.6-sol"),
    /// but the actual backend is always this NIM model.
    /// Default: qwen/qwen3-next-80b-a3b-instruct (fast + great coder).
    #[arg(long, default_value = DEFAULT_BACKEND_MODEL)]
    backend_model: String,

    /// Inject `chat_template_kwargs.enable_thinking = true` and
    /// `reasoning_budget = 99999999` for reasoning-capable NIM models
    /// (inkling, nemotron, deepseek-r1, mistral-nemotron).
    /// ON by default because the default backend (thinkingmachines/inkling)
    /// is a reasoning model that needs it to produce final answers.
    /// Turn OFF if you switch to a non-reasoning model like Qwen3 or Llama.
    #[arg(long, default_value_t = true)]
    enable_thinking: bool,

    /// Max requests per minute to send to the upstream.
    /// NIM free tier = 40. Set to 0 to disable rate limiting.
    #[arg(long, default_value_t = DEFAULT_RPM)]
    rpm: u32,

    /// Verbose logging (dumps every request/response body to stderr).
    #[arg(long, short = 'v')]
    verbose: bool,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    let filter = if args.verbose {
        "codex_nim_proxy=debug,info"
    } else {
        "codex_nim_proxy=info,warn"
    };
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new(filter)),
        )
        .with_target(false)
        .init();

    let cfg = ProxyConfig::resolve(
        args.port,
        args.upstream_base_url,
        args.api_key,
        args.verbose,
        args.rpm,
        Some(args.backend_model),
        args.enable_thinking,
    )?;

    serve(cfg).await
}
