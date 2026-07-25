//! Configuration for the NIM proxy.

use std::env;
use std::net::SocketAddr;

use anyhow::Result;
use anyhow::bail;

/// Default upstream NIM endpoint.
pub const DEFAULT_UPSTREAM_BASE_URL: &str = "https://integrate.api.nvidia.com/v1";

/// Default local port for the proxy.
pub const DEFAULT_PROXY_PORT: u16 = 8765;

/// Default NIM model that ALL codex requests are silently routed to,
/// regardless of what codex thinks it's using. Codex's UI will show
/// whatever model codex picked (e.g. "gpt-5.6-sol"), but every request
/// actually hits this NIM model on the upstream.
pub const DEFAULT_BACKEND_MODEL: &str = "thinkingmachines/inkling";

/// Environment variable that holds the NIM API key.
pub const NIM_API_KEY_ENV: &str = "NVIDIA_API_KEY";

#[derive(Debug, Clone)]
pub struct ProxyConfig {
    pub bind_addr: SocketAddr,
    pub upstream_base_url: String,
    pub api_key: String,
    pub request_timeout_secs: u64,
    pub verbose: bool,
    /// Max requests per minute to send to the upstream. NIM free tier = 40.
    /// Set to 0 to disable rate limiting.
    pub rpm: u32,
    /// The NIM model slug that ALL incoming requests are silently rerouted
    /// to. Whatever codex sends as `model` (e.g. "gpt-5.6-sol") is replaced
    /// with this value before forwarding to NIM. This is the core "override
    /// bridge" behavior: codex's UI keeps showing its own model name, but
    /// the actual backend is always this NIM model.
    pub backend_model: String,
    /// When true, inject `chat_template_kwargs.enable_thinking = true` and
    /// `reasoning_budget = 99_999_999` for reasoning-capable NIM models
    /// (nemotron, deepseek-r1, mistral-nemotron). OFF by default because
    /// it makes non-reasoning models slow.
    pub enable_thinking: bool,
}

impl ProxyConfig {
    pub fn resolve(
        port: u16,
        upstream_base_url: Option<String>,
        api_key_override: Option<String>,
        verbose: bool,
        rpm: u32,
        backend_model: Option<String>,
        enable_thinking: bool,
    ) -> Result<Self> {
        let bind_addr: SocketAddr = ([127, 0, 0, 1], port).into();

        let upstream_base_url = upstream_base_url
            .unwrap_or_else(|| DEFAULT_UPSTREAM_BASE_URL.to_string())
            .trim_end_matches('/')
            .to_string();

        let api_key = match api_key_override {
            Some(k) if !k.trim().is_empty() => k,
            _ => match env::var(NIM_API_KEY_ENV) {
                Ok(v) if !v.trim().is_empty() => v,
                _ => bail!(
                    "NVIDIA NIM API key not found. Set the {NIM_API_KEY_ENV} environment \
                     variable to your `nvapi-...` key, or pass --api-key."
                ),
            },
        };

        let backend_model = backend_model
            .unwrap_or_else(|| DEFAULT_BACKEND_MODEL.to_string());

        Ok(Self {
            bind_addr,
            upstream_base_url,
            api_key,
            request_timeout_secs: 600,
            verbose,
            rpm,
            backend_model,
            enable_thinking,
        })
    }

    pub fn upstream_chat_url(&self) -> String {
        format!("{}/chat/completions", self.upstream_base_url)
    }

    pub fn upstream_models_url(&self) -> String {
        format!("{}/models", self.upstream_base_url)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chat_url_is_built_correctly() {
        let cfg = ProxyConfig {
            bind_addr: "127.0.0.1:0".parse().unwrap(),
            upstream_base_url: "https://example.com/v1".to_string(),
            api_key: "k".to_string(),
            request_timeout_secs: 1,
            verbose: false,
            rpm: 40,
            backend_model: "qwen/qwen3".to_string(),
            enable_thinking: false,
        };
        assert_eq!(cfg.upstream_chat_url(), "https://example.com/v1/chat/completions");
        assert_eq!(cfg.upstream_models_url(), "https://example.com/v1/models");
    }

    #[test]
    fn trailing_slash_is_trimmed() {
        let r = ProxyConfig::resolve(
            DEFAULT_PROXY_PORT,
            Some("https://example.com/v1/".to_string()),
            Some("k".to_string()),
            false,
            40,
            None,
            false,
        )
        .expect("resolve");
        assert_eq!(r.upstream_base_url, "https://example.com/v1");
        assert_eq!(r.backend_model, DEFAULT_BACKEND_MODEL);
    }
}
