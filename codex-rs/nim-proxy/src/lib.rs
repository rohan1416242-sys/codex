//! `codex-nim-proxy` — local translator that lets Codex CLI talk to
//! NVIDIA NIM (and any other OpenAI Chat-Completions-only provider) by
//! exposing a Responses-API endpoint on localhost and translating to/from
//! Chat Completions on the upstream side.

pub mod catalog;
pub mod config;
pub mod rate_limit;
pub mod server;
pub mod translate;

pub use config::ProxyConfig;
pub use rate_limit::RateLimiter;
pub use rate_limit::DEFAULT_RPM;
pub use server::serve;
