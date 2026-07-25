//! HTTP server that exposes the local Responses-API endpoint.

use std::time::Duration;

use anyhow::Result;
use axum::Router;
use axum::body::Body;
use axum::extract::State;
use axum::http::HeaderMap;
use axum::http::HeaderValue;
use axum::http::StatusCode;
use axum::http::header::CONTENT_TYPE;
use axum::response::Response;
use axum::routing::get;
use axum::routing::post;
use reqwest::header::HeaderName;
use serde_json::Value;
use tokio_stream::wrappers::ReceiverStream;
use tracing::debug;
use tracing::info;
use tracing::warn;

use crate::config::ProxyConfig;
use crate::rate_limit::RateLimiter;
use crate::translate::ChatToResponsesStream;
use crate::translate::spawn_stream_converter;
use crate::translate::translate_request;

const ACCEPT: &str = "accept";
const CACHE_CONTROL: &str = "cache-control";
const CONNECTION: &str = "connection";

#[derive(Clone)]
pub struct ProxyState {
    pub cfg: ProxyConfig,
    pub client: reqwest::Client,
    pub rate_limiter: Option<RateLimiter>,
}

pub async fn serve(cfg: ProxyConfig) -> Result<()> {
    let bind_addr = cfg.bind_addr;
    let verbose = cfg.verbose;

    let client = reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(30))
        .pool_idle_timeout(Duration::from_secs(600))
        .pool_max_idle_per_host(32)
        .tcp_nodelay(true)
        .tcp_keepalive(Duration::from_secs(60))
        .build()?;

    let state = ProxyState {
        cfg: cfg.clone(),
        client,
        rate_limiter: if cfg.rpm > 0 {
            Some(RateLimiter::new(cfg.rpm))
        } else {
            None
        },
    };

    if let Some(limiter) = state.rate_limiter.clone() {
        crate::rate_limit::spawn_stats_logger(limiter);
    }

    let app = Router::new()
        .route("/v1/responses", post(handle_responses))
        .route("/v1/chat/completions", post(handle_passthrough_chat))
        .route("/v1/models", get(handle_models))
        .route("/health", get(handle_health))
        .with_state(state);

    info!("codex-nim-proxy listening on http://{bind_addr}");
    info!("forwarding to upstream {}", cfg.upstream_base_url);
    info!("backend model: {} (ALL codex requests rerouted here)", cfg.backend_model);
    info!(
        "enable_thinking: {} (only affects nemotron / deepseek-r1 / mistral-nemotron)",
        cfg.enable_thinking
    );
    if cfg.rpm > 0 {
        info!("rate limit: {} requests per minute (NIM free tier cap)", cfg.rpm);
    } else {
        info!("rate limit: disabled (unlimited)");
    }
    if verbose {
        info!("verbose logging enabled");
    }

    let listener = tokio::net::TcpListener::bind(bind_addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}

async fn handle_health() -> &'static str {
    "ok\n"
}

async fn handle_models(State(state): State<ProxyState>) -> Response<Body> {
    let upstream_ids: Vec<String> = match state
        .client
        .get(state.cfg.upstream_models_url())
        .bearer_auth(&state.cfg.api_key)
        .timeout(Duration::from_secs(8))
        .send()
        .await
    {
        Ok(r) if r.status().is_success() => match r.json::<Value>().await {
            Ok(v) => v
                .get("data")
                .and_then(Value::as_array)
                .map(|arr| {
                    arr.iter()
                        .filter_map(|m| m.get("id").and_then(Value::as_str).map(String::from))
                        .collect()
                })
                .unwrap_or_default(),
            Err(_) => Vec::new(),
        },
        _ => Vec::new(),
    };

    let body = crate::catalog::build_models_response(&upstream_ids);
    Response::builder()
        .status(StatusCode::OK)
        .header(CONTENT_TYPE, "application/json")
        .body(Body::from(body.to_string()))
        .unwrap_or_else(|_| fallback_body())
}

async fn handle_responses(
    State(state): State<ProxyState>,
    _headers: HeaderMap,
    body: axum::body::Bytes,
) -> Response<Body> {
    if let Some(limiter) = &state.rate_limiter {
        limiter.acquire().await;
    }

    let responses_body: Value = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(e) => {
            return error_response(StatusCode::BAD_REQUEST, &format!("invalid JSON body: {e}"));
        }
    };

    let incoming_model = responses_body
        .get("model")
        .and_then(Value::as_str)
        .unwrap_or("unknown")
        .to_string();

    let want_stream = responses_body
        .get("stream")
        .and_then(Value::as_bool)
        .unwrap_or(true);

    // Log the model override: codex sent `incoming_model`, we'll forward
    // to NIM using `state.cfg.backend_model`.
    info!(
        "model override: codex=`{incoming_model}` → backend=`{}`",
        state.cfg.backend_model
    );
    debug!(
        "responses request: stream={want_stream} body_bytes={}",
        body.len()
    );
    if state.cfg.verbose {
        info!("responses request body: {responses_body}");
    }

    let (chat_body, response_model) = match translate_request(
        &responses_body,
        &state.cfg.backend_model,
        state.cfg.enable_thinking,
    ) {
        Ok((b, m)) => (b, m),
        Err(e) => {
            return error_response(StatusCode::BAD_REQUEST, &format!("translation error: {e}"));
        }
    };

    if state.cfg.verbose {
        info!("translated chat request body: {chat_body}");
    }

    // `response_model` is the INCOMING model name (what codex sent, e.g.
    // "gpt-5.6-sol"). We use this in response events so codex's UI sees
    // the "expected" model name and can't tell a proxy is in the middle.
    // This is the KEY to the invisible bridge.
    let model = response_model;

    let chat_url = state.cfg.upstream_chat_url();
    let req = state
        .client
        .post(&chat_url)
        .bearer_auth(&state.cfg.api_key)
        .header(CONTENT_TYPE, "application/json")
        .header(ACCEPT, "text/event-stream")
        .json(&chat_body);

    info!("sending upstream request to {chat_url}");
    let upstream = match req.send().await {
        Ok(r) => {
            info!("upstream responded with status {}", r.status());
            r
        }
        Err(e) => {
            warn!("upstream chat request failed: {e}");
            return failed_sse_response(&model, &format!("upstream connect error: {e}"));
        }
    };

    let status = upstream.status();
    if !status.is_success() {
        let body_text = upstream.text().await.unwrap_or_default();
        let preview: String = body_text.chars().take(500).collect();
        warn!("upstream returned {status}: {preview}");
        return failed_sse_response(&model, &format!("upstream {status}: {body_text}"));
    }

    if !want_stream {
        return non_streaming_response(upstream, &model).await;
    }

    info!("entering streaming path");
    let byte_stream = upstream.bytes_stream();
    let rx = spawn_stream_converter(byte_stream, model.clone());
    info!("spawned stream converter, returning SSE body");

    let body = Body::from_stream(ReceiverStream::new(rx));
    Response::builder()
        .status(StatusCode::OK)
        .header(CONTENT_TYPE, "text/event-stream")
        .header(CACHE_CONTROL, "no-cache")
        .header(CONNECTION, "keep-alive")
        .header("X-Accel-Buffering", "no")
        .body(body)
        .unwrap_or_else(|_| fallback_body())
}

async fn handle_passthrough_chat(
    State(state): State<ProxyState>,
    body: axum::body::Bytes,
) -> Response<Body> {
    if let Some(limiter) = &state.rate_limiter {
        limiter.acquire().await;
    }

    let url = state.cfg.upstream_chat_url();
    let resp = state
        .client
        .post(&url)
        .bearer_auth(&state.cfg.api_key)
        .header(CONTENT_TYPE, "application/json")
        .body(body)
        .send()
        .await;

    match resp {
        Ok(r) => forward_response(r).await,
        Err(e) => error_response(StatusCode::BAD_GATEWAY, &format!("upstream error: {e}")),
    }
}

async fn forward_response(resp: reqwest::Response) -> Response<Body> {
    let status =
        StatusCode::from_u16(resp.status().as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
    let mut headers = HeaderMap::new();
    for (k, v) in resp.headers().iter() {
        if matches!(
            k.as_str(),
            "content-length" | "transfer-encoding" | "connection"
        ) {
            continue;
        }
        if let Ok(name) = HeaderName::from_bytes(k.as_str().as_bytes())
            && let Ok(val) = HeaderValue::from_bytes(v.as_bytes())
        {
            headers.append(name, val);
        }
    }
    let bytes = match resp.bytes().await {
        Ok(b) => b,
        Err(e) => {
            return error_response(
                StatusCode::BAD_GATEWAY,
                &format!("failed reading upstream body: {e}"),
            );
        }
    };
    let mut response = Response::new(Body::from(bytes));
    *response.status_mut() = status;
    *response.headers_mut() = headers;
    response
}

async fn non_streaming_response(upstream: reqwest::Response, model: &str) -> Response<Body> {
    let bytes = match upstream.bytes().await {
        Ok(b) => b,
        Err(e) => {
            return error_response(
                StatusCode::BAD_GATEWAY,
                &format!("error reading upstream: {e}"),
            );
        }
    };

    if bytes.len() > 16 * 1024 * 1024 {
        return error_response(
            StatusCode::PAYLOAD_TOO_LARGE,
            "non-streaming response exceeded 16 MiB",
        );
    }

    let text = String::from_utf8_lossy(&bytes);
    let mut full_content = String::new();
    let mut tool_call_args: std::collections::BTreeMap<u32, (String, String, String)> =
        std::collections::BTreeMap::new();
    let mut usage: Option<Value> = None;

    for line in text.lines() {
        let Some(data) = line.strip_prefix("data: ") else {
            continue;
        };
        let data = data.trim();
        if data == "[DONE]" {
            break;
        }
        let Ok(chunk) = serde_json::from_str::<Value>(data) else {
            continue;
        };
        if let Some(c) = chunk
            .get("choices")
            .and_then(|v| v.as_array())
            .and_then(|a| a.first())
        {
            if let Some(s) = c
                .get("delta")
                .and_then(|d| d.get("content"))
                .and_then(Value::as_str)
            {
                full_content.push_str(s);
            }
            if let Some(tcs) = c
                .get("delta")
                .and_then(|d| d.get("tool_calls"))
                .and_then(Value::as_array)
            {
                for tc in tcs {
                    let idx = tc.get("index").and_then(Value::as_u64).unwrap_or(0) as u32;
                    let entry = tool_call_args
                        .entry(idx)
                        .or_insert_with(|| (String::new(), String::new(), String::new()));
                    if let Some(id) = tc.get("id").and_then(Value::as_str)
                        && !id.is_empty()
                    {
                        entry.0 = id.to_string();
                    }
                    if let Some(name) = tc
                        .get("function")
                        .and_then(|f| f.get("name"))
                        .and_then(Value::as_str)
                        && !name.is_empty()
                    {
                        entry.1 = name.to_string();
                    }
                    if let Some(args) = tc
                        .get("function")
                        .and_then(|f| f.get("arguments"))
                        .and_then(Value::as_str)
                    {
                        entry.2.push_str(args);
                    }
                }
            }
        }
        if let Some(u) = chunk.get("usage") {
            usage = Some(u.clone());
        }
    }

    let response_id = format!("resp_{}", rand_id());
    let mut output: Vec<Value> = Vec::new();
    if !full_content.is_empty() {
        output.push(serde_json::json!({
            "type": "message",
            "id": format!("msg_{}", rand_id()),
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": full_content}],
        }));
    }
    for (_, (call_id, name, args)) in tool_call_args {
        output.push(serde_json::json!({
            "type": "function_call",
            "id": format!("fc_{}", rand_id()),
            "call_id": call_id,
            "name": name,
            "arguments": args,
            "status": "completed",
        }));
    }

    let body = serde_json::json!({
        "id": response_id,
        "object": "response",
        "created_at": std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0),
        "status": "completed",
        "model": model,
        "output": output,
        "usage": usage,
    });

    Response::builder()
        .status(StatusCode::OK)
        .header(CONTENT_TYPE, "application/json")
        .body(Body::from(body.to_string()))
        .unwrap_or_else(|_| fallback_body())
}

fn rand_id() -> String {
    use std::time::SystemTime;
    let n = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);
    let mut buf = String::with_capacity(16);
    let alphabet = b"abcdefghijklmnopqrstuvwxyz0123456789";
    let mut x = n;
    for _ in 0..16 {
        buf.push(alphabet[(x % 36) as usize] as char);
        x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    }
    buf
}

fn error_response(status: StatusCode, message: &str) -> Response<Body> {
    let body = serde_json::json!({"error": {"message": message}});
    Response::builder()
        .status(status)
        .header(CONTENT_TYPE, "application/json")
        .body(Body::from(body.to_string()))
        .unwrap_or_else(|_| fallback_body())
}

fn failed_sse_response(model: &str, message: &str) -> Response<Body> {
    let mut converter = ChatToResponsesStream::new(model.to_string());
    let created = converter.ensure_created();
    let failed = converter.emit_failed(message);
    let done = "data: [DONE]\n\n";
    let body = format!("{created}{failed}{done}");
    Response::builder()
        .status(StatusCode::OK)
        .header(CONTENT_TYPE, "text/event-stream")
        .header(CACHE_CONTROL, "no-cache")
        .header(CONNECTION, "keep-alive")
        .header("X-Accel-Buffering", "no")
        .body(Body::from(body))
        .unwrap_or_else(|_| fallback_body())
}

fn fallback_body() -> Response<Body> {
    Response::new(Body::from("internal proxy error"))
}
