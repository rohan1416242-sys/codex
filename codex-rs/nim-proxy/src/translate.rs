//! Translation between OpenAI Responses API and Chat Completions API.

use std::sync::Arc;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;

use anyhow::Result;
use futures::StreamExt;
use serde::Deserialize;
use serde::Serialize;
use serde_json::Map;
use serde_json::Value;
use tokio::sync::mpsc;
use tracing::debug;
use tracing::warn;

fn next_seq(seq: &Arc<AtomicU64>) -> u64 {
    seq.fetch_add(1, Ordering::Relaxed)
}

fn gen_id(prefix: &str) -> String {
    use std::time::SystemTime;
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0)
        ^ COUNTER.fetch_add(1, Ordering::Relaxed);
    let mut buf = String::with_capacity(22);
    let alphabet = b"abcdefghijklmnopqrstuvwxyz0123456789";
    let mut x = n;
    for _ in 0..22 {
        buf.push(alphabet[(x % 36) as usize] as char);
        x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    }
    format!("{prefix}_{buf}")
}

// ============================================================================
// Request translation: Responses → Chat Completions
//
// TRUE INVISIBLE BRIDGE:
//   - Codex sends `model: "gpt-5.6-sol"` → we silently replace with
//     `backend_model` (e.g. "thinkingmachines/inkling") before forwarding
//   - Response events come back with `model: "gpt-5.6-sol"` (the original)
//     so codex's UI has NO idea a proxy is in the middle
//   - No params injected by default (transparent passthrough)
//   - All codex-specific fields (store, include, service_tier, etc.) are
//     stripped — NIM doesn't understand them and would 400
//   - Image inputs translated from Responses format to OpenAI Chat format
// ============================================================================

pub fn translate_request(
    responses_body: &Value,
    backend_model: &str,
    enable_thinking: bool,
) -> Result<(Value, String)> {
    let obj = responses_body
        .as_object()
        .ok_or_else(|| anyhow::anyhow!("responses request body must be a JSON object"))?;

    // Capture the incoming model name (what codex thinks it's using). We
    // return this so the response stream can use it in response events —
    // this is what makes the bridge INVISIBLE: codex sends "gpt-5.6-sol",
    // response events say "gpt-5.6-sol", but the actual NIM call used
    // `backend_model`.
    let incoming_model = obj
        .get("model")
        .and_then(Value::as_str)
        .unwrap_or("unknown")
        .to_string();

    let instructions = obj
        .get("instructions")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();

    let input = obj.get("input").cloned().unwrap_or(Value::Array(vec![]));
    let messages = build_chat_messages(&instructions, &input)?;

    let mut chat_body = Map::new();
    // Override model with backend_model — this is the core bridge behavior.
    chat_body.insert("model".to_string(), Value::String(backend_model.to_string()));
    chat_body.insert("messages".to_string(), Value::Array(messages));
    chat_body.insert("stream".to_string(), Value::Bool(true));

    if let Some(tools) = obj.get("tools").and_then(Value::as_array) {
        let translated: Vec<Value> = tools
            .iter()
            .filter_map(|t| translate_tool_definition(t))
            .collect();
        if !translated.is_empty() {
            chat_body.insert("tools".to_string(), Value::Array(translated));
        }
    }

    if let Some(tc) = obj.get("tool_choice") {
        if let Some(s) = tc.as_str() {
            chat_body.insert("tool_choice".to_string(), Value::String(s.to_string()));
        } else if let Some(o) = tc.as_object() {
            if o.get("type").and_then(Value::as_str) == Some("function") {
                let mut nested = Map::new();
                nested.insert("type".to_string(), Value::String("function".to_string()));
                let mut func = Map::new();
                if let Some(name) = o.get("name").and_then(Value::as_str) {
                    func.insert("name".to_string(), Value::String(name.to_string()));
                }
                nested.insert("function".to_string(), Value::Object(func));
                chat_body.insert("tool_choice".to_string(), Value::Object(nested));
            }
        }
    }

    if let Some(p) = obj.get("parallel_tool_calls") {
        chat_body.insert("parallel_tool_calls".to_string(), p.clone());
    }

    // Pass through sampling params codex sent (if any). We do NOT inject
    // defaults — transparent passthrough.
    for field in ["temperature", "top_p", "stop", "user"] {
        if let Some(v) = obj.get(field) {
            chat_body.insert(field.to_string(), v.clone());
        }
    }
    // max_output_tokens (Responses API) → max_tokens (Chat Completions)
    if let Some(v) = obj.get("max_output_tokens") {
        chat_body.insert("max_tokens".to_string(), v.clone());
    } else if let Some(v) = obj.get("max_tokens") {
        chat_body.insert("max_tokens".to_string(), v.clone());
    }

    // Only inject thinking params when explicitly enabled. These unlock
    // `reasoning_content` on Nemotron / DeepSeek-R1 / Mistral-Nemotron /
    // Inkling, but they make non-reasoning models super slow if sent.
    // NOTE: Only `chat_template_kwargs.enable_thinking` is universally
    // supported. `reasoning_budget` is rejected by some models (e.g.
    // thinkingmachines/inkling returns 400 "Unsupported parameter"), so
    // we don't send it — NIM will use the model's default reasoning budget.
    if enable_thinking {
        let mut ctk = Map::new();
        ctk.insert("enable_thinking".to_string(), Value::Bool(true));
        chat_body.insert("chat_template_kwargs".to_string(), Value::Object(ctk));
    }

    // NOTE: We intentionally DO NOT forward these codex-specific fields
    // because NIM rejects unknown fields with 400:
    //   - store, stream_options, include, service_tier, prompt_cache_key
    //   - text, client_metadata, previous_response_id, reasoning
    // Codex doesn't need them to be honored — they're optimizations.

    // ─── drop_params: NIM validation fixes (like LiteLLM's drop_params) ───
    // NIM returns 400 "When using tool_choice, tools must be set" if codex
    // sends tool_choice without tools (which happens on simple "hi" turns
    // where no tools are needed). Drop tool_choice in that case.
    let has_tools = chat_body
        .get("tools")
        .and_then(Value::as_array)
        .is_some_and(|t| !t.is_empty());
    if !has_tools {
        chat_body.remove("tool_choice");
        // parallel_tool_calls without tools is also rejected by some models.
        chat_body.remove("parallel_tool_calls");
    }

    // Some NIM models reject `web_search_options` (codex sends it for
    // models that advertise web search). Drop it — NIM doesn't support it.
    chat_body.remove("web_search_options");

    Ok((Value::Object(chat_body), incoming_model))
}

fn translate_tool_definition(tool: &Value) -> Option<Value> {
    let obj = tool.as_object()?;
    let kind = obj.get("type").and_then(Value::as_str).unwrap_or("function");
    if kind != "function" {
        return None;
    }

    let (name, parameters, description) =
        if let Some(f) = obj.get("function").and_then(Value::as_object) {
            (
                f.get("name").and_then(Value::as_str).map(String::from),
                f.get("parameters")
                    .cloned()
                    .unwrap_or(Value::Object(Default::default())),
                f.get("description").and_then(Value::as_str).map(String::from),
            )
        } else {
            (
                obj.get("name").and_then(Value::as_str).map(String::from),
                obj.get("parameters")
                    .cloned()
                    .unwrap_or(Value::Object(Default::default())),
                obj.get("description").and_then(Value::as_str).map(String::from),
            )
        };

    let name = name?;
    let mut function_obj = Map::new();
    function_obj.insert("name".to_string(), Value::String(name));
    function_obj.insert("parameters".to_string(), parameters);
    if let Some(desc) = description {
        function_obj.insert("description".to_string(), Value::String(desc));
    }
    let mut out = Map::new();
    out.insert("type".to_string(), Value::String("function".to_string()));
    out.insert("function".to_string(), Value::Object(function_obj));
    Some(Value::Object(out))
}

fn build_chat_messages(instructions: &str, input: &Value) -> Result<Vec<Value>> {
    let mut messages: Vec<Value> = Vec::new();

    if !instructions.trim().is_empty() {
        messages.push(serde_json::json!({
            "role": "system",
            "content": instructions,
        }));
    }

    let items = match input {
        Value::Array(arr) => arr,
        Value::String(s) => {
            messages.push(serde_json::json!({ "role": "user", "content": s }));
            return Ok(messages);
        }
        _ => return Ok(messages),
    };

    let mut pending_assistant: Option<Map<String, Value>> = None;

    let flush_assistant =
        |pending: &mut Option<Map<String, Value>>, msgs: &mut Vec<Value>| {
            if let Some(mut a) = pending.take() {
                if !a.contains_key("content") {
                    a.insert("content".to_string(), Value::Null);
                }
                msgs.push(Value::Object(a));
            }
        };

    for item in items {
        let obj = match item.as_object() {
            Some(o) => o,
            None => continue,
        };
        let kind = obj.get("type").and_then(Value::as_str).unwrap_or("");

        match kind {
            "message" => {
                flush_assistant(&mut pending_assistant, &mut messages);
                let role = obj
                    .get("role")
                    .and_then(Value::as_str)
                    .unwrap_or("user")
                    .to_string();
                let content = build_chat_content(obj.get("content"));
                messages.push(serde_json::json!({ "role": role, "content": content }));
            }
            "function_call" => {
                let assistant = pending_assistant.get_or_insert_with(|| {
                    let mut m = Map::new();
                    m.insert("role".to_string(), Value::String("assistant".to_string()));
                    m
                });
                let tool_calls = assistant
                    .entry("tool_calls".to_string())
                    .or_insert_with(|| Value::Array(vec![]));
                if let Some(arr) = tool_calls.as_array_mut() {
                    let id = obj
                        .get("call_id")
                        .and_then(Value::as_str)
                        .unwrap_or("call")
                        .to_string();
                    let name = obj
                        .get("name")
                        .and_then(Value::as_str)
                        .unwrap_or("")
                        .to_string();
                    let arguments = obj
                        .get("arguments")
                        .and_then(Value::as_str)
                        .unwrap_or("{}")
                        .to_string();
                    arr.push(serde_json::json!({
                        "id": id,
                        "type": "function",
                        "function": {"name": name, "arguments": arguments}
                    }));
                }
            }
            "function_call_output" => {
                flush_assistant(&mut pending_assistant, &mut messages);
                let call_id = obj
                    .get("call_id")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string();
                let output = obj
                    .get("output")
                    .map(|v| match v {
                        Value::String(s) => s.clone(),
                        other => other.to_string(),
                    })
                    .unwrap_or_default();
                messages.push(serde_json::json!({
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": output,
                }));
            }
            "reasoning" => {}
            _ => {
                debug!("translate_request: skipping unknown item type `{kind}`");
            }
        }
    }
    flush_assistant(&mut pending_assistant, &mut messages);

    Ok(messages)
}

/// Build the `content` field for a Chat Completions message.
///
/// Returns either:
///   - A plain string (if the message is text-only) — most efficient
///   - An array of content parts (if the message has images) — OpenAI format
///
/// Translates Responses-API content types to Chat Completions:
///   - `input_text` / `output_text` / `text` → `{"type":"text","text":...}`
///   - `input_image` → `{"type":"image_url","image_url":{"url":...}}`
///   - `input_audio` → dropped (NIM doesn't support audio in chat completions)
fn build_chat_content(content: Option<&Value>) -> Value {
    let Some(content) = content else {
        return Value::String(String::new());
    };
    match content {
        Value::String(s) => Value::String(s.clone()),
        Value::Array(arr) => {
            let mut parts: Vec<Value> = Vec::new();
            let mut text_only = true;
            for part in arr {
                let kind = part.get("type").and_then(Value::as_str).unwrap_or("");
                match kind {
                    "input_text" | "output_text" | "text" => {
                        if let Some(t) = part.get("text").and_then(Value::as_str) {
                            parts.push(serde_json::json!({"type": "text", "text": t}));
                        }
                    }
                    "input_image" => {
                        text_only = false;
                        if let Some(url) = part.get("image_url").and_then(Value::as_str) {
                            parts.push(serde_json::json!({
                                "type": "image_url",
                                "image_url": {"url": url}
                            }));
                        }
                    }
                    _ => {}
                }
            }
            // If text-only, return a plain string (more efficient + compatible)
            if text_only {
                let s: String = parts
                    .iter()
                    .filter_map(|p| p.get("text").and_then(Value::as_str).map(String::from))
                    .collect::<Vec<_>>()
                    .join("\n");
                Value::String(s)
            } else {
                Value::Array(parts)
            }
        }
        _ => Value::String(content.to_string()),
    }
}

/// Legacy helper — kept for backwards compat. Use `build_chat_content` for
/// new code since it handles images.
#[allow(dead_code)]
fn message_content_to_string(content: Option<&Value>) -> String {
    match build_chat_content(content) {
        Value::String(s) => s,
        other => other.to_string(),
    }
}

// ============================================================================
// Response stream translation: Chat Completions SSE → Responses SSE
// ============================================================================

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponsesEvent {
    #[serde(rename = "response.created")]
    Created {
        sequence_number: u64,
        response: CreatedResponseBody,
    },
    #[serde(rename = "response.output_item.added")]
    OutputItemAdded {
        sequence_number: u64,
        output_index: u32,
        item: Value,
    },
    #[serde(rename = "response.output_text.delta")]
    OutputTextDelta {
        sequence_number: u64,
        item_id: String,
        output_index: u32,
        content_index: u32,
        delta: String,
    },
    /// Streaming reasoning delta — Nemotron / DeepSeek-R1 / Mistral-Nemotron.
    #[serde(rename = "response.reasoning_text.delta")]
    ReasoningTextDelta {
        sequence_number: u64,
        item_id: String,
        output_index: u32,
        content_index: u32,
        delta: String,
    },
    #[serde(rename = "response.function_call_arguments.delta")]
    FunctionCallArgumentsDelta {
        sequence_number: u64,
        item_id: String,
        output_index: u32,
        delta: String,
    },
    #[serde(rename = "response.output_item.done")]
    OutputItemDone {
        sequence_number: u64,
        output_index: u32,
        item: Value,
    },
    #[serde(rename = "response.completed")]
    Completed {
        sequence_number: u64,
        response: CompletedResponseBody,
    },
    #[serde(rename = "response.failed")]
    Failed {
        sequence_number: u64,
        response: FailedResponseBody,
    },
}

#[derive(Debug, Clone, Serialize)]
pub struct CreatedResponseBody {
    pub id: String,
    pub object: String,
    pub created_at: u64,
    pub status: String,
    pub model: String,
    pub output: Vec<Value>,
}

#[derive(Debug, Clone, Serialize)]
pub struct CompletedResponseBody {
    pub id: String,
    pub object: String,
    pub created_at: u64,
    pub status: String,
    pub model: String,
    pub output: Vec<Value>,
    pub usage: Option<UsageSummary>,
}

#[derive(Debug, Clone, Serialize)]
pub struct FailedResponseBody {
    pub id: String,
    pub object: String,
    pub created_at: u64,
    pub status: String,
    pub error: Value,
    pub output: Vec<Value>,
}

#[derive(Debug, Clone, Serialize)]
pub struct UsageSummary {
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub total_tokens: u64,
}

pub fn format_sse_event(event: &ResponsesEvent) -> String {
    let json = serde_json::to_string(event).unwrap_or_else(|_| "{}".to_string());
    format!("data: {json}\n\n")
}

pub struct ChatToResponsesStream {
    seq: Arc<AtomicU64>,
    response_id: String,
    created_at: u64,
    model: String,
    message_item_id: Option<String>,
    message_output_index: u32,
    message_text_buf: String,
    reasoning_item_id: Option<String>,
    reasoning_output_index: u32,
    reasoning_text_buf: String,
    tool_calls: std::collections::BTreeMap<u32, ToolCallState>,
    next_tool_output_index: u32,
    created_emitted: bool,
    final_usage: Option<UsageSummary>,
    completed_emitted: bool,
}

#[derive(Debug, Default)]
struct ToolCallState {
    item_id: String,
    call_id: String,
    name: String,
    arguments_buf: String,
    output_index: u32,
    added_emitted: bool,
}

impl ChatToResponsesStream {
    pub fn new(model: String) -> Self {
        let response_id = gen_id("resp");
        let created_at = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        Self {
            seq: Arc::new(AtomicU64::new(0)),
            response_id,
            created_at,
            model,
            message_item_id: None,
            message_output_index: 0,
            message_text_buf: String::new(),
            reasoning_item_id: None,
            reasoning_output_index: 0,
            reasoning_text_buf: String::new(),
            tool_calls: Default::default(),
            next_tool_output_index: 1,
            created_emitted: false,
            final_usage: None,
            completed_emitted: false,
        }
    }

    pub fn ensure_created(&mut self) -> String {
        if self.created_emitted {
            return String::new();
        }
        self.created_emitted = true;
        let body = CreatedResponseBody {
            id: self.response_id.clone(),
            object: "response".to_string(),
            created_at: self.created_at,
            status: "in_progress".to_string(),
            model: self.model.clone(),
            output: vec![],
        };
        let event = ResponsesEvent::Created {
            sequence_number: next_seq(&self.seq),
            response: body,
        };
        format_sse_event(&event)
    }

    pub fn handle_chunk(&mut self, chunk: &ChatChunk) -> String {
        let mut out = String::new();
        out.push_str(&self.ensure_created());

        if let Some(usage) = &chunk.usage {
            self.final_usage = Some(UsageSummary {
                input_tokens: usage.prompt_tokens,
                output_tokens: usage.completion_tokens,
                total_tokens: usage.total_tokens,
            });
        }

        let Some(choice) = chunk.choices.first() else {
            return out;
        };

        // ─── Reasoning deltas (Nemotron, DeepSeek-R1, Mistral-Nemotron) ────
        if let Some(reasoning) = choice.delta.reasoning_content.as_deref() {
            if !reasoning.is_empty() {
                if self.reasoning_item_id.is_none() {
                    self.reasoning_item_id = Some(gen_id("rs"));
                    let item_id = self.reasoning_item_id.clone().unwrap_or_default();
                    let added_item = serde_json::json!({
                        "type": "reasoning",
                        "id": item_id,
                        "summary": [],
                        "content": [],
                        "status": "in_progress",
                    });
                    out.push_str(&format_sse_event(&ResponsesEvent::OutputItemAdded {
                        sequence_number: next_seq(&self.seq),
                        output_index: self.reasoning_output_index,
                        item: added_item,
                    }));
                }
                let item_id = self.reasoning_item_id.clone().unwrap_or_default();
                self.reasoning_text_buf.push_str(reasoning);
                out.push_str(&format_sse_event(&ResponsesEvent::ReasoningTextDelta {
                    sequence_number: next_seq(&self.seq),
                    item_id,
                    output_index: self.reasoning_output_index,
                    content_index: 0,
                    delta: reasoning.to_string(),
                }));
            }
        }

        if let Some(text) = choice.delta.content.as_deref() {
            if !text.is_empty() {
                if self.message_item_id.is_none() {
                    self.message_item_id = Some(gen_id("msg"));
                    let item_id = self.message_item_id.clone().unwrap_or_default();
                    let added_item = serde_json::json!({
                        "type": "message",
                        "id": item_id,
                        "role": "assistant",
                        "status": "in_progress",
                        "content": [],
                    });
                    out.push_str(&format_sse_event(&ResponsesEvent::OutputItemAdded {
                        sequence_number: next_seq(&self.seq),
                        output_index: self.message_output_index,
                        item: added_item,
                    }));
                }
                let item_id = self.message_item_id.clone().unwrap_or_default();
                self.message_text_buf.push_str(text);
                out.push_str(&format_sse_event(&ResponsesEvent::OutputTextDelta {
                    sequence_number: next_seq(&self.seq),
                    item_id,
                    output_index: self.message_output_index,
                    content_index: 0,
                    delta: text.to_string(),
                }));
            }
        }

        if let Some(tool_calls) = &choice.delta.tool_calls {
            for tc in tool_calls {
                let idx = tc.index;
                let state = self.tool_calls.entry(idx).or_insert_with(|| {
                    let n = self.next_tool_output_index;
                    self.next_tool_output_index += 1;
                    ToolCallState {
                        item_id: gen_id("fc"),
                        call_id: tc.id.clone().unwrap_or_else(|| gen_id("call")),
                        name: String::new(),
                        arguments_buf: String::new(),
                        output_index: n,
                        added_emitted: false,
                    }
                });

                if let Some(id) = &tc.id
                    && !id.is_empty()
                {
                    state.call_id = id.clone();
                }
                if let Some(name) = tc.function.as_ref().and_then(|f| f.name.as_deref()) {
                    if !name.is_empty() && state.name.is_empty() {
                        state.name = name.to_string();
                    }
                }

                if !state.name.is_empty() && !state.added_emitted {
                    state.added_emitted = true;
                    let added_item = serde_json::json!({
                        "type": "function_call",
                        "id": state.item_id,
                        "call_id": state.call_id,
                        "name": state.name,
                        "arguments": "",
                        "status": "in_progress",
                    });
                    out.push_str(&format_sse_event(&ResponsesEvent::OutputItemAdded {
                        sequence_number: next_seq(&self.seq),
                        output_index: state.output_index,
                        item: added_item,
                    }));
                }

                if let Some(args) = tc.function.as_ref().and_then(|f| f.arguments.as_deref()) {
                    if !args.is_empty() {
                        state.arguments_buf.push_str(args);
                        out.push_str(&format_sse_event(
                            &ResponsesEvent::FunctionCallArgumentsDelta {
                                sequence_number: next_seq(&self.seq),
                                item_id: state.item_id.clone(),
                                output_index: state.output_index,
                                delta: args.to_string(),
                            },
                        ));
                    }
                }
            }
        }

        if let Some(_reason) = &choice.finish_reason {
            if let Some(item_id) = self.reasoning_item_id.take() {
                let text = std::mem::take(&mut self.reasoning_text_buf);
                let done_item = serde_json::json!({
                    "type": "reasoning",
                    "id": item_id,
                    "summary": [],
                    "content": [{"type": "reasoning_text", "text": text}],
                    "status": "completed",
                });
                out.push_str(&format_sse_event(&ResponsesEvent::OutputItemDone {
                    sequence_number: next_seq(&self.seq),
                    output_index: self.reasoning_output_index,
                    item: done_item,
                }));
            }

            if let Some(item_id) = self.message_item_id.take() {
                let text = std::mem::take(&mut self.message_text_buf);
                let done_item = serde_json::json!({
                    "type": "message",
                    "id": item_id,
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": text}],
                });
                out.push_str(&format_sse_event(&ResponsesEvent::OutputItemDone {
                    sequence_number: next_seq(&self.seq),
                    output_index: self.message_output_index,
                    item: done_item,
                }));
            }

            let mut indices: Vec<u32> = self.tool_calls.keys().copied().collect();
            indices.sort_unstable();
            for idx in indices {
                let state = self.tool_calls.remove(&idx).unwrap_or_default();
                if !state.added_emitted {
                    warn!("tool call index {idx} never received a name; dropping");
                    continue;
                }
                let done_item = serde_json::json!({
                    "type": "function_call",
                    "id": state.item_id,
                    "call_id": state.call_id,
                    "name": state.name,
                    "arguments": state.arguments_buf,
                    "status": "completed",
                });
                out.push_str(&format_sse_event(&ResponsesEvent::OutputItemDone {
                    sequence_number: next_seq(&self.seq),
                    output_index: state.output_index,
                    item: done_item,
                }));
            }

            out.push_str(&self.emit_completed());
        }

        out
    }

    fn emit_completed(&mut self) -> String {
        if self.completed_emitted {
            return String::new();
        }
        self.completed_emitted = true;

        let output: Vec<Value> = Vec::new();
        let body = CompletedResponseBody {
            id: self.response_id.clone(),
            object: "response".to_string(),
            created_at: self.created_at,
            status: "completed".to_string(),
            model: self.model.clone(),
            output,
            usage: self.final_usage.clone(),
        };
        let event = ResponsesEvent::Completed {
            sequence_number: next_seq(&self.seq),
            response: body,
        };
        format_sse_event(&event)
    }

    pub fn emit_failed(&mut self, error_message: &str) -> String {
        let error = serde_json::json!({
            "code": "upstream_error",
            "message": error_message,
        });
        let body = FailedResponseBody {
            id: self.response_id.clone(),
            object: "response".to_string(),
            created_at: self.created_at,
            status: "failed".to_string(),
            error,
            output: vec![],
        };
        let event = ResponsesEvent::Failed {
            sequence_number: next_seq(&self.seq),
            response: body,
        };
        format_sse_event(&event)
    }
}

// ============================================================================
// Chat Completions SSE chunk shapes
// ============================================================================

#[derive(Debug, Deserialize)]
pub struct ChatChunk {
    pub id: Option<String>,
    pub model: Option<String>,
    pub choices: Vec<ChatChoice>,
    pub usage: Option<ChatUsage>,
}

#[derive(Debug, Deserialize)]
pub struct ChatChoice {
    pub index: u32,
    pub delta: ChatDelta,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Deserialize, Default)]
pub struct ChatDelta {
    #[serde(default)]
    pub role: Option<String>,
    #[serde(default)]
    pub content: Option<String>,
    #[serde(default)]
    pub tool_calls: Option<Vec<ChatToolCall>>,
    #[serde(default)]
    pub reasoning_content: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct ChatToolCall {
    pub index: u32,
    pub id: Option<String>,
    #[serde(default)]
    pub function: Option<ChatToolCallFunction>,
}

#[derive(Debug, Deserialize, Default)]
pub struct ChatToolCallFunction {
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub arguments: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct ChatUsage {
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub total_tokens: u64,
}

pub fn spawn_stream_converter(
    upstream_stream: impl futures::Stream<Item = Result<bytes::Bytes, reqwest::Error>> + Send + 'static,
    // The model name to use in response events. This should be the
    // INCOMING model (what codex sent), NOT the backend model — so codex's
    // UI sees the "expected" model name and can't tell a proxy is in the
    // middle.
    response_model: String,
) -> mpsc::Receiver<Result<bytes::Bytes, std::io::Error>> {
    let (tx, rx) = mpsc::channel::<Result<bytes::Bytes, std::io::Error>>(256);

    tokio::spawn(async move {
        let mut state = ChatToResponsesStream::new(response_model.clone());
        let mut line_buf = String::new();
        let mut stream = upstream_stream.boxed();

        let created = state.ensure_created();
        if !created.is_empty() {
            let _ = tx.send(Ok(bytes::Bytes::from(created))).await;
        }

        while let Some(chunk_result) = stream.next().await {
            let chunk = match chunk_result {
                Ok(c) => c,
                Err(e) => {
                    let failed = state.emit_failed(&format!("upstream read error: {e}"));
                    let _ = tx.send(Ok(bytes::Bytes::from(failed))).await;
                    let _ = tx.send(Ok(bytes::Bytes::from("data: [DONE]\n\n"))).await;
                    return;
                }
            };

            line_buf.push_str(&String::from_utf8_lossy(&chunk));
            loop {
                let Some(newline_pos) = line_buf.find('\n') else {
                    break;
                };
                let line: String = line_buf.drain(..=newline_pos).collect();
                let trimmed = line.trim_end_matches(['\r', '\n']);

                if trimmed.is_empty() {
                    continue;
                }
                if let Some(data) = trimmed.strip_prefix("data: ") {
                    let data = data.trim();
                    if data == "[DONE]" {
                        let _ = state.reasoning_item_id.take();
                        let _ = state.message_item_id.take();
                        let completed = state.emit_completed();
                        if !completed.is_empty() {
                            let _ = tx.send(Ok(bytes::Bytes::from(completed))).await;
                        }
                        let _ = tx.send(Ok(bytes::Bytes::from("data: [DONE]\n\n"))).await;
                        return;
                    }
                    match serde_json::from_str::<ChatChunk>(data) {
                        Ok(chat_chunk) => {
                            let out = state.handle_chunk(&chat_chunk);
                            if !out.is_empty() {
                                let _ = tx.send(Ok(bytes::Bytes::from(out))).await;
                            }
                        }
                        Err(e) => {
                            debug!("failed to parse chat chunk `{data}`: {e}");
                        }
                    }
                }
            }
        }

        let _ = state.reasoning_item_id.take();
        let _ = state.message_item_id.take();
        let completed = state.emit_completed();
        if !completed.is_empty() {
            let _ = tx.send(Ok(bytes::Bytes::from(completed))).await;
        }
        let _ = tx.send(Ok(bytes::Bytes::from("data: [DONE]\n\n"))).await;
    });

    rx
}

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;

    #[test]
    fn translates_basic_request() {
        let body = serde_json::json!({
            "model": "gpt-5.6-sol",
            "instructions": "Be helpful.",
            "input": [
                {"type": "message", "role": "user", "content": [
                    {"type": "input_text", "text": "hello"}
                ]}
            ],
            "tool_choice": "auto",
            "parallel_tool_calls": false
        });
        let (chat, incoming) = translate_request(&body, "qwen/qwen3-next-80b-a3b-instruct", false).unwrap();
        let messages = chat.get("messages").unwrap().as_array().unwrap();
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0]["role"], "system");
        assert_eq!(messages[0]["content"], "Be helpful.");
        assert_eq!(messages[1]["role"], "user");
        assert_eq!(messages[1]["content"], "hello");
        // CRITICAL: model name is OVERRIDDEN with backend model, not what codex sent.
        assert_eq!(chat["model"], "qwen/qwen3-next-80b-a3b-instruct");
        assert_ne!(chat["model"], "gpt-5.6-sol");
        // But we return the incoming model name so response events can use it
        // (this is what makes the bridge INVISIBLE).
        assert_eq!(incoming, "gpt-5.6-sol");
        assert_eq!(chat["stream"], true);
        // tool_choice is DROPPED because no tools are present in this request
        // (NIM rejects tool_choice without tools with a 400).
        assert!(chat.get("tool_choice").is_none());
    }

    #[test]
    fn translates_function_call_history() {
        let body = serde_json::json!({
            "model": "x",
            "input": [
                {"type": "message", "role": "user", "content": [
                    {"type": "input_text", "text": "list files"}
                ]},
                {"type": "function_call", "name": "shell", "arguments": "{\"cmd\":\"ls\"}", "call_id": "call_1"},
                {"type": "function_call_output", "call_id": "call_1", "output": "file.txt"}
            ]
        });
        let (chat, _) = translate_request(&body, "qwen/qwen3", false).unwrap();
        let messages = chat.get("messages").unwrap().as_array().unwrap();
        assert_eq!(messages.len(), 3);
        assert_eq!(messages[0]["role"], "user");
        assert_eq!(messages[1]["role"], "assistant");
        assert_eq!(messages[1]["tool_calls"][0]["id"], "call_1");
        assert_eq!(messages[1]["tool_calls"][0]["function"]["name"], "shell");
        assert_eq!(
            messages[1]["tool_calls"][0]["function"]["arguments"],
            "{\"cmd\":\"ls\"}"
        );
        assert_eq!(messages[2]["role"], "tool");
        assert_eq!(messages[2]["tool_call_id"], "call_1");
        assert_eq!(messages[2]["content"], "file.txt");
    }

    #[test]
    fn translates_flat_tool_definition() {
        let body = serde_json::json!({
            "model": "x",
            "input": [],
            "tools": [
                {"type": "function", "name": "shell", "parameters": {"type": "object"}, "description": "Run a shell command"}
            ]
        });
        let (chat, _) = translate_request(&body, "qwen/qwen3", false).unwrap();
        let tools = chat.get("tools").unwrap().as_array().unwrap();
        assert_eq!(tools[0]["type"], "function");
        assert_eq!(tools[0]["function"]["name"], "shell");
        assert_eq!(tools[0]["function"]["description"], "Run a shell command");
    }

    #[test]
    fn handles_string_input() {
        let body = serde_json::json!({
            "model": "x",
            "input": "hello world"
        });
        let (chat, _) = translate_request(&body, "qwen/qwen3", false).unwrap();
        let messages = chat.get("messages").unwrap().as_array().unwrap();
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0]["role"], "user");
        assert_eq!(messages[0]["content"], "hello world");
    }

    #[test]
    fn enable_thinking_injects_reasoning_params() {
        let body = serde_json::json!({"model": "x", "input": "hi"});
        let (chat, _) = translate_request(&body, "nemotron", true).unwrap();
        assert_eq!(chat["chat_template_kwargs"]["enable_thinking"], true);
        // Note: reasoning_budget is NOT sent because some models (inkling)
        // reject it with 400 "Unsupported parameter".
        assert!(chat.get("reasoning_budget").is_none());
    }

    #[test]
    fn enable_thinking_off_by_default_omits_reasoning_params() {
        let body = serde_json::json!({"model": "x", "input": "hi"});
        let (chat, _) = translate_request(&body, "qwen", false).unwrap();
        // Qwen3 should NOT have thinking params injected by default — this
        // was the cause of the "1m 59s for hi" slowness.
        assert!(chat.get("chat_template_kwargs").is_none());
        assert!(chat.get("reasoning_budget").is_none());
    }

    #[test]
    fn strips_codex_specific_fields() {
        let body = serde_json::json!({
            "model": "gpt-5.6-sol",
            "input": "hi",
            "store": true,
            "include": ["reasoning.encrypted_content"],
            "service_tier": "priority",
            "prompt_cache_key": "abc123",
            "text": {"format": {"type": "text"}},
            "client_metadata": {"turn_id": "xyz"},
            "reasoning": {"effort": "high"}
        });
        let (chat, _) = translate_request(&body, "qwen", false).unwrap();
        // None of these codex-specific fields should appear in the chat body
        // (NIM would reject them with 400).
        assert!(chat.get("store").is_none());
        assert!(chat.get("include").is_none());
        assert!(chat.get("service_tier").is_none());
        assert!(chat.get("prompt_cache_key").is_none());
        assert!(chat.get("text").is_none());
        assert!(chat.get("client_metadata").is_none());
        assert!(chat.get("reasoning").is_none());
    }

    #[test]
    fn translates_image_input() {
        let body = serde_json::json!({
            "model": "gpt-5.6-sol",
            "input": [{
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "What's in this image?"},
                    {"type": "input_image", "image_url": "data:image/png;base64,iVBOR..."}
                ]
            }]
        });
        let (chat, _) = translate_request(&body, "qwen/qwen2.5-vl", false).unwrap();
        let messages = chat.get("messages").unwrap().as_array().unwrap();
        // Content should be an array (because it has an image), not a plain string.
        assert!(messages[0]["content"].is_array());
        let content = messages[0]["content"].as_array().unwrap();
        assert_eq!(content.len(), 2);
        assert_eq!(content[0]["type"], "text");
        assert_eq!(content[1]["type"], "image_url");
        assert!(content[1]["image_url"]["url"].as_str().unwrap().starts_with("data:image/png"));
    }

    #[test]
    fn text_only_content_returns_plain_string() {
        let body = serde_json::json!({
            "model": "gpt-5.6-sol",
            "input": [{
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "hello"},
                    {"type": "input_text", "text": "world"}
                ]
            }]
        });
        let (chat, _) = translate_request(&body, "qwen", false).unwrap();
        let messages = chat.get("messages").unwrap().as_array().unwrap();
        // Text-only content should be a plain string (joined), not an array.
        assert!(messages[0]["content"].is_string());
        assert_eq!(messages[0]["content"], "hello\nworld");
    }

    #[test]
    fn drops_tool_choice_when_no_tools() {
        // Codex sends tool_choice="auto" even on simple "hi" turns where no
        // tools are needed. NIM rejects this with 400. We must drop it.
        let body = serde_json::json!({
            "model": "gpt-5.6-sol",
            "input": "hi",
            "tool_choice": "auto",
            "parallel_tool_calls": false
        });
        let (chat, _) = translate_request(&body, "inkling", false).unwrap();
        assert!(chat.get("tool_choice").is_none(), "tool_choice must be dropped when no tools");
        assert!(chat.get("parallel_tool_calls").is_none(), "parallel_tool_calls must be dropped when no tools");
    }

    #[test]
    fn keeps_tool_choice_when_tools_present() {
        let body = serde_json::json!({
            "model": "gpt-5.6-sol",
            "input": "list files",
            "tool_choice": "auto",
            "parallel_tool_calls": true,
            "tools": [
                {"type": "function", "name": "shell", "parameters": {"type": "object"}, "description": "Run a shell command"}
            ]
        });
        let (chat, _) = translate_request(&body, "inkling", false).unwrap();
        assert_eq!(chat["tool_choice"], "auto");
        assert_eq!(chat["parallel_tool_calls"], true);
    }

    #[test]
    fn drops_web_search_options() {
        let body = serde_json::json!({
            "model": "gpt-5.6-sol",
            "input": "hi",
            "web_search_options": {"search_context_size": "medium"}
        });
        let (chat, _) = translate_request(&body, "inkling", false).unwrap();
        assert!(chat.get("web_search_options").is_none());
    }

    #[test]
    fn stream_emits_created_first() {
        let mut state = ChatToResponsesStream::new("test-model".to_string());
        let created = state.ensure_created();
        assert!(created.contains("response.created"));
        assert!(created.contains("test-model"));
        let again = state.ensure_created();
        assert!(again.is_empty());
    }

    #[test]
    fn stream_translates_text_delta() {
        let mut state = ChatToResponsesStream::new("m".to_string());
        let chunk = ChatChunk {
            id: Some("c1".to_string()),
            model: Some("m".to_string()),
            choices: vec![ChatChoice {
                index: 0,
                delta: ChatDelta {
                    role: Some("assistant".to_string()),
                    content: Some("Hello".to_string()),
                    tool_calls: None,
                    reasoning_content: None,
                },
                finish_reason: None,
            }],
            usage: None,
        };
        let out = state.handle_chunk(&chunk);
        assert!(out.contains("response.created"));
        assert!(out.contains("response.output_item.added"));
        assert!(out.contains("response.output_text.delta"));
        assert!(out.contains("\"delta\":\"Hello\""));
    }

    #[test]
    fn stream_translates_tool_call() {
        let mut state = ChatToResponsesStream::new("m".to_string());

        let chunk1 = ChatChunk {
            id: Some("c1".to_string()),
            model: Some("m".to_string()),
            choices: vec![ChatChoice {
                index: 0,
                delta: ChatDelta {
                    role: Some("assistant".to_string()),
                    content: None,
                    tool_calls: Some(vec![ChatToolCall {
                        index: 0,
                        id: Some("call_42".to_string()),
                        function: Some(ChatToolCallFunction {
                            name: Some("shell".to_string()),
                            arguments: Some("{\"cmd\":".to_string()),
                        }),
                    }]),
                    reasoning_content: None,
                },
                finish_reason: None,
            }],
            usage: None,
        };
        let out1 = state.handle_chunk(&chunk1);
        assert!(out1.contains("response.output_item.added"));
        assert!(out1.contains("\"name\":\"shell\""));
        assert!(out1.contains("response.function_call_arguments.delta"));
        assert!(out1.contains("\"delta\":\"{\\\"cmd\\\":\""));

        let chunk2 = ChatChunk {
            id: Some("c2".to_string()),
            model: Some("m".to_string()),
            choices: vec![ChatChoice {
                index: 0,
                delta: ChatDelta {
                    role: None,
                    content: None,
                    tool_calls: Some(vec![ChatToolCall {
                        index: 0,
                        id: None,
                        function: Some(ChatToolCallFunction {
                            name: None,
                            arguments: Some("\"ls\"}".to_string()),
                        }),
                    }]),
                    reasoning_content: None,
                },
                finish_reason: Some("tool_calls".to_string()),
            }],
            usage: Some(ChatUsage {
                prompt_tokens: 10,
                completion_tokens: 5,
                total_tokens: 15,
            }),
        };
        let out2 = state.handle_chunk(&chunk2);
        assert!(out2.contains("response.function_call_arguments.delta"));
        assert!(out2.contains("response.output_item.done"));
        assert!(out2.contains("\"arguments\":\"{\\\"cmd\\\":\\\"ls\\\"}\""));
        assert!(out2.contains("response.completed"));
        assert!(out2.contains("\"input_tokens\":10"));
    }

    #[test]
    fn stream_emits_failed_on_demand() {
        let mut state = ChatToResponsesStream::new("m".to_string());
        let out = state.emit_failed("upstream 500");
        assert!(out.contains("response.failed"));
        assert!(out.contains("upstream 500"));
    }
}
