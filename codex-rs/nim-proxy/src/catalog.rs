//! Curated catalog of NIM-hosted models with display names + context info.

use std::sync::LazyLock;

use serde::Deserialize;
use serde::Serialize;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CuratedModel {
    pub id: String,
    pub display_name: String,
    pub description: String,
    pub context_window: u32,
    pub reasoning: bool,
    pub tool_calls: bool,
    pub use_case: ModelUseCase,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ModelUseCase {
    Coding,
    Reasoning,
    Fast,
    Specialty,
}

pub static CURATED_MODELS: LazyLock<Vec<CuratedModel>> = LazyLock::new(|| {
    vec![
        CuratedModel {
            id: "qwen/qwen3-next-80b-a3b-instruct".to_string(),
            display_name: "Qwen3-Next 80B (MoE)".to_string(),
            description: "Best overall — fast MoE model with strong coding ability".to_string(),
            context_window: 262144,
            reasoning: false,
            tool_calls: true,
            use_case: ModelUseCase::Coding,
        },
        CuratedModel {
            id: "deepseek-ai/deepseek-v4-pro".to_string(),
            display_name: "DeepSeek V4 Pro".to_string(),
            description: "Excellent coder, slightly slower but very thorough".to_string(),
            context_window: 128000,
            reasoning: false,
            tool_calls: true,
            use_case: ModelUseCase::Coding,
        },
        CuratedModel {
            id: "meta/llama-3.3-70b-instruct".to_string(),
            display_name: "Llama 3.3 70B".to_string(),
            description: "Solid baseline instruct model".to_string(),
            context_window: 128000,
            reasoning: false,
            tool_calls: true,
            use_case: ModelUseCase::Coding,
        },
        CuratedModel {
            id: "qwen/qwen3.5-397b-a17b".to_string(),
            display_name: "Qwen3.5 397B (MoE)".to_string(),
            description: "Large MoE — high capability, higher latency".to_string(),
            context_window: 262144,
            reasoning: false,
            tool_calls: true,
            use_case: ModelUseCase::Coding,
        },
        CuratedModel {
            id: "nvidia/llama-3.3-nemotron-super-49b-v1.5".to_string(),
            display_name: "Nemotron Super 49B v1.5".to_string(),
            description: "NVIDIA reasoning model — slow but thinks deeply".to_string(),
            context_window: 128000,
            reasoning: true,
            tool_calls: true,
            use_case: ModelUseCase::Reasoning,
        },
        CuratedModel {
            id: "nvidia/llama-3.1-nemotron-ultra-253b-v1".to_string(),
            display_name: "Nemotron Ultra 253B".to_string(),
            description: "Massive reasoning model — for hardest problems".to_string(),
            context_window: 128000,
            reasoning: true,
            tool_calls: true,
            use_case: ModelUseCase::Reasoning,
        },
        CuratedModel {
            id: "nvidia/llama-3.1-nemotron-nano-8b-v1".to_string(),
            display_name: "Nemotron Nano 8B".to_string(),
            description: "Fast lightweight reasoning model".to_string(),
            context_window: 128000,
            reasoning: true,
            tool_calls: true,
            use_case: ModelUseCase::Fast,
        },
        CuratedModel {
            id: "mistralai/mistral-nemotron".to_string(),
            display_name: "Mistral Nemotron".to_string(),
            description: "Mistral + NVIDIA reasoning tuning".to_string(),
            context_window: 128000,
            reasoning: true,
            tool_calls: true,
            use_case: ModelUseCase::Reasoning,
        },
        CuratedModel {
            id: "meta/codellama-70b".to_string(),
            display_name: "Code Llama 70B".to_string(),
            description: "Meta's code-specialized Llama — pure code completion".to_string(),
            context_window: 16000,
            reasoning: false,
            tool_calls: false,
            use_case: ModelUseCase::Specialty,
        },
        CuratedModel {
            id: "ibm/granite-34b-code-instruct".to_string(),
            display_name: "Granite 34B Code".to_string(),
            description: "IBM's enterprise code model — solid for refactors".to_string(),
            context_window: 8000,
            reasoning: false,
            tool_calls: false,
            use_case: ModelUseCase::Specialty,
        },
        CuratedModel {
            id: "deepseek-ai/deepseek-coder-6.7b-instruct".to_string(),
            display_name: "DeepSeek Coder 6.7B".to_string(),
            description: "Fast small coder — great for quick edits".to_string(),
            context_window: 16000,
            reasoning: false,
            tool_calls: false,
            use_case: ModelUseCase::Fast,
        },
        CuratedModel {
            id: "google/codegemma-7b".to_string(),
            display_name: "CodeGemma 7B".to_string(),
            description: "Google's code model — fast, lightweight".to_string(),
            context_window: 8000,
            reasoning: false,
            tool_calls: false,
            use_case: ModelUseCase::Fast,
        },
    ]
});

pub fn build_models_response(upstream_ids: &[String]) -> serde_json::Value {
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut data: Vec<serde_json::Value> = Vec::new();

    for m in CURATED_MODELS.iter() {
        if seen.insert(m.id.clone()) {
            data.push(serde_json::json!({
                "id": m.id,
                "object": "model",
                "created": 0,
                "owned_by": "nvidia",
                "display_name": m.display_name,
                "description": m.description,
                "context_window": m.context_window,
                "reasoning": m.reasoning,
                "tool_calls": m.tool_calls,
                "use_case": m.use_case,
                "curated": true,
            }));
        }
    }

    for id in upstream_ids {
        if seen.insert(id.clone()) {
            data.push(serde_json::json!({
                "id": id,
                "object": "model",
                "created": 0,
                "owned_by": "nvidia",
                "curated": false,
            }));
        }
    }

    serde_json::json!({"object": "list", "data": data})
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn curated_models_have_unique_ids() {
        let mut seen = std::collections::HashSet::new();
        for m in CURATED_MODELS.iter() {
            assert!(seen.insert(m.id.clone()), "duplicate id: {}", m.id);
        }
    }

    #[test]
    fn build_response_merges_curated_and_upstream() {
        let upstream = vec![
            "qwen/qwen3-next-80b-a3b-instruct".to_string(),
            "some/other-model".to_string(),
        ];
        let resp = build_models_response(&upstream);
        let data = resp.get("data").unwrap().as_array().unwrap();
        assert!(data.len() >= CURATED_MODELS.len() + 1);
        assert_eq!(data[0]["id"], "qwen/qwen3-next-80b-a3b-instruct");
        assert_eq!(data[0]["curated"], true);
        assert_eq!(data[data.len() - 1]["id"], "some/other-model");
        assert_eq!(data[data.len() - 1]["curated"], false);
    }

    #[test]
    fn build_response_works_with_empty_upstream() {
        let resp = build_models_response(&[]);
        let data = resp.get("data").unwrap().as_array().unwrap();
        assert_eq!(data.len(), CURATED_MODELS.len());
    }
}
