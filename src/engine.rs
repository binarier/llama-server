use std::{convert::Infallible, sync::Arc};

use llm::{Model, InferenceParameters, InferenceSessionConfig, InferenceStats, InferenceResponse, InferenceFeedback, samplers::TopPTopK};
use rand::thread_rng;

use crate::api::ChatMessage;

pub struct Engine {
    model: Box<dyn Model>,
    config: InferenceSessionConfig,
}

impl Engine {
    pub fn new(model: Box<dyn Model>, config: InferenceSessionConfig) -> Self {
        Self { 
            model, 
            config,
        }
    }

    pub fn chat(&self, messages: &[ChatMessage], temperature: f32, callback: impl FnMut(InferenceResponse) -> std::result::Result<InferenceFeedback, std::convert::Infallible>) -> anyhow::Result<InferenceStats> {
        let mut session = self.model.start_session(self.config);
    
        let mut rng = thread_rng();
    
        let init_prompt = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n";
        let mut prompt = init_prompt.to_string();

        for message in messages {
            match message.role.as_str() {
                "user" | "system" => {
                    prompt += format!("### Instruction:\n\n{}\n", message.content).as_str();
                },
                "assistant" => {
                    prompt += format!("### Response:\n\n{}\n", message.content).as_str();
                }
                _ => {
                    panic!("unknown role");
                }
            }
        }
        prompt += "### Response:\n\n";

        // println!("prompt:{}", prompt);
        let parameters = InferenceParameters {
            sampler: Arc::new(TopPTopK {
                temperature,
                ..
                Default::default()
            }),
        };
    
        let res = session.infer::<Infallible>(
            self.model.as_ref(),
            &mut rng,
            &llm::InferenceRequest {
                prompt: prompt.as_str().into(),
                parameters: &parameters,
                play_back_previous_tokens: false,
                maximum_token_count: None,
            },
            // OutputRequest
            &mut Default::default(),
            callback,
        );
        println!();
        let stats = res.map_err(|x| anyhow::anyhow!("{}", x))?;
        Ok(stats)
    }
}
