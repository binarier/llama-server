use std::{path::PathBuf, convert::Infallible, io::Write, sync::Mutex, time::{SystemTime, UNIX_EPOCH}};

use clap::{arg, value_parser};
use llm::{TokenizerSource, ModelParameters, LoadProgress, ModelArchitecture, InferenceSessionConfig, InferenceParameters, InferenceResponse, InferenceFeedback, Model, InferenceStats};
use pretty_env_logger::env_logger;
use rand::thread_rng;
use rocket::response::stream::{EventStream, Event};
use rocket_cors::CorsOptions;
use serde::{Deserialize, Serialize};
use rocket::serde::json::Json;
use serde_json::json;

#[macro_use] extern crate rocket;

lazy_static::lazy_static! {
    static ref ENGINE: Mutex<Option<Engine>> = Mutex::new(None);
}

#[derive(Deserialize, Serialize, Clone, Debug)]
pub struct ChatMessage {
    role: String,
    content: String,
    name: Option<String>,
}

#[derive(Deserialize)]
pub struct ChatCompletionRequest {
    pub model: Option<String>,
    pub messages: Vec<ChatMessage>,
    pub temperature: Option<f32>,
    // top_p: Optional[float] = 1.0
    // top_k: Optional[int] = 40
    // n: Optional[int] = 1
    // max_tokens: Optional[int] = 128
    // num_beams: Optional[int] = 4
    // stop: Optional[Union[str, List[str]]] = None
    pub stream: Option<bool>,
    // repetition_penalty: Optional[float] = 1.0
    // user: Optional[str] = None
}


// {
//     "id": "chatcmpl-123",
//     "object": "chat.completion",
//     "created": 1677652288,
//     "choices": [{
//       "index": 0,
//       "message": {
//         "role": "assistant",
//         "content": "\n\nHello there, how may I assist you today?",
//       },
//       "finish_reason": "stop"
//     }],
//     "usage": {
//       "prompt_tokens": 9,
//       "completion_tokens": 12,
//       "total_tokens": 21
//     }
//   }
  
#[derive(Serialize)]
pub struct ChatChoice {
    pub index: usize,
    pub message: ChatMessage,
    pub finish_reason: Option<String>,
}

#[derive(Serialize)]
pub struct ChatUsage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

#[derive(Serialize)]
pub struct ChatResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub choices: Vec<ChatChoice>,
    pub usage: Option<ChatUsage>,
}


#[post("/v1/chat/completions", format = "application/json", data = "<request>")]    // std::result::Result<Json<ChatResponse>, Status>
pub fn create_chat_completion(request: Json<ChatCompletionRequest>) -> EventStream![] {
    let req = request.messages.clone();
    let (tx, mut rx) = tokio::sync::mpsc::channel::<String>(1);
    let ts = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
    let uuid = uuid::Uuid::new_v4().to_string();

    info!("msg:{:?}", req);

    std::thread::spawn(move || {
        let engine = ENGINE.lock().unwrap();
        let engine = engine.as_ref().unwrap();
        engine.chat2(&req, |r| match &r {
            InferenceResponse::InferredToken(t) => {
                print!("{t}");
                std::io::stdout().flush().unwrap();

                tx.blocking_send(t.to_string()).unwrap();
                Ok(InferenceFeedback::Continue)
            }
            _ => Ok(InferenceFeedback::Continue),
        }).ok();
    });

    //    {"id":"chatcmpl-7c7MaL1Pc0XPGGAPAzLL0ds61jGjQ","object":"chat.completion.chunk","created":1689319124,"model":"gpt-4-0613","choices":[{"index":0,"delta":{"content":"！"},"finish_reason":null}]}
    EventStream! {
        let mut first = true;
        while let Some(msg) = rx.recv().await {

            let mut x = json!({
                "id" : uuid,
                "object": "chat.completion.chunk",
                "create": ts,
                "choices" : [
                    {
                        "index" : 0,
                        "delta" : {
                            "content": msg
                        }
                    }
                ]
            });

            if first {
                first = false;
                x["choices"][0]["delta"]["role"] = json!("assistant");
            }
        
            let s = serde_json::to_string(&x).unwrap();

            yield Event::data(s);
        }
        log::info!("[{}] Done", uuid);
        yield Event::data("[DONE]");
    }

    // } else {
    //     match engine.chat(&request.messages) {
    //         Ok((response, stats)) => {
    //             let resp = ChatResponse {
    //                 id: "chatcmpl-123".to_string(),
    //                 object: "chat.completion".to_string(),
    //                 created: 1677652288,
    //                 choices: vec![ChatChoice {
    //                     index: 0,
    //                     message: ChatMessage {
    //                         role: "assistant".to_string(),
    //                         content: response,
    //                         name: None,
    //                     },
    //                     finish_reason: "stop".to_string(),
    //                 }],
    //                 usage: ChatUsage {
    //                     prompt_tokens: stats.prompt_tokens,
    //                     completion_tokens: stats.predict_tokens,
    //                     total_tokens: 0,
    //                 },
    //             };
    //             Ok(Json(resp))
    //         },
    //         Err(e) => {
    //             log::error!("Error: {}", e);
    //             Err(Status::InternalServerError)
    //         }
    //     }
    // }
}

#[launch]
fn rocket() -> _ {
    env_logger::builder()
        .filter_level(log::LevelFilter::Info)
        .parse_default_env()
        .init();

    let matches = clap::command!() // requires `cargo` feature
        .arg(
            arg!(
                -m --model <FILE> "model file"
            )
            .required(true)
            .value_parser(value_parser!(PathBuf)),
        )
        .get_matches();

    let model_file = matches.get_one::<PathBuf>("model").unwrap();

    // let model_file = Path::new("../7B_ggml/ggml-model-f16.bin");
    // let model_file = Path::new("../LLaMA/33B_merged-1/ggml-model-f16.bin");
    // let tk = Path::new("../7B_ggml/tokenizer.model");
    let params = ModelParameters {
        prefer_mmap: true,
        context_size: 512,
        lora_adapters: None,
        use_gpu: true,
        gpu_layers: None,
    };
    let mut sp = Some(spinoff::Spinner::new(
        spinoff::spinners::Dots2,
        "Loading model...",
        None,
    ));
    let now = std::time::Instant::now();
    let mut prev_load_time = now;

    let model = llm::load_dynamic(
        Some(ModelArchitecture::Llama),
        model_file,
        TokenizerSource::Embedded,
        params,
        |progress| match progress {
            LoadProgress::HyperparametersLoaded => {
                if let Some(sp) = sp.as_mut() {
                    sp.update_text("Loaded hyperparameters")
                };
            }
            LoadProgress::ContextSize { bytes } => log::debug!(
                "ggml ctx size = {}",
                bytesize::to_string(bytes as u64, false)
            ),
            LoadProgress::LoraApplied { name, source } => {
                if let Some(sp) = sp.as_mut() {
                    sp.update_text(format!(
                        "Patched tensor {} via LoRA from '{}'",
                        name,
                        source.file_name().unwrap().to_str().unwrap()
                    ));
                }
            }
            LoadProgress::TensorLoaded {
                current_tensor,
                tensor_count,
                ..
            } => {
                if prev_load_time.elapsed().as_millis() > 500 {
                    // We don't want to re-render this on every message, as that causes the
                    // spinner to constantly reset and not look like it's spinning (and
                    // it's obviously wasteful).
                    if let Some(sp) = sp.as_mut() {
                        sp.update_text(format!(
                            "Loaded tensor {}/{tensor_count}",
                            current_tensor + 1,
                        ));
                    };
                    prev_load_time = std::time::Instant::now();
                }
            }
            LoadProgress::Loaded {
                file_size,
                tensor_count,
            } => {
                if let Some(sp) = sp.take() {
                    sp.success(&format!(
                        "Loaded {tensor_count} tensors ({}) after {}ms",
                        bytesize::to_string(file_size, false),
                        now.elapsed().as_millis()
                    ));
                };
            }
        },
    );


    let model = model.unwrap();

    *ENGINE.lock().unwrap() = Some(Engine {
        model
    });

    rocket::build()
        .mount("/", routes![create_chat_completion])
        .attach(CorsOptions::default().to_cors().unwrap())
        // .manage(Engine {
        //     model,
        // })
}


pub struct Engine {
    model: Box<dyn Model>,
}

impl Engine {
    pub fn new(model: Box<dyn Model>) -> Self {
        Self { model }
    }

    pub fn chat(&self, messages: &[ChatMessage]) -> anyhow::Result<(String, InferenceStats)> {
        let config = InferenceSessionConfig::default();

        let mut session = self.model.start_session(config);
    
        let mut rng = thread_rng();
    
        let init_prompt = include_str!("alpaca-initial.txt");
        let mut prompt = init_prompt.to_string();

        for message in messages {
            match message.role.as_str() {
                "user" => {
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
            // n_threads: 32,
            ..
            Default::default()
        };
    
        let mut output = String::new();
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
            |r| match &r {
                InferenceResponse::InferredToken(t) => {
                    print!("{t}");
                    std::io::stdout().flush().unwrap();

                    output.push_str(t);
    
                    Ok(InferenceFeedback::Continue)
                }
                _ => Ok(InferenceFeedback::Continue),
            },
        );
        println!();
        let stats = res.map_err(|x| anyhow::anyhow!("{}", x))?;
        Ok((output, stats))
    }

    pub fn chat2(&self, messages: &[ChatMessage], callback: impl FnMut(InferenceResponse) -> std::result::Result<InferenceFeedback, std::convert::Infallible>) -> anyhow::Result<InferenceStats> {
        let config = InferenceSessionConfig { 
            n_threads: 64,
            ..Default::default()
        };

        let mut session = self.model.start_session(config);
    
        let mut rng = thread_rng();
    
        let init_prompt = include_str!("alpaca-initial.txt");
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
            // n_threads: 32,
            ..
            Default::default()
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