use std::{path::PathBuf, convert::Infallible, io::Write, sync::Mutex, time::{SystemTime, UNIX_EPOCH}};

use clap::{arg, value_parser};
use crossbeam::channel;
use llm::{TokenizerSource, ModelParameters, LoadProgress, ModelArchitecture, InferenceSessionConfig, InferenceParameters, InferenceResponse, InferenceFeedback, Model, InferenceStats};
use log::info;
use pretty_env_logger::env_logger;
use rand::thread_rng;
use serde::{Deserialize, Serialize};
use serde_json::json;
use tiny_http::{Server, Header, StatusCode, HTTPVersion};

lazy_static::lazy_static! {
    static ref ENGINE: Mutex<Option<Engine>> = Mutex::new(None);
    static ref SSE_HEADERS: Vec<Header> = vec![
        Header::from_bytes(&b"Content-Type"[..], &b"text/event-stream"[..]).unwrap(),
        Header::from_bytes(&b"Cache-Control"[..], &b"no-cache"[..]).unwrap(),
        Header::from_bytes(&b"Connection"[..], &b"keep-alive"[..]).unwrap(),
    ];
}

#[derive(Deserialize, Serialize, Clone, Debug)]
pub struct ChatMessage {
    role: String,
    content: String,
    name: Option<String>,
}

#[derive(Deserialize, Debug)]
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

fn handle_sse_request(mut request: tiny_http::Request) {

    let req: ChatCompletionRequest = serde_json::from_reader(request.as_reader()).unwrap();
    info!("msg:{:#?}", req);
    let msgs = req.messages;

    // 设置响应头

    let (tx, rx) = channel::unbounded::<String>();

    let ts = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
    let uuid = uuid::Uuid::new_v4().to_string();

    crossbeam::scope(|s| {

        s.spawn(move |_| {
            let engine = ENGINE.lock().unwrap();
            let engine = engine.as_ref().unwrap();
            engine.chat2(&msgs, |r| match &r {
                InferenceResponse::InferredToken(t) => {
                    print!("{t}");
                    std::io::stdout().flush().unwrap();
    
                    tx.send(t.to_string()).unwrap();
                    Ok(InferenceFeedback::Continue)
                }
                _ => Ok(InferenceFeedback::Continue),
            }).ok();

        });

        s.spawn(move |_| {
            let mut writer = request.into_writer();

            write_message_header(&mut writer, &HTTPVersion(1, 1), &StatusCode(200), &SSE_HEADERS).unwrap();

            writer.flush().unwrap();

            let mut first = true;
            for msg in rx {
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
                let formatted_event = format!("data: {}\n\n", s);
                writer.write_all(formatted_event.as_bytes()).unwrap();
                writer.flush().unwrap();
            }
            writer.write_all("data: [DONE]\n\n".as_bytes()).unwrap();
            writer.flush().unwrap();
        });
    }).expect("spawn failed");
}

fn main() {
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
        .arg(
            arg!(
                -t --threads <NUM> "number of threads"
            )
            .value_parser(value_parser!(usize))
        )
        .get_matches();

    let model_file = matches.get_one::<PathBuf>("model").unwrap();

    let parameters: InferenceParameters = Default::default();
    let mut config: InferenceSessionConfig = Default::default();

    if let Some(n_threads) = matches.get_one::<usize>("threads") {
        info!("using {} threads", n_threads);
        config.n_threads = *n_threads;
    }

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
        model,
        parameters,
        config,
    });

    let server = Server::http("0.0.0.0:8000").unwrap();
    println!("Server listening on port 8000...");

    loop {
        // 接收连接
        let request = match server.recv() {
            Ok(request) => request,
            Err(e) => {
                println!("Error while handling request: {}", e);
                continue;
            }
        };

        // 如果请求路径是 SSE 的端点，则处理 SSE 请求
        if request.url() == "/v1/chat/completions" {
            handle_sse_request(request);
        } else {
            // 处理其他请求（可选）
            // handle_other_requests(request);
        }
    }
}




pub struct Engine {
    model: Box<dyn Model>,
    parameters: InferenceParameters, 
    config: InferenceSessionConfig,
}

impl Engine {
    pub fn new(model: Box<dyn Model>, parameters: InferenceParameters, config: InferenceSessionConfig) -> Self {
        Self { 
            model, 
            parameters,
            config,
        }
    }

    pub fn chat(&self, messages: &[ChatMessage]) -> anyhow::Result<(String, InferenceStats)> {

        let mut session = self.model.start_session(self.config);
    
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

        let mut output = String::new();
        let res = session.infer::<Infallible>(
            self.model.as_ref(),
            &mut rng,
            &llm::InferenceRequest {
                prompt: prompt.as_str().into(),
                parameters: &self.parameters,
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
        let mut session = self.model.start_session(self.config);
    
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
    
        let res = session.infer::<Infallible>(
            self.model.as_ref(),
            &mut rng,
            &llm::InferenceRequest {
                prompt: prompt.as_str().into(),
                parameters: &self.parameters,
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

fn write_message_header<W>(
    mut writer: W,
    http_version: &HTTPVersion,
    status_code: &StatusCode,
    headers: &[Header],
) -> std::io::Result<()>
where
    W: Write,
{
    // writing status line
    write!(
        &mut writer,
        "HTTP/{}.{} {} {}\r\n",
        http_version.0,
        http_version.1,
        status_code.0,
        status_code.default_reason_phrase()
    )?;

    // writing headers
    for header in headers.iter() {
        writer.write_all(header.field.as_str().as_ref())?;
        write!(&mut writer, ": ")?;
        writer.write_all(header.value.as_str().as_ref())?;
        write!(&mut writer, "\r\n")?;
    }

    // separator between header and data
    write!(&mut writer, "\r\n")?;

    Ok(())
}
