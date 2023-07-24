mod engine;
mod api;

use std::{path::PathBuf, io::Write, time::{SystemTime, UNIX_EPOCH, Duration}};

use anyhow::Result;
use clap::{arg, value_parser};
use crossbeam::channel;
use llm::{TokenizerSource, ModelParameters, LoadProgress, InferenceSessionConfig, InferenceResponse, InferenceFeedback, ModelArchitecture};
use log::{info, warn};
use parking_lot::Mutex;
use pretty_env_logger::env_logger;
use serde_json::json;
use tiny_http::{Server, Header, StatusCode, HTTPVersion, Response, Request};

use crate::{api::ChatCompletionRequest, engine::Engine};

lazy_static::lazy_static! {
    static ref SSE_HEADERS: Vec<Header> = vec![
        Header::from_bytes(&b"Content-Type"[..], &b"text/event-stream"[..]).unwrap(),
        Header::from_bytes(&b"Cache-Control"[..], &b"no-cache"[..]).unwrap(),
        Header::from_bytes(&b"Connection"[..], &b"keep-alive"[..]).unwrap(),
    ];
    static ref JSON_HEADER: Header = Header::from_bytes(&b"Content-Type"[..], &b"application/json"[..]).unwrap();
}

fn main() -> Result<()> {
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

    let mut config: InferenceSessionConfig = Default::default();

    if let Some(n_threads) = matches.get_one::<usize>("threads") {
        info!("using {} threads", n_threads);
        config.n_threads = *n_threads;
    }

    let model_params = ModelParameters {
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
        model_params,
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
    )?;

    let engine = Mutex::new(Engine::new(model, config));

    let server = Server::http("0.0.0.0:8000").unwrap();
    println!("Server listening on port 8000...");

    crossbeam::scope(|s| {
        loop {
            // 接收连接
            let request = match server.recv() {
                Ok(request) => request,
                Err(e) => {
                    warn!("Error while handling request: {}", e);
                    continue;
                }
            };

            if request.url() == "/v1/chat/completions" {
                s.spawn(|_| {
                    handle_chat_completions(request, &engine);
                });
            } else {
                // 处理其他请求（可选）
                // handle_other_requests(request);
            }
        }
    }).expect("scope failed");
    Ok(())
}

fn handle_chat_completions(mut request: Request, engine: &Mutex<Engine>) {
    match serde_json::from_reader::<_, ChatCompletionRequest>(request.as_reader()) {
        Ok(ccr) => {
            handle_stream_request(ccr, request, engine).ok();
        },
        Err(x) => {
            log::error!("error: {}", x);
            request.respond(
                Response::from_string(x.to_string()).with_status_code(400)).ok();
        },
    }
}

fn handle_stream_request(req: ChatCompletionRequest, request: Request, engine: &Mutex<Engine>) -> Result<()> {

    let msgs = req.messages;

    let (tx, rx) = channel::unbounded::<String>();
    let ts = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
    let uuid = uuid::Uuid::new_v4().to_string();
    let temperature = req.temperature.unwrap_or(1f32);

    crossbeam::scope(|s| {

        match engine.try_lock_for(Duration::from_secs(60)) {
            Some(engine) => {
                // 建线程处理响应
                s.spawn(|_| {
                    if req.stream {

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
                    } else {
                        let txt = rx.into_iter().collect::<Vec<String>>().join("");
                        let json = json!({
                            "id": "uuid",
                            "object": "chat.completion",
                            "created": ts,
                            "choices": [{
                                "index": 0,
                                "message": {
                                "role": "assistant",
                                "content": txt,
                                },
                                "finish_reason": "stop"
                            }],
                            "usage": {
                                "prompt_tokens": 0,
                                "completion_tokens": 0,
                                "total_tokens": 0
                            }
                        });
            
                        let response = Response::from_string(serde_json::to_string(&json).expect("invalid json"))
                            .with_status_code(200)
                            .with_header(JSON_HEADER.clone());
                        request.respond(response).ok();
                    } 
                });

                // 主线程跑llm
                let stats = engine.chat(&msgs, temperature, |r| match &r {
                        InferenceResponse::InferredToken(t) => {
                            print!("{t}");
                            std::io::stdout().flush().unwrap();
            
                            tx.send(t.to_string()).unwrap();
                            Ok(InferenceFeedback::Continue)
                        }
                        _ => Ok(InferenceFeedback::Continue),
                });
                drop(engine);
                drop(tx);

                match stats {
                    Ok(stats) => info!("stats: {:#?}", stats),
                    Err(x) => warn!("infer failed: {}", x),
                }
            },
            None => {
                warn!("engine is busy");
                request.respond(
                    Response::from_string("engine is busy").with_status_code(503)).ok();
            },
        }
    }).map_err(|_| anyhow::anyhow!("scope failed"))?;
    Ok(())
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
