pub mod engine;
mod api;

use std::{path::PathBuf, io::Write, time::{SystemTime, UNIX_EPOCH, Duration}, sync::{atomic::{AtomicBool, Ordering}, Arc}};

use anyhow::{Result, bail};
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
        .arg(
            arg!(
                -b --batch <NUM> "size of batch"
            )
            .default_value("256")
            .value_parser(value_parser!(usize))
        )
        .arg(
            arg!(
                --context <NUM> "number of context tokens"
            )
            .default_value("1024")
            .value_parser(value_parser!(usize))
        )
        .arg(
            arg!(
                --storage <path> "session path"
            )
            .value_parser(value_parser!(PathBuf))
        )
        .arg(
            arg!(
                --legecy "use legacy model prompt"
            )
        )
        .get_matches();

    let model_file = matches.get_one::<PathBuf>("model").unwrap();

    let storage_path = matches.get_one::<PathBuf>("storage");

    if let Some(sp) = &storage_path {
        if !sp.exists() {
            std::fs::create_dir_all(sp)?;
        } else if !sp.is_dir() {
            bail!("storage path must be a directory");
        }
    }

    let mut config: InferenceSessionConfig = Default::default();
    let context_size = matches.get_one::<usize>("context").unwrap();
    let legecy = matches.get_flag("legecy");

    if let Some(n_threads) = matches.get_one::<usize>("threads") {
        info!("using {} threads", n_threads);
        config.n_threads = *n_threads;
    }

    if let Some(n_batch) = matches.get_one::<usize>("batch") {
        info!("using {} batch size", n_batch);
        config.n_batch = *n_batch;
    }

    info!("using context size of {}", context_size);

    let model_params = ModelParameters {
        context_size: *context_size,
        use_gpu: true,
        ..Default::default()
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
            LoadProgress::ContextSize { bytes } => log::info!(
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

    // let s = model.start_session(config);

    let engine = Mutex::new(Engine::new(model, config, !legecy, storage_path.cloned()));

    // let test = true;

    // if test {
    //     let engine = engine.lock();

    //     let mut ps: Option<InferenceSession> = Some(s);

    //     let mut rl = rustyline::Editor::<(), DefaultHistory>::new()?;

    //     loop {
        
    //         ps = match rl.readline(">> ") {
    //             Ok(raw_line) => {
    //                 let (mut x0, _) = engine.chat(ps, &[ChatMessage {
    //                     role: "user".into(),
    //                     content: raw_line,
    //                     name: None,
    //                 }], 0.5, llm::conversation_inference_callback("asdflasdfljlj", print_token)).unwrap();
    //                 let s = String::from_utf8_lossy(x0.decoded_tokens());
    //                 println!("s:{}", s);


    //                 let s0 = unsafe { x0.get_snapshot() };

    //                 let mut v = Vec::new();
    //                 bincode::serialize_into(&mut v, &s0).unwrap();
    //                 println!("len:{}", v.len());

    //                 let s1 = bincode::deserialize_from(&v[..]).unwrap();
    //                 if s0.to_owned() == s1 {
    //                     println!("same");
    //                 }
    //                 let x1 = InferenceSession::from_snapshot(s1, engine.model.as_ref())?;

    //                 Some(x1)
    //             }
    //             Err(ReadlineError::Eof) | Err(ReadlineError::Interrupted) => {
    //                 break;
    //             }
    //             Err(err) => {
    //                 log::error!("{err}");
    //                 break;
    //             }
    //         };
    
    //     }
    //     // let (ps, _) = engine.chat(None, &[ChatMessage {
    //     //     role: "user".into(),
    //     //     content: "我在上海".into(),
    //     //     name: None,
    //     // }], 0.5, cb).unwrap();

    //     // let s = String::from_utf8_lossy(ps.decoded_tokens());
    //     // println!("s:{}", s);

    //     // let (ps, _) = engine.chat(Some(ps), &[ChatMessage {
    //     //     role: "user".into(),
    //     //     content: "我在什么国家？".into(),
    //     //     name: None,
    //     // }], 0.5, cb).unwrap();
    //     return Ok(());
    // }

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
            } else if request.url() == "/v1/models" {
                s.spawn(move |_| {
                    let json_data = handle_models(legecy);
                    let response = Response::from_string(json_data);
                    if let Err(x) = request.respond(response) {
                        log::error!("error responding to request: {}", x);
                    }
                });
            } else {
                // 处理其他请求（可选）
                // handle_other_requests(request);
            }
        }
    }).expect("scope failed");
    Ok(())
}

fn handle_models(legecy: bool) -> String {
    let ret = json!({
        "data": [
          {
            "id": if legecy { "llama" } else { "llama2" },
            "object": "model",
            "owned_by": "hs",
            "permission": []
          }
        ],
        "object": "list"
      });
    ret.to_string()
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

    crossbeam::scope(move |s| {

        match engine.try_lock_for(Duration::from_secs(60)) {
            Some(engine) => {
                let terminate = Arc::new(AtomicBool::new(false));

                let t = terminate.clone();
                // 建线程处理响应
                s.spawn(move |_| {
                    if req.stream {

                        let mut writer: Box<dyn Write + Send> = request.into_writer();
            
                        if let Err(x) = write_message_header(&mut writer, &HTTPVersion(1, 1), &StatusCode(200), &SSE_HEADERS) {
                            warn!("error while writing header: {}", x);
                            t.store(true, Ordering::Relaxed);
                            return;
                        }
            
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
                            if let Err(x) = writer.write_all(formatted_event.as_bytes()).and_then(|_| writer.flush()) {
                                warn!("error while writing event: {}", x);
                                t.store(true, Ordering::Relaxed);
                                break;
                            }
                        }
                        if !t.load(Ordering::Relaxed) {
                            if let Err(x) = writer.write_all("data: [DONE]\n\n".as_bytes()).and_then(|_| writer.flush()) {
                                warn!("error while writing event: {}", x);
                                t.store(true, Ordering::Relaxed);
                            }
                        }
                        writer.flush().unwrap();
                        drop(writer);
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
                let stats = engine.chat(None, &msgs, temperature, |r| {
                    match (terminate.load(Ordering::Relaxed), &r) {
                        (true, _) => {
                            warn!("terminated");
                            Ok(InferenceFeedback::Halt)
                        }
                        (_, InferenceResponse::EotToken) => {
                            warn!("eot");
                            Ok(InferenceFeedback::Halt)
                        }
                        (_, InferenceResponse::InferredToken(t)) => {
                            print!("{t}");
                            std::io::stdout().flush().unwrap();
            
                            tx.send(t.to_string()).unwrap();
                            Ok(InferenceFeedback::Continue)
                        }
                        _ => Ok(InferenceFeedback::Continue),
                    }
                });
                drop(engine);
                drop(tx);

                match stats {
                    Ok((_ps, stats)) => {
                        info!("stats: {:#?}", stats);
                        println!("{} tokens / {} ms, {:.2} tokens / s", stats.predict_tokens, stats.predict_duration.as_millis(), stats.predict_tokens as f64 * 1000.0 / stats.predict_duration.as_millis() as f64);
                    },
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

    writer.flush()?;

    Ok(())
}
