use std::{convert::Infallible, sync::Arc, fs::File, io::{BufWriter, BufReader}, time::Instant, path::{Path, PathBuf}};

use anyhow::Result;
use llm::{Model, InferenceParameters, InferenceSessionConfig, InferenceStats, InferenceResponse, InferenceFeedback, samplers::TopPTopK, InferenceSession};
use log::info;
use rand::thread_rng;
use zstd::{Encoder, Decoder};

use crate::api::ChatMessage;

pub struct Engine {
    pub model: Box<dyn Model>,
    pub config: InferenceSessionConfig,
    llama2: bool,
    _storage_path: Option<PathBuf>,
}

impl Engine {
    pub fn new(model: Box<dyn Model>, config: InferenceSessionConfig, llama2: bool, storage_path: Option<PathBuf>) -> Self {
        Self { 
            model, 
            config,
            llama2,
            _storage_path: storage_path,
        }
    }

    pub fn chat(&self, persisted_session: Option<InferenceSession>, messages: &[ChatMessage], temperature: f32, callback: impl FnMut(InferenceResponse) -> std::result::Result<InferenceFeedback, std::convert::Infallible>) -> Result<(InferenceSession, InferenceStats)> {
        let mut rng = thread_rng();

        // let mut session = None;

        // // if let Some(cid) = &conv_id {
        // //     if let Some(path) = &self.storage_path {
        // //         let path = path.join(cid);
        // //         if path.exists() {
        // //             match read_session(self.model.as_ref(), &path) {
        // //                 Ok(s) => session = Some(s),
        // //                 Err(err) => {
        // //                     log::warn!("could not load session from {}: {}", path.display(), err);
        // //                 },
        // //             }
        // //         }
        // //     }
        // // }
        let has_session = persisted_session.is_some();

        let messages = if !has_session {
            messages
        } else {
            // 有session取最后条
            &messages[messages.len()-1..]
        };

        let mut session = persisted_session.unwrap_or_else(|| self.model.start_session(self.config));

        // let p = Path::new("/tmp/session");
        // write_session(session, p)?;
        // let mut session = read_session(self.model.as_ref(), p).unwrap();

        let mut prompt = String::new();


        if !self.llama2 {
            prompt += "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n";

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
        } else {
            // let mut system = "You are a helpful assistant. 你是一个乐于助人的助手。\n\n";
            let mut system = "You are a helpful assistant. 你是一个乐于助人的助手。请你提供专业、有逻辑、内容真实、有价值的详细回复。";
            let mut first = true;
            for message in messages {
                match message.role.as_str() {
                    "system" => {
                        if !message.content.is_empty() {
                            system = message.content.as_str();
                        }
                    },
                    "user" => {
                        if first && !has_session {
                            prompt += format!("[INST] <<SYS>>\n{}\n<</SYS>>\n\n{}[/INST]", system, message.content).as_str();
                            first = false;
                        } else {
                            prompt += format!("[INST]{}[/INST]", message.content).as_str();
                        }
                    },
                    "assistant" => {
                        prompt += message.content.as_str();
                    }
                    _ => {
                        panic!("unknown role");
                    }
                }
            }
        }

        let parameters = InferenceParameters {
            sampler: Arc::new(TopPTopK {
                temperature,
                ..
                Default::default()
            }),
        };

        info!("PROMPT:[{}] {}", has_session, prompt);
    
        let res = session.infer::<Infallible>(
            self.model.as_ref(),
            &mut rng,
            &llm::InferenceRequest {
                prompt: prompt.as_str().into(),
                parameters: &parameters,
                play_back_previous_tokens: has_session,
                maximum_token_count: None,
            },
            // OutputRequest
            &mut Default::default(),
            callback,
        );
        println!();

        // let start = Instant::now();
        // let snap = unsafe { session.get_snapshot() };
        // let mut f = File::create("snapshot.bin")?;

        // let mut t: RleVec<u8> = RleVec::new();
        // bincode::serialize_into(&mut t, &snap)?;
        // serde_json::to_writer(&f, &t)?;

        // let mut e = ZlibEncoder::new(Vec::new(), Compression::default());
        // bincode::serialize_into(&mut e, &snap)?;
        // let v = e.finish()?;
        // f.write_all(&v)?;

        // let mut v = Vec::new();
        // bincode::serialize_into(&mut v, &snap)?;
        // info!("snapshot size: {} bytes", v.len());
        // let r: Vec<Vec<u8>> = v.par_chunks(4 << 20)
        //     .map(|x| {
        //         let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
        //         encoder.write_all(x).unwrap();
        //         encoder.finish().unwrap()
        //     })
        //     .collect();
        // bincode::serialize_into(&mut f, &r)?;




        // drop(f);

        // if let Some(cid) = &conv_id {
        //     if let Some(path) = &self.storage_path {
        //         let wpath = path.join(cid);
        //         write_session(session, &wpath)?;
        //     } else {
        //         warn!("no storage path specified, session will not be saved");
        //     }
        // }

        // info!("snapshot took {:?} ms", start.elapsed().as_millis());

        let stats = res.map_err(|x| anyhow::anyhow!("{}", x))?;
        Ok((session, stats))
    }
}


pub fn write_session(mut session: InferenceSession, path: &Path) -> Result<()> {
    // SAFETY: the session is consumed here, so nothing else can access it.
    let start = Instant::now();
    let snapshot = unsafe { session.get_snapshot() };
    let file = File::create(path)?;
    let encoder = Encoder::new(BufWriter::new(file), 1)?.auto_finish();
    bincode::serialize_into(encoder, &snapshot)?;
    log::info!("Successfully wrote session to {path:?} in {} ms", start.elapsed().as_millis());
    Ok(())
}

pub fn read_session(model: &dyn Model, path: &Path) -> Result<InferenceSession> {
    let start = Instant::now();
    let file = File::open(path)?;
    let decoder = Decoder::new(BufReader::new(file))?;
    let snapshot = bincode::deserialize_from(decoder)?;
    let session = InferenceSession::from_snapshot(snapshot, model)?;
    log::info!("Loaded inference session from {path:?} in {} ms", start.elapsed().as_millis());
    Ok(session)
}
