use std::{convert::Infallible, fs::File, io::{BufWriter, BufReader}, time::Instant, path::{Path, PathBuf}, str::FromStr, sync::{Mutex, Arc}};

use anyhow::{Result, Context, bail};
use llm::{Model, InferenceParameters, InferenceSessionConfig, InferenceStats, InferenceResponse, InferenceFeedback, samplers::ConfiguredSamplers, InferenceSession};
use log::info;
use mysql::{Pool, prelude::*};
use rand::thread_rng;
use regex::Regex;
use zstd::{Encoder, Decoder};

use crate::api::ChatMessage;

pub struct Engine {
    pub model: Box<dyn Model>,
    pub config: InferenceSessionConfig,
    llama2: bool,
    _storage_path: Option<PathBuf>,
    admindb: Option<Pool>,
}

impl Engine {
    pub fn new(model: Box<dyn Model>, config: InferenceSessionConfig, llama2: bool, storage_path: Option<PathBuf>, admindb_url: Option<String>) -> Result<Self> {
        let admindb = if let Some(url) = admindb_url {
            let re = Regex::new(r"^mysql://.*:.*@.*:.*/.*$")?;
            if !re.is_match(&url) {
                bail!("invalid db url");
            }
            info!("connecting to db {}", hide_password(&url));
            Some(Pool::new(url.as_str()).context("连接管理数据库")?)
        } else {
            None
        };

        Ok(Self { 
            model, 
            config,
            llama2,
            _storage_path: storage_path,
            admindb,
        })
    }

    pub fn chat(&self, persisted_session: Option<InferenceSession>, messages: &[ChatMessage], temperature: f32, mut callback: impl FnMut(InferenceResponse) -> std::result::Result<InferenceFeedback, std::convert::Infallible>) -> Result<(InferenceSession, InferenceStats)> {
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

        let session_id = uuid::Uuid::new_v4().to_string();

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

        // let parameters = InferenceParameters {
        //     sampler: Arc::new(TopPTopK {
        //         temperature,
        //         top_p: 0.9,
        //         top_k: 40,
        //         repeat_penalty: 1.1,
        //         ..
        //         Default::default()
        //     }),
        // };

        let sampler_opts = format!("topp:p=0.9/topk:k=40/temperature:{}", temperature);
        let samplers =  ConfiguredSamplers::from_str(&sampler_opts)?;
        let parameters = InferenceParameters {
            sampler: Arc::new(Mutex::new(samplers.builder.into_chain()))
        };

        info!("PROMPT:[{}] {}", has_session, prompt);

        if let Err(x) = admindb_insert(&self.admindb, &session_id, &prompt, &sampler_opts) {
            log::warn!("admindb insert error: {}", x);
        }

        let mut infer_tokens = String::new();

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
            |r| {
                if let InferenceResponse::InferredToken(t) = &r {
                    infer_tokens += t.as_str();
                }
                callback(r)
            }
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

        if let Err(x) = admindb_update(&self.admindb, &session_id, &infer_tokens, &stats) {
            log::warn!("admindb update error: {}", x);
        }

        Ok((session, stats))
    }
}


fn admindb_insert(db: &Option<Pool>, session_id: &String, input_tokens: &String, sampler_opts: &String) -> Result<()> {
    if let Some(db) = db {
        let mut conn = db.get_conn()?;
        conn.exec_drop("INSERT INTO OA_CHAT (SESSION_ID, INPUT_TOKENS, SAMPLER_OPTS, LAST_UPDATE) VALUES (?, ?, ?, NOW())", (session_id, input_tokens, sampler_opts))?;
    }

    Ok(())
}

fn admindb_update(db: &Option<Pool>, session_id: &String, infer_tokens: &String, stats: &InferenceStats) -> Result<()> {
    if let Some(db) = db {
        let mut conn = db.get_conn()?;
        conn.exec_drop("UPDATE OA_CHAT SET INFER_TOKENS = ?, STATS_PROMPT_DURATION_MS = ?, STATS_PROMPT_TOKENS_PER_SECOND=?, STATS_INFER_DURATION_MS = ?, STATS_INFER_TOKENS_PER_SECOND = ?, LAST_UPDATE = NOW() WHERE SESSION_ID = ?", 
            (infer_tokens, 
                stats.feed_prompt_duration.as_millis(), stats.prompt_tokens as f64 / stats.feed_prompt_duration.as_millis() as f64 * 1000f64,
                stats.predict_duration.as_millis(), stats.predict_tokens as f64 / stats.predict_duration.as_millis() as f64 * 1000f64,
                session_id
            ))?;
    }

    Ok(())
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

fn hide_password(mysql_conn_str: &str) -> String {
    let end = mysql_conn_str.find('@').unwrap_or(mysql_conn_str.len());
    let start = mysql_conn_str[..end].rfind(':').unwrap_or(0) + 1;
    let mut masked_conn_str = mysql_conn_str.to_string();
    masked_conn_str.replace_range(start..end, "****");
    masked_conn_str
}