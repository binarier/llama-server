use std::{convert::Infallible, sync::Arc, fs::File, io::BufWriter, time::Instant, path::Path};

use anyhow::Result;
use llm::{Model, InferenceParameters, InferenceSessionConfig, InferenceStats, InferenceResponse, InferenceFeedback, samplers::TopPTopK, InferenceSession};
use log::info;
use rand::thread_rng;
use zstd::Encoder;

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

    pub fn chat(&self, messages: &[ChatMessage], temperature: f32, callback: impl FnMut(InferenceResponse) -> std::result::Result<InferenceFeedback, std::convert::Infallible>) -> Result<InferenceStats> {
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

        let start = Instant::now();
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

        write_session(session, Path::new("session.bin"))?;

        info!("snapshot took {:?} ms", start.elapsed().as_millis());

        let stats = res.map_err(|x| anyhow::anyhow!("{}", x))?;
        Ok(stats)
    }
}


pub fn write_session(mut session: InferenceSession, path: &Path) -> Result<()> {
    // SAFETY: the session is consumed here, so nothing else can access it.
    let snapshot = unsafe { session.get_snapshot() };
    let file = File::create(path)?;
    let encoder = Encoder::new(BufWriter::new(file), 1)?.auto_finish();
    bincode::serialize_into(encoder, &snapshot)?;
    log::info!("Successfully wrote session to {path:?}");
    Ok(())
}
