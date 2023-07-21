use serde::{Deserialize, Serialize};


#[derive(Deserialize, Serialize, Clone, Debug)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
    pub name: Option<String>,
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
    #[serde(default)]
    pub stream: bool,
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