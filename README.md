# Llama Server

基于Rust的Llama Server，可以提供基本的类OpenAI的API服务，以及一些开发中的扩展功能。

项目开发了一个API层，调用Rust实现的LLM引擎 https://github.com/rustformers/llm ，该引擎使用了 [llama.cpp](https://github.com/ggerganov/llama.cpp) 项目的底层ggml实现，用于模型的数据结构管理及张量的运算，以及对显卡的支持，然后在 tokenizer及sampler等方向都是纯rust实现，并且项目还在快速更新中，所以这个项目使用submodule方式引入这些依赖项，以便于快速跟进更新。

显卡支持方面，目前实现了cublas的支持，并且可以支持把权重数据分布在多显卡上，以得到更大的显存。但在多显卡运行时，由于总线带宽原因，推理速度会下降。

因为显卡占用的关系，推理引擎目前是独占运行的，所以如果有并发请求上来，会直接返回503。

OpenAI的API参考[这里](https://platform.openai.com/docs/api-reference)，目前只实现了 model和chat接口，后续会有更多的接口实现。

模型目前只支持ggml格式，模型的获取参考 https://github.com/ymcui/Chinese-LLaMA-Alpaca-2 ，格式转换参考 https://github.com/ymcui/Chinese-LLaMA-Alpaca-2/wiki/manual_conversion_zh

性能方面，目前13B的llama2模型，使用q8_0量化（显存大小原因，f16放不下），在4090上可以达到130 token/s。

## 依赖项及编译方法

* 需要安装 rust 1.70 或以上版本
* 使用 cublas 库支持，如果打开cublas feature，需要安装好相应的cuda开发库
* 或者使用 docker 编译运行镜像，见 Dockerfile

```
$ sudo apt install pkg-config libssl-dev
```

```
$ git submodule update --init --recursive
$ cargo build --release --features cublas
```


## 运行方法

使用 Nvidia 运行：

```
cargo run --release --features cublas -- -m <path_to_ggml_model> --context 1024
```

使用 CPU 运行：

```
cargo run --release -- -m <path_to_ggml_model> --context 1024 -t 32
```

context 是上下文大小，即一次聊天包含prompt在内的最多token数。目前llama2模型支持4096上下文，更大的上下文使用NTK方法实现，会在后续版本加入支持。

## 演示环境

访问 https://chat.llamax.site 可以看到演示环境，可以回答一些简单的问题。

目前演示环境使用的是  [Chinese-LLaMA-Alpaca-2 v2.0版本](https://github.com/ymcui/Chinese-LLaMA-Alpaca-2) 提供的中文化的llama2的对话模型，使用了13b参数的q5k的量化版本。

演示界面使用了 https://github.com/mckaywrigley/chatbot-ui 项目提供的界面，通过OpenAI标准接口访问后端服务。

由于演示环境运行在测试用的的国产显卡上，响应速度大约为 6 token/s，用起来会有些慢，但不影响功能。

演示环境有k8s集群，所以可以支持一定的并发用户数，不用担心503的问题。

## 待开发功能

### session快照功能。

由于OpenAI的API中，每次聊天都会把完整上下文传回，所以需要把完整上下文当作prompt一起送入引擎进行初始推理，在显卡性能有限的环境下，这会导致对话越来越慢，因为每次需要初始推荐的token越来越多。

所以考虑通过一个比如redis数据库记录每个对话对应的session id，然后在每次请求时，如果对话已经推理过，则直接从redis里取出之前的上下文数据，只需要把新的prompt加入到上下文中，然后继续推理即可。

在1024 context size下，上下文数据大约100M左右，但因为是稀疏数据，可以压缩到几百k，所以可以实现session的缓存。