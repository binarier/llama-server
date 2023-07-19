# OpenAI API Server

## 系统依赖项
```
$ sudo apt install pkg-config libssl-dev
```

## 编译
```
$ git submodule update --init --recursive
$ cargo build --release --features cublas
```

## CI/CD

使用AutoDevOps的话，需要设置变量 GIT_SUBMODULE_STRATEGY = recursive

## 运行

docker

```
docker run --runtime=nvidia --rm -v (model_path):/models -p 8000:8000 git.highsharp.com:5050/opensource/openai-server/master:latest 
```
