ARG CUDA_VERSION=12.2.0
ARG UBUNTU_DIST=20.04

FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_DIST} as builder

ARG RUST_TOOLCHAIN=1.70.0

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8

#视情况修改
ENV http_proxy=http://192.168.3.180:7779
ENV https_proxy=http://192.168.3.180:7779

RUN apt-get update
RUN apt-get install -y curl pkg-config libssl-dev
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs -o /tmp/setup.sh && sh /tmp/setup.sh -y --default-toolchain ${RUST_TOOLCHAIN}

WORKDIR /build
COPY . .
RUN . $HOME/.cargo/env && cargo build --release --features cublas

FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_DIST}
WORKDIR /app
COPY --from=builder /build/target/release/llama-server /app

ENV MODEL_PATH=/models/ggml-model-f16.bin

ENV SERVER_OPTS=""

CMD /app/llama-server -m MODEL_PATH ${SERVER_OPTS}
