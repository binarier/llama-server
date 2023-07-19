ARG CUDA_VERSION=12.2.0
ARG UBUNTU_DIST=20.04

FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_DIST} as builder

ARG RUST_TOOLCHAIN=1.70.0
ARG APT_PROXY=http://192.168.24.2:8000/

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV http_proxy=http://192.168.3.180:7779
ENV https_proxy=http://192.168.3.180:7779

RUN echo Acquire::http::Proxy \"$APT_PROXY\"\; > /etc/apt/apt.conf.d/00-aptproxy
RUN apt-get update
RUN apt-get install -y curl pkg-config libssl-dev
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs -o /tmp/setup.sh && sh /tmp/setup.sh -y --default-toolchain ${RUST_TOOLCHAIN}

WORKDIR /build
COPY . .
RUN . $HOME/.cargo/env && cargo build --release --features cublas

FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_DIST}
WORKDIR /app
COPY --from=builder /build/target/release/openai-server /app
COPY --from=builder /build/Rocket.toml /app

CMD [ "/app/openai-server", "-m", "/models/ggml-model-f16.bin" ]
