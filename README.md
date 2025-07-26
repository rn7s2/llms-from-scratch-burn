# LLMs from scratch - Rust & Burn

[Burn: A next generation Deep Learning Framework](https://github.com/tracel-ai/burn).

Many thanks to [LLMs from scratch - Rust (candle)](https://github.com/nerdai/llms-from-scratch-rs).

## Backends

Burn & Wgpu should be compatible with any operating system and will use the GPU.

## Usage

Clone this repo, and use cargo to run examples `ch02` - `ch07`:

`cargo run --example ch02`

by default, `Wgpu` will be used. In some cases, `Wgpu` backend is known to generate wrong results. If that happens, you can use the `ndarray` feature to recheck the results:

`cargo run -F ndarray --example ch02`

## Note

for ch05, `gpt2-small-124M.safetensors` can be downloaded from [Hugging Face](https://huggingface.co/rasbt/gpt2-from-scratch-pytorch/tree/main)
