use std::f64::consts::PI;

use anyhow::Result;
use burn::config::Config;
use burn::module::{Module, Param};
use burn::nn::loss::CrossEntropyLossConfig;
use burn::nn::{Dropout, DropoutConfig, Embedding, EmbeddingConfig, Linear, LinearConfig};
use burn::tensor::activation::softmax;
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::{DType, Int, TensorData};
use burn::tensor::{Tensor, backend::Backend};
use burn::train::{ClassificationOutput, TrainOutput, TrainStep, ValidStep};
use rand::Rng;
use rand::distr::weighted::WeightedIndex;
use safetensors::SafeTensors;

use crate::attention::{MultiHeadAttention, MultiHeadAttentionConfig};
use crate::dataset::GPTDatasetV1Batch;
use crate::tokenizer::{self, ITokenizer};

#[derive(Module, Debug)]
pub struct GPTModel<B: Backend> {
    vocab_size: usize,
    context_length: usize,
    tok_emb: Embedding<B>,
    pos_emb: Embedding<B>,
    drop_emb: Dropout,
    trf_blocks: Vec<TransformerBlock<B>>,
    final_norm: LayerNorm<B>,
    out_head: Linear<B>,
}

impl<B: Backend> GPTModel<B> {
    pub fn forward(&self, in_idx: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let device = in_idx.device();
        let dims = in_idx.dims();
        let (_batch_size, seq_len) = (dims[0], dims[1] as i64);

        let tok_embeds = self.tok_emb.forward(in_idx);
        let pos_embeds = self
            .pos_emb
            .forward(Tensor::arange(0..seq_len, &device).unsqueeze());

        let x = tok_embeds + pos_embeds;
        let mut x = self.drop_emb.forward(x);
        for trf_block in self.trf_blocks.iter() {
            x = trf_block.forward(x);
        }
        let x = self.final_norm.forward(x);
        self.out_head.forward(x)
    }

    fn forward_train(
        &self,
        input_ids: Tensor<B, 2, Int>,
        target_ids: Tensor<B, 2, Int>,
    ) -> ClassificationOutput<B> {
        let logits = self.forward(input_ids);
        let targets = target_ids;

        let device = &logits.device();

        let logits_flat = logits.flatten::<2>(0, 1);
        let targets_flat = targets.flatten::<1>(0, 1);

        let loss = CrossEntropyLossConfig::new()
            .init(device)
            .forward(logits_flat.clone(), targets_flat.clone());
        ClassificationOutput::new(loss, logits_flat, targets_flat)
    }
}

impl<B: AutodiffBackend> TrainStep<GPTDatasetV1Batch<B>, ClassificationOutput<B>> for GPTModel<B> {
    fn step(&self, batch: GPTDatasetV1Batch<B>) -> TrainOutput<ClassificationOutput<B>> {
        let item = self.forward_train(batch.input_ids, batch.target_ids);

        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> ValidStep<GPTDatasetV1Batch<B>, ClassificationOutput<B>> for GPTModel<B> {
    fn step(&self, batch: GPTDatasetV1Batch<B>) -> ClassificationOutput<B> {
        let start_context = "Every effort moves you";
        let token_ids = generate_text(
            self,
            text_to_token_ids(start_context, &crate::tokenizer::TOKENIZER),
            25,
            self.context_length,
            1.4,
            Some(50),
        );
        println!(
            "Output preview: {:?}",
            token_ids_to_text(token_ids, &crate::tokenizer::TOKENIZER)
        );

        self.forward_train(batch.input_ids, batch.target_ids)
    }
}

#[derive(Config, Debug)]
pub struct GPTModelConfig {
    pub vocab_size: usize,
    pub context_length: usize,
    pub emb_dim: usize,
    pub n_heads: usize,
    pub n_layers: usize,
    pub drop_rate: f64,
    pub qkv_bias: bool,
}

impl GPTModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> GPTModel<B> {
        GPTModel {
            vocab_size: self.vocab_size,
            context_length: self.context_length,
            tok_emb: EmbeddingConfig::new(self.vocab_size, self.emb_dim).init(device),
            pos_emb: EmbeddingConfig::new(self.vocab_size, self.emb_dim).init(device),
            drop_emb: DropoutConfig::new(self.drop_rate).init(),
            trf_blocks: (0..self.n_layers)
                .map(|_| {
                    TransformerBlockConfig::new(
                        self.context_length,
                        self.emb_dim,
                        self.n_heads,
                        self.drop_rate,
                        self.qkv_bias,
                    )
                    .init(device)
                })
                .collect(),
            final_norm: LayerNormConfig::new(self.emb_dim).init(device),
            out_head: LinearConfig::new(self.emb_dim, self.vocab_size)
                .with_bias(false)
                .init(device),
        }
    }

    pub fn init_pretrained<B: Backend>(&self, pth_path: &str, device: &B::Device) -> GPTModel<B> {
        let buffer = std::fs::read(pth_path).unwrap();
        let tensors = SafeTensors::deserialize(&buffer).expect("Failed to deserialize tensors");

        fn tensor_data(tensors: &SafeTensors<'_>, name: &str) -> TensorData {
            let tensor = tensors.tensor(name).unwrap();
            let bytes = tensor.data();
            let mut shape = tensor.shape().to_vec();
            let dtype = tensor.dtype();

            if name.ends_with(".weight") && !name.contains("_emb") {
                shape.reverse();
            }

            println!("load tensor: {} -> {:?}", name, shape);

            TensorData::from_bytes(
                bytes.to_vec(),
                shape,
                match dtype {
                    safetensors::Dtype::BOOL => DType::Bool,
                    safetensors::Dtype::F32 => DType::F32,
                    safetensors::Dtype::F64 => DType::F64,
                    safetensors::Dtype::I32 => DType::I32,
                    safetensors::Dtype::I64 => DType::I64,
                    _ => panic!("Unsupported dtype"),
                },
            )
        }

        let tok_emb = Embedding {
            weight: Param::from_data(tensor_data(&tensors, "tok_emb.weight"), device),
        };
        let pos_emb = Embedding {
            weight: Param::from_data(tensor_data(&tensors, "pos_emb.weight"), device),
        };
        let drop_emb = DropoutConfig::new(self.drop_rate).init();

        let trf_blocks = (0..self.n_layers)
            .map(|i| {
                let attn = MultiHeadAttention {
                    d_out: self.emb_dim,
                    num_heads: self.n_heads,
                    head_dim: self.emb_dim / self.n_heads,
                    w_query: Linear {
                        weight: Param::from_data(
                            tensor_data(&tensors, &format!("trf_blocks.{}.att.W_query.weight", i)),
                            device,
                        ),
                        bias: Some(Param::from_data(
                            tensor_data(&tensors, &format!("trf_blocks.{}.att.W_query.bias", i)),
                            device,
                        )),
                    },
                    w_key: Linear {
                        weight: Param::from_data(
                            tensor_data(&tensors, &format!("trf_blocks.{}.att.W_key.weight", i)),
                            device,
                        ),
                        bias: Some(Param::from_data(
                            tensor_data(&tensors, &format!("trf_blocks.{}.att.W_key.bias", i)),
                            device,
                        )),
                    },
                    w_value: Linear {
                        weight: Param::from_data(
                            tensor_data(&tensors, &format!("trf_blocks.{}.att.W_value.weight", i)),
                            device,
                        ),
                        bias: Some(Param::from_data(
                            tensor_data(&tensors, &format!("trf_blocks.{}.att.W_value.bias", i)),
                            device,
                        )),
                    },
                    out_proj: Linear {
                        weight: Param::from_data(
                            tensor_data(&tensors, &format!("trf_blocks.{}.att.out_proj.weight", i)),
                            device,
                        ),
                        bias: Some(Param::from_data(
                            tensor_data(&tensors, &format!("trf_blocks.{}.att.out_proj.bias", i)),
                            device,
                        )),
                    },
                    dropout: DropoutConfig::new(self.drop_rate).init(),
                    mask: Tensor::from_data(
                        tensor_data(&tensors, &format!("trf_blocks.{}.att.mask", i)),
                        device,
                    ),
                };
                let ff = FeedForward {
                    linear1: Linear {
                        weight: Param::from_data(
                            tensor_data(&tensors, &format!("trf_blocks.{}.ff.layers.0.weight", i)),
                            device,
                        ),
                        bias: Some(Param::from_data(
                            tensor_data(&tensors, &format!("trf_blocks.{}.ff.layers.0.bias", i)),
                            device,
                        )),
                    },
                    gelu: GELU {},
                    linear2: Linear {
                        weight: Param::from_data(
                            tensor_data(&tensors, &format!("trf_blocks.{}.ff.layers.2.weight", i)),
                            device,
                        ),
                        bias: Some(Param::from_data(
                            tensor_data(&tensors, &format!("trf_blocks.{}.ff.layers.2.bias", i)),
                            device,
                        )),
                    },
                };
                let norm1 = LayerNorm {
                    eps: 1e-5,
                    scale: Param::from_data(
                        tensor_data(&tensors, &format!("trf_blocks.{}.norm1.scale", i)),
                        device,
                    ),
                    shift: Param::from_data(
                        tensor_data(&tensors, &format!("trf_blocks.{}.norm1.shift", i)),
                        device,
                    ),
                };
                let norm2 = LayerNorm {
                    eps: 1e-5,
                    scale: Param::from_data(
                        tensor_data(&tensors, &format!("trf_blocks.{}.norm2.scale", i)),
                        device,
                    ),
                    shift: Param::from_data(
                        tensor_data(&tensors, &format!("trf_blocks.{}.norm2.shift", i)),
                        device,
                    ),
                };
                let dropout_shortcut = DropoutConfig::new(self.drop_rate).init();

                TransformerBlock {
                    attn,
                    ff,
                    norm1,
                    norm2,
                    dropout_shortcut,
                }
            })
            .collect();

        let final_norm = LayerNorm {
            eps: 1e-5,
            scale: Param::from_data(tensor_data(&tensors, "final_norm.shift"), device),
            shift: Param::from_data(tensor_data(&tensors, "final_norm.scale"), device),
        };
        let out_head = Linear {
            weight: Param::from_data(tensor_data(&tensors, "out_head.weight"), device),
            bias: None,
        };

        GPTModel {
            vocab_size: self.vocab_size,
            context_length: self.context_length,
            tok_emb,
            pos_emb,
            drop_emb,
            trf_blocks,
            final_norm,
            out_head,
        }
    }
}

#[derive(Module, Debug)]
pub struct TransformerBlock<B: Backend> {
    attn: MultiHeadAttention<B>,
    ff: FeedForward<B>,
    norm1: LayerNorm<B>,
    norm2: LayerNorm<B>,
    dropout_shortcut: Dropout,
}

impl<B: Backend> TransformerBlock<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let shortcut = x.clone();
        let x = self.norm1.forward(x);
        let x = self.attn.forward(x);
        let x = self.dropout_shortcut.forward(x);
        let x = x.add(shortcut);

        let shortcut = x.clone();
        let x = self.norm2.forward(x);
        let x = self.ff.forward(x);
        let x = self.dropout_shortcut.forward(x);
        let x = x.add(shortcut);

        x
    }
}

#[derive(Config, Debug)]
pub struct TransformerBlockConfig {
    pub context_length: usize,
    pub emb_dim: usize,
    pub n_heads: usize,
    pub drop_rate: f64,
    pub qkv_bias: bool,
}

impl TransformerBlockConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> TransformerBlock<B> {
        TransformerBlock {
            attn: MultiHeadAttentionConfig::new(
                self.emb_dim,
                self.emb_dim,
                self.context_length,
                self.drop_rate,
                self.n_heads,
                self.qkv_bias,
            )
            .init(device),
            ff: FeedForwardConfig::new(self.emb_dim).init(device),
            norm1: LayerNormConfig::new(self.emb_dim).init(device),
            norm2: LayerNormConfig::new(self.emb_dim).init(device),
            dropout_shortcut: DropoutConfig::new(self.drop_rate).init(),
        }
    }
}

#[derive(Module, Clone, Debug)]
pub struct GELU {}

impl GELU {
    pub fn forward<B: Backend, const D: usize>(&self, x: Tensor<B, D>) -> Tensor<B, D> {
        x.clone().mul_scalar(0.5).mul(
            x.clone()
                .add(x.powf_scalar(3).mul_scalar(0.044715))
                .mul_scalar((2.0 / PI).sqrt())
                .tanh()
                .add_scalar(1),
        )
    }
}

#[derive(Module, Debug)]
pub struct FeedForward<B: Backend> {
    linear1: Linear<B>,
    gelu: GELU,
    linear2: Linear<B>,
}

impl<B: Backend> FeedForward<B> {
    pub fn forward<const D: usize>(&self, input: Tensor<B, D>) -> Tensor<B, D> {
        let x = self.linear1.forward(input);
        let x = self.gelu.forward(x);
        self.linear2.forward(x)
    }
}

#[derive(Config, Debug)]
pub struct FeedForwardConfig {
    pub emb_dim: usize,
}

impl FeedForwardConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> FeedForward<B> {
        FeedForward {
            linear1: LinearConfig::new(self.emb_dim, 4 * self.emb_dim).init(device),
            gelu: GELU {},
            linear2: LinearConfig::new(4 * self.emb_dim, self.emb_dim).init(device),
        }
    }
}

#[derive(Module, Debug)]
pub struct LayerNorm<B: Backend> {
    eps: f64,
    scale: Param<Tensor<B, 1>>,
    shift: Param<Tensor<B, 1>>,
}

impl<B: Backend> LayerNorm<B> {
    pub fn forward<const D: usize>(&self, input: Tensor<B, D>) -> Tensor<B, D> {
        let (var, mean) = input.clone().var_mean_bias(D - 1);
        let norm_input = input.sub(mean).div(var.add_scalar(self.eps).sqrt());
        norm_input
            .mul(self.scale.val().unsqueeze())
            .add(self.shift.val().unsqueeze())
    }
}

#[derive(Config, Debug)]
pub struct LayerNormConfig {
    pub emb_dim: usize,
}

impl LayerNormConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> LayerNorm<B> {
        LayerNorm {
            eps: 1e-5,
            scale: Param::from_tensor(Tensor::ones([self.emb_dim], device)),
            shift: Param::from_tensor(Tensor::zeros([self.emb_dim], device)),
        }
    }
}

pub fn generate_text_simple<B: Backend>(
    model: &GPTModel<B>,
    mut idx: Tensor<B, 2, Int>,
    max_new_tokens: usize,
    context_size: usize,
) -> Tensor<B, 2, Int> {
    for _ in 0..max_new_tokens {
        let [n_batches, n_tokens] = idx.clone().dims();
        let idx_cond = idx.clone().slice([
            0..n_batches,
            n_tokens.max(context_size) - context_size..n_tokens,
        ]);

        let logits = model.forward(idx_cond);
        let last_logits = logits
            .slice([0..n_batches, n_tokens - 1..n_tokens, 0..model.vocab_size])
            .squeeze::<2>(1);

        let probas = softmax(last_logits, 1);
        let idx_next = probas.argmax(1);
        idx = Tensor::cat(vec![idx, idx_next], 1);
    }

    idx
}

pub fn generate_text<B: Backend>(
    model: &GPTModel<B>,
    mut idx: Tensor<B, 2, Int>,
    max_new_tokens: usize,
    context_size: usize,
    temperature: f64,
    top_k: Option<usize>,
) -> Tensor<B, 2, Int> {
    for _ in 0..max_new_tokens {
        let [n_batches, n_tokens] = idx.clone().dims();
        let idx_cond = idx.clone().slice([
            0..n_batches,
            n_tokens.max(context_size) - context_size..n_tokens,
        ]);

        let logits = model.forward(idx_cond);
        let mut last_logits = logits
            .slice([0..n_batches, n_tokens - 1..n_tokens, 0..model.vocab_size])
            .squeeze::<2>(1);

        if let Some(top_k) = top_k {
            let top_k_logits = last_logits.clone().topk(top_k, 1);
            let min_val = top_k_logits.slice([0..n_batches, top_k - 1..top_k]);
            last_logits = last_logits
                .clone()
                .mask_fill(last_logits.lower(min_val), f64::NEG_INFINITY);
        }

        let idx_next = if temperature > 0.0 {
            last_logits = last_logits.div_scalar(temperature);
            let probas = softmax(last_logits, 1);

            // sample from the distribution for each batch
            let mut next_idxs = Vec::new();
            for i in 0..n_batches {
                let mut rng = rand::rng();
                let batch_probas_data = probas
                    .clone()
                    .slice([i..i + 1, 0..model.vocab_size])
                    .to_data();
                let idx = if let Ok(batch_probas) = batch_probas_data.to_vec::<f32>() {
                    if let Ok(dist) = WeightedIndex::new(batch_probas) {
                        rng.sample(dist)
                    } else {
                        0
                    }
                } else {
                    if let Ok(dist) = WeightedIndex::new(batch_probas_data.to_vec::<f64>().unwrap())
                    {
                        rng.sample(dist)
                    } else {
                        0
                    }
                };
                next_idxs.push(Tensor::<B, 1, Int>::from([idx]));
            }
            Tensor::stack(next_idxs, 0)
        } else {
            last_logits.argmax(1)
        };

        idx = Tensor::cat(vec![idx, idx_next], 1);
    }

    idx
}

pub fn text_to_token_ids<B: Backend>(
    text: &str,
    tokenizer: &tokenizer::BpeTokenizer,
) -> Tensor<B, 2, Int> {
    let encoded = tokenizer.encode(text);
    Tensor::<B, 1, Int>::from(&encoded[..]).unsqueeze()
}

pub fn token_ids_to_text<B: Backend>(
    token_ids: Tensor<B, 2, Int>,
    tokenizer: &tokenizer::BpeTokenizer,
) -> Result<String> {
    let out_ids = token_ids.squeeze::<1>(0).to_data();
    let u32_ids = if let Ok(i32_ids) = out_ids.to_vec::<i32>() {
        i32_ids.iter().map(|id| *id as u32).collect::<Vec<_>>()
    } else {
        let i64_ids = out_ids.to_vec::<i64>().unwrap();
        i64_ids.iter().map(|id| *id as u32).collect::<Vec<_>>()
    };
    tokenizer.decode(&u32_ids)
}
