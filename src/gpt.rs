use std::f64::consts::PI;

use burn::config::Config;
use burn::module::{Module, Param};
use burn::nn::{Dropout, DropoutConfig, Embedding, EmbeddingConfig, Linear, LinearConfig};
use burn::tensor::Int;
use burn::tensor::activation::softmax;
use burn::tensor::{Tensor, backend::Backend};

use crate::attention::{MultiHeadAttention, MultiHeadAttentionConfig};

#[derive(Module, Debug)]
pub struct GPTModel<B: Backend> {
    pub vocab_size: usize,
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
