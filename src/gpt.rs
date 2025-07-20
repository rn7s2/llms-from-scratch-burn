use burn::config::Config;
use burn::module::Module;
use burn::nn::{Dropout, DropoutConfig, Embedding, EmbeddingConfig, Linear, LinearConfig};
use burn::tensor::Int;
use burn::tensor::{Tensor, backend::Backend};

#[derive(Module, Debug)]
pub struct DummyGPTModel<B: Backend> {
    tok_emb: Embedding<B>,
    pos_emb: Embedding<B>,
    drop_emb: Dropout,
    trf_blocks: Vec<DummyTransformerBlock<B>>,
    final_norm: DummyLayerNorm<B>,
    out_head: Linear<B>,
}

impl<B: Backend> DummyGPTModel<B> {
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
pub struct DummyGPTModelConfig {
    pub vocab_size: usize,
    pub context_length: usize,
    pub emb_dim: usize,
    pub n_heads: usize,
    pub n_layers: usize,
    pub drop_rate: f64,
    pub qkv_bias: bool,
}

impl DummyGPTModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> DummyGPTModel<B> {
        DummyGPTModel {
            tok_emb: EmbeddingConfig::new(self.vocab_size, self.emb_dim).init(device),
            pos_emb: EmbeddingConfig::new(self.vocab_size, self.emb_dim).init(device),
            drop_emb: DropoutConfig::new(self.drop_rate).init(),
            trf_blocks: (0..self.n_layers)
                .map(|_| DummyTransformerBlock {
                    _placeholder: LinearConfig::new(self.emb_dim, self.emb_dim).init(device),
                })
                .collect(),
            final_norm: DummyLayerNorm {
                _placeholder: LinearConfig::new(self.emb_dim, self.emb_dim).init(device),
            },
            out_head: LinearConfig::new(self.emb_dim, self.vocab_size)
                .with_bias(false)
                .init(device),
        }
    }
}

#[derive(Module, Debug)]
pub struct DummyTransformerBlock<B: Backend> {
    _placeholder: Linear<B>,
}

impl<B: Backend> DummyTransformerBlock<B> {
    pub fn forward<const D: usize>(&self, input: Tensor<B, D>) -> Tensor<B, D> {
        input
    }
}

#[derive(Module, Debug)]
pub struct DummyLayerNorm<B: Backend> {
    _placeholder: Linear<B>,
}

impl<B: Backend> DummyLayerNorm<B> {
    pub fn forward<const D: usize>(&self, input: Tensor<B, D>) -> Tensor<B, D> {
        input
    }
}
