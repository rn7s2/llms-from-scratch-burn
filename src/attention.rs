use burn::config::Config;
use burn::module::Module;
use burn::nn::{Dropout, DropoutConfig, Linear, LinearConfig};
use burn::tensor::Bool;
use burn::tensor::activation::softmax;
use burn::tensor::cast::ToElement;
use burn::tensor::{Tensor, backend::Backend};

#[derive(Module, Debug)]
pub struct SelfAttentionV2<B: Backend> {
    pub w_query: Linear<B>,
    pub w_key: Linear<B>,
    pub w_value: Linear<B>,
}

impl<B: Backend> SelfAttentionV2<B> {
    pub fn forward<const D: usize>(&self, input: Tensor<B, D>) -> Tensor<B, D> {
        assert_eq!(D, 2);

        let inputs = Tensor::<B, D>::from(input);
        let queries = self.w_query.forward(inputs.clone());
        let keys = self.w_key.forward(inputs.clone());
        let values = self.w_key.forward(inputs);

        let attn_scores = queries.matmul(keys.clone().transpose());
        let attn_weights = softmax(
            attn_scores.div_scalar(keys.shape().dims[1].to_f64().sqrt()),
            1,
        );

        attn_weights.matmul(values)
    }
}

#[derive(Config, Debug)]
pub struct SelfAttentionV2Config {}

impl SelfAttentionV2Config {
    pub fn init<B: Backend>(
        &self,
        d_in: usize,
        d_out: usize,
        qkv_bias: bool,
        device: &B::Device,
    ) -> SelfAttentionV2<B> {
        SelfAttentionV2 {
            w_query: LinearConfig::new(d_in, d_out)
                .with_bias(qkv_bias)
                .init(device),
            w_key: LinearConfig::new(d_in, d_out)
                .with_bias(qkv_bias)
                .init(device),
            w_value: LinearConfig::new(d_in, d_out)
                .with_bias(qkv_bias)
                .init(device),
        }
    }
}

#[derive(Module, Debug)]
pub struct CausalAttention<B: Backend> {
    d_out: usize,
    w_query: Linear<B>,
    w_key: Linear<B>,
    w_value: Linear<B>,
    dropout: Dropout,
    mask: Tensor<B, 2, Bool>,
}

impl<B: Backend> CausalAttention<B> {
    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let [_b, num_tokens, _d_in] = input.shape().dims();

        let inputs = Tensor::<B, 3>::from(input);
        let queries = self.w_query.forward(inputs.clone());
        let keys = self.w_key.forward(inputs.clone());
        let values = self.w_key.forward(inputs);

        let attn_scores = queries.matmul(keys.clone().transpose());
        let masked = attn_scores.mask_fill(
            self.mask
                .clone()
                .slice([..num_tokens, ..num_tokens])
                .unsqueeze(),
            -f64::INFINITY,
        );

        let attn_weights = softmax(masked.div_scalar(self.d_out.to_f64().sqrt()), 1);
        let attn_weights = self.dropout.forward(attn_weights);

        attn_weights.matmul(values)
    }
}

#[derive(Config, Debug)]
pub struct CausalAttentionConfig {}

impl CausalAttentionConfig {
    pub fn init<B: Backend>(
        &self,
        d_in: usize,
        d_out: usize,
        context_length: usize,
        dropout: f64,
        qkv_bias: bool,
        device: &B::Device,
    ) -> CausalAttention<B> {
        CausalAttention {
            d_out,
            w_query: LinearConfig::new(d_in, d_out)
                .with_bias(qkv_bias)
                .init(device),
            w_key: LinearConfig::new(d_in, d_out)
                .with_bias(qkv_bias)
                .init(device),
            w_value: LinearConfig::new(d_in, d_out)
                .with_bias(qkv_bias)
                .init(device),
            dropout: DropoutConfig::new(dropout).init(),
            mask: Tensor::<B, 2, Bool>::tril_mask([context_length, context_length], 0, device),
        }
    }
}

#[derive(Module, Debug)]
pub struct MultiHeadAttention<B: Backend> {
    heads: Vec<CausalAttention<B>>,
}

impl<B: Backend> MultiHeadAttention<B> {
    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let outputs = self
            .heads
            .iter()
            .map(|h| h.forward(input.clone()))
            .collect();
        Tensor::<B, 3>::cat(outputs, 2)
    }
}

#[derive(Config, Debug)]
pub struct MultiHeadAttentionConfig {}

impl MultiHeadAttentionConfig {
    pub fn init<B: Backend>(
        &self,
        d_in: usize,
        d_out: usize,
        context_length: usize,
        dropout: f64,
        num_heads: usize,
        qkv_bias: bool,
        device: &B::Device,
    ) -> MultiHeadAttention<B> {
        MultiHeadAttention {
            heads: (0..num_heads)
                .map(|_| {
                    CausalAttentionConfig::new().init(
                        d_in,
                        d_out,
                        context_length,
                        dropout,
                        qkv_bias,
                        device,
                    )
                })
                .collect(),
        }
    }
}
