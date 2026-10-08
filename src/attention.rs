use burn::config::Config;
use burn::module::Module;
use burn::nn::{Dropout, DropoutConfig, Linear, LinearConfig};
use burn::tensor::Bool;
use burn::tensor::activation::softmax;
use burn::tensor::cast::ToElement;
use burn::tensor::{Device, Tensor};

#[derive(Module, Debug)]
pub struct SelfAttentionV2 {
    pub w_query: Linear,
    pub w_key: Linear,
    pub w_value: Linear,
}

impl SelfAttentionV2 {
    pub fn forward<const D: usize>(&self, input: Tensor<D>) -> Tensor<D> {
        let inputs = Tensor::<D>::from(input);
        let queries = self.w_query.forward(inputs.clone());
        let keys = self.w_key.forward(inputs.clone());
        let values = self.w_value.forward(inputs);

        let attn_scores = queries.matmul(keys.clone().transpose());
        let attn_weights = softmax(
            attn_scores.div_scalar(keys.dims()[D - 1].to_f64().sqrt()),
            D - 1,
        );

        attn_weights.matmul(values)
    }
}

#[derive(Config, Debug)]
pub struct SelfAttentionV2Config {}

impl SelfAttentionV2Config {
    pub fn init(
        &self,
        d_in: usize,
        d_out: usize,
        qkv_bias: bool,
        device: &Device,
    ) -> SelfAttentionV2 {
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
pub struct CausalAttention {
    #[module(skip)]
    d_out: usize,
    w_query: Linear,
    w_key: Linear,
    w_value: Linear,
    dropout: Dropout,
    mask: Tensor<2, Bool>,
}

impl CausalAttention {
    pub fn forward<const D: usize>(&self, input: Tensor<D>) -> Tensor<D> {
        let dims = input.dims();
        let num_tokens = dims[D - 2];

        let inputs = Tensor::<D>::from(input);
        let queries = self.w_query.forward(inputs.clone());
        let keys = self.w_key.forward(inputs.clone());
        let values = self.w_value.forward(inputs);

        let attn_scores = queries.matmul(keys.clone().transpose());
        let masked = attn_scores.mask_fill(
            self.mask
                .clone()
                .slice([..num_tokens, ..num_tokens])
                .unsqueeze(),
            f64::NEG_INFINITY,
        );

        let attn_weights = softmax(masked.div_scalar(self.d_out.to_f64().sqrt()), D - 1);
        let attn_weights = self.dropout.forward(attn_weights);

        attn_weights.matmul(values)
    }
}

#[derive(Config, Debug)]
pub struct CausalAttentionConfig {}

impl CausalAttentionConfig {
    pub fn init(
        &self,
        d_in: usize,
        d_out: usize,
        context_length: usize,
        dropout: f64,
        qkv_bias: bool,
        device: &Device,
    ) -> CausalAttention {
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
            mask: Tensor::<2, Bool>::tril_mask([context_length, context_length], 0, device),
        }
    }
}

#[derive(Module, Debug)]
pub struct NaiveMultiHeadAttention {
    heads: Vec<CausalAttention>,
}

impl NaiveMultiHeadAttention {
    pub fn forward<const D: usize>(&self, input: Tensor<D>) -> Tensor<D> {
        let outputs = self
            .heads
            .iter()
            .map(|h| h.forward(input.clone()))
            .collect();
        Tensor::<D>::cat(outputs, D - 1)
    }
}

#[derive(Module, Debug)]
pub struct MultiHeadAttention {
    #[module(skip)]
    pub d_out: usize,
    #[module(skip)]
    pub num_heads: usize,
    #[module(skip)]
    pub head_dim: usize,
    pub w_query: Linear,
    pub w_key: Linear,
    pub w_value: Linear,
    pub out_proj: Linear,
    pub dropout: Dropout,
    pub mask: Tensor<2, Bool>,
}

impl MultiHeadAttention {
    pub fn forward(&self, input: Tensor<3>) -> Tensor<3> {
        let [b, num_tokens, _d_in] = input.dims();

        let inputs = Tensor::<3>::from(input);
        let queries = self.w_query.forward(inputs.clone());
        let keys = self.w_key.forward(inputs.clone());
        let values = self.w_value.forward(inputs);

        let queries = queries
            .reshape([b, num_tokens, self.num_heads, self.head_dim])
            .swap_dims(1, 2);
        let keys = keys
            .reshape([b, num_tokens, self.num_heads, self.head_dim])
            .swap_dims(1, 2);
        let values = values
            .reshape([b, num_tokens, self.num_heads, self.head_dim])
            .swap_dims(1, 2);

        let attn_scores = queries.matmul(keys.clone().transpose());
        let masked = attn_scores.mask_fill(
            self.mask
                .clone()
                .slice([..num_tokens, ..num_tokens])
                .unsqueeze(),
            f64::NEG_INFINITY,
        );

        let attn_weights = softmax(masked.div_scalar(self.head_dim.to_f64().sqrt()), 3);
        let attn_weights = self.dropout.forward(attn_weights);

        let context_vecs = attn_weights.matmul(values).swap_dims(1, 2);
        let context_vecs = context_vecs.reshape([b, num_tokens, self.d_out]);
        self.out_proj.forward(context_vecs)
    }
}

#[derive(Config, Debug)]
pub struct MultiHeadAttentionConfig {
    d_in: usize,
    d_out: usize,
    context_length: usize,
    dropout: f64,
    num_heads: usize,
    qkv_bias: bool,
}

impl MultiHeadAttentionConfig {
    pub fn init(&self, device: &Device) -> MultiHeadAttention {
        MultiHeadAttention {
            d_out: self.d_out,
            num_heads: self.num_heads,
            head_dim: self.d_out / self.num_heads,
            w_query: LinearConfig::new(self.d_in, self.d_out)
                .with_bias(self.qkv_bias)
                .init(device),
            w_key: LinearConfig::new(self.d_in, self.d_out)
                .with_bias(self.qkv_bias)
                .init(device),
            w_value: LinearConfig::new(self.d_in, self.d_out)
                .with_bias(self.qkv_bias)
                .init(device),
            out_proj: LinearConfig::new(self.d_out, self.d_out)
                .with_bias(self.qkv_bias)
                .init(device),
            dropout: DropoutConfig::new(self.dropout).init(),
            mask: Tensor::<2, Bool>::tril_mask(
                [self.context_length, self.context_length],
                0,
                device,
            ),
        }
    }

    pub fn init_naive(&self, device: &Device) -> NaiveMultiHeadAttention {
        NaiveMultiHeadAttention {
            heads: (0..self.num_heads)
                .map(|_| {
                    CausalAttentionConfig::new().init(
                        self.d_in,
                        self.d_out,
                        self.context_length,
                        self.dropout,
                        self.qkv_bias,
                        device,
                    )
                })
                .collect(),
        }
    }
}
