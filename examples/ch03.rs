use burn::{
    nn::DropoutConfig,
    tensor::{Bool, Distribution, Tensor, activation::softmax},
};
use llms_from_scratch_burn::{
    TrainBackend,
    attention::{CausalAttentionConfig, MultiHeadAttentionConfig, SelfAttentionV2Config},
};

fn main() {
    let device = Default::default();

    const INPUTS: [[f64; 3]; 6] = [
        [0.43, 0.15, 0.89], // Your     (x^1)
        [0.55, 0.87, 0.66], // journey  (x^2)
        [0.57, 0.85, 0.64], // starts   (x^3)
        [0.22, 0.58, 0.33], // with     (x^4)
        [0.77, 0.25, 0.10], // one      (x^5)
        [0.05, 0.80, 0.55], // step     (x^6)
    ];
    let inputs = Tensor::<TrainBackend, 2>::from(INPUTS);

    // 3.3. attending to different parts of the input with self-attention
    println!("3.3. attending to different parts of the input with self-attention");

    // 3.3.1. attention weights for the second token (a single query)
    println!("\n3.3.1. attention weights for the second token (a single query)");

    let query_2 = Tensor::<TrainBackend, 1>::from_data(INPUTS[1], &device)
        .unsqueeze::<2>()
        .transpose();
    println!("{}", query_2);

    let attn_scores_2 = inputs.clone().matmul(query_2).transpose().squeeze::<1>(0);
    println!("Attention scores: {}", attn_scores_2);

    let attn_weights_2 = softmax(attn_scores_2, 0);
    println!("Attention weights: {}", attn_weights_2);
    println!("Sum: {}", attn_weights_2.clone().sum());

    let context_vec_2 = attn_weights_2
        .unsqueeze::<2>()
        .matmul(inputs.clone())
        .squeeze::<1>(0);
    println!("Context vector: {}", context_vec_2);

    // 3.3.2. attention weights for all input tokens
    println!("\n3.3.2. attention weights for all input tokens");

    let attn_scores = inputs
        .clone()
        .matmul(inputs.clone().transpose())
        .transpose();
    println!("Attention scores: {}", attn_scores);

    let attn_weights = softmax(attn_scores, 1);
    println!("Attention weights: {}", attn_weights);
    println!("Sum: {}", attn_weights.clone().sum_dim(1).transpose());

    let context_vecs = attn_weights.matmul(inputs.clone());
    println!("Context vectors: {}", context_vecs);

    // 3.4. self-attention with trainable weights
    println!("\n3.4. self-attention with trainable weights");

    const DIM_IN: usize = 3;
    const DIM_OUT: usize = 2;

    // 3.4.1. computing the attention weights step by step
    println!("\n3.4.1. computing the attention weights step by step");

    let distribution = Distribution::Uniform(0.0, 1.0); // Any random value between 0.0 and 1.0
    let w_query = Tensor::<TrainBackend, 2>::random([DIM_IN, DIM_OUT], distribution, &device);
    let w_key = Tensor::<TrainBackend, 2>::random([DIM_IN, DIM_OUT], distribution, &device);
    let w_value = Tensor::<TrainBackend, 2>::random([DIM_IN, DIM_OUT], distribution, &device);

    let query_2 = Tensor::<TrainBackend, 1>::from_data(INPUTS[1], &device)
        .unsqueeze::<2>()
        .matmul(w_query.clone());
    println!("{}", query_2);

    let keys = inputs.clone().matmul(w_key.clone());
    let values = inputs.clone().matmul(w_value.clone());
    println!("{}\n{}", keys, values);

    let attn_scores_2 = query_2.matmul(keys.transpose());
    println!("Attention scores: {}", attn_scores_2);

    let attn_weights_2 = softmax(attn_scores_2.div_scalar((DIM_OUT as f64).sqrt()), 1);
    println!("Attention weights: {}", attn_weights_2);
    println!("Sum: {}", attn_weights_2.clone().sum_dim(1));

    let context_vec_2 = attn_weights_2.matmul(values).squeeze::<1>(0);
    println!("Context vector: {}", context_vec_2);

    // 3.4.2. implementing a compact SelfAttention class
    println!("\n3.4.2. implementing a compact SelfAttention class");

    let model = SelfAttentionV2Config::new().init::<TrainBackend>(DIM_IN, DIM_OUT, false, &device);
    println!("{}", model);

    let sa_v2 = model.forward(inputs.clone());
    println!("{}", sa_v2);

    // 3.5. hiding future words with causal attention
    println!("\n3.5. hiding future words with causal attention");

    // 3.5.1. applying a causal attention mask
    println!("\n3.5.1. applying a causal attention mask");

    let queries = model.w_query.forward(inputs.clone());
    let keys = model.w_key.forward(inputs.clone());

    let attn_scores = queries.matmul(keys.transpose());
    let attn_weights = softmax(attn_scores.clone().div_scalar((DIM_OUT as f64).sqrt()), 1);
    println!("Attention weights: {}", attn_weights);
    println!("Sum: {}", attn_weights.clone().sum_dim(1));

    const CONTEXT_LEN: usize = INPUTS.len();
    let mask_simple =
        Tensor::<TrainBackend, 2>::tril(Tensor::ones([CONTEXT_LEN, CONTEXT_LEN], &device), 0);
    println!("Mask: {}", mask_simple);

    let masked_simple = attn_weights * mask_simple.clone();
    println!("Masked attention weights: {}", masked_simple);

    let row_sums = masked_simple.clone().sum_dim(1);
    let masked_simple_norm = masked_simple.div(row_sums);
    println!(
        "Masked attention weights normalized: {}",
        masked_simple_norm
    );

    let mask = Tensor::<TrainBackend, 2, Bool>::tril_mask([CONTEXT_LEN, CONTEXT_LEN], 0, &device);
    println!("Mask: {}", mask);
    let masked = attn_scores.mask_fill(mask, -f64::INFINITY);
    println!("Masked attention scores: {}", masked);

    let attn_weights = softmax(masked.div_scalar((DIM_OUT as f64).sqrt()), 1);
    println!("Attention weights: {}", attn_weights);
    println!("Sum: {}", attn_weights.clone().sum_dim(1));

    // 3.5.2. masking additional attention weights with dropout
    println!("\n3.5.2. masking additional attention weights with dropout");

    let dropout = DropoutConfig::new(0.5).init();
    let attn_weights_dropout = dropout.forward(attn_weights);
    println!("Attention weights dropout: {}", attn_weights_dropout);

    // 3.5.3. implementing a compact causal self-attention class
    println!("\n3.5.3. implementing a compact causal self-attention class");

    let batch = Tensor::stack::<3>(vec![inputs.clone(), inputs.clone()], 0);
    println!("{}", batch);

    let model = CausalAttentionConfig::new().init::<TrainBackend>(
        DIM_IN,
        DIM_OUT,
        CONTEXT_LEN,
        0.0,
        false,
        &device,
    );
    println!("{}", model);

    let ca = model.forward(batch.clone());
    println!("{}", ca);

    // 3.6. extending single-head attention to multi-head attention
    println!("\n3.6. extending single-head attention to multi-head attention");

    // 3.6.1. stacking multiple single-head attention layers
    println!("3.6.1. stacking multiple single-head attention layers");

    let mha = MultiHeadAttentionConfig::new(DIM_IN, DIM_OUT, CONTEXT_LEN, 0.0, 2, false)
        .init_naive::<TrainBackend>(&device);
    println!("{}", mha);

    let context_vecs = mha.forward(batch.clone());
    println!("{}", context_vecs);

    // 3.6.2. implementing multi-head attention with weight splits
    println!("\n3.6.2. implementing multi-head attention with weight splits");

    let mha = MultiHeadAttentionConfig::new(DIM_IN, DIM_OUT, CONTEXT_LEN, 0.0, 2, false)
        .init::<TrainBackend>(&device);
    println!("{}", mha);

    let context_vecs = mha.forward(batch);
    println!("{}", context_vecs);
}
