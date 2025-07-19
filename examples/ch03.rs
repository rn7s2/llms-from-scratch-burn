use burn::tensor::{Tensor, activation::softmax};

type Backend = burn::backend::Wgpu;

fn main() {
    let device = Default::default();

    let inputs = [
        [0.43, 0.15, 0.89], // Your     (x^1)
        [0.55, 0.87, 0.66], // journey  (x^2)
        [0.57, 0.85, 0.64], // starts   (x^3)
        [0.22, 0.58, 0.33], // with     (x^4)
        [0.77, 0.25, 0.10], // one      (x^5)
        [0.05, 0.80, 0.55], // step     (x^6)
    ];

    // 3.3. attending to different parts of the input with self-attention
    // 3.3.1. attention weights for the second token (a single query)
    let query_2 = Tensor::<Backend, 1>::from_data(inputs[1], &device)
        .unsqueeze::<2>()
        .transpose();
    println!("{}", query_2);

    let attn_scores_2 = Tensor::<Backend, 2>::from_data(inputs, &device)
        .matmul(query_2)
        .transpose()
        .squeeze::<1>(0);
    println!("Attention scores: {}", attn_scores_2);

    let attn_weights_2 = softmax(attn_scores_2, 0);
    println!("Attention weights: {}", attn_weights_2);
    println!("Sum: {}", attn_weights_2.clone().sum());

    let context_vec_2 = attn_weights_2
        .unsqueeze::<2>()
        .matmul(Tensor::<Backend, 2>::from_data(inputs, &device))
        .squeeze::<1>(0);
    println!("Context vector: {}", context_vec_2);

    println!();

    // 3.3.2. attention weights for all input tokens
    let attn_scores = Tensor::<Backend, 2>::from_data(inputs, &device)
        .matmul(Tensor::<Backend, 2>::from_data(inputs, &device).transpose())
        .transpose();
    println!("Attention scores: {}", attn_scores);

    let attn_weights = softmax(attn_scores, 1);
    println!("Attention weights: {}", attn_weights);
    println!("Sum: {}", attn_weights.clone().sum_dim(1).transpose());

    let context_vecs = attn_weights.matmul(Tensor::<Backend, 2>::from_data(inputs, &device));
    println!("Context vectors: {}", context_vecs);

    println!();

    // 3.4. self-attention with trainable weights
    todo!()
}
