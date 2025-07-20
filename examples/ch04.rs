use burn::tensor::{Distribution, Int, Tensor};
use llms_from_scratch_burn::{
    Backend,
    gpt::{FeedForwardConfig, GELU, GPTModelConfig, LayerNormConfig, TransformerBlockConfig},
    tokenizer::{self, ITokenizer},
};

fn main() {
    let device = &Default::default();

    // 4.2 normalizing activations with layer normalization
    println!("\n4.2 normalizing activations with layer normalization");

    let batch_example = Tensor::<Backend, 2>::random([2, 5], Distribution::Default, device);

    let ln = LayerNormConfig::new(5).init::<Backend>(device);
    let x = ln.forward(batch_example.clone());
    println!("Normalized layer outputs:\n{}", x);

    let (var, mean) = x.var_mean_bias(1);
    println!("Mean:\n{}, Var:\n{}", mean, var);

    // 4.3 implementing a feed forward network with GELU activations
    println!("\n4.3 implementing a feed forward network with GELU activations");

    let x = Tensor::<Backend, 1>::from_floats(
        &(0..=100)
            .map(|x| (x as f64) / (100.0 / 6.0) - 3.0)
            .collect::<Vec<_>>()[..],
        device,
    );
    println!("Input:\n{}", x);

    let gelu = GELU {};
    println!("GELU outputs:\n{}", gelu.forward(x));

    let x = Tensor::<Backend, 3>::random([2, 3, 768], Distribution::Default, device);
    let ff = FeedForwardConfig::new(768).init::<Backend>(device);
    println!("Feed Forward outputs:\n{}", ff.forward(x));

    // 4.5 connecting attention and linear layers in a transformer block
    todo!("refine MultiHeadAttention, reduce dim_out.");

    let x = Tensor::<Backend, 3>::random([2, 4, 768], Distribution::Default, device);
    let block = TransformerBlockConfig::new(1024, 768, 12, 0.1, false).init::<Backend>(device);
    println!("Transformer Block outputs:\n{}", block.forward(x));

    // 4.6 coding the GPT model
    println!("\n4.6 coding the GPT model");

    let tokenizer = tokenizer::BpeTokenizer::new();

    let mut batch = vec![];
    let txt1 = "Every effort moves you";
    let txt2 = "Every day holds a";
    batch.push(Tensor::<Backend, 1, Int>::from(&tokenizer.encode(txt1)[..]));
    batch.push(Tensor::<Backend, 1, Int>::from(&tokenizer.encode(txt2)[..]));

    let batch = Tensor::stack::<2>(batch, 0);
    println!("{}", batch);

    let model = GPTModelConfig::new(50257, 1024, 768, 12, 12, 0.1, false).init::<Backend>(device);

    let logits = model.forward(batch);
    println!("{}", logits);
}
