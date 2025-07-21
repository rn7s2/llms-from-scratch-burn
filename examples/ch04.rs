use burn::{
    module::Module,
    tensor::{Distribution, Int, Tensor, activation::softmax},
};
use llms_from_scratch_burn::{
    InferenceBackend as Backend,
    gpt::{
        FeedForwardConfig, GELU, GPTModel, GPTModelConfig, LayerNormConfig, TransformerBlockConfig,
    },
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

    let gpt_config_124m = GPTModelConfig::new(50257, 1024, 768, 12, 12, 0.1, false);
    let model = gpt_config_124m.init::<Backend>(device);

    let logits = model.forward(batch);
    println!("{}", logits);

    println!("Total number of parameters: {}", model.num_params());

    // 4.7 generating text
    println!("\n4.7 generating text");

    let start_context = "Hello, I am";

    let encoded = tokenizer.encode(start_context);
    let encoded_tensor = Tensor::<Backend, 1, Int>::from(&encoded[..]).unsqueeze::<2>();
    println!("encoded_tensor: {}", encoded_tensor);

    let out = generate_text_simple(&model, encoded_tensor, 6, gpt_config_124m.context_length);
    println!("Output: {}", out);

    let out_ids = out.squeeze::<1>(0).to_data();
    let u32_ids = if let Ok(i32_ids) = out_ids.to_vec::<i32>() {
        i32_ids.iter().map(|id| *id as u32).collect::<Vec<_>>()
    } else {
        let i64_ids = out_ids.to_vec::<i64>().unwrap();
        i64_ids.iter().map(|id| *id as u32).collect::<Vec<_>>()
    };
    let decoded_text = tokenizer.decode(&u32_ids).unwrap();
    println!("Output text: {}", decoded_text);
}

fn generate_text_simple(
    model: &GPTModel<Backend>,
    mut idx: Tensor<Backend, 2, Int>,
    max_new_tokens: usize,
    context_size: usize,
) -> Tensor<Backend, 2, Int> {
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
