use burn::tensor::{Int, Tensor};
use llms_from_scratch_burn::{
    Backend,
    gpt::{GPTModelConfig, generate_text_simple},
    tokenizer::{self, ITokenizer},
};

fn main() {
    let device = &Default::default();

    // 5.1 Evaluating generative text models
    println!("5.1 Evaluating generative text models");

    // 5.1.1 Using GPT to generate text
    println!("5.1.1 Using GPT to generate text");

    let gpt_config_124m = GPTModelConfig::new(50257, 1024, 768, 12, 12, 0.1, false);
    let model = gpt_config_124m.init::<Backend>(device);

    let tokenizer = tokenizer::BpeTokenizer::new();

    let start_context = "Every effort moves you";

    let token_ids = generate_text_simple(
        &model,
        text_to_token_ids(start_context, &tokenizer),
        6,
        gpt_config_124m.context_length,
    );
    println!("Output text:\n{}", token_ids_to_text(token_ids, &tokenizer));

    todo!()
}

fn text_to_token_ids(text: &str, tokenizer: &tokenizer::BpeTokenizer) -> Tensor<Backend, 2, Int> {
    let encoded = tokenizer.encode(text);
    Tensor::<Backend, 1, Int>::from(&encoded[..]).unsqueeze()
}

fn token_ids_to_text(
    token_ids: Tensor<Backend, 2, Int>,
    tokenizer: &tokenizer::BpeTokenizer,
) -> String {
    let out_ids = token_ids.squeeze::<1>(0).to_data();
    let u32_ids = if let Ok(i32_ids) = out_ids.to_vec::<i32>() {
        i32_ids.iter().map(|id| *id as u32).collect::<Vec<_>>()
    } else {
        let i64_ids = out_ids.to_vec::<i64>().unwrap();
        i64_ids.iter().map(|id| *id as u32).collect::<Vec<_>>()
    };
    tokenizer.decode(&u32_ids).unwrap()
}
