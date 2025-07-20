use burn::tensor::{Int, Tensor};
use llms_from_scratch_burn::{
    Backend,
    gpt::DummyGPTModelConfig,
    tokenizer::{self, ITokenizer},
};

fn main() {
    // 4.1 coding an LLM architecture
    println!("4.1 coding an LLM architecture");

    let tokenizer = tokenizer::BpeTokenizer::new();

    let mut batch = vec![];
    let txt1 = "Every effort moves you";
    let txt2 = "Every day holds a";
    batch.push(Tensor::<Backend, 1, Int>::from(&tokenizer.encode(txt1)[..]));
    batch.push(Tensor::<Backend, 1, Int>::from(&tokenizer.encode(txt2)[..]));

    let batch = Tensor::stack::<2>(batch, 0);
    println!("{}", batch);

    let device = &Default::default();
    let model =
        DummyGPTModelConfig::new(50257, 1024, 768, 12, 12, 0.1, false).init::<Backend>(device);

    let logits = model.forward(batch);
    println!("{}", logits);

    // 4.2 normalizing activations with layer normalization
    println!("\n4.2 normalizing activations with layer normalization");

    todo!()
}
