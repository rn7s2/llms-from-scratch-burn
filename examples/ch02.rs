use burn::backend::Wgpu;
use burn::data::dataloader::Dataset;
use burn::data::dataloader::batcher::Batcher;
use llms_from_scratch_burn::{
    dataset,
    tokenizer::{self, ITokenizer},
};

fn main() {
    // 2.4. SimpleTokenizerV2
    let vocab = tokenizer::SimpleTokenizerV2::fetch_vocab();
    let tokenizer = tokenizer::SimpleTokenizerV2::new(&vocab);

    let text1 = "Hello, do you like tea?";
    let text2 = "In the sunlit terraces of the palace.";
    let text = [text1, text2].join(" <|endoftext|> ");
    println!("{}", text);

    let ids = tokenizer.encode(&text);
    println!("encoded: {:?}", ids);

    let decoded = tokenizer.decode(&ids).unwrap();
    println!("decoded: {}", decoded);

    println!();

    // 2.5. BPE Tokenizer
    let tokenizer = tokenizer::BpeTokenizer::new();

    let text = "Hello, do you like tea? <|endoftext|> In the sunlit terracesof someunknownPlace.";
    println!("{}", text);

    let ids = tokenizer.encode(&text);
    println!("encoded: {:?}", ids);

    let decoded = tokenizer.decode(&ids).unwrap();
    println!("decoded: {}", decoded);

    println!();

    // 2.6. Dataset & Batcher
    let path = "assets/the-verdict.txt";
    let text = std::fs::read_to_string(path).unwrap();

    let dataset = dataset::GPTDatasetV1::<4>::new(&text, &tokenizer, 4);
    let batcher = dataset::GPTDatasetV1Batcher::default();

    type Backend = Wgpu;
    let device = Default::default();
    let batch: dataset::GPTDatasetV1Batch<Backend> =
        batcher.batch((0..3).map(|i| dataset.get(i).unwrap()).collect(), &device);
    println!(
        "{:?}, {:?}",
        batch.input_ids.to_data().to_vec::<i32>(),
        batch.target_ids.to_data().to_vec::<i32>()
    );
}
