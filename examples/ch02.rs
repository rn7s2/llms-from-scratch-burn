use burn::data::dataloader::{Dataset, batcher::Batcher};
use burn::nn::{Embedding, EmbeddingConfig};
use burn::prelude::*;
use llms_from_scratch_burn::{
    dataset,
    tokenizer::{self, ITokenizer},
};

type Backend = burn::backend::Wgpu;

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
    const MAX_LENGTH: usize = 4;
    const STRIDE: usize = 4;
    const VOCAB_SIZE: usize = 50257;
    const OUT_DIM: usize = 256;

    let path = "assets/the-verdict.txt";
    let text = std::fs::read_to_string(path).unwrap();

    let dataset = dataset::GPTDatasetV1::<MAX_LENGTH>::new(&text, &tokenizer, STRIDE);
    let batcher = dataset::GPTDatasetV1Batcher::default();

    let device = Default::default();
    let batch: dataset::GPTDatasetV1Batch<Backend> =
        batcher.batch((0..8).map(|i| dataset.get(i).unwrap()).collect(), &device);
    println!("{}\n{}", batch.input_ids, batch.target_ids);

    println!();

    // 2.8. Encoding word positions
    let token_embedding_layer: Embedding<Backend> =
        EmbeddingConfig::new(VOCAB_SIZE, OUT_DIM).init(&device);
    let token_embeddings = token_embedding_layer.forward(batch.input_ids);
    println!("token embeddings: {}", token_embeddings.clone());

    let pos_embedding_layer: Embedding<Backend> =
        EmbeddingConfig::new(MAX_LENGTH, OUT_DIM).init(&device);
    let pos_embeddings =
        pos_embedding_layer.forward(Tensor::arange(0..MAX_LENGTH as i64, &device).unsqueeze());
    println!("pos embeddings: {}", pos_embeddings.clone());

    let input_embeddings =
        token_embeddings.clone() + pos_embeddings.expand(token_embeddings.shape());
    println!("input embeddings: {}", input_embeddings);
}
