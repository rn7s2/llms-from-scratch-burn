use burn::{
    nn::loss::CrossEntropyLossConfig,
    optim::{AdamConfig, decay::WeightDecayConfig},
    tensor::{Int, Tensor, activation::softmax},
};
use llms_from_scratch_burn::{
    MAX_LENGTH, TrainBackend, TrainingConfig,
    gpt::{GPTModelConfig, generate_text_simple, text_to_token_ids, token_ids_to_text},
    tokenizer::{self, ITokenizer},
};

fn main() {
    let device = Default::default();

    // 5.1 Evaluating generative text models
    println!("5.1 Evaluating generative text models");

    // 5.1.1 Using GPT to generate text
    println!("5.1.1 Using GPT to generate text");

    let gpt_config_124m = GPTModelConfig::new(50257, 1024, 768, 12, 12, 0.1, false);
    let model = gpt_config_124m.init::<TrainBackend>(&device);

    let tokenizer = tokenizer::BpeTokenizer::new();

    let start_context = "Every effort moves you";

    let token_ids = generate_text_simple(
        &model,
        text_to_token_ids(start_context, &tokenizer),
        6,
        gpt_config_124m.context_length,
    );
    println!(
        "Output text:\n{:?}",
        token_ids_to_text(token_ids, &tokenizer)
    );

    // 5.1.2 Calculating the text generation loss: cross-entropy and perplexity
    println!("\n5.1.2 Calculating the text generation loss: cross-entropy and perplexity");

    let inputs = Tensor::<TrainBackend, 2, Int>::from([
        [16833, 3626, 6100], // ["every effort moves",
        [40, 1107, 588],     //  "I really like"]
    ]);
    let targets = Tensor::<TrainBackend, 2, Int>::from([
        [3626, 6100, 345],  // [" effort moves you",
        [1107, 588, 11311], //  " really like chocolate"]
    ]);

    let logits = model.forward(inputs);
    let probas = softmax(logits.clone(), 2);
    println!("{}", probas);

    let token_ids = probas.clone().argmax(2);
    println!("Token IDs:\n{}", token_ids);

    println!(
        "Targets batch 1: {:?}",
        token_ids_to_text(targets.clone().slice([0..1, 0..3]), &tokenizer)
    );
    println!(
        "Outputs batch 1: {:?}",
        token_ids_to_text(token_ids.slice([0..1]).squeeze(2), &tokenizer)
    );

    let target_probas = probas
        .gather(2, targets.clone().unsqueeze_dim(2))
        .squeeze::<2>(2);
    println!("Target probabilities:\n{}", target_probas);

    let log_probas = target_probas.flatten::<1>(0, 1).log();
    println!("Log probabilities:\n{}", log_probas);

    let avg_log_probas = log_probas.mean();
    println!("Average log probability: {}", avg_log_probas.into_scalar());

    // Logits have shape (batch_size, num_tokens, vocab_size)
    println!("Logits shape: {:?}", logits.dims());

    // Targets have shape (batch_size, num_tokens)
    println!("Targets shape: {:?}", targets.dims());

    let logits_flat = logits.flatten::<2>(0, 1);
    let targets_flat = targets.flatten::<1>(0, 1);

    println!("Flattened logits: {:?}", logits_flat.dims());
    println!("Flattened targets: {:?}", targets_flat.dims());

    let cross_entropy_loss = CrossEntropyLossConfig::new().init(&device);
    let loss = cross_entropy_loss
        .forward(logits_flat, targets_flat)
        .into_scalar();
    println!("Loss: {}", loss);
    println!("Perplexity: {}", loss.exp());

    // 5.1.3 Calculating the training and validation set losses
    println!("\n5.1.3 Calculating the training and validation set losses");

    let path = "assets/the-verdict.txt";
    let text = std::fs::read_to_string(path).unwrap();

    let total_characters = text.len();
    let total_tokens = tokenizer.encode(&text).len();

    println!("Characters: {}", total_characters);
    println!("Tokens: {}", total_tokens);

    let optimizer = AdamConfig::new().with_weight_decay(Some(WeightDecayConfig::new(0.1)));
    llms_from_scratch_burn::train::<TrainBackend>(
        &text,
        "artifacts",
        TrainingConfig::new(
            GPTModelConfig::new(50257, MAX_LENGTH, 768, 12, 12, 0.1, false),
            optimizer,
        )
        .with_num_epochs(2000)
        .with_restore_epoch(None),
        device,
    );
}
