use burn::{
    config::Config,
    data::{dataloader::DataLoaderBuilder, dataset::Dataset},
    module::Module,
    optim::AdamConfig,
    record::CompactRecorder,
    tensor::backend::AutodiffBackend,
    train::{LearnerBuilder, metric::LossMetric},
};

use crate::{
    dataset::{GPTDatasetV1, GPTDatasetV1Batcher},
    gpt::{GPTModel, GPTModelConfig},
    tokenizer,
};

#[derive(Config)]
pub struct TrainingConfig {
    pub model: GPTModelConfig,
    #[config(default = 0.1)]
    pub valid_ratio: f64,
    pub optimizer: AdamConfig,
    #[config(default = 10)]
    pub num_epochs: usize,
    #[config(default = 2)]
    pub batch_size: usize,
    #[config(default = 1)]
    pub num_workers: usize,
    #[config(default = 123)]
    pub seed: u64,
    #[config(default = 0.0004)]
    pub learning_rate: f64,
}

fn create_artifact_dir(artifact_dir: &str) {
    // Remove existing artifacts before to get an accurate learner summary
    std::fs::remove_dir_all(artifact_dir).ok();
    std::fs::create_dir_all(artifact_dir).ok();
}

pub fn train<B: AutodiffBackend>(
    text: &str,
    artifact_dir: &str,
    config: TrainingConfig,
    device: B::Device,
) -> GPTModel<B> {
    assert_eq!(config.model.context_length, 256);

    create_artifact_dir(artifact_dir);
    config
        .save(format!("{artifact_dir}/config.json"))
        .expect("Config should be saved successfully");

    B::seed(config.seed);

    let batcher = GPTDatasetV1Batcher::default();

    let tokenizer = tokenizer::BpeTokenizer::new();
    let (train, valid) =
        GPTDatasetV1::<256>::new(text, &tokenizer, 256).split_train_valid(config.valid_ratio);

    println!("Train dataset size: {}", train.len());
    println!("Valid dataset size: {}", valid.len());

    let dataloader_train = DataLoaderBuilder::new(batcher.clone())
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(train);

    let dataloader_valid = DataLoaderBuilder::new(batcher)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(valid);

    let learner = LearnerBuilder::new(artifact_dir)
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .devices(vec![device.clone()])
        .num_epochs(config.num_epochs)
        .summary()
        .build(
            config.model.init::<B>(&device),
            config.optimizer.init(),
            config.learning_rate,
        );

    let model_trained = learner.fit(dataloader_train, dataloader_valid);

    model_trained
        .clone()
        .save_file(format!("{artifact_dir}/model"), &CompactRecorder::new())
        .expect("Trained model should be saved successfully");

    model_trained
}
