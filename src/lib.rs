pub mod attention;
mod backend;
pub mod dataset;
pub mod gpt;
pub mod tokenizer;
mod train;

pub use backend::{Backend, TrainBackend};
pub use train::{MAX_LENGTH, TrainingConfig, train};

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {}
}
