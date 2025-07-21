pub mod attention;
mod backend;
pub mod dataset;
pub mod gpt;
pub mod tokenizer;

pub use backend::{Backend, InferenceBackend};

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {}
}
