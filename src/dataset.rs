use burn::{
    data::{dataloader::batcher::Batcher, dataset::Dataset},
    prelude::*,
};

use crate::tokenizer::ITokenizer;

/// GPT Dataset V1
///
/// M: max_length
pub struct GPTDatasetV1<const M: usize> {
    input_ids: Vec<[u32; M]>,
    target_ids: Vec<[u32; M]>,
}

impl<const M: usize> GPTDatasetV1<M> {
    pub fn new<T: ITokenizer>(text: &str, tokenizer: &T, stride: usize) -> Self {
        let token_ids = tokenizer.encode(text);
        assert!(
            token_ids.len() > M,
            "number of tokenized inputs must at least be equal to max_length+1"
        );

        let mut input_ids = vec![];
        let mut target_ids = vec![];
        for i in (0..token_ids.len() - M).step_by(stride) {
            let mut input_chunk = [0; M];
            let mut target_chunk = [0; M];
            input_chunk.copy_from_slice(&token_ids[i..i + M]);
            target_chunk.copy_from_slice(&token_ids[i + 1..i + M + 1]);
            input_ids.push(input_chunk);
            target_ids.push(target_chunk);
        }

        Self {
            input_ids,
            target_ids,
        }
    }
}

type GPTDatasetV1Item<const M: usize> = ([u32; M], [u32; M]);

impl<const M: usize> Dataset<GPTDatasetV1Item<M>> for GPTDatasetV1<M> {
    fn len(&self) -> usize {
        self.input_ids.len()
    }

    fn get(&self, index: usize) -> Option<GPTDatasetV1Item<M>> {
        Some((self.input_ids[index], self.target_ids[index]))
    }
}

#[derive(Clone, Default)]
pub struct GPTDatasetV1Batcher {}

#[derive(Clone, Debug)]
pub struct GPTDatasetV1Batch<B: Backend> {
    pub input_ids: Tensor<B, 2, Int>,
    pub target_ids: Tensor<B, 2, Int>,
}

impl<B: Backend, const M: usize> Batcher<B, GPTDatasetV1Item<M>, GPTDatasetV1Batch<B>>
    for GPTDatasetV1Batcher
{
    fn batch(&self, items: Vec<GPTDatasetV1Item<M>>, device: &B::Device) -> GPTDatasetV1Batch<B> {
        let input_chunks = items
            .iter()
            .map(|item| Tensor::<B, 1, Int>::from_ints(item.0, device))
            .collect::<Vec<_>>();
        let target_chunks = items
            .iter()
            .map(|item| Tensor::<B, 1, Int>::from_ints(item.1, device))
            .collect::<Vec<_>>();

        let input_ids = Tensor::stack(input_chunks, 0);
        let target_ids = Tensor::stack(target_chunks, 0);
        GPTDatasetV1Batch {
            input_ids,
            target_ids,
        }
    }
}
