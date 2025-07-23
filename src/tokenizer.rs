use std::collections::{BTreeSet, HashMap};

use anyhow::{Result, anyhow};
use regex::Regex;
use tiktoken_rs::{CoreBPE, r50k_base};

pub trait ITokenizer {
    fn encode(&self, text: &str) -> Vec<u32>;
    fn decode(&self, ids: &[u32]) -> Result<String>;
}

pub struct SimpleTokenizerV2 {
    str_to_id: HashMap<String, u32>,
    id_to_str: HashMap<u32, String>,
}

impl SimpleTokenizerV2 {
    pub fn new(vocab: &HashMap<String, u32>) -> Self {
        Self {
            str_to_id: vocab.clone(),
            id_to_str: vocab.iter().map(|(k, v)| (*v, k.clone())).collect(),
        }
    }

    pub fn fetch_vocab() -> HashMap<String, u32> {
        let path = "assets/the-verdict.txt";
        let content = std::fs::read_to_string(path).unwrap();
        let mut tokenized = SimpleTokenizerV2::tokenize(&content);

        tokenized.extend(["<|endoftext|>".to_string(), "<|unk|>".to_string()]);

        assert_eq!(tokenized.len(), 4690 + 2);

        let tokens = BTreeSet::from_iter(tokenized);
        let vocab: HashMap<String, u32> = HashMap::from_iter(
            tokens
                .iter()
                .enumerate()
                .map(|(idx, token)| (token.clone(), idx as u32)),
        );

        assert_eq!(vocab.len(), 1132);
        assert_eq!(vocab.get("Among"), Some(&15));

        vocab
    }

    pub fn tokenize(content: &str) -> Vec<String> {
        let re = Regex::new(r#"([,.:;?_!"()\']|--|\s)"#).unwrap();

        let mut parts = Vec::new();

        let mut last_idx = 0;
        for cap in re.find_iter(&content) {
            parts.push(&content[last_idx..cap.start()]);
            parts.push(cap.as_str().trim());
            last_idx = cap.end();
        }

        parts
            .into_iter()
            .filter(|s| !s.is_empty())
            .map(|s| s.to_string())
            .collect()
    }
}

impl ITokenizer for SimpleTokenizerV2 {
    fn encode(&self, text: &str) -> Vec<u32> {
        Self::tokenize(text)
            .iter()
            .map(|token| {
                *self
                    .str_to_id
                    .get(token)
                    .unwrap_or(self.str_to_id.get("<|unk|>").unwrap())
            })
            .collect()
    }

    fn decode(&self, ids: &[u32]) -> Result<String> {
        let strs = ids
            .iter()
            .map(|id| self.id_to_str.get(id))
            .collect::<Vec<_>>();
        if strs.iter().any(Option::is_none) {
            return Err(anyhow!("some of the ids not in vocab"));
        }

        let mut joined = strs
            .iter()
            .map(|s| s.unwrap().clone())
            .collect::<Vec<_>>()
            .join(" ");

        let re = Regex::new(r#"\s+([,.?!"()\'])"#).unwrap();
        while let Some(cap) = re.find(&joined) {
            let b = joined.replace(cap.as_str(), cap.as_str().trim_start());
            joined = b;
        }

        Ok(joined)
    }
}

pub struct BpeTokenizer {
    bpe: CoreBPE,
}

impl BpeTokenizer {
    pub fn new() -> Self {
        Self {
            bpe: r50k_base().unwrap(),
        }
    }
}

impl ITokenizer for BpeTokenizer {
    fn encode(&self, text: &str) -> Vec<u32> {
        self.bpe.encode_with_special_tokens(text)
    }

    fn decode(&self, ids: &[u32]) -> Result<String> {
        self.bpe.decode(Vec::from(ids))
    }
}
