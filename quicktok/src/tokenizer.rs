pub mod basic;

use std::io::Result;

pub trait Tokenizer {
    fn train(&mut self, text: &str, vocab_size: usize);
    fn encode(&self, text: &str) -> Vec<usize>;
    fn decode(&self, ids: Vec<usize>) -> Result<String>;
    fn load(&mut self, model_file_path: &str) -> Result<()>;
    fn save(&self, model_file_path: &str, vocab_file_path: Option<&str>) -> Result<()>;
}
