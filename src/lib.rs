use std::{
    collections::HashMap,
    fs::File,
    io::{BufRead, BufReader, BufWriter, Error, ErrorKind, Result, Write},
    sync::Mutex,
};

use indexmap::IndexMap;
use pyo3::prelude::*;
use rayon::prelude::*;

static VERSION: &str = "quicktok v1";

fn populate_counts(counts: &mut HashMap<(usize, usize), usize>, ids: &[usize]) {
    counts.clear();
    for pair in ids.windows(2) {
        *counts.entry((pair[0], pair[1])).or_insert(0) += 1;
    }
}

fn get_most_common_pair(
    counts: &mut HashMap<(usize, usize), usize>,
    ids: &[usize],
) -> (usize, usize) {
    populate_counts(counts, ids);
    counts
        .iter()
        .max_by_key(|&(_, &count)| count)
        .map(|(&pair, _)| pair)
        .unwrap()
}

fn parallel_get_most_common_pair(ids: &[usize], num_chunks: usize) -> (usize, usize) {
    let chunk_size = ids.len() / num_chunks;
    let counts = Mutex::new(HashMap::new());

    ids.par_chunks(chunk_size)
        .enumerate()
        .for_each(|(i, chunk)| {
            let mut local_counts = HashMap::new();

            for pair in chunk.windows(2) {
                *local_counts.entry((pair[0], pair[1])).or_insert(0) += 1;
            }

            if i < num_chunks - 1 {
                let boundary_pair = (*chunk.last().unwrap(), ids[i * chunk_size + chunk.len()]);
                *local_counts.entry(boundary_pair).or_insert(0) += 1;
            }

            let mut counts = counts.lock().unwrap();

            for (pair, count) in local_counts {
                *counts.entry(pair).or_insert(0) += count;
            }
        });

    let counts = counts.into_inner().unwrap();
    let (pair, _) = counts.iter().max_by_key(|&(_, &count)| count).unwrap();
    *pair
}

fn merge(ids: &mut Vec<usize>, pair: (usize, usize), new_id: usize) {
    let mut i = 0;

    while i < ids.len() {
        if ids[i] == pair.0 && i < ids.len() - 1 && ids[i + 1] == pair.1 {
            ids[i] = new_id;
            ids[i + 1] = usize::MAX;
            i += 2;
        } else {
            i += 1;
        }
    }

    ids.retain(|&id| id != usize::MAX);
}

fn render_token(token: &[usize]) -> String {
    let token = token.iter().map(|idx| *idx as u8).collect::<Vec<u8>>();
    let string = String::from_utf8_lossy(&token);
    string
        .chars()
        .map(|c| {
            if c.is_control() {
                format!("\\u{:04x}", c as u32)
            } else {
                c.to_string()
            }
        })
        .collect()
}

pub trait Tokenizer {
    fn train(&mut self, text: &str, vocab_size: usize);
    fn encode(&self, text: &str) -> Vec<usize>;
    fn decode(&self, ids: Vec<usize>) -> Result<String>;
    fn load(&mut self, model_file_path: &str) -> Result<()>;
    fn save(&self, model_file_path: &str, vocab_file_path: Option<&str>) -> Result<()>;
}

#[pyclass]
pub struct BasicTokenizer {
    merges: IndexMap<(usize, usize), usize>,
    pattern: String,
    special_tokens: IndexMap<String, usize>,
    vocab: HashMap<usize, Vec<usize>>,
    num_threads: usize,
}

#[pymethods]
impl BasicTokenizer {
    #[new]
    #[pyo3(signature = (num_threads=1))]
    pub fn new(num_threads: usize) -> Self {
        BasicTokenizer {
            merges: IndexMap::new(),
            pattern: "".to_string(),
            special_tokens: IndexMap::new(),
            vocab: HashMap::new(),
            num_threads,
        }
    }

    fn build_vocab(&mut self) {
        for idx in 0..256 {
            self.vocab.insert(idx, vec![idx]);
        }

        for (&(p0, p1), &idx) in &self.merges {
            if let (Some(token0), Some(token1)) = (self.vocab.get(&p0), self.vocab.get(&p1)) {
                let mut new_token = token0.clone();
                new_token.extend_from_slice(token1);
                self.vocab.insert(idx, new_token);
            }
        }

        for (special, &idx) in &self.special_tokens {
            self.vocab.insert(
                idx,
                special.as_bytes().iter().map(|b| *b as usize).collect(),
            );
        }
    }

    fn save_model(&self, file_path: &str) -> Result<()> {
        let file = File::create(file_path)?;
        let mut writer = BufWriter::new(file);
        writeln!(writer, "{}", VERSION)?;
        writeln!(writer, "{}", self.pattern)?;
        writeln!(writer, "{}", self.special_tokens.len())?;

        for (special, &idx) in &self.special_tokens {
            writeln!(writer, "{} {}", special, idx)?;
        }

        for (&(idx1, idx2), _) in &self.merges {
            writeln!(writer, "{} {}", idx1, idx2)?;
        }

        Ok(())
    }

    fn save_vocab(&self, vocab_file_path: &str) -> Result<()> {
        let inverted_merges: HashMap<usize, &(usize, usize)> =
            self.merges.iter().map(|(pair, &idx)| (idx, pair)).collect();
        let file = File::create(vocab_file_path)?;
        let mut writer = BufWriter::new(file);
        let mut sorted_vocab = self
            .vocab
            .iter()
            .map(|(idx, token)| (*idx, token))
            .collect::<Vec<(usize, &Vec<usize>)>>();
        sorted_vocab.sort();

        for (idx, token) in &sorted_vocab {
            let s = render_token(token);

            if let Some((idx1, idx2)) = inverted_merges.get(idx) {
                let s1 = render_token(self.vocab.get(idx1).unwrap());
                let s2 = render_token(self.vocab.get(idx2).unwrap());
                writeln!(writer, "[{s1}][{s2}] -> [{s}] {idx}")?;
            } else {
                writeln!(writer, "[{s}] {idx}")?;
            }
        }

        Ok(())
    }

    // https://pyo3.rs/main/trait_bounds.html#implementation-of-the-trait-bounds-for-the-python-class

    fn train(&mut self, text: &str, vocab_size: usize) {
        Tokenizer::train(self, text, vocab_size);
    }

    fn encode(&self, text: &str) -> Vec<usize> {
        Tokenizer::encode(self, text)
    }

    fn decode(&self, ids: Vec<usize>) -> Result<String> {
        Tokenizer::decode(self, ids)
    }

    fn load(&mut self, model_file_path: &str) -> Result<()> {
        Tokenizer::load(self, model_file_path)
    }

    fn save(&self, model_file_path: &str, vocab_file_path: Option<&str>) -> Result<()> {
        Tokenizer::save(self, model_file_path, vocab_file_path)
    }
}

impl Tokenizer for BasicTokenizer {
    fn train(&mut self, text: &str, vocab_size: usize) {
        assert!(vocab_size >= 256, "vocab_size must be at least 256");
        let num_merges = vocab_size - 256;
        let mut ids: Vec<usize> = text.as_bytes().iter().map(|b| *b as usize).collect();
        self.vocab = (0..=255_usize).map(|idx| (idx, vec![idx])).collect();
        let mut counts = HashMap::new();

        for i in 0..num_merges {
            let pair = if self.num_threads > 1 {
                parallel_get_most_common_pair(&ids, self.num_threads)
            } else {
                get_most_common_pair(&mut counts, &ids)
            };
            let idx = 256 + i;
            merge(&mut ids, pair, idx);
            self.merges.insert(pair, idx);
            let new_token = [self.vocab[&pair.0].clone(), self.vocab[&pair.1].clone()].concat();
            self.vocab.insert(idx, new_token.clone());
        }
    }

    fn encode(&self, text: &str) -> Vec<usize> {
        let mut ids: Vec<usize> = text.as_bytes().iter().map(|&b| b as usize).collect();
        let mut counts = HashMap::new();

        while ids.len() >= 2 {
            populate_counts(&mut counts, &ids);
            let mut min_merge_index = usize::MAX;
            let mut best_pair = (0, 0);

            for pair in counts.keys() {
                if let Some(&merge_index) = self.merges.get(pair) {
                    if merge_index < min_merge_index {
                        min_merge_index = merge_index;
                        best_pair = *pair;
                    }
                }
            }

            if min_merge_index == usize::MAX {
                break;
            }

            merge(&mut ids, best_pair, min_merge_index);
        }

        ids
    }

    fn decode(&self, ids: Vec<usize>) -> Result<String> {
        let mut result = String::new();

        for idx in &ids {
            let token = self.vocab.get(idx).ok_or(Error::new(
                ErrorKind::InvalidInput,
                format!("invalid token ID: {}", idx),
            ))?;
            result += &render_token(token);
        }

        Ok(result)
    }

    fn load(&mut self, model_file_path: &str) -> Result<()> {
        let file = File::open(model_file_path)?;
        let reader = BufReader::new(file);
        let mut lines = reader.lines();
        let version = lines
            .next()
            .ok_or(Error::new(ErrorKind::InvalidData, "missing version"))??;

        if version != VERSION {
            return Err(Error::new(ErrorKind::InvalidData, "invalid version"));
        }

        self.pattern = lines
            .next()
            .ok_or(Error::new(ErrorKind::InvalidData, "missing pattern"))??;
        let num_special: usize = lines
            .next()
            .ok_or(Error::new(
                ErrorKind::InvalidData,
                "missing number of special tokens",
            ))??
            .parse()
            .map_err(|e| Error::new(ErrorKind::InvalidData, e))?;

        for _ in 0..num_special {
            let line = lines.next().ok_or(Error::new(
                ErrorKind::InvalidData,
                "not enough special tokens",
            ))??;
            let parts: Vec<&str> = line.split_whitespace().collect();

            if parts.len() != 2 {
                return Err(Error::new(
                    ErrorKind::InvalidData,
                    "invalid special token entry",
                ));
            }

            let special = parts[0]
                .parse()
                .map_err(|e| Error::new(ErrorKind::InvalidData, e))?;
            let special_idx = parts[1]
                .parse()
                .map_err(|e| Error::new(ErrorKind::InvalidData, e))?;
            self.special_tokens.insert(special, special_idx);
        }

        let mut idx = 256;

        for line in lines {
            let line = line?;
            let parts: Vec<&str> = line.split_whitespace().collect();

            if parts.len() != 2 {
                return Err(Error::new(ErrorKind::InvalidData, "invalid merge entry"));
            }

            let idx1 = parts[0]
                .parse()
                .map_err(|e| Error::new(ErrorKind::InvalidData, e))?;
            let idx2 = parts[1]
                .parse()
                .map_err(|e| Error::new(ErrorKind::InvalidData, e))?;
            self.merges.insert((idx1, idx2), idx);
            idx += 1;
        }

        self.build_vocab();
        Ok(())
    }

    fn save(&self, model_file_path: &str, vocab_file_path: Option<&str>) -> Result<()> {
        self.save_model(model_file_path)?;

        if let Some(vocab_file_path) = vocab_file_path {
            self.save_vocab(vocab_file_path)?;
        }

        Ok(())
    }
}

#[pymodule]
fn quicktok(_py: Python, module: &PyModule) -> PyResult<()> {
    module.add_class::<BasicTokenizer>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use tempfile::NamedTempFile;

    use super::*;

    #[test]
    fn test_wikipedia_example() {
        let mut tokenizer = BasicTokenizer::new(1);
        let text = "aaabdaaabac";
        let expected_merges = 3;
        tokenizer.train(text, 256 + expected_merges);
        let ids = tokenizer.encode(text);
        let expected_ids = vec![258, 100, 258, 97, 99];
        assert_eq!(
            ids, expected_ids,
            "the encoded token IDs do not match the expected values"
        );
        let decoded_text = tokenizer.decode(ids).unwrap();
        assert_eq!(
            decoded_text, text,
            "decoded text does not match the original text"
        );
    }

    #[test]
    fn test_save_load() {
        let text = r#"
<|endoftext|>The llama (/ˈlɑːmə/; Spanish pronunciation: [ˈʎama] or [ˈʝama]) (Lama glama) is a domesticated South American camelid, widely used as a meat and pack animal by Andean cultures since the pre-Columbian era.
Llamas are social animals and live with others as a herd. Their wool is soft and contains only a small amount of lanolin.[2] Llamas can learn simple tasks after a few repetitions. When using a pack, they can carry about 25 to 30% of their body weight for 8 to 13 km (5–8 miles).[3] The name llama (in the past also spelled "lama" or "glama") was adopted by European settlers from native Peruvians.[4]
The ancestors of llamas are thought to have originated from the Great Plains of North America about 40 million years ago, and subsequently migrated to South America about three million years ago during the Great American Interchange. By the end of the last ice age (10,000–12,000 years ago), camelids were extinct in North America.[3] As of 2007, there were over seven million llamas and alpacas in South America and over 158,000 llamas and 100,000 alpacas, descended from progenitors imported late in the 20th century, in the United States and Canada.[5]
<|fim_prefix|>In Aymara mythology, llamas are important beings. The Heavenly Llama is said to drink water from the ocean and urinates as it rains.[6] According to Aymara eschatology,<|fim_suffix|> where they come from at the end of time.[6]<|fim_middle|> llamas will return to the water springs and ponds<|endofprompt|>
"#;
        let mut tokenizer = BasicTokenizer::new(1);
        tokenizer.train(text, 256 + 64);
        let original_ids = tokenizer.encode(text);
        let model_file = NamedTempFile::new().expect("failed to create temporary model file");
        let model_file_path = model_file.path().to_str().unwrap();
        let vocab_file = NamedTempFile::new().expect("failed to create temporary vocab file");
        let vocab_file_path = vocab_file.path().to_str().unwrap();
        tokenizer
            .save(model_file_path, Some(vocab_file_path))
            .expect("failed to save the tokenizer model");
        let mut loaded_tokenizer = BasicTokenizer::new(1);
        loaded_tokenizer
            .load(&model_file_path)
            .expect("failed to load the tokenizer model");
        let loaded_ids = loaded_tokenizer.encode(text);
        assert_eq!(
            original_ids, loaded_ids,
            "token IDs from the original and loaded tokenizer do not match"
        );
    }
}
