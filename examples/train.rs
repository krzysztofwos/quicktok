use std::{fs, path::Path, time::Instant};

use quicktok::{BasicTokenizer, Tokenizer};

fn main() {
    let data_dir_path = Path::new("data");
    let data_file_path = data_dir_path.join("taylorswift.txt");
    let out_dir_path = Path::new("models");
    let text = fs::read_to_string(&data_file_path).expect("failed to read data file");
    fs::create_dir_all(&out_dir_path).expect("failed to create output directory");
    let model_file_path = out_dir_path.join("taylorswift.model");
    let vocab_file_path = out_dir_path.join("taylorswift.vocab");
    let num_threads = 1;
    let start_time = Instant::now();
    let mut tokenizer = BasicTokenizer::new(num_threads);
    tokenizer.train(&text, 512);
    tokenizer
        .save(
            model_file_path.to_str().unwrap(),
            Some(vocab_file_path.to_str().unwrap()),
        )
        .expect("failed to save tokenizer model and vocabulary");
    let duration = start_time.elapsed();
    println!("Training took {:.2} seconds", duration.as_secs_f32());
}
