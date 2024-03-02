use std::{fs, path::PathBuf, time::Instant};

use clap::Parser;
use quicktok::{BasicTokenizer, Tokenizer};

#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Args {
    data_file_path: PathBuf,

    model_file_path: PathBuf,

    #[clap(long, default_value_t = 1)]
    num_threads: usize,

    #[clap(long, default_value_t = 512)]
    vocab_size: usize,

    #[clap(long)]
    save_vocab: bool,
}

fn main() {
    let args = Args::parse();
    let text = fs::read_to_string(&args.data_file_path).expect("failed to read data file");
    let output_dir_path = args
        .model_file_path
        .parent()
        .expect("failed to get output directory");
    fs::create_dir_all(output_dir_path).expect("failed to create output directory");
    let vocab_file_path = args.model_file_path.with_extension("vocab");
    let start_time = Instant::now();
    let mut tokenizer = BasicTokenizer::new(args.num_threads);
    tokenizer.train(&text, args.vocab_size);
    tokenizer
        .save(
            args.model_file_path.to_str().unwrap(),
            if args.save_vocab {
                Some(vocab_file_path.to_str().unwrap())
            } else {
                None
            },
        )
        .expect("failed to save tokenizer model and vocabulary");
    let duration = start_time.elapsed();
    println!("Training took {:.2} seconds", duration.as_secs_f32());
}
