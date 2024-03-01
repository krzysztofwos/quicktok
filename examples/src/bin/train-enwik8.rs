use std::{
    fs::{self, File},
    io::BufReader,
    path::PathBuf,
    time::Instant,
};

use clap::Parser;
use quicktok::{BasicTokenizer, Tokenizer};
use zip::ZipArchive;

#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Args {
    #[clap(long, default_value = "data")]
    data_dir_path: PathBuf,

    #[clap(long, default_value = "models/enwik8.model")]
    model_file_path: PathBuf,

    #[clap(long, default_value_t = 1)]
    num_threads: usize,

    #[clap(long, default_value_t = 2048)]
    vocab_size: usize,

    #[clap(long)]
    save_vocab: bool,

    #[clap(long, default_value = "http://mattmahoney.net/dc/enwik8.zip")]
    enwik8_url: String,
}

fn main() {
    let args = Args::parse();
    let zip_file_path = args.data_dir_path.join("enwik8.zip");
    let data_file_path = args.data_dir_path.join("enwik8");

    if !zip_file_path.exists() {
        let response =
            reqwest::blocking::get(args.enwik8_url).expect("failed to download enwik8 Zip file");
        fs::write(
            &zip_file_path,
            response.bytes().expect("failed to read response"),
        )
        .expect("failed to write zip file");
    }

    if !data_file_path.exists() {
        let zip_file = File::open(&zip_file_path).expect("failed to open Zip file");
        let mut archive =
            ZipArchive::new(BufReader::new(zip_file)).expect("failed to read Zip file");
        archive
            .extract(&args.data_dir_path)
            .expect("failed to extract Zip file");
    }

    let text = fs::read_to_string(&data_file_path).expect("failed to read data file");
    let output_dir_path = args.model_file_path.parent().unwrap();
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
