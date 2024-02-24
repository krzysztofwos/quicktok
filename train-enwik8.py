import time
import zipfile
from pathlib import Path

import requests
import typer
from quicktok import BasicTokenizer

ENWIK8_URL = "http://mattmahoney.net/dc/enwik8.zip"


def main(
    data_dir_path: Path = Path("data"),
    model_file_path: Path = Path("models/enwik8.model"),
    vocab_file_path: Path | None = None,
    vocab_size: int = 2048,
    num_threads: int = 1,
):
    zip_file_path = data_dir_path / "enwik8.zip"
    data_file_path = data_dir_path / "enwik8"

    if not zip_file_path.exists():
        response = requests.get(ENWIK8_URL)
        zip_file_path.write_bytes(response.content)

    if not data_file_path.exists():
        with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
            zip_ref.extractall(data_dir_path)

    text = data_file_path.read_text(encoding="utf-8")
    model_file_path.parent.mkdir(parents=True, exist_ok=True)
    model_file_path = str(model_file_path)
    vocab_file_path = str(vocab_file_path) if vocab_file_path else None
    t0 = time.time()
    tokenizer = BasicTokenizer(num_threads)
    tokenizer.train(text, vocab_size)
    tokenizer.save(model_file_path, vocab_file_path)
    t1 = time.time()
    print(f"Training took {t1 - t0:.2f} seconds")


if __name__ == "__main__":
    typer.run(main)
