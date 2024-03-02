#!/usr/bin/env python

import time
import zipfile
from pathlib import Path

import requests
import typer

from quicktok import BasicTokenizer


def main(
    data_dir_path: Path = Path("data"),
    model_file_path: Path = Path("models/enwik8.model"),
    max_bytes: int | None = None,
    num_threads: int = 1,
    vocab_size: int = 2048,
    save_vocab: bool = False,
    enwik8_url: str = "http://mattmahoney.net/dc/enwik8.zip",
):
    zip_file_path = data_dir_path / "enwik8.zip"
    data_file_path = data_dir_path / "enwik8"

    if not zip_file_path.exists():
        response = requests.get(enwik8_url)
        zip_file_path.write_bytes(response.content)

    if not data_file_path.exists():
        with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
            zip_ref.extractall(data_dir_path)

    data = data_file_path.read_bytes()

    if max_bytes:
        data = data[:max_bytes]
        print(f"Training on the first {len(data)} bytes")

    text = data.decode("utf-8", errors="replace")
    model_file_path.parent.mkdir(parents=True, exist_ok=True)
    vocab_file_path = str(model_file_path.with_suffix(".vocab")) if save_vocab else None
    t0 = time.time()
    tokenizer = BasicTokenizer(num_threads)
    tokenizer.train(text, vocab_size)
    tokenizer.save(str(model_file_path), vocab_file_path)
    t1 = time.time()
    print(f"Training took {t1 - t0:.2f} seconds")


if __name__ == "__main__":
    typer.run(main)
