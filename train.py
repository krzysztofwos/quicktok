#!/usr/bin/env python

import time
from pathlib import Path

import typer

from quicktok import BasicTokenizer


def main(
    data_file_path: Path,
    model_file_path: Path = Path("models/taylorswift.model"),
    num_threads: int = 1,
    vocab_size: int = 512,
    save_vocab: bool = False,
):
    data_dir_path = Path("data")
    data_file_path = data_dir_path / "taylorswift.txt"

    text = data_file_path.read_text(encoding="utf-8")
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
