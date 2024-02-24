<p align="center" style="font-size: 48px; font-weight: bold;">Work in Progress</p>

# quicktok

Rust implementation of Andrej Karpathy's [minbpe](https://github.com/karpathy/minbpe).

## Usage

Install [Rust](https://www.rust-lang.org/):

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

Install the package:

```bash
pip install .
```

### Basic example

```python
from pathlib import Path

from quicktok import BasicTokenizer

text = Path("data/taylorswift.txt").read_text(encoding="utf-8")
tokenizer = BasicTokenizer()
tokenizer.train(text, 512)
tokenizer.save("basic.model", "basic.vocab")
```

### Multi-threaded example

Install script dependencies:

```bash
pip install -r requirements.txt
```

Run training script:

```bash
python train-enwik8.py --vocab-file-path models/enwik8.vocab --num-threads 8
```

## Performance

Performance measured on AMD Ryzen 9 3950X 16-Core processor using the `train-enwik8.py` script.

| Threads | Training Time (seconds) | Speedup |
| ------: | ----------------------: | ------: |
|       1 |                 2673.07 |   1.00x |
|       2 |                 1411.15 |   1.89x |
|       4 |                  869.31 |   3.08x |
|       8 |                  704.82 |   3.79x |
