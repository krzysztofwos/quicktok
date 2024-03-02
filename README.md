<p align="center" style="font-size: 48px; font-weight: bold;">Work in Progress</p>

# quicktok &emsp; [![Python Build Status]][Python Actions] [![Rust Build Status]][Rust Actions] [![Latest Version PyPI]][PyPI] [![Latest Version crates.io]][crates.io]

[Python Build Status]: https://img.shields.io/github/actions/workflow/status/krzysztofwos/quicktok/python-ci.yml?branch=main&label=Python
[Python Actions]: https://github.com/krzysztofwos/quicktok/actions/workflows/python-ci.yml?query=branch%3Amain
[Rust Build Status]: https://img.shields.io/github/actions/workflow/status/krzysztofwos/quicktok/rust-ci.yml?branch=main&label=Rust
[Rust Actions]: https://github.com/krzysztofwos/quicktok/actions/workflows/rust-ci.yml?query=branch%3Amain
[Latest Version PyPI]: https://img.shields.io/pypi/v/quicktok?label=PyPI
[PyPI]: https://pypi.org/project/quicktok/
[Latest Version crates.io]: https://img.shields.io/crates/v/quicktok
[crates.io]: https://crates.io/crates/quicktok

Rust implementation of Andrej Karpathy's [minbpe](https://github.com/karpathy/minbpe).

## Usage

Install the package:

```bash
pip install quicktok
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
./train-enwik8.py --num-threads 4 --save-vocab
```

## Rust examples

### Basic

```bash
cargo run --release --package quicktok-examples --bin train -- \
    data/taylorswift.txt \
    models/taylorswift.model \
    --save-vocab
```

### Multi-threaded

```bash
cargo run --release --package quicktok-examples --bin train-enwik8 -- --num-threads 4 --save-vocab
```

## Performance

Performance measured on AMD Ryzen 9 3950X 16-core processor using the `train-enwik8.py` script.

| Threads | Training Time (seconds) | Speedup |
| ------: | ----------------------: | ------: |
|       1 |                 2673.07 |   1.00x |
|       2 |                 1411.15 |   1.89x |
|       4 |                  869.31 |   3.08x |
|       8 |                  704.82 |   3.79x |

minbpe's BasicTokenizer takes 27002.18 seconds to train on the same dataset, making quicktok 38x faster.

## Development

The repository contains a [Dev Container](https://containers.dev/overview) configuration compatible with Visual Studio Code, JetBrains RustRover, and other [tools](https://containers.dev/supporting).

To use it in VS Code, install the [Dev Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) extension, open the repository, and click "Reopen in Container" in the notification that appears in the bottom-right corner of the window.

The Dev Container contains all the necessary tools and dependencies. It can also be used with [GitHub Codespaces](https://github.com/features/codespaces), allowing development directly in the browser.

## License

MIT
