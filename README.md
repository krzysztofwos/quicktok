<p align="center" style="font-size: 48px; font-weight: bold;">Work in Progress</p>

# quicktok

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
python train-enwik8.py --vocab-file-path models/enwik8.vocab --num-threads 8
```

## Rust example

```bash
cargo run --release --example train
```

## Performance

Performance measured on AMD Ryzen 9 3950X 16-core processor using the `train-enwik8.py` script.

| Threads | Training Time (seconds) | Speedup |
| ------: | ----------------------: | ------: |
|       1 |                 2673.07 |   1.00x |
|       2 |                 1411.15 |   1.89x |
|       4 |                  869.31 |   3.08x |
|       8 |                  704.82 |   3.79x |

## Development

The repository contains a [Dev Container](https://containers.dev/overview) configuration compatible with Visual Studio Code, JetBrains RustRover, and other [tools](https://containers.dev/supporting).

To use it in VS Code, install the [Dev Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) extension, open the repository, and click "Reopen in Container" in the notification that appears in the bottom-right corner of the window.

The Dev Container contains all the necessary tools and dependencies. It can also be used with [GitHub Codespaces](https://github.com/features/codespaces), allowing development directly in the browser.

## License

MIT
