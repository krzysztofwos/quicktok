import time
from pathlib import Path

from quicktok import BasicTokenizer

data_dir_path = Path("data")
data_file_path = data_dir_path / "taylorswift.txt"
out_dir_path = Path("models")

text = data_file_path.read_text(encoding="utf-8")
out_dir_path.mkdir(exist_ok=True)
model_file_path = str(out_dir_path / "taylorswift.model")
vocab_file_path = str(out_dir_path / "taylorswift.vocab")
t0 = time.time()
tokenizer = BasicTokenizer()
tokenizer.train(text, 512)
tokenizer.save(model_file_path, vocab_file_path)
t1 = time.time()
print(f"Training took {t1 - t0:.2f} seconds")
