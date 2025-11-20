from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
import jax.numpy as jnp
from pathlib import Path
import numpy as np
import sys
import os
""" to run: python -m data_preprocess.tokenizer """

DIR = Path(__file__).parent

try:
        sys.path.append(str(DIR.parent))
        from config import Config as config
        VOCAB_SIZE = config.vocab_size
except ImportError:
        VOCAB_SIZE = 12000
        print("Using default vocab_size: 12000")

class BPETokenizer:
    def __init__(self, vocab_size: int = VOCAB_SIZE):
        self.vocab_size = vocab_size
        self.tokenizer = None
    
    def load_data(self, data_dir: str = None):
        if data_dir is None:
            data_dir = DIR / "data"
        else:
            data_dir = DIR / data_dir
            
        data_dir = Path(data_dir)
        
        with open(data_dir / "train.txt", "r", encoding="utf-8") as f:
            train_text = f.read()
        with open(data_dir / "valid.txt", "r", encoding="utf-8") as f:
            valid_text = f.read()
        with open(data_dir / "test.txt", "r", encoding="utf-8") as f:
            test_text = f.read()
        
        all_text = train_text + valid_text + test_text
        return train_text, valid_text, test_text, all_text
    
    def train_tokenizer(self, all_text: str):
        self.tokenizer = Tokenizer(BPE(unk_token="<unk>"))
        self.tokenizer.pre_tokenizer = Whitespace()
        
        trainer = BpeTrainer(
            vocab_size=self.vocab_size,
            special_tokens=["<pad>", "<unk>", "<bos>", "<eos>"],
            min_frequency=2
        )
        
        self.tokenizer.train_from_iterator([all_text], trainer=trainer)
    
    def encode(self, text: str) -> jnp.ndarray:
        if self.tokenizer is None:
            raise ValueError("Tokenizer not trained.")
        encoded = self.tokenizer.encode(text)
        return jnp.array(encoded.ids, dtype=jnp.int32)
    
    def decode(self, tokens: jnp.ndarray) -> str:
        if self.tokenizer is None:
            raise ValueError("Tokenizer not trained.")
        if hasattr(tokens, 'tolist'):
            tokens = tokens.tolist()
        return self.tokenizer.decode(tokens)
    
    def tokenize_splits(self, train_text: str, valid_text: str, test_text: str):
        train_tokens = self.encode(train_text)
        valid_tokens = self.encode(valid_text)
        test_tokens = self.encode(test_text)
        return train_tokens, valid_tokens, test_tokens
    
    def get_vocab_size(self) -> int:
        if self.tokenizer is None:
            raise ValueError("Tokenizer not trained.")
        return self.tokenizer.get_vocab_size()
    
    def save_tokenizer(self, save_path: str = None):
        if self.tokenizer is None:
            raise ValueError("Tokenizer not trained.")
        if save_path is None:
            save_path = DIR / "outputs" / "tokenizer"
        else:
            save_path = DIR / save_path
            
        Path(save_path).mkdir(parents=True, exist_ok=True)
        self.tokenizer.save(f"{save_path}/bpe_tokenizer.json")
    
    def save_data(self, train_tokens: jnp.ndarray, valid_tokens: jnp.ndarray, test_tokens: jnp.ndarray):
        save_dir = DIR / "outputs" / "tokenized_data"
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        np.save(f"{save_dir}/train_tokens.npy", np.array(train_tokens))
        np.save(f"{save_dir}/valid_tokens.npy", np.array(valid_tokens))
        np.save(f"{save_dir}/test_tokens.npy", np.array(test_tokens))

def main():
    tokenizer = BPETokenizer()
    
    train_text, valid_text, test_text, all_text = tokenizer.load_data()
    tokenizer.train_tokenizer(all_text)
    train_tokens, valid_tokens, test_tokens = tokenizer.tokenize_splits(train_text, valid_text, test_text)
    
    tokenizer.save_tokenizer()
    tokenizer.save_data(train_tokens, valid_tokens, test_tokens)

if __name__ == "__main__":
    main()