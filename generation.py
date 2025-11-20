from model import NGCTransformer
import jax
import jax.numpy as jnp
import tiktoken
import os
import requests
import numpy as np
from config import Config as config

#data loading
input_file_path = os.path.join(os.path.dirname(__file__), 'input.txt')
if not os.path.exists(input_file_path):
    data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    with open(input_file_path, 'w', encoding='utf-8') as f:
        f.write(requests.get(data_url).text)

with open(input_file_path, 'r', encoding='utf-8') as f:
    data = f.read()
n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]

enc = tiktoken.get_encoding("gpt2")
train_ids = enc.encode_ordinary(train_data)
val_ids = enc.encode_ordinary(val_data)

print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")
vocab_size = enc.n_vocab
print(f"vocab size: {vocab_size}")

data_dir = os.path.join(os.path.dirname(__file__), 'data')
os.makedirs(data_dir, exist_ok=True)
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(data_dir, 'train.bin'))
val_ids.tofile(os.path.join(data_dir, 'val.bin'))


def get_batch(split, seq_len=128, batch_size=1, key=jax.random.PRNGKey(0)):
    script_dir = os.path.dirname(__file__)
    data_dir = os.path.join(script_dir, "data")
    data_path = os.path.join(data_dir, f'{split}.bin')
    data = np.memmap(data_path, dtype=np.uint16, mode='r')
    key, subkey = jax.random.split(key)
    ix = jax.random.randint(subkey, (batch_size,), 0, len(data) - seq_len)
    x = jnp.stack([jnp.array(data[i:i+seq_len], dtype=jnp.int32) for i in ix])
    y = jnp.stack([jnp.array(data[i+1:i+1+seq_len], dtype=jnp.int32) for i in ix])
    return x, y, key

def generate_text(
    model,
    prompt: str,
    max_new_tokens: int = 100,
    seq_len: int = 8,
    temperature: float = 1.0,
    key=None
):
    """
    Generates text using the provided NGCTransformer model.

    Args:
        model: Instance of NGCTransformer.
        prompt: Input prompt as a string.
        max_new_tokens: Number of tokens to generate.
        seq_len: Context window size (must match training).
        temperature: Sampling temperature (1.0 = standard softmax).
        key: JAX PRNG key for stochastic sampling (if None, uses greedy argmax).

    Returns:
        Generated text as a string (prompt + continuation).
    """
    # Encode prompt
    prompt_ids = enc.encode_ordinary(prompt)
    prompt_tensor = jnp.array([prompt_ids], dtype=jnp.int32)  # shape: (1, seq_len)

    current_tokens = prompt_tensor
    current_key = key

    for _ in range(max_new_tokens):
        # Truncate or pad to fit within seq_len
        if current_tokens.shape[1] > seq_len:
            input_seq = current_tokens[:, -seq_len:]
        else:
            input_seq = current_tokens

        # Pad to exactly seq_len if needed 
        if input_seq.shape[1] < seq_len:
            pad_len = seq_len - input_seq.shape[1]
            input_seq = jnp.pad(input_seq, ((0, 0), (0, pad_len)), constant_values=0)
        dummy_target = jnp.zeros((model.batch_size * model.seq_len, vocab_size))  

        # Run inference 
        y_mu_inf, _, _ = model.process(input_seq, dummy_target, adapt_synapses=False)

      
        logits = y_mu_inf.reshape(model.batch_size, model.seq_len, vocab_size)

        # Get logits for the last **real** token in the input (not padding)
        actual_len = min(current_tokens.shape[1], seq_len)
        last_pos = actual_len - 1
        next_logits = logits[0, last_pos, :] / temperature

        if current_key is not None:
            probs = jax.nn.softmax(next_logits)
            current_key, subkey = jax.random.split(current_key)
            next_token = jax.random.choice(subkey, a=vocab_size, p=probs)
        else:
            next_token = jnp.argmax(next_logits)

        # Append token
        current_tokens = jnp.concatenate([current_tokens, next_token[None, None]], axis=1)

    # Decode full sequence to string
    generated_ids = current_tokens[0].tolist()
    return enc.decode(generated_ids)
dkey = jax.random.PRNGKey(0)
model = NGCTransformer(dkey, config.batch_size, config.seq_len, config.n_embed, config.vocab_size, T=10,
                 dt=1., tau_m=10., act_fx="tanh", eta=0.001, exp_dir="exp",
                 model_name="ngc transformer", loadDir=None, pos_learnable= True)
# Example usage
prompt = "The king said: "
generated = generate_text(
    model=model,
    prompt=prompt,
    max_new_tokens=200,
    seq_len=config.seq_len,        
    temperature=0.8,
    key=jax.random.PRNGKey(42)  
)
print(generated)