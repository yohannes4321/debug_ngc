from config import Config as config
from layers.attention import Attention
from layers.mlp import MLP
from jax import jit, random
import jax.numpy as jnp
from utils.model_util import ReshapeComponent

@jit
def rms_normalize(x, eps=1e-6):
    """
    RMS Normalization
    """
    return x * jnp.rsqrt(jnp.mean(jnp.square(x), axis=-1, keepdims=True) + eps)

class Block:
    def __init__(self, dkey, block_id, n_embed, seq_len, vocab_size,
                 batch_size, n_heads, dropout_rate, eta, optim_type, wub, wlb, **kwargs):
        
        dkey, attn_key, mlp_key = random.split(dkey, 3)
        prefix = f"block{block_id}_"
    
        self.attention = Attention(dkey=attn_key, n_embed=n_embed, seq_len=seq_len,
                                 batch_size=batch_size, n_heads=n_heads,
                                 dropout_rate=dropout_rate, eta=eta, optim_type= optim_type, wub=wub, wlb=wlb, prefix=prefix)
        self.mlp = MLP(dkey=mlp_key, n_embed=n_embed, seq_len=seq_len,
                      batch_size=batch_size, eta=eta, optim_type=optim_type, wub=wub, wlb=wlb, prefix=prefix)
        self.rms_norm = rms_normalize 
        
        self.reshape_2d_to_3d_q = ReshapeComponent(f"{prefix}reshape_2d_to_3d_q",
                                            input_shape=(batch_size * seq_len, n_embed),
                                            output_shape=(batch_size, seq_len, n_embed))
        self.reshape_2d_to_3d_k = ReshapeComponent(f"{prefix}reshape_2d_to_3d_k",    
                                            input_shape=(batch_size * seq_len, n_embed),
                                            output_shape=(batch_size, seq_len, n_embed))    
        self.reshape_2d_to_3d_v = ReshapeComponent(f"{prefix}reshape_2d_to_3d_v",
                                            input_shape=(batch_size * seq_len, n_embed),
                                            output_shape=(batch_size, seq_len, n_embed))
        self.reshape_3d_to_2d_attnout= ReshapeComponent(f"{prefix}reshape_3d_to_2d_attnout",
                                            input_shape=(batch_size, seq_len, n_embed),
                                            output_shape=(batch_size * seq_len, n_embed))
        self.reshape_3d_to_2d = ReshapeComponent(f"{prefix}reshape_3d_to_2d",
                                            input_shape=(batch_size, seq_len, n_embed),
                                            output_shape=(batch_size * seq_len, n_embed))        