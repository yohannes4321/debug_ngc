from jax import random, numpy as jnp, jit
import jax
from functools import partial
from ngclearn.utils.optim import get_opt_init_fn, get_opt_step_fn
from ngclearn.components.jaxComponent import JaxComponent
from ngclearn import resolver, Compartment
from ngcsimlib.compilers.process import transition
from ngclearn.utils import tensorstats
from functools import partial
from jax import jit, lax, random
import jax
import jax.numpy as jnp
from jax.ops import segment_sum
# from config import vocab_size
@partial(jit, static_argnums=[0, 1])
def _create_sinusoidal_embeddings(seq_len, embed_dim):
    """Create fixed sinusoidal position embeddings"""
    position = jnp.arange(seq_len)[:, None]
    div_term = jnp.exp(jnp.arange(0, embed_dim, 2) * 
                      (-jnp.log(10000.0) / embed_dim))
    
    embeddings = jnp.zeros((seq_len, embed_dim))
    embeddings = embeddings.at[:, 0::2].set(jnp.sin(position * div_term))
    embeddings = embeddings.at[:, 1::2].set(jnp.cos(position * div_term))
    return embeddings

@partial(jit, static_argnums=[4, 5, 6, 7])
def _compute_embedding_updates(inputs, post, word_weights, pos_weights, 
                              vocab_size, seq_len, embed_dim, batch_size, pos_learnable):
    """
    Compute updates for word and position embeddings
    """
    # Flatten tokens and errors
    flat_tokens = (inputs.reshape(-1) )
    unique_tokens = jnp.unique(flat_tokens)
    flat_errors = post.reshape(batch_size * seq_len, embed_dim)
    
    # Word embeddings update
    d_word_weights = jnp.zeros((vocab_size, embed_dim), dtype=jnp.float32)
    d_word_weights = d_word_weights.at[unique_tokens].add(flat_errors)
    
    # Position embeddings update
    d_pos_weights = jnp.zeros((seq_len, embed_dim), dtype=jnp.float32)
    batch_positions = jnp.tile(jnp.arange(seq_len), batch_size)
    d_pos_weights = jax.lax.cond(
        pos_learnable,
        lambda _: d_pos_weights.at[batch_positions].add(flat_errors),
        lambda _: d_pos_weights,
        operand=None
    )
    
    return d_word_weights, d_pos_weights

# @partial(jit, static_argnums=[4, 5, 6, 7])
# def _compute_embedding_updates(inputs, post, word_weights, pos_weights,
#                                vocab_size, seq_len, embed_dim, batch_size, pos_learnable):
#     """
#     Compute updates for word and position embeddings (JIT-safe).

#     Args:
#       inputs: int32 array (batch_size, seq_len) token ids
#       post: float array (batch_size, seq_len, embed_dim) post-synaptic signals
#       word_weights: (vocab_size, embed_dim) - unused for update calc but kept for signature
#       pos_weights: (seq_len, embed_dim) - unused here as well
#       vocab_size, seq_len, embed_dim, batch_size: ints
#       pos_learnable: boolean (treated as traced value; use lax.cond)

#     Returns:
#       d_word_weights: (vocab_size, embed_dim) accumulated updates per token id
#       d_pos_weights: (seq_len, embed_dim) accumulated updates per position
#     """

#     # Flatten tokens and errors: shapes (batch*seq,), (batch*seq, embed_dim)
#     flat_tokens = inputs.reshape(-1).astype(jnp.int32)
#     flat_errors = post.reshape(batch_size * seq_len, embed_dim)

#     # -------- WORD EMBEDDING GRADIENT (correctly accumulates repeats) --------
#     # segment_sum sums rows in flat_errors grouped by flat_tokens.
#     # result shape: (vocab_size, embed_dim)
#     d_word_weights = segment_sum(flat_errors, flat_tokens, num_segments=vocab_size)

#     # -------- POSITION EMBEDDING GRADIENT (use lax.cond for JIT) --------
#     batch_positions = jnp.tile(jnp.arange(seq_len, dtype=jnp.int32), batch_size)

#     def pos_true(_):
#         # accumulate errors per position (num_s-egments=seq_len)
#         return segment_sum(flat_errors, batch_positions, num_segments=seq_len)

#     def pos_false(_):
#         return jnp.zeros((seq_len, embed_dim), dtype=flat_errors.dtype)

#     d_pos_weights = lax.cond(pos_learnable, pos_true, pos_false, operand=None)

#     return d_word_weights, d_pos_weights

class EmbeddingSynapse(JaxComponent):
    """
    A synaptic cable that handles both word and position embeddings.
    Combines word embeddings with learnable or fixed positional embeddings.

    | --- Synapse Compartments: ---
    | inputs - input token indices (takes in external signals)
    | outputs - output embedding signals (combined word + position embeddings)
    | word_weights - word embedding matrix
    | pos_weights - position embedding matrix  
    | post - post-synaptic signals for learning (takes in external signals)
    | key - JAX PRNG key
    | --- Synaptic Plasticity Compartments: ---
    | dWordWeights - current delta matrix for word embedding changes
    | dPosWeights - current delta matrix for position embedding changes
    | word_opt_params - optimizer statistics for word embeddings
    | pos_opt_params - optimizer statistics for position embeddings

    Args:
        name: the string name of this component

        vocab_size: size of vocabulary for word embeddings

        seq_len: sequence length for position embeddings

        embed_dim: dimensionality of embeddings

        batch_size: batch size dimension

        pos_learnable: whether position embeddings are learnable or fixed

        eta: global learning rate 

        optim_type: optimization scheme (Default: "sgd")

        weight_scale: scaling factor for weight initialization (Default: 0.02)
    """

    def __init__(
            self, name, vocab_size, seq_len, embed_dim, batch_size,
            pos_learnable, eta, optim_type, weight_scale=0.02,
            **kwargs
    ):
        super().__init__(name, **kwargs)

        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        self.batch_size = batch_size
        self.pos_learnable = pos_learnable
        self.eta = eta
        self.weight_scale = weight_scale
        self.optim_type = optim_type

        key = self.key.value
        word_key, pos_key = random.split(key, 2)
        
        word_weights = random.normal(word_key, (vocab_size, embed_dim)) * weight_scale
        
        if pos_learnable:
            pos_weights = random.normal(pos_key, (seq_len, embed_dim)) * weight_scale
        else:
            pos_weights = _create_sinusoidal_embeddings(seq_len, embed_dim)

        ## Compartments
        self.inputs = Compartment(jnp.zeros((batch_size, seq_len), dtype=jnp.int32))
        self.outputs = Compartment(jnp.zeros((batch_size, seq_len, embed_dim)))
        self.word_weights = Compartment(word_weights)
        self.pos_weights = Compartment(pos_weights)
        self.post = Compartment(jnp.zeros((batch_size, seq_len, embed_dim)))
        
        self.dWordWeights = Compartment(jnp.zeros((vocab_size, embed_dim)))
        self.dPosWeights = Compartment(jnp.zeros((seq_len, embed_dim)))
        
        # Optimization
        self.opt = get_opt_step_fn(optim_type, eta=self.eta)
        self.word_opt_params = Compartment(
            get_opt_init_fn(optim_type)([self.word_weights.value])
        )
        if pos_learnable:
            self.pos_opt_params = Compartment(
                get_opt_init_fn(optim_type)([self.pos_weights.value])
            )
        else:
            self.pos_opt_params = Compartment(None)

    @transition(output_compartments=["outputs"])
    @staticmethod
    def advance_state(inputs, word_weights, pos_weights, seq_len, embed_dim):
        """
        Forward pass: output = word_embedding[inputs] + position_embedding[positions]
        """
        batch_size = inputs.shape[0]
        
        flat_tokens = inputs.reshape(-1).astype(jnp.int32)
        word_embeds_flat = word_weights[flat_tokens]
        word_embeds = word_embeds_flat.reshape(batch_size, seq_len, embed_dim)
        
        positions = jnp.arange(seq_len)
        pos_embeds = pos_weights[positions]
        pos_embeds_batch = jnp.broadcast_to(pos_embeds, (batch_size, seq_len, embed_dim))
        
        combined_embeddings = word_embeds + pos_embeds_batch
        return combined_embeddings

    @transition(output_compartments=["word_weights", "pos_weights", "dWordWeights", "dPosWeights", 
                                   "word_opt_params", "pos_opt_params"])
    @staticmethod
    def evolve(opt, pos_learnable, vocab_size, seq_len, embed_dim, batch_size,
               inputs, post, word_weights, pos_weights, word_opt_params, pos_opt_params):
        """
        Learning step: Hebbian updates for both word and position embeddings
        """
        # Compute embedding updates
        inputs= inputs.astype(jnp.int32)
        d_word_weights, d_pos_weights = _compute_embedding_updates(
            inputs, post, word_weights, pos_weights, vocab_size, seq_len, 
            embed_dim, batch_size, pos_learnable
        )
        
        word_opt_params, [new_word_weights] = opt(
            word_opt_params, [word_weights], [d_word_weights]
        )
        
        new_pos_weights = pos_weights
        new_pos_opt_params = pos_opt_params
        
        if pos_learnable:
            pos_opt_params, [new_pos_weights] = opt(
                pos_opt_params, [pos_weights], [d_pos_weights]
            )
            new_pos_opt_params = pos_opt_params
        
        return new_word_weights, new_pos_weights, d_word_weights, d_pos_weights, word_opt_params, new_pos_opt_params

    @transition(output_compartments=["inputs", "outputs", "post", "dWordWeights", "dPosWeights"])
    @staticmethod
    def reset(batch_size, seq_len, embed_dim, vocab_size):
        """
        Reset compartments to zeros
        """
        inputs = jnp.zeros((batch_size, seq_len), dtype=jnp.int32)
        outputs = jnp.zeros((batch_size, seq_len, embed_dim))
        post = jnp.zeros((batch_size, seq_len, embed_dim))
        dWordWeights = jnp.zeros((vocab_size, embed_dim))
        dPosWeights = jnp.zeros((seq_len, embed_dim))
        return inputs, outputs, post, dWordWeights, dPosWeights

    @classmethod
    def help(cls):
        """Component help function"""
        properties = {
            "synapse_type": "EmbeddingSynapse - combines word and position embeddings "
                           "into a single embedding representation"
        }
        compartment_props = {
            "inputs": 
                {"inputs": "Input token indices (batch_size, seq_len)",
                 "post": "Post-synaptic error signals for learning"},
            "states":
                {"word_weights": "Word embedding matrix (vocab_size, embed_dim)",
                 "pos_weights": "Position embedding matrix (seq_len, embed_dim)",
                 "key": "JAX PRNG key"},
            "analytics":
                {"dWordWeights": "Word embedding adjustment matrix",
                 "dPosWeights": "Position embedding adjustment matrix"},
            "outputs":
                {"outputs": "Combined embeddings (batch_size, seq_len, embed_dim)"},
        }
        hyperparams = {
            "vocab_size": "Size of vocabulary",
            "seq_len": "Maximum sequence length", 
            "embed_dim": "Dimensionality of embeddings",
            "batch_size": "Batch size dimension",
            "pos_learnable": "Whether position embeddings are learnable",
            "eta": "Global learning rate",
            "optim_type": "Optimization scheme",
            "weight_scale": "Weight initialization scale"
        }
        info = {cls.__name__: properties,
                "compartments": compartment_props,
                "dynamics": "outputs = word_embedding[inputs] + position_embedding[positions]",
                "hyperparameters": hyperparams}
        return info

    def __repr__(self):
        comps = [varname for varname in dir(self) if Compartment.is_compartment(getattr(self, varname))]
        maxlen = max(len(c) for c in comps) + 5
        lines = f"[{self.__class__.__name__}] PATH: {self.name}\n"
        for c in comps:
            stats = tensorstats(getattr(self, c).value)
            if stats is not None:
                line = [f"{k}: {v}" for k, v in stats.items()]
                line = ", ".join(line)
            else:
                line = "None"
            lines += f"  {f'({c})'.ljust(maxlen)}{line}\n"
        return lines


if __name__ == "__main__":
   

    # ---- Dummy configuration ----
    vocab_size = 11710 
    seq_len = 12
    embed_dim = 2
    batch_size = 64
    pos_learnable = True
    weight_scale = 0.02

    # ---- PRNG key ----
    key = random.PRNGKey(0)
    word_key, pos_key = random.split(key, 2)

    # ---- Initialize embeddings ----
    word_weights = random.normal(word_key, (vocab_size, embed_dim)) * weight_scale
    if pos_learnable:
        pos_weights = random.normal(pos_key, (seq_len, embed_dim)) * weight_scale
    else:
        pos_weights = _create_sinusoidal_embeddings(seq_len, embed_dim)

    # ---- Dummy input tokens ----
    inputs = random.randint(key, (batch_size, seq_len), 0, vocab_size)
    synapse = EmbeddingSynapse(
        name="test_synapse",
        vocab_size=vocab_size,
        seq_len=seq_len,
        embed_dim=embed_dim,
        batch_size=batch_size,
        pos_learnable=pos_learnable,
        eta=0.01,
        optim_type="sgd",
        weight_scale=0.02
    )
    
    optim_type="sgd"
    eta=0.01
    post = random.normal(key, (batch_size, seq_len, embed_dim))
    # print("post)
    opt = get_opt_step_fn(optim_type, eta=eta)
    word_opt_params = get_opt_init_fn(optim_type)([word_weights])
    pos_opt_params = get_opt_init_fn(optim_type)([pos_weights])
    # ---- Call advance_state to get combined embeddings ----
  
    print("Word weights intial:", word_weights)
    # print("postion weights intial value ",pos_weights)
    # print("seperation____after i call evolve")

    new_word_weights, new_pos_weights, d_word_weights, d_pos_weights, \
        new_word_opt_params, new_pos_opt_params = synapse.evolve(
            opt, pos_learnable, vocab_size, seq_len, embed_dim, batch_size,
            inputs, post, word_weights, pos_weights, word_opt_params, pos_opt_params
    )
        # Flatten for processing
   

    print("New word weights shape:", new_word_weights)
    # print("New pos weights shape:", new_pos_weights)
    # print("dWordWeights shape:", d_word_weights)
    # print("dPosWeights shape:", d_pos_weights)






    


#    ` print(flat_tokens.min(), flat_tokens.max())  `# should be within [0, vocab_size-1]
    flat_tokens = inputs.reshape(-1).astype(jnp.int32)
    flat_errors = post.reshape(batch_size * seq_len, embed_dim).astype(jnp.float32)  # do NOT int32

    d_word_weights = jnp.zeros((vocab_size, embed_dim), dtype=jnp.float32)

    d_word_weights = d_word_weights.at[flat_tokens].add(flat_errors)
    
    print("Tokens actually used:", unique_tokens)
    print("Corresponding rows in d_word_weights:")
    print(d_word_weights[unique_tokens])
   


