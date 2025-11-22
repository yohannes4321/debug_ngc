from jax import random, numpy as jnp, jit
import jax
from functools import partial
from ngclearn.utils.optim import get_opt_init_fn, get_opt_step_fn
from ngclearn.components.jaxComponent import JaxComponent
from ngclearn import resolver, Compartment
from ngcsimlib.compilers.process import transition
from ngclearn.utils.model_utils import drop_out, softmax, gelu, layer_normalize


class NormalizeComponent(JaxComponent):
    """Component that reshapes tensors for ngc-learn wiring"""
    
    def __init__(self, name, input_shape, output_shape, **kwargs):
        super().__init__(name, **kwargs)
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.inputs = Compartment(jnp.zeros(input_shape))
        self.outputs = Compartment(jnp.zeros(output_shape))
        
    
    @transition(output_compartments=["outputs"])
    @staticmethod
    def advance_state(inputs, output_shape):
        feature_dim = inputs.shape[-1]

        ln_in_mu = jnp.zeros((1, feature_dim))
        ln_in_scale = jnp.ones((1, feature_dim))
        inputs = layer_normalize(inputs, ln_in_mu, ln_in_scale)
        return inputs
    
    @transition(output_compartments=["inputs", "outputs"])
    @staticmethod  
    def reset(input_shape, output_shape):
        return jnp.zeros(input_shape), jnp.zeros(output_shape)