import jax
from ngclearn.utils.model_utils import scanner
from ngcsimlib.compilers import compile_command, wrap_command
from ngcsimlib.context import Context
from ngclearn.utils import JaxProcess
from ngclearn.utils.io_utils import makedir
from jax import numpy as jnp, random, jit
from ngclearn.components import GaussianErrorCell as ErrorCell, RateCell, HebbianSynapse, StaticSynapse
import ngclearn.utils.weight_distribution as dist
from config import Config as config
from layers.embedding import EMBEDDING
from layers.attention import Attention
from layers.blocks import Block
from utils.attention_utils import AttentionBlock
from utils.embed_utils import EmbeddingSynapse
from layers.mlp import MLP
from layers.output import Output
from utils.model_util import ReshapeComponent
from projection.projection import Projection

class NGCTransformer:
    """
    Predictive Coding Transformer following PCN architecture from:
    Whittington & Bogacz (2017) - "An approximation of the error backpropagation 
    algorithm in a predictive coding network with local hebbian synaptic plasticity"

    Architecture:
    z_embed -(W_embed)-> e_embed, z_qkv -(W_q,W_k,W_v - > W_attn_out)-> e_attn, z_mlp -(W_mlp1,W_mlp2)-> e_mlp, z_out -(W_out)-> e_out
    e_attn -(E_attn)-> z_qkv <- e_embed, e_mlp -(E_mlp)-> z_mlp <- e_attn, e_out -(E_out)-> z_out <- e_mlp

    Args:
        dkey: JAX seeding key
        vocab_size: vocabulary size
        seq_len: sequence length
        n_embed: embedding dimension
        n_heads: number of attention heads
        batch_size: batch size
        n_layers: number of transformer blocks
        dt: integration time constant
        tau_m: membrane time constant
        eta: learning rate for Hebbian synapses
        exp_dir: experimental directory
        model_name: unique model name
    """

    def __init__(self, dkey, batch_size, seq_len, n_embed, vocab_size, n_layers, n_heads,  T,
                 dt, tau_m, act_fx, eta, exp_dir,
                 model_name, loadDir, pos_learnable, optim_type, wlb, wub , dropout_rate, **kwargs):
        self.exp_dir = exp_dir
        self.model_name = model_name
        self.nodes = None
        self.n_layers = n_layers
        self.T = T
        makedir(exp_dir)
        makedir(exp_dir + "/filters")

        dkey, *subkeys = random.split(dkey, 50)

        if loadDir is not None:
            self.load_from_disk(loadDir)
        else:
            with Context("Circuit") as self.circuit:
                
                self.embedding = EMBEDDING(dkey=subkeys[0], vocab_size=vocab_size, seq_len=seq_len, embed_dim=n_embed, batch_size=batch_size, pos_learnable=pos_learnable, eta=eta, optim_type=optim_type)
                
                self.blocks = []
                for i in range(n_layers):
                    key, subkey = random.split(subkeys[1 + i])
                    block=Block(dkey=subkey, block_id= i, n_embed=n_embed, seq_len=seq_len,
                                batch_size=batch_size, vocab_size=vocab_size, n_heads=n_heads, dropout_rate=dropout_rate, eta=eta, optim_type=optim_type, wub=wub, wlb=wlb)
                    self.blocks.append(block)
                    
                self.output = Output(dkey=subkeys[3], n_embed=n_embed, seq_len=seq_len, batch_size=batch_size, vocab_size=vocab_size, eta=eta, optim_type=optim_type, wlb=wlb, wub=wub)
                
                self.z_target=RateCell("z_target", n_units= vocab_size, tau_m=0., act_fx="identity", batch_size=batch_size * seq_len) 
               
                
                self.reshape_4d_to_2d = ReshapeComponent("reshape_4d_to_2d",
                                            input_shape=(batch_size, seq_len, n_embed, 1),
                                            output_shape=(batch_size * seq_len, n_embed))
                
                self.reshape_3d_to_2d_embed = ReshapeComponent("reshape_3d_to_2d_embed",
                                            input_shape=(batch_size, seq_len, n_embed),
                                            output_shape=(batch_size * seq_len, n_embed))
                self.reshape_2d_to_3d_embed= ReshapeComponent("reshape_2d_to_3d_embed",
                                            input_shape=(batch_size * seq_len, n_embed),
                                            output_shape=(batch_size, seq_len, n_embed))
                
                
                self.embedding.W_embed.inputs << self.embedding.z_embed.zF  
                self.reshape_3d_to_2d_embed.inputs << self.embedding.W_embed.outputs   
                self.embedding.e_embed.mu << self.reshape_3d_to_2d_embed.outputs
                self.embedding.e_embed.target << self.blocks[0].attention.z_qkv.z
                
                # self.reshape_4d_to_2d.inputs << self.attention.z_qkv.zF
                for blocks in range(n_layers):
                    block= self.blocks[blocks]
                    block.attention.W_q.inputs << block.attention.z_qkv.zF
                    block.attention.W_k.inputs << block.attention.z_qkv.zF
                    block.attention.W_v.inputs << block.attention.z_qkv.zF
                    
                    block.reshape_2d_to_3d_q.inputs << block.attention.W_q.outputs
                    block.reshape_2d_to_3d_k.inputs << block.attention.W_k.outputs
                    block.reshape_2d_to_3d_v.inputs << block.attention.W_v.outputs
                    
                    block.attention.attn_block.inputs_q << block.reshape_2d_to_3d_q.outputs
                    block.attention.attn_block.inputs_k << block.reshape_2d_to_3d_k.outputs       
                    block.attention.attn_block.inputs_v << block.reshape_2d_to_3d_v.outputs
                    
                    block.reshape_3d_to_2d.inputs << block.attention.attn_block.outputs
                    block.attention.W_attn_out.inputs << block.reshape_3d_to_2d.outputs
                    block.attention.e_attn.mu << block.attention.W_attn_out.outputs
                    block.attention.e_attn.target << block.mlp.z_mlp.z

                    block.mlp.W_mlp1.inputs << block.mlp.z_mlp.zF
                    block.mlp.e_mlp1.mu << block.mlp.W_mlp1.outputs
                    block.mlp.e_mlp1.target << block.mlp.z_mlp2.z
                    
                    
                    block.mlp.W_mlp2.inputs << block.mlp.z_mlp2.zF
                    block.mlp.e_mlp.mu << block.mlp.W_mlp2.outputs
                    
                    if blocks == n_layers -1:
                        block.mlp.e_mlp.target << self.output.z_out.z
                    else:
                        block.mlp.e_mlp.target << self.blocks[blocks + 1].attention.z_qkv.z 
                    block.attention.E_attn.inputs << block.attention.e_attn.dmu
                
                    block.mlp.E_mlp1.inputs << block.mlp.e_mlp1.dmu
                    block.mlp.E_mlp.inputs << block.mlp.e_mlp.dmu
                    block.attention.z_qkv.j << block.attention.E_attn.outputs
                    
                    if blocks == 0:
                        block.attention.z_qkv.j_td << self.embedding.e_embed.dtarget
                    else:
                        block.attention.z_qkv.j_td << block.mlp.e_mlp.dtarget
                    block.mlp.z_mlp2.j << block.mlp.E_mlp.outputs
                    block.mlp.z_mlp.j << block.mlp.E_mlp1.outputs
                    block.mlp.z_mlp.j_td << block.attention.e_attn.dtarget
                    block.mlp.z_mlp2.j_td << block.mlp.e_mlp1.dtarget
                    block.attention.W_q.pre << block.attention.z_qkv.zF
                    block.attention.W_q.post << block.attention.e_attn.dmu
                    
                    block.attention.W_k.pre << block.attention.z_qkv.zF
                    block.attention.W_k.post << block.attention.e_attn.dmu
                    
                    block.attention.W_v.pre << block.attention.z_qkv.zF
                    block.attention.W_v.post << block.attention.e_attn.dmu
                    block.reshape_3d_to_2d_attnout.inputs << block.attention.attn_block.outputs
                    block.attention.W_attn_out.pre << block.reshape_3d_to_2d_attnout.outputs
                    block.attention.W_attn_out.post << block.attention.e_attn.dmu
                                
                    block.mlp.W_mlp1.pre << block.mlp.z_mlp.zF
                    block.mlp.W_mlp1.post << block.mlp.e_mlp1.dmu
                    block.mlp.W_mlp2.pre << block.mlp.z_mlp2.zF
                    block.mlp.W_mlp2.post << block.mlp.e_mlp.dmu
                        
                self.output.W_out.inputs << self.output.z_out.zF
                self.output.e_out.mu << self.output.W_out.outputs
                self.output.e_out.target << self.z_target.z
            
                self.output.E_out.inputs << self.output.e_out.dmu
                
                
                # self.embedding.z_embed.j << self.embedding.E_embed.outputs
                # self.embedding.z_embed.j_td << td_error.e_attn.dtarget
                
                
                self.output.z_out.j << self.output.E_out.outputs
                self.output.z_out.j_td << self.blocks[n_layers - 1].mlp.e_mlp.dtarget
                
                
                # self.W_embed.pre << self.z_embed.zF
                self.reshape_2d_to_3d_embed.inputs << self.embedding.e_embed.dmu 
                self.embedding.W_embed.post << self.reshape_2d_to_3d_embed.outputs 
                
                self.output.W_out.pre << self.output.z_out.zF
                self.output.W_out.post << self.output.e_out.dmu      
               
        
        
        
                ## PROJECTION PHASE ##
                
                self.projection = Projection(dkey=subkeys[29], n_embed=n_embed, seq_len=seq_len, batch_size=batch_size,
                                             vocab_size=vocab_size, eta=eta, optim_type=optim_type, pos_learnable=pos_learnable, wub=wub, wlb=wlb, n_blocks=n_layers, n_heads=n_heads, dropout_rate=dropout_rate)
                
                
                self.projection.Q_embed.inputs << self.projection.q_embed.zF
                self.projection.reshape_3d_to_2d_proj.inputs << self.projection.Q_embed.outputs
                for b in range(n_layers):
                    block_proj= self.projection.blocks[b]
                    if b ==0:
                       block_proj.q_qkv.j << self.projection.reshape_3d_to_2d_proj.outputs
                    else:
                       block_proj.q_qkv.j << self.projection.blocks[b - 1].Q_mlp2.outputs
                       
                    block_proj.Q_q.inputs << block_proj.q_qkv.zF
                    block_proj.Q_k.inputs << block_proj.q_qkv.zF
                    block_proj.Q_v.inputs << block_proj.q_qkv.zF
                    block_proj.q_attn_block.inputs_q << block_proj.Q_q.outputs
                    block_proj.q_attn_block.inputs_k << block_proj.Q_k.outputs
                    block_proj.q_attn_block.inputs_v << block_proj.Q_v.outputs
                    
                    block_proj.reshape_3d_to_2d_proj1.inputs << block_proj.q_attn_block.outputs
                    block_proj.Q_attn_out.inputs << block_proj.reshape_3d_to_2d_proj1.outputs
                    block_proj.q_mlp.j << block_proj.Q_attn_out.outputs
                    
                    block_proj.Q_mlp1.inputs << block_proj.q_mlp.zF
                    block_proj.q_mlp2.j << block_proj.Q_mlp1.outputs
                    block_proj.Q_mlp2.inputs << block_proj.q_mlp2.zF
                self.projection.q_out.j << self.projection.blocks[n_layers - 1].Q_mlp2.outputs
                self.projection.Q_out.inputs << self.projection.q_out.zF
                self.projection.q_target.j <<self.projection.Q_out.outputs
                self.projection.eq_target.target << self.projection.q_target.z
                self.projection.eq_target.mu << self.projection.Q_out.outputs
                
                # Create the processes by iterating through all blocks
                advance_process = JaxProcess(name="advance_process")
                reset_process = JaxProcess(name="reset_process")
                embedding_evolve_process = (JaxProcess(name="embedding_evolve_process")
                                            >> self.embedding.W_embed.evolve) 
                evolve_process = JaxProcess(name="evolve_process")
                project_process = JaxProcess(name="project_process")
                for i in range(n_layers):
                    block = self.blocks[i]
                    
                    advance_process >> block.attention.E_attn.advance_state
                    advance_process >> block.mlp.E_mlp.advance_state
                    advance_process >> block.attention.z_qkv.advance_state
                    advance_process >> block.mlp.z_mlp.advance_state
                    advance_process >> block.mlp.z_mlp2.advance_state
                    advance_process >> block.attention.W_q.advance_state
                    advance_process >> block.attention.W_k.advance_state
                    advance_process >> block.attention.W_v.advance_state
                    advance_process >> block.reshape_2d_to_3d_q.advance_state
                    advance_process >> block.reshape_2d_to_3d_k.advance_state
                    advance_process >> block.reshape_2d_to_3d_v.advance_state
                    advance_process >> block.attention.attn_block.advance_state
                    advance_process >> block.reshape_3d_to_2d.advance_state
                    advance_process >> block.reshape_3d_to_2d_attnout.advance_state
                    advance_process >> block.attention.W_attn_out.advance_state
                    advance_process >> block.mlp.W_mlp1.advance_state
                    advance_process >> block.mlp.W_mlp2.advance_state
                    advance_process >> block.attention.e_attn.advance_state
                    advance_process >> block.mlp.e_mlp.advance_state
                    
                    reset_process >> block.attention.z_qkv.reset
                    reset_process >> block.mlp.z_mlp.reset
                    reset_process >> block.mlp.z_mlp2.reset
                    reset_process >> block.attention.e_attn.reset
                    reset_process >> block.mlp.e_mlp.reset
                    reset_process >> block.mlp.e_mlp1.reset
                    reset_process >> block.reshape_3d_to_2d.reset
                    reset_process >> block.reshape_2d_to_3d_q.reset
                    reset_process >> block.reshape_2d_to_3d_k.reset
                    reset_process >> block.reshape_2d_to_3d_v.reset
                    reset_process >> block.reshape_3d_to_2d_attnout.reset
                    
                    evolve_process >> block.attention.W_q.evolve
                    evolve_process >> block.attention.W_k.evolve
                    evolve_process >> block.attention.W_v.evolve
                    evolve_process >> block.attention.W_attn_out.evolve
                    evolve_process >> block.mlp.W_mlp1.evolve
                    evolve_process >> block.mlp.W_mlp2.evolve

                # Add non-block components to advance_process, reset_process, evolve_process
                advance_process >> self.output.E_out.advance_state
                advance_process >> self.embedding.z_embed.advance_state
                advance_process >> self.output.z_out.advance_state
                advance_process >> self.embedding.W_embed.advance_state
                advance_process >> self.reshape_3d_to_2d_embed.advance_state
                advance_process >> self.reshape_2d_to_3d_embed.advance_state
                advance_process >> self.output.W_out.advance_state
                advance_process >> self.embedding.e_embed.advance_state
                advance_process >> self.output.e_out.advance_state

                reset_process >> self.projection.q_embed.reset
                reset_process >> self.projection.q_out.reset
                reset_process >> self.projection.q_target.reset
                reset_process >> self.projection.eq_target.reset
                reset_process >> self.embedding.z_embed.reset
                reset_process >> self.output.z_out.reset
                reset_process >> self.embedding.e_embed.reset
                reset_process >> self.output.e_out.reset
                reset_process >> self.reshape_3d_to_2d_embed.reset
                reset_process >> self.reshape_2d_to_3d_embed.reset

                evolve_process >> self.output.W_out.evolve
                project_process >> self.projection.q_embed.advance_state
                project_process >> self.projection.Q_embed.advance_state
                project_process >> self.projection.reshape_3d_to_2d_proj.advance_state
                for b in range(n_layers):
                    block_proj= self.projection.blocks[b]
                    project_process >> block_proj.q_qkv.advance_state
                    project_process >> block_proj.Q_q.advance_state
                    project_process >> block_proj.Q_k.advance_state
                    project_process >> block_proj.Q_v.advance_state
                    project_process >> block_proj.q_attn_block.advance_state
                    project_process >> block_proj.reshape_3d_to_2d_proj1.advance_state
                    project_process >> block_proj.Q_attn_out.advance_state
                    project_process >> block_proj.q_mlp.advance_state
                    project_process >> block_proj.q_mlp2.advance_state
                    project_process >> block_proj.Q_mlp1.advance_state
                    project_process >> block_proj.Q_mlp2.advance_state
                    reset_process >> block_proj.q_qkv.reset
                    reset_process >> block_proj.q_attn_block.reset
                    reset_process >> block_proj.q_mlp.reset
                    reset_process >> block_proj.q_mlp2.reset 
                project_process >> self.projection.q_out.advance_state
                project_process >> self.projection.Q_out.advance_state
                project_process >> self.projection.q_target.advance_state
                project_process >> self.projection.eq_target.advance_state
                
                processes = (reset_process, advance_process, embedding_evolve_process, evolve_process, project_process)        

                self._dynamic(processes)
    
    def _dynamic(self, processes):
        vars = self.circuit.get_components( "reshape_3d_to_2d_embed", "reshape_2d_to_3d_embed",
            "q_embed", "q_out", "reshape_3d_to_2d_proj", "q_target", "eq_target","Q_embed", "Q_out",
                                           "z_embed", "z_out", "e_embed", "e_out", "W_embed", "W_out", "E_out")
        (self.reshape_3d_to_2d_embed,  self.reshape_2d_to_3d_embed, self.q_embed, self.q_out, self.reshape_3d_to_2d_proj, 
        self.q_target, self.eq_target, self.Q_embed, self.Q_out,
        self.embedding.z_embed, self.output.z_out, self.embedding.e_embed, self.output.e_out, self.embedding.W_embed,
        self.output.W_out, self.output.E_out) = vars
        
        self.block_components = []  
    
        for i in range(self.n_layers):
            var2 = self.circuit.get_components(
                f"block{i}_z_qkv", f"block{i}_e_attn", f"block{i}_W_q", f"block{i}_W_k", f"block{i}_W_v",
                f"block{i}_W_attn_out", f"block{i}_E_attn", f"block{i}_z_mlp", f"block{i}_e_mlp",
                f"block{i}_W_mlp1", f"block{i}_W_mlp2", f"block{i}_E_mlp", f"block{i}_e_mlp1", f"block{i}_E_mlp1",
                f"block{i}_z_mlp2", f"block{i}_attn_block",
                f"block{i}_reshape_2d_to_3d_q", f"block{i}_reshape_2d_to_3d_k", f"block{i}_reshape_2d_to_3d_v",
                f"block{i}_reshape_3d_to_2d", f"block{i}_reshape_3d_to_2d_attnout",
                f"proj_block{i}_q_qkv", f"proj_block{i}_Q_q", f"proj_block{i}_Q_k", f"proj_block{i}_Q_v",
                f"proj_block{i}_Q_attn_out", f"proj_block{i}_q_attn_block",
                f"proj_block{i}_reshape_3d_to_2d_proj1", f"proj_block{i}_q_mlp", f"proj_block{i}_Q_mlp1",
                f"proj_block{i}_q_mlp2", f"proj_block{i}_Q_mlp2"    
            )
            
            self.block_components.append(var2)
        
        all_nodes = list(vars)
        for block_vars in self.block_components:
            all_nodes.extend(block_vars)
        self.nodes = all_nodes

        reset_proc, advance_proc, embedding_evolve_process, evolve_proc, project_proc = processes

        self.circuit.wrap_and_add_command(jit(reset_proc.pure), name="reset")
        self.circuit.wrap_and_add_command(jit(advance_proc.pure), name="advance")
        self.circuit.wrap_and_add_command(jit(project_proc.pure), name="project")
        self.circuit.wrap_and_add_command(jit(evolve_proc.pure), name="evolve")
        self.circuit.wrap_and_add_command(jit(embedding_evolve_process.pure), name="evolve_embedding")


        @Context.dynamicCommand
        def clamp_input(x):
            self.embedding.z_embed.j.set(x)
            self.q_embed.j.set(x) 
        
        @Context.dynamicCommand
        def clamp_target(y):
            self.z_target.j.set(y)

        @Context.dynamicCommand
        def clamp_infer_target(y):
            self.eq_target.target.set(y)
        
    def save_to_disk(self, params_only=False):
        """
        Saves current model parameter values to disk

        Args:
            params_only: if True, save only param arrays to disk (and not JSON sim/model structure)
        """
        if params_only:
            model_dir = "{}/{}/custom".format(self.exp_dir, self.model_name)
            self.embedding.W_embed.save(model_dir)
            self.blocks = []
            for j in range(self.n_layers):
                block = self.circuit.get_component(f"block{j}_W_q")
                block.save(model_dir)
                block = self.circuit.get_component(f"block{j}_W_k")
                block.save(model_dir)
                block = self.circuit.get_component(f"block{j}_W_v")
                block.save(model_dir)
                block = self.circuit.get_component(f"block{j}_W_attn_out")
                block.save(model_dir)
                block = self.circuit.get_component(f"block{j}_W_mlp1")
                block.save(model_dir)
                block = self.circuit.get_component(f"block{j}_W_mlp2")
                block.save(model_dir)    
            self.output.W_out.save(model_dir)
        else:
            self.circuit.save_to_json(self.exp_dir, model_name=self.model_name, overwrite=True)

    def load_from_disk(self, model_directory):
        """
        Loads parameter/config values from disk to this model

        Args:
            model_directory: directory/path to saved model parameter/config values
        """
        print(" > Loading model from ",model_directory)
        with Context("Circuit") as self.circuit:
            self.circuit.load_from_dir(model_directory)
            processes = (
                self.circuit.reset_process, self.circuit.advance_process,
                self.circuit.evolve_process, self.circuit.project_process
            )
            self._dynamic(processes)

    def process(self, obs, lab, adapt_synapses=True):
        eps = 0.001
        _lab = jnp.clip(lab, eps, 1. - eps)
        self.circuit.reset()

        ## pin/tie inference synapses to be exactly equal to the forward ones
        self.Q_embed.word_weights.set(self.embedding.W_embed.word_weights.value)
        if self.embedding.W_embed.pos_learnable:
           self.Q_embed.pos_weights.set(self.embedding.W_embed.pos_weights.value)
        for i in range(self.n_layers):
            block_proj= self.projection.blocks[i]
            block= self.blocks[i]
            block_proj.Q_q.weights.set(block.attention.W_q.weights.value)
            block_proj.Q_q.biases.set(block.attention.W_q.biases.value)
            block_proj.Q_k.weights.set(block.attention.W_k.weights.value)
            block_proj.Q_k.biases.set(block.attention.W_k.biases.value)
            block_proj.Q_v.weights.set(block.attention.W_v.weights.value)
            block_proj.Q_v.biases.set(block.attention.W_v.biases.value)
            block_proj.Q_attn_out.weights.set(block.attention.W_attn_out.weights.value)
            block_proj.q_attn_block.inputs_q.set(block.attention.attn_block.inputs_q.value)
            block_proj.q_attn_block.inputs_k.set(block.attention.attn_block.inputs_k.value)
            block_proj.q_attn_block.inputs_v.set(block.attention.attn_block.inputs_v.value)
            block_proj.Q_attn_out.biases.set(block.attention.W_attn_out.biases.value)
            block_proj.Q_mlp1.weights.set(block.mlp.W_mlp1.weights.value)
            block_proj.Q_mlp1.biases.set(block.mlp.W_mlp1.biases.value)
            block_proj.Q_mlp2.weights.set(block.mlp.W_mlp2.weights.value)
            block_proj.Q_mlp2.biases.set(block.mlp.W_mlp2.biases.value)
            
            ## pin/tie feedback synapses to transpose of forward ones

            block.attention.E_attn.weights.set(jnp.transpose(block.attention.W_attn_out.weights.value))
            block.mlp.E_mlp.weights.set(jnp.transpose(block.mlp.W_mlp2.weights.value))  
            block.mlp.E_mlp1.weights.set(jnp.transpose(block.mlp.W_mlp1.weights.value))
  
        self.projection.Q_out.weights.set(self.output.W_out.weights.value)
        self.projection.Q_out.biases.set(self.output.W_out.biases.value)
        self.projection.q_target.j_td.set(jnp.zeros((config.batch_size * config.seq_len, config.vocab_size)))
        
        ## pin/tie feedback synapses to transpose of forward ones
       
        self.output.E_out.weights.set(jnp.transpose(self.output.W_out.weights.value))
        
        ## Perform P-step (projection step)
        self.circuit.clamp_input(obs)
        self.circuit.clamp_infer_target(_lab)
        self.circuit.project(t=0., dt=1.) 
        ## initialize dynamics of generative model latents to projected states for the errors it's 0
        self.blocks[0].attention.z_qkv.z.set(self.projection.blocks[0].q_qkv.z.value)
        self.blocks[0].mlp.z_mlp.z.set(self.projection.blocks[0].q_mlp.z.value)
        self.blocks[0].mlp.z_mlp2.z.set(self.projection.blocks[0].q_mlp2.z.value)
        self.output.e_out.dmu.set(self.projection.eq_target.dmu.value)
        self.output.e_out.dtarget.set(self.projection.eq_target.dtarget.value)
        
        
        ## get projected prediction (from the P-step)
        y_mu_inf = self.q_target.z.value

        EFE = 0. ## expected free energy
        y_mu = 0.
        if adapt_synapses:
            for ts in range(0, self.T):
                self.circuit.clamp_input(obs) ## clamp input data to z_embed & q_embed input compartments
                self.circuit.clamp_target(_lab) ## clamp target data to z_target
                self.circuit.advance(t=ts, dt=1.)

            y_mu = self.output.e_out.mu.value ## get settled prediction

            L1 = self.embedding.e_embed.L.value
            L4 = self.output.e_out.L.value
            # Sum errors from ALL blocks
            block_errors = 0.
            for i in range(self.n_layers):
                block = self.blocks[i]
                block_errors += block.attention.e_attn.L.value + block.mlp.e_mlp.L.value + block.mlp.e_mlp1.L.value

            EFE = L4 + block_errors + L1

            if adapt_synapses == True:
                self.circuit.evolve()
                self.circuit.evolve_embedding()
        ## skip E/M steps if just doing test-time inference
        return y_mu_inf, y_mu, EFE

    def get_latents(self):
        return self.q_out.z.value

   