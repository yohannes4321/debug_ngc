class Config:
    SEED = 42
    n_embed = 32
    seq_len = 12
    batch_size = 5
    vocab_size = 11710 # data vocab size + special tokens = 11706 + 4
    n_heads = 2
    n_layers = 6
    dropout_rate = 0.1
    eta = 0.0001
    exp_dir = "exp" 
    pos_learnable = True
    optim_type = "adam"
    num_iter = 20
    wub = 0.01
    wlb = -0.01
    tau_m = 10.
    act_fx = "tanh"