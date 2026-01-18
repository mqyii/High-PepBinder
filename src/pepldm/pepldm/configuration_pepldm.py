from transformers.configuration_utils import PretrainedConfig


class PepLDMConfig(PretrainedConfig):
    model_type = "pepldm"

    def __init__(
        self,
        objective: str = "epsilon",
        input_dim: int = 1152,
        hidden_size: int = 768,
        depth: int = 16,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        use_self_conditioning: bool = True,
        timestep_embedding_strategy: str = "sinusoidal",
        use_skip_connect: bool = True,
        attention_mode: str = "standard",
        conditional: bool = True,
        cond_dim: int = 2304,
    ):
        super().__init__()

        self.objective = objective
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.use_self_conditioning = use_self_conditioning
        self.timestep_embedding_strategy = timestep_embedding_strategy
        self.use_skip_connect = use_skip_connect
        self.attention_mode = attention_mode
        self.conditional = conditional
        self.cond_dim = cond_dim
