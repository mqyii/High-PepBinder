from transformers import PretrainedConfig


class ESMCConfig(PretrainedConfig):
    model_type = "esmc"

    def __init__(
        self,
        vocab_size: int = 64,
        hidden_size: int = 1152,
        num_attention_heads: int = 18,
        num_hidden_layers: int = 36,
        num_labels: int = 1,
        problem_type: str = "regression",
        dropout: float = 0.1,
        initializer_range: float = 0.02,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_labels = num_labels
        self.problem_type = problem_type
        self.dropout = dropout
        self.initializer_range = initializer_range
        self.tie_word_embeddings = False