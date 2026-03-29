import torch
import torch.nn as nn

class LinearLoRA(nn.Module):
    def __init__(self, original_linear, rank):
        super(LinearLoRA, self).__init__()
        self.original_linear = original_linear
        model_dim = original_linear.out_features
        
        # Low-rank matrices
        self.A_ = nn.Parameter(torch.randn(model_dim, rank))
        self.B_ = nn.Parameter(torch.randn(rank, model_dim))

    def forward(self, x):
        # Original linear transformation
        transformed_x = self.original_linear(x)
        # Low-rank adaptation
        low_rank_x = x @ self.A_ @ self.B_
        # Adding the original transformation and the low-rank adaptation
        return transformed_x + low_rank_x

class LoRAWrapper(nn.Module):
    def __init__(self, model, rank):
        super(LoRAWrapper, self).__init__()
        self.model = model
        
        # Wrap the final projection layer (o_proj) in each attention layer with LoRA
        for layer in self.model.model.layers:
            attn = layer.self_attn  # Assuming self_attn is where the o_proj is located
            attn.o_proj = LinearLoRA(attn.o_proj, rank)  # Applying LoRA to the final output projection

    def forward(self, input_ids, attention_mask=None):
        # Forward pass through the modified model
        return self.model(input_ids, attention_mask=attention_mask)





