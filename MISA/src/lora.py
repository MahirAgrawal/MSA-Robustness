import torch
import torch.nn as nn
import math


class LoRALinear(nn.Module):
    """
    Linear layer with LoRA adaptation.
    Replaces W with W + BA where B and A are low-rank matrices.
    """
    def __init__(self, in_features, out_features, lora_rank=8, lora_alpha=16, 
                 lora_dropout=0.1, bias=True):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.scaling = lora_alpha / lora_rank
        
        # Main weight matrix (frozen during training if using LoRA)
        self.weight = nn.Parameter(torch.zeros(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
            
        # LoRA matrices  (standard convention: A is rank×in, B is out×rank)
        # Forward: x @ A.T @ B.T  ==  F.linear(F.linear(x, A), B)
        # Merge:   W + B @ A  (both already out×in after the matmul)
        self.lora_A = nn.Parameter(torch.empty(lora_rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, lora_rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))  # B starts zero → no change at init
        
        # Dropout for regularization
        self.dropout = nn.Dropout(lora_dropout)
        
        # Initialize weight with standard linear initialization
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        """
        Forward pass: output = x @ (W + scaling * B @ A)^T + b
        """
        # Main linear transformation
        result = torch.nn.functional.linear(x, self.weight, self.bias)
        
        # LoRA path: F.linear(x, A) → shape (…, rank), then F.linear(…, B) → (…, out)
        lora_result = torch.nn.functional.linear(
            torch.nn.functional.linear(self.dropout(x), self.lora_A),
            self.lora_B
        ) * self.scaling
        
        return result + lora_result

    def merge_weights(self):
        """Merge LoRA weights into the main weight matrix (for inference optimization)"""
        # lora_B is (out×rank), lora_A is (rank×in) → product is (out×in), same shape as self.weight
        merged_weight = self.weight.data + (self.lora_B @ self.lora_A) * self.scaling
        self.weight.data = merged_weight


class LoRASequential(nn.Module):
    """
    Sequential module with optional LoRA adaptation on linear layers
    """
    def __init__(self, *args, lora_config=None, **kwargs):
        super().__init__()
        self.layers = nn.Sequential(*args, **kwargs)
        self.lora_config = lora_config or {}
        self._apply_lora()
        
    def _apply_lora(self):
        """Apply LoRA to all linear layers in the sequential module"""
        if not self.lora_config.get('enabled', True):
            return
            
        for name, module in self.layers.named_modules():
            if isinstance(module, nn.Linear):
                parent_name = name.rsplit('.', 1)[0] if '.' in name else ''
                # Replace linear layers with LoRA versions
                # This would require more complex surgery, so we keep original structure
                # and add LoRA through a wrapper approach instead
                
    def forward(self, x):
        return self.layers(x)


def add_lora_to_linear(module, lora_rank=8, lora_alpha=16, lora_dropout=0.1):
    """
    Recursively add LoRA to all linear layers in a module
    """
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            # Replace linear layer with LoRA version
            lora_layer = LoRALinear(
                child.in_features,
                child.out_features,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                bias=child.bias is not None
            )
            # Copy original weights
            lora_layer.weight.data = child.weight.data.clone()
            if child.bias is not None:
                lora_layer.bias.data = child.bias.data.clone()
            
            # Replace the layer
            setattr(module, name, lora_layer)
        else:
            # Recursively apply to child modules
            add_lora_to_linear(child, lora_rank, lora_alpha, lora_dropout)


def merge_lora_weights(module):
    """
    Merge LoRA weights into main weights for all LoRALinear layers
    Useful for inference optimization
    """
    for name, child in module.named_children():
        if isinstance(child, LoRALinear):
            child.merge_weights()
        else:
            merge_lora_weights(child)


def get_lora_params(module):
    """
    Get all LoRA parameters from a module
    Useful for training only LoRA parameters
    """
    lora_params = []
    for name, param in module.named_parameters():
        if 'lora_' in name:
            lora_params.append(param)
    return lora_params


class LoRAConfig:
    """Configuration class for LoRA"""
    def __init__(self, 
                 lora_rank=8,
                 lora_alpha=16,
                 lora_dropout=0.1,
                 target_modules=None,
                 lora_enabled=True,
                 train_only_lora=False):
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.target_modules = target_modules or ['project', 'private', 'shared', 'fusion']
        self.lora_enabled = lora_enabled
        self.train_only_lora = train_only_lora
