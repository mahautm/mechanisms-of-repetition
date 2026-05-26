import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class HookedModel(torch.nn.Module):
    def __init__(self, model, layer=None):
        super().__init__()
        self.model = model
        self.attn_outputs = []
        self.mlp_outputs = []  # Store MLP outputs separately
        self.final_layer_logits = None  # Special variable for final layer logits
        self.all_logits = []  # Accumulate logits across all generation steps
        self.hooks = []
        
        # Determine whether to hook attention or MLP based on layer name
        if layer and 'mlp' in layer:
            self._register_mlp_hooks(layer)
        else:
            self._register_attention_hooks(layer)
        
        self._register_final_layer_hook()  # Always hook final layer
        self.unembed_module = self._identify_unembed_module()

    def _identify_unembed_module(self):
        if hasattr(self.model, 'lm_head'):
            return self.model.lm_head
        elif hasattr(self.model, 'head'):
            return self.model.head
        elif hasattr(self.model, 'embed_out'):
            return self.model.embed_out
        else:
            raise ValueError("Model does not have a recognizable unembedding layer.")

    def _register_final_layer_hook(self):
        """Always register a hook on the final layer (lm_head) to capture logits"""
        if hasattr(self.model, 'lm_head'):
            hook = self.model.lm_head.register_forward_hook(self._final_layer_hook_fn())
            self.hooks.append(hook)
        elif hasattr(self.model, 'head'):
            hook = self.model.head.register_forward_hook(self._final_layer_hook_fn())
            self.hooks.append(hook)
        elif hasattr(self.model, 'embed_out'):
            hook = self.model.embed_out.register_forward_hook(self._final_layer_hook_fn())
            self.hooks.append(hook)

    def _final_layer_hook_fn(self):
        """Hook function specifically for capturing final layer logits"""
        def fn(module, input, output):
            # Store the most recent logits (for backward compatibility)
            self.final_layer_logits = output
            # Also accumulate all logits across generation steps
            self.all_logits.append(output.clone().detach())
        return fn

    def _register_mlp_hooks(self, layer=None):
        """Register hooks on MLP layers"""
        for name, module in self.model.named_modules():
            # Handle different model architectures
            is_mlp_layer = False
            
            # For Pythia models: gpt_neox.layers.X.mlp.dense_4h_to_h (final linear layer of MLP)
            if 'gpt_neox' in name and 'mlp' in name and 'dense_4h_to_h' in name and (layer is None or layer in name):
                is_mlp_layer = True
            # For Mistral/LLaMA models: model.layers.X.mlp.down_proj (final linear layer of MLP)
            elif 'model.layers' in name and 'mlp' in name and 'down_proj' in name and (layer is None or layer in name):
                is_mlp_layer = True
            # For other architectures - add as needed
            # TODO: Add more model-specific MLP layer patterns
            
            if is_mlp_layer:
                hook = module.register_forward_hook(self._mlp_hook_fn(name))
                self.hooks.append(hook)
                print(f"Registered MLP hook on: {name}")

    def _mlp_hook_fn(self, name):
        """Hook function for MLP layers"""
        def fn(module, input, output):
            self.mlp_outputs.append((name, output))
        return fn

    def _register_attention_hooks(self, layer=None, only_layers=None):
        for name, module in self.model.named_modules():
            if only_layers and name not in only_layers:
                continue
            if hasattr(module, 'attention') and (layer is None or layer == name):
                hook = module.attention.register_forward_hook(self._hook_fn(name))
                self.hooks.append(hook)
            elif hasattr(module, 'self_attn') and (layer is None or layer in name):
                hook = module.self_attn.register_forward_hook(self._hook_fn(name))
                self.hooks.append(hook)

    def _hook_fn(self, name):
        def fn(module, input, output):
            self.attn_outputs.append((name, output))
        return fn

    def clear(self):
        self.attn_outputs.clear()
        self.mlp_outputs.clear()  # Clear MLP outputs too
        self.final_layer_logits = None  # Clear final layer logits
        self.all_logits.clear()  # Clear accumulated logits

    def forward(self, *args, **kwargs):
        self.clear()
        return self.model(*args, **kwargs)

    def generate(self, *args, **kwargs):
        self.clear()
        return self.model.generate(*args, **kwargs)

    def add_hooks(self, hooks_names):
        for name, module in self.model.named_modules():
            if name in hooks_names:
                hook = module.register_forward_hook(self._hook_fn(name))
                self.hooks.append(hook)
    def unembed(self, x):
        return self.unembed_module(x)
    
    def get_final_logits(self):
        """Convenient method to access final layer logits"""
        return self.final_layer_logits
    
    def get_all_logits(self):
        """Get all accumulated logits from generation process"""
        if not self.all_logits:
            return None
        # Concatenate along sequence dimension if multiple forward passes
        return torch.cat(self.all_logits, dim=1) if len(self.all_logits) > 1 else self.all_logits[0]
    
    def get_logits_at_positions(self, positions):
        """Get logits at specific sequence positions
        
        Args:
            positions: List of positions to extract (relative to original sequence length)
        
        Returns:
            Tensor of shape [batch, len(positions), vocab_size] if available
        """
        all_logits = self.get_all_logits()
        if all_logits is None:
            return None
        
        # Handle positions that might be out of bounds
        seq_len = all_logits.shape[1]
        valid_positions = [pos for pos in positions if 0 <= pos < seq_len]
        
        if not valid_positions:
            return None
            
        return all_logits[:, valid_positions, :]
    
    def get_logits_count(self):
        """Get the number of logit tensors captured during generation"""
        return len(self.all_logits)
    
    def has_final_logits(self):
        """Check if final layer logits are available"""
        return self.final_layer_logits is not None
    
    def get_mlp_outputs(self):
        """Get captured MLP outputs"""
        return self.mlp_outputs
    
    def has_mlp_outputs(self):
        """Check if MLP outputs are available"""
        return bool(self.mlp_outputs)
    
    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

def load_model_and_tokenizer(model_name, revision=None, use_bfloat16=False):
    if use_bfloat16:
        model = AutoModelForCausalLM.from_pretrained(model_name, revision=revision, torch_dtype=torch.bfloat16, trust_remote_code=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, revision=revision, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # Check if model needs right padding based on config
    if hasattr(model.config, 'pad_side'):
        tokenizer.padding_side = model.config.pad_side
    elif hasattr(model.config, 'padding_side'):
        tokenizer.padding_side = model.config.padding_side
    else:
        # Default to left padding for most causal LMs, right padding for encoder-decoder
        tokenizer.padding_side = 'left' if model.config.is_decoder else 'right'
    
    # Set pad token for open-ended generation
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Ensure model config matches tokenizer
    model.config.pad_token_id = tokenizer.pad_token_id
    
    # Set model to evaluation mode for generation
    model.eval()
    
    return model, tokenizer

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
