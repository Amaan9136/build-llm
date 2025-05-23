import torch
from torch.utils.data import Dataset
import json
import os
import random

class TextDataset(Dataset):
    """
    Dataset for training language models on text data.
    Supports both plain text files and JSON format with text field.
    """
    def __init__(self, file_path, tokenizer, context_length=1024):
        """
        Initialize the dataset.
        
        Args:
            file_path (str): Path to the data file (text or JSON)
            tokenizer: Tokenizer object with encode method
            context_length (int): Maximum context length for training
        """
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.context_length = context_length
        
        # Load data
        self.text_data = self._load_data(file_path)
        
        # Tokenize full text
        self.tokens = self.tokenizer.encode(self.text_data)
        
        # Calculate number of samples based on context length
        self.num_samples = max(1, len(self.tokens) - context_length)
    
    def _load_data(self, file_path):
        """Load data from file (text or JSON)"""
        extension = os.path.splitext(file_path)[1].lower()
        
        if extension == '.txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        
        elif extension == '.json':
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Handle different JSON formats
            if isinstance(data, list):
                combined_text = ""
                for item in data:
                    if isinstance(item, dict) and 'text' in item:
                        combined_text += item['text'] + "\n\n"
                    elif isinstance(item, str):
                        combined_text += item + "\n\n"
                return combined_text
            
            elif isinstance(data, dict) and 'text' in data:
                return data['text']
            
            else:
                # Fallback: convert the entire JSON to string
                return json.dumps(data)
        
        else:
            raise ValueError(f"Unsupported file format: {extension}")
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Get a slice of tokens starting at the index
        x = self.tokens[idx:idx + self.context_length]
        y = self.tokens[idx + 1:idx + self.context_length + 1]
        
        # Pad if necessary
        if len(x) < self.context_length:
            pad_length = self.context_length - len(x)
            x = x + [0] * pad_length
            y = y + [0] * pad_length
        
        # Convert to tensors
        x = torch.tensor(x, dtype=torch.long)
        y = torch.tensor(y, dtype=torch.long)
        
        return {'input_ids': x, 'labels': y}


class CharTokenizer:
    """
    Simple character-level tokenizer for LLM training.
    Maps each character to a unique integer.
    """
    def __init__(self):
        self.char_to_idx = {}
        self.idx_to_char = {}
        self.vocab_size = 0
    
    def fit(self, text):
        """Build vocabulary from text"""
        unique_chars = sorted(set(text))
        self.char_to_idx = {c: i for i, c in enumerate(unique_chars)}
        self.idx_to_char = {i: c for i, c in enumerate(unique_chars)}
        self.vocab_size = len(unique_chars)
        return self
    
    def encode(self, text):
        """Convert text to token IDs"""
        return [self.char_to_idx.get(c, 0) for c in text]
    
    def decode(self, ids):
        """Convert token IDs back to text"""
        return ''.join([self.idx_to_char.get(i, '') for i in ids])


class LLMTrainer:
    """
    Trainer class for language models.
    Handles model creation, training, and inference.
    """
    # Add to LLMTrainer.__init__ in python_classes/text_dataset.py
    def __init__(self, n_layer=12, n_head=12, n_embd=768, n_ctx=1024, vocab_size=50257, 
                use_gpu=True, fp16=False, mixed_precision=False, device_id=0):
        """
        Initialize the trainer with model configuration.
        
        Args:
            n_layer (int): Number of transformer layers
            n_head (int): Number of attention heads
            n_embd (int): Embedding dimension
            n_ctx (int): Context length
            vocab_size (int): Vocabulary size
            use_gpu (bool): Whether to use GPU for training
            fp16 (bool): Whether to use FP16 precision
            mixed_precision (bool): Whether to use mixed precision training
            device_id (int): GPU device ID to use
        """
        self.config = type('Config', (), {
            'n_layer': n_layer,
            'n_head': n_head,
            'n_embd': n_embd,
            'n_ctx': n_ctx,
            'vocab_size': vocab_size
        })
        
        self.device = torch.device(f'cuda:{device_id}' if torch.cuda.is_available() and use_gpu else 'cpu')
        self.model = self._create_model()
        self.tokenizer = None
        
        # Add support for mixed precision training
        self.scaler = None
        if mixed_precision and self.device.type == 'cuda':
            self.scaler = torch.cuda.amp.GradScaler()

    def _create_model(self):
        """Create a GPT-style language model"""
        # For simplicity, we'll use a basic transformer model
        # In a real implementation, you'd use a proper GPT model
        from torch import nn
        import math
        
        class GPTConfig:
            def __init__(self, vocab_size, n_ctx, n_embd, n_head, n_layer):
                self.vocab_size = vocab_size
                self.n_ctx = n_ctx
                self.n_embd = n_embd
                self.n_head = n_head
                self.n_layer = n_layer
        
        class CausalSelfAttention(nn.Module):
            def __init__(self, config):
                super().__init__()
                assert config.n_embd % config.n_head == 0
                self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
                self.c_proj = nn.Linear(config.n_embd, config.n_embd)
                self.n_head = config.n_head
                self.n_embd = config.n_embd
                self.register_buffer(
                    "mask", 
                    torch.tril(torch.ones(config.n_ctx, config.n_ctx))
                    .view(1, 1, config.n_ctx, config.n_ctx)
                )
        
            def forward(self, x):
                B, T, C = x.size()
                
                # Calculate query, key, values
                q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
                k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
                q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
                v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
                
                # Causal self-attention
                att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
                att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
                att = torch.softmax(att, dim=-1)
                y = att @ v
                y = y.transpose(1, 2).contiguous().view(B, T, C)
                
                # Output projection
                y = self.c_proj(y)
                return y
        
        class Block(nn.Module):
            def __init__(self, config):
                super().__init__()
                self.ln_1 = nn.LayerNorm(config.n_embd)
                self.attn = CausalSelfAttention(config)
                self.ln_2 = nn.LayerNorm(config.n_embd)
                self.mlp = nn.Sequential(
                    nn.Linear(config.n_embd, 4 * config.n_embd),
                    nn.GELU(),
                    nn.Linear(4 * config.n_embd, config.n_embd)
                )
            
            def forward(self, x):
                x = x + self.attn(self.ln_1(x))
                x = x + self.mlp(self.ln_2(x))
                return x
        
        class GPTModel(nn.Module):
            def __init__(self, config):
                super().__init__()
                self.transformer = nn.ModuleDict(dict(
                    wte = nn.Embedding(config.vocab_size, config.n_embd),
                    wpe = nn.Embedding(config.n_ctx, config.n_embd),
                    h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                    ln_f = nn.LayerNorm(config.n_embd)
                ))
                self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
                self.transformer.wte.weight = self.lm_head.weight
                
                self.config = config
            
            def forward(self, idx, labels=None):
                device = idx.device
                b, t = idx.size()
                pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)
                
                # Forward the GPT model
                tok_emb = self.transformer.wte(idx)
                pos_emb = self.transformer.wpe(pos)
                x = tok_emb + pos_emb
                
                for block in self.transformer.h:
                    x = block(x)
                
                x = self.transformer.ln_f(x)
                logits = self.lm_head(x)
                
                loss = None
                if labels is not None:
                    loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
                
                return {'logits': logits, 'loss': loss}
            
            def generate(self, idx, max_new_tokens, temperature=1.0):
                for _ in range(max_new_tokens):
                    # Crop context if needed
                    idx_cond = idx if idx.size(1) <= self.config.n_ctx else idx[:, -self.config.n_ctx:]
                    
                    # Forward pass
                    with torch.no_grad():
                        logits = self.forward(idx_cond)['logits']
                    
                    # Get logits for the last token and apply temperature
                    logits = logits[:, -1, :] / temperature
                    
                    # Apply softmax to get probabilities
                    probs = torch.softmax(logits, dim=-1)
                    
                    # Sample from the distribution
                    idx_next = torch.multinomial(probs, num_samples=1)
                    
                    # Append to the sequence
                    idx = torch.cat((idx, idx_next), dim=1)
                
                return idx
        
        # Create and return model
        config = GPTConfig(
            vocab_size=self.config.vocab_size,
            n_ctx=self.config.n_ctx,
            n_embd=self.config.n_embd,
            n_head=self.config.n_head,
            n_layer=self.config.n_layer
        )
        model = GPTModel(config)
        model = model.to(self.device)
        return model
    
    def _create_char_tokenizer(self, file_path):
        """Create and fit a character-level tokenizer"""
        tokenizer = CharTokenizer()
        
        # Load data for tokenizer
        # Handle case when file_path is a list (multiple files)
        if isinstance(file_path, list):
            all_text = ""
            for path in file_path:
                all_text += self._load_text_from_file(path) + "\n\n"
            text = all_text
        else:
            # Single file case
            text = self._load_text_from_file(file_path)
        
        # Fit tokenizer
        tokenizer.fit(text)
        
        # Update vocab size in config
        self.config.vocab_size = tokenizer.vocab_size
        
        # Recreate model with new vocab size
        self.model = self._create_model()
        
        return tokenizer

    def _load_text_from_file(self, file_path):
        """Helper to load text from a single file"""
        extension = os.path.splitext(file_path)[1].lower()
        
        if extension == '.txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        
        elif extension == '.json':
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, list):
                text = ""
                for item in data:
                    if isinstance(item, dict) and 'text' in item:
                        text += item['text'] + "\n\n"
                    elif isinstance(item, str):
                        text += item + "\n\n"
                return text
            
            elif isinstance(data, dict) and 'text' in data:
                return data['text']
            
            else:
                return json.dumps(data)
        
        else:
            raise ValueError(f"Unsupported file format: {extension}")

    def save_model(self, save_dir):
        """Save model and tokenizer to directory"""
        os.makedirs(save_dir, exist_ok=True)
        
        # Save model state
        torch.save(self.model.state_dict(), os.path.join(save_dir, 'model.pt'))
        
        # Save config
        config_dict = {
            'n_layer': self.config.n_layer,
            'n_head': self.config.n_head,
            'n_embd': self.config.n_embd,
            'n_ctx': self.config.n_ctx,
            'vocab_size': self.config.vocab_size
        }
        with open(os.path.join(save_dir, 'config.json'), 'w') as f:
            json.dump(config_dict, f)
        
        # Save tokenizer if available
        if self.tokenizer is not None:
            tokenizer_dict = {
                'char_to_idx': self.tokenizer.char_to_idx,
                'idx_to_char': self.tokenizer.idx_to_char,
                'vocab_size': self.tokenizer.vocab_size
            }
            with open(os.path.join(save_dir, 'tokenizer.json'), 'w') as f:
                json.dump(tokenizer_dict, f)
    
    def load_model(self, load_dir):
        """Load model and tokenizer from directory"""
        # Load config
        with open(os.path.join(load_dir, 'config.json'), 'r') as f:
            config_dict = json.load(f)
        
        # Update config
        self.config = type('Config', (), config_dict)
        
        # Recreate model
        self.model = self._create_model()
        
        # Load model state
        self.model.load_state_dict(torch.load(
            os.path.join(load_dir, 'model.pt'),
            map_location=self.device
        ))
        
        # Load tokenizer if available
        tokenizer_path = os.path.join(load_dir, 'tokenizer.json')
        if os.path.exists(tokenizer_path):
            with open(tokenizer_path, 'r') as f:
                tokenizer_dict = json.load(f)
            
            self.tokenizer = CharTokenizer()
            self.tokenizer.char_to_idx = {k: int(v) for k, v in tokenizer_dict['char_to_idx'].items()}
            self.tokenizer.idx_to_char = {int(k): v for k, v in tokenizer_dict['idx_to_char'].items()}
            self.tokenizer.vocab_size = tokenizer_dict['vocab_size']
    
    def generate(self, input_ids, max_length=100, temperature=0.8):
        """Generate text from input_ids"""
        self.model.eval()
        input_ids = input_ids.unsqueeze(0) if input_ids.dim() == 1 else input_ids
        
        return self.model.generate(input_ids, max_length, temperature)[0]