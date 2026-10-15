import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import math

class LanguageModelingDataset(Dataset):
    def __init__(self, file_path, tokenizer, context_length, split='train'):
        super().__init__()
        with open(file_path, 'r') as f:
            text = f.read()

        tokens = tokenizer.encode(str(text))
        
        if split == 'train':
            self.tokens = torch.tensor(tokens[ :int(0.8*len(tokens))])
        elif split =='valid':
            self.tokens = torch.tensor(tokens[int(0.8*len(tokens)): int(0.9*len(tokens))])
        else:
            self.tokens = torch.tensor(tokens[int(0.9*len(tokens)): ])
        # self.tokens = torch.tensor(tokens) if isinstance(tokens, list) else tokens
        self.context_length = context_length
    def __len__(self):
        return len(self.tokens) - self.context_length
    def __getitem__(self, index):
        start_idx = index
        end_idx = start_idx + self.context_length
        x = self.tokens[start_idx: end_idx]
        y = self.tokens[start_idx+1: end_idx+1]
        return x, y 

class PositionalEncoding(nn.Module):
    def __init__(self, max_len, dmodel):
        super().__init__()
        positions = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, dmodel, 2) * (-math.log(10000)/dmodel))
        penc = torch.zeros(1, max_len, dmodel)
        penc[0, :, 0::2] = torch.sin(positions*div)
        penc[0, :, 1::2] = torch.cos(positions*div)
        self.register_buffer('penc', penc)
    def forward(self, x):
        x = x + self.penc[:, :x.shape[1], :]
        return x

class AttentionHead(nn.Module):
    def __init__(self, dmodel, headsize, dropout):
        super().__init__()
        self.dmodel = dmodel
        self.wk = nn.Linear(dmodel, headsize)
        self.wv = nn.Linear(dmodel, headsize)
        self.wo = nn.Linear(dmodel, headsize)
        self.wq = nn.Linear(dmodel, headsize)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        _, T, _ = x.shape
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)
        qk = q@k.transpose(-1, -2)
        tril = torch.tril(torch.ones(T, T))
        self.register_buffer('tril', tril)
        qk = qk.masked_fill(tril == 0, float('-inf')) * q.shape[-1]**-0.5
        qk = F.softmax(qk, dim=-1)
        qk = self.dropout(qk)
        qkv = qk@v
        return qkv

class MultiHeadAttention(nn.Module):
    def __init__(self, numheads, dmodel, dropout):
        super().__init__()
        head_dim = dmodel//numheads
        self.heads = nn.ModuleList([AttentionHead(dmodel, head_dim, dropout) for i in range(numheads)])
        self.lin_proj = nn.Linear(head_dim*numheads, dmodel)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        z = []
        for head in self.heads:
            h = head
            x_h = h(x)
            z.append(x_h)
        return self.dropout(self.lin_proj(torch.concat([s for s in z], dim=-1)))       

class LayerNorm(nn.Module):
    def __init__(self, dmodel):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dmodel))
        self.beta = nn.Parameter(torch.zeros(dmodel))
        self.eps = 1e-7
    def forward(self, x):
        mean = x.mean(dim=-1).unsqueeze(-1)
        var = x.var(dim=-1).unsqueeze(-1)
        layernorm = (x-mean)/(var+self.eps)**0.5
        return self.gamma*layernorm + self.beta     # element-wise multipplication

class FFN(nn.Module):
    def __init__(self, dmodel, dropout):
        super().__init__()
        self.l1 = nn.Linear(dmodel, dmodel*4)
        self.l2 = nn.Linear(dmodel*4, dmodel)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        x = self.dropout(self.l2(torch.relu(self.l1(x))))
        return x

class Block(nn.Module):
    def __init__(self, dmodel, numheads, dropout):
        super().__init__()
        self.mha = MultiHeadAttention(numheads, dmodel, dropout)
        self.ln1 = LayerNorm(dmodel)
        self.ln2 = LayerNorm(dmodel)
        self.ffn = FFN(dmodel, dropout)
    def forward(self, x):
        x = x + self.mha(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

class Transformer(nn.Module):
    def __init__(self, num_layers, max_seq_len, dmodel, numheads, vocab_size, dropout):
        super().__init__()
        self.enc = nn.Embedding(vocab_size, dmodel)
        nn.init.normal_(self.enc.weight, mean=0.0, std=0.02)
        self.penc = PositionalEncoding(max_seq_len, dmodel)
        self.lm_head = nn.Linear(dmodel, vocab_size, bias=False)
        self.layers = nn.Sequential(*[
            Block(dmodel, numheads, dropout) for _ in range(num_layers)
        ])
        self.ln_final = LayerNorm(dmodel)
        self.lm_head.weight = self.enc.weight   # weight tying
        
    def forward(self, x):
        x = self.enc(x)
        x = self.penc(x)
        # x = x + penc
        x = self.layers(x)
        x = self.ln_final(x)
        x = self.lm_head(x)
        return x

    def generate(self, x, context_length, max_tokens, decoding='greedy', t=0.7, k=5):
        """
        Generates tokens autoregressively.
        
        Args:
            x (torch.Tensor): Input sequence of token IDs of shape (batch_size, seq_len).
            max_tokens (int): Number of new tokens to generate.
            
        Returns:
            torch.Tensor: The input sequence concatenated with the generated tokens.
        """
        # Set to evaluation mode to disable dropout/batchnorm if any
        self.eval()
        with torch.no_grad():
            for _ in range(max_tokens):
                x = x[:, -context_length:]

                # Forward pass to get logits
                logits = self(x)
                
                # Get the logits for the last token in the sequence
                next_token_logits = logits[:, -1, :] 
                
                # Greedy decoding: pick the token with the highest logit
                if decoding == 'greedy':
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                elif decoding == 'temp':
                    scaled_logits = next_token_logits/t     # t<1 -> decisive output, t>1 -> creative output
                    scaled_logits = F.softmax(scaled_logits)
                    next_token = torch.multinomial(scaled_logits, num_samples=1)
                elif decoding == 'top-k':
                    top_k_v, top_k_i = torch.topk(next_token_logits, k)
                    next_token_logits = torch.where(next_token_logits < top_k_v[:, -1:], float('-inf'), next_token_logits)
                    next_token_logits = F.softmax(next_token_logits)
                    next_token = torch.multinomial(next_token_logits, num_samples=1)
                else:
                    raise ValueError("Decoding scheme not supported. Choose from [greedy, temp, top-k]", decoding)               
                
                # Append the predicted token to the sequence
                x = torch.cat([x, next_token], dim=1)
                
        return x