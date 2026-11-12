import torch
import math

def train_model(model, dataloader, loss_fn, optimizer, epoch):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.train()
    train_loss = 0
    train_acc = 0
    total_tokens = 0
    for batch, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)
        logits = model(x)
        y = y.view(-1)
        logits = logits.view(-1, logits.size(-1))
        loss = loss_fn(logits, y)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        pred_tokens = torch.argmax(logits, dim=-1)
        
        train_loss += loss.item()*y.numel()
        train_acc += (pred_tokens == y).sum().item()
        total_tokens += y.numel()

    avg_loss = train_loss/total_tokens
    avg_acc = train_acc/total_tokens
    perplexity = math.exp(avg_loss)
    print(f"Epoch {epoch} | Loss: {avg_loss:.4f} | Acc: {avg_acc:.4f} | Perplexity: {perplexity:.4f}")
    return avg_loss, avg_acc