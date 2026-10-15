import torch
import math

def evaluate_model(model, dataloader, loss_fn, epoch):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.eval()
    eval_loss = 0
    eval_acc = 0
    perplexity = 0
    total_tokens = 0
    for batch, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            logits = model(x)
            y = y.view(-1)
            logits = logits.view(-1, logits.shape[-1])
            loss = loss_fn(logits, y)

            pred_tokens = torch.argmax(logits, dim=-1)

            eval_loss += loss.item()*y.numel()
            eval_acc += (pred_tokens == y).sum().item()
            total_tokens += y.numel()

    eval_loss = eval_loss/total_tokens
    eval_acc = eval_acc/total_tokens
    perplexity = math.exp(eval_loss)
    print(f"Epoch {epoch} | Loss: {eval_loss:.4f} | Acc: {eval_acc:.4f} | Perplexity: {perplexity:.4f}")
    return eval_loss, eval_acc, perplexity