import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import argparse

# define the transformer-based mask predictor
class MaskedDiffusionModel(nn.Module):
    def __init__(self, vocab_size, emb_dim, nhead, num_layers, dropout=0.1, mask_token_id=0):
        super(MaskedDiffusionModel, self).__init__()
        self.mask_token_id = mask_token_id
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=nhead, dropout=dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(emb_dim, vocab_size)
        
    def forward(self, x):
        emb = self.embedding(x)
        out = self.transformer(emb)
        logits = self.fc(out)
        return logits

# randomly mask tokens in an input sequence
def mask_tokens(x, mask_token_id, t):
    mask = (torch.rand_like(x, dtype=torch.float) < t).long()
    x_masked = x.clone()
    x_masked[mask == 1] = mask_token_id
    return x_masked, mask

# a single training step implementing the forward (masking) process
def train_step(model, optimizer, x, mask_token_id, device):
    t = random.uniform(0, 1)
    x_masked, mask = mask_tokens(x, mask_token_id, t)
    
    x_masked = x_masked.transpose(0, 1).to(device)
    logits = model(x_masked)
    
    x = x.transpose(0, 1).to(device)
    
    loss_all = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), x.reshape(-1), reduction='none')
    mask = mask.transpose(0, 1).to(device).reshape(-1).float()
    loss = (loss_all * mask).sum() / (mask.sum() + 1e-8)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()

# given a prompt, iteratively “denoise” a fully masked response
def sample(model, prompt, response_len, num_steps, mask_token_id, device):
    model.eval()
    with torch.no_grad():
        prompt = prompt.to(device)
        batch_size = prompt.shape[0]
        response = torch.full((batch_size, response_len), mask_token_id, dtype=torch.long, device=device)
        
        for step in range(num_steps):
            x_combined = torch.cat([prompt, response], dim=1)
            x_input = x_combined.transpose(0, 1)
            logits = model(x_input)
            response_logits = logits[prompt.shape[1]:, 0, :]
            predictions = torch.argmax(response_logits, dim=-1)
            response = predictions.unsqueeze(0)
            
            re_mask_prob = 0.2
            random_mask = (torch.rand(response.shape, device=device) < re_mask_prob).long()
            response = torch.where(random_mask == 1, torch.full_like(response, mask_token_id), response)
        
        x_combined = torch.cat([prompt, response], dim=1)
        x_input = x_combined.transpose(0, 1)
        logits = model(x_input)
        response_logits = logits[prompt.shape[1]:, 0, :]
        final_predictions = torch.argmax(response_logits, dim=-1)
        return final_predictions.cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description="Masked Diffusion Language Model")
    parser.add_argument("--mode", type=str, choices=["train", "sample"], default="train", help="Mode: train or sample")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs (train mode)")
    parser.add_argument("--num_steps", type=int, default=10, help="Number of sampling steps (sample mode)")
    args = parser.parse_args()
    
    vocab_size = 10000
    emb_dim = 256
    nhead = 8
    num_layers = 4
    mask_token_id = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = MaskedDiffusionModel(vocab_size, emb_dim, nhead, num_layers, mask_token_id=mask_token_id).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    if args.mode == "train":
        batch_size = 32
        seq_len = 50
        num_batches = 100
        for epoch in range(args.epochs):
            epoch_loss = 0
            for batch in range(num_batches):
                x_dummy = torch.randint(1, vocab_size, (batch_size, seq_len))
                loss_value = train_step(model, optimizer, x_dummy, mask_token_id, device)
                epoch_loss += loss_value
            print(f"Epoch {epoch+1}/{args.epochs}, Average Loss: {epoch_loss/num_batches:.4f}")
        torch.save(model.state_dict(), "masked_diffusion_model.pt")
        
    elif args.mode == "sample":
        model.load_state_dict(torch.load("masked_diffusion_model.pt", map_location=device))
        prompt_len = 10
        response_len = 20
        prompt = torch.randint(1, vocab_size, (1, prompt_len))
        sampled_response = sample(model, prompt, response_len, args.num_steps, mask_token_id, device)
        print("Prompt:", prompt.cpu().numpy())
        print("Sampled Response:", sampled_response)

if __name__ == "__main__":
    main()
