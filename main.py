#!/usr/bin/env python3
import os
import math
import random
import argparse
import logging
from typing import Tuple, Callable, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler

from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def top_k_top_p_filtering(logits: torch.Tensor, top_k: int = 0, top_p: float = 0.0, filter_value: float = -float('Inf')) -> torch.Tensor:
    batch_size, vocab_size = logits.size()
    if top_k > 0:
        top_k = min(top_k, vocab_size)
        kth_values = torch.topk(logits, top_k, dim=-1)[0][:, -1].unsqueeze(-1)
        logits = torch.where(logits < kth_values, torch.full_like(logits, filter_value), logits)
    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits = logits.masked_fill(indices_to_remove, filter_value)
    return logits

def sample_logits(logits: torch.Tensor, temperature: float = 1.0, top_k: int = 0, top_p: float = 0.0) -> torch.Tensor:
    logits = logits / temperature
    logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)

class MaskedDiffusionTransformer(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, nhead: int, num_layers: int, dropout: float = 0.1, mask_token_id: int = 0):
        super().__init__()
        self.mask_token_id = mask_token_id
        self.embedding = nn.Embedding(vocab_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_emb = self.embedding(x)
        encoded = self.encoder(x_emb)
        logits = self.fc_out(encoded)
        return logits

class MaskingProcess:
    def __init__(self, mask_token_id: int, min_mask_prob: float = 0.1, max_mask_prob: float = 0.5):
        self.mask_token_id = mask_token_id
        self.min_mask_prob = min_mask_prob
        self.max_mask_prob = max_mask_prob
    def mask_tokens(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mask_prob = random.uniform(self.min_mask_prob, self.max_mask_prob)
        mask = (torch.rand_like(x, dtype=torch.float) < mask_prob).long()
        x_masked = x.masked_fill(mask.bool(), self.mask_token_id)
        return x_masked, mask

class DummyDataset(Dataset):
    def __init__(self, num_samples: int, seq_length: int, vocab_size: int):
        self.data = torch.randint(1, vocab_size - 1, (num_samples, seq_length))
    def __len__(self) -> int:
        return self.data.size(0)
    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.data[idx]

class TextDataset(Dataset):
    def __init__(self, file_path: str, tokenizer: Callable[[str], List[int]], seq_length: int):
        self.seq_length = seq_length
        with open(file_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
        self.data = []
        for line in lines:
            tokens = tokenizer(line)
            if len(tokens) < seq_length:
                tokens += [0] * (seq_length - len(tokens))
            else:
                tokens = tokens[:seq_length]
            self.data.append(torch.tensor(tokens, dtype=torch.long))
    def __len__(self) -> int:
        return len(self.data)
    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.data[idx]

class Trainer:
    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer, criterion: nn.Module, masking_process: MaskingProcess, device: torch.device, scaler: GradScaler, scheduler=None):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.masking_process = masking_process
        self.device = device
        self.scaler = scaler
        self.scheduler = scheduler
    def train_epoch(self, dataloader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        pbar = tqdm(dataloader, desc="Training", leave=False)
        for batch in pbar:
            batch = batch.to(self.device)
            masked_batch, _ = self.masking_process.mask_tokens(batch)
            self.optimizer.zero_grad()
            with autocast():
                outputs = self.model(masked_batch)
                loss = self.criterion(outputs.view(-1, outputs.size(-1)), batch.view(-1))
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            if self.scheduler is not None:
                self.scheduler.step()
            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        return total_loss / len(dataloader)

class Sampler:
    def __init__(self, model: nn.Module, masking_process: MaskingProcess, device: torch.device, temperature: float = 1.0, top_k: int = 50, top_p: float = 0.9):
        self.model = model
        self.masking_process = masking_process
        self.device = device
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
    def generate_text(self, prompt: torch.Tensor, response_len: int, num_steps: int, re_mask_prob: float = 0.2) -> torch.Tensor:
        self.model.eval()
        with torch.no_grad():
            batch_size = prompt.size(0)
            response = torch.full((batch_size, response_len), self.masking_process.mask_token_id, dtype=torch.long, device=self.device)
            for _ in range(num_steps):
                combined = torch.cat([prompt, response], dim=1)
                logits = self.model(combined)
                response_logits = logits[:, prompt.size(1):, :]
                sampled_tokens = []
                for pos in range(response_logits.size(1)):
                    pos_logits = response_logits[:, pos, :]
                    next_token = sample_logits(pos_logits, temperature=self.temperature, top_k=self.top_k, top_p=self.top_p)
                    sampled_tokens.append(next_token)
                response = torch.cat(sampled_tokens, dim=1)
                remask = (torch.rand(response.shape, device=self.device) < re_mask_prob).long()
                response = torch.where(remask.bool(), torch.full_like(response, self.masking_process.mask_token_id), response)
            combined = torch.cat([prompt, response], dim=1)
            final_logits = self.model(combined)
            final_response_logits = final_logits[:, prompt.size(1):, :]
            final_tokens = []
            for pos in range(final_response_logits.size(1)):
                pos_logits = final_response_logits[:, pos, :]
                token = sample_logits(pos_logits, temperature=self.temperature, top_k=self.top_k, top_p=self.top_p)
                final_tokens.append(token)
            return torch.cat(final_tokens, dim=1)

def simple_tokenizer(text: str) -> List[int]:
    return [int(token) for token in text.split() if token.isdigit()]

def main():
    parser = argparse.ArgumentParser(description="Production-Ready Masked Diffusion Transformer")
    parser.add_argument("--mode", type=str, choices=["train", "sample"], default="train")
    parser.add_argument("--dataset_file", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seq_length", type=int, default=50)
    parser.add_argument("--prompt_length", type=int, default=10)
    parser.add_argument("--response_length", type=int, default=20)
    parser.add_argument("--sample_steps", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--mask_token_id", type=int, default=0)
    parser.add_argument("--min_mask_prob", type=float, default=0.1)
    parser.add_argument("--max_mask_prob", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.9)
    args = parser.parse_args()

    VOCAB_SIZE = 10000
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MaskedDiffusionTransformer(vocab_size=VOCAB_SIZE, d_model=args.d_model, nhead=args.nhead, num_layers=args.num_layers, dropout=args.dropout, mask_token_id=args.mask_token_id).to(device)
    masking_process = MaskingProcess(mask_token_id=args.mask_token_id, min_mask_prob=args.min_mask_prob, max_mask_prob=args.max_mask_prob)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.95)
    scaler = GradScaler() if device.type == "cuda" else None
    criterion = nn.CrossEntropyLoss()

    if args.dataset_file and os.path.exists(args.dataset_file):
        dataset = TextDataset(args.dataset_file, tokenizer=simple_tokenizer, seq_length=args.seq_length)
        logger.info(f"Loaded dataset from {args.dataset_file} with {len(dataset)} samples.")
    else:
        logger.warning("No valid dataset_file provided. Using DummyDataset.")
        dataset = DummyDataset(num_samples=1000, seq_length=args.seq_length, vocab_size=VOCAB_SIZE)

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    if args.mode == "train":
        logger.info("Starting training...")
        trainer = Trainer(model, optimizer, criterion, masking_process, device, scaler, scheduler)
        for epoch in range(1, args.epochs + 1):
            epoch_loss = trainer.train_epoch(dataloader)
            logger.info(f"Epoch {epoch}/{args.epochs}, Loss: {epoch_loss:.4f}")
            torch.save(model.state_dict(), f"checkpoint_epoch_{epoch}.pt")
        torch.save(model.state_dict(), "masked_diffusion_transformer.pt")
        logger.info("Training complete. Model saved.")
    elif args.mode == "sample":
        checkpoint_path = "masked_diffusion_transformer.pt"
        if os.path.exists(checkpoint_path):
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
            logger.info("Loaded model checkpoint.")
        else:
            logger.error("Model checkpoint not found!")
            return
        sampler = Sampler(model, masking_process, device, temperature=args.temperature, top_k=args.top_k, top_p=args.top_p)
        prompt = torch.randint(1, VOCAB_SIZE - 1, (args.batch_size, args.prompt_length)).to(device)
        generated = sampler.generate_text(prompt, response_len=args.response_length, num_steps=args.sample_steps, re_mask_prob=0.2)
        logger.info("Sampled Prompt:\n%s", prompt.cpu().numpy())
        logger.info("Generated Response:\n%s", generated.cpu().numpy())

if __name__ == "__main__":
    main()
