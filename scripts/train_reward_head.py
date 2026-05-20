
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.internvl_reward import InternVLRewardModel, RLAIFVPreferenceDataset, pairwise_loss


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', default='/root/private_data/models/reward-model-exam/OpenGVLab/InternVL2_5-2B')
    parser.add_argument('--data-dir', default='/root/private_data/datasets/reward-model-exam/rlaif-v')
    parser.add_argument('--output-dir', default='outputs/checkpoints/reward_head_overfit')
    parser.add_argument('--log-path', default='outputs/logs/train_reward_head_overfit.jsonl')
    parser.add_argument('--limit', type=int, default=32)
    parser.add_argument('--shard-limit', type=int, default=1)
    parser.add_argument('--include-indices', default='', help='Optional JSON/JSONL/TXT file with RLAIF-V global train indices to keep.')
    parser.add_argument('--exclude-indices', default='', help='Optional JSON/JSONL/TXT file with RLAIF-V global train indices to skip.')
    parser.add_argument(
        '--limit-after-exclude',
        action='store_true',
        help='Apply --limit after filtering excluded global indices. Useful for dedup ablations with matched sample count.',
    )
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--max-tiles', type=int, default=2)
    parser.add_argument('--margin', type=float, default=0.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save-every', type=int, default=0)
    parser.add_argument('--use-lora', action='store_true')
    parser.add_argument('--lora-r', type=int, default=8)
    parser.add_argument('--lora-alpha', type=int, default=16)
    parser.add_argument('--lora-dropout', type=float, default=0.05)
    parser.add_argument('--score-head-type', choices=['linear', 'mlp'], default='linear')
    parser.add_argument('--pooling', choices=['final', 'mean', 'final_mean_concat'], default='final')
    parser.add_argument('--mlp-hidden-ratio', type=float, default=0.25)
    parser.add_argument('--mlp-dropout', type=float, default=0.1)
    args = parser.parse_args()

    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path(args.log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    def read_index_file(path_text: str) -> set[int]:
        if not path_text:
            return set()
        path = Path(path_text)
        text = path.read_text(encoding='utf-8').strip()
        if not text:
            return set()
        if path.suffix == '.json':
            data = json.loads(text)
            if isinstance(data, dict):
                data = data.get('indices', data.get('include_indices', data.get('exclude_indices', [])))
            return {int(item) for item in data}
        return {int(line.strip()) for line in text.splitlines() if line.strip()}

    include_indices = set()
    if args.include_indices:
        include_indices = read_index_file(args.include_indices)
        print(f'include_indices={len(include_indices)} path={Path(args.include_indices)}', flush=True)

    exclude_indices = set()
    if args.exclude_indices:
        exclude_path = Path(args.exclude_indices)
        exclude_indices = read_index_file(args.exclude_indices)
        print(f'exclude_indices={len(exclude_indices)} path={exclude_path}', flush=True)

    dataset = RLAIFVPreferenceDataset(
        args.data_dir,
        limit=args.limit,
        shard_limit=args.shard_limit,
        include_indices=include_indices,
        exclude_indices=exclude_indices,
        limit_after_exclude=args.limit_after_exclude,
    )
    indices = list(range(len(dataset)))
    print(f'dataset_size={len(dataset)} epochs={args.epochs} lr={args.lr} max_tiles={args.max_tiles}', flush=True)

    model = InternVLRewardModel(
        args.model_path,
        freeze_backbone=True,
        use_flash_attn=False,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        score_head_type=args.score_head_type,
        pooling=args.pooling,
        mlp_hidden_ratio=args.mlp_hidden_ratio,
        mlp_dropout=args.mlp_dropout,
    ).cuda().train()
    print('params', model.trainable_parameters_summary(), flush=True)
    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    global_step = 0
    start = time.time()
    with log_path.open('w', encoding='utf-8') as log_f:
        for epoch in range(args.epochs):
            random.shuffle(indices)
            progress = tqdm(indices, desc=f'epoch {epoch+1}/{args.epochs}')
            for idx in progress:
                sample = dataset[idx]
                optimizer.zero_grad(set_to_none=True)
                score_chosen = model.score(sample.image_bytes, sample.question, sample.chosen, max_tiles=args.max_tiles)
                score_rejected = model.score(sample.image_bytes, sample.question, sample.rejected, max_tiles=args.max_tiles)
                loss = pairwise_loss(score_chosen, score_rejected, margin=args.margin)
                loss.backward()
                optimizer.step()

                gap = (score_chosen - score_rejected).detach().float().item()
                record = {
                    'step': global_step,
                    'epoch': epoch,
                    'idx': idx,
                    'loss': float(loss.detach().float().cpu()),
                    'score_chosen': float(score_chosen.detach().float().cpu()),
                    'score_rejected': float(score_rejected.detach().float().cpu()),
                    'score_gap': gap,
                    'question_len': len(sample.question),
                    'chosen_len': len(sample.chosen),
                    'rejected_len': len(sample.rejected),
                }
                log_f.write(json.dumps(record, ensure_ascii=False) + '\n')
                log_f.flush()
                progress.set_postfix(loss=f'{record["loss"]:.4f}', gap=f'{gap:.3f}')
                global_step += 1

                if args.save_every and global_step % args.save_every == 0:
                    torch.save(model.score_head.state_dict(), output_dir / f'score_head_step_{global_step}.pt')
                    if args.use_lora:
                        model.backbone.language_model.save_pretrained(output_dir / f'lora_step_{global_step}')

    torch.save(model.score_head.state_dict(), output_dir / 'score_head_final.pt')
    if args.use_lora:
        model.backbone.language_model.save_pretrained(output_dir / 'lora_final')
    meta = {
        'model_path': args.model_path,
        'data_dir': args.data_dir,
        'limit': args.limit,
        'epochs': args.epochs,
        'lr': args.lr,
        'max_tiles': args.max_tiles,
        'margin': args.margin,
        'seed': args.seed,
        'include_indices': args.include_indices,
        'include_indices_n': len(include_indices),
        'exclude_indices': args.exclude_indices,
        'exclude_indices_n': len(exclude_indices),
        'limit_after_exclude': args.limit_after_exclude,
        'effective_dataset_size': len(dataset),
        'use_lora': args.use_lora,
        'lora_r': args.lora_r,
        'lora_alpha': args.lora_alpha,
        'lora_dropout': args.lora_dropout,
        'score_head_type': args.score_head_type,
        'pooling': args.pooling,
        'mlp_hidden_ratio': args.mlp_hidden_ratio,
        'mlp_dropout': args.mlp_dropout,
        'elapsed_sec': time.time() - start,
        'cuda_max_mem_gb': torch.cuda.max_memory_allocated() / 1024**3,
    }
    (output_dir / 'training_meta.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(meta, ensure_ascii=False, indent=2), flush=True)
    print(f'wrote {output_dir}', flush=True)
    print(f'log {log_path}', flush=True)


if __name__ == '__main__':
    main()
