
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.internvl_reward import InternVLRewardModel


def target_original_choice(human_ranking) -> int:
    ranking = np.asarray(human_ranking)
    return int(np.where(ranking == 0)[0][0])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', default='/root/private_data/models/reward-model-exam/OpenGVLab/InternVL2_5-2B')
    parser.add_argument('--score-head', required=True)
    parser.add_argument('--lora-adapter', default='')
    parser.add_argument('--data-path', default='/root/private_data/datasets/reward-model-exam/VL-RewardBench/data/test-00000-of-00001.parquet')
    parser.add_argument('--output', default='outputs/eval/reward_head_eval.jsonl')
    parser.add_argument('--summary', default='outputs/eval/reward_head_eval_summary.json')
    parser.add_argument('--limit', type=int, default=20)
    parser.add_argument('--max-tiles', type=int, default=2)
    parser.add_argument('--score-head-type', choices=['linear', 'mlp'], default='linear')
    parser.add_argument('--pooling', choices=['final', 'mean', 'final_mean_concat'], default='final')
    parser.add_argument('--mlp-hidden-ratio', type=float, default=0.25)
    parser.add_argument('--mlp-dropout', type=float, default=0.1)
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.data_path)
    if args.limit and args.limit > 0:
        df = df.head(args.limit)
    print(f'eval_rows={len(df)} max_tiles={args.max_tiles}', flush=True)

    model = InternVLRewardModel(
        args.model_path,
        freeze_backbone=True,
        use_flash_attn=False,
        score_head_type=args.score_head_type,
        pooling=args.pooling,
        mlp_hidden_ratio=args.mlp_hidden_ratio,
        mlp_dropout=args.mlp_dropout,
    ).cuda().eval()
    if args.lora_adapter:
        model.load_lora_adapter(args.lora_adapter, trainable=False)
        model.cuda().eval()
    state = torch.load(args.score_head, map_location='cpu')
    model.score_head.load_state_dict(state)

    correct = 0
    rows = []
    start = time.time()
    with output.open('w', encoding='utf-8') as f:
        for _, row in tqdm(df.iterrows(), total=len(df)):
            item = row.to_dict()
            image_bytes = item['image']['bytes']
            query = item['query']
            responses = list(item['response'])
            with torch.inference_mode():
                score0 = model.score(image_bytes, query, responses[0], max_tiles=args.max_tiles)
                score1 = model.score(image_bytes, query, responses[1], max_tiles=args.max_tiles)
            score0_f = float(score0.detach().float().cpu())
            score1_f = float(score1.detach().float().cpu())
            pred = 0 if score0_f >= score1_f else 1
            target = target_original_choice(item['human_ranking'])
            ok = pred == target
            correct += int(ok)
            record = {
                'id': item['id'],
                'query_source': item.get('query_source', ''),
                'target': target,
                'pred': pred,
                'correct': ok,
                'score0': score0_f,
                'score1': score1_f,
                'score_gap_0_minus_1': score0_f - score1_f,
                'query': query,
                'response0': responses[0],
                'response1': responses[1],
            }
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
            rows.append(record)

    out_df = pd.DataFrame(rows)
    by_source = {}
    if 'query_source' in out_df.columns:
        for source, g in out_df.groupby('query_source'):
            by_source[str(source)] = {'n': int(len(g)), 'accuracy': float(g['correct'].mean())}
    summary = {
        'n': int(len(rows)),
        'accuracy': float(correct / max(1, len(rows))),
        'score_head': args.score_head,
        'lora_adapter': args.lora_adapter,
        'max_tiles': args.max_tiles,
        'score_head_type': args.score_head_type,
        'pooling': args.pooling,
        'mlp_hidden_ratio': args.mlp_hidden_ratio,
        'mlp_dropout': args.mlp_dropout,
        'elapsed_sec': time.time() - start,
        'cuda_max_mem_gb': torch.cuda.max_memory_allocated() / 1024**3,
        'by_query_source': by_source,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f'wrote {output}', flush=True)


if __name__ == '__main__':
    main()
