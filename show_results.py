import os
# --- FIX 1: force UTF-8 decoding on Windows ---
os.environ["PYTHONIOENCODING"] = "utf-8"

import pandas as pd
import numpy as np
import plotly.express as px
import datasets
import tiktoken
import datetime
import argparse
import math
import prettytable as pt

from glob import glob
from tqdm import tqdm

from sklearn.linear_model import LogisticRegression
from collections import defaultdict, Counter
from bench_utils import (
    load_model_answers,
    model_name_to_id,
    load_image_categoeis,
    load_question_categoeis,
    load_model_judgements,
)
from functools import partial


def compute_mle_elo(df, baseline, SCALE=400, BASE=10, INIT_RATING=1000):
    models = pd.concat([df["model_a"], df["model_b"]]).unique()
    models = pd.Series(np.arange(len(models)), index=models)

    df = pd.concat([df, df], ignore_index=True)
    p = len(models.index)
    n = df.shape[0]

    X = np.zeros([n, p])
    X[np.arange(n), models[df["model_a"]]] = +math.log(BASE)
    X[np.arange(n), models[df["model_b"]]] = -math.log(BASE)

    Y = np.zeros(n)
    Y[df["winner"] == "model_a"] = 1.0

    tie_idx = (df["winner"] == "tie") | (df["winner"] == "tie (bothbad)")
    tie_idx[len(tie_idx)//2:] = False
    Y[tie_idx] = 1.0

    lr = LogisticRegression(fit_intercept=False, penalty=None, tol=1e-8)
    lr.fit(X, Y)

    elo_scores = SCALE * lr.coef_[0] + INIT_RATING

    if baseline in models.index:
        elo_scores += 1000 - elo_scores[models[baseline]]

    return pd.Series(elo_scores, index=models.index).sort_values(ascending=False)


def get_bootstrap_result(battles, func_compute_elo, num_round):
    rows = [func_compute_elo(battles)]
    for _ in tqdm(range(num_round), desc="bootstrap"):
        try:
            rows.append(func_compute_elo(battles.sample(frac=1.0, replace=True)))
        except Exception:
            break
    df = pd.DataFrame(rows)
    return df[df.median().sort_values(ascending=False).index]


def predict_win_rate(elo_ratings, SCALE=400, BASE=10):
    names = sorted(elo_ratings.keys())
    wins = defaultdict(dict)
    for a in names:
        for b in names:
            ea = 1 / (1 + BASE ** ((elo_ratings[b] - elo_ratings[a]) / SCALE))
            wins[a][b] = ea
            wins[b][a] = 1 - ea
    return pd.DataFrame(wins).T


def get_win_rate_column(df, column, baseline):
    elo_dict = df.set_index("model")[column].to_dict()
    win_rate_table = predict_win_rate(elo_dict)
    return win_rate_table[baseline].fillna(0.5).apply(lambda x: round(x * 100, 2))


def get_battles_from_judgement(judge_name, baseline, model_judgements,
                              first_game_only=False, WEIGHT=3, bench_name="vision_bench"):
    battles = []
    for model in model_judgements:
        df = pd.DataFrame.from_dict(model_judgements[model], orient="index")
        for _, row in df.iterrows():
            for gi, game in enumerate(row["games"][:1 if first_game_only else 2]):
                winner = None
                weight = 1
                if game["score"] == "A=B":
                    winner = "tie"
                elif game["score"] == "A>B":
                    winner = baseline
                elif game["score"] == "A>>B":
                    winner = baseline
                    weight = WEIGHT
                elif game["score"] == "B>A":
                    winner = row["model"]
                elif game["score"] == "B>>A":
                    winner = row["model"]
                    weight = WEIGHT
                if winner:
                    battles += [{
                        "question_id": row["question_id"],
                        "model_a": baseline,
                        "model_b": row["model"],
                        "winner": "model_a" if winner == baseline else
                                  "model_b" if winner == row["model"] else "tie"
                    }] * weight
    df = pd.DataFrame(battles)
    df.to_json(f"data/{bench_name}_battles.jsonl", lines=True, orient="records")
    return df


def get_reward_from_judgement(model_judgements, first_game_only=False):
    rewards = {}
    for model, judgements in model_judgements.items():
        df = pd.DataFrame.from_dict(judgements, orient="index")
        total = []
        wins = []
        wins_or_tie = []
        counts = Counter()
        for _, row in df.iterrows():
            reward = 0
            for game in row["games"][:1 if first_game_only else 2]:
                if game["score"] == "B>A":
                    reward += 50
                    counts["better"] += 1
                elif game["score"] == "B>>A":
                    reward += 100
                    counts["much better"] += 1
                elif game["score"] == "A>B":
                    reward -= 50
                    counts["worse"] += 1
                elif game["score"] == "A>>B":
                    reward -= 100
                    counts["much worse"] += 1
                else:
                    counts["tie"] += 1
            total.append(reward)
            wins.append(reward > 0)
            wins_or_tie.append(reward >= 0)
        rewards[model] = {
            "reward": np.mean(total),
            "win_rate": np.mean(wins),
            "win_or_tie_rate": np.mean(wins_or_tie),
            "vote_type_counts": counts
        }
    return rewards


def run_elo_simulation(model_answers, model_judgements, args):
    battles = get_battles_from_judgement(
        args.judge_name, args.baseline, model_judgements,
        args.first_game_only, args.weight, args.bench_name
    )

    rewards = get_reward_from_judgement(model_judgements, args.first_game_only)
    elo = compute_mle_elo(battles, args.baseline)
    elo_func = partial(compute_mle_elo, baseline=args.baseline)

    boot = get_bootstrap_result(battles, elo_func, args.num_rounds)

    stats = []
    for model in elo.index:
        mid = model_name_to_id(model)
        avg_tokens = np.mean([v["token_len"] for v in model_answers.get(mid, {}).values()])
        stats.append({
            "model": model,
            "score": elo[model],
            "lower": np.percentile(boot[model], 2.5),
            "upper": np.percentile(boot[model], 97.5),
            "reward": rewards[mid]["reward"],
            "win_rate": rewards[mid]["win_rate"],
            "avg_tokens": int(avg_tokens),
        })

    df = pd.DataFrame(stats).sort_values("score", ascending=False)

    table = pt.PrettyTable()
    table.field_names = ["Model", "Score", "95% CI", "Win Rate", "Reward", "Avg Tokens"]
    for _, r in df.iterrows():
        table.add_row([
            r["model"],
            round(r["score"], 1),
            f"({round(r['lower']-r['score'],1)}, {round(r['upper']-r['score'],1)})",
            f"{round(r['win_rate']*100,2)}%",
            round(r["reward"], 2),
            r["avg_tokens"],
        ])
    print(table)

    # --- FIX 2: UTF-8 safe write ---
    with open("elo_leaderboard.md", "w", encoding="utf-8") as f:
        f.write(table.get_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench-name", default="vision_bench_0617")
    parser.add_argument("--judge-name", default="gpt-4o")
    parser.add_argument("--baseline", default="claude-3-sonnet-20240229")
    parser.add_argument("--num-rounds", type=int, default=100)
    parser.add_argument("--weight", type=int, default=3)
    parser.add_argument("--first-game-only", action="store_true")
    args = parser.parse_args()

    answers = load_model_answers(os.path.join("data", args.bench_name, "model_answers"))
    judgements = load_model_judgements(
        os.path.join("data", args.bench_name, "model_judgements",
                     f"judge_{args.judge_name}_reference_{args.baseline}"),
        SAMPLE_START=0,
        MAX_SAMPLE_BENCH_SIZE=500,
    )

    run_elo_simulation(answers, judgements, args)
