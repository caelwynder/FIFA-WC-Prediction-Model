# scripts/06_simulate_tournament.py
#
# Tournament odds for the knockout bracket: for every team, the probability
# of reaching R16 / QF / SF / the Final and of winning the trophy.
#
# Computed EXACTLY by dynamic programming over the fixed bracket (Brandes et
# al. 2025, "Stop Simulating! Efficient Computation of Tournament Winning
# Probabilities"): P(team survives round r) = P(team alive) x sum over
# possible opponents of P(opponent alive) x P(team beats opponent). A 10,000-
# run Monte Carlo cross-check validates the DP (max deviation should be under
# ~1.5 percentage points, within sampling noise).
#
# Per-match win probability folds the tandem models together:
#   P(A beats B) = P(A wins) + P(draw) x P(A wins the shootout)
#
# Writes the odds into website/data.json under "odds".

import json
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR.parent / "website" / "data.json"

N_SIMS = 10_000
SEED = 42


def load_data():
    return json.loads(DATA_FILE.read_text())


def match_win_prob(pred: list[float]) -> float:
    """P(home-slot side advances) from [pH, pD, pA, gH, gA, penH]."""
    p_h, p_d, _, _, _, pen_h = pred
    return p_h + p_d * pen_h


def pair_win_prob(pairs: dict, a: str, b: str) -> float:
    """P(a advances past b) on neutral ground, a in the home slot."""
    return match_win_prob(pairs[f"{a}|{b}"])


def exact_odds(data: dict) -> dict:
    """DP over the bracket. Seeds: team i's R32 match is i//2, home if i%2==0.
    reach[r][i] = P(team i is alive going INTO round r) (r=0 -> R32)."""
    teams = []
    for f in data["r32"]:
        teams += [f["home"], f["away"]]
    n = len(teams)  # 32
    pairs = data["pairs"]

    reach = np.zeros((6, n))
    reach[0] = 1.0

    for r in range(5):  # R32, R16, QF, SF, F
        block = 2 ** (r + 1)
        for i in range(n):
            if reach[r][i] == 0:
                continue
            base = (i // block) * block
            half = block // 2
            in_upper = (i - base) < half
            opp_range = range(base + half, base + block) if in_upper else range(base, base + half)

            p_survive = 0.0
            for j in opp_range:
                if reach[r][j] == 0:
                    continue
                if r == 0:
                    # real fixture prediction (true venue); i,j are in the same match
                    pred = data["r32"][i // 2]["pred"]
                    w = match_win_prob(pred) if in_upper else 1 - match_win_prob(pred)
                else:
                    # hypothetical pairing on neutral ground; upper side takes the home slot
                    w = pair_win_prob(pairs, teams[i], teams[j]) if in_upper \
                        else 1 - pair_win_prob(pairs, teams[j], teams[i])
                p_survive += reach[r][j] * w
            reach[r + 1][i] = reach[r][i] * p_survive

    return {teams[i]: [float(reach[r][i]) for r in range(1, 6)] for i in range(n)}


def monte_carlo_odds(data: dict, n_sims: int = N_SIMS, seed: int = SEED) -> dict:
    rng = np.random.default_rng(seed)
    teams = []
    for f in data["r32"]:
        teams += [f["home"], f["away"]]
    pairs = data["pairs"]
    champs = {t: 0 for t in teams}

    for _ in range(n_sims):
        alive = list(teams)
        rnd = 0
        while len(alive) > 1:
            nxt = []
            for m in range(len(alive) // 2):
                a, b = alive[2 * m], alive[2 * m + 1]
                if rnd == 0:
                    w = match_win_prob(data["r32"][m]["pred"])
                else:
                    w = pair_win_prob(pairs, a, b)
                nxt.append(a if rng.random() < w else b)
            alive = nxt
            rnd += 1
        champs[alive[0]] += 1

    return {t: c / n_sims for t, c in champs.items()}


def main():
    data = load_data()

    odds = exact_odds(data)
    mc = monte_carlo_odds(data)

    total_champ = sum(v[4] for v in odds.values())
    assert abs(total_champ - 1.0) < 1e-6, f"champion odds sum to {total_champ}"

    max_dev = max(abs(odds[t][4] - mc[t]) for t in odds)
    print(f"Exact DP vs {N_SIMS}-run Monte Carlo: max champion-odds deviation {max_dev:.4f}")
    assert max_dev < 0.015, "Monte Carlo disagrees with the exact DP beyond sampling noise"

    print("\nChampion odds (exact):")
    board = sorted(odds.items(), key=lambda kv: kv[1][4], reverse=True)
    for t, o in board[:10]:
        print(f"  {t:15s} champ {o[4]*100:5.1f}%  (final {o[3]*100:5.1f}%, SF {o[2]*100:5.1f}%)")

    data["odds"] = {t: [round(x, 4) for x in v] for t, v in odds.items()}
    DATA_FILE.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")))
    print(f"\n✅ Odds written into {DATA_FILE}")


if __name__ == "__main__":
    main()
