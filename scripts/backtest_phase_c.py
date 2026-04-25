# -*- coding: utf-8 -*-
"""
Phase C バックテスト (Stage 1/2)
- v4 スコアから PL 3連単 120通り確率
- 市場暗黙確率 q (案 β: レースごと Z、<100 組合せなら α=1.333)
- EV, Edge で買い目選定
- 3000 円均等配分 (100 円単位)
"""
from __future__ import annotations
import io, sys, json, runpy, time
from pathlib import Path
from itertools import permutations
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
OUT = BASE / "notebooks" / "output"
sys.path.insert(0, str(BASE))
sys.stdout.reconfigure(encoding="utf-8")
import warnings; warnings.filterwarnings("ignore")

from scripts.db import get_connection

# 設定 (確定済み)
TAU = 0.8
EV_THRESHOLD = 1.10
EDGE_THRESHOLD = 0.01
BUDGET = 3000
MIN_UNIT = 100
ALPHA_FALLBACK = 1.333
MIN_COMBOS_FOR_Z = 100   # β案: 100未満なら α=1.333 に fallback

TYPE_RELIABILITY = {
    "型1_逃げ本命": 1.00,
    "型2_イン残りヒモ荒れ": 0.90,
    "型3_頭荒れ": 0.80,
    "型4_ノイズ": 0.50,    # 実質フィルタ
}

# 120 通り (a,b,c) 順列、a!=b!=c, 1..6
PERMS = [(a, b, c) for a, b, c in permutations([1,2,3,4,5,6], 3)]
assert len(PERMS) == 120
PERM_STR = {p: f"{p[0]}-{p[1]}-{p[2]}" for p in PERMS}


def pl_trifecta_probs(p_lane):
    """PL 3 連単 120 通り確率。p_lane は長さ 6 の 1 着確率配列 (合計 1)。"""
    p = np.asarray(p_lane, dtype=float)
    probs = {}
    for (a, b, c) in PERMS:
        pa = p[a-1]
        pb = p[b-1] / max(1 - pa, 1e-9)
        pc = p[c-1] / max(1 - pa - p[b-1], 1e-9)
        probs[(a, b, c)] = pa * pb * pc
    return probs


def softmax_with_tau(s, tau):
    s_t = s / tau
    s_t = s_t - s_t.max()
    e = np.exp(s_t)
    return e / e.sum()


def classify_v4(top1_lane, G_S, O_S):
    if top1_lane == 0:
        if G_S > 1.0 and O_S < 0.3: return "型1_逃げ本命"
        elif G_S > 0.6 and O_S > 0.2: return "型2_イン残りヒモ荒れ"
        else: return "型4_ノイズ"
    else:
        if O_S > 0.3 and G_S > 0.4: return "型3_頭荒れ"
        else: return "型4_ノイズ"


def compute_indices(S):
    mean = S.mean()
    sigma_raw = S.std(ddof=0)
    sigma = max(sigma_raw, 0.3)
    top1_lane = int(S.argmax())
    S_sorted = -np.sort(-S)
    top1, top2, top3 = S_sorted[0], S_sorted[1], S_sorted[2]
    outer_max = float(S[3:6].max())
    return {
        "top1_lane": top1_lane,
        "F_S": (S[0] - mean) / sigma,
        "G_S": (top1 - top2) / sigma,
        "O_S": (outer_max - mean) / sigma,
        "N_S": 1.0 - (top1 - top3) / (2.0 * sigma),
        "sigma_S_raw": sigma_raw,
    }


def select_bets(candidates, budget=BUDGET, min_unit=MIN_UNIT):
    """候補リスト [(combo, p_adj, q, ev_adj, edge_adj, odds)] を受け取り、
    全候補均等配分 (min_unit 単位)。予算不足なら EV 上位から min_unit ずつ。"""
    if not candidates:
        return []
    n = len(candidates)
    per = (budget // n // min_unit) * min_unit
    if per >= min_unit:
        # 全候補に per 円
        return [(c, per) for c in candidates]
    # fallback: EV 上位 budget/min_unit 件まで min_unit 円ずつ
    cands_sorted = sorted(candidates, key=lambda x: -x[3])  # ev_adj 降順
    k = budget // min_unit
    return [(c, min_unit) for c in cands_sorted[:k]]


def run_backtest(mode: str, beta_v4, X_test, pi_test, keys_test,
                 odds_df, races_meta):
    """
    mode: "stage1" (r=1.0 全部) or "stage2" (型別 r)
    """
    results = []
    total_races = len(keys_test)
    t0 = time.time()

    # keys_test index → race key dict
    for i in range(total_races):
        key = keys_test.iloc[i]
        d = pd.Timestamp(key["date"]).date()
        stad = int(key["stadium"])
        rno = int(key["race_number"])
        # v4 スコア
        S = X_test[i] @ beta_v4  # (6,)
        idx = compute_indices(S)
        type_d = classify_v4(idx["top1_lane"], idx["G_S"], idx["O_S"])
        r = TYPE_RELIABILITY[type_d] if mode == "stage2" else 1.0

        # 1 着確率 (softmax, τ=0.8)
        p_lane = softmax_with_tau(S, TAU)
        # 120 通り 3 連単確率
        probs = pl_trifecta_probs(p_lane)

        race_odds = odds_df.get((d, stad, rno))
        if race_odds is None or len(race_odds) < MIN_COMBOS_FOR_Z:
            Z_use = ALPHA_FALLBACK
            race_odds = race_odds or {}
            reason = "data_insufficient"
        else:
            # 0 オッズは除外して Z 計算
            valid_odds = [o for o in race_odds.values() if o and o > 0]
            if len(valid_odds) < MIN_COMBOS_FOR_Z:
                Z_use = ALPHA_FALLBACK
                reason = "data_insufficient"
            else:
                Z_use = sum(1.0 / o for o in valid_odds)
                reason = None

        # 実績
        actual = races_meta.get((d, stad, rno))
        if actual is None:
            results.append({"date": d, "stadium": stad, "race_number": rno,
                            "type_d": type_d, "verdict": "skip",
                            "skip_reason": "no_result",
                            "n_candidates": 0, "total_stake": 0,
                            "total_payout": 0, "profit": 0,
                            "max_ev": 0.0, "max_edge": 0.0, "bets": "[]"})
            continue

        actual_combo = (actual[0], actual[1], actual[2])

        # 型4 自動見送り
        if type_d == "型4_ノイズ":
            results.append({"date": d, "stadium": stad, "race_number": rno,
                            "type_d": type_d, "verdict": "skip",
                            "skip_reason": "type4",
                            "n_candidates": 0, "total_stake": 0,
                            "total_payout": 0, "profit": 0,
                            "max_ev": 0.0, "max_edge": 0.0, "bets": "[]"})
            continue

        # 候補計算
        candidates = []
        max_ev, max_edge = 0.0, -1.0
        for combo in PERMS:
            odds = race_odds.get(combo)
            if odds is None or odds <= 0:
                continue
            p = probs[combo]
            p_adj = p * r
            q = 1.0 / (odds * Z_use)
            ev_adj = p_adj * odds
            edge_adj = p_adj - q
            if ev_adj > max_ev: max_ev = ev_adj
            if edge_adj > max_edge: max_edge = edge_adj
            if ev_adj >= EV_THRESHOLD and edge_adj >= EDGE_THRESHOLD:
                candidates.append((combo, p_adj, q, ev_adj, edge_adj, odds))

        if not candidates:
            results.append({"date": d, "stadium": stad, "race_number": rno,
                            "type_d": type_d, "verdict": "skip",
                            "skip_reason": "no_candidate",
                            "n_candidates": 0, "total_stake": 0,
                            "total_payout": 0, "profit": 0,
                            "max_ev": float(max_ev), "max_edge": float(max_edge), "bets": "[]"})
            continue

        # 買い目選定
        bets = select_bets(candidates)
        total_stake = sum(stake for _, stake in bets)
        # 払戻 (的中時 = odds × stake)
        total_payout = 0
        for (combo, p_adj, q, ev_adj, edge_adj, odds), stake in bets:
            if combo == actual_combo:
                total_payout += int(stake * odds)
        profit = total_payout - total_stake
        bets_detail = [{"combo": PERM_STR[c], "stake": stake,
                        "p_adj": round(p_adj, 4), "ev": round(ev_adj, 3),
                        "edge": round(edge_adj, 4), "odds": odds}
                       for (c, p_adj, q, ev_adj, edge_adj, odds), stake in bets]
        results.append({"date": d, "stadium": stad, "race_number": rno,
                        "type_d": type_d, "verdict": "bet",
                        "skip_reason": "",
                        "n_candidates": len(candidates),
                        "total_stake": int(total_stake),
                        "total_payout": int(total_payout),
                        "profit": int(profit),
                        "max_ev": float(max_ev),
                        "max_edge": float(max_edge),
                        "bets": json.dumps(bets_detail, ensure_ascii=False)})
        if (i+1) % 2000 == 0:
            print(f"   {i+1:,}/{total_races:,}  elapsed {time.time()-t0:.1f}s")
    return pd.DataFrame(results)


def summarize(df, label):
    n_total = len(df)
    n_bet = (df["verdict"] == "bet").sum()
    n_skip = (df["verdict"] == "skip").sum()
    n_skip_type4 = (df["skip_reason"] == "type4").sum()
    n_skip_nocand = (df["skip_reason"] == "no_candidate").sum()
    n_skip_nores = (df["skip_reason"] == "no_result").sum()
    n_skip_data = (df["skip_reason"] == "data_insufficient").sum()
    bet_df = df[df["verdict"] == "bet"]
    total_stake = int(bet_df["total_stake"].sum())
    total_payout = int(bet_df["total_payout"].sum())
    profit = total_payout - total_stake
    roi = (total_payout / total_stake) if total_stake else 0
    n_hit = (bet_df["total_payout"] > 0).sum()

    print(f"\n{'=' * 80}")
    print(f"{label} サマリ")
    print(f"{'=' * 80}")
    print(f"  全レース: {n_total:,}")
    print(f"  ベット: {n_bet:,} ({n_bet/n_total*100:.2f}%)")
    print(f"  見送り: {n_skip:,}")
    print(f"    型4:             {n_skip_type4:,}")
    print(f"    候補ゼロ:        {n_skip_nocand:,}")
    print(f"    結果なし:        {n_skip_nores:,}")
    print(f"    データ不備:      {n_skip_data:,}")
    print(f"  総賭金: ¥{total_stake:,}")
    print(f"  総払戻: ¥{total_payout:,}")
    print(f"  純損益: ¥{profit:+,}")
    print(f"  ROI:    {roi:.4f} ({(roi-1)*100:+.2f}%)")
    print(f"  的中数 / ベット: {n_hit:,} / {n_bet:,} ({n_hit/n_bet*100 if n_bet else 0:.2f}%)")

    # 型別
    print("\n  型別 サマリ:")
    print(f"    {'型':>22s} {'N':>6s} {'ベット率':>9s} {'的中率':>7s} {'ROI':>7s} {'profit':>10s}")
    type_rows = []
    for t in ["型1_逃げ本命","型2_イン残りヒモ荒れ","型3_頭荒れ","型4_ノイズ"]:
        sub = df[df["type_d"] == t]
        n = len(sub)
        bsub = sub[sub["verdict"] == "bet"]
        nb = len(bsub)
        stakes = int(bsub["total_stake"].sum())
        payouts = int(bsub["total_payout"].sum())
        roi_t = payouts/stakes if stakes else 0
        hit_t = (bsub["total_payout"] > 0).sum() / nb * 100 if nb else 0
        pr = payouts - stakes
        print(f"    {t:>22s} {n:>6,} {nb/n*100 if n else 0:>8.2f}% {hit_t:>6.2f}% {roi_t:>7.4f} ¥{pr:>+10,}")
        type_rows.append({"type":t, "n_total":n, "n_bet":nb,
                          "bet_rate_%": nb/n*100 if n else 0,
                          "hit_rate_%": hit_t, "roi": roi_t, "profit": pr})

    # 月別
    bet_df_ = bet_df.copy()
    bet_df_["month"] = pd.to_datetime(bet_df_["date"]).dt.to_period("M").astype(str)
    month_rows = []
    print("\n  月別 サマリ:")
    print(f"    {'月':>8s} {'N_bet':>7s} {'stake':>10s} {'payout':>10s} {'ROI':>7s}")
    for m in sorted(bet_df_["month"].unique()):
        msub = bet_df_[bet_df_["month"] == m]
        st = int(msub["total_stake"].sum()); pa = int(msub["total_payout"].sum())
        ri = pa/st if st else 0
        print(f"    {m:>8s} {len(msub):>7,} ¥{st:>9,} ¥{pa:>9,} {ri:>7.4f}")
        month_rows.append({"month":m, "n_bet":len(msub),
                           "total_stake":st, "total_payout":pa, "roi":ri})

    return {
        "n_total": n_total, "n_bet": int(n_bet),
        "bet_rate_%": n_bet/n_total*100, "total_stake": total_stake,
        "total_payout": total_payout, "profit": profit, "roi": roi,
        "hit_rate_%": n_hit/n_bet*100 if n_bet else 0,
        "n_skip_type4": int(n_skip_type4), "n_skip_nocand": int(n_skip_nocand),
        "n_skip_nores": int(n_skip_nores), "n_skip_data": int(n_skip_data),
        "type_rows": type_rows, "month_rows": month_rows,
    }


# ==================== メイン ====================

# v4 学習済 β + X_test, pi_test, keys_test を runpy で取得
print("=" * 80)
print("Phase C-1 / C-2 Backtest")
print("=" * 80)
print("\n[0] v4 training 読み込み (runpy)...")
_buf = io.StringIO(); _o = sys.stdout; sys.stdout = _buf
try:
    ns = runpy.run_path(str(BASE / "notebooks" / "pl_v4_training.py"))
finally:
    sys.stdout = _o
beta_v4 = ns["beta_v4"]
X_test_v4 = ns["X_test_v4"]
pi_test_v4 = ns["pi_test_v4"]
keys_test_v4 = ns["keys_test_v4"].reset_index(drop=True)
print(f"  β_v4 shape={beta_v4.shape}, X_test {X_test_v4.shape}")

# test 期間のオッズを一括ロード
print("\n[1] test 期間 trifecta_odds ロード ...")
conn = get_connection()
t0 = time.time()
odds = pd.read_sql_query("""
    SELECT date, stadium, race_number, combination, odds_1min
    FROM trifecta_odds
    WHERE date BETWEEN '2026-01-01' AND '2026-04-18'
      AND odds_1min IS NOT NULL
""", conn.native)
odds["date"] = pd.to_datetime(odds["date"]).dt.date
print(f"   odds rows {len(odds):,} ({time.time()-t0:.1f}s)")

# レースごとに (a,b,c) -> odds の dict 化
odds_by_race = {}
for (d, s, r), g in odds.groupby(["date","stadium","race_number"]):
    book = {}
    for _, row in g.iterrows():
        try:
            a, b, c = map(int, row["combination"].split("-"))
            book[(a, b, c)] = float(row["odds_1min"])
        except Exception:
            continue
    odds_by_race[(d, s, r)] = book
print(f"   race keys: {len(odds_by_race):,}")

# race_results をロード (1, 2, 3 着枠)
print("\n[2] race_results ロード ...")
res = pd.read_sql_query("""
    SELECT date, stadium, race_number, rank, boat
    FROM race_results
    WHERE date BETWEEN '2026-01-01' AND '2026-04-18'
      AND rank BETWEEN 1 AND 3
""", conn.native)
res["date"] = pd.to_datetime(res["date"]).dt.date
res_meta = {}
for (d, s, r), g in res.groupby(["date","stadium","race_number"]):
    g_sorted = g.sort_values("rank")
    if len(g_sorted) < 3: continue
    res_meta[(d, s, r)] = (int(g_sorted.iloc[0]["boat"]),
                            int(g_sorted.iloc[1]["boat"]),
                            int(g_sorted.iloc[2]["boat"]))
conn.close()
print(f"   races with 1-3 着: {len(res_meta):,}")

# --- Stage 1 ---
print("\n[3] Stage 1 (r=1.0) 実行 ...")
df1 = run_backtest("stage1", beta_v4, X_test_v4, pi_test_v4,
                   keys_test_v4, odds_by_race, res_meta)
df1.to_csv(OUT/"backtest_stage1_per_race.csv", index=False, encoding="utf-8-sig")
s1 = summarize(df1, "Stage 1 (基本 EV のみ, r=1.0)")

# --- Stage 2 ---
print("\n[4] Stage 2 (型別 r) 実行 ...")
df2 = run_backtest("stage2", beta_v4, X_test_v4, pi_test_v4,
                   keys_test_v4, odds_by_race, res_meta)
df2.to_csv(OUT/"backtest_stage2_per_race.csv", index=False, encoding="utf-8-sig")
s2 = summarize(df2, "Stage 2 (型信頼度 r 導入)")

# --- 集計 CSV ---
pd.DataFrame([{"stage":"stage1", **{k:v for k,v in s1.items() if not isinstance(v, list)}},
              {"stage":"stage2", **{k:v for k,v in s2.items() if not isinstance(v, list)}}]).to_csv(
    OUT/"backtest_summary.csv", index=False, encoding="utf-8-sig")
pd.DataFrame([{"stage":"stage1", **r} for r in s1["type_rows"]] +
              [{"stage":"stage2", **r} for r in s2["type_rows"]]).to_csv(
    OUT/"backtest_by_type.csv", index=False, encoding="utf-8-sig")
pd.DataFrame([{"stage":"stage1", **r} for r in s1["month_rows"]] +
              [{"stage":"stage2", **r} for r in s2["month_rows"]]).to_csv(
    OUT/"backtest_by_month.csv", index=False, encoding="utf-8-sig")

# --- Stage 1 vs Stage 2 比較 ---
print("\n" + "=" * 80)
print("Stage 1 vs Stage 2 比較")
print("=" * 80)
print(f"  {'指標':>18} {'Stage1':>10} {'Stage2':>10} {'差':>8}")
for k in ["bet_rate_%","roi","hit_rate_%","total_stake","total_payout","profit"]:
    v1 = s1[k]; v2 = s2[k]
    diff = v2 - v1 if isinstance(v1, (int, float)) else ""
    if isinstance(v1, float):
        print(f"  {k:>18} {v1:>10.4f} {v2:>10.4f} {diff:>+8.4f}")
    else:
        print(f"  {k:>18} {v1:>10,} {v2:>10,} {diff:>+8,}")

# 最終レポート
rep = f"""# Phase C 最終レポート

## 実行日
{pd.Timestamp.now():%Y-%m-%d %H:%M}

## バックテスト設定
- test 期間: 2026-01-01 〜 2026-04-18
- モデル: v4_newH (12 特徴量)
- τ = {TAU}
- EV 閾値: {EV_THRESHOLD}, Edge 閾値: {EDGE_THRESHOLD:+.2f}
- 予算: {BUDGET} 円/レース, 最低単位: {MIN_UNIT} 円
- Z (案 β): 120 通り揃いでレースごと実測、<100 組合せなら α={ALPHA_FALLBACK} fallback

## 全体比較

| 指標 | Stage 1 (r=1.0) | Stage 2 (型別 r) |
|---|---:|---:|
| ベット率 | {s1['bet_rate_%']:.2f}% | {s2['bet_rate_%']:.2f}% |
| 的中率 (ベット中) | {s1['hit_rate_%']:.2f}% | {s2['hit_rate_%']:.2f}% |
| ROI | **{s1['roi']:.4f}** | **{s2['roi']:.4f}** |
| 総賭金 | ¥{s1['total_stake']:,} | ¥{s2['total_stake']:,} |
| 総払戻 | ¥{s1['total_payout']:,} | ¥{s2['total_payout']:,} |
| 純損益 | ¥{s1['profit']:+,} | ¥{s2['profit']:+,} |

## 型別 Stage 2

| 型 | N | ベット率 | 的中率 | ROI | profit |
|---|---:|---:|---:|---:|---:|
""" + "\n".join(
    [f"| {r['type']} | {r['n_total']:,} | {r['bet_rate_%']:.2f}% | {r['hit_rate_%']:.2f}% | {r['roi']:.4f} | ¥{r['profit']:+,} |"
     for r in s2["type_rows"]]
) + f"""

## MVP 目標 (仕様書 §13.4.3)
- ROI > 1.0: Stage1 = {'✅' if s1['roi']>1.0 else '❌'}  Stage2 = {'✅' if s2['roi']>1.0 else '❌'}
- ベット率 10-30%: Stage1 = {'✅' if 10 <= s1['bet_rate_%'] <= 30 else '❌'}  Stage2 = {'✅' if 10 <= s2['bet_rate_%'] <= 30 else '❌'}
- 見送り率 20-40%: Stage1 見送り = {(s1['n_skip_type4']+s1['n_skip_nocand'])/s1['n_total']*100:.1f}%, Stage2 = {(s2['n_skip_type4']+s2['n_skip_nocand'])/s2['n_total']*100:.1f}%

## 月別 Stage 2
| 月 | N_bet | stake | payout | ROI |
|---|---:|---:|---:|---:|
""" + "\n".join(
    [f"| {r['month']} | {r['n_bet']:,} | ¥{r['total_stake']:,} | ¥{r['total_payout']:,} | {r['roi']:.4f} |"
     for r in s2["month_rows"]]
)
(OUT/"phase_c_final_report.md").write_text(rep, encoding="utf-8")
print(f"\nsaved: {OUT/'phase_c_final_report.md'}")
print("完了")
