# -*- coding: utf-8 -*-
"""指定日の全レース予想 v2 — 型別最適券種を割当.

型ごとに過去検証で最も ROI が高かった券種を採用:
  型1 (本命): 3連単 (trifecta), test ROI 0.983
  型2 (穴・イン残り): 2連複 (quinella), test ROI 1.053
  型3 (穴・頭荒れ): 3連複 (trio), test ROI 1.265 (小サンプル注意)
  型4 (ノイズ): 見送り

スコア定義:
  本命度 = G_S (1着確定スコア)
  穴度 = O_S (2-6 号艇優位スコア)
  ノイズ度 = 1/(1 + max(G_S, O_S))  (低いほどノイズ)

使い方:
  python -m scripts.daily_prediction_v2 --date 2026-04-25
"""
from __future__ import annotations
import argparse, os, io, sys, runpy, pickle, sqlite3, warnings
from pathlib import Path
from itertools import permutations
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
BASE = Path(__file__).resolve().parent.parent
OUT = BASE / "notebooks" / "output"
PRED = BASE.parent / "predictions"
PRED.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(BASE))
sys.stdout.reconfigure(encoding="utf-8")

TAU = 0.8
EV_THR = 1.50; EDGE_THR = 0.02; ODDS_CAP = 10.0
TYPE_RELIABILITY = {"型1": 1.0, "型2": 0.9, "型3": 0.8}
ALPHA_FALLBACK = 1.333  # vig 25%
BUDGET = 3000

# 型分類閾値 (案 D: case δ 含む)
T1=(1.0,0.3); T2=(0.6,0.2); T3=(0.3,0.4)

# 型別最適券種 (前回検証より)
TYPE_BET_TYPE = {
    "型1": "trifecta",   # 3連単 ROI 0.983
    "型2": "quinella",   # 2連複 ROI 1.053
    "型3": "trio",       # 3連複 ROI 1.265
    "型4": None,
}

# 各券種の組み合わせ
PERMS_TRI = [(a,b,c) for a,b,c in permutations([1,2,3,4,5,6], 3)]      # 120
COMBS_QU  = sorted([tuple(sorted([a,b])) for a,b in permutations([1,2,3,4,5,6],2) if a<b])    # 15
COMBS_TR  = sorted([tuple(sorted([a,b,c])) for a,b,c in permutations([1,2,3,4,5,6],3) if a<b<c])  # 20

STADIUM_NAME = {1:"桐生",2:"戸田",3:"江戸川",4:"平和島",5:"多摩川",6:"浜名湖",
                7:"蒲郡",8:"常滑",9:"津",10:"三国",11:"びわこ",12:"住之江",
                13:"尼崎",14:"鳴門",15:"丸亀",16:"児島",17:"宮島",18:"徳山",
                19:"下関",20:"若松",21:"芦屋",22:"福岡",23:"唐津",24:"大村"}


def compute_indices(S):
    std_S = max(S.std(ddof=0), 0.3)
    s_sorted = np.sort(S)[::-1]
    return ((s_sorted[0]-s_sorted[1])/std_S,
            (S[3:6].max()-S.mean())/std_S,
            int(S.argmax()))

def classify_full(idx):
    G, O, top1 = idx
    if top1 == 0:
        if G > T1[0] and O < T1[1]: return "型1"
        if G > T2[0] and O > T2[1]: return "型2"
        return "型4"
    else:
        if O > T3[0] and G > T3[1]: return "型3"
        return "型4"

def compute_pl_probs_lane(p_lane):
    """3連単 120 通りの確率."""
    p = np.asarray(p_lane); arr = np.zeros(120)
    for i, (a,b,c) in enumerate(PERMS_TRI):
        pa = p[a-1]
        pb = p[b-1]/max(1-pa, 1e-9)
        pc = p[c-1]/max(1-pa-p[b-1], 1e-9)
        arr[i] = pa*pb*pc
    return arr

def compute_quinella_probs(p_lane):
    p = np.asarray(p_lane); arr = np.zeros(15)
    for i, (a,b) in enumerate(COMBS_QU):
        pa = p[a-1]; pb = p[b-1]
        arr[i] = pa*(p[b-1]/max(1-pa,1e-9)) + pb*(p[a-1]/max(1-pb,1e-9))
    return arr

def compute_trio_probs(p_lane):
    """3連複 20通り."""
    p = np.asarray(p_lane); arr = np.zeros(20)
    for i, (a,b,c) in enumerate(COMBS_TR):
        s = 0.0
        for (x,y,z) in permutations([a,b,c], 3):
            px = p[x-1]; py = p[y-1]/max(1-px,1e-9); pz = p[z-1]/max(1-px-p[y-1],1e-9)
            s += px*py*pz
        arr[i] = s
    return arr

def fmt_combo_tri(c): return f"{c[0]}-{c[1]}-{c[2]}"
def fmt_combo_qu(c):  return f"{c[0]}={c[1]}"
def fmt_combo_tr(c):  return f"{c[0]}={c[1]}={c[2]}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True, help="予想日 YYYY-MM-DD")
    args = ap.parse_args()
    target = args.date

    print("=" * 80)
    print(f"日次予想 v2 — {target} (全レース型分類 + 型別最適券種)")
    print("=" * 80)

    # DB 確認
    conn = sqlite3.connect(str(BASE/"boatrace.db"))
    n_rc = conn.execute(f"SELECT COUNT(DISTINCT stadium||'|'||race_number) FROM race_cards WHERE date = '{target}'").fetchone()[0]
    conn.close()
    print(f"\n[0] race_cards for {target}: {n_rc} races")
    if n_rc == 0:
        print(f"  ❌ race_cards に {target} のデータなし")
        return 1

    # v4_ext_fixed runpy
    print(f"\n[1] v4_ext_fixed runpy (test = {target}) ...")
    os.environ["V4_TRAIN_FROM"] = "2020-02-01"
    os.environ["V4_TRAIN_TO"]   = "2025-06-30"
    os.environ["V4_TEST_FROM"]  = target
    os.environ["V4_TEST_TO"]    = target
    os.environ["V4_OUT_SUFFIX"] = "_ext_fixed"

    _buf = io.StringIO(); _o = sys.stdout; sys.stdout = _buf
    try: ns = runpy.run_path(str(BASE / "notebooks" / "pl_v4_training.py"))
    finally: sys.stdout = _o

    beta = ns["beta_v4"]
    X_te = ns["X_test_v4"]; keys_te = ns["keys_test_v4"].reset_index(drop=True)
    N = len(keys_te)
    print(f"  β={beta.shape}  X_test={X_te.shape}  ({N} races for {target})")
    if N == 0:
        print(f"  ❌ 推論対象なし")
        return 1

    # Isotonic per bet type
    isotonics = {}
    for bt in ["trifecta","quinella","trio"]:
        with open(OUT/f"calibration_isotonic_{bt}.pkl", "rb") as f:
            isotonics[bt] = pickle.load(f)

    # 各レース予想
    print(f"\n[2] 各レース予想 ...")
    rows = []
    for i in range(N):
        k = keys_te.iloc[i]
        d = pd.Timestamp(k["date"]).date().isoformat()
        stadium = int(k["stadium"]); rno = int(k["race_number"])
        S = X_te[i] @ beta
        idx = compute_indices(S)
        G_S, O_S, top1 = idx
        t = classify_full(idx)
        s_t = S/TAU; s_t = s_t - s_t.max()
        p_lane = np.exp(s_t)/np.exp(s_t).sum()

        # スコア
        honmei_score = G_S         # 本命度 (1号艇優位)
        ana_score = O_S            # 穴度 (2-6 号艇優位)
        # ノイズ度: 両方とも低い (G_S<0.5 AND O_S<0.3) ほど高い
        noise_score = 1 / (1 + max(G_S, O_S * 2))

        bt = TYPE_BET_TYPE.get(t)
        bet_type_jp = {"trifecta":"3連単","quinella":"2連複","trio":"3連複",None:"見送り"}[bt]

        # 推奨買い目
        top_combo_str = "—"
        top_p = 0; pred_odds = float('inf'); req_odds = float('inf')
        if bt is not None:
            rel = TYPE_RELIABILITY.get(t, 0.5)
            if bt == "trifecta":
                probs = compute_pl_probs_lane(p_lane)
                combos = PERMS_TRI; fmt = fmt_combo_tri
            elif bt == "quinella":
                probs = compute_quinella_probs(p_lane)
                combos = COMBS_QU; fmt = fmt_combo_qu
            else:  # trio
                probs = compute_trio_probs(p_lane)
                combos = COMBS_TR; fmt = fmt_combo_tr
            probs_cal = isotonics[bt].transform(probs.astype(np.float64))
            p_adj_arr = probs_cal * rel
            j = int(np.argmax(p_adj_arr))
            top_combo_str = fmt(combos[j])
            top_p = p_adj_arr[j]
            Z = ALPHA_FALLBACK
            pred_odds = 1/(top_p * Z) if top_p > 0 else float('inf')
            # EV >= 1.50 必要オッズ
            req_odds = 1.50 / top_p if top_p > 0 else float('inf')

        rows.append({
            "date": d, "stadium": stadium,
            "stadium_name": STADIUM_NAME.get(stadium, f"場{stadium}"),
            "race": rno, "type": t,
            "honmei": honmei_score, "ana": ana_score, "noise": noise_score,
            "bet_type": bet_type_jp, "combo": top_combo_str,
            "top_p": top_p, "pred_odds": pred_odds, "req_odds": req_odds,
        })

    df = pd.DataFrame(rows).sort_values(["stadium","race"])
    df_score = df[df["type"] != "型4"].copy()
    df_skip = df[df["type"] == "型4"].copy()

    # 型別集計
    print(f"\n[3] 型別集計")
    print(f"  全レース数: {len(df)}")
    for t in ["型1", "型2", "型3", "型4"]:
        n = (df["type"]==t).sum()
        bt = TYPE_BET_TYPE.get(t)
        bt_jp = {"trifecta":"3連単","quinella":"2連複","trio":"3連複",None:"見送り"}[bt]
        print(f"  {t}: {n:>3}  推奨券種: {bt_jp}")

    # 注目レース表 (型1/2/3)
    print(f"\n[4] 注目レース ({len(df_score)} races, スコア順)")
    df_print = df_score.copy()
    # スコア優先順: 型1 = honmei, 型2 = ana, 型3 = ana
    def primary_score(rw):
        if rw["type"] == "型1": return rw["honmei"]
        else: return rw["ana"]
    df_print["primary"] = df_print.apply(primary_score, axis=1)
    df_print = df_print.sort_values("primary", ascending=False)
    print(f"  {'会場':<8} {'R':>3} {'型':<5} {'本命度':>6} {'穴度':>6} "
          f"{'ノイズ度':>7} {'券種':<6} {'買い目':<10} {'予想o':>7} {'必要o':>7}")
    for _, rw in df_print.head(40).iterrows():
        print(f"  {rw['stadium_name']:<8} {rw['race']:>3} {rw['type']:<5} "
              f"{rw['honmei']:>6.2f} {rw['ana']:>6.2f} {rw['noise']:>7.3f} "
              f"{rw['bet_type']:<6} {rw['combo']:<10} "
              f"{rw['pred_odds']:>7.2f} {rw['req_odds']:>7.2f}")

    # CSV 出力
    out_csv = PRED / f"{target}_full.csv"
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"\n  saved: {out_csv}")

    # MD 出力
    md_lines = [f"# {target} 全レース予想 (型別最適券種)\n"]
    md_lines.append(f"型別最適券種:\n")
    md_lines.append(f"- 型1 (本命) → 3連単  (test ROI 0.983)")
    md_lines.append(f"- 型2 (穴イン残) → 2連複 (test ROI 1.053)")
    md_lines.append(f"- 型3 (穴頭荒) → 3連複 (test ROI 1.265, 小サンプル注意)")
    md_lines.append(f"- 型4 (ノイズ) → 見送り\n")
    md_lines.append(f"## 注目レース ({len(df_score)} races)\n")
    md_lines.append("| 会場 | R | 型 | 本命度 | 穴度 | ノイズ度 | 券種 | 買い目 | 予想oz | 必要oz |")
    md_lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for _, rw in df_print.iterrows():
        md_lines.append(f"| {rw['stadium_name']} | {rw['race']}R | {rw['type']} | "
                         f"{rw['honmei']:.2f} | {rw['ana']:.2f} | {rw['noise']:.3f} | "
                         f"{rw['bet_type']} | {rw['combo']} | "
                         f"{rw['pred_odds']:.2f} | {rw['req_odds']:.2f} |")
    md_lines.append(f"\n## 見送りレース ({len(df_skip)} races)\n")
    md_lines.append("| 会場 | R | 型 | 本命度 | 穴度 |")
    md_lines.append("|---|---|---|---|---|")
    for _, rw in df_skip.iterrows():
        md_lines.append(f"| {rw['stadium_name']} | {rw['race']}R | 型4 | "
                         f"{rw['honmei']:.2f} | {rw['ana']:.2f} |")
    (PRED/f"{target}_full.md").write_text("\n".join(md_lines), encoding="utf-8")
    print(f"  saved: {PRED}/{target}_full.md")

    # X 投稿用 TOP10 (型1/2/3 のみ、スコア順)
    tw_lines = [f"【{target[5:]} 数理モデル予想 注目10選】"]
    for i, (_, rw) in enumerate(df_print.head(10).iterrows(), 1):
        bt_short = {"3連単":"3単","2連複":"2複","3連複":"3複"}.get(rw['bet_type'], '')
        tw_lines.append(f"{i}.{rw['stadium_name']}{rw['race']}R {rw['type']} "
                         f"{bt_short}{rw['combo']} 必要{rw['req_odds']:.1f}↑")
    tw_lines.append("※1分前oz確認")
    tw_lines.append("#ボートレース #競艇予想 #数理モデル")
    tw_text = "\n".join(tw_lines)
    (PRED/f"{target}_twitter.txt").write_text(tw_text, encoding="utf-8")
    print(f"\n--- X 投稿用 ({len(tw_text)} chars) ---")
    print(tw_text)

    return 0


if __name__ == "__main__":
    sys.exit(main())
