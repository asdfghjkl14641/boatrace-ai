# -*- coding: utf-8 -*-
"""指定日の 2連複 注目レース予想 (v4_ext_fixed 運用戦略).

戦略:
  券種: 2連複
  判定: EV ≥ 1.50 AND Edge ≥ 0.02
  型フィルタ: no_t3 (型1 + 型2)
  オッズ上限: 10.0
  予算: 3000円/レース、均等配分

使い方:
  python -m scripts.daily_prediction --date 2026-04-25
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
TYPE_FILTER = {"型1", "型2"}
TYPE_RELIABILITY = {"型1": 1.0, "型2": 0.9}
BUDGET = 3000; MIN_UNIT = 100
ALPHA_FALLBACK = 1.333  # 2連複 Z フォールバック (控除率 25%)

COMBS_QU = sorted([tuple(sorted([a,b])) for a,b in permutations([1,2,3,4,5,6],2) if a<b])
T1=(1.0,0.3); T2=(0.6,0.2)

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

def classify_12(idx):
    """型1 or 型2 のみ返す (型3/4 は None)."""
    G, O, top1 = idx
    if top1 == 0:
        if G > T1[0] and O < T1[1]: return "型1"
        if G > T2[0] and O > T2[1]: return "型2"
    return None

def compute_quinella_probs(p_lane):
    p = np.asarray(p_lane); arr = np.zeros(15)
    for i, (a,b) in enumerate(COMBS_QU):
        pa = p[a-1]; pb = p[b-1]
        arr[i] = pa*(p[b-1]/max(1-pa,1e-9)) + pb*(p[a-1]/max(1-pb,1e-9))
    return arr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True, help="予想日 YYYY-MM-DD")
    ap.add_argument("--quick", action="store_true",
                    help="既存 v4_ext_fixed_features_v2.csv を使う (高速だが対象日が無いとエラー)")
    args = ap.parse_args()
    target = args.date

    print("=" * 80)
    print(f"日次予想 — {target}")
    print("=" * 80)

    # DB から target 日の race_cards 確認
    conn = sqlite3.connect(str(BASE/"boatrace.db"))
    n_rc = conn.execute(f"SELECT COUNT(DISTINCT stadium||'|'||race_number) FROM race_cards WHERE date = '{target}'").fetchone()[0]
    conn.close()
    print(f"\n[0] race_cards for {target}: {n_rc} races")
    if n_rc == 0:
        print(f"  ❌ race_cards に {target} のデータなし")
        print(f"     GH Actions で LZH B 取得 or Windows 手動取得が必要")
        return 1

    # v4_ext_fixed runpy — target 日を test に指定
    print(f"\n[1] v4_ext_fixed runpy (test = {target}) ...")
    os.environ["V4_TRAIN_FROM"] = "2020-02-01"
    os.environ["V4_TRAIN_TO"]   = "2025-06-30"
    os.environ["V4_TEST_FROM"]  = target
    os.environ["V4_TEST_TO"]    = target
    os.environ["V4_OUT_SUFFIX"] = "_ext_fixed"

    _buf = io.StringIO(); _o = sys.stdout; sys.stdout = _buf
    try:
        ns = runpy.run_path(str(BASE / "notebooks" / "pl_v4_training.py"))
    finally:
        sys.stdout = _o

    beta = ns["beta_v4"]
    X_te = ns["X_test_v4"]; keys_te = ns["keys_test_v4"].reset_index(drop=True)
    N = len(keys_te)
    print(f"  β={beta.shape}  X_test={X_te.shape}  ({N} races for {target})")
    if N == 0:
        print(f"  ❌ 推論対象レースなし ({target} の race_cards はあるが特徴量計算が失敗した可能性)")
        return 1

    # Isotonic quinella
    with open(OUT/"calibration_isotonic_quinella.pkl", "rb") as f:
        iso = pickle.load(f)

    # 予想: 各レース
    print(f"\n[2] 各レース予想 ...")
    results = []
    for i in range(N):
        k = keys_te.iloc[i]
        d = pd.Timestamp(k["date"]).date().isoformat()
        stadium = int(k["stadium"]); rno = int(k["race_number"])
        S = X_te[i] @ beta
        idx = compute_indices(S)
        t = classify_12(idx)
        if t is None: continue  # 型3/4 除外
        rel = TYPE_RELIABILITY[t]
        s_t = S/TAU; s_t = s_t - s_t.max()
        p_lane = np.exp(s_t)/np.exp(s_t).sum()
        probs = compute_quinella_probs(p_lane)
        probs_cal = iso.transform(probs.astype(np.float64))

        # 各組み合わせの情報
        Z = ALPHA_FALLBACK  # 当日のオッズ流通前はフォールバック
        combo_rows = []
        for j, combo in enumerate(COMBS_QU):
            p_adj = probs_cal[j] * rel
            pred_odds = 1/(p_adj * Z) if p_adj > 0 else float('inf')  # モデル予想オッズ
            # EV >= 1.50 に必要な実オッズ (p_adj × O >= 1.50 → O >= 1.50 / p_adj)
            req_odds_ev = 1.50 / p_adj if p_adj > 0 else float('inf')
            # Edge >= 0.02 (p_adj - 1/(O*Z) >= 0.02 → O >= 1/(Z*(p_adj-0.02)))
            req_odds_edge = 1/(Z*(p_adj-0.02)) if p_adj > 0.02 else float('inf')
            # odds <= 10 制約
            req_odds = max(req_odds_ev, req_odds_edge)
            # 買いのレンジ: [req_odds, 10.0]
            buyable = req_odds <= ODDS_CAP
            combo_rows.append({
                "combo": f"{combo[0]}={combo[1]}", "p": probs_cal[j], "p_adj": p_adj,
                "pred_odds": pred_odds, "req_odds": req_odds, "buyable": buyable
            })
        combo_df = pd.DataFrame(combo_rows).sort_values("p_adj", ascending=False)
        buyable = combo_df[combo_df["buyable"]].head(5)

        results.append({
            "date": d, "stadium": stadium, "stadium_name": STADIUM_NAME.get(stadium, f"場{stadium}"),
            "race": rno, "type": t, "G_S": idx[0], "O_S": idx[1],
            "top_combo": combo_df.iloc[0]["combo"],
            "top_p": combo_df.iloc[0]["p_adj"],
            "top_pred_odds": combo_df.iloc[0]["pred_odds"],
            "top_req_odds": combo_df.iloc[0]["req_odds"],
            "buyable_combos": len(buyable),
            "buyable_detail": buyable.to_dict("records") if len(buyable) else [],
        })

    if not results:
        print("  該当レースなし (型1/型2 のレースが無い日)")
        return 0

    res_df = pd.DataFrame(results).sort_values(["stadium", "race"])
    print(f"  注目レース (型1/型2): {len(res_df)} races")
    print()
    print(f"  {'会場':<8} {'R':>3} {'型':<5} {'G_S':>5} {'O_S':>5} "
          f"{'top買い目':<10} {'top p':>7} {'予想o':>7} {'必要o':>7} {'買い':>4}")
    for _, rw in res_df.iterrows():
        print(f"  {rw['stadium_name']:<8} {rw['race']:>3} {rw['type']:<5} "
              f"{rw['G_S']:>5.2f} {rw['O_S']:>5.2f} {rw['top_combo']:<10} "
              f"{rw['top_p']:>7.4f} {rw['top_pred_odds']:>7.2f} {rw['top_req_odds']:>7.2f} "
              f"{'◯' if rw['buyable_combos']>0 else '×':>4}")

    # ========== 出力 ==========
    # 詳細 MD
    md_lines = [f"# {target} 2連複 注目レース 数理モデル予想\n"]
    md_lines.append(f"戦略: EV ≥ 1.50 AND Edge ≥ 0.02 AND odds ≤ 10, 型フィルタ: 型1 + 型2\n")
    md_lines.append(f"予算: 3000円/レース, 均等配分\n")
    md_lines.append(f"\n## 注目レース ({len(res_df)} races)\n")
    md_lines.append("| 会場 | R | 型 | G_S | O_S | top買い目 | モデル予想オッズ | 買いの基準オッズ | 買い可否 |")
    md_lines.append("|---|---|---|---|---|---|---|---|---|")
    for _, rw in res_df.iterrows():
        mark = "◯" if rw["buyable_combos"]>0 else "×"
        md_lines.append(f"| {rw['stadium_name']} | {rw['race']}R | {rw['type']} | "
                        f"{rw['G_S']:.2f} | {rw['O_S']:.2f} | {rw['top_combo']} | "
                        f"{rw['top_pred_odds']:.2f} | {rw['top_req_odds']:.2f} | {mark} |")
    md_lines.append("")
    md_lines.append("## 解説")
    md_lines.append("- **モデル予想オッズ**: Plackett-Luce + Isotonic キャリブレーション × 型信頼度 → 1/(p×1.333)")
    md_lines.append("- **買いの基準オッズ**: EV≥1.50 かつ Edge≥0.02 かつ odds≤10 を満たすための実オッズ")
    md_lines.append("- **買い可否**: 1分前オッズが基準を満たせば買い、満たさなければ見送り")
    md_lines.append("- **注意**: 実買いは1分前オッズで最終判断。このリストは候補選定のみ。")
    (PRED/f"{target}.md").write_text("\n".join(md_lines), encoding="utf-8")

    # Twitter スレッド用 — 自己紹介 + 5レース/ツイート
    buyable_df = res_df[res_df["buyable_combos"]>0].copy()
    buyable_df["score"] = buyable_df["top_p"]
    picks = buyable_df.sort_values("score", ascending=False)

    tweets = []
    # ツイート 1: 自己紹介
    intro = (
        f"【{target[5:]} ボートレース 2連複 数理モデル予想】\n\n"
        f"古典的統計モデルによる予想 (Plackett-Luce + Isotonic)。\n"
        f"バックテスト ROI 1.21倍 (95%CI 1.13-1.30, 検証 44,064レース)。\n\n"
        f"注目 {len(picks)} レース、EV 順に発信。\n"
        f"※実オッズで EV 確認 / 投票自己責任\n\n"
        f"#ボートレース #競艇予想 #数理モデル"
    )
    tweets.append(intro)

    # ツイート 2+: 5 レースずつ
    for batch_start in range(0, len(picks), 5):
        batch = picks.iloc[batch_start:batch_start+5]
        page = batch_start // 5 + 1
        total_pages = (len(picks) + 4) // 5
        lines = [f"【注目レース {target[5:]} ({page}/{total_pages})】"]
        for i, (_, rw) in enumerate(batch.iterrows(), batch_start+1):
            lines.append(f"{i}.{rw['stadium_name']}{rw['race']}R 型{rw['type'][1]} "
                         f"2連複{rw['top_combo']} 予{rw['top_pred_odds']:.1f}倍 "
                         f"(実{rw['top_req_odds']:.1f}倍↑で買い)")
        tweets.append("\n".join(lines))

    tw_text = "\n---\n".join(tweets)
    (PRED/f"{target}_twitter.txt").write_text(tw_text, encoding="utf-8")

    print(f"\n[3] 出力 ...")
    print(f"  詳細: predictions/{target}.md ({len(res_df)} races, buyable {len(picks)})")
    print(f"  X用: predictions/{target}_twitter.txt ({len(tweets)} tweets)")
    # 文字数チェック
    for i, t in enumerate(tweets):
        mark = " ⚠ 280超" if len(t) > 280 else ""
        print(f"    tweet {i+1}: {len(t)} chars{mark}")
    print()
    print(f"--- X 投稿用 (各ツイート --- で区切り) ---")
    print(tw_text)
    print(f"--- 文字数: {len(tw_text)} ---")
    return 0


if __name__ == "__main__":
    sys.exit(main())
