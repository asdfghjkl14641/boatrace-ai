# 2連複 最終戦略レポート

## 核心メッセージ
- **パターン α**: 運用基準到達 (train選定→test評価で CI下≥0.95, 差<0.05)
- train 選定 → test 評価の分離徹底
- 戦略空間: 400 組み合わせ (EV × 型 × top-N × odds cap)

## Top 5 train 戦略 → test 評価

| EV | 型 | top | odds | train ROI | test ROI | CI下 | CI上 | 差 | train n | test n |
|---|---|---|---|---|---|---|---|---|---|---|
| 1.50 | t1+t2 | 2 | 10.0 | 1.1970 | **1.2094** | 1.1249 | 1.3012 | +0.0124 | 25,360 | 3,419 |
| 1.50 | no_t3 | 2 | 10.0 | 1.1970 | **1.2094** | 1.1249 | 1.3012 | +0.0124 | 25,360 | 3,419 |
| 1.50 | no_t3 | all | 10.0 | 1.1968 | **1.2106** | 1.1254 | 1.3023 | +0.0138 | 25,360 | 3,419 |
| 1.50 | no_t3 | 5 | 10.0 | 1.1968 | **1.2106** | 1.1254 | 1.3023 | +0.0138 | 25,360 | 3,419 |
| 1.50 | no_t3 | 3 | 10.0 | 1.1968 | **1.2106** | 1.1254 | 1.3023 | +0.0138 | 25,360 | 3,419 |


## 最良戦略

```
券種: 2連複 (quinella)
モデル: v4_ext_fixed
キャリブレーション: Isotonic (quinella train fit)
判定: EV ≥ 1.50 AND Edge ≥ 0.02
型フィルタ: no_t3
買い目絞り: top-all
オッズ上限: 10.0
予算: 3000円/レース, EV 比例配分
```

### 期待性能
- **train ROI**: 1.1968 (CI [1.1634, 1.2283])
- **test ROI**: 1.2106 (CI [1.1254, 1.3023])
- 差: +0.0138
- train 月別 std: 0.1459
- test 月別 std: 0.1159
- ベット率 (test): 16.53%

## 会場 sanity check
- train top5: [3, 8, 9, 19, 21]
- test top5:  [3, 5, 9, 14, 15]
- overlap: [3, 9] (2/5)
- → 会場フィルタは 過学習的

## 次アクション
1. 運用候補戦略確定, 仕様書更新
2. ペーパートレード開始


## 出力ファイル
- quinella_train_strategy_search.csv
- quinella_train_test_evaluation.csv
- quinella_stadium_train_test_check.csv
- quinella_candidates_train.pkl / quinella_candidates_test.pkl
