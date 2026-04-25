#!/usr/bin/env bash
# 残り 35 ヶ月を GH Actions へ投入.
# Step 1 でサンプリング済: 2020-02, 2021-10, 2022-06, 2023-04
set -e
cd "$(dirname "$0")/.."

MONTHS=(
  "2020-03-01:2020-03-31" "2020-04-01:2020-04-30" "2020-05-01:2020-05-31"
  "2020-06-01:2020-06-30" "2020-07-01:2020-07-31" "2020-08-01:2020-08-31"
  "2020-09-01:2020-09-30" "2020-10-01:2020-10-31" "2020-11-01:2020-11-30"
  "2020-12-01:2020-12-31"
  "2021-01-01:2021-01-31" "2021-02-01:2021-02-28" "2021-03-01:2021-03-31"
  "2021-04-01:2021-04-30" "2021-05-01:2021-05-31" "2021-06-01:2021-06-30"
  "2021-07-01:2021-07-31" "2021-08-01:2021-08-31" "2021-09-01:2021-09-30"
  "2021-11-01:2021-11-30" "2021-12-01:2021-12-31"
  "2022-01-01:2022-01-31" "2022-02-01:2022-02-28" "2022-03-01:2022-03-31"
  "2022-04-01:2022-04-30" "2022-05-01:2022-05-31"
  "2022-07-01:2022-07-31" "2022-08-01:2022-08-31" "2022-09-01:2022-09-30"
  "2022-10-01:2022-10-31" "2022-11-01:2022-11-30" "2022-12-01:2022-12-31"
  "2023-01-01:2023-01-31" "2023-02-01:2023-02-28" "2023-03-01:2023-03-31"
)

echo "Launching ${#MONTHS[@]} months ..."
> /tmp/bulk_run_ids.txt
for m in "${MONTHS[@]}"; do
  IFS=':' read -r start end <<< "$m"
  echo "  → $start : $end"
  gh workflow run "LZH to SQLite artifact" -f start="$start" -f end="$end" -f include_b=true 2>&1 | grep -oE "runs/[0-9]+" | head -1 >> /tmp/bulk_run_ids.txt
  sleep 1.0
done

echo ""
echo "All triggered. Waiting 10s for GH to register ..."
sleep 10

# 最新 40 runs から workflow_dispatch の in_progress/queued を取得
gh run list --workflow="LZH to SQLite artifact" --limit 45 --json databaseId,status,conclusion,createdAt \
  | python -c "
import json, sys, re
runs = json.load(sys.stdin)
# in_progress/queued な run だけ抽出
alive = [r for r in runs if r['status'] not in ('completed',)]
print(f'alive runs: {len(alive)}')
for r in alive[:40]:
    print(f\"  {r['databaseId']} {r['status']} {r['createdAt']}\")"
