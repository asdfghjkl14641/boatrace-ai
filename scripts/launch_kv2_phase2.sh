#!/usr/bin/env bash
# 2023-05 〜 2025-06 の 26 ヶ月を GH Actions へ投入 (K parser v2 効果を全 train 範囲に)
set -e
cd "$(dirname "$0")/.."

MONTHS=(
  "2023-05-01:2023-05-31" "2023-06-01:2023-06-30" "2023-07-01:2023-07-31"
  "2023-08-01:2023-08-31" "2023-09-01:2023-09-30" "2023-10-01:2023-10-31"
  "2023-11-01:2023-11-30" "2023-12-01:2023-12-31"
  "2024-01-01:2024-01-31" "2024-02-01:2024-02-29" "2024-03-01:2024-03-31"
  "2024-04-01:2024-04-30" "2024-05-01:2024-05-31" "2024-06-01:2024-06-30"
  "2024-07-01:2024-07-31" "2024-08-01:2024-08-31" "2024-09-01:2024-09-30"
  "2024-10-01:2024-10-31" "2024-11-01:2024-11-30" "2024-12-01:2024-12-31"
  "2025-01-01:2025-01-31" "2025-02-01:2025-02-28" "2025-03-01:2025-03-31"
  "2025-04-01:2025-04-30" "2025-05-01:2025-05-31" "2025-06-01:2025-06-30"
)

echo "Launching ${#MONTHS[@]} months ..."
for m in "${MONTHS[@]}"; do
  IFS=':' read -r start end <<< "$m"
  echo "  → $start : $end"
  gh workflow run "LZH to SQLite artifact" -f start="$start" -f end="$end" -f include_b=true 2>&1 | grep -oE "runs/[0-9]+" | head -1
  sleep 1.0
done

echo ""
echo "All triggered. Waiting 10s ..."
sleep 10
gh run list --workflow="LZH to SQLite artifact" --limit 30 --json databaseId,status,createdAt \
  | python -c "
import json, sys
runs = json.load(sys.stdin)
alive = [r for r in runs if r['status'] != 'completed']
print(f'alive: {len(alive)}')
for r in alive[:30]:
    print(f\"  {r['databaseId']} {r['status']}\")"
