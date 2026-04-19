# -*- coding: utf-8 -*-
"""mvp_backtest.py を .ipynb に変換する。
# %% コメントブロックごとにセル化。
# %% [markdown] はマークダウンセル、# %% はコードセル。
"""
import json
import re
from pathlib import Path

SRC = Path(__file__).parent / "mvp_backtest.py"
DST = Path(__file__).parent / "mvp_backtest.ipynb"

text = SRC.read_text(encoding="utf-8")
lines = text.splitlines()

cells = []
cur_type = None
cur_src = []
for ln in lines:
    if ln.startswith("# %% [markdown]"):
        if cur_src:
            cells.append((cur_type, cur_src))
        cur_type = "markdown"
        cur_src = []
        continue
    if ln.startswith("# %%"):
        if cur_src:
            cells.append((cur_type, cur_src))
        cur_type = "code"
        cur_src = []
        continue
    if cur_type == "markdown":
        # "# " を取り除いて markdown として扱う
        cur_src.append(re.sub(r"^# ?", "", ln))
    else:
        cur_src.append(ln)

if cur_src:
    cells.append((cur_type, cur_src))

nb = {
    "cells": [],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.x"}
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
for t, src in cells:
    # 末尾空行を削る
    while src and src[-1].strip() == "":
        src.pop()
    if not src:
        continue
    cell = {
        "cell_type": t,
        "metadata": {},
        "source": [l + "\n" for l in src[:-1]] + [src[-1]],
    }
    if t == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    nb["cells"].append(cell)

DST.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"生成: {DST} ({len(nb['cells'])}セル)")
