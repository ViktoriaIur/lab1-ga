# -*- coding: utf-8 -*-
"""Сводная визуализация эффективности ГА по 3 функциям и 2 размерностям."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

with open("/mnt/user-data/outputs/experiment_results.json", encoding="utf-8") as f:
    data = json.load(f)

rows = data["summary"]
functions = sorted(set(r["function"] for r in rows))
dims = sorted(set(r["n"] for r in rows))

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# --- Надёжность (успешность) ---
ax = axes[0]
width = 0.35
x = np.arange(len(functions))
for i, n in enumerate(dims):
    vals = [next(r["success_rate_%"] for r in rows if r["function"] == fn and r["n"] == n)
            for fn in functions]
    ax.bar(x + (i - 0.5) * width, vals, width, label=f"n={n}")
ax.set_xticks(x)
ax.set_xticklabels(functions)
ax.set_ylabel("Надёжность, %")
ax.set_title("Доля успешных запусков (5 запусков на конфигурацию)")
ax.legend()
ax.grid(axis="y", alpha=0.3)

# --- Среднее число вычислений функции ---
ax = axes[1]
for i, n in enumerate(dims):
    vals = [next(r["mean_evaluations"] for r in rows if r["function"] == fn and r["n"] == n)
            for fn in functions]
    ax.bar(x + (i - 0.5) * width, vals, width, label=f"n={n}")
ax.set_xticks(x)
ax.set_xticklabels(functions)
ax.set_ylabel("Среднее число вычислений f(x)")
ax.set_title("Вычислительная стоимость сходимости")
ax.legend()
ax.grid(axis="y", alpha=0.3)

fig.tight_layout()
fig.savefig("/mnt/user-data/outputs/experiment_summary.png", dpi=150)
print("Сохранено: experiment_summary.png")
