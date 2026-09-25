"""Общие средства CLI, воспроизводимости и сохранения экспериментов."""
import argparse
import csv
import json
import platform
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent

def setup():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    parser.add_argument("--runs", type=int, help="Переопределить число запусков")
    parser.add_argument("--seed", type=int, help="Первый seed серии")
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    if args.runs is not None: cfg["runs"] = args.runs
    if args.seed is not None: cfg["seed"] = args.seed
    if cfg["runs"] < 1 or cfg["population"] < 30 or cfg["generations"] < 1 or cfg["seed"] < 0:
        parser.error("runs >= 1, population >= 30, generations >= 1, seed >= 0")
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "used_config.json", cfg)
    write_json(args.output / "environment.json", {"python": platform.python_version(), "numpy": np.__version__, "matplotlib": matplotlib.__version__, "platform": platform.platform()})
    return cfg, args.output

def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")

def write_csv(path, rows):
    if not rows: return
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)

def summarize(rows, metrics):
    result=[]
    for method in dict.fromkeys(r["method"] for r in rows):
        subset=[r for r in rows if r["method"] == method]
        for metric in metrics:
            a=np.array([r[metric] for r in subset], dtype=float)
            result.append(dict(method=method, metric=metric, n=len(a), minimum=float(a.min()), mean=float(a.mean()), median=float(np.median(a)), std=float(a.std(ddof=1)) if len(a)>1 else 0.0, maximum=float(a.max())))
    return result

def style():
    plt.rcParams.update({"font.family":"DejaVu Sans", "axes.spines.top":False, "axes.spines.right":False, "figure.facecolor":"#f7f9fc", "axes.facecolor":"#f7f9fc", "axes.grid":True, "grid.alpha":0.18, "font.size":11, "savefig.dpi":160})

def convergence(path, histories, ylabel="Лучшее значение", title="Сходимость: среднее и диапазон min–max"):
    style(); fig,ax=plt.subplots(figsize=(10,5))
    for name, values in histories.items():
        a=np.asarray(values); x=np.arange(a.shape[1])
        line=ax.plot(x,a.mean(axis=0),label=name)[0]
        ax.fill_between(x,a.min(axis=0),a.max(axis=0),color=line.get_color(),alpha=.12)
    ax.set(xlabel="Поколение (0 — начальная популяция)",ylabel=ylabel,title=title)
    ax.legend(); fig.tight_layout(); fig.savefig(path); plt.close(fig)

def save_histories(path, histories):
    write_csv(path,[dict(method=m,run=i,generation=g,best=float(v)) for m,hs in histories.items() for i,h in enumerate(hs) for g,v in enumerate(h)])

def table(summary):
    lines=["| Метод | Метрика | n | min | Среднее | Медиана | std (ddof=1) | max |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in summary:
        lines.append("| {method} | {metric} | {n} | {minimum:.6g} | {mean:.6g} | {median:.6g} | {std:.6g} | {maximum:.6g} |".format(**r))
    return "\n".join(lines)
