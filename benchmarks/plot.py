"""Gráficos e tabelas a partir dos resultados da suíte (benchmarks/suite.py).

Gera, na pasta da suíte (ou em --out):
    time_vs_rows_light.png / _dark.png     tempo mediano x nº de linhas, um painel por nº de colunas
    memory_vs_rows_light.png / _dark.png   pico de memória x nº de linhas, idem
    summary.md                             tabelas completas (tempo, memória, razão de tempo) + metodologia

Leitura dos gráficos: eixos log-log; linha = mediana das execuções; faixa = intervalo interquartil
(Q1-Q3); `×` no topo = execução interrompida (estouro de RAM ou timeout) naquele tamanho.
Cada engine tem cor E formato de marcador próprios, então a identidade nunca depende só da cor.

Uso:
    uv run python benchmarks/plot.py benchmarks/results/full
    uv run python benchmarks/plot.py benchmarks/results/full --theme light --mem-metric delta_mb

Requer matplotlib (grupo dev).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FuncFormatter, NullFormatter  # noqa: E402

# Paleta validada com scripts/validate_palette.js (--pairs all, modos claro e escuro): azul, amarelo,
# magenta e verde, nessa ordem. Em claro o contraste de amarelo/magenta fica < 3:1 e, em escuro, o CVD
# do par verde/amarelo fica na faixa de aviso; a compensação é o marcador próprio de cada engine, a
# legenda sempre visível e a tabela completa (summary.md).
THEMES: Dict[str, Dict[str, Any]] = {
    "light": {
        "surface": "#fcfcfb",
        "ink": "#0b0b0b",
        "ink2": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
        "series": ["#2a78d6", "#eda100", "#e87ba4", "#008300"],
        "extra": ["#eb6834", "#1baf7a", "#4a3aa7", "#e34948"],
    },
    "dark": {
        "surface": "#1a1a19",
        "ink": "#ffffff",
        "ink2": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
        "series": ["#3987e5", "#c98500", "#d55181", "#008300"],
        "extra": ["#d95926", "#199e70", "#9085e9", "#e66767"],
    },
}
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "h"]
# Ordem fixa das cores por engine (a cor segue a entidade, nunca a posição na tela)
ENGINE_ORDER = [
    "datacompolars",
    "datacompolars (janela única)",
    "datacompolars (janelas)",
    "datacompolars (janelas + cache)",
    "datacompolars (columnwise)",
    "datacompolars (columnwise + janelas)",
    "datacompolars (columnwise + janelas + cache)",
    "datacompy (Polars)",
    "datacompy (pandas)",
    "diffly",
    "diffly (eager)",
]
FONT_STACK = ["Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"]
MEM_LABELS = {
    "peak_median_mb": "Pico de memória (MB, mediana)",
    "peak_mb": "Pico de memória (MB, máximo das execuções)",
    "delta_mb": "Memória acima da base (MB, mediana)",
}


# ----------------------------------------------------------------------------- dados
def load(path: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Path]:
    """Lê results.jsonl (última linha de cada caso vale) e meta.json da pasta da suíte."""
    results = path / "results.jsonl" if path.is_dir() else path
    if not results.exists():
        raise SystemExit(f"não encontrei {results}")
    latest: Dict[Tuple[int, int, str], Dict[str, Any]] = {}
    for line in results.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        latest[(r["rows"], r["cols"], r["engine"])] = r
    meta_file = results.parent / "meta.json"
    meta = json.loads(meta_file.read_text(encoding="utf-8")) if meta_file.exists() else {}
    return list(latest.values()), meta, results.parent


def engine_labels(rows: List[Dict[str, Any]]) -> List[str]:
    labels = {r.get("label") or r["engine"] for r in rows}
    ordered = [e for e in ENGINE_ORDER if e in labels]
    return ordered + sorted(labels - set(ordered))


def style_for(labels: List[str], theme: str) -> Dict[str, Dict[str, Any]]:
    t = THEMES[theme]
    colors = t["series"] + t["extra"]
    if len(labels) > len(t["series"]):
        print(
            f"[aviso] {len(labels)} engines: só as 4 primeiras cores foram validadas em todos os pares; "
            "confie nos marcadores e na tabela para as demais."
        )
    return {
        lab: {"color": colors[i % len(colors)], "marker": MARKERS[i % len(MARKERS)]} for i, lab in enumerate(labels)
    }


def runs_text(rows: List[Dict[str, Any]]) -> str:
    """'10' ou '10 (3 a partir de 10M linhas)' quando os tamanhos maiores usaram menos repetições."""
    by_rows: Dict[int, int] = {}
    for r in rows:
        by_rows[r["rows"]] = max(by_rows.get(r["rows"], 0), int(r.get("n_runs") or 0))
    if not by_rows:
        return "0"
    base = max(by_rows.values())
    fewer = {k: v for k, v in by_rows.items() if v < base}
    if not fewer:
        return str(base)
    return f"{base} ({min(fewer.values())} a partir de {fmt_rows(min(fewer))} linhas)"


def fmt_rows(v: float, _pos: Optional[int] = None) -> str:
    for div, suffix in ((1e6, "M"), (1e3, "k")):
        if v >= div:
            return f"{v / div:g}{suffix}"
    return f"{v:g}"


def fmt_value(v: float, _pos: Optional[int] = None) -> str:
    return f"{v:,.0f}" if v >= 1 else f"{v:g}"


# ----------------------------------------------------------------------------- gráficos
def draw_figure(
    rows: List[Dict[str, Any]],
    metric: str,
    ylabel: str,
    title: str,
    subtitle: str,
    theme: str,
    out_path: Path,
    dpi: int,
) -> None:
    t = THEMES[theme]
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": FONT_STACK})

    labels = engine_labels(rows)
    style = style_for(labels, theme)
    rows_vals = sorted({r["rows"] for r in rows})
    cols_vals = sorted({r["cols"] for r in rows})
    band = metric == "median_s"

    # limites comuns a todos os painéis (comparáveis entre si)
    values = [
        v
        for r in rows
        if r.get("status") == "ok"
        for v in ([r.get(metric), r.get("q1_s"), r.get("q3_s")] if band else [r.get(metric)])
        if isinstance(v, (int, float)) and v > 0
    ]
    if not values:
        print(f"[aviso] sem dados para {metric}; gráfico não gerado")
        return
    any_killed = any(r.get("status") != "ok" for r in rows)
    y_lo, y_hi = min(values) / 1.7, max(values) * (4.0 if any_killed else 1.8)
    killed_y = max(values) * 2.3

    n = len(cols_vals)
    ncols = min(4, n)
    nrows = math.ceil(n / ncols)
    header_in, footer_in = 1.35, 0.55
    fig_w, panel_h = 4.1 * ncols, 3.5
    fig_h = panel_h * nrows + header_in + footer_in
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), squeeze=False, sharex=True, sharey=True)
    fig.patch.set_facecolor(t["surface"])

    for idx, cols_n in enumerate(cols_vals):
        ax = axes[idx // ncols][idx % ncols]
        ax.set_facecolor(t["surface"])
        ax.set_xscale("log")
        ax.set_yscale("log")
        for lab in labels:
            pts = sorted(
                (r for r in rows if r["cols"] == cols_n and (r.get("label") or r["engine"]) == lab),
                key=lambda r: r["rows"],
            )
            ok_pts = [r for r in pts if r.get("status") == "ok" and isinstance(r.get(metric), (int, float))]
            color, marker = style[lab]["color"], style[lab]["marker"]
            xs = [r["rows"] for r in ok_pts]
            ys = [r[metric] for r in ok_pts]
            if band and ok_pts and all("q1_s" in r and "q3_s" in r for r in ok_pts):
                ax.fill_between(xs, [r["q1_s"] for r in ok_pts], [r["q3_s"] for r in ok_pts], color=color, alpha=0.10, lw=0)
            if xs:
                ax.plot(
                    xs,
                    ys,
                    color=color,
                    lw=1.44,  # 2px a 2x
                    solid_capstyle="round",
                    solid_joinstyle="round",
                    marker=marker,
                    markersize=6.2,  # >= 8px a 2x
                    markerfacecolor=color,
                    markeredgecolor=t["surface"],  # anel de 2px na cor da superfície
                    markeredgewidth=1.4,
                    zorder=3,
                )
            for r in pts:
                if r.get("status") != "ok":
                    ax.plot([r["rows"]], [killed_y], marker="x", color=color, markersize=7, markeredgewidth=1.8, zorder=4, clip_on=False)
        ax.set_title(f"{cols_n} colunas", loc="left", fontsize=10.5, color=t["ink"], pad=8, fontweight="semibold")
        ax.set_xticks(rows_vals)
        ax.xaxis.set_major_formatter(FuncFormatter(fmt_rows))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.yaxis.set_major_formatter(FuncFormatter(fmt_value))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_ylim(y_lo, y_hi)
        ax.grid(True, which="major", color=t["grid"], lw=0.7, ls="-")
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(t["axis"])
        ax.tick_params(axis="both", which="both", colors=t["muted"], labelsize=8.5, length=0)
        if idx % ncols == 0:
            ax.set_ylabel(ylabel, color=t["ink2"], fontsize=9)
        if idx // ncols == nrows - 1 or idx + ncols >= n:
            ax.set_xlabel("Linhas", color=t["ink2"], fontsize=9)
            ax.tick_params(axis="x", labelbottom=True)

    for idx in range(n, nrows * ncols):  # painéis sobrando
        axes[idx // ncols][idx % ncols].set_visible(False)

    # cabeçalho: título, subtítulo e legenda (sempre presente com >= 2 engines)
    fig.text(0.012, 1 - 0.18 / fig_h, title, ha="left", va="top", fontsize=15, color=t["ink"], fontweight="semibold")
    fig.text(0.012, 1 - 0.55 / fig_h, subtitle, ha="left", va="top", fontsize=9.5, color=t["ink2"])
    if len(labels) >= 2:
        handles = [
            Line2D(
                [0], [0], color=style[lab]["color"], lw=1.44, marker=style[lab]["marker"], markersize=6.2,
                markerfacecolor=style[lab]["color"], markeredgecolor=t["surface"], markeredgewidth=1.4, label=lab,
            )
            for lab in labels
        ]
        leg = fig.legend(
            handles=handles, loc="upper left", bbox_to_anchor=(0.006, 1 - 0.85 / fig_h), ncol=len(labels),
            frameon=False, fontsize=9.5, handlelength=2.4, columnspacing=1.8,
        )
        for text in leg.get_texts():
            text.set_color(t["ink"])  # texto em tinta, nunca na cor da série

    notes = ["eixos log-log", "faixa = intervalo interquartil" if band else None,
             "× = execução interrompida (RAM/timeout)" if any_killed else None,
             "tabela completa em summary.md"]
    fig.text(0.012, 0.012 + 0.12 / fig_h, " · ".join(x for x in notes if x), ha="left", va="bottom", fontsize=8.5, color=t["muted"])

    fig.tight_layout(rect=(0, footer_in / fig_h, 1, 1 - header_in / fig_h))
    fig.savefig(out_path, dpi=dpi, facecolor=t["surface"])
    plt.close(fig)
    print(f"gerado: {out_path}")


# ----------------------------------------------------------------------------- tabela
def cell(r: Optional[Dict[str, Any]], metric: str) -> str:
    if r is None:
        return "—"
    status = r.get("status")
    if status == "killed":
        return "✗ " + str(r.get("killed", "interrompido")).replace("rss>", "RAM>").replace("free<", "RAM livre<")
    if status == "error":
        return "✗ erro"
    v = r.get(metric)
    if not isinstance(v, (int, float)):
        return "—"
    text = fmt_seconds(v) if metric.endswith("_s") else f"{v:,.0f}"
    return text + (" ⚠" if r.get("ok") is False else "")


def fmt_seconds(v: float) -> str:
    """Segundos com algarismos significativos suficientes para milissegundos (0.00412) até horas (1,234)."""
    if v >= 100:
        return f"{v:,.0f}"
    if v >= 10:
        return f"{v:.1f}"
    if v >= 1:
        return f"{v:.2f}"
    return f"{v:.3g}"


def md_table(headers: List[str], body: List[List[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] + ["---:"] * (len(headers) - 1)) + "|"]
    out += ["| " + " | ".join(row) + " |" for row in body]
    return "\n".join(out)


def build_summary(rows: List[Dict[str, Any]], meta: Dict[str, Any], mem_metric: str, name: str) -> str:
    labels = engine_labels(rows)
    index = {(r["rows"], r["cols"], r.get("label") or r["engine"]): r for r in rows}
    rows_vals = sorted({r["rows"] for r in rows})
    cols_vals = sorted({r["cols"] for r in rows})
    n_runs = max((r.get("n_runs", 0) for r in rows), default=0)

    inv = (meta.get("invocations") or [{}])[-1]
    machine, libs, git = inv.get("machine", {}), inv.get("libs", {}), inv.get("git", {})
    impl = sorted({r["impl"] for r in rows if r.get("impl")})

    lines = [f"# Resultados do benchmark — {name}", ""]
    if machine:
        lines += [
            f"- **Máquina:** {machine.get('cpu', '?')} — {machine.get('cores_physical', '?')} núcleos físicos / "
            f"{machine.get('cores_logical', '?')} lógicos, {machine.get('ram_gb', '?')} GB de RAM, {machine.get('platform', '')}",
            f"- **Python:** {machine.get('python', '?')} | " + ", ".join(f"{k} {v}" for k, v in libs.items() if v != "não instalado"),
        ]
    if git.get("commit"):
        lines.append(f"- **Commit:** `{git['commit'][:10]}` ({git.get('branch', '?')}){' — com alterações não commitadas' if git.get('dirty') else ''}")
    if impl:
        lines.append("- **Implementações usadas:** " + ", ".join(f"`{i}`" for i in impl))
    lines += [
        f"- **Execuções:** {runs_text(rows)} medidas por caso (mais 1 de aquecimento descartada), cada uma em processo novo. Tabelas mostram a **mediana**.",
        "- **O que é medido:** leitura dos dados + comparação + geração do relatório, com igualdade exata (tolerância zero). "
        "Imports ficam fora do cronômetro. Os dados são parquet com ~1% de linhas divergentes e ~2% de nulos.",
        "- **Paralelismo:** datacompolars, diffly e datacompy (Polars) usam todos os núcleos via Polars; datacompy (pandas) é single-thread.",
        "- **Legenda:** `✗ RAM>…` / `✗ timeout>…` = execução interrompida; `✗ erro` = exceção; `—` = não medido ou pulado "
        "(uma engine que falha em R×C é pulada nos casos maiores); `⚠` = contagens diferentes do gabarito (resultado não confiável).",
        "",
    ]

    def table(metric: str) -> str:
        body: List[List[str]] = []
        for rv in rows_vals:
            for cv in cols_vals:
                if not any((rv, cv, lab) in index for lab in labels):
                    continue
                body.append([f"{rv:,}", str(cv)] + [cell(index.get((rv, cv, lab)), metric) for lab in labels])
        return md_table(["Linhas", "Colunas"] + labels, body)

    lines += ["## Tempo mediano (s)", "", table("median_s"), ""]
    lines += [f"## {MEM_LABELS[mem_metric]}", "", table(mem_metric), ""]

    base_label = labels[0]
    others = labels[1:]
    if others and rows_vals:
        top = rows_vals[-1]
        ratio_rows: List[List[str]] = []
        for cv in cols_vals:
            base = index.get((top, cv, base_label))
            if not base or base.get("status") != "ok":
                continue
            cells = []
            for lab in others:
                o = index.get((top, cv, lab))
                cells.append(f"{o['median_s'] / base['median_s']:.1f}×" if o and o.get("status") == "ok" and base["median_s"] > 0 else cell(o, "median_s"))
            ratio_rows.append([str(cv)] + cells)
        if ratio_rows:
            lines += [
                f"## Tempo relativo ao {base_label}, com {top:,} linhas",
                "",
                f"Valores > 1× significam mais lento que {base_label}.",
                "",
                md_table(["Colunas"] + others, ratio_rows),
                "",
            ]
    return "\n".join(lines)


# ----------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", type=Path, help="pasta da suíte (com results.jsonl) ou o próprio results.jsonl")
    ap.add_argument("--theme", choices=["light", "dark", "both"], default="both")
    ap.add_argument("--mem-metric", choices=list(MEM_LABELS), default="peak_median_mb")
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--out", type=Path, default=None, help="pasta de saída (default: a pasta da suíte)")
    ap.add_argument(
        "--hide-failed", action="store_true",
        help="nos GRÁFICOS, omite as execuções interrompidas/com erro (sem o marcador ×): a engine simplesmente "
        "não aparece nesse tamanho. O summary.md continua mostrando as falhas",
    )
    ap.add_argument(
        "--engines", default=None,
        help="só estas engines (ids ou rótulos, separados por vírgula), ex.: hash,hash_windows,diffly. "
        "Use com --out para gerar um segundo conjunto de gráficos sem sobrescrever o primeiro",
    )
    args = ap.parse_args()

    rows, meta, suite_dir = load(args.path)
    if args.engines:
        wanted = {e.strip() for e in args.engines.split(",") if e.strip()}
        rows = [r for r in rows if r["engine"] in wanted or r.get("label") in wanted]
    if not rows:
        raise SystemExit("results.jsonl está vazio (ou nenhuma engine casou com --engines)")
    out_dir = args.out or suite_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    name = meta.get("name") or suite_dir.name

    n_runs = max((r.get("n_runs", 0) for r in rows), default=0)
    inv = (meta.get("invocations") or [{}])[-1]
    machine = inv.get("machine", {})
    where = f"{machine.get('cpu', '')} · {machine.get('ram_gb', '?')} GB RAM".strip(" ·") if machine else ""
    subtitle = f"Mediana de {runs_text(rows)} execuções por ponto, cada uma em processo novo" + (f" · {where}" if where else "")

    chart_rows = [r for r in rows if r.get("status") == "ok"] if args.hide_failed else rows
    themes = ["light", "dark"] if args.theme == "both" else [args.theme]
    for theme in themes:
        draw_figure(chart_rows, "median_s", "Tempo mediano (s)", "Tempo de comparação x tamanho do dataset", subtitle, theme, out_dir / f"time_vs_rows_{theme}.png", args.dpi)
        draw_figure(chart_rows, args.mem_metric, MEM_LABELS[args.mem_metric].split(" (")[0] + " (MB)", "Memória x tamanho do dataset", subtitle, theme, out_dir / f"memory_vs_rows_{theme}.png", args.dpi)

    (out_dir / "summary.md").write_text(build_summary(rows, meta, args.mem_metric, name), encoding="utf-8")
    print(f"gerado: {out_dir / 'summary.md'}")


if __name__ == "__main__":
    main()