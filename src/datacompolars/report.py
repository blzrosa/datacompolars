"""Relatório: um modelo neutro de blocos (build_blocks) renderizado em texto, markdown, HTML e JSON.

O conteúdo é montado uma única vez; cada renderizador só decide a apresentação. Texto e Markdown
respeitam `max_columns` / `max_sample_columns`; HTML e JSON trazem tudo.
"""
from __future__ import annotations

import html as _html
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from .results import ComparisonResult
from .settings import ReportSettings

# ----------------------------------------------------------------------------- modelo neutro
@dataclass
class Bar:
    frac: float


Cell = Union[str, int, float, bool, None, Bar]


@dataclass
class Heading:
    text: str
    level: int = 2


@dataclass
class Verdict:
    level: str  # ok | diff | error
    text: str
    detail: str = ""


@dataclass
class Facts:
    items: List[Tuple[str, str]]


@dataclass
class Lines:
    lines: List[str]
    kind: str = "plain"  # plain | note | warn | error


@dataclass
class Table:
    headers: List[str]
    rows: List[List[Cell]]
    align: List[str]  # "l" | "r" | "c" por coluna
    caption: str = ""


Block = Union[Heading, Verdict, Facts, Lines, Table]


# ----------------------------------------------------------------------------- formatação de valores
def _n(x: int) -> str:
    return f"{x:,}"


def _pct(frac: float) -> str:
    if frac <= 0:
        return "0%"
    if frac >= 1:
        return "100%"
    p = frac * 100
    return "<0.01%" if p < 0.01 else f"{p:.2f}%"


def _dur(seconds: float) -> str:
    if seconds < 1:
        return f"{seconds * 1000:.0f} ms"
    if seconds < 60:
        return f"{seconds:.2f} s"
    m, s = divmod(seconds, 60)
    return f"{int(m)}m {s:.0f}s"


def _value(v: Any, limit: int) -> str:
    if v is None:
        return "null"
    if isinstance(v, float):
        return "NaN" if v != v else f"{v:.6g}"
    s = str(v).replace("\n", " ").replace("\r", " ")
    return s if len(s) <= limit else s[: limit - 3] + "..."


def _names(names: Sequence[str], limit: int = 20) -> str:
    shown = ", ".join(names[:limit])
    return shown + (f" (+{len(names) - limit} more)" if len(names) > limit else "")


# ----------------------------------------------------------------------------- conteúdo
def build_blocks(r: ComparisonResult, st: ReportSettings, full: bool = False) -> List[Block]:
    L, R = r.left_name, r.right_name
    rows, cols = r.rows, r.columns
    value_limit = 80 if full else 30
    b: List[Block] = [
        Heading("DataComPolars Comparison Report", 1),
        Facts(
            [
                ("Compared", f"{L}  vs  {R}"),
                ("Join key(s)", ", ".join(r.join_columns)),
                ("Generated", r.created_at.strftime("%Y-%m-%d %H:%M:%S UTC")),
            ]
        ),
    ]

    # --- veredito
    if r.aborted:
        b.append(Verdict("error", "COMPARISON ABORTED", "The row comparison could not be performed."))
    elif r.is_match:
        b.append(Verdict("ok", "IDENTICAL", "Same rows, same values and same columns."))
    else:
        parts = []
        if rows.mismatched:
            parts.append(f"{_n(rows.mismatched)} of {_n(rows.common)} common rows differ ({_pct(rows.mismatch_rate)})")
        if rows.left_only:
            parts.append(f"{_n(rows.left_only)} rows only in {L}")
        if rows.right_only:
            parts.append(f"{_n(rows.right_only)} rows only in {R}")
        if cols.left_only or cols.right_only:
            parts.append("the column sets differ")
        if cols.type_mismatches:
            parts.append(f"{len(cols.type_mismatches)} columns have different types")
        b.append(Verdict("diff", "DIFFERENCES FOUND", "; ".join(parts) + "."))

    if r.fatal_errors:
        b += [Heading("Errors"), Lines(r.fatal_errors, "error")]
    if r.warnings:
        b += [Heading("Warnings"), Lines(r.warnings, "warn")]

    # --- linhas
    if not r.aborted:
        common = rows.common
        share = (lambda x: _pct(x / common)) if common else (lambda x: "")
        frac = (lambda x: Bar(x / common)) if common else (lambda x: "")
        b += [
            Heading("Rows"),
            Table(
                ["Category", "Rows", "% of common", ""],
                [
                    ["In common", _n(common), "", ""],
                    ["  equal", _n(rows.matched), share(rows.matched), frac(rows.matched)],
                    ["  with differences", _n(rows.mismatched), share(rows.mismatched), frac(rows.mismatched)],
                    [f"Only in {L}", _n(rows.left_only), "", ""],
                    [f"Only in {R}", _n(rows.right_only), "", ""],
                    [f"Total in {L}", _n(rows.left_total), "", ""],
                    [f"Total in {R}", _n(rows.right_total), "", ""],
                ],
                ["l", "r", "r", "l"],
            ),
        ]

    # --- colunas
    b += [
        Heading("Columns"),
        Facts(
            [
                ("Compared", _n(len(cols.compared))),
                (f"Only in {L}", _n(len(cols.left_only))),
                (f"Only in {R}", _n(len(cols.right_only))),
                ("Different types (skipped)", _n(len(cols.type_mismatches))),
            ]
        ),
    ]
    notes = []
    if cols.left_only:
        notes.append(f"Only in {L}: {_names(cols.left_only)}")
    if cols.right_only:
        notes.append(f"Only in {R}: {_names(cols.right_only)}")
    if notes:
        b.append(Lines(notes, "note"))
    if cols.type_mismatches:
        b.append(
            Table(
                ["Column", f"Type in {L}", f"Type in {R}"],
                [[t.column, t.left, t.right] for t in cols.type_mismatches],
                ["l", "l", "l"],
                "Columns with different types (not compared)",
            )
        )

    # --- divergências por coluna
    if r.column_stats:
        shown = r.column_stats if full else r.column_stats[: st.max_columns]
        b += [
            Heading("Differences by column"),
            Table(
                ["Column", "Type", "Differences", "% of common", ""],
                [[c.column, c.dtype, _n(c.mismatches), _pct(c.rate), Bar(c.rate)] for c in shown],
                ["l", "l", "r", "r", "l"],
            ),
        ]
        if len(shown) < len(r.column_stats):
            b.append(
                Lines(
                    [f"{len(r.column_stats) - len(shown)} more columns with differences (use the JSON or HTML report for the full list)."],
                    "note",
                )
            )
    elif rows.mismatched and not r.execution.column_details_computed:
        b += [Heading("Differences by column"), Lines(["Per-column details were not computed (column_details=False)."], "note")]

    # --- amostras
    if r.samples and (full or st.max_sample_columns > 0):
        shown_s = r.samples if full else r.samples[: st.max_sample_columns]
        b.append(Heading("Sample differing rows"))
        for t in shown_s:
            b.append(
                Table(
                    t.headers,
                    [[_value(v, value_limit) for v in row] for row in t.rows],
                    ["l"] * len(t.headers),
                    f"{t.column}: {_n(t.total_mismatches)} differing rows, showing {len(t.rows)}",
                )
            )
        if len(shown_s) < len(r.samples):
            b.append(Lines([f"Samples for {len(r.samples) - len(shown_s)} more columns omitted."], "note"))

    # --- execução
    ex = r.execution
    facts: List[Tuple[str, str]] = []
    if ex.path:
        facts.append(("Method", f"{ex.path}  ({ex.path_reason})" if ex.path_reason else ex.path))
    facts.append(("Execution windows", f"{ex.windows}" + (f"  (~{_n(ex.rows_per_window)} rows each)" if ex.rows_per_window else "")))
    facts.append(("Column details", "computed" if ex.column_details_computed else "skipped"))
    facts.append(("Total time", _dur(ex.total_seconds)))
    if ex.timings:
        facts.append(("Time by phase", ", ".join(f"{k} {_dur(v)}" for k, v in ex.timings.items())))
    if ex.polars_version:
        facts.append(("Polars", ex.polars_version))
    b += [Heading("Execution"), Facts(facts)]
    return b


# ----------------------------------------------------------------------------- texto
_UNICODE = dict(h="─", v="│", tl="┌", tr="┐", bl="└", br="┘", tj="┬", bj="┴", lj="├", rj="┤", x="┼",
                dh="═", full="█", empty="░", ell="…", ok="✔", diff="✘", error="✖", warn="!", note="·")
_ASCII = dict(h="-", v="|", tl="+", tr="+", bl="+", br="+", tj="+", bj="+", lj="+", rj="+", x="+",
              dh="=", full="#", empty=".", ell="...", ok="[OK]", diff="[DIFF]", error="[ERROR]", warn="!", note="-")
_BAR_WIDTH = 12


def _use_unicode(style: str) -> bool:
    if style != "auto":
        return style == "unicode"
    enc = getattr(sys.stdout, "encoding", None) or "ascii"
    try:
        "─█░✔✘✖…·".encode(enc)
        return True
    except (UnicodeEncodeError, LookupError):
        return False


def _trunc(s: str, w: int, ell: str) -> str:
    return s if len(s) <= w else s[: max(0, w - len(ell))] + ell


def _pad(s: str, w: int, align: str) -> str:
    return s.rjust(w) if align == "r" else s.center(w) if align == "c" else s.ljust(w)


def _text_table(t: Table, width: int, cs: Dict[str, str]) -> List[str]:
    def cell(c: Cell) -> str:
        if isinstance(c, Bar):
            k = round(max(0.0, min(1.0, c.frac)) * _BAR_WIDTH)
            k = max(k, 1) if c.frac > 0 else 0
            return cs["full"] * k + cs["empty"] * (_BAR_WIDTH - k)
        return "" if c is None else str(c)

    body = [[cell(c) for c in row] for row in t.rows]
    n = len(t.headers)
    widths = [max([len(t.headers[i])] + [len(row[i]) for row in body]) for i in range(n)]
    budget = width - 2  # recuo
    while sum(widths) + 3 * n + 1 > budget and max(widths) > 8:
        widths[widths.index(max(widths))] -= 1

    def line(left: str, mid: str, right: str) -> str:
        return left + mid.join(cs["h"] * (w + 2) for w in widths) + right

    def row(cells: Sequence[str], aligns: Sequence[str]) -> str:
        parts = [" " + _pad(_trunc(c, w, cs["ell"]), w, a) + " " for c, w, a in zip(cells, widths, aligns)]
        return cs["v"] + cs["v"].join(parts) + cs["v"]

    out = []
    if t.caption:
        out.append(f"  {t.caption}")
    out.append("  " + line(cs["tl"], cs["tj"], cs["tr"]))
    out.append("  " + row(t.headers, ["l"] * n))
    out.append("  " + line(cs["lj"], cs["x"], cs["rj"]))
    out += ["  " + row(r_, t.align) for r_ in body]
    out.append("  " + line(cs["bl"], cs["bj"], cs["br"]))
    return out


def render_text(r: ComparisonResult, st: ReportSettings) -> str:
    unicode_ok = _use_unicode(st.style)
    cs = _UNICODE if unicode_ok else _ASCII
    W = st.width
    out: List[str] = []

    for blk in build_blocks(r, st):
        if isinstance(blk, Heading):
            if blk.level == 1:
                out += [cs["dh"] * W, f"  {blk.text}", cs["dh"] * W]
            else:
                out += ["", blk.text.upper(), cs["h"] * len(blk.text)]
        elif isinstance(blk, Verdict):
            title = f"{cs[blk.level]} {blk.text}"
            inner = [title] + textwrap.wrap(blk.detail, W - 6) if blk.detail else [title]
            box_w = max(len(x) for x in inner) + 2
            out += ["", cs["tl"] + cs["h"] * box_w + cs["tr"]]
            out += [cs["v"] + " " + x.ljust(box_w - 1) + cs["v"] for x in inner]
            out += [cs["bl"] + cs["h"] * box_w + cs["br"]]
        elif isinstance(blk, Facts):
            lw = max(len(k) for k, _ in blk.items)
            out += [""] if out and out[-1] != "" and not isinstance(blk, Heading) else []
            for k, v in blk.items:
                out.append(textwrap.fill(f"{k.ljust(lw)}  {v}", W, initial_indent="  ", subsequent_indent="  " + " " * (lw + 2)))
        elif isinstance(blk, Lines):
            mark = {"error": cs["error"], "warn": cs["warn"], "note": cs["note"], "plain": cs["note"]}[blk.kind]
            for ln in blk.lines:
                out.append(textwrap.fill(ln, W, initial_indent=f"  {mark} ", subsequent_indent="    "))
        elif isinstance(blk, Table):
            out += [""] + _text_table(blk, W, cs)

    text = "\n".join(out).strip("\n") + "\n"
    return text if unicode_ok else text.encode("ascii", "replace").decode("ascii")


# ----------------------------------------------------------------------------- markdown
def _md(s: Any) -> str:
    return str(s).replace("|", "\\|").replace("\n", " ")


def render_markdown(r: ComparisonResult, st: ReportSettings) -> str:
    out: List[str] = []
    icon = {"ok": "✅", "diff": "⚠️", "error": "❌"}
    for blk in build_blocks(r, st):
        if isinstance(blk, Heading):
            out += ["", "#" * blk.level + " " + blk.text, ""]
        elif isinstance(blk, Verdict):
            out += ["", f"> **{icon[blk.level]} {blk.text}**", f"> {blk.detail}" if blk.detail else "", ""]
        elif isinstance(blk, Facts):
            out += [f"- **{k}:** {_md(v)}" for k, v in blk.items]
        elif isinstance(blk, Lines):
            pre = {"error": "❌ ", "warn": "⚠️ ", "note": "", "plain": ""}[blk.kind]
            out += [""] + [f"- {pre}{_md(x)}" for x in blk.lines]
        elif isinstance(blk, Table):
            def cell(c: Cell) -> str:
                if isinstance(c, Bar):
                    k = round(max(0.0, min(1.0, c.frac)) * _BAR_WIDTH)
                    return "`" + "█" * k + "░" * (_BAR_WIDTH - k) + "`"
                return _md("" if c is None else c)

            sep = {"l": ":---", "r": "---:", "c": ":---:"}
            out += [""]
            if blk.caption:
                out += [f"**{_md(blk.caption)}**", ""]
            out.append("| " + " | ".join(_md(h) for h in blk.headers) + " |")
            out.append("| " + " | ".join(sep[a] for a in blk.align) + " |")
            out += ["| " + " | ".join(cell(c) for c in row) + " |" for row in blk.rows]
    text = "\n".join(out)
    while "\n\n\n" in text:
        text = text.replace("\n\n\n", "\n\n")
    return text.strip() + "\n"


# ----------------------------------------------------------------------------- html
_CSS = """
:root{--bg:#f6f7f9;--card:#fff;--fg:#1c2330;--muted:#667085;--line:#e4e7ec;--accent:#3b6fd4;
--ok:#12805c;--ok-bg:#e3f6ee;--diff:#a15c07;--diff-bg:#fdf1dc;--error:#b42318;--error-bg:#fde7e5;--bar:#e4e7ec}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#0f141b;--card:#171e29;--fg:#e6e9ef;
--muted:#98a2b3;--line:#2a3342;--accent:#7aa2f7;--ok:#4fd1a1;--ok-bg:#12302a;--diff:#f2b45a;--diff-bg:#33280f;
--error:#ff8a80;--error-bg:#3a1b19;--bar:#2a3342}}
:root[data-theme="dark"]{--bg:#0f141b;--card:#171e29;--fg:#e6e9ef;--muted:#98a2b3;--line:#2a3342;--accent:#7aa2f7;
--ok:#4fd1a1;--ok-bg:#12302a;--diff:#f2b45a;--diff-bg:#33280f;--error:#ff8a80;--error-bg:#3a1b19;--bar:#2a3342}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
main{max-width:1000px;margin:0 auto;padding:28px 18px 60px}
h1{font-size:24px;margin:0 0 12px}
h2{font-size:13px;letter-spacing:.08em;text-transform:uppercase;color:var(--muted);margin:30px 0 10px;
border-bottom:1px solid var(--line);padding-bottom:6px}
h3{font-size:14px;margin:18px 0 6px;font-weight:600}
dl{display:grid;grid-template-columns:max-content 1fr;gap:4px 18px;margin:8px 0}
dt{color:var(--muted)} dd{margin:0;word-break:break-word}
.verdict{border-radius:10px;padding:14px 18px;margin:16px 0;border:1px solid var(--line)}
.verdict b{font-size:17px;display:block}
.verdict.ok{background:var(--ok-bg);color:var(--ok)} .verdict.diff{background:var(--diff-bg);color:var(--diff)}
.verdict.error{background:var(--error-bg);color:var(--error)}
.verdict span{color:var(--fg);opacity:.85}
ul.lines{margin:6px 0;padding-left:20px} ul.error li{color:var(--error)} ul.warn li{color:var(--diff)} ul.note li{color:var(--muted)}
.tw{overflow-x:auto;background:var(--card);border:1px solid var(--line);border-radius:10px;margin:6px 0 14px}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
th,td{padding:7px 12px;text-align:left;white-space:nowrap;border-bottom:1px solid var(--line)}
th{font-size:12px;color:var(--muted);font-weight:600;text-transform:uppercase;letter-spacing:.04em}
tr:last-child td{border-bottom:0} td.r,th.r{text-align:right} td.c,th.c{text-align:center}
.bar{width:120px;height:8px;border-radius:4px;background:var(--bar);overflow:hidden}
.bar i{display:block;height:100%;background:var(--accent)}
"""


def render_html(r: ComparisonResult, st: ReportSettings) -> str:
    e = _html.escape
    body: List[str] = []
    icon = {"ok": "&#10004;", "diff": "&#9888;", "error": "&#10006;"}
    for blk in build_blocks(r, st, full=True):
        if isinstance(blk, Heading):
            body.append(f"<h{blk.level}>{e(blk.text)}</h{blk.level}>")
        elif isinstance(blk, Verdict):
            detail = f"<span>{e(blk.detail)}</span>" if blk.detail else ""
            body.append(f'<div class="verdict {blk.level}"><b>{icon[blk.level]} {e(blk.text)}</b>{detail}</div>')
        elif isinstance(blk, Facts):
            items = "".join(f"<dt>{e(k)}</dt><dd>{e(v)}</dd>" for k, v in blk.items)
            body.append(f"<dl>{items}</dl>")
        elif isinstance(blk, Lines):
            items = "".join(f"<li>{e(x)}</li>" for x in blk.lines)
            body.append(f'<ul class="lines {blk.kind}">{items}</ul>')
        elif isinstance(blk, Table):
            def cell(c: Cell) -> str:
                if isinstance(c, Bar):
                    w = 0.0 if c.frac <= 0 else max(2.0, min(100.0, c.frac * 100))
                    return f'<div class="bar"><i style="width:{w:.1f}%"></i></div>'
                return e("" if c is None else str(c)).replace("  ", "&nbsp;&nbsp;")

            head = "".join(f'<th class="{a}">{e(h)}</th>' for h, a in zip(blk.headers, blk.align))
            rows = "".join(
                "<tr>" + "".join(f'<td class="{a}">{cell(c)}</td>' for c, a in zip(row, blk.align)) + "</tr>"
                for row in blk.rows
            )
            cap = f"<h3>{e(blk.caption)}</h3>" if blk.caption else ""
            body.append(f'{cap}<div class="tw"><table><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table></div>')
    title = e(f"DataComPolars: {r.left_name} vs {r.right_name}")
    return (
        '<!DOCTYPE html>\n<html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>{title}</title><style>{_CSS}</style></head><body><main>\n"
        + "\n".join(body)
        + "\n</main></body></html>\n"
    )


# ----------------------------------------------------------------------------- saída
def render_json(r: ComparisonResult, st: ReportSettings) -> str:
    return r.to_json()


_RENDERERS: Dict[str, Callable[[ComparisonResult, ReportSettings], str]] = {
    "text": render_text,
    "markdown": render_markdown,
    "html": render_html,
    "json": render_json,
}


def _safe_print(text: str) -> None:
    try:
        print(text)
    except UnicodeEncodeError:
        enc = getattr(sys.stdout, "encoding", None) or "ascii"
        print(text.encode(enc, "replace").decode(enc))


def emit(result: ComparisonResult, settings: Optional[ReportSettings] = None) -> str:
    """Renderiza o relatório, salva (UTF-8) se `save_path` e imprime se `print_output`. Devolve o texto."""
    settings = settings or ReportSettings()
    text = _RENDERERS[settings.resolved_format()](result, settings)
    if settings.save_path is not None:
        path = Path(settings.save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    if settings.print_output:
        _safe_print(text)
    return text
