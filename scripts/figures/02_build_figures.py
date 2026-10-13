"""Paper figures built from committed result files only.

Usage: python scripts/figures/02_build_figures.py [--out figures]   (from the repository root)

Every number drawn is read from a result file; nothing is typed in by hand. The figures:
  fig1_pipeline         candidates to briefings in the represcription, with the count kept at each stage
  fig2_hypergraph       one represcription triad as an open triangle in the paper hypergraph
                        (figures/data/hypergraph_example.json, from scripts/figures/01_extract_hypergraph.py);
                        caption: numbers are papers holding the pair, shaded areas are example papers (hyperedges)
  fig3_verdict          third-study T3 verdict steps S1-S4 with both bootstrap intervals and thresholds
  fig4_precision_k      precision and lift at each k for the engine, the nPMI baseline and random
  fig5_measurement      label accuracy by attempt, and the S3 interval before and after the label fix
  fig6_judging          earlier and blind re-judged Q1 by field, and overall Q1 of studies four and five
  fig7_outcomes         concept-frame outcomes of studies one, four and five, and the riskiest DVF axis
  fig8_timeline         preregistration freezes, prospective registrations and the planned verdicts
Each figure is written as PDF and PNG; figures/manifest.json lists the input files and their hashes.
"""
import argparse
import datetime
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon

# Shared code lives in the triadrx package at the project root
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from triadrx.study import load_study

COLORS = {"engine": "#1f5f8b", "baseline": "#d08c2a", "random": "#8a8a8a", "earlier": "#b0b0b0", "now": "#1f5f8b",
          "pass": "#2e7d32", "fail": "#c62828", "neutral": "#555555"}
STUDY_NAMES = {"first": "Study 1", "represcription": "Study 4", "faithfulness": "Study 5"}
plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42, "svg.hashsalt": "triadrx"})


class Inputs:
    """Result files read for the figures, with their hashes for the manifest."""

    def __init__(self):
        self.used = {}

    def json(self, path: str) -> dict:
        p = ROOT / path
        self.used[path] = hashlib.sha256(p.read_bytes()).hexdigest()
        return json.loads(p.read_text())


def save(fig, out: Path, name: str) -> list[str]:
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight", metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(out / f"{name}.png", bbox_inches="tight", dpi=200, metadata={"Software": None})
    plt.close(fig)
    return [f"{name}.pdf", f"{name}.png"]


def short(term: str) -> str:
    kind, name = term.split(":", 1)
    return f"{name} ({'method' if kind == 'M' else 'task'})"


def fig_pipeline(inp: Inputs, study: dict, out: Path) -> list[str]:
    run = study["represcription"]["run_dir"]
    main = inp.json(f"{run}/results/applied_main.json")
    papers = inp.json(f"{run}/results/evidence_papers.json")["stats"]
    dvf = inp.json(f"{run}/results/dvf.json")["candidates"]
    gen = inp.json(f"{run}/results/briefing_generation.json")["summary"]
    v_missing = sum(1 for c in dvf if c["axes"]["V"]["score"] is None)
    top_n = study["applied"]["briefing"]["top_n"]
    stages = [("Generator\ncandidates", f"{main['redundancy']['candidates']:,}"),
              ("Redundancy\nfilter", f"{main['redundancy']['kept']:,}"),
              ("Registered\nprescriptions", f"{len(main['top_k_prescription'])}"),
              ("Evidence\nbundles", f"{papers['candidates']} triads\n{papers['distinct_papers']:,} papers"),
              ("DVF\nscores", f"{len(dvf)} triads\nV missing {v_missing}"),
              ("Concept\nframes", f"top {top_n} x {gen['runs'] // top_n} runs\ncomplete {gen['frame_complete']}"),
              ("Briefings\nand MVPs", f"{gen['briefing_valid']} valid")]
    fig, ax = plt.subplots(figsize=(7.2, 1.7))
    ax.set_xlim(0, len(stages)), ax.set_ylim(0, 1), ax.axis("off")
    for i, (title, count) in enumerate(stages):
        ax.add_patch(FancyBboxPatch((i + 0.07, 0.08), 0.86, 0.84, boxstyle="round,pad=0.02", fc="#eef3f7", ec=COLORS["engine"]))
        ax.text(i + 0.5, 0.70, title, ha="center", va="center", fontsize=6.5, weight="bold")
        ax.text(i + 0.5, 0.32, count, ha="center", va="center", fontsize=6.5)
        if i:
            ax.annotate("", (i + 0.06, 0.5), (i - 0.06, 0.5), arrowprops={"arrowstyle": "->", "color": COLORS["neutral"]})
    return save(fig, out, "fig1_pipeline")


def fig_hypergraph(inp: Inputs, out: Path) -> list[str]:
    hg = inp.json("figures/data/hypergraph_example.json")
    a, b, c = hg["triad"]
    pos = {a: (0.0, 1.3), b: (-1.3, -0.7), c: (1.3, -0.7)}
    centre = (sum(x for x, _ in pos.values()) / 3, sum(y for _, y in pos.values()) / 3)
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.axis("off"), ax.set_aspect("equal")
    hull = {"fc": COLORS["engine"], "ec": COLORS["engine"], "alpha": 0.07, "lw": 7, "joinstyle": "round", "zorder": 0}
    for key, pair in hg["pairs"].items():
        x, y = key.split("|")
        (x1, y1), (x2, y2) = pos[x], pos[y]
        ax.plot([x1, x2], [y1, y2], color=COLORS["engine"], lw=1.6, zorder=1)
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        dx, dy = mx - centre[0], my - centre[1]
        norm = (dx * dx + dy * dy) ** 0.5 or 1.0
        dx, dy = dx / norm, dy / norm
        ax.text(mx + 0.13 * dx, my + 0.13 * dy, f"{pair['papers']:,}", ha="center", va="center", fontsize=7, color=COLORS["engine"],
                weight="bold", bbox={"fc": "white", "ec": "none", "pad": 0.6})
        example = next((e for e in pair["examples"] if e["other_terms"]), None)
        if example:
            other = (mx + 1.05 * dx, my + 1.05 * dy)
            ax.add_patch(Polygon([pos[x], pos[y], other], closed=True, **hull))
            ax.scatter([other[0]], [other[1]], s=60, color="white", edgecolor=COLORS["neutral"], lw=1, zorder=2)
            ax.text(other[0] + 0.4 * dx, other[1] + 0.4 * dy, f"{example['other_terms'][0].split(':', 1)[1]}\npaper {example['paper_id']}",
                    ha="center", va="center", fontsize=6, color=COLORS["neutral"])
    offsets = {a: (0, 0.24, "center"), b: (0, -0.26, "center"), c: (0, -0.26, "center")}
    for t, (x, y) in pos.items():
        ox, oy, ha = offsets[t]
        ax.scatter([x], [y], s=260, color="white", edgecolor=COLORS["engine"], lw=1.6, zorder=2)
        ax.text(x + ox, y + oy, short(t), ha=ha, va="center", fontsize=7, weight="bold")
    ax.text(centre[0], centre[1] - 0.15, f"all three together:\n{hg['papers_with_all_three']} papers\n(open triangle: a candidate)",
            ha="center", va="center", fontsize=7, color=COLORS["fail"])
    ax.set_xlim(-2.6, 2.6), ax.set_ylim(-2.3, 2.3)
    return save(fig, out, "fig2_hypergraph")


def verdict_steps(v: dict, k: str) -> list[tuple]:
    by = v["by_k"][k]
    e = by["engine"]
    return [("S1 recall gain", v["S1_recall"]["point"], v["S1_recall"]),
            ("S2 lift vs random", e["lift"], by["S2_lift_ci"]),
            ("S3 precision gain vs nPMI", e["precision"] - by["baseline_precision"], by["S3_diff_ci"]),
            ("S4 contribution of H", e["precision"] - by["engine_no_h_precision"], by["S4_diff_ci"])]


def fig_verdict(inp: Inputs, study: dict, out: Path) -> list[str]:
    v = inp.json("results/engine_v3_verdict_T3.json")
    k = str(study["matching"]["verdict"]["k"])
    thresholds = {s["step"]: s["threshold"] for s in study["engine_reliability"]["verdict"]["sequence"]}
    keys = ["S1_recall", "S2_random", "S3_npmi", "S4_h_contribution"]
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 1.9))
    for ax, key, (title, point, ci) in zip(axes, keys, verdict_steps(v, k)):
        for y, (name, part) in enumerate((("candidate", ci["candidate"]), ("term block", ci["term_block"]))):
            ax.plot([part["lower"], part["upper"]], [y, y], color=COLORS["engine"], lw=2)
            ax.text(part["upper"], y + 0.18, name, fontsize=6, ha="right", color=COLORS["neutral"])
        ax.scatter([point], [0.5], marker="D", color="black", s=14, zorder=3)
        ax.axvline(thresholds[key], color=COLORS["fail"], ls="--", lw=1)
        reading = v["reading"][key]
        ax.set_title(f"{title}\n{reading}", fontsize=7, color=COLORS["pass"] if reading == "pass" else COLORS["neutral"])
        ax.set_yticks([]), ax.set_ylim(-0.6, 1.6)
    fig.suptitle(f"T3 verdict at top k = {int(k):,} (dashed: threshold)", fontsize=8, y=1.08)
    return save(fig, out, "fig3_verdict")


def fig_precision_k(inp: Inputs, study: dict, out: Path) -> list[str]:
    v = inp.json("results/engine_v3_verdict_T3.json")
    verdict_k = study["matching"]["verdict"]["k"]
    ks = sorted(int(k) for k in v["by_k"])
    base_rate = v["labels"]["pool_positive"] / v["labels"]["pool_candidates"]
    eng = [v["by_k"][str(k)]["engine"]["precision"] for k in ks]
    npmi = [v["by_k"][str(k)]["baseline_precision"] for k in ks]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.4))
    ax1.plot(ks, eng, "o-", color=COLORS["engine"], label="engine")
    ax1.plot(ks, npmi, "s-", color=COLORS["baseline"], label="nPMI baseline")
    ax1.axhline(base_rate, color=COLORS["random"], ls=":", label="random")
    ax1.set_xscale("log"), ax1.set_xlabel("top k"), ax1.set_ylabel("precision"), ax1.legend(frameon=False)
    ax2.plot(ks, [p / base_rate for p in eng], "o-", color=COLORS["engine"], label="engine")
    ax2.plot(ks, [p / base_rate for p in npmi], "s-", color=COLORS["baseline"], label="nPMI baseline")
    ax2.axhline(1, color=COLORS["random"], ls=":", label="random")
    ax2.set_xscale("log"), ax2.set_xlabel("top k"), ax2.set_ylabel("lift over random"), ax2.legend(frameon=False)
    for ax in (ax1, ax2):
        ax.axvline(verdict_k, color=COLORS["neutral"], ls="--", lw=0.8)
        ax.text(verdict_k, 0.97, "verdict k ", fontsize=6, color=COLORS["neutral"], va="top", ha="right", transform=ax.get_xaxis_transform())
    return save(fig, out, "fig4_precision_k")


def fig_measurement(inp: Inputs, study: dict, out: Path) -> list[str]:
    attempts = [("Study 2 rule, first", "results/label_evidence_T1_score.json"),
                ("Study 2 rule, adopted", "results/label_evidence_T1_attempt1_score.json"),
                ("Study 3 rule", "results/label_v3_T1_score.json"),
                ("Study 3 classifier, 1st check", "results/label_v3_T1_attempt1_score.json"),
                ("Study 3 classifier, 2nd check", "results/label_v3_T1_attempt2_score.json")]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.6), gridspec_kw={"width_ratios": [1.3, 1]})
    for y, (name, path) in enumerate(attempts):
        s = inp.json(path)
        rp = s["rule_positive"]
        color = COLORS["engine"] if s.get("adopted") else COLORS["earlier"]
        ax1.plot([rp["wilson_lower"], rp["wilson_upper"]], [y, y], color=color, lw=3)
        ax1.scatter([rp["combination_share"]], [y], color="black", s=12, zorder=3)
        ax1.text(1.02, y, f"{rp['combination_share']:.2f} (n={rp['n']})", va="center", fontsize=6, transform=ax1.get_yaxis_transform())
    ax1.axvline(study["matching"]["label_v3"]["target_positive_lower"], color=COLORS["fail"], ls="--", lw=1)
    ax1.set_yticks(range(len(attempts)), [a[0] for a in attempts]), ax1.invert_yaxis()
    ax1.set_xlabel("combination share among label positives (Wilson 95%)"), ax1.set_xlim(0.3, 1.0)
    ax1.set_title("a  Label accuracy (blue: adopted; dashed: target)", loc="left")
    names = []
    for y, (name, path, k_key, level_key) in enumerate((("Study 2 (label 0.57)", "results/engine_verdict_T3.json", "engine_reliability", "evaluation.ci_level"),
                                                        ("Study 3 (label 0.85)", "results/engine_v3_verdict_T3.json", "matching", "matching.t3_reuse.ci_level"))):
        v = inp.json(path)
        k = str(study[k_key]["verdict"]["k"])
        by = v["by_k"][k]
        level = study
        for part in level_key.split("."):
            level = level[part]
        for dy, part in ((-0.12, "candidate"), (0.12, "term_block")):
            ci = by["S3_diff_ci"][part]
            ax2.plot([ci["lower"], ci["upper"]], [y + dy, y + dy], color=COLORS["engine"] if part == "term_block" else COLORS["earlier"], lw=3)
        ax2.scatter([by["engine"]["precision"] - by["baseline_precision"]], [y], color="black", s=12, zorder=3)
        ax2.text(by["S3_diff_ci"]["term_block"]["lower"], y + 0.3, f"term-block lower {by['S3_diff_ci']['term_block']['lower']:.3f} (CI {level:.1%})",
                 fontsize=6, color=COLORS["neutral"])
        names.append(name)
    ax2.axvline(0, color=COLORS["fail"], ls="--", lw=1)
    ax2.set_yticks(range(len(names)), names), ax2.set_ylim(-0.6, 1.8), ax2.set_xlabel("precision gain over nPMI at top k (S3)")
    ax2.set_title("b  S3 interval (grey: candidate; blue: term block)", loc="left")
    fig.tight_layout()
    return save(fig, out, "fig5_measurement")


FIELD_NAMES = {"concept_frame.components": "frame: component roles", "concept_frame.target_task": "frame: target task",
               "concept_frame.combination": "frame: combination", "concept_frame.limitation": "frame: limitation",
               "concept_frame.insufficient_explanation": "frame: why insufficient", "summary": "summary",
               "building_blocks.open_source": "open-source use", "target_user": "target user", "jtbd": "job to be done",
               "value_prop": "value proposition"}


def fig_judging(inp: Inputs, study: dict, out: Path) -> list[str]:
    run = study["faithfulness"]["run_dir"]
    by_field = {}
    for path in (f"{run}/results/rejudge.json", f"{run}/results/rejudge_2.json"):
        for name, fields in inp.json(path)["by_study"].items():
            for field, v in fields.items():
                by_field.setdefault(field, {})[name] = v
    full = inp.json(f"{run}/results/rejudge_full.json")["by_study"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.8), gridspec_kw={"width_ratios": [1.4, 1]})
    y, ticks, labels = 0, [], []
    for field in [f for f in FIELD_NAMES if f in by_field]:
        for name in ("represcription", "faithfulness"):
            v = by_field[field].get(name)
            if not v:
                continue
            e, n = v["earlier_supported"] / v["n"], v["now_supported"] / v["n"]
            ax1.plot([e, n], [y, y], color=COLORS["earlier"], lw=1.5)
            ax1.scatter([e], [y], color=COLORS["earlier"], s=16, zorder=3)
            ax1.scatter([n], [y], facecolor=COLORS["now"] if e != n else "white", edgecolor=COLORS["now"], s=16, zorder=4, lw=1.2)
            ticks.append(y), labels.append(f"{FIELD_NAMES[field]}, {STUDY_NAMES[name]} (n={v['n']})")
            y += 1
        y += 0.6
    ax1.set_yticks(ticks, labels, fontsize=6), ax1.invert_yaxis()
    ax1.set_xlim(-0.03, 1.03), ax1.set_xlabel("share supported (grey: earlier; blue: blind re-judging; ring: unchanged)")
    ax1.set_title("a  Same sentences, same judge, different time", loc="left")
    for x, key in enumerate(("represcription", "faithfulness")):
        q = inp.json(f"{study[key]['run_dir']}/results/q1.json")
        now = full[key]["overall"]["now"]
        for dx, rate, ci, color in ((-0.12, q["rate"], q["wilson_95"], COLORS["earlier"]), (0.12, now["rate"], now["wilson_95"], COLORS["now"])):
            ax2.plot([x + dx, x + dx], ci, color=color, lw=3)
            ax2.scatter([x + dx], [rate], color="black", s=12, zorder=3)
            ax2.text(x + dx + 0.04, rate, f"{rate:.2f}", fontsize=6, va="center")
    ax2.set_xticks([0, 1], [STUDY_NAMES["represcription"] + "\n(prompt v3)", STUDY_NAMES["faithfulness"] + "\n(prompt v4)"])
    ax2.set_ylim(0.2, 0.9), ax2.set_ylabel("Q1 (Wilson 95%)"), ax2.set_xlim(-0.5, 1.6)
    ax2.set_title("b  Overall Q1: separate (grey) vs mixed blind (blue)", loc="left")
    fig.tight_layout()
    return save(fig, out, "fig6_judging")


def segment_label(ax, x: float, bottom: float, value: int, inside_min: int = 7) -> None:
    """Count inside a stacked segment, or beside the bar when the segment is too thin for text."""
    if not value:
        return
    if value >= inside_min:
        ax.text(x, bottom + value / 2, str(value), ha="center", va="center", fontsize=6)
    else:
        ax.annotate(str(value), (x + 0.3, bottom + value / 2), (x + 0.42, bottom + value / 2), fontsize=6, va="center",
                    arrowprops={"arrowstyle": "-", "lw": 0.5, "color": COLORS["neutral"]})


def fig_outcomes(inp: Inputs, study: dict, out: Path) -> list[str]:
    gens = [("Study 1", "results/briefing_generation.json"),
            ("Study 4", f"{study['represcription']['run_dir']}/results/briefing_generation.json"),
            ("Study 5", f"{study['faithfulness']['run_dir']}/results/briefing_generation.json")]
    parts = [("complete", "frame_complete", COLORS["engine"]), ("comparison only", "comparison_only", "#9ecae1"),
             ("term mismatch", "term_mismatch", COLORS["baseline"]), ("no combination evidence", "no_combination_evidence", "#fdd0a2"),
             ("other", "other", COLORS["random"])]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.8))
    for x, (name, path) in enumerate(gens):
        s = inp.json(path)["summary"]
        bottom = 0
        for label, key, color in parts:
            value = s[key] if key == "frame_complete" else s["insufficient_reasons"][key]
            ax1.bar(x, value, bottom=bottom, color=color, label=label if x == 0 else None, width=0.6)
            segment_label(ax1, x, bottom, value)
            bottom += value
    ax1.set_xticks(range(len(gens)), [g[0] for g in gens]), ax1.set_ylabel("runs")
    ax1.set_title("a  Concept-frame outcomes", loc="left")
    ax1.legend(frameon=False, fontsize=6, bbox_to_anchor=(0.5, -0.14), loc="upper center", ncol=3)
    cmp = inp.json(f"{study['represcription']['run_dir']}/results/comparison_first_study.json")["dvf"]
    for x, key in enumerate(("first", "fourth")):
        d = cmp[key]
        missing = d["riskiest_reason"].get("missing", 0)
        bars = [("V (missing)", missing, "#c6dbef"), ("V (lowest)", d["riskiest_axis"].get("V", 0) - missing, COLORS["engine"]),
                ("D", d["riskiest_axis"].get("D", 0), COLORS["baseline"]), ("F", d["riskiest_axis"].get("F", 0), COLORS["pass"])]
        bottom = 0
        for label, value, color in bars:
            ax2.bar(x, value, bottom=bottom, color=color, label=label if x == 0 else None, width=0.6)
            segment_label(ax2, x, bottom, value)
            bottom += value
    ax2.set_xticks([0, 1], ["Study 1", "Study 4"]), ax2.set_ylabel("prescriptions")
    ax2.set_title("b  Riskiest DVF axis", loc="left")
    ax2.legend(frameon=False, fontsize=6, bbox_to_anchor=(0.5, -0.14), loc="upper center", ncol=4)
    fig.tight_layout()
    return save(fig, out, "fig7_outcomes")


def commit_date(ref: str) -> datetime.date:
    text = subprocess.run(["git", "log", "-1", "--format=%cs", ref], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    return datetime.date.fromisoformat(text)


def freeze_commit(path: str) -> str:
    for line in (ROOT / path).read_text().splitlines():
        if line.startswith("고정 커밋:"):
            return line.split(":", 1)[1].strip()
    raise ValueError(f"no freeze commit in {path}")


def fig_timeline(inp: Inputs, study: dict, out: Path) -> list[str]:
    preregs = [("Study 1", "preregistration.md"), ("Study 2", study["engine_reliability"]["prereg_file"]),
               ("Study 3", study["matching"]["prereg_file"]), ("Study 4", study["represcription"]["prereg_file"]),
               ("Study 5", study["faithfulness"]["prereg_file"])]
    events = [(commit_date(freeze_commit(path)), f"{name} frozen", "freeze") for name, path in preregs]
    run = study["represcription"]["run_dir"]
    regs = [("Study 1", study["prospective"]["registration"]),
            ("Study 4", f"{run}/" + inp.json(f"{run}/config/study.json")["prospective"]["registration"])]
    for name, path in regs:
        inp.json(path)
        log = subprocess.run(["git", "log", "--diff-filter=A", "--format=%cs", "--", path], cwd=ROOT, capture_output=True, text=True, check=True).stdout.split()
        events.append((datetime.date.fromisoformat(log[-1]), f"{name} registration", "registration"))
    years = study["prospective"]["buffer_years"] + study["prospective"]["label_years"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.0), gridspec_kw={"width_ratios": [3, 1]})
    ordered = sorted(events, key=lambda e: (e[0], e[2] != "freeze"))
    side = {"freeze": 0, "registration": 0}
    for x, (date, label, kind) in enumerate(ordered):
        above = kind == "freeze"
        level = 0.22 + 0.3 * (side[kind] % 2)
        side[kind] += 1
        ax1.scatter([x], [0], color=COLORS["engine"] if above else COLORS["pass"], marker="o" if above else "s", s=26, zorder=3)
        ax1.plot([x, x], [0, level if above else -level], color=COLORS["neutral"], lw=0.6)
        ax1.text(x, level + 0.03 if above else -level - 0.03, f"{label}\n{date.isoformat()}", ha="center",
                 va="bottom" if above else "top", fontsize=6)
    ax1.axhline(0, color=COLORS["neutral"], lw=0.8)
    ax1.set_xlim(-0.6, len(ordered) - 0.4), ax1.set_ylim(-1.0, 1.0), ax1.axis("off")
    ax1.set_title("Preregistration freezes (above) and prospective registrations (below), in time order", fontsize=7)
    ax2.axis("off")
    ax2.text(0.5, 0.5, f"Planned verdicts\n{years} years after the reference time\n(buffer {study['prospective']['buffer_years']}"
             f" + label window {study['prospective']['label_years']})", ha="center", va="center", fontsize=7,
             bbox={"boxstyle": "round", "fc": "#eef3f7", "ec": COLORS["engine"]})
    return save(fig, out, "fig8_timeline")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="figures")
    args = parser.parse_args()
    out = ROOT / args.out
    study = load_study()
    inp = Inputs()
    written = []
    written += fig_pipeline(inp, study, out)
    written += fig_hypergraph(inp, out)
    written += fig_verdict(inp, study, out)
    written += fig_precision_k(inp, study, out)
    written += fig_measurement(inp, study, out)
    written += fig_judging(inp, study, out)
    written += fig_outcomes(inp, study, out)
    written += fig_timeline(inp, study, out)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    (out / "manifest.json").write_text(json.dumps({"git_commit": commit, "figures": written, "inputs_sha256": inp.used}, indent=1))
    print("Saved", len(written), "files to", out)


if __name__ == "__main__":
    main()
