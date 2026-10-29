# analyze_report.py
#
# Quantifies how well each preprocessing step worked, from report.txt.
#
# Run from the project root:
#     python analyze_report.py
#     python analyze_report.py --report report.txt --db ShapeDatabase --out analysis_out


import argparse
import re
from pathlib import Path

import pandas as pd

STEP_ORDER = [
    "0. load", "1. clean", "2. orient", "3. fill_holes", "4. align",
    "5. flip", "6. scale", "7. ensure_outward", "8. simplify", "9. write",
]
ALL_STEPS = list(STEP_ORDER)
LINE_RE = re.compile(r"^(?P<path>.+?\.obj) \u2014 FAILED: (?P<detail>.*)$")
# message may contain parentheses/semicolons
STEP_RE = re.compile(r"(\d+\. \w+) \((.*?)\)(?=; \d+\. \w+ \(|; <internal> \(|$)")
SEVERITY_COLS = ["nonmanifold", "boundary", "verts", "bbox"]


def group_of(name: str) -> str:
    c = name[:1]
    if c == "D":
        return "(no VNs)"
    if c == "m":
        return "(with VNs)"
    return "other"


def short(step: str) -> str:
    """'3. fill_holes' -> 'fill_holes'"""
    return step.split(". ", 1)[-1]


def parse_report(path: Path) -> pd.DataFrame:
    """One row per (file, failed step). Last entry per file wins."""
    last = {}
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = LINE_RE.match(raw.strip())
        if not m:
            continue
        p = m.group("path").replace("\\", "/")
        last[p] = m.group("detail")

    rows = []
    for p, detail in last.items():
        parts = p.split("/")
        cls = parts[-2] if len(parts) >= 2 else "?"
        name = Path(parts[-1]).stem
        for step, msg in STEP_RE.findall(detail):
            rows.append({
                "path": p, "class": cls, "name": name, "group": group_of(name),
                "step": step, "msg": msg,
                "crash": msg.startswith("CRASH") or msg.startswith("check crashed"),
            })
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # numeric severity extracted from the message
    def num(pattern):
        return df["msg"].str.extract(pattern, expand=False).astype(float)

    df["nonmanifold"] = num(r"non-manifold edges = (\d+)")
    df["boundary"] = num(r"boundary edges = (\d+)")
    df["verts"] = num(r"^v=(\d+) \(target")
    df["bbox"] = num(r"max bbox dim = ([\d.]+)")
    df["crash_type"] = df["msg"].str.extract(r"CRASH: (\w+)", expand=False)
    return df


def inventory(db: Path) -> pd.DataFrame:
    rows = []
    for f in sorted(db.rglob("*.obj")):
        rel = f.relative_to(db.parent).as_posix()
        rows.append({"path": rel, "class": f.parent.name, "name": f.stem,
                     "group": group_of(f.stem)})
    return pd.DataFrame(rows)


def step_index(s):
    return ALL_STEPS.index(s) if s in ALL_STEPS else 99


# ===========================================================================
# Master table: everything in ONE row per shape (wide), plus a long table
# ===========================================================================
def build_master_table(inv: pd.DataFrame, fails: pd.DataFrame) -> pd.DataFrame:
    """One row per shape in the DB (passing shapes included).

    Columns:
      path, class, name, group
      any_fail, n_failed_steps, failed_steps ("1. clean|3. fill_holes")
      crash_step, crash_si, crash_type   (first crash; pipeline aborts there)
      fail_<step>                        bool per step
      <metric>@<step>                    severity values, e.g. nonmanifold@clean,
                                         boundary@fill_holes, verts@simplify
    """
    m = inv[["path", "class", "name", "group"]].drop_duplicates("path").copy()

    # failed steps per shape
    steps = (fails.groupby("path")["step"]
             .apply(lambda s: sorted(set(s), key=step_index)))
    m["n_failed_steps"] = m["path"].map(steps.map(len)).fillna(0).astype(int)
    m["any_fail"] = m["n_failed_steps"] > 0
    m["failed_steps"] = m["path"].map(steps.map("|".join)).fillna("")

    # first crash per shape
    cr = fails[fails["crash"]].assign(si=lambda d: d["step"].map(step_index))
    first = cr.sort_values("si").drop_duplicates("path").set_index("path")
    m["crash_si"] = m["path"].map(first["si"])  # NaN = no crash
    m["crash_step"] = m["path"].map(first["step"])
    m["crash_type"] = m["path"].map(first["crash_type"])

    # one boolean column per step
    failed_pairs = set(zip(fails["path"], fails["step"]))
    for step in STEP_ORDER:
        m[f"fail_{short(step)}"] = [(p, step) in failed_pairs for p in m["path"]]

    # severity metrics, one column per (metric, step)
    for col in SEVERITY_COLS:
        sub = fails[fails[col].notna()]
        if sub.empty:
            continue
        pv = sub.pivot_table(index="path", columns="step", values=col, aggfunc="max")
        for step in pv.columns:
            m[f"{col}@{short(step)}"] = m["path"].map(pv[step])
    return m


def build_long_table(inv: pd.DataFrame, fails: pd.DataFrame) -> pd.DataFrame:
    """One row per (shape, failed step) with the raw message and severity."""
    return fails.sort_values(["path", "step"],
                             key=lambda s: s.map(step_index) if s.name == "step" else s)


def export_csvs(master: pd.DataFrame, long: pd.DataFrame, out: Path):
    out.mkdir(exist_ok=True, parents=True)
    master.to_csv(out / "master.csv", index=False)
    long.to_csv(out / "failures_long.csv", index=False)
    print(f"Wrote {out / 'master.csv'} ({len(master)} rows, {master.shape[1]} cols)")
    print(f"Wrote {out / 'failures_long.csv'} ({len(long)} rows)\n")


# ===========================================================================
# Analysis from the master table only (works on a re-loaded CSV too)
# ===========================================================================
def step_rates(master: pd.DataFrame, by=None) -> pd.DataFrame:
    """Per-step evaluated/failed/fail_rate, optionally split by a column
    ('group' or 'class'). A shape counts as evaluated if it didn't crash
    at an earlier step."""
    rows = []
    groups = [("ALL", master)] if by is None else list(master.groupby(by))
    for gname, g in groups:
        for step in STEP_ORDER:
            si = step_index(step)
            col = f"fail_{short(step)}"
            if col not in g.columns:
                continue
            ev = int((g["crash_si"].isna() | (g["crash_si"] >= si)).sum())
            nf = int(g[col].sum())
            row = {"step": step, "evaluated": ev, "failed": nf,
                   "fail_rate": nf / ev if ev else float("nan")}
            if by is not None:
                row[by] = gname
            rows.append(row)
    return pd.DataFrame(rows)


def analyze_master(master: pd.DataFrame):
    """Console summary computed purely from the master table."""
    pd.set_option("display.width", 200, "display.max_rows", 200,
                  "display.max_columns", 30, "display.float_format", "{:.3f}".format)
    print("=== MASTER: per-step fail rate (ALL) ===")
    print(step_rates(master).to_string(index=False), "\n")
    print("=== MASTER: per-step fail rate by group ===")
    print(step_rates(master, "group").to_string(index=False), "\n")

    sev_cols = [c for c in master.columns if "@" in c]
    if sev_cols:
        print("=== MASTER: severity summary ===")
        print(master[sev_cols].describe(percentiles=[.5]).T.to_string(), "\n")


def make_plots(master: pd.DataFrame, out: Path, top_classes: int = 25):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    out.mkdir(exist_ok=True, parents=True)
    labels = [short(s) for s in STEP_ORDER]

    # 1. fail rate per step, by group
    sr = step_rates(master, "group")
    piv = sr.pivot(index="step", columns="group", values="fail_rate").reindex(STEP_ORDER)
    ax = piv.plot(kind="bar", figsize=(10, 4))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("fail rate (of shapes reaching step)")
    ax.set_title("Per-step failure rate by group")
    plt.tight_layout();
    plt.savefig(out / "fail_rate_per_step.png", dpi=150);
    plt.close()

    # 2. class x step heatmap (only classes with any failure, top N)
    cs = step_rates(master, "class")
    hm = cs.pivot(index="class", columns="step", values="fail_rate").reindex(columns=STEP_ORDER)
    hm = hm.loc[hm.fillna(0).sum(axis=1).sort_values(ascending=False).index]
    hm = hm[hm.fillna(0).sum(axis=1) > 0].head(top_classes)
    if len(hm):
        fig, ax = plt.subplots(figsize=(9, 0.3 * len(hm) + 2))
        im = ax.imshow(hm.fillna(0).values, aspect="auto", cmap="Reds", vmin=0, vmax=1)
        ax.set_xticks(range(len(hm.columns)));
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticks(range(len(hm)));
        ax.set_yticklabels(hm.index)
        fig.colorbar(im, label="fail rate")
        ax.set_title("Failure rate by class and step")
        plt.tight_layout();
        plt.savefig(out / "class_by_step_heatmap.png", dpi=150);
        plt.close()

    # 3. shapes with any failure per class
    bc = (master.groupby("class")["any_fail"].agg(["size", "sum"])
          .assign(rate=lambda d: d["sum"] / d["size"])
          .sort_values("rate", ascending=False).head(top_classes))
    ax = bc["rate"][::-1].plot(kind="barh", figsize=(7, 0.3 * len(bc) + 2))
    ax.set_xlabel("share of shapes with >=1 failed step")
    ax.set_title(f"Worst {len(bc)} classes")
    plt.tight_layout();
    plt.savefig(out / "worst_classes.png", dpi=150);
    plt.close()

    # 4. number of failed steps per shape
    ax = master["n_failed_steps"].value_counts().sort_index().plot(kind="bar", figsize=(6, 4))
    ax.set_xlabel("failed steps per shape");
    ax.set_ylabel("shapes")
    plt.tight_layout();
    plt.savefig(out / "failed_steps_hist.png", dpi=150);
    plt.close()

    # 5. severity distributions (log x since counts are heavy-tailed)
    sev_cols = [c for c in master.columns if "@" in c and master[c].notna().any()]
    if sev_cols:
        n = len(sev_cols);
        cols = min(3, n);
        rows = int(np.ceil(n / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 3.2 * rows), squeeze=False)
        for ax, c in zip(axes.ravel(), sev_cols):
            v = master[c].dropna()
            v = v[v > 0] if (v > 0).any() else v
            bins = np.logspace(np.log10(v.min()), np.log10(v.max()), 30) if v.max() > v.min() else 10
            ax.hist(v, bins=bins)
            if v.max() > v.min():
                ax.set_xscale("log")
            ax.set_title(c, fontsize=9)
        for ax in axes.ravel()[n:]:
            ax.axis("off")
        plt.tight_layout();
        plt.savefig(out / "severity_hists.png", dpi=150);
        plt.close()

    # 6. crashes by step / type
    cr = master[master["crash_step"].notna()]
    if len(cr):
        ct = cr.groupby(["crash_step", "crash_type"]).size().unstack(fill_value=0)
        ct = ct.reindex([s for s in STEP_ORDER if s in ct.index])
        ax = ct.plot(kind="bar", stacked=True, figsize=(8, 4))
        ax.set_xticklabels([short(s) for s in ct.index], rotation=45, ha="right")
        ax.set_title("Crashes by step and exception type")
        plt.tight_layout();
        plt.savefig(out / "crashes.png", dpi=150);
        plt.close()

    print(f"Plots written to: {out.resolve()}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", default="report.txt")
    ap.add_argument("--db", default="ShapeDatabase")
    ap.add_argument("--out", default="analysis_out/preprocessing")
    ap.add_argument("--csv", action="store_true", help="write master.csv + failures_long.csv")
    ap.add_argument("--plots", action="store_true", help="write PNG plots")
    ap.add_argument("--from-csv", default=None,
                    help="skip parsing; load a master.csv and just analyse/plot it")
    ap.add_argument("--steps", nargs="+", default=None,
                    help="step numbers to include in the analysis")
    a = ap.parse_args()

    if a.steps:
        keep = {int(x) for x in a.steps}
        STEP_ORDER[:] = [s for s in ALL_STEPS if int(s.split(".")[0]) in keep]

    out = Path(a.out)
    out.mkdir(parents=True,                 exist_ok=True)

    if a.from_csv:
        master = pd.read_csv(a.from_csv)
        analyze_master(master)
        if a.plots:
            make_plots(master, out)
        return

    pd.set_option("display.width", 200, "display.max_rows", 200,
                  "display.max_columns", 30, "display.float_format", "{:.3f}".format)

    fails = parse_report(Path(a.report))
    inv = inventory(Path(a.db)) if Path(a.db).exists() else pd.DataFrame()
    if fails.empty:
        print("No failures parsed from report. Nothing to analyse.")
        return

    # If the db is missing, fall back to the failing shapes only (rates meaningless)
    if inv.empty:
        print("WARNING: --db not found, rates use failing shapes only as the universe.\n")
        inv = fails[["path", "class", "name", "group"]].drop_duplicates()

    N = len(inv)
    failed_paths = fails["path"].nunique()
    print(f"Shapes in DB: {N} | with >=1 failed step: {failed_paths} "
          f"({failed_paths / N:.1%}) | fully passed: {N - failed_paths} "
          f"({(N - failed_paths) / N:.1%})\n")

    # --- master table (single source for CSV export and plots) -------------
    master = build_master_table(inv, fails)
    if a.csv:
        export_csvs(master, build_long_table(inv, fails), out)
    if a.plots:
        make_plots(master, out)


    # --- which step each shape crashed at (pipeline aborts there) ------------
    crash_at = (fails[fails["crash"]].assign(si=lambda d: d["step"].map(step_index))
                .groupby("path")["si"].min())
    inv = inv.copy()
    inv["crash_si"] = inv["path"].map(crash_at)  # NaN = no crash

    # =======================================================================
    # 1. Per-step metrics (overall + by group)
    # =======================================================================
    def evaluated(si, subset):
        """Shapes that reached step index si (no crash at an earlier step;
        a crash AT the step counts as reaching it)."""
        c = subset["crash_si"]
        return int(((c.isna()) | (c >= si)).sum())

    rows = []
    for step in STEP_ORDER:
        si = step_index(step)
        sf = fails[fails["step"] == step]
        for gname, sub_inv in [("ALL", inv)] + list(inv.groupby("group")):
            sub_f = sf if gname == "ALL" else sf[sf["group"] == gname]
            ev = evaluated(si, sub_inv)
            nfail = sub_f["path"].nunique()
            ncrash = sub_f[sub_f["crash"]]["path"].nunique()
            rows.append({"step": step, "group": gname, "evaluated": ev,
                         "failed": nfail, "crashed": ncrash,
                         "check_failed": nfail - ncrash,
                         "fail_rate": nfail / ev if ev else float("nan"),
                         "pass_rate": 1 - nfail / ev if ev else float("nan")})
    per_step = pd.DataFrame(rows)
    print("=== 1. Per-step failure metrics ===")
    print(per_step.to_string(index=False), "\n")

    # Per-step, per-class failure rate (denominator = shapes in class that reached step)
    rows = []
    for step in STEP_ORDER:
        si = step_index(step)
        sf = fails[fails["step"] == step]
        for cls, ci in inv.groupby("class"):
            ev = evaluated(si, ci)
            nfail = sf[sf["class"] == cls]["path"].nunique()
            if nfail:
                rows.append({"step": step, "class": cls, "n_class": len(ci),
                             "evaluated": ev, "failed": nfail,
                             "fail_rate": nfail / ev if ev else float("nan")})
    per_class = pd.DataFrame(rows)
    print("=== 2. Worst classes per step (top 8 by fail rate) ===")
    for step in STEP_ORDER:
        sub = per_class[per_class["step"] == step].sort_values(
            ["fail_rate", "failed"], ascending=False).head(8)
        if len(sub):
            print(f"\n-- {step}")
            print(sub.drop(columns="step").to_string(index=False))
    print()

    # 3. Overall per-class and per-group summary (any failure)
    inv["any_fail"] = inv["path"].isin(set(fails["path"]))
    by_class = inv.groupby("class").agg(n=("path", "size"),
                                        failed=("any_fail", "sum"))
    by_class["fail_rate"] = by_class["failed"] / by_class["n"]
    by_class = by_class.sort_values("fail_rate", ascending=False)
    print("=== 3a. Per-class: shapes with any failed step ===")
    print(by_class.to_string(), "\n")

    by_group = inv.groupby("group").agg(n=("path", "size"), failed=("any_fail", "sum"))
    by_group["fail_rate"] = by_group["failed"] / by_group["n"]
    print("=== 3b. Objects with no VNs vs with VNs ===")
    print(by_group.to_string(), "\n")

    # 4. Severity metrics for the failures
    print("=== 4. Severity (distribution among FAILING shapes) ===")
    sev_rows = []
    for step, col in [("1. clean", "nonmanifold"), ("3. fill_holes", "boundary"),
                      ("8. simplify", "boundary"), ("8. simplify", "verts")]:
        sub = fails[(fails["step"] == step) & fails[col].notna()]
        for gname, g in [("ALL", sub)] + list(sub.groupby("group")):
            if len(g):
                d = g[col].describe(percentiles=[.5])
                sev_rows.append({"step": step, "metric": col, "group": gname,
                                 "n": int(d["count"]), "mean": d["mean"],
                                 "median": d["50%"], "max": d["max"]})
    sev = pd.DataFrame(sev_rows)
    print(sev.to_string(index=False), "\n")

    # =======================================================================
    # 5. Crashes by exception type
    # =======================================================================
    cr = fails[fails["crash"]]
    if len(cr):
        ct = (cr.groupby(["step", "crash_type", "group"]).size()
              .unstack("group", fill_value=0))
        ct["total"] = ct.sum(axis=1)
        print("=== 5. Crashes by step / exception type ===")
        print(ct.to_string(), "\n")


    # 6. Co-occurrence and root-cause structure
    steps_per_shape = fails.groupby("path")["step"].apply(lambda s: tuple(sorted(s, key=step_index)))
    combos = steps_per_shape.value_counts().rename("shapes").to_frame()
    combos["share_of_failures"] = combos["shapes"] / combos["shapes"].sum()
    print("=== 6. Most common failure combinations ===")
    print(combos.head(15).to_string(), "\n")

    # 7. Simplify detail: vertex target hit vs boundary tolerance
    simp = fails[fails["step"] == "8. simplify"]
    if len(simp):
        v_miss = simp["verts"].notna().sum()
        print("=== 7. Simplify failures ===")
        print(f"shapes failing simplify: {simp['path'].nunique()}")
        print(f"  with message parsed (vertex/boundary check): {v_miss}")
        print(f"  crashed: {simp['crash'].sum()}\n")



if __name__ == "__main__":
    main()
