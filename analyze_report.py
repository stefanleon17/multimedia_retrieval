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

LINE_RE = re.compile(r"^(?P<path>.+?\.obj) \u2014 FAILED: (?P<detail>.*)$")
# "3. fill_holes (boundary edges = 74)" ; message may contain parentheses/semicolons
STEP_RE = re.compile(r"(\d+\. \w+) \((.*?)\)(?=; \d+\. \w+ \(|; <internal> \(|$)")


def group_of(name: str) -> str:
    c = name[:1]
    if c == "D":
        return "(no VNs)"
    if c == "m":
        return "(with VNs)"
    return "other"


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
    return STEP_ORDER.index(s) if s in STEP_ORDER else 99


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", default="report.txt")
    ap.add_argument("--db", default="ShapeDatabase")
    ap.add_argument("--out", default="analysis_out")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(exist_ok=True)
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
    #per_step.to_csv(out / "per_step_metrics.csv", index=False)
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
    #per_class.to_csv(out / "per_step_per_class.csv", index=False)
    print("=== 2. Worst classes per step (top 8 by fail rate) ===")
    for step in STEP_ORDER:
        sub = per_class[per_class["step"] == step].sort_values(
            ["fail_rate", "failed"], ascending=False).head(8)
        if len(sub):
            print(f"\n-- {step}")
            print(sub.drop(columns="step").to_string(index=False))
    print()

    # class x step heatmap table of fail rates
    pivot = per_class.pivot(index="class", columns="step", values="fail_rate").fillna(0)
    pivot = pivot[[s for s in STEP_ORDER if s in pivot.columns]]
    #pivot.to_csv(out / "class_by_step_fail_rate.csv")

    # 3. Overall per-class and per-group summary (any failure)
    inv["any_fail"] = inv["path"].isin(set(fails["path"]))
    by_class = inv.groupby("class").agg(n=("path", "size"),
                                        failed=("any_fail", "sum"))
    by_class["fail_rate"] = by_class["failed"] / by_class["n"]
    by_class = by_class.sort_values("fail_rate", ascending=False)
    #by_class.to_csv(out / "per_class_summary.csv")
    print("=== 3a. Per-class: shapes with any failed step ===")
    print(by_class.to_string(), "\n")

    by_group = inv.groupby("group").agg(n=("path", "size"), failed=("any_fail", "sum"))
    by_group["fail_rate"] = by_group["failed"] / by_group["n"]
    #by_group.to_csv(out / "per_group_summary.csv")
    print("=== 3b. Objects with no VNs vs with VNs ===")
    print(by_group.to_string(), "\n")

    # class x group failure rate
    cg = inv.groupby(["class", "group"]).agg(n=("path", "size"),
                                             failed=("any_fail", "sum"))
    cg["fail_rate"] = cg["failed"] / cg["n"]
    #cg.to_csv(out / "class_by_group.csv")

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
    #sev.to_csv(out / "severity.csv", index=False)
    print(sev.to_string(index=False), "\n")

    # =======================================================================
    # 5. Crashes by exception type
    # =======================================================================
    cr = fails[fails["crash"]]
    if len(cr):
        ct = (cr.groupby(["step", "crash_type", "group"]).size()
              .unstack("group", fill_value=0))
        ct["total"] = ct.sum(axis=1)
        #ct.to_csv(out / "crash_types.csv")
        print("=== 5. Crashes by step / exception type ===")
        print(ct.to_string(), "\n")


    # 6. Co-occurrence and root-cause structure
    steps_per_shape = fails.groupby("path")["step"].apply(lambda s: tuple(sorted(s, key=step_index)))
    combos = steps_per_shape.value_counts().rename("shapes").to_frame()
    combos["share_of_failures"] = combos["shapes"] / combos["shapes"].sum()
    #combos.to_csv(out / "failure_combinations.csv")
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

    #print(f"CSV outputs written to: {out.resolve()}")


if __name__ == "__main__":
    main()
