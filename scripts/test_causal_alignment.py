"""Regression test for the rolling-24h blend's view alignment.

Terminology: the design is the **rolling-24h** blend; the variant it replaces is
the **calendar-day** blend. The module/function names keep a ``causal`` suffix
because output paths already carry it, and because "causal" is the precise
signal-processing term — but only once the reference point is stated, which is
why prose prefers "rolling 24h". Here causality is **with respect to the
decision moment**: the calendar-day variant pulls in views closing up to 23h
after the target anchor's own cutoff. Against the EXECUTION moment (a further
~24h out) both variants are causal and neither has look-ahead; the calendar-day
one is merely fresher than its own design intends.

The blend in ``run_hour_designs.py::scores_ens_causal`` is the one place in the
hour-of-day work where a one-character slip silently produces a better-looking
backtest: writing ``hp < hour`` instead of ``hp <= hour``, or dropping the lag
entirely, hands every anchor a signal fresher than its own decision moment. That
does not raise, does not produce NaNs, and does not look wrong in any metric —
it just quietly inflates Sharpe. (The topk sweep showed the calendar-day-aligned
variant's edge was width luck; this test is what keeps that variant from
creeping back in by accident.)

Binance stamps a kline with its OPEN time, so the bar a view h' takes on day d
closes at ``d (h'+1):00``. The intended contract, for a target anchor h on
decision day d:

    view h' <= h  ->  day d       cutoff  d (h'+1):00
    view h' >  h  ->  day d - 1   cutoff  d-1 (h'+1):00

which lays the 24 cutoffs out as 24 consecutive hourly points ending exactly at
the target's own cutoff — a genuine trailing 24-hour window.

Checks, in increasing cost:

  A. timing algebra (no data): for every target anchor, the 24 cutoffs are
     distinct consecutive hours, the newest equals the target's own, none is
     later than it, and all land strictly before the execution timestamp;
  B. lag count: exactly ``23 - h`` views are pulled back a day;
  C. production vs oracle: ``scores_ens_causal`` matches an independent
     reimplementation of the contract above, elementwise;
  D. the h23 identity: at target h23 nothing is newer, so the rolling-24h and
     the calendar-day blends must agree bit for bit — and at h00 (23 views
     lagged) they must NOT.

``--explain`` additionally prints the alignment table the checks assert — the
contract as documentation, derived from clock arithmetic alone, so it stays
correct by construction rather than by being kept in sync by hand.

Run from orange-quant/::
    ../.venv/bin/python scripts/test_causal_alignment.py           # h00/h08/h23
    ../.venv/bin/python scripts/test_causal_alignment.py --all     # all 24
    ../.venv/bin/python scripts/test_causal_alignment.py --explain [--explain-target H]
Exits non-zero on the first failed assertion.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_hour_designs as R  # noqa: E402

HOURS = list(range(24))
ATOL = 1e-12


class Failure(AssertionError):
    pass


def check(cond: bool, msg: str) -> None:
    if not cond:
        raise Failure(msg)
    print(f"  ok  {msg}")


# --------------------------------------------------------------------------
# A/B — timing algebra, no data needed
# --------------------------------------------------------------------------
def cutoff(view_h: int, day: np.datetime64) -> np.datetime64:
    """A bar stamped ``day view_h:00`` is an open time; it closes an hour later."""
    return np.datetime64(day, "h") + np.timedelta64(view_h + 1, "h")


def _fmt(ts: np.datetime64) -> str:
    return str(np.datetime64(ts, "h")).replace("T", " ") + ":00"


def explain(target: int, day: np.datetime64) -> None:
    """Print the alignment the checks below assert — the table as documentation.

    Pure clock arithmetic, no data loaded: this is the contract itself, not a
    measurement of it.
    """
    one = np.timedelta64(1, "D")
    own = cutoff(target, day)
    nc_max = max(cutoff(hp, day) for hp in HOURS)
    exec_ts = cutoff(target, day + one)

    print(f"\n[explain] 滚动24h 视角对齐 — 目标锚点 h{target:02d}，决策日 {day}")
    print("  时间戳是开盘时间，故视角 h′ 在 `d h′:00` 的线收在 `d (h′+1):00`。\n")
    print(f"  {'视角':<5} {'取哪天':<12} {'滚动24h截止':<18} {'按日历日截止':<18}")
    print("  " + "─" * 60)

    order = sorted(HOURS, key=lambda hp: cutoff(hp, day if hp <= target else day - one))
    prev_day = None
    for hp in order:
        d = day if hp <= target else day - one
        if prev_day is not None and d != prev_day:
            print("  " + "─" * 18 + f" 日界：h{target:02d} 之后的视角回退一日 " + "─" * 18)
        prev_day = d
        c, nc = cutoff(hp, d), cutoff(hp, day)
        mark = ""
        if c == own:
            mark = "  ← 最新 = 目标自身"
        elif hp == order[0]:
            mark = "  ← 最旧"
        print(f"  h{hp:02d}   {str(d):<12} {_fmt(c):<18} {_fmt(nc):<18}{mark}")

    print(f"\n  滚动24h窗口   : {_fmt(own - np.timedelta64(23, 'h'))} ~ {_fmt(own)}"
          f"   （24 个连续整点，回溯 24 小时）")
    print(f"  目标自身截止  : {_fmt(own)}")
    print(f"  按日历日最新  : {_fmt(nc_max)}"
          f"   比目标自身晚 {(nc_max - own) / np.timedelta64(1, 'h'):.0f}h")
    print(f"  执行时刻(t+1) : {_fmt(exec_ts)}"
          f"   两版最新信息距执行 {(exec_ts - own) / np.timedelta64(1, 'h'):.0f}h / "
          f"{(exec_ts - nc_max) / np.timedelta64(1, 'h'):.0f}h（均 > 0，无前视）")


def test_timing(day: np.datetime64) -> None:
    print("\n[A/B] 时刻对齐与回退计数")
    one = np.timedelta64(1, "D")
    for h in HOURS:
        days = [day if hp <= h else day - one for hp in HOURS]
        cuts = sorted(cutoff(hp, d) for hp, d in zip(HOURS, days))
        own = cutoff(h, day)
        n_lag = sum(1 for hp in HOURS if hp > h)

        if len(set(cuts)) != 24:
            raise Failure(f"h{h:02d}: 24 个截止时刻不互异")
        gaps = {(cuts[i + 1] - cuts[i]) / np.timedelta64(1, "h") for i in range(23)}
        if gaps != {1.0}:
            raise Failure(f"h{h:02d}: 截止时刻不连续，间隔={sorted(gaps)}")
        if cuts[-1] != own:
            raise Failure(f"h{h:02d}: 最新视角 {cuts[-1]} != 自身截止 {own}")
        if cuts[0] != own - np.timedelta64(23, "h"):
            raise Failure(f"h{h:02d}: 窗口不是回溯 24 小时，最旧={cuts[0]}")
        if n_lag != 23 - h:
            raise Failure(f"h{h:02d}: 回退视角数 {n_lag} != {23 - h}")

        # execution is the close of the NEXT decision day's bar
        exec_ts = cutoff(h, day + one)
        if not cuts[-1] < exec_ts:
            raise Failure(f"h{h:02d}: 最新信息 {cuts[-1]} 未早于执行 {exec_ts}")

        # the calendar-day variant must be strictly fresher for every h < 23,
        # which is exactly the confound this design removes
        nc = max(cutoff(hp, day) for hp in HOURS)
        if h < 23 and not nc > own:
            raise Failure(f"h{h:02d}: 按日历日对齐未表现出新鲜度差（{nc} vs {own}）")
        if h == 23 and nc != own:
            raise Failure("h23: 按日历日对齐应与自身截止相同")

    check(True, "24 个目标锚点：截止时刻连续、最新者=自身、窗口=回溯24h")
    check(True, "24 个目标锚点：回退视角数 = 23 − h")
    check(True, "24 个目标锚点：最新信息严格早于执行时刻（无前视）")
    check(True, "按日历日对齐在 h00–h22 上确实更新鲜，在 h23 上相等")


# --------------------------------------------------------------------------
# C/D — production code vs an independent oracle
# --------------------------------------------------------------------------
def oracle(hour: int, zmaps: dict, n_stocks: int) -> np.ndarray:
    """The contract, rewritten from the docstring rather than from the source."""
    ds = R.ds_for(hour)
    t0, t1 = R.decision_range(ds)
    one = np.timedelta64(1, "D")
    out = np.full((t1 - t0 + 1, n_stocks), np.nan)
    for k, d in enumerate(ds.dates[t0 : t1 + 1]):
        d = np.datetime64(d, "D")
        rows = []
        for hp in HOURS:
            r = zmaps[hp].get(d if hp <= hour else d - one)
            if r is not None:
                rows.append(r)
        if rows:
            stack = np.array(rows)
            with np.errstate(invalid="ignore"):
                out[k] = np.nanmean(stack, axis=0)
    return out


def test_blend(targets: list[int]) -> None:
    print("\n[C/D] 生产实现 vs 独立复算")
    score_fn = R.base_scorer("pooled-ens-causal")
    zmaps = R.build_zmaps(score_fn)
    blend = R.build_ens_blend(score_fn)

    for h in targets:
        n = R.ds_for(h).n_stocks
        prod = R.scores_ens_causal(h, zmaps)
        ref = oracle(h, zmaps, n)
        both = np.isfinite(prod) & np.isfinite(ref)
        if np.isfinite(prod).sum() != np.isfinite(ref).sum():
            raise Failure(f"h{h:02d}: 生产与复算的有效元素数不同")
        gap = float(np.max(np.abs(prod[both] - ref[both])))
        if gap > ATOL:
            raise Failure(f"h{h:02d}: 生产与复算最大差 {gap:.3e} > {ATOL:.0e}")
        print(f"  ok  h{h:02d} 生产实现 == 独立复算 (max|Δ| = {gap:.1e})")

    # the physically meaningful invariant: at h23 nothing is newer
    for h, must_match in ((23, True), (0, False)):
        if h not in targets:
            continue
        prod = R.scores_ens_causal(h, zmaps)
        cal = R.scores_ens(h, blend)
        both = np.isfinite(prod) & np.isfinite(cal)
        gap = float(np.max(np.abs(prod[both] - cal[both])))
        if must_match and gap > ATOL:
            raise Failure(f"h23: 滚动24h应与按日历日对齐完全相同，实测差 {gap:.3e}")
        if not must_match and gap <= ATOL:
            raise Failure(f"h{h:02d}: 滚动24h与按日历日对齐不应相同（回退未生效）")
        verdict = "完全相同" if must_match else f"不同 (max|Δ| = {gap:.3f})"
        print(f"  ok  h{h:02d} 滚动24h vs 按日历日对齐：{verdict}")


def main() -> int:
    ap = argparse.ArgumentParser(description="滚动24h视角对齐回归测试")
    ap.add_argument("--all", action="store_true", help="全部 24 个目标锚点（慢）")
    ap.add_argument("--date", default="2026-04-15", help="时刻算术用的样例日期")
    ap.add_argument("--explain", action="store_true",
                    help="顺带打印视角对齐表（纯时刻算术，不加载数据）")
    ap.add_argument("--explain-target", type=int, default=8, metavar="H",
                    help="--explain 展示哪个目标锚点（默认 8）")
    args = ap.parse_args()

    day = np.datetime64(args.date)
    targets = HOURS if args.all else [0, 8, 23]
    print(f"滚动24h视角对齐回归测试 — 目标锚点 {targets}")
    try:
        if args.explain:
            if not 0 <= args.explain_target <= 23:
                raise Failure(f"--explain-target 需在 0..23，收到 {args.explain_target}")
            explain(args.explain_target, day)
        test_timing(day)
        test_blend(targets)
    except Failure as e:
        print(f"\nFAIL: {e}")
        return 1
    print("\nPASS — 滚动 24h 视角对齐符合契约")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
