"""View 3: cumulative GPU time by kernel and by class.

Copies belong in this table. They used to be excluded, and since a copy counts as
GPU-busy it sat in no gap either, so a 4.7 GB device-to-host copy appeared in no
view at all. Three of the fixes that shipped from this engine were host-side data
movement, and it found none of them.
"""

from __future__ import annotations

import sqlite3

from .common import classify, fmt_bw, fmt_ms, fmt_size, table

NAME_WIDTH = 56


def _cls(cat: str, name: str) -> str:
    """Classify a transfer by its category, never by its name.

    "Memcpy HtoD" matches the element-wise name pattern on `copy` and would be
    filed as compute. Host transfers are split from device-to-device because they
    are different machines: host crosses PCIe at tens of GB/s and is where the
    findings are, device-to-device is HBM at thousands. Averaging them hid a
    10 GB/s pageable copy inside a 2900 GB/s number.
    """
    if cat in ("memcpy", "memset"):
        return "transfer H<->D" if ("HtoD" in name or "DtoH" in name) else "transfer D2D"
    return classify(name)


def render(con: sqlite3.Connection, top: int = 15, cutoff: float = 0.01) -> str:
    rows = con.execute(
        "SELECT cat, name, COUNT(*), SUM(dur_ns), SUM(COALESCE(nbytes,0)) FROM events "
        "WHERE cat IN ('kernel','comm','memcpy','memset') GROUP BY cat, name "
        "ORDER BY 4 DESC").fetchall()
    out = ["== 3. Kernel table =="]
    if not rows:
        return "\n".join(out + ["  (no kernels)"])
    total = sum(t for _, _, _, t, _ in rows)

    by_cat: dict[str, list[int]] = {}
    for cat, name, n, t, b in rows:
        c = by_cat.setdefault(_cls(cat, name), [0, 0, 0])
        c[0] += t
        c[1] += n
        c[2] += b
    out.append(table([[c, v[1], fmt_ms(v[0]), f"{v[0] / total:.0%}", fmt_size(v[2]),
                       fmt_bw(v[2], v[0])]
                      for c, v in sorted(by_cat.items(), key=lambda x: -x[1][0])],
                     ["class", "count", "time_ms", "share", "size", "GB/s"]))

    shown = [r for r in rows[:top] if r[3] >= cutoff * total]
    out.append(table([[n[:NAME_WIDTH], c, fmt_ms(t), f"{t / total:.0%}", _cls(cat, n),
                       fmt_size(b), fmt_bw(b, t)]
                      for cat, n, c, t, b in shown],
                     ["kernel", "count", "time_ms", "share", "class", "size", "GB/s"]))
    if len(rows) > len(shown):
        out.append(f"  +{len(rows) - len(shown)} more kernels below {cutoff:.0%}")

    pageable = _pageable_note(rows)
    if pageable:
        out.append(pageable)
    return "\n".join(out)


def _pageable_note(rows: list[tuple]) -> str:
    """Call out pageable host copies, compared only against pinned HOST copies.

    A pageable copy is synchronous: the driver stages it through a bounce buffer,
    so it cannot overlap with compute and runs at roughly half the bandwidth of a
    pinned one. Comparing it against all pinned traffic instead made it look 300x
    slower, because that bucket was mostly device-to-device at HBM speed.
    """
    host = [r for r in rows if r[0] == "memcpy" and ("HtoD" in r[1] or "DtoH" in r[1])]
    pageable = [r for r in host if "pageable" in r[1].lower()]
    if not pageable:
        return ""
    pinned = [r for r in host if "pageable" not in r[1].lower()]
    pt, pb = sum(r[3] for r in pageable), sum(r[4] for r in pageable)
    qt, qb = sum(r[3] for r in pinned), sum(r[4] for r in pinned)
    rest = (f"pinned host transfers are {fmt_ms(qt)}ms at {fmt_bw(qb, qt)} GB/s"
            if pinned else "there are no pinned host transfers to compare against")
    return (f"  PAGEABLE host transfers: {sum(r[2] for r in pageable)} copies, {fmt_ms(pt)}ms, "
            f"{fmt_size(pb)} at {fmt_bw(pb, pt)} GB/s. Pageable is synchronous, so it cannot "
            f"overlap with compute and runs at roughly half the bandwidth of a pinned copy; "
            f"{rest}")
