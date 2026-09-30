"""Two steps, from exported traces to the five views.

    python -m tools.view load   <trace...> -o <ws>/commits/<sha>/store.db
    python -m tools.view render <store.db> -o <ws>/commits/<sha>/views [--capacity-gb N]
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

from . import FILES, write
from .load import build


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("load", help="exported traces into a store, one file per rank")
    p.add_argument("traces", nargs="+", type=Path)
    p.add_argument("-o", "--out", type=Path, required=True)

    p = sub.add_parser("render", help="the five views out of a store")
    p.add_argument("store", type=Path)
    p.add_argument("-o", "--out-dir", type=Path, required=True)
    p.add_argument("--capacity-gb", type=float, default=None,
                   help="torch.cuda.get_device_properties(d).total_memory / 2**30")

    a = ap.parse_args()
    if a.cmd == "load":
        print(build(a.traces, a.out))
        return
    con = sqlite3.connect(f"file:{a.store}?mode=ro", uri=True)
    write(con, a.out_dir, a.capacity_gb)
    print(a.out_dir)
    for name, what in FILES:
        print(f"  {name}  {what}")


if __name__ == "__main__":
    main()
