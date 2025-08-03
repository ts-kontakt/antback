#!/usr/bin/env python3
"""
rolling_demo.py
A *visual* tour of RollingArray, NamedRollingArrays and
PerTickerMultiRollingArrays – watch the buffers shift!
"""

import numpy as np
from collections import defaultdict
from antback import RollingArray, NamedRollingArrays, PerTickerNamedRollingArrays



# ------------------------------------------------------------------
# 2. Demo helpers
# ------------------------------------------------------------------
def banner(title: str):
    print("\n" + "=" * 65)
    print(title)
    print("=" * 65)


def show(label, obj):
    print(f"{label:>15} → {obj}")


# ------------------------------------------------------------------
# 3. RollingArray – single buffer, size = 5
# ------------------------------------------------------------------
banner("RollingArray – size 5")
ra = RollingArray(5)
show("fresh", ra.values())

for v in [10, 20, 30, 40, 50, 60, 70]:
    ra.append(v)
    show(f"append({v})", ra.values())

# ------------------------------------------------------------------
# 4. NamedRollingArrays – many keys, one buffer each
# ------------------------------------------------------------------
banner("NamedRollingArrays – buffer size 3")
nra = NamedRollingArrays(3)

# Key A
for v in [1, 2, 3, 4]:
    nra.append("A", v)
    show(f'nra.append("A",{v})', nra.get("A"))

# Key B
for v in [9, 8]:
    nra.append("B", v)
    show(f'nra.append("B", {v})', nra.get("B"))

show("repr snapshot", nra)

# ------------------------------------------------------------------
# 5. PerTickerMultiRollingArrays – keys hold many named series
# ------------------------------------------------------------------
banner("PerTickerMultiRollingArrays – size 4")
per_ticker = PerTickerNamedRollingArrays(4)

# AAPL close
ticks = [100, 101, 102, 103, 104, 105]
for px in ticks:
    per_ticker.append("AAPL", "close", px)
    show(f'per_ticker.append("AAPL", "close", {px})', per_ticker.get("AAPL", "close"))

# AAPL volume (same loop, different series)
vols = [1_000, 2_000, 3_000, 4_000, 5_000, 6_000]
for vol in vols:
    per_ticker.append("AAPL", "volume", vol)
    show(f'per_ticker.append("AAPL", "volume", {vol})', per_ticker.get("AAPL", "volume"))

# TSLA open
per_ticker.append("TSLA", "open", 200)
per_ticker.append("TSLA", "open", 205)
show('per_ticker.append("TSLA", "open", 200)', per_ticker.get("TSLA", "open"))

show("full repr snapshot", per_ticker)

# ------------------------------------------------------------------
banner("Demo finished!")