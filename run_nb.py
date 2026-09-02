"""Execute a notebook's code cells in order, headless. Scratch helper, not part of the library."""
import json
import re
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, ".")
MAGIC = re.compile(r"^\s*[%!]")

path = sys.argv[1]
nb = json.load(open(path))
ns = {"__name__": "__main__"}
t0 = time.time()
n = 0
for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] != "code":
        continue
    code = "\n".join(l for l in "".join(cell["source"]).split("\n") if not MAGIC.match(l))
    if not code.strip():
        continue
    try:
        exec(compile(code, f"{path} cell {i}", "exec"), ns)
    except Exception as exc:
        print(f"\nFAILED at cell {i}: {type(exc).__name__}: {exc}")
        raise
    plt.close("all")
    n += 1
print(f"\n{path}: {n} code cells, {time.time() - t0:.1f}s, no error")
