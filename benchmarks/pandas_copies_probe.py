"""Time and peak private memory of a query in a session forked from a built
engine, and its sorted result saved for comparison.

    python pandas_copies_probe.py ENGINE QUERY.dl OUT.pkl [REPEATS]
"""
import gc, os, pickle, sys, threading, time
from pathlib import Path
import nl_core
from neurolang_apps.utils.engine_registry import build_engine

engine, qfile, out = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
repeats = int(sys.argv[4]) if len(sys.argv) > 4 else 1
query = qfile.read_text()
nl = build_engine(engine, Path.home() / "neurolang_data", resolution=2)
gc.collect(); gc.freeze()


def private_mb(pid="self"):
    kb = 0
    with open(f"/proc/{pid}/smaps_rollup") as f:
        for line in f:
            if line.startswith(("Private_Clean", "Private_Dirty")):
                kb += int(line.split()[1])
    return kb / 1024


for r in range(repeats):
    rfd, wfd = os.pipe()
    pid = os.fork()
    if pid == 0:
        idle, peak, stop = private_mb(), [0], threading.Event()

        def sample():
            while not stop.is_set():
                peak[0] = max(peak[0], private_mb())
                time.sleep(0.05)

        th = threading.Thread(target=sample); th.start()
        t = time.time()
        with nl.scope:
            res = nl.execute_datalog_program(query)
        wall = time.time() - t
        stop.set(); th.join()
        df = res.as_pandas_dataframe()
        df = df.sort_values(list(df.columns)).reset_index(drop=True)
        if r == 0:
            pickle.dump(df, open(out, "wb"))
        os.write(wfd, f"{wall:.2f} {idle:.0f} {peak[0]:.0f} {len(df)}".encode())
        os._exit(0)
    os.waitpid(pid, 0)
    wall, idle, peak, rows = os.read(rfd, 100).decode().split()
    print(f"{engine} {qfile.name} run {r}: wall {wall} s, private idle {idle} MB, "
          f"peak {peak} MB (+{int(peak) - int(idle)}), rows {rows}", flush=True)
