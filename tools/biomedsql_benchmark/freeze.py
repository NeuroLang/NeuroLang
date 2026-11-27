"""Hash what the test run depends on, before it: ``python freeze.py`` writes FROZEN.json.

``run_agent.py`` refuses to start a test run when these hashes differ from FROZEN.json,
and ``python freeze.py check`` verifies them.
"""

import hashlib
import json
import re
import sys

from common import HERE, sha256

ENGINES = (
    HERE.parents[1] / "packages/neurolang_apps/neurolang_apps/utils/engines"
)
FILES = [
    "items.json",
    "scorer.py",
    "common.py",
    "run_agent.py",
    "judge_runs.py",
    "analyse.py",
    "guidance.txt",
    "bank.json",
    "llm.yaml",
    "oracle.py",
    "families.py",
]


def hashes():
    out = {
        name: sha256(HERE / name) for name in FILES if name != "run_agent.py"
    }
    out["engine/biomedsql/init.py"] = sha256(ENGINES / "biomedsql/init.py")
    out["engine/biomedsql/guidance.yaml"] = sha256(
        ENGINES / "biomedsql/guidance.yaml"
    )
    text = (ENGINES / "engines.yaml").read_text()
    block = re.search(
        r"\n  biomedsql:\n.*?(?=\n  # Example template usage)", text, re.S
    ).group(0)
    out["engine/engines.yaml: biomedsql"] = hashlib.sha256(
        block.encode()
    ).hexdigest()
    run = (HERE / "run_agent.py").read_text()
    # the runner may change in how it reports, not in what it asks: hash its settings
    settings = "".join(
        re.findall(
            r"^(?:PRICE|CAP_USD|MAX_STEPS|QUERY_TIMEOUT)\b.*$", run, re.M
        )
    )
    out["run_agent.py: settings"] = hashlib.sha256(
        settings.encode()
    ).hexdigest()
    return out


def check():
    frozen = json.load(open(HERE / "FROZEN.json"))
    now = hashes()
    return [k for k in frozen if frozen[k] != now.get(k)]


if __name__ == "__main__":
    if sys.argv[1:] == ["check"]:
        changed = check()
        print("changed since the freeze:", changed or "nothing")
        sys.exit(1 if changed else 0)
    (HERE / "FROZEN.json").write_text(json.dumps(hashes(), indent=1))
    print(json.dumps(hashes(), indent=1))
