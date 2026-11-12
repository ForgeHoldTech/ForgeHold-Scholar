import subprocess
import sys
from pathlib import Path

import uvicorn


PIPELINE = [
    (Path("data/catalog.json"), "fetch_papers.py"),
    (Path("output/corpus.json"), "ingest.py"),
    (Path("output/corpus.index.json"), "build_index.py"),
]

HOST = "0.0.0.0"
PORT = 8000


def prepare() -> None:
    for required, script in PIPELINE:
        if required.exists():
            print(f"Found {required}, skipping {script}", flush=True)
            continue

        print(f"Running {script}", flush=True)
        completed = subprocess.run([sys.executable, script], check=False)
        if completed.returncode != 0 or not required.exists():
            print(f"{script} did not produce {required}", flush=True)
            sys.exit(1)


def main() -> None:
    prepare()
    uvicorn.run("app:app", host=HOST, port=PORT)


if __name__ == "__main__":
    main()