# route_distill/cli.py
import argparse
import sys

from .core import store
from .distiller import Distiller


def main(argv=None):
    p = argparse.ArgumentParser(prog="distill")
    p.add_argument("command", choices=["train", "report", "status"])
    p.add_argument("--store", default="routes.jsonl")
    p.add_argument("--model", default="route_model.pkl")
    a = p.parse_args(argv)

    d = Distiller(store=a.store, model_path=a.model)
    if a.command == "train":
        d.train()
        print(d.report())
    elif a.command == "report":
        print(d.report())
    elif a.command == "status":
        print(f"{store.count(a.store)} logged decisions; "
              f"threshold={d.threshold}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
