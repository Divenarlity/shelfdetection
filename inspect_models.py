import argparse
import json
from pathlib import Path
from src.model_inspector import inspect_directory

parser = argparse.ArgumentParser()
parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
parser.add_argument("--output", type=Path, default=Path("outputs/model_report.json"))
args = parser.parse_args()
result = inspect_directory(args.root, args.output)
print(json.dumps(result, ensure_ascii=False, indent=2, default=str))

