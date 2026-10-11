from __future__ import annotations

import argparse
import json
from pathlib import Path

from puripuly_heart.app.adapters.nemotron.artifacts import convert_artifact

parser = argparse.ArgumentParser()
parser.add_argument("source", type=Path)
parser.add_argument("destination", type=Path)
args = parser.parse_args()
print(json.dumps(convert_artifact(args.source, args.destination), indent=2))
