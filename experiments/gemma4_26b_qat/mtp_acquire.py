from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import urllib.request

from followup_probes import headroom

REVISION = '7b92b5b28818151e8669af2e45e88d6086f490dd'
DRAFTER_NAME = 'mtp-gemma-4-26B-A4B-it-Q4_0.gguf'
DRAFTER_BYTES = 251939328
DRAFTER_SHA256 = '7272d97595f0d4c74bd7b623492b7dbdaafd8b7c72f329a8270ba4eca68f768a'
TARGET_NAME = 'gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf'
TARGET_BYTES = 14249047104
TARGET_SHA256 = 'a7c5bc715f5ff8e99a3e8901ce7d2b42b402c669bf24f7c5250747633d0f5891'
SERVER_SHA256 = '7b36a3bc3a24e552d0fcc18441893bc198abaeb0fa77d653241d762077c995c6'
URL = f'https://huggingface.co/unsloth/gemma-4-26B-A4B-it-qat-GGUF/resolve/{REVISION}/MTP/{DRAFTER_NAME}?download=true'


def identity(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return {'filename': path.name, 'bytes': path.stat().st_size, 'sha256': digest.hexdigest()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--adapter', required=True)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent / 'results/mtp/prerequisites.json')
    parser.add_argument('--baseline', default='c451f1b04a940f8d3e297ca58d13eefc338baee8')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve prerequisite evidence; choose a fresh output path')
    before = headroom(args.adapter)
    if before['model_runtime_processes']:
        raise RuntimeError('Model runtime conflict; no acquisition or launch started')
    drafter = args.root / 'models' / DRAFTER_NAME
    acquired = not drafter.exists()
    if acquired:
        partial = drafter.with_suffix('.gguf.download')
        with urllib.request.urlopen(URL, timeout=120) as response, partial.open('xb') as destination:
            while block := response.read(8 * 1024 * 1024):
                destination.write(block)
        observed = identity(partial)
        if observed['bytes'] != DRAFTER_BYTES or observed['sha256'] != DRAFTER_SHA256:
            raise RuntimeError('Pinned drafter identity mismatch; retained external download, no launch')
        partial.rename(drafter)
    draft_identity = identity(drafter)
    target_identity = identity(args.root / 'models' / TARGET_NAME)
    server = args.root / 'runtimes/b11537/llama-server.exe'
    server_identity = identity(server)
    assert (draft_identity['bytes'], draft_identity['sha256']) == (DRAFTER_BYTES, DRAFTER_SHA256), 'Drafter mismatch'
    assert (target_identity['bytes'], target_identity['sha256']) == (TARGET_BYTES, TARGET_SHA256), 'Target mismatch'
    assert server_identity['sha256'] == SERVER_SHA256, 'Pinned executable mismatch'
    version = subprocess.run([str(server), '--version'], capture_output=True, text=True, encoding='utf-8', errors='replace', check=True)
    result = {'baseline_commit': 'c451f1b04a940f8d3e297ca58d13eefc338baee8', 'observed_utc': datetime.now(timezone.utc).isoformat(), 'cohort': 'owner_authorized_mtp_followup', 'revision': REVISION, 'target': target_identity, 'drafter': draft_identity, 'drafter_acquired_now': acquired, 'drafter_url': URL, 'root_alias_identity': 'Owner/researcher verified root mtp-gemma-4-26B-A4B-it.gguf is identical to this pinned MTP/Q4_0 object.', 'runtime': server_identity, 'runtime_version': version.stdout.strip() + version.stderr.strip(), 'runtime_source_commit': '8e2d31e0eb658d0382d3106a61369dff81f0c080', 'python_version': sys.version, 'adapter': args.adapter, 'preflight': before, 'artifact_root': '<ARTIFACT_ROOT>', 'scope': 'Native-only new cohort. Prior c451f1b0 evidence unchanged; no runtime/backend/model-target/driver/product/settings change. Draft file size is not measured VRAM.'}
    result['baseline_commit'] = args.baseline
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
