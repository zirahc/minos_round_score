#!/usr/bin/env python3
"""Download one Minos round from Hugging Face and score it with GATK.

The round id is the primary key of public.rounds, for example
2026-09-02T21:44:00+00:00. This script reads that row, downloads the reveal
files from the row's huggingface link, applies gatk_updates.json on top of
minos_subnet/configs/gatk.conf, runs GATK, and writes the v2 score.

This folder is the sibling of minos_subnet. The GATK template comes from that
checkout. Before the round download, the chromosome reference (FASTA, index,
dictionary, and RTG SDF) is checked under minos_subnet/datasets/reference/
and any missing files are downloaded. The gatk.conf file on disk is not modified.

  python score_round.py "2026-09-02T21:44:00+00:00"
  python score_round.py "2026-09-02T21:44:00+00:00" --updates gatk_updates.json

gatk_updates.json is one config, a list of configs, or a sweep:

  {"min_base_quality_score": 20, "pcr_indel_model": "NONE"}

  {"name": "higher-base-quality", "updates": {"min_base_quality_score": 20}}

  {"name": "stand-call-conf", "sweep": {"param": "standard_min_confidence_threshold_for_calling", "start": 20, "stop": 40, "step": 5}}

Set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY (or SUPABASE_KEY) in .env.
HF_TOKEN is only needed when the model repo is private.
Quote the round id in PowerShell so the + sign is kept.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent
DOWNLOADS = REPO_ROOT / "downloads"
RESULTS = REPO_ROOT / "results"
ENV_PATH = REPO_ROOT / ".env"
COLLECTOR_ENV = REPO_ROOT.parent / "minos_round_collector" / ".env"

_subnet_raw = (os.getenv("MINOS_SUBNET") or "").strip()
SUBNET_ROOT = (
    Path(_subnet_raw).expanduser().resolve()
    if _subnet_raw
    else (REPO_ROOT.parent / "minos_subnet").resolve()
)
if not SUBNET_ROOT.is_dir():
    print(f"ERROR: minos_subnet not found at {SUBNET_ROOT}", flush=True)
    print("This folder must sit next to minos_subnet, or set MINOS_SUBNET.", flush=True)
    raise SystemExit(2)

sys.path.insert(0, str(SUBNET_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from score_gatk_folders import (  # noqa: E402
    _jsonable,
    _load_gatk_config,
    _print_summary,
    score_folder,
)
from templates.tool_params import (  # noqa: E402
    GATK_QUALITY_PARAMS,
    REGION_PATTERN,
    validate_and_build_flags,
)

GATK_CONF = SUBNET_ROOT / "configs" / "gatk.conf"
REFERENCE_DIR = SUBNET_ROOT / "datasets" / "reference"
DEFAULT_REF_BASE = "https://api.theminos.ai/reference"
# api.theminos.ai/reference rejects the default Python urllib user agent.
REF_USER_AGENT = "minos-installer/0.1 (+https://github.com/minos-protocol/minos_subnet)"
FASTA_EXTS = ("fa", "fa.fai", "dict")
SDF_FILES = (
    "done",
    "mainIndex",
    "nameIndex0",
    "namedata0",
    "namepointer0",
    "progress",
    "seqdata0",
    "seqpointer0",
    "sequenceIndex0",
    "summary.txt",
)
ROUND_COLUMNS = (
    "round_id,region,huggingface,status,"
    "rank_1_tool_name,rank_1_combined_final,rank_1_snp_final,rank_1_indel_final"
)
PROVENANCE_KEYS = frozenset({
    "name",
    "updates",
    "sweep",
    "search_category",
    "hypothesis",
    "suggested_by",
    "study_name",
    "parent_config_id",
    "optuna_trial_number",
    "status",
})


def main(argv: Optional[List[str]] = None) -> int:
    _load_env()
    args = _parse_args(argv)
    if not GATK_CONF.is_file():
        print(f"ERROR: GATK config not found: {GATK_CONF}", flush=True)
        return 2

    updates_path = Path(args.updates)
    if not updates_path.is_absolute():
        updates_path = REPO_ROOT / updates_path
    try:
        experiments = load_gatk_updates(updates_path)
        base_config = _load_gatk_config(str(GATK_CONF))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", flush=True)
        return 2

    prepared, rejected = _prepare_configs(base_config["gatk_options"], experiments)
    if rejected:
        return 2
    if not prepared:
        print(f"ERROR: {updates_path} produced no GATK configs", flush=True)
        return 2

    try:
        row = fetch_round(args.round_id)
    except (urllib.error.URLError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", flush=True)
        return 2
    if row is None:
        print(f"ERROR: no rounds row for round_id {args.round_id!r}", flush=True)
        return 2

    link = (row.get("huggingface") or "").strip()
    if not link:
        print(f"ERROR: rounds row {args.round_id!r} has no huggingface link", flush=True)
        return 2
    region = (row.get("region") or "").strip()
    if args.region:
        region = args.region.strip()
    if not region or not REGION_PATTERN.match(region):
        print(
            f"ERROR: round has no usable region ({region!r}). Pass --region.",
            flush=True,
        )
        return 2

    try:
        remote = parse_huggingface_link(link)
    except ValueError as exc:
        print(f"ERROR: {exc}", flush=True)
        return 2

    print("=" * 72, flush=True)
    print("  SCORE ONE ROUND", flush=True)
    print("=" * 72, flush=True)
    print(f"  Round:      {row.get('round_id')}", flush=True)
    print(f"  Region:     {region}", flush=True)
    print(f"  Status:     {row.get('status')}", flush=True)
    print(f"  HuggingFace {link}", flush=True)
    chrom = region.split(":", 1)[0]
    print(f"  GATK base:  {GATK_CONF}", flush=True)
    print(f"  Updates:    {updates_path.name}  ({len(prepared)} config(s))", flush=True)
    print(f"  Reference:  {REFERENCE_DIR / chrom}", flush=True)
    _print_rank1(row)
    print(flush=True)

    if not ensure_reference(chrom):
        return 2

    try:
        folder = download_round(remote, force=args.force)
    except Exception as exc:  # noqa: BLE001 — hub and network errors, reported below
        print(f"ERROR: download failed: {exc}", flush=True)
        return 2
    (folder / "region.txt").write_text(region + "\n", encoding="utf-8")
    print(f"  Folder:     {folder}", flush=True)

    out_path = Path(args.json_out) if args.json_out else RESULTS / f"{remote['folder']}.json"
    if not out_path.is_absolute():
        out_path = REPO_ROOT / out_path

    results: List[Dict[str, Any]] = []
    for index, (name, updates, options) in enumerate(prepared, 1):
        print(f"\n{'#' * 72}", flush=True)
        print(f"  [{index}/{len(prepared)}] {name}", flush=True)
        print(f"{'#' * 72}", flush=True)
        _print_updates(base_config["gatk_options"], updates)
        record = score_folder(
            folder=folder,
            tool_config={"gatk_options": options},
            region_override=region,
            region_padding=args.region_padding,
        )
        record["round_id"] = row.get("round_id")
        record["config_name"] = name
        record["gatk_updates"] = updates
        record["gatk_options"] = options
        results.append(record)
        _write_output(out_path, row, link, folder, region, results)

    _print_summary(results)
    _print_rank1(row)
    print(f"  Wrote {out_path}", flush=True)
    failed = sum(1 for item in results if not item.get("ok"))
    return 1 if failed else 0


def _parse_args(argv: Optional[List[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download one Supabase round from Hugging Face, apply gatk_updates.json, and score GATK.",
    )
    parser.add_argument(
        "round_id",
        help="rounds.round_id, e.g. 2026-09-02T21:44:00+00:00",
    )
    parser.add_argument(
        "--updates",
        default="gatk_updates.json",
        help="GATK update JSON (default: gatk_updates.json next to this script)",
    )
    parser.add_argument(
        "--region",
        default=None,
        help="Override the region stored on the rounds row.",
    )
    parser.add_argument(
        "--region-padding",
        type=int,
        default=100_000,
        help="Used only if the region has to be inferred. The rounds row region is preferred.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Download the round files again even when they are already on disk.",
    )
    parser.add_argument(
        "--json-out",
        default=None,
        help="Where to write the score JSON (default: results/<round folder>.json).",
    )
    return parser.parse_args(argv)


def _load_env() -> None:
    """Load .env files. Existing environment variables win. Never print values."""
    _load_env_file(ENV_PATH)
    _load_env_file(COLLECTOR_ENV)


def _load_env_file(path: Path) -> None:
    if not path.is_file():
        return
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def fetch_round(round_id: str) -> Optional[Dict[str, Any]]:
    """One public.rounds row, or None when the id is not in the table."""
    base = os.environ.get("SUPABASE_URL", "").rstrip("/")
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or os.environ.get("SUPABASE_KEY") or ""
    if not base or not key:
        raise ValueError(
            "set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY (or SUPABASE_KEY) in .env"
        )
    query = urllib.parse.quote(round_id, safe="")
    request = urllib.request.Request(
        f"{base}/rest/v1/rounds?round_id=eq.{query}&select={ROUND_COLUMNS}",
        headers={
            "apikey": key,
            "Authorization": f"Bearer {key}",
            "Accept": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            rows = json.load(response)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")[:500]
        raise ValueError(f"supabase HTTP {exc.code}: {body}") from exc
    if not isinstance(rows, list):
        raise ValueError("supabase did not return a list of rounds")
    if not rows:
        return None
    row = rows[0]
    if not isinstance(row, dict):
        raise ValueError("supabase round row was not an object")
    return row


def parse_huggingface_link(url: str) -> Dict[str, str]:
    """Split a collected huggingface tree URL into repo, revision, and folder.

    https://huggingface.co/eliteminer/minos_ch20/tree/main/2026-09-02T21-44-00+00-00
    """
    parsed = urllib.parse.urlparse(url.strip())
    if parsed.netloc not in ("huggingface.co", "www.huggingface.co"):
        raise ValueError(f"not a huggingface.co link: {url}")
    parts = [urllib.parse.unquote(part) for part in parsed.path.split("/") if part]
    repo_type = "model"
    if parts and parts[0] == "datasets":
        repo_type = "dataset"
        parts = parts[1:]
    if len(parts) < 5 or parts[2] != "tree":
        raise ValueError(f"expected a /tree/<revision>/<folder> link, got: {url}")
    folder = "/".join(parts[4:]).strip("/")
    if not folder:
        raise ValueError(f"huggingface link has no round folder: {url}")
    return {
        "repo_id": f"{parts[0]}/{parts[1]}",
        "repo_type": repo_type,
        "revision": parts[3],
        "folder": folder,
    }


def ensure_reference(chrom: str) -> bool:
    """Download FASTA, index, dictionary, and RTG SDF when any file is missing.

    Files land in minos_subnet/datasets/reference/<chrom>/, which is where
    scoring looks. Files already on disk are left in place.
    """
    ref_base = (os.getenv("REF_S3_BASE") or DEFAULT_REF_BASE).rstrip("/")
    fa_dir = REFERENCE_DIR / chrom
    sdf_dir = fa_dir / f"{chrom}.sdf"
    fasta_files = [fa_dir / f"{chrom}.{ext}" for ext in FASTA_EXTS]
    sdf_files = [sdf_dir / name for name in SDF_FILES]
    missing = [path for path in fasta_files + sdf_files if not _file_ready(path)]
    if not missing:
        print(f"  Reference {chrom} already present", flush=True)
        return True

    print(f"  Reference {chrom} is missing {len(missing)} file(s). Downloading them first.", flush=True)
    fa_dir.mkdir(parents=True, exist_ok=True)
    for path in fasta_files:
        if _file_ready(path):
            print(f"    reuse {path.name}", flush=True)
            continue
        if not _download_ref_file(f"{ref_base}/{chrom}/{path.name}", path):
            print(f"  ERROR: could not download {path.name}", flush=True)
            return False
    if any(not _file_ready(path) for path in sdf_files):
        sdf_dir.mkdir(parents=True, exist_ok=True)
        for path in sdf_files:
            if _file_ready(path):
                print(f"    reuse {chrom}.sdf/{path.name}", flush=True)
                continue
            url = f"{ref_base}/{chrom}/{chrom}.sdf/{path.name}"
            if not _download_ref_file(url, path):
                print(f"  ERROR: could not download {chrom}.sdf/{path.name}", flush=True)
                return False
    print(f"  Reference {chrom} ready", flush=True)
    return True


def _file_ready(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def _download_ref_file(url: str, dest: Path) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": REF_USER_AGENT})
    written = 0
    next_report = 32 * 1024 * 1024
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            if not (200 <= response.status < 300):
                return False
            with tmp.open("wb") as handle:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    handle.write(chunk)
                    written += len(chunk)
                    if written >= next_report:
                        print(f"    {dest.name}: {written / (1024 * 1024):.0f} MB", flush=True)
                        next_report += 32 * 1024 * 1024
        if written <= 0:
            tmp.unlink(missing_ok=True)
            return False
        tmp.replace(dest)
        print(f"    {dest.name}: {written / (1024 * 1024):.1f} MB", flush=True)
        return True
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        print(f"    ERROR: {url} ({exc})", flush=True)
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        return False


def download_round(remote: Dict[str, str], force: bool) -> Path:
    """Download the reveal files into downloads/<folder>/. Reuse complete files."""
    try:
        from huggingface_hub import hf_hub_download, list_repo_files
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is not installed. Run: pip install -r requirements.txt"
        ) from exc

    folder_name = remote["folder"]
    local = DOWNLOADS / folder_name
    local.mkdir(parents=True, exist_ok=True)
    token = os.environ.get("HF_TOKEN") or None
    filenames = list_repo_files(
        repo_id=remote["repo_id"],
        repo_type=remote["repo_type"],
        revision=remote["revision"],
        token=token,
    )
    prefix = folder_name + "/"
    names = sorted(
        rel[len(prefix):]
        for rel in filenames
        if rel.startswith(prefix) and "/" not in rel[len(prefix):] and rel[len(prefix):]
    )
    if not names:
        raise RuntimeError(
            f"no files under {remote['repo_id']} {remote['revision']}/{folder_name}"
        )

    for name in names:
        destination = local / name
        if not force and destination.is_file() and destination.stat().st_size > 0:
            print(f"  reuse {name}", flush=True)
            continue
        print(f"  download {name}", flush=True)
        hf_hub_download(
            repo_id=remote["repo_id"],
            filename=f"{folder_name}/{name}",
            repo_type=remote["repo_type"],
            revision=remote["revision"],
            token=token,
            local_dir=str(DOWNLOADS),
            force_download=force,
        )
        if not destination.is_file() or destination.stat().st_size == 0:
            raise RuntimeError(f"download did not produce {destination}")

    missing = [
        name for name in ("input.bam", "truth.vcf.gz", "mutations.vcf.gz")
        if not (local / name).is_file()
    ]
    if missing:
        raise RuntimeError(f"round folder is missing {', '.join(missing)}")
    return local


def load_gatk_updates(path: Path) -> List[Tuple[str, Dict[str, Any]]]:
    """Named update dicts from gatk_updates.json. Sweeps become one dict per value."""
    if not path.is_file():
        raise FileNotFoundError(
            f"write {path.name} with the GATK keys to change, or pass --updates. "
            'Example: {"min_base_quality_score": 20}'
        )
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict) and isinstance(raw.get("experiments"), list):
        raw = raw["experiments"]
    if isinstance(raw, dict):
        raw = [raw]
    if not isinstance(raw, list):
        raise ValueError("gatk_updates.json must be an object or a list of objects")

    experiments: List[Tuple[str, Dict[str, Any]]] = []
    for index, item in enumerate(raw, 1):
        if not isinstance(item, dict):
            raise ValueError(f"entry {index} is not an object")
        experiments.extend(_expand_item(item, index))
    return experiments


def _expand_item(item: Dict[str, Any], index: int) -> List[Tuple[str, Dict[str, Any]]]:
    if "sweep" in item:
        return _expand_sweep(item, index)
    if "updates" in item:
        name = str(item.get("name") or f"update-{index}")
        updates = item["updates"]
        if not isinstance(updates, dict):
            raise ValueError(f"{name}: 'updates' must be an object of param -> value")
        return [(name, updates)]
    name = str(item.get("name") or f"update-{index}")
    updates = {key: value for key, value in item.items() if key not in PROVENANCE_KEYS}
    return [(name, updates)]


def _expand_sweep(item: Dict[str, Any], index: int) -> List[Tuple[str, Dict[str, Any]]]:
    sweep = item["sweep"]
    if not isinstance(sweep, dict):
        raise ValueError(f"entry {index}: 'sweep' must be an object")
    param = sweep.get("param")
    if not isinstance(param, str) or not param:
        raise ValueError(f"entry {index}: sweep.param is required")
    base_name = str(item.get("name") or param)
    return [
        (f"{base_name}-{value}", {param: value})
        for value in _sweep_values(sweep, index)
    ]


def _sweep_values(sweep: Dict[str, Any], index: int) -> List[Any]:
    if "values" in sweep:
        values = sweep["values"]
        if not isinstance(values, list) or not values:
            raise ValueError(f"entry {index}: sweep.values must be a non-empty list")
        return list(values)
    if "start" not in sweep or "stop" not in sweep:
        raise ValueError(f"entry {index}: sweep needs values, or start and stop")
    start, stop = sweep["start"], sweep["stop"]
    step = sweep.get("step", 1)
    if isinstance(start, bool) or isinstance(stop, bool) or isinstance(step, bool):
        raise ValueError(f"entry {index}: sweep start/stop/step must be numeric")
    if not isinstance(start, (int, float)) or not isinstance(stop, (int, float)):
        raise ValueError(f"entry {index}: sweep start/stop must be numeric")
    if not isinstance(step, (int, float)) or step == 0:
        raise ValueError(f"entry {index}: sweep step must be a non-zero number")
    if (stop - start) * step < 0:
        raise ValueError(f"entry {index}: sweep step has the wrong sign")
    values: List[Any] = []
    current: Any = start
    whole = all(isinstance(item, int) and not isinstance(item, bool) for item in (start, stop, step))
    while (step > 0 and current <= stop) or (step < 0 and current >= stop):
        values.append(int(current) if whole else current)
        current = current + step
        if len(values) > 10_000:
            raise ValueError(f"entry {index}: sweep produced more than 10000 values")
    if not values:
        raise ValueError(f"entry {index}: sweep produced no values")
    return values


def _prepare_configs(
    base: Dict[str, Any],
    experiments: List[Tuple[str, Dict[str, Any]]],
) -> Tuple[List[Tuple[str, Dict[str, Any], Dict[str, Any]]], bool]:
    """Merge each update onto the base GATK options and reject invalid params."""
    prepared: List[Tuple[str, Dict[str, Any], Dict[str, Any]]] = []
    rejected = False
    for name, updates in experiments:
        options = dict(base)
        unknown = [key for key in updates if key not in options and key not in GATK_QUALITY_PARAMS]
        options.update(updates)
        check = validate_and_build_flags("gatk", options)
        if unknown or not check["valid"]:
            rejected = True
            print(f"ERROR: {name} is not a valid GATK config", flush=True)
            for key in unknown:
                print(f"  unknown parameter {key}", flush=True)
            for error in check["errors"]:
                print(f"  {error}", flush=True)
            continue
        prepared.append((name, updates, options))
    return prepared, rejected


def _print_updates(base: Dict[str, Any], updates: Dict[str, Any]) -> None:
    if not updates:
        print("  GATK config unchanged (empty updates).", flush=True)
        return
    print("  GATK config updates:", flush=True)
    for key, value in updates.items():
        old = base.get(key, "(missing)")
        print(f"    {key}: {old} -> {value}", flush=True)


def _print_rank1(row: Dict[str, Any]) -> None:
    tool = row.get("rank_1_tool_name")
    score = row.get("rank_1_combined_final")
    if tool is None and score is None:
        return
    print(
        f"  Official rank 1: {tool}  combined_final={score}  "
        f"snp={row.get('rank_1_snp_final')}  indel={row.get('rank_1_indel_final')}",
        flush=True,
    )


def _write_output(
    path: Path,
    row: Dict[str, Any],
    link: str,
    folder: Path,
    region: str,
    results: List[Dict[str, Any]],
) -> None:
    payload = {
        "round_id": row.get("round_id"),
        "region": region,
        "huggingface": link,
        "folder": str(folder),
        "status": row.get("status"),
        "rank_1_tool_name": row.get("rank_1_tool_name"),
        "rank_1_combined_final": row.get("rank_1_combined_final"),
        "rank_1_snp_final": row.get("rank_1_snp_final"),
        "rank_1_indel_final": row.get("rank_1_indel_final"),
        "runs": [
            {
                "config_name": item.get("config_name"),
                "gatk_updates": item.get("gatk_updates"),
                "ok": item.get("ok"),
                "error": item.get("error"),
                "variant_count": item.get("variant_count"),
                "region": item.get("region"),
                "scores": item.get("scores"),
                "metrics": item.get("metrics"),
                "gatk_options": item.get("gatk_options"),
            }
            for item in results
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
