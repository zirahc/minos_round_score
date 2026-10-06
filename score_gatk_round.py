#!/usr/bin/env python3
"""Run GATK on the one downloaded round and print a v2 validator-style score.

Scores the round directory score_round.py last wrote under
minos_subnet/datasets/rounds/. Pass a folder to score a different download.
Tool is always GATK. Config defaults to minos_subnet/configs/gatk.conf.

This is a local self-scorer: no wallet, no chain, no submission.

Examples:
  python score_gatk_round.py
  python score_gatk_round.py 2026-09-02T21-44-00+00-00
  python score_gatk_round.py --config configs/gatk.conf
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent
_subnet_raw = (os.getenv("MINOS_SUBNET") or "").strip()
SUBNET_ROOT = (
    Path(_subnet_raw).expanduser().resolve()
    if _subnet_raw
    else (REPO_ROOT.parent / "minos_subnet").resolve()
)
# This checkout sits next to minos_subnet. Reference FASTA and the GATK
# template live there; BASE_DIR follows the genomics_config.py location.
sys.path.insert(0, str(SUBNET_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from base import BASE_DIR, GENOMICS_CONFIG, MINER_CONFIG, require_docker
from neurons import CHROMOSOME_PATTERN, safe_chrom
from templates import load_template
from templates.tool_params import REGION_PATTERN
from utils.scoring import (
    GERMLINE_FP_SCALE,
    V2_CORE_WEIGHT,
    V2_GERMLINE_WEIGHT,
    AdvancedScorer,
    HappyScorer,
    difficulty_class_counts,
    difficulty_weighted_f1,
    parse_happy_vcf,
)

PRACTICE_DIR = SUBNET_ROOT / "datasets" / "rounds"
SKIP_ROUNDS = frozenset({"round_2aeeddc1d86288f3"})
DEFAULT_CONFIG = SUBNET_ROOT / "configs" / "gatk.conf"
SAMTOOLS_IMAGE = "quay.io/biocontainers/samtools:1.20--h50ea8bc_0"
REGION_PADDING = 100_000


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    rounds_dir = Path(args.practice_dir).resolve() if args.practice_dir else PRACTICE_DIR
    folder = _resolve_downloaded_round(args.folder, rounds_dir)
    if folder is None:
        return 2
    files = _discover_files(folder)
    if files.get("error"):
        print(f"ERROR: {files['error']}", flush=True)
        return 2

    try:
        require_docker()
    except RuntimeError as e:
        print(f"ERROR: {e}", flush=True)
        return 2

    try:
        tool_config = _load_gatk_config(args.config)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as e:
        print(f"ERROR: could not load GATK config: {e}", flush=True)
        return 2

    n_params = len(tool_config.get("gatk_options", {}))
    print("=" * 72, flush=True)
    print("  GATK LOCAL FOLDER SCORER  (v2 only)", flush=True)
    print("=" * 72, flush=True)
    print(f"  Config:           {args.config}  ({n_params} params)", flush=True)
    print(f"  Round:            {folder}", flush=True)
    print(flush=True)

    result = score_folder(
        folder=folder,
        tool_config=tool_config,
        region_override=args.region,
        region_padding=args.region_padding,
    )
    _print_summary([result])
    if args.json_out:
        _write_json(Path(args.json_out), [_v2_save_payload(result)])

    return 0 if result.get("ok") else 1


def _parse_args(argv: Optional[List[str]]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run GATK on the one downloaded round and print the v2 score.",
    )
    p.add_argument(
        "folder",
        nargs="?",
        default=None,
        help="Round directory or its name under datasets/rounds. "
             "Defaults to the round score_round.py last downloaded.",
    )
    p.add_argument(
        "--practice-dir",
        default=str(PRACTICE_DIR),
        help="Parent folder that contains downloaded rounds "
             f"(default: {PRACTICE_DIR})",
    )
    p.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="GATK .conf or JSON (default: configs/gatk.conf)",
    )
    p.add_argument(
        "--region",
        default=None,
        help="Override the round region, e.g. chr20:45000000-50000000. "
             "Otherwise read region.txt / sample.json, or infer from the mutations VCF.",
    )
    p.add_argument(
        "--region-padding",
        type=int,
        default=REGION_PADDING,
        help="Bases added on each side when the region is inferred from a VCF "
             f"(default {REGION_PADDING}).",
    )
    p.add_argument(
        "--json-out",
        default=None,
        help="Write the v2 score for this round to this JSON file.",
    )
    return p.parse_args(argv)


def score_folder(
    folder: Path,
    tool_config: Dict[str, Any],
    region_override: Optional[str],
    region_padding: int,
) -> Dict[str, Any]:
    folder = folder.resolve()
    record: Dict[str, Any] = {"folder": str(folder), "ok": False}

    files = _discover_files(folder)
    if files.get("error"):
        print(f"   ERROR: {files['error']}", flush=True)
        record["error"] = files["error"]
        return record

    bam_path = files["bam"]
    truth_path = files["truth"]
    mutations_path = files["mutations"]
    record.update({
        "bam": str(bam_path),
        "truth": str(truth_path),
        "mutations": str(mutations_path),
    })

    region = region_override or files.get("region") or _infer_region(
        mutations_path or truth_path, padding=region_padding
    )
    if not region or not REGION_PATTERN.match(region):
        err = (
            f"could not resolve a valid region for {folder.name}. "
            "Pass --region, or put one line like chr20:45000000-50000000 in region.txt."
        )
        print(f"   ERROR: {err}", flush=True)
        record["error"] = err
        return record
    record["region"] = region

    chrom = safe_chrom(region)
    if chrom is None:
        err = f"region {region!r} does not name a supported chromosome (chr1-22, chrX, chrY, chrM)"
        print(f"   ERROR: {err}", flush=True)
        record["error"] = err
        return record

    ref_path = _resolve_reference(chrom)
    if ref_path is None:
        err = (
            f"reference not found for {chrom}. "
            f"Place it at datasets/reference/{chrom}/{chrom}.fa "
            "(start-miner.sh --practice fetches these)."
        )
        print(f"   ERROR: {err}", flush=True)
        record["error"] = err
        return record
    sdf_path = _resolve_sdf(chrom, ref_path)
    if sdf_path is None:
        err = (
            f"RTG SDF not found for {chrom}. "
            f"Expected datasets/reference/{chrom}/{chrom}.sdf "
            "(start-miner.sh --practice fetches these)."
        )
        print(f"   ERROR: {err}", flush=True)
        record["error"] = err
        return record

    print(f"   BAM:        {bam_path.name}", flush=True)
    print(f"   Truth:      {truth_path.name}", flush=True)
    print(f"   Mutations:  {mutations_path.name}", flush=True)
    print(f"   Region:     {region}", flush=True)
    print(f"   Reference:  {ref_path}", flush=True)
    print(f"   SDF:        {sdf_path}", flush=True)

    if not _ensure_bam_index(bam_path):
        record["error"] = "failed to index BAM"
        return record
    if not _ensure_fasta_index(ref_path):
        record["error"] = "failed to index reference"
        return record

    out_dir = folder / "score_run"
    out_dir.mkdir(parents=True, exist_ok=True)
    output_vcf = out_dir / "output.vcf.gz"

    run_config = {
        **tool_config,
        "timeout": GENOMICS_CONFIG.get("variant_calling_timeout", 1800),
        "threads": MINER_CONFIG.get("num_threads", 4),
        "ref_build": "GRCh38",
    }

    print(f"\n   Running GATK HaplotypeCaller...", flush=True)
    t0 = time.time()
    template = load_template("gatk")
    result = template.variant_call(
        bam_path=bam_path,
        reference_path=ref_path,
        output_vcf_path=output_vcf,
        region=region,
        config=run_config,
    )
    elapsed = time.time() - t0
    if not result.get("success"):
        err = result.get("error", "unknown GATK failure")
        print(f"   ERROR: variant calling failed: {err}", flush=True)
        record["error"] = err
        record["gatk_seconds"] = elapsed
        return record

    query_vcf = _find_query_vcf(out_dir, output_vcf)
    if query_vcf is None:
        err = "GATK reported success but no output VCF was produced"
        print(f"   ERROR: {err}", flush=True)
        record["error"] = err
        return record

    variant_count = result.get("variant_count", 0)
    print(f"   Variants called: {variant_count}  ({elapsed:.0f}s)", flush=True)
    print(f"   VCF: {query_vcf}", flush=True)
    record["gatk_seconds"] = elapsed
    record["variant_count"] = variant_count
    record["query_vcf"] = str(query_vcf)

    print(f"\n   Scoring with hap.py...", flush=True)
    scorer = HappyScorer()
    metrics = scorer.score_vcf(
        truth_vcf=str(truth_path),
        query_vcf=str(query_vcf),
        reference_fasta=str(ref_path),
        region=region,
        reference_sdf=str(sdf_path),
        mutations_vcf=str(mutations_path),
    )
    if metrics is None:
        err = "hap.py returned no valid metrics"
        print(f"   ERROR: {err}", flush=True)
        record["error"] = err
        return record

    happy = _public_happy_metrics(metrics)
    record["metrics"] = happy
    _print_happy_metrics(happy)

    detailed = _v2_breakdown(metrics, truth_path)
    if detailed.get("error") or detailed.get("advanced_score") is None:
        err = detailed.get("error") or "v2 score could not be computed"
        print(f"   ERROR: {err}", flush=True)
        record["error"] = err
        record["scores"] = detailed
        _print_v2_score(detailed)
        _write_json(out_dir / "score_details.json", _v2_save_payload(record))
        return record

    record["ok"] = True
    record["scores"] = detailed
    _print_v2_score(detailed)
    _write_json(out_dir / "score_details.json", _v2_save_payload(record))
    return record


def _resolve_downloaded_round(raw: Optional[str], rounds_dir: Path) -> Optional[Path]:
    """The folder argument, or the round recorded in datasets/rounds/.latest."""
    if raw:
        path = Path(raw)
        if not path.is_absolute():
            named = rounds_dir / raw
            path = named if named.exists() else Path(raw)
        path = path.resolve()
        if not path.is_dir():
            print(f"ERROR: round folder not found: {path}", flush=True)
            return None
        return path

    marker = rounds_dir / ".latest"
    if not marker.is_file():
        print(
            f"ERROR: no downloaded round recorded at {marker}. "
            "Run score_round.py first, or pass the round folder.",
            flush=True,
        )
        return None
    text = marker.read_text(encoding="utf-8").strip()
    if not text:
        print(f"ERROR: {marker} is empty.", flush=True)
        return None
    path = Path(text)
    if not path.is_absolute():
        path = rounds_dir / text
    path = path.resolve()
    if not path.is_dir():
        print(f"ERROR: downloaded round folder not found: {path}", flush=True)
        return None
    return path


def _list_round_folders(practice_dir: Path) -> Tuple[Optional[List[Path]], List[str]]:
    """Immediate child directories of practice_dir that look like scored samples."""
    if not practice_dir.is_dir():
        print(f"ERROR: practice directory not found: {practice_dir}", flush=True)
        return None, []

    rounds: List[Path] = []
    skipped: List[str] = []
    for child in sorted(practice_dir.iterdir()):
        if not child.is_dir() or child.name.startswith("."):
            continue
        if child.name in SKIP_ROUNDS:
            print(f"  Skipping round {child.name}", flush=True)
            continue
        files = _discover_files(child)
        if files.get("error"):
            skipped.append(child.name)
            continue
        rounds.append(child)
    return rounds, skipped


def _discover_files(folder: Path) -> Dict[str, Any]:
    if not folder.is_dir():
        return {"error": f"not a directory: {folder}"}

    bam = _pick_file(folder, preferred=("input.bam",), globs=("*.bam",))
    truth = _pick_file(
        folder,
        preferred=("truth.vcf.gz", "truth.vcf"),
        globs=("truth*.vcf.gz", "truth*.vcf", "*truth*.vcf.gz"),
    )
    mutations = _pick_file(
        folder,
        preferred=("mutations.vcf.gz", "mutations.vcf"),
        globs=("mutations*.vcf.gz", "*mutations*.vcf.gz", "*mut*.vcf.gz"),
    )
    missing = [name for name, path in (("BAM", bam), ("truth VCF", truth), ("mutations VCF", mutations)) if path is None]
    if missing:
        return {"error": f"folder {folder} is missing {', '.join(missing)}"}

    region = _read_region_sidecar(folder)
    return {"bam": bam, "truth": truth, "mutations": mutations, "region": region}


def _pick_file(folder: Path, preferred: Tuple[str, ...], globs: Tuple[str, ...]) -> Optional[Path]:
    for name in preferred:
        path = folder / name
        if path.is_file() and path.stat().st_size > 0:
            return path
    matches: List[Path] = []
    for pattern in globs:
        for path in folder.glob(pattern):
            if not path.is_file() or path.stat().st_size <= 0:
                continue
            if path.name.endswith(".bai") or path.name.endswith(".tbi"):
                continue
            matches.append(path)
    unique = sorted({p.resolve() for p in matches})
    return unique[0] if unique else None


def _read_region_sidecar(folder: Path) -> Optional[str]:
    for name in ("region.txt", "region"):
        path = folder / name
        if path.is_file():
            text = path.read_text(encoding="utf-8").strip().splitlines()
            if text and REGION_PATTERN.match(text[0].strip()):
                return text[0].strip()
    for name in ("sample.json", "meta.json"):
        path = folder / name
        if not path.is_file():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        region = data.get("region") if isinstance(data, dict) else None
        if isinstance(region, str) and REGION_PATTERN.match(region):
            return region
    return None


def _infer_region(vcf_path: Path, padding: int) -> Optional[str]:
    opener = gzip.open if str(vcf_path).endswith(".gz") else open
    chrom = None
    min_pos = None
    max_pos = None
    try:
        with opener(vcf_path, "rt") as fh:
            for line in fh:
                if not line.strip() or line.startswith("#"):
                    continue
                parts = line.split("\t")
                if len(parts) < 2:
                    continue
                contig = parts[0]
                try:
                    pos = int(parts[1])
                except ValueError:
                    continue
                if chrom is None:
                    chrom = contig
                if contig != chrom:
                    continue
                min_pos = pos if min_pos is None else min(min_pos, pos)
                max_pos = pos if max_pos is None else max(max_pos, pos)
    except OSError:
        return None
    if chrom is None or min_pos is None or max_pos is None:
        return None
    if not CHROMOSOME_PATTERN.match(chrom):
        return None
    start = max(1, min_pos - max(0, padding))
    end = max_pos + max(0, padding)
    if end < start:
        return None
    region = f"{chrom}:{start}-{end}"
    print(
        f"   Inferred region {region} from {vcf_path.name} "
        f"(span {min_pos}-{max_pos}, padding {padding})",
        flush=True,
    )
    return region


def _load_gatk_config(config_path: Optional[str]) -> Dict[str, Any]:
    """Load a GATK .conf or JSON into the dict templates.gatk expects."""
    if not config_path:
        from utils.config_loader import extract_tool_options
        return {"gatk_options": extract_tool_options("gatk")}

    p = Path(config_path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")

    text = p.read_text(encoding="utf-8")
    stripped = text.lstrip()
    if stripped.startswith("{"):
        data = json.loads(text)
        if "gatk_options" in data and isinstance(data["gatk_options"], dict):
            return {"gatk_options": data["gatk_options"]}
        return {"gatk_options": {k: v for k, v in data.items() if not isinstance(v, (dict, list))}}

    options: Dict[str, Any] = {}
    for line_num, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(f"{p}:{line_num}: expected key=value, got: {raw!r}")
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip()
        if val.lower() in ("true", "false"):
            options[key] = val.lower() == "true"
        else:
            try:
                options[key] = int(val)
            except ValueError:
                try:
                    options[key] = float(val)
                except ValueError:
                    options[key] = val
    return {"gatk_options": options}


def _resolve_reference(chrom: str) -> Optional[Path]:
    if not CHROMOSOME_PATTERN.match(chrom):
        return None
    ref_path = BASE_DIR / "datasets" / "reference" / chrom / f"{chrom}.fa"
    if ref_path.exists():
        return ref_path
    legacy = BASE_DIR / "datasets" / "reference" / "chr20.fa"
    if chrom == "chr20" and legacy.exists():
        return legacy
    return None


def _resolve_sdf(chrom: str, ref_path: Path) -> Optional[Path]:
    for cand in (
        BASE_DIR / "datasets" / "reference" / chrom / f"{chrom}.sdf",
        BASE_DIR / "datasets" / "reference" / f"{chrom}.sdf",
        ref_path.parent / f"{chrom}.sdf",
    ):
        if cand.is_dir():
            return cand
    return None


def _ensure_bam_index(bam_path: Path) -> bool:
    if Path(f"{bam_path}.bai").exists() or bam_path.with_suffix(".bam.bai").exists():
        return True
    print(f"   Creating BAM index for {bam_path.name}...", flush=True)
    try:
        subprocess.run(
            [
                "docker", "run", "--rm",
                "-v", f"{bam_path.parent}:/data",
                SAMTOOLS_IMAGE,
                "samtools", "index", f"/data/{bam_path.name}",
            ],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=600,
        )
        return Path(f"{bam_path}.bai").exists() or bam_path.with_suffix(".bam.bai").exists()
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        print(f"   ERROR: failed to index BAM: {e}", flush=True)
        return False


def _ensure_fasta_index(ref_path: Path) -> bool:
    if Path(f"{ref_path}.fai").exists():
        return True
    print(f"   Creating reference index for {ref_path.name}...", flush=True)
    try:
        subprocess.run(
            [
                "docker", "run", "--rm",
                "-v", f"{ref_path.parent}:/data",
                SAMTOOLS_IMAGE,
                "samtools", "faidx", f"/data/{ref_path.name}",
            ],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=600,
        )
        return Path(f"{ref_path}.fai").exists()
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        print(f"   ERROR: failed to index reference: {e}", flush=True)
        return False


def _find_query_vcf(out_dir: Path, preferred: Path) -> Optional[Path]:
    if preferred.exists():
        return preferred
    for ext in (".vcf.gz", ".vcf"):
        alt = out_dir / f"output{ext}"
        if alt.exists():
            return alt
    return None


def _v2_breakdown(metrics: Dict[str, Any], truth_path: Optional[Path] = None) -> Dict[str, Any]:
    happy_vcf = metrics.get("happy_vcf_path")
    out: Dict[str, Any] = {"scoring_version": "v2", "advanced_score": None, "combined_final": None}
    if not happy_vcf or not Path(happy_vcf).exists():
        out["error"] = "hap.py VCF missing; v2 cannot be computed"
        return out
    try:
        records = parse_happy_vcf(
            happy_vcf,
            truth_vcf_path=str(truth_path) if truth_path else None,
        )
    except Exception as e:  # noqa: BLE001
        out["error"] = f"hap.py VCF parse failed: {e}"
        return out
    if records is None:
        out["error"] = "hap.py VCF could not be parsed completely"
        return out

    class_counts = difficulty_class_counts(records) if records else {}
    core = difficulty_weighted_f1(class_counts) if class_counts else None
    fp_per_target = metrics.get("fp_per_target")
    germline = None
    try:
        fp_rate = float(fp_per_target) if fp_per_target is not None else None
    except (TypeError, ValueError):
        fp_rate = None
    if fp_rate is not None and math.isfinite(fp_rate) and fp_rate >= 0:
        germline = math.exp(-fp_rate / GERMLINE_FP_SCALE)

    score = AdvancedScorer.compute_score_v2(metrics, class_counts)
    combined = (score / 100.0) if isinstance(score, (int, float)) else None
    would_record = (
        isinstance(combined, float)
        and math.isfinite(combined)
        and 0.0 < combined <= 1.0
    )
    out.update({
        "difficulty_class_counts": class_counts,
        "core": core,
        "fp_per_target": fp_per_target,
        "germline": germline,
        "weights": {"core": V2_CORE_WEIGHT, "germline": V2_GERMLINE_WEIGHT},
        "advanced_score": score,
        "combined_final": combined,
        "would_record": would_record,
    })
    return out


def _v2_save_payload(record: Dict[str, Any]) -> Dict[str, Any]:
    scores = record.get("scores") or {}
    metrics = record.get("metrics") if isinstance(record.get("metrics"), dict) else {}
    combined = scores.get("combined_final")
    payload = {
        "folder": record.get("folder"),
        "region": record.get("region"),
        "tool_name": "gatk",
        "variant_count": record.get("variant_count"),
        "ok": record.get("ok"),
        "scorer": "AdvancedV2",
        "scoring_version": "v2",
        "scoring_status": "scored" if record.get("ok") else "error",
        "snp_final": metrics.get("snp_final", metrics.get("f1_snp")),
        "indel_final": metrics.get("indel_final", metrics.get("f1_indel")),
        "score": combined,
        "core": scores.get("core"),
        "fp_per_target": scores.get("fp_per_target", metrics.get("fp_per_target")),
        "germline": scores.get("germline"),
        "difficulty_class_counts": scores.get("difficulty_class_counts"),
        "weights": scores.get("weights"),
        "advanced_score": scores.get("advanced_score"),
        "combined_final": combined,
        "would_record": scores.get("would_record"),
        "metrics": metrics,
    }
    if record.get("error"):
        payload["error"] = record["error"]
    return payload


_HAPPY_ALIASES = (
    ("snp_f1", "f1_snp"),
    ("snp_precision", "precision_snp"),
    ("snp_recall", "recall_snp"),
    ("snp_tp", "tp_snp"),
    ("snp_fp", "fp_snp"),
    ("snp_fn", "fn_snp"),
    ("indel_f1", "f1_indel"),
    ("indel_precision", "precision_indel"),
    ("indel_recall", "recall_indel"),
    ("indel_tp", "tp_indel"),
    ("indel_fp", "fp_indel"),
    ("indel_fn", "fn_indel"),
    ("snp_final", "f1_snp"),
    ("indel_final", "f1_indel"),
    ("ti_tv_ratio", "titv_query_snp"),
    ("het_hom_ratio", "hethom_query_snp"),
)

_HAPPY_HIGHLIGHT = (
    "f1_snp", "precision_snp", "recall_snp", "tp_snp", "fp_snp", "fn_snp",
    "truth_total_snp", "query_total_snp", "target_total_snp",
    "f1_indel", "precision_indel", "recall_indel", "tp_indel", "fp_indel", "fn_indel",
    "truth_total_indel", "query_total_indel", "target_total_indel",
    "weighted_f1", "snp_final", "indel_final",
    "region_fp_snp", "region_fp_indel", "region_fp_total",
    "fp_per_target", "snp_fp_per_target", "overcall_penalty",
    "titv_query_snp", "titv_truth_snp", "ti_tv_ratio",
    "hethom_query_snp", "hethom_truth_snp", "hethom_query_indel", "hethom_truth_indel",
    "het_hom_ratio", "frac_na_snp", "frac_na_indel", "query_unk_snp", "query_unk_indel",
)


def _public_happy_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Keep hap.py's full metric dict and add evaluation-style aliases."""
    out: Dict[str, Any] = {}
    for key, value in metrics.items():
        if isinstance(value, Path):
            out[key] = str(value)
        else:
            out[key] = value
    for dest, src in _HAPPY_ALIASES:
        if dest not in out and src in out:
            out[dest] = out[src]
    out.setdefault("scorer", "AdvancedV2")
    return out


def _print_happy_metrics(metrics: Dict[str, Any]) -> None:
    print(f"\n   {'=' * 64}", flush=True)
    print("   HAP.PY METRICS", flush=True)
    print(f"   {'=' * 64}", flush=True)
    print(
        f"   SNP    F1={_fmt(metrics.get('f1_snp'))}  "
        f"prec={_fmt(metrics.get('precision_snp'))}  "
        f"recall={_fmt(metrics.get('recall_snp'))}  "
        f"TP={_fmt(metrics.get('tp_snp'))}  "
        f"FP={_fmt(metrics.get('fp_snp'))}  "
        f"FN={_fmt(metrics.get('fn_snp'))}",
        flush=True,
    )
    print(
        f"   INDEL  F1={_fmt(metrics.get('f1_indel'))}  "
        f"prec={_fmt(metrics.get('precision_indel'))}  "
        f"recall={_fmt(metrics.get('recall_indel'))}  "
        f"TP={_fmt(metrics.get('tp_indel'))}  "
        f"FP={_fmt(metrics.get('fp_indel'))}  "
        f"FN={_fmt(metrics.get('fn_indel'))}",
        flush=True,
    )
    print(
        f"   REGION FP  snp={_fmt(metrics.get('region_fp_snp'))}  "
        f"indel={_fmt(metrics.get('region_fp_indel'))}  "
        f"total={_fmt(metrics.get('region_fp_total'))}  "
        f"fp_per_target={_fmt(metrics.get('fp_per_target'))}",
        flush=True,
    )
    print("   all hap.py keys:", flush=True)
    shown = set()
    for key in list(_HAPPY_HIGHLIGHT) + sorted(metrics):
        if key in shown or key not in metrics:
            continue
        shown.add(key)
        value = metrics[key]
        if isinstance(value, dict):
            print(f"      {key:28s} {json.dumps(value, default=str)}", flush=True)
        else:
            print(f"      {key:28s} {_fmt(value)}", flush=True)
    print(f"   {'=' * 64}", flush=True)


def _print_v2_score(scores: Dict[str, Any]) -> None:
    print(f"\n   {'=' * 64}", flush=True)
    print("   V2 SCORE", flush=True)
    print(f"   {'=' * 64}", flush=True)
    print(f"   {'core':28s} {_fmt(scores.get('core'))}", flush=True)
    print(f"   {'fp_per_target':28s} {_fmt(scores.get('fp_per_target'))}", flush=True)
    print(f"   {'germline':28s} {_fmt(scores.get('germline'))}", flush=True)
    print(f"   {'advanced_score':28s} {_fmt(scores.get('advanced_score'))}", flush=True)
    print(f"   {'combined_final':28s} {_fmt(scores.get('combined_final'))}", flush=True)
    counts = scores.get("difficulty_class_counts") or {}
    if counts:
        print("   difficulty class counts:", flush=True)
        for cls, vals in counts.items():
            print(
                f"      {cls:16s} tp={vals.get('tp', 0)}  "
                f"fn={vals.get('fn', 0)}  fp={vals.get('fp', 0)}",
                flush=True,
            )
    if scores.get("would_record"):
        print("   verdict                     would record (this is what a validator stores)", flush=True)
    else:
        print("   verdict                     unavailable / out of range — a validator discards this", flush=True)
    print(f"   {'=' * 64}\n", flush=True)


def _print_summary(results: List[Dict[str, Any]]) -> None:
    print(f"\n{'=' * 72}", flush=True)
    print("  SUMMARY  (v2)", flush=True)
    print(f"{'=' * 72}", flush=True)
    print(
        f"  {'folder':32s} {'ok':4s} {'vars':>6s} {'snp_f1':>10s} "
        f"{'indel_f1':>10s} {'v2':>10s} {'final':>10s}",
        flush=True,
    )
    for r in results:
        name = Path(r.get("folder", "")).name[:32]
        ok = "yes" if r.get("ok") else "no"
        nvars = r.get("variant_count", "")
        scores = r.get("scores") or {}
        metrics = r.get("metrics") or {}
        v2 = scores.get("advanced_score")
        final = scores.get("combined_final")
        print(
            f"  {name:32s} {ok:4s} {str(nvars):>6s} "
            f"{_fmt(metrics.get('f1_snp')):>10s} "
            f"{_fmt(metrics.get('f1_indel')):>10s} "
            f"{_fmt(v2):>10s} {_fmt(final):>10s}",
            flush=True,
        )
        if r.get("error"):
            print(f"      error: {r['error']}", flush=True)
    print(f"{'=' * 72}\n", flush=True)


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return str(value)
        if value == math.floor(value) and abs(value) >= 1:
            return f"{value:.1f}" if value != int(value) else str(int(value))
        return f"{value:.6f}"
    return str(value)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def _write_json(path: Path, payload: Any) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"   Wrote {path}", flush=True)
    except OSError as e:
        print(f"   WARNING: could not write {path}: {e}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
