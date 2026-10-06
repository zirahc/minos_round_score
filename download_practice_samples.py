#!/usr/bin/env python3
"""Download every Minos practice sample (BAM + truth + mutations).

Talks to the platform's practice namespace with an ephemeral keypair — no
wallet, no chain, no submission. For each chromosome in the download, the
reference FASTA and RTG SDF are fetched first when they are missing.
Samples and references are written under $MINOS_SUBNET/datasets/.

This does not score anything and does not earn TAO.

Examples:
  python download_practice_samples.py
  python download_practice_samples.py --type chr18
  python download_practice_samples.py --type 18 --type chr19
  python download_practice_samples.py --list-types
  python download_practice_samples.py --force
  python download_practice_samples.py --sample-id <sample-id>
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import secrets
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))


def _load_local_env() -> None:
    """Load tuner .env files. Never print values."""
    try:
        from dotenv import load_dotenv
        load_dotenv(REPO_ROOT / ".env")
        load_dotenv(REPO_ROOT / ".env.tuner")
        return
    except ImportError:
        pass
    for name in (".env", ".env.tuner"):
        path = REPO_ROOT / name
        if not path.is_file():
            continue
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for raw in lines:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
                value = value[1:-1]
            if key and key not in os.environ:
                os.environ[key] = value


_load_local_env()

_subnet_raw = (os.getenv("MINOS_SUBNET") or "").strip()
if not _subnet_raw:
    print(
        "ERROR: set MINOS_SUBNET in .env to the minos_subnet checkout. "
        "Downloads go to that checkout's datasets/ folder.",
        flush=True,
    )
    sys.exit(2)
SUBNET_ROOT = Path(_subnet_raw).expanduser().resolve()
if not SUBNET_ROOT.is_dir():
    print(f"ERROR: MINOS_SUBNET is not a directory: {SUBNET_ROOT}", flush=True)
    sys.exit(2)
sys.path.insert(0, str(SUBNET_ROOT))

from bittensor_wallet import Keypair

from utils.file_utils import compute_sha256, download_file_with_fallback
from utils.path_utils import safe_round_dir_name
from utils.platform_client import (
    MinerPlatformClient,
    PlatformClientError,
    PlatformConfig,
)

DATASETS_DIR = SUBNET_ROOT / "datasets"
PRACTICE_DIR = DATASETS_DIR / "practice"
REFERENCE_DIR = DATASETS_DIR / "reference"
DEFAULT_PLATFORM_URL = "https://api.theminos.ai"
DEFAULT_REF_BASE = "https://api.theminos.ai/reference"
SAMTOOLS_IMAGE = "quay.io/biocontainers/samtools:1.20--h50ea8bc_0"
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

# Metadata written next to the files. Never include presigned URLs.
_META_KEYS = ("sample_id", "chromosome", "region", "num_mutations")


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    platform_url = (args.platform_url or os.getenv("PLATFORM_URL") or DEFAULT_PLATFORM_URL).rstrip("/")

    print("=" * 72, flush=True)
    print("  DOWNLOAD PRACTICE SAMPLES", flush=True)
    print("=" * 72, flush=True)
    print(f"  Platform:  {platform_url}", flush=True)
    print(f"  Subnet:    {SUBNET_ROOT}", flush=True)
    print(f"  Practice:  {PRACTICE_DIR}", flush=True)
    print(f"  Reference: {REFERENCE_DIR}", flush=True)
    if args.type:
        print(f"  Type:      {', '.join(_parse_types(args.type))}", flush=True)
    print(flush=True)

    try:
        return asyncio.run(
            _run(
                platform_url,
                sample_id=args.sample_id,
                types=args.type,
                list_types=args.list_types,
                force=args.force,
            )
        )
    except KeyboardInterrupt:
        print("\nInterrupted.", flush=True)
        return 130


def _parse_args(argv: Optional[List[str]]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download every Minos practice sample into datasets/practice.",
    )
    p.add_argument(
        "--type",
        action="append",
        default=[],
        metavar="CHR",
        help=(
            "Download only this chromosome type (chr18, chr19, ...). "
            "Repeat or comma-separate for several: --type chr18 --type chr19"
        ),
    )
    p.add_argument(
        "--list-types",
        action="store_true",
        help="List available chromosome types on the menu and exit.",
    )
    p.add_argument(
        "--sample-id",
        default=None,
        help="Download only this sample id instead of the full menu.",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Re-download even when a complete copy is already on disk.",
    )
    p.add_argument(
        "--platform-url",
        default=None,
        help=f"Platform base URL (default: PLATFORM_URL env or {DEFAULT_PLATFORM_URL}).",
    )
    return p.parse_args(argv)


def _normalize_type(raw: str) -> str:
    """Accept chr18, CHR18, or 18 → chr18."""
    value = (raw or "").strip()
    if not value:
        return ""
    lowered = value.lower()
    if lowered.startswith("chr"):
        rest = lowered[3:]
    else:
        rest = lowered
    if rest in ("x", "y", "m"):
        rest = rest.upper()
    return f"chr{rest}" if rest else ""


def _parse_types(raw_types: Optional[List[str]]) -> List[str]:
    wanted: List[str] = []
    seen = set()
    for chunk in raw_types or []:
        for part in str(chunk).split(","):
            chrom = _normalize_type(part)
            if chrom and chrom not in seen:
                seen.add(chrom)
                wanted.append(chrom)
    return wanted


def _sample_type(sample: Dict[str, Any]) -> str:
    chrom = sample.get("chromosome")
    if isinstance(chrom, str) and chrom.strip():
        return _normalize_type(chrom)
    region = sample.get("region")
    if isinstance(region, str) and ":" in region:
        return _normalize_type(region.split(":", 1)[0])
    return ""


def _print_types(samples: List[Dict[str, Any]]) -> None:
    counts: Dict[str, int] = {}
    unknown = 0
    for sample in samples:
        chrom = _sample_type(sample)
        if not chrom:
            unknown += 1
            continue
        counts[chrom] = counts.get(chrom, 0) + 1
    print("  Available types:", flush=True)
    if not counts and not unknown:
        print("    (none)", flush=True)
        return
    for chrom in sorted(counts, key=lambda c: (len(c), c)):
        print(f"    {chrom:8s}  {counts[chrom]} sample(s)", flush=True)
    if unknown:
        print(f"    (unknown) {unknown} sample(s)", flush=True)


async def _run(
    platform_url: str,
    sample_id: Optional[str],
    types: Optional[List[str]],
    list_types: bool,
    force: bool,
) -> int:
    try:
        client = MinerPlatformClient(
            keypair=Keypair.create_from_uri(f"//practice-{secrets.token_hex(4)}"),
            config=PlatformConfig(
                base_url=platform_url,
                timeout=float(os.getenv("PLATFORM_TIMEOUT", "60")),
            ),
            demo=False,
        )
    except ValueError as e:
        print(f"ERROR: invalid PLATFORM_URL: {e}", flush=True)
        return 2

    try:
        listing = await client.list_practice_samples()
    except PlatformClientError as e:
        print(f"ERROR: {e}", flush=True)
        return 2

    samples = listing.get("samples") or []
    if not samples:
        print("No practice samples available on this platform.", flush=True)
        return 2

    if list_types:
        _print_types(samples)
        return 0

    wanted_types = _parse_types(types)
    chosen = samples
    if wanted_types:
        chosen = [s for s in chosen if _sample_type(s) in wanted_types]
        if not chosen:
            print(
                f"ERROR: no practice samples of type {', '.join(wanted_types)}.",
                flush=True,
            )
            _print_types(samples)
            return 2

    if sample_id:
        match = next((s for s in chosen if s.get("sample_id") == sample_id), None)
        if not match:
            known = [s.get("sample_id") for s in chosen]
            scope = f" of type {', '.join(wanted_types)}" if wanted_types else ""
            print(
                f"ERROR: sample_id {sample_id!r} is not in the menu{scope}: {known}",
                flush=True,
            )
            return 2
        chosen = [match]

    chroms = _chroms_for_samples(chosen)
    if not _ensure_references(chroms):
        return 2

    print(f"  Samples:   {len(chosen)} of {len(samples)} on the menu", flush=True)
    for i, sample in enumerate(chosen, 1):
        print(
            f"    {i}. {sample.get('sample_id')}  "
            f"[{sample.get('chromosome')} {sample.get('region')}]  "
            f"mutations={sample.get('num_mutations')}",
            flush=True,
        )
    print(flush=True)

    ok = 0
    failed: List[str] = []
    for i, meta in enumerate(chosen, 1):
        sid = meta.get("sample_id")
        if not sid:
            print(f"[{i}/{len(chosen)}] ERROR: menu entry has no sample_id", flush=True)
            failed.append("(missing sample_id)")
            continue
        print(f"[{i}/{len(chosen)}] {sid}", flush=True)
        try:
            full = await client.get_practice_sample(sid)
        except PlatformClientError as e:
            print(f"   ERROR: {e}", flush=True)
            failed.append(sid)
            continue
        files = _download_practice_files(full, force=force)
        if files is None:
            failed.append(sid)
            continue
        _write_sidecars(files["dir"], full)
        print(f"   Saved {sid} -> {files['dir']}", flush=True)
        print(f"     BAM:       {files['bam'].name}", flush=True)
        print(f"     Truth:     {files['truth'].name}", flush=True)
        print(
            f"     Mutations: {files['mutations'].name if files['mutations'] else '(none)'}",
            flush=True,
        )
        ok += 1

    print(flush=True)
    print(f"Done. {ok} downloaded, {len(failed)} failed.", flush=True)
    if failed:
        print("Failed:", flush=True)
        for sid in failed:
            print(f"  - {sid}", flush=True)
        return 1
    return 0


def _chroms_for_samples(samples: List[Dict[str, Any]]) -> List[str]:
    chroms: List[str] = []
    seen = set()
    for sample in samples:
        chrom = _sample_type(sample)
        if chrom and chrom not in seen:
            seen.add(chrom)
            chroms.append(chrom)
    return chroms


def _ensure_references(chroms: List[str]) -> bool:
    """Download FASTA + RTG SDF for each chromosome that is not already on disk."""
    if not chroms:
        print("ERROR: chosen samples have no chromosome, so no reference can be fetched.", flush=True)
        return False
    ref_base = (os.getenv("REF_S3_BASE") or DEFAULT_REF_BASE).rstrip("/")
    print(f"  Reference: {REFERENCE_DIR}", flush=True)
    for chrom in chroms:
        if not _ensure_one_reference(chrom, ref_base):
            return False
    print(flush=True)
    return True


def _ensure_one_reference(chrom: str, ref_base: str) -> bool:
    fa_dir = REFERENCE_DIR / chrom
    fa_path = fa_dir / f"{chrom}.fa"
    sdf_dir = fa_dir / f"{chrom}.sdf"
    fasta_ready = fa_path.is_file() and fa_path.stat().st_size > 0
    sdf_ready = (sdf_dir / "seqdata0").is_file()
    if fasta_ready and sdf_ready and _fasta_sidecars_ready(fa_dir, chrom):
        print(f"  Reference {chrom} already present", flush=True)
        return True

    print(f"  Fetching reference for {chrom}...", flush=True)
    fa_dir.mkdir(parents=True, exist_ok=True)
    for ext in FASTA_EXTS:
        dest = fa_dir / f"{chrom}.{ext}"
        if dest.is_file() and dest.stat().st_size > 0:
            continue
        url = f"{ref_base}/{chrom}/{chrom}.{ext}"
        if not _download_ref_file(url, dest):
            print(f"   ERROR: could not download {chrom}.{ext}", flush=True)
            return False
    if not sdf_ready:
        sdf_dir.mkdir(parents=True, exist_ok=True)
        for name in SDF_FILES:
            dest = sdf_dir / name
            if dest.is_file() and dest.stat().st_size > 0:
                continue
            url = f"{ref_base}/{chrom}/{chrom}.sdf/{name}"
            if not _download_ref_file(url, dest):
                print(f"   ERROR: could not download {chrom}.sdf/{name}", flush=True)
                return False
    print(f"  Reference {chrom} ready", flush=True)
    return True


def _fasta_sidecars_ready(fa_dir: Path, chrom: str) -> bool:
    for ext in ("fa.fai", "dict"):
        path = fa_dir / f"{chrom}.{ext}"
        if not path.is_file() or path.stat().st_size == 0:
            return False
    return True


def _download_ref_file(url: str, dest: Path) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": REF_USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            if not (200 <= resp.status < 300):
                return False
            with tmp.open("wb") as out:
                while True:
                    chunk = resp.read(1024 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)
        if not tmp.is_file() or tmp.stat().st_size == 0:
            tmp.unlink(missing_ok=True)
            return False
        tmp.replace(dest)
        return True
    except (urllib.error.URLError, OSError, TimeoutError) as e:
        print(f"   ERROR: {url} ({e})", flush=True)
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        return False


def _download_practice_files(sample: dict, force: bool = False) -> Optional[dict]:
    """Download BAM + truth + mutations into datasets/practice/<hashed id>/."""
    sample_id = sample["sample_id"]
    out_dir = PRACTICE_DIR / safe_round_dir_name(sample_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    prefer_hippius = os.getenv("STORAGE_PRIMARY_BACKEND", "hippius").lower() != "aws_s3"
    bam_path = out_dir / "input.bam"
    truth_path = out_dir / "truth.vcf.gz"
    mutations_path = out_dir / "mutations.vcf.gz"

    bam_indexed = Path(f"{bam_path}.bai").exists() or bam_path.with_suffix(".bam.bai").exists()
    if not force and _reuse_ok(bam_path, sample.get("bam_sha256")) and bam_indexed:
        print(f"   BAM already downloaded — reusing {bam_path.name}", flush=True)
    else:
        if prefer_hippius:
            bam_url = sample.get("bam_presigned_url_backup") or sample.get("bam_presigned_url")
            bam_backup = sample.get("bam_presigned_url")
        else:
            bam_url = sample.get("bam_presigned_url")
            bam_backup = sample.get("bam_presigned_url_backup")
        if not bam_url and not bam_backup:
            print("   ERROR: sample has no BAM URL", flush=True)
            return None
        print("   Downloading BAM...", flush=True)
        primary, backup = _coalesce_urls(bam_url, bam_backup)
        got = download_file_with_fallback(
            primary, bam_path, backup_url=backup,
            expected_sha256=sample.get("bam_sha256"), show_progress=True,
        )
        if not got or not got.exists():
            print("   ERROR: failed to download BAM", flush=True)
            return None
        bam_index = Path(str(bam_path) + ".bai")
        _download_index_best_effort(
            sample.get("bam_index_presigned_url"),
            sample.get("bam_index_presigned_url_backup"),
            bam_index,
        )
        if not bam_index.exists() and not bam_path.with_suffix(".bam.bai").exists():
            if not _ensure_bam_index(bam_path):
                return None

    truth_indexed = Path(str(truth_path) + ".tbi").exists()
    if not force and _reuse_ok(truth_path, sample.get("truth_vcf_sha256")) and truth_indexed:
        print(f"   Truth already downloaded — reusing {truth_path.name}", flush=True)
    else:
        truth_url = sample.get("truth_vcf_presigned_url")
        truth_backup = sample.get("truth_vcf_presigned_url_backup")
        if not truth_url and not truth_backup:
            print("   ERROR: sample has no truth VCF URL", flush=True)
            return None
        print("   Downloading truth VCF...", flush=True)
        primary, backup = _coalesce_urls(truth_url, truth_backup)
        got = download_file_with_fallback(
            primary, truth_path, backup_url=backup,
            expected_sha256=sample.get("truth_vcf_sha256"), show_progress=True,
        )
        if not got or not got.exists():
            print("   ERROR: failed to download truth VCF", flush=True)
            return None
        _download_index_best_effort(
            sample.get("truth_vcf_index_presigned_url"),
            sample.get("truth_vcf_index_presigned_url_backup"),
            Path(str(truth_path) + ".tbi"),
        )

    have_mutations = False
    mut_indexed = Path(str(mutations_path) + ".tbi").exists()
    if not force and _reuse_ok(mutations_path, sample.get("mutations_vcf_sha256")) and mut_indexed:
        print(f"   Mutations already downloaded — reusing {mutations_path.name}", flush=True)
        have_mutations = True
    else:
        mut_url = sample.get("mutations_vcf_presigned_url")
        mut_backup = sample.get("mutations_vcf_presigned_url_backup")
        if mut_url or mut_backup:
            print("   Downloading mutations VCF...", flush=True)
            primary, backup = _coalesce_urls(mut_url, mut_backup)
            got = download_file_with_fallback(
                primary, mutations_path, backup_url=backup,
                expected_sha256=sample.get("mutations_vcf_sha256"), show_progress=True,
            )
            if got and got.exists():
                have_mutations = True
                _download_index_best_effort(
                    sample.get("mutations_vcf_index_presigned_url"),
                    sample.get("mutations_vcf_index_presigned_url_backup"),
                    Path(str(mutations_path) + ".tbi"),
                )

    return {
        "bam": bam_path,
        "truth": truth_path,
        "mutations": mutations_path if have_mutations else None,
        "dir": out_dir,
    }


def _write_sidecars(out_dir: Path, sample: dict) -> None:
    """Write region.txt and sample.json so local scoring can resolve the region.

    Only public metadata is stored — no URLs, signatures, or hashes.
    """
    region = sample.get("region")
    if isinstance(region, str) and region.strip():
        (out_dir / "region.txt").write_text(region.strip() + "\n", encoding="utf-8")
    meta = {key: sample.get(key) for key in _META_KEYS if sample.get(key) is not None}
    if meta:
        (out_dir / "sample.json").write_text(
            json.dumps(meta, indent=2) + "\n", encoding="utf-8"
        )


def _coalesce_urls(primary, backup) -> Tuple[Any, Any]:
    if not primary and backup:
        return backup, None
    return primary, backup


def _reuse_ok(path: Path, expected_sha256) -> bool:
    try:
        if not path.exists() or path.stat().st_size == 0:
            return False
    except OSError:
        return False
    if expected_sha256:
        try:
            if compute_sha256(path) != expected_sha256:
                print(f"   {path.name} on disk fails sha256 — will re-download.", flush=True)
                return False
        except Exception:
            return False
    return True


def _download_index_best_effort(url, backup, dest: Path) -> None:
    if not url and not backup:
        return
    primary, bkp = _coalesce_urls(url, backup)
    try:
        download_file_with_fallback(primary, dest, backup_url=bkp, show_progress=False)
    except Exception as e:
        print(f"   Index download for {dest.name} failed (non-fatal): {e}", flush=True)


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
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as e:
        print(f"   ERROR: failed to index BAM: {e}", flush=True)
        return False


if __name__ == "__main__":
    sys.exit(main())
