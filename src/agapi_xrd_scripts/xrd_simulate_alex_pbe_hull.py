import argparse
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from jarvis.analysis.diffraction.xrd import XRD
from jarvis.core.atoms import Atoms

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kw):
        return it


# ---------------------------------------------------------------------------
# Single-structure helpers (unchanged behaviour)
# ---------------------------------------------------------------------------

def load_atoms(poscar=None, cif=None, jid=None, db=None):
    if poscar is not None:
        from jarvis.io.vasp.inputs import Poscar
        return Poscar.from_file(poscar).atoms
    if cif is not None:
        from jarvis.io.cif import CifParser
        return CifParser(cif).get_atoms()
    if jid is not None:
        from jarvis.db.figshare import data as figshare_data
        dataset = db or "dft_3d"
        for entry in figshare_data(dataset):
            if entry["jid"] == jid:
                return Atoms.from_dict(entry["atoms"])
        raise ValueError(f"JID {jid!r} not found in dataset {dataset!r}")
    raise ValueError("One of --poscar, --cif, --jid, or --dataset must be provided")


def simulate_dense_xrd(atoms, wavelength=1.54184, two_theta_min=5.0,
                       two_theta_max=90.0, step=0.01, sigma=0.1):
    two_theta_peaks, _d, intensities = XRD(
        wavelength=wavelength, thetas=[two_theta_min, two_theta_max]
    ).simulate(atoms=atoms)

    two_theta_peaks = np.array(two_theta_peaks, dtype=float)
    intensities = np.array(intensities, dtype=float)
    intensities /= intensities.max()

    grid = np.arange(two_theta_min, two_theta_max + step, step)
    dense = np.zeros(len(grid), dtype=float)
    for x0, amp in zip(two_theta_peaks, intensities):
        dense += amp * np.exp(-0.5 * ((grid - x0) / sigma) ** 2)

    if dense.max() > 0:
        dense /= dense.max()

    return grid, dense


def write_spectrum(grid, dense, output=None):
    lines = [f"{x:.5f} {y:.6f}" for x, y in zip(grid, dense)]
    text = "\n".join(lines)
    if output is not None:
        with open(output, "w") as f:
            f.write(text + "\n")
    else:
        print(text)


def save_plot(grid, dense, path, formula):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(grid, dense, lw=0.8)
    ax.set_xlabel("2θ (degrees)")
    ax.set_ylabel("Intensity (normalized)")
    ax.set_title(f"Simulated XRD — {formula}")
    ax.set_xlim(grid[0], grid[-1])
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Dataset-mode helpers
# ---------------------------------------------------------------------------

def _process_entry_worker(payload: dict):
    """Run in a worker process. payload contains atoms_dict + sim params."""
    import json as _json
    from jarvis.core.atoms import Atoms as _Atoms
    from jarvis.analysis.diffraction.xrd import XRD as _XRD
    import numpy as _np

    jid = payload["jid"]
    try:
        atoms = _Atoms.from_dict(payload["atoms_dict"])
        wavelength = payload["wavelength"]
        two_theta_min = payload["two_theta_min"]
        two_theta_max = payload["two_theta_max"]
        step = payload["step"]
        sigma = payload["sigma"]

        two_theta_peaks, _d, intensities = _XRD(
            wavelength=wavelength, thetas=[two_theta_min, two_theta_max]
        ).simulate(atoms=atoms)

        two_theta_peaks = _np.array(two_theta_peaks, dtype=float)
        intensities = _np.array(intensities, dtype=float)
        intensities /= intensities.max()

        grid = _np.arange(two_theta_min, two_theta_max + step, step)
        dense = _np.zeros(len(grid), dtype=float)
        for x0, amp in zip(two_theta_peaks, intensities):
            dense += amp * _np.exp(-0.5 * ((grid - x0) / sigma) ** 2)
        if dense.max() > 0:
            dense /= dense.max()

        return {
            "jid": jid,
            "formula": atoms.composition.reduced_formula,
            "atoms_json": _json.dumps(payload["atoms_dict"]),
            "xrd_intensities": dense.astype(_np.float32).tolist(),
        }
    except Exception as exc:
        print(f"[warn] {jid} failed: {exc}", file=sys.stderr)
        return {"__error__": True, "jid": jid, "error": str(exc)}


def _get_processed_jids(output_dir: Path) -> set:
    import pyarrow.parquet as pq
    jids = set()
    for p in output_dir.glob("train-*.parquet"):
        try:
            t = pq.read_table(str(p), columns=["jid"])
            jids.update(t.column("jid").to_pylist())
        except Exception:
            pass
    return jids


def _write_shard(records: list, path: Path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    schema = pa.schema([
        pa.field("jid", pa.string()),
        pa.field("formula", pa.string()),
        pa.field("atoms_json", pa.string()),
        pa.field("xrd_intensities", pa.list_(pa.float32())),
    ])
    arrays = {
        "jid": pa.array([r["jid"] for r in records], type=pa.string()),
        "formula": pa.array([r["formula"] for r in records], type=pa.string()),
        "atoms_json": pa.array([r["atoms_json"] for r in records], type=pa.string()),
        "xrd_intensities": pa.array(
            [r["xrd_intensities"] for r in records],
            type=pa.list_(pa.float32()),
        ),
    }
    table = pa.table(arrays, schema=schema)
    pq.write_table(table, str(path), compression="snappy")


def _write_metadata(output_dir: Path, args, grid: np.ndarray):
    meta = {
        "dataset": args.dataset,
        "wavelength": args.wavelength,
        "two_theta_min": args.two_theta_min,
        "two_theta_max": args.two_theta_max,
        "step": args.step,
        "sigma": args.sigma,
        "two_theta_grid": grid.tolist(),
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)


def run_dataset_mode(args):
    from jarvis.db.figshare import data as figshare_data

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    errors_path = output_dir / "errors.jsonl"

    job_start_epoch = time.time()
    print(f"  Computation start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)

    print(f"Downloading dataset '{args.dataset}' ...", flush=True)
    entries = figshare_data(args.dataset)
    print(f"  {len(entries)} entries loaded.", flush=True)

    processed_jids: set = set()
    if args.resume:
        processed_jids = _get_processed_jids(output_dir)
        print(f"  Resume: {len(processed_jids)} already processed.", flush=True)

    already_done = len(processed_jids)
    grand_total = len(entries)
    to_process = [e for e in entries if e.get("jid", "") not in processed_jids]
    print(f"  {len(to_process)} entries to simulate.", flush=True)

    session_start_epoch = time.time()

    if not to_process:
        print("Nothing to do.", flush=True)
        return

    total_shards = math.ceil(len(to_process) / args.shard_size)
    # Existing shards offset when resuming
    existing_shards = len(list(output_dir.glob("train-*.parquet")))

    sim_params = {
        "wavelength": args.wavelength,
        "two_theta_min": args.two_theta_min,
        "two_theta_max": args.two_theta_max,
        "step": args.step,
        "sigma": args.sigma,
    }

    # Build the shared grid for metadata
    grid = np.arange(args.two_theta_min, args.two_theta_max + args.step, args.step)

    records: list = []
    shard_idx = existing_shards
    grand_total_shards = existing_shards + total_shards
    n_session_done = 0

    # Build payloads lazily to avoid materialising all 100k+ dicts at once.
    def _make_payload(i, e):
        return {"jid": e.get("jid", f"entry_{i}"), "atoms_dict": e["atoms"], **sim_params}

    # Bounded in-flight queue: avoids pickling all entries before tqdm starts
    # and prevents OOM from holding O(N) Future objects simultaneously.
    MAX_IN_FLIGHT = args.num_workers * 4
    payload_iter = (_make_payload(i, e) for i, e in enumerate(to_process))
    in_flight: dict = {}  # future -> jid

    def _fill_queue(executor):
        while len(in_flight) < MAX_IN_FLIGHT:
            try:
                p = next(payload_iter)
                fut = executor.submit(_process_entry_worker, p)
                in_flight[fut] = p["jid"]
            except StopIteration:
                break

    with open(errors_path, "a") as err_f:
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            _fill_queue(executor)
            with tqdm(total=len(to_process), desc="Simulating XRD") as pbar:
                while in_flight:
                    # as_completed on a snapshot; we mutate in_flight each iteration
                    for future in as_completed(list(in_flight)):
                        jid = in_flight.pop(future)
                        _fill_queue(executor)
                        pbar.update(1)
                        n_session_done += 1

                        result = future.result()
                        if result is None or result.get("__error__"):
                            err_f.write(json.dumps({
                                "jid": result["jid"] if result else jid,
                                "error": result.get("error", "unknown") if result else "None returned",
                            }) + "\n")
                            err_f.flush()
                            continue

                        records.append(result)

                        if len(records) >= args.shard_size:
                            shard_path = output_dir / f"train-{shard_idx:05d}-of-{grand_total_shards:05d}.parquet"
                            _write_shard(records[:args.shard_size], shard_path)
                            n_done_total = already_done + n_session_done
                            elapsed = time.time() - job_start_epoch
                            rate = n_session_done / max(time.time() - session_start_epoch, 1e-6)
                            remaining = grand_total - n_done_total
                            eta_s = remaining / rate if rate > 0 else float("inf")
                            eta_str = str(timedelta(seconds=int(eta_s))) if eta_s != float("inf") else "unknown"
                            print(
                                f"  [{datetime.now().strftime('%H:%M:%S')}] Wrote {shard_path.name} | "
                                f"{n_done_total}/{grand_total} ({100*n_done_total/grand_total:.1f}%) | "
                                f"rate {rate:.1f} struct/s | elapsed {timedelta(seconds=int(elapsed))} | "
                                f"ETA {eta_str}",
                                flush=True,
                            )
                            records = records[args.shard_size:]
                            shard_idx += 1

        if records:
            shard_path = output_dir / f"train-{shard_idx:05d}-of-{grand_total_shards:05d}.parquet"
            _write_shard(records, shard_path)
            print(f"  Wrote {shard_path.name}", flush=True)

    total_elapsed = time.time() - job_start_epoch
    print(
        f"\n  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
        f"Total elapsed: {timedelta(seconds=int(total_elapsed))} | "
        f"{already_done + n_session_done}/{grand_total} structures processed",
        flush=True,
    )

    _write_metadata(output_dir, args, grid)
    print(f"\nDone. Output in {output_dir}/", flush=True)
    print("Load with: from datasets import load_dataset; "
          f"ds = load_dataset('parquet', data_files='{output_dir}/train-*.parquet')")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Simulate powder XRD spectra — single structure or entire JARVIS dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--poscar", metavar="FILE", help="Path to POSCAR/CONTCAR file")
    src.add_argument("--cif", metavar="FILE", help="Path to CIF file")
    src.add_argument("--jid", metavar="JID", help="JARVIS JID (e.g. JVASP-1002)")
    src.add_argument("--dataset", metavar="KEY",
                     help="JARVIS figshare dataset key; simulate XRD for all entries "
                          "(e.g. alex_pbe_hull, dft_3d)")

    # Common simulation parameters
    p.add_argument("--db", default="dft_3d",
                   help="JARVIS figshare dataset name (used with --jid only)")
    p.add_argument("--wavelength", type=float, default=1.54184, metavar="ANGSTROM",
                   help="X-ray wavelength in Angstroms (Cu Kα = 1.54184)")
    p.add_argument("--two-theta-min", type=float, default=5.0, metavar="DEG")
    p.add_argument("--two-theta-max", type=float, default=90.0, metavar="DEG")
    p.add_argument("--step", type=float, default=0.01, metavar="DEG",
                   help="Grid step size in degrees")
    p.add_argument("--sigma", type=float, default=0.1, metavar="DEG",
                   help="Gaussian peak broadening sigma in degrees")

    # Single-structure output options
    p.add_argument("--output", "-o", metavar="FILE",
                   help="(single mode) Write spectrum to FILE instead of stdout")
    p.add_argument("--plot", metavar="FILE",
                   help="(single mode) Save matplotlib figure to FILE")

    # Dataset-mode output options
    p.add_argument("--output-dir", metavar="DIR",
                   help="(dataset mode) Directory for Parquet shards")
    p.add_argument("--shard-size", type=int, default=1000, metavar="N",
                   help="(dataset mode) Entries per Parquet shard")
    p.add_argument("--num-workers", type=int, default=1, metavar="N",
                   help="(dataset mode) Parallel worker processes")
    p.add_argument("--resume", action="store_true",
                   help="(dataset mode) Skip JIDs already written to --output-dir")

    return p.parse_args()


def main():
    args = parse_args()

    if args.dataset is not None:
        if not args.output_dir:
            print("error: --output-dir is required when using --dataset", file=sys.stderr)
            sys.exit(1)
        run_dataset_mode(args)
        return

    atoms = load_atoms(
        poscar=args.poscar,
        cif=args.cif,
        jid=args.jid,
        db=args.db,
    )

    grid, dense = simulate_dense_xrd(
        atoms,
        wavelength=args.wavelength,
        two_theta_min=args.two_theta_min,
        two_theta_max=args.two_theta_max,
        step=args.step,
        sigma=args.sigma,
    )

    write_spectrum(grid, dense, output=args.output)

    if args.plot:
        formula = atoms.composition.reduced_formula
        save_plot(grid, dense, args.plot, formula)


if __name__ == "__main__":
    main()
