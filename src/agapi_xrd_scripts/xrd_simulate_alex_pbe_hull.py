import argparse
import numpy as np
from jarvis.analysis.diffraction.xrd import XRD
from jarvis.core.atoms import Atoms


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
    raise ValueError("One of --poscar, --cif, or --jid must be provided")


def simulate_dense_xrd(atoms, wavelength=1.54184, two_theta_min=5.0, two_theta_max=90.0, step=0.01, sigma=0.1):
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


def parse_args():
    p = argparse.ArgumentParser(
        description="Simulate a dense powder XRD spectrum from a crystal structure.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--poscar", metavar="FILE", help="Path to POSCAR/CONTCAR file")
    src.add_argument("--cif", metavar="FILE", help="Path to CIF file")
    src.add_argument("--jid", metavar="JID", help="JARVIS JID (e.g. JVASP-1002)")

    p.add_argument("--db", default="dft_3d", help="JARVIS figshare dataset name (used with --jid)")
    p.add_argument("--wavelength", type=float, default=1.54184, metavar="ANGSTROM",
                   help="X-ray wavelength in Angstroms (default: Cu Kα)")
    p.add_argument("--two-theta-min", type=float, default=5.0, metavar="DEG")
    p.add_argument("--two-theta-max", type=float, default=90.0, metavar="DEG")
    p.add_argument("--step", type=float, default=0.01, metavar="DEG",
                   help="Grid step size in degrees")
    p.add_argument("--sigma", type=float, default=0.1, metavar="DEG",
                   help="Gaussian peak broadening sigma in degrees")
    p.add_argument("--output", "-o", metavar="FILE",
                   help="Write spectrum to FILE instead of stdout")
    p.add_argument("--plot", metavar="FILE",
                   help="Save a matplotlib figure of the spectrum to FILE")

    return p.parse_args()


def main():
    args = parse_args()

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
