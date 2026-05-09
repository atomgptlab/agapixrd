for D in ./slurm/alex*; do sbatch --partition=nvl "$D"; done
for D in ./slurm/rruff*; do sbatch --partition=nvl "$D"; done
