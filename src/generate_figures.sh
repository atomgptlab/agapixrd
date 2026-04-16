echo "plot_refinement_mae.py"
python src/plotting_scripts/plot_refinement_mae.py
echo "plot_refinement_jsd_18panel.py"
python src/plotting_scripts/plot_refinement_jsd_18panel.py
echo "plot_match_rate_crystal_systems.py"
python src/plotting_scripts/plot_match_rate_crystal_systems.py --max-val 10
echo "plot_match_rate.py"
python src/plotting_scripts/plot_match_rate.py
echo "plot_refinement_mae_bmgn_vs_bmgn_alignnff.py"
python src/plotting_scripts/plot_refinement_mae_bmgn_vs_bmgn_alignnff.py
echo "stoich"
bash src/plotting_scripts/stoich.sh
echo "end script"

