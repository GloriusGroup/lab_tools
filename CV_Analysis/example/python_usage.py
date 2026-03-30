"""Example: using cv_analyzer from Python."""

from pathlib import Path
from cv_analyzer.parser import parse_cv_file, extract_metadata
from cv_analyzer.analysis import analyze_cv
from cv_analyzer.reference import convert_potential_array, convert_potential
from cv_analyzer.plotting import plot_cv
from cv_analyzer.exporter import export_to_excel

# ---------------------------------------------------------------------------
# 1. Parse measurement and (optional) blank
# ---------------------------------------------------------------------------
data_dir = Path(__file__).parent / "irreversible"

metadata, potential, current = parse_cv_file(data_dir / "thioamide_2_1.csv")
_, blank_pot, blank_cur = parse_cv_file(data_dir / "blank1.csv")

# ---------------------------------------------------------------------------
# 2. Run analysis (blank subtraction is optional)
# ---------------------------------------------------------------------------
result = analyze_cv(potential, current, blank_pot, blank_cur)

# ---------------------------------------------------------------------------
# 3. Inspect results
# ---------------------------------------------------------------------------
for i, peak in enumerate(result.peaks, 1):
    label = "Oxidation" if peak.peak_type == "anodic" else "Reduction"
    print(f"{label} Peak {i}:")
    print(f"  Onset potential  = {peak.onset_potential:.4f} V")
    print(f"  Peak potential   = {peak.potential:.4f} V")
    print(f"  Peak current     = {peak.current:.4f} uA")
    print(f"  Half-peak pot.   = {peak.half_peak_potential:.4f} V")
    print()

print(f"Reversible: {result.is_reversible}")
if result.is_reversible:
    print(f"  Standard potential (E0) = {result.standard_potential:.4f} V")
    print(f"  Peak separation (dEp)   = {result.peak_separation * 1000:.1f} mV")

# ---------------------------------------------------------------------------
# 4. Convert to a different reference electrode
# ---------------------------------------------------------------------------
from_ref = "Ag/AgCl"
to_ref = "SCE"

out_potential = convert_potential_array(potential, from_ref, to_ref)

# Convert peak potentials for display
for pk in result.peaks:
    pk.potential = convert_potential(pk.potential, from_ref, to_ref)
    pk.half_peak_potential = convert_potential(pk.half_peak_potential, from_ref, to_ref)
    pk.onset_potential = convert_potential(pk.onset_potential, from_ref, to_ref)
if result.standard_potential is not None:
    result.standard_potential = convert_potential(result.standard_potential, from_ref, to_ref)

# ---------------------------------------------------------------------------
# 5. Generate plot and Excel export
# ---------------------------------------------------------------------------
out_dir = Path(__file__).parent / "output" / "irreversible"
out_dir.mkdir(parents=True, exist_ok=True)

file_metadata = extract_metadata(data_dir / "thioamide_2_1.csv", potential, current)

plot_cv(
    out_potential, current, result,
    blank_potential=convert_potential_array(blank_pot, from_ref, to_ref),
    blank_current=blank_cur,
    output_path=out_dir / "example_plot.png",
    ref_label=to_ref,
    title="Example CV Analysis",
)
print(f"\nPlot saved to {out_dir / 'example_plot.png'}")

export_to_excel(
    out_potential, current, result, file_metadata,
    output_path=out_dir / "example_data.xlsx",
    ref_label=to_ref,
)
print(f"Excel saved to {out_dir / 'example_data.xlsx'}")
