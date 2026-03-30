"""Generate annotated CV plots."""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path

from cv_analyzer.analysis import CVResult

# Colour scheme
_CV_COLOR = "#1a1a2e"        # dark navy for the CV trace
_BLANK_COLOR = "#b0b0b0"     # grey for blank
_ANODIC_COLOR = "#c0392b"    # red for oxidation
_CATHODIC_COLOR = "#27ae60"  # green for reduction
_E0_COLOR = "#e67e22"        # orange for standard potential


def plot_cv(potential, current, result: CVResult,
            blank_potential=None, blank_current=None,
            output_path=None, ref_label="Ag/AgCl",
            title=None):
    """Create an annotated CV plot and save to file.

    Parameters
    ----------
    potential, current : array-like
        CV data (already converted to output reference if needed).
    result : CVResult
        Analysis results (potentials already converted).
    blank_potential, blank_current : array-like, optional
        Blank data for overlay.
    output_path : str or Path, optional
        Save path. If None, defaults to cv_plot.png in cwd.
    ref_label : str
        Label for reference electrode on x-axis.
    title : str, optional
        Plot title.
    """
    # Use a wider figure to make room for the legend on the right
    fig, ax = plt.subplots(figsize=(10, 5.5))

    # --- Data traces -------------------------------------------------------
    if blank_potential is not None and blank_current is not None:
        ax.plot(blank_potential, blank_current,
                color=_BLANK_COLOR, linewidth=1, label="Blank", zorder=1)

    ax.plot(potential, current, color=_CV_COLOR, linewidth=1.3,
            label="CV", zorder=2)

    # --- Peak annotations --------------------------------------------------
    info_lines = []          # for the value text box
    legend_handles = []      # custom legend entries for annotation lines
    legend_labels = []

    ox_count = 0
    red_count = 0

    for peak in result.peaks:
        is_anodic = peak.peak_type == "anodic"
        color = _ANODIC_COLOR if is_anodic else _CATHODIC_COLOR

        if is_anodic:
            ox_count += 1
            tag = f"Ox{ox_count}"
        else:
            red_count += 1
            tag = f"Red{red_count}"

        # Vertical dash-dot line at E_onset
        ax.axvline(
            peak.onset_potential, color=color, linestyle="-.",
            linewidth=1, alpha=0.7, zorder=3,
        )
        # Vertical dashed line at Ep
        ax.axvline(
            peak.potential, color=color, linestyle="--",
            linewidth=1, alpha=0.85, zorder=3,
        )
        # Vertical dotted line at Ep/2
        ax.axvline(
            peak.half_peak_potential, color=color, linestyle=":",
            linewidth=1, alpha=0.75, zorder=3,
        )

        # Marker at peak apex
        ax.plot(peak.potential, peak.current, "o",
                color=color, markersize=6, markeredgecolor="white",
                markeredgewidth=0.6, zorder=5)

        # Legend entries
        legend_handles.append(Line2D([], [], color=color, linestyle="-.",
                                     linewidth=1))
        legend_labels.append(f"{tag}  $E_{{onset}}$ = {peak.onset_potential:+.3f} V")

        legend_handles.append(Line2D([], [], color=color, linestyle="--",
                                     linewidth=1))
        legend_labels.append(f"{tag}  $E_p$ = {peak.potential:+.3f} V")

        legend_handles.append(Line2D([], [], color=color, linestyle=":",
                                     linewidth=1))
        legend_labels.append(f"{tag}  $E_{{p/2}}$ = {peak.half_peak_potential:+.3f} V")

        # Info-box text
        delta_mv = abs(peak.potential - peak.half_peak_potential) * 1000
        info_lines.append(f"{tag}:  $E_{{onset}}$ = {peak.onset_potential:+.3f} V")
        info_lines.append(f"       $E_p$ = {peak.potential:+.3f} V")
        info_lines.append(f"       $E_{{p/2}}$ = {peak.half_peak_potential:+.3f} V")
        info_lines.append(f"       $|E_p - E_{{p/2}}|$ = {delta_mv:.0f} mV")

    # Standard potential (reversible)
    if result.is_reversible and result.standard_potential is not None:
        ax.axvline(result.standard_potential, color=_E0_COLOR,
                   linestyle="-", linewidth=1.8, alpha=0.9, zorder=4)
        legend_handles.append(Line2D([], [], color=_E0_COLOR, linestyle="-",
                                     linewidth=1.8))
        legend_labels.append(f"$E^0$ = {result.standard_potential:+.3f} V")

        info_lines.append("")
        info_lines.append(f"$E^0$ = {result.standard_potential:+.3f} V")
        sep_mv = result.peak_separation * 1000
        info_lines.append(f"$\\Delta E_p$ = {sep_mv:.0f} mV")

    # --- Info text box (inside plot, top-left) ----------------------------
    if info_lines:
        text = "\n".join(info_lines)
        bbox_props = dict(
            boxstyle="round,pad=0.45", facecolor="ivory",
            edgecolor="0.6", alpha=0.88,
        )
        ax.text(
            0.02, 0.97, text, transform=ax.transAxes,
            fontsize=7.5, verticalalignment="top",
            fontfamily="monospace", bbox=bbox_props, zorder=6,
        )

    # --- Legend: placed OUTSIDE the plot to avoid overlap ------------------
    data_handles, data_labels = ax.get_legend_handles_labels()
    all_handles = data_handles + legend_handles
    all_labels = data_labels + legend_labels

    if all_handles:
        ax.legend(
            all_handles, all_labels,
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            fontsize=7.5, framealpha=0.9,
            edgecolor="0.7", borderaxespad=0,
        )

    # --- Axis styling ------------------------------------------------------
    ax.set_xlabel(f"Potential (V vs {ref_label})", fontsize=10)
    ax.set_ylabel("Current ($\\mu$A)", fontsize=10)
    ax.tick_params(labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if title:
        ax.set_title(title, fontsize=11, fontweight="semibold")

    # Shrink axes to make room for the external legend
    fig.subplots_adjust(right=0.72)

    # --- Save --------------------------------------------------------------
    if output_path is None:
        output_path = Path("cv_plot.png")
    fig.savefig(str(output_path), dpi=200, bbox_inches="tight")
    plt.close(fig)
    return Path(output_path)
