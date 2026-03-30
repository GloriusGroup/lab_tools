"""Command-line interface for CV Analyzer."""

import argparse
import sys
from pathlib import Path
from typing import List

from cv_analyzer import __version__
from cv_analyzer.parser import parse_cv_file, extract_metadata
from cv_analyzer.analysis import analyze_cv, CVResult
from cv_analyzer.reference import (
    convert_potential,
    convert_potential_array,
    list_references,
)
from cv_analyzer.plotting import plot_cv
from cv_analyzer.exporter import export_to_excel


def _build_parser():
    parser = argparse.ArgumentParser(
        prog="cv-analyzer",
        description="Cyclic Voltammetry Analysis Tool",
    )
    parser.add_argument(
        "files", nargs="+", type=Path,
        help="One or more CV measurement CSV files to analyze.",
    )
    parser.add_argument(
        "--blank", type=Path, default=None,
        help="Optional blank measurement CSV for background subtraction.",
    )
    parser.add_argument(
        "--reference-electrode", default="Ag/AgCl",
        metavar="REF",
        help=(
            "Reference electrode used during measurement "
            f"(default: Ag/AgCl). Supported: {', '.join(list_references())}"
        ),
    )
    parser.add_argument(
        "--output-reference", default="SCE",
        metavar="REF",
        help=(
            "Reference electrode for reported potentials "
            f"(default: SCE). Supported: {', '.join(list_references())}"
        ),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Directory for output files (default: same as input file).",
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}",
    )
    return parser


def _resolve_output_dir(input_path: Path, output_dir_arg):
    if output_dir_arg is not None:
        d = Path(output_dir_arg)
    else:
        d = input_path.parent
    d.mkdir(parents=True, exist_ok=True)
    return d


def _print_header(text, char="="):
    print(text)
    print(char * len(text))


def _process_single(filepath: Path, args, batch_index=None, batch_total=None):
    """Analyse one CV file and write outputs. Returns True on success."""
    prefix = ""
    if batch_total and batch_total > 1:
        prefix = f"[{batch_index}/{batch_total}] "

    print(f"\n{prefix}{filepath.name}")
    print("-" * (len(prefix) + len(filepath.name)))

    # --- parse measurement -------------------------------------------------
    try:
        meta, potential, current = parse_cv_file(filepath)
    except Exception as exc:
        print(f"  ERROR parsing file: {exc}")
        return False

    # --- parse blank -------------------------------------------------------
    blank_pot, blank_cur = None, None
    if args.blank is not None:
        try:
            _, blank_pot, blank_cur = parse_cv_file(args.blank)
        except Exception as exc:
            print(f"  WARNING: could not parse blank ({exc}), proceeding without it")

    # --- extract metadata --------------------------------------------------
    file_metadata = extract_metadata(filepath, potential, current)

    # --- analyse -----------------------------------------------------------
    result = analyze_cv(potential, current, blank_pot, blank_cur)

    # --- convert potentials to output reference ----------------------------
    from_ref = args.reference_electrode
    to_ref = args.output_reference
    need_convert = from_ref != to_ref

    def conv(v):
        return convert_potential(v, from_ref, to_ref) if need_convert else v

    out_potential = (
        convert_potential_array(potential, from_ref, to_ref)
        if need_convert else potential
    )

    # Convert peak potentials in-place for display / plotting
    for pk in result.peaks:
        pk.potential = conv(pk.potential)
        pk.half_peak_potential = conv(pk.half_peak_potential)
        pk.onset_potential = conv(pk.onset_potential)
    if result.standard_potential is not None:
        result.standard_potential = conv(result.standard_potential)
    for pair in result.matched_pairs:
        pair["standard_potential"] = conv(pair["standard_potential"])

    # Also convert blank for plotting
    out_blank_pot = None
    out_blank_cur = blank_cur
    if blank_pot is not None:
        out_blank_pot = (
            convert_potential_array(blank_pot, from_ref, to_ref)
            if need_convert else blank_pot
        )

    # --- print results -----------------------------------------------------
    ref_note = f" (converted from {from_ref})" if need_convert else ""
    print(f"  Reference: {to_ref}{ref_note}")
    print(f"  Data points: {len(potential)}")
    print()

    if not result.peaks:
        print("  No peaks detected.")
    else:
        for i, pk in enumerate(result.peaks, 1):
            label = "Oxidation" if pk.peak_type == "anodic" else "Reduction"
            print(f"  {label} Peak {i}:")
            print(f"    Onset potential (E_onset) = {pk.onset_potential:+.4f} V vs {to_ref}")
            print(f"    Peak potential     (Ep)  = {pk.potential:+.4f} V vs {to_ref}")
            print(f"    Peak current       (Ip)  = {pk.current:.4f} uA")
            print(f"    Half-peak pot.   (Ep/2)  = {pk.half_peak_potential:+.4f} V vs {to_ref}")
            delta = abs(pk.potential - pk.half_peak_potential) * 1000
            print(f"    |Ep - Ep/2|              = {delta:.1f} mV")
            print()

    if result.is_reversible:
        print(f"  Reversibility: REVERSIBLE (quasi-reversible)")
        print(f"    Standard potential (E0) = {result.standard_potential:+.4f} V vs {to_ref}")
        print(f"    Peak separation (dEp)   = {result.peak_separation * 1000:.1f} mV")
    else:
        print("  Reversibility: IRREVERSIBLE (no matching cathodic/anodic pair)")
    print()

    # --- output files ------------------------------------------------------
    out_dir = _resolve_output_dir(filepath, args.output_dir)
    stem = filepath.stem

    plot_path = out_dir / f"{stem}_cv_plot.png"
    plot_cv(
        out_potential, current, result,
        blank_potential=out_blank_pot,
        blank_current=out_blank_cur,
        output_path=plot_path,
        ref_label=to_ref,
        title=f"CV: {filepath.name}",
    )
    print(f"  Plot:  {plot_path}")

    xlsx_path = out_dir / f"{stem}_cv_data.xlsx"
    export_to_excel(out_potential, current, result, file_metadata,
                    xlsx_path, ref_label=to_ref)
    print(f"  Excel: {xlsx_path}")

    return True


def main(argv=None):
    parser = _build_parser()
    args = parser.parse_args(argv)

    # Expand directories into contained CSV files
    files: List[Path] = []
    for p in args.files:
        if p.is_dir():
            files.extend(sorted(p.glob("*.csv")))
        else:
            files.append(p)

    if not files:
        parser.error("No CSV files found.")

    _print_header(f"CV Analyzer v{__version__}")

    n = len(files)
    if n > 1:
        print(f"Batch mode: processing {n} files ...")

    ok = 0
    for i, f in enumerate(files, 1):
        if _process_single(f, args, batch_index=i, batch_total=n):
            ok += 1

    if n > 1:
        print(f"\nBatch complete: {ok}/{n} files processed successfully.")
