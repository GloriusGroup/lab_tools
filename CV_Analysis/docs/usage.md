# CV Analyzer -- Usage Guide

## Installation

From the repository root (`lab_tools/`):

```bash
pip install -e ./CV_Analysis
```

Or install all lab tools at once:

```bash
pip install -e .
```

### Dependencies

* Python >= 3.9
* numpy, scipy, matplotlib, openpyxl

All are installed automatically via `pip install`.

---

## Quick Start

```bash
# Analyse a single CV file (default: measured vs Ag/AgCl, output vs SCE)
cv-analyzer measurement.csv

# Equivalent via python -m
python -m cv_analyzer measurement.csv
```

This produces two files next to the input CSV:

| File | Contents |
|------|----------|
| `measurement_cv_plot.png` | Annotated CV plot |
| `measurement_cv_data.xlsx` | Two-sheet Excel workbook |

---

## CLI Options

```
cv-analyzer [OPTIONS] FILE [FILE ...]
```

| Option | Default | Description |
|--------|---------|-------------|
| `FILE` | *(required)* | One or more `.csv` files **or** a directory (all `.csv` inside are processed). |
| `--blank PATH` | *none* | Blank measurement CSV for background subtraction. |
| `--reference-electrode REF` | `Ag/AgCl` | Reference electrode used during the measurement. |
| `--output-reference REF` | `SCE` | Reference electrode for all reported potentials. |
| `--output-dir DIR` | same as input | Directory for output files. |
| `--version` | | Print version and exit. |

### Supported reference electrodes

`Ag/AgCl` (3 M KCl), `Ag/AgCl_sat` (sat. KCl), `SCE`, `SHE`, `NHE`

---

## Batch Mode

Pass multiple files or a directory:

```bash
# Multiple files
cv-analyzer file1.csv file2.csv file3.csv

# Entire directory
cv-analyzer path/to/csv_folder/

# With shared blank
cv-analyzer path/to/csv_folder/ --blank blank.csv --output-dir results/
```

Each file produces its own plot and Excel workbook.

---

## What Gets Extracted

### For every detected peak

| Value | Symbol | Description |
|-------|--------|-------------|
| Onset potential | E_onset | Tangent-intersection of baseline and rising slope |
| Peak potential | E_p | Potential at maximum (oxidation) or minimum (reduction) current |
| Peak current | I_p | Current at the peak (raw value) |
| Half-peak potential | E_p/2 | Potential where baseline-corrected current = I_p/2 |
| \|E_p - E_p/2\| | | Diagnostic for transfer coefficient |

### Reversibility assessment

If a matching anodic/cathodic peak pair is found with
|E_p,a - E_p,c| < 200 mV, the system is classified as **(quasi-)reversible** and the following are reported:

| Value | Symbol | Description |
|-------|--------|-------------|
| Standard potential | E^0 | (E_p,a + E_p,c) / 2 |
| Peak separation | dE_p | \|E_p,a - E_p,c\| |

If no matching pair exists, the system is classified as **irreversible**.

---

## Output Files

### Plot (`*_cv_plot.png`)

* Main CV trace (dark line) and optional blank (grey).
* Vertical annotation lines for each peak:
  * **dash-dot (-.-)** -- onset potential
  * **dashed (--)** -- peak potential
  * **dotted (:)** -- half-peak potential
* If reversible: **solid line** at E^0.
* Colour coding: red = oxidation, green = reduction, orange = E^0.
* Info box (top-left) with numeric values.
* Legend (right side, outside plot) mapping every line to its value.

### Excel workbook (`*_cv_data.xlsx`)

**Sheet 1 -- "Raw Data"**

| Potential (V vs REF) | Current (uA) |
|----------------------|--------------|
| ... | ... |

**Sheet 2 -- "Analysis"**

Contains three sections:

1. **Measurement Metadata** -- identifier, source file, data points,
   potential/current units, start/end/min/max potential, step size,
   vertex potentials, number of sweeps, output reference electrode.
2. **Reversibility Assessment** -- classification and (if reversible)
   E^0 and dE_p.
3. **Detected Peaks** -- table with onset, E_p, I_p, E_p/2, |E_p - E_p/2|
   for every peak.

---

## Input File Format

The analyzer expects semicolon-delimited CSV files with quoted values:

```
"identifier_string"
""
"Potential (V)";"Current (uA)"
"0.100";"0.01185"
"0.105";"0.02445"
...
```

* **Line 1**: identifier / metadata string
* **Line 2**: empty
* **Line 3**: column headers
* **Line 4+**: data rows

---

## Analysis Method Details

### Peak detection

1. The CV trace is split into monotonic segments at vertex potentials.
2. Each segment is smoothed with a Savitzky-Golay filter (adaptive window).
3. `scipy.signal.find_peaks` detects local maxima (oxidation) or minima (reduction).
4. Peaks are filtered by prominence (>2 % of segment and global current range) and by position (must sit in the correct half of the segment's current range to avoid diffusion-tail artefacts).

### Half-peak potential (E_p/2)

Determined via `scipy.signal.peak_widths` at `rel_height=0.5`. This finds the potential on the ascending side of the peak where the current equals the baseline level plus half the peak prominence -- i.e. the standard electrochemical definition of E_p/2.

### Onset potential (E_onset)

Tangent-intersection method:
1. A baseline tangent is fitted to the first 20 % of the pre-peak region.
2. A slope tangent is fitted at the inflection point (steepest rise) of the peak.
3. Their intersection defines E_onset.

If the intersection falls outside the valid range, a 10 %-of-peak-current threshold crossing is used as fallback.

### Reference electrode conversion

```
E(vs target) = E(vs source) + E_ref(source vs SHE) - E_ref(target vs SHE)
```
