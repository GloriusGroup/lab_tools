"""Parse CV data files (semicolon-delimited CSV with quoted values)."""

import numpy as np
from pathlib import Path
from typing import Dict, Any


def parse_cv_file(filepath):
    """Parse a CV CSV file.

    Expected format:
        Line 1: metadata/identifier string
        Line 2: empty
        Line 3: "Potential (V)";"Current (uA)"
        Line 4+: "value";"value"

    Parameters
    ----------
    filepath : str or Path
        Path to the CSV file.

    Returns
    -------
    metadata : str
        First-line identifier.
    potential : np.ndarray
        Potential values in V.
    current : np.ndarray
        Current values in uA.
    """
    filepath = Path(filepath)
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    metadata = lines[0].strip().strip('"').rstrip("\r\n")

    potential = []
    current = []
    for line in lines[3:]:
        line = line.strip()
        if not line:
            continue
        parts = line.split(";")
        if len(parts) < 2:
            continue
        try:
            v = float(parts[0].strip('"'))
            i = float(parts[1].strip('"'))
            potential.append(v)
            current.append(i)
        except ValueError:
            continue

    return metadata, np.array(potential), np.array(current)


def extract_metadata(filepath, potential, current):
    """Derive measurement metadata from the file header and data arrays.

    Parameters
    ----------
    filepath : str or Path
        Original file path.
    potential, current : np.ndarray
        Parsed data arrays.

    Returns
    -------
    dict
        Keys: identifier, filename, n_points, potential_unit, current_unit,
              start_potential, end_potential, min_potential, max_potential,
              potential_step, vertex_potentials, n_sweeps.
    """
    filepath = Path(filepath)
    with open(filepath, "r", encoding="utf-8") as f:
        first_line = f.readline().strip().strip('"').rstrip("\r\n")

    # Parse header row for units
    with open(filepath, "r", encoding="utf-8") as f:
        lines = [f.readline() for _ in range(3)]
    header = lines[2].strip() if len(lines) >= 3 else ""
    pot_unit = "V"
    cur_unit = "uA"
    if header:
        parts = header.split(";")
        if len(parts) >= 1:
            h = parts[0].strip('"')
            if "(" in h and ")" in h:
                pot_unit = h.split("(")[1].split(")")[0]
        if len(parts) >= 2:
            h = parts[1].strip('"')
            if "(" in h and ")" in h:
                cur_unit = h.split("(")[1].split(")")[0]

    # Potential step
    if len(potential) >= 2:
        steps = np.abs(np.diff(potential))
        step = float(np.median(steps[steps > 0])) if np.any(steps > 0) else 0.0
    else:
        step = 0.0

    # Vertex potentials (where sweep direction changes)
    vertices = []
    if len(potential) >= 3:
        dp = np.diff(potential)
        sign = np.sign(dp)
        for i in range(1, len(sign)):
            if sign[i] == 0:
                sign[i] = sign[i - 1]
        sign_changes = np.where(np.diff(sign) != 0)[0]
        for idx in sign_changes:
            vertices.append(float(potential[idx + 1]))

    # Number of half-sweeps (segments between vertices)
    n_sweeps = len(vertices) + 1

    return dict(
        identifier=first_line,
        filename=filepath.name,
        n_points=len(potential),
        potential_unit=pot_unit,
        current_unit=cur_unit,
        start_potential=float(potential[0]) if len(potential) else None,
        end_potential=float(potential[-1]) if len(potential) else None,
        min_potential=float(np.min(potential)) if len(potential) else None,
        max_potential=float(np.max(potential)) if len(potential) else None,
        potential_step_V=step,
        vertex_potentials=vertices,
        n_sweeps=n_sweeps,
    )
