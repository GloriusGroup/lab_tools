"""Reference electrode potential conversions."""

import numpy as np

# Potentials in V vs SHE at 25 degC
REFERENCE_POTENTIALS = {
    "Ag/AgCl": 0.210,       # 3 M KCl (most common laboratory reference)
    "Ag/AgCl_sat": 0.197,   # saturated KCl
    "SCE": 0.241,           # saturated calomel electrode
    "SHE": 0.000,           # standard hydrogen electrode
    "NHE": 0.000,           # normal hydrogen electrode (= SHE)
}


def list_references():
    """Return supported reference electrode names."""
    return list(REFERENCE_POTENTIALS.keys())


def convert_potential(E, from_ref, to_ref):
    """Convert a single potential value between reference electrodes.

    E(vs to_ref) = E(vs from_ref) + E_ref(from_ref vs SHE) - E_ref(to_ref vs SHE)
    """
    if from_ref not in REFERENCE_POTENTIALS:
        raise ValueError(
            f"Unknown reference electrode: '{from_ref}'. "
            f"Supported: {list_references()}"
        )
    if to_ref not in REFERENCE_POTENTIALS:
        raise ValueError(
            f"Unknown reference electrode: '{to_ref}'. "
            f"Supported: {list_references()}"
        )
    offset = REFERENCE_POTENTIALS[from_ref] - REFERENCE_POTENTIALS[to_ref]
    return E + offset


def convert_potential_array(potentials, from_ref, to_ref):
    """Convert an array of potentials between reference electrodes."""
    offset = REFERENCE_POTENTIALS[from_ref] - REFERENCE_POTENTIALS[to_ref]
    return np.asarray(potentials) + offset
