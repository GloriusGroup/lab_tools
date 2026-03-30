"""CV peak analysis: detection, half-peak potentials, onset, reversibility."""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional
from scipy.signal import savgol_filter, find_peaks
from scipy.interpolate import interp1d


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class CVPeak:
    """A single detected CV peak."""

    peak_type: str              # "anodic" or "cathodic"
    potential: float            # Ep  (V)
    current: float              # Ip  (uA, raw value at peak)
    net_current: float          # Ip above/below baseline (prominence, uA)
    half_peak_potential: float  # Ep/2  (V)
    onset_potential: float      # E_onset  (V)
    segment_index: int          # which scan segment the peak belongs to


@dataclass
class CVResult:
    """Complete CV analysis result."""

    peaks: List[CVPeak]
    is_reversible: bool
    standard_potential: Optional[float] = None   # E0  (V), if reversible
    peak_separation: Optional[float] = None      # |Ep,a - Ep,c|  (V)
    matched_pairs: List[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _adaptive_smooth(current, n_points):
    """Savitzky-Golay smooth with adaptive window."""
    window = min(31, max(5, n_points // 10))
    if window % 2 == 0:
        window += 1
    if n_points < window:
        return current.copy()
    return savgol_filter(current, window, polyorder=3)


def find_vertex_indices(potential):
    """Return indices where the sweep direction reverses."""
    dp = np.diff(potential)
    step = np.median(np.abs(dp[dp != 0]))
    dp_clean = np.where(np.abs(dp) < step * 0.1, 0, dp)
    sign = np.sign(dp_clean)
    for i in range(1, len(sign)):
        if sign[i] == 0:
            sign[i] = sign[i - 1]
    changes = np.where(np.diff(sign) != 0)[0]
    return changes + 1


def split_segments(potential, current):
    """Split a CV trace into monotonic-potential segments."""
    vertices = find_vertex_indices(potential)
    bounds = np.concatenate(([0], vertices, [len(potential) - 1]))
    segments = []
    for i in range(len(bounds) - 1):
        s, e = int(bounds[i]), int(bounds[i + 1]) + 1
        if e - s < 10:
            continue
        seg_pot = potential[s:e]
        seg_cur = current[s:e]
        direction = "anodic" if seg_pot[-1] > seg_pot[0] else "cathodic"
        segments.append(dict(
            potential=seg_pot,
            current=seg_cur,
            start_idx=s,
            end_idx=e,
            direction=direction,
        ))
    return segments


def subtract_blank(potential, current, blank_potential, blank_current):
    """Subtract interpolated blank from measurement."""
    interp_fn = interp1d(
        blank_potential, blank_current,
        kind="linear", fill_value="extrapolate",
    )
    return current - interp_fn(potential)


# ---------------------------------------------------------------------------
# Onset potential
# ---------------------------------------------------------------------------

def _compute_onset_potential(potential, smoothed, peak_idx):
    """Compute onset potential via tangent-intersection method.

    1. Fit a baseline tangent to the flat region before the peak.
    2. Fit a slope tangent to the steepest rising part of the peak.
    3. The intersection of these two lines is the onset potential.
    """
    # We only look at the ascending side (indices 0 .. peak_idx)
    pre = smoothed[:peak_idx + 1]
    pot = potential[:peak_idx + 1]

    if len(pre) < 10:
        return potential[0]  # fallback

    # --- baseline tangent: fit to the first 20% of the pre-peak region -----
    n_base = max(int(len(pre) * 0.20), 5)
    base_pot = pot[:n_base]
    base_cur = pre[:n_base]
    base_coeffs = np.polyfit(base_pot, base_cur, 1)  # [slope, intercept]

    # --- slope tangent: fit at the inflection point of the rising edge -----
    # Compute first derivative dI/dE
    dI = np.gradient(pre, pot)
    # Smooth the derivative
    dI_smooth = _adaptive_smooth(dI, len(dI))

    # The steepest point is where |dI/dE| is maximum in the rising region
    # (use second half of the pre-peak data to avoid baseline region)
    half = max(len(dI_smooth) // 2, 1)
    search_region = dI_smooth[half:]
    inflection_local = np.argmax(np.abs(search_region))
    inflection_idx = half + inflection_local

    # Fit a tangent at the inflection point using ±5 surrounding points
    margin = min(5, inflection_idx, len(pre) - inflection_idx - 1)
    if margin < 2:
        return potential[0]  # fallback

    tang_pot = pot[inflection_idx - margin : inflection_idx + margin + 1]
    tang_cur = pre[inflection_idx - margin : inflection_idx + margin + 1]
    tang_coeffs = np.polyfit(tang_pot, tang_cur, 1)

    # --- intersection of the two lines: a1*E + b1 = a2*E + b2 -------------
    a1, b1 = base_coeffs
    a2, b2 = tang_coeffs
    if abs(a2 - a1) < 1e-12:
        return potential[0]  # parallel lines, fallback

    e_onset = (b1 - b2) / (a2 - a1)

    # Sanity: onset must lie between segment start and peak potential
    e_min = min(pot[0], pot[-1])
    e_max = max(pot[0], pot[-1])
    if not (e_min <= e_onset <= e_max):
        # Fallback: use the 10% net-current crossing
        net = pre - np.polyval(base_coeffs, pot)
        threshold = net[peak_idx] * 0.10
        crossings = np.where(net >= threshold)[0]
        if len(crossings):
            e_onset = pot[crossings[0]]
        else:
            e_onset = pot[0]

    return float(e_onset)


# ---------------------------------------------------------------------------
# Per-segment peak detection
# ---------------------------------------------------------------------------

def _fit_pre_peak_baseline(potential, smoothed, peak_idx, frac=0.20):
    """Fit a linear baseline to the early part of the pre-peak region.

    Returns (slope, intercept) coefficients for np.polyval.
    """
    n_base = max(int(peak_idx * frac), 5)
    if n_base > peak_idx:
        n_base = max(peak_idx, 2)
    return np.polyfit(potential[:n_base], smoothed[:n_base], 1)


def _compute_half_peak_potential(potential, smoothed, peak_idx):
    """Compute Ep/2 using a proper linear baseline.

    1. Fit a linear baseline to the first 20 % of the pre-peak region.
    2. Extrapolate it under the peak to get net current.
    3. Ip_net = smoothed[peak] - baseline[peak].
    4. Find where net current = Ip_net / 2 on the ascending side.
    """
    if peak_idx < 5:
        return potential[peak_idx]

    baseline_coeffs = _fit_pre_peak_baseline(potential, smoothed, peak_idx)
    baseline = np.polyval(baseline_coeffs, potential[:peak_idx + 1])
    net = smoothed[:peak_idx + 1] - baseline

    ip_net = net[peak_idx]
    if abs(ip_net) < 1e-12:
        return potential[peak_idx]

    half_level = ip_net / 2.0

    # Find the last crossing of net == half_level on the ascending side
    diff = net - half_level
    crossings = np.where(np.diff(np.sign(diff)))[0]
    if len(crossings) == 0:
        # Fallback: closest point
        closest = np.argmin(np.abs(diff))
        return float(potential[closest])

    # Use the last crossing (closest to peak)
    ci = crossings[-1]
    # Linear interpolation between ci and ci+1
    d0, d1 = diff[ci], diff[ci + 1]
    if abs(d1 - d0) > 1e-15:
        frac = -d0 / (d1 - d0)
    else:
        frac = 0.5
    ep2 = potential[ci] + frac * (potential[ci + 1] - potential[ci])
    return float(ep2)


def _detect_peaks_in_segment(potential, current, direction,
                             min_prominence_frac=0.02):
    """Detect peaks in one monotonic segment.

    For anodic segments (E increasing) we look for current maxima (oxidation).
    For cathodic segments (E decreasing) we look for current minima (reduction).

    Returns list of dicts with peak info.
    """
    n = len(current)
    smoothed = _adaptive_smooth(current, n)
    current_range = np.ptp(smoothed)
    if current_range == 0:
        return []

    min_prom = current_range * min_prominence_frac
    min_dist = max(n // 20, 3)

    results = []

    # Midpoint filter: anodic peaks must be above, cathodic below.
    midpoint = (np.max(smoothed) + np.min(smoothed)) / 2

    if direction == "anodic":
        idx, props = find_peaks(smoothed, prominence=min_prom, distance=min_dist)
        if len(idx) == 0:
            return []
        for i, pk in enumerate(idx):
            if pk < n * 0.03 or pk > n * 0.97:
                continue
            if smoothed[pk] < midpoint:
                continue
            ep2 = _compute_half_peak_potential(potential, smoothed, pk)
            onset = _compute_onset_potential(potential, smoothed, pk)
            # Net current from proper baseline
            base_coeffs = _fit_pre_peak_baseline(potential, smoothed, pk)
            ip_net = smoothed[pk] - np.polyval(base_coeffs, potential[pk])
            results.append(dict(
                peak_type="anodic",
                potential=potential[pk],
                current=current[pk],
                net_current=abs(ip_net),
                half_peak_potential=ep2,
                onset_potential=onset,
            ))
    else:
        # Cathodic: find minima by inverting signal
        neg_smoothed = -smoothed
        idx, props = find_peaks(neg_smoothed, prominence=min_prom, distance=min_dist)
        if len(idx) == 0:
            return []
        for i, pk in enumerate(idx):
            if pk < n * 0.03 or pk > n * 0.97:
                continue
            if smoothed[pk] > midpoint:
                continue
            # Use -smoothed so the same baseline logic applies (looking at a maximum)
            ep2 = _compute_half_peak_potential(potential, neg_smoothed, pk)
            onset = _compute_onset_potential(potential, neg_smoothed, pk)
            base_coeffs = _fit_pre_peak_baseline(potential, neg_smoothed, pk)
            ip_net = neg_smoothed[pk] - np.polyval(base_coeffs, potential[pk])
            results.append(dict(
                peak_type="cathodic",
                potential=potential[pk],
                current=current[pk],
                net_current=abs(ip_net),
                half_peak_potential=ep2,
                onset_potential=onset,
            ))

    return results


# ---------------------------------------------------------------------------
# Reversibility matching
# ---------------------------------------------------------------------------

def _match_reversible_pairs(peaks, max_separation=0.200):
    """Try to match anodic/cathodic peak pairs.

    A pair is considered (quasi-)reversible if the peak separation
    |Ep,a - Ep,c| < max_separation (default 200 mV).
    """
    anodic = [p for p in peaks if p.peak_type == "anodic"]
    cathodic = [p for p in peaks if p.peak_type == "cathodic"]

    if not anodic or not cathodic:
        return []

    pairs = []
    used_cathodic = set()
    for ap in sorted(anodic, key=lambda p: -abs(p.net_current)):
        best = None
        best_sep = max_separation
        for j, cp in enumerate(cathodic):
            if j in used_cathodic:
                continue
            sep = abs(ap.potential - cp.potential)
            if sep < best_sep:
                best_sep = sep
                best = j
        if best is not None:
            used_cathodic.add(best)
            cp = cathodic[best]
            pairs.append(dict(
                anodic=ap,
                cathodic=cp,
                separation=abs(ap.potential - cp.potential),
                standard_potential=(ap.potential + cp.potential) / 2,
            ))
    return pairs


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def analyze_cv(potential, current,
               blank_potential=None, blank_current=None):
    """Run full CV analysis.

    Parameters
    ----------
    potential, current : array-like
        Raw CV data.
    blank_potential, blank_current : array-like, optional
        Blank measurement for subtraction.

    Returns
    -------
    CVResult
    """
    potential = np.asarray(potential, dtype=float)
    current = np.asarray(current, dtype=float)

    if blank_potential is not None and blank_current is not None:
        current = subtract_blank(
            potential, current,
            np.asarray(blank_potential, dtype=float),
            np.asarray(blank_current, dtype=float),
        )

    segments = split_segments(potential, current)

    all_peaks: List[CVPeak] = []
    for seg_idx, seg in enumerate(segments):
        raw = _detect_peaks_in_segment(
            seg["potential"], seg["current"], seg["direction"],
        )
        for r in raw:
            all_peaks.append(CVPeak(
                peak_type=r["peak_type"],
                potential=r["potential"],
                current=r["current"],
                net_current=r["net_current"],
                half_peak_potential=r["half_peak_potential"],
                onset_potential=r["onset_potential"],
                segment_index=seg_idx,
            ))

    # Global noise filter: discard peaks that are insignificant on the
    # full CV scale.  Two checks:
    #   1. net_current (from baseline fit) >= 2 % of full current range
    #   2. |raw peak current| >= 2 % of maximum absolute current
    # The second check catches cases where a bad baseline extrapolation
    # inflates the net_current of a tiny artefact.
    global_range = np.ptp(current)
    max_abs_current = max(abs(np.max(current)), abs(np.min(current)))
    if global_range > 0:
        min_net = 0.02 * global_range
        min_abs = 0.02 * max_abs_current
        all_peaks = [
            p for p in all_peaks
            if abs(p.net_current) >= min_net and abs(p.current) >= min_abs
        ]

    pairs = _match_reversible_pairs(all_peaks)
    is_reversible = len(pairs) > 0

    standard_potential = None
    peak_separation = None
    if pairs:
        best = max(pairs, key=lambda p: abs(p["anodic"].net_current))
        standard_potential = best["standard_potential"]
        peak_separation = best["separation"]

    return CVResult(
        peaks=all_peaks,
        is_reversible=is_reversible,
        standard_potential=standard_potential,
        peak_separation=peak_separation,
        matched_pairs=pairs,
    )
