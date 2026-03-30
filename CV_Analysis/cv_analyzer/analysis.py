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
# Local windowing for onset / Ep/2
# ---------------------------------------------------------------------------

def _local_pre_peak_window(potential, smoothed, peak_idx, max_v=1.0):
    """Return a local window of the pre-peak region.

    Limits the lookback to at most *max_v* volts from the peak potential,
    so that baseline fits use data near the actual peak rather than
    the far end of a long segment.

    Returns (pot_window, cur_window, local_peak_idx).
    """
    n_pre = peak_idx + 1
    if n_pre < 15:
        return potential[:n_pre], smoothed[:n_pre], peak_idx

    pot_range = abs(potential[0] - potential[peak_idx])
    if pot_range <= max_v:
        return potential[:n_pre], smoothed[:n_pre], peak_idx

    # Find the first index within max_v of the peak
    distances = np.abs(potential[:n_pre] - potential[peak_idx])
    within = distances <= max_v
    first = int(np.argmax(within))

    # Ensure at least 15 points
    if n_pre - first < 15:
        first = max(0, n_pre - 15)

    local_pk = peak_idx - first
    return potential[first:n_pre], smoothed[first:n_pre], local_pk


# ---------------------------------------------------------------------------
# Onset potential
# ---------------------------------------------------------------------------

def _compute_onset_potential(potential, smoothed, peak_idx):
    """Compute onset potential via tangent-intersection method.

    1. Fit a baseline tangent to the flat region before the peak.
    2. Fit a slope tangent to the steepest rising part of the peak.
    3. The intersection of these two lines is the onset potential.

    Works for both anodic (increasing potential) and cathodic (decreasing
    potential) segments — for cathodic peaks the caller passes -smoothed
    so the peak appears as a maximum, and the pre-peak region (indices
    0..peak_idx) corresponds to more-positive potentials (right side on
    the CV plot).
    """
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
    dI = np.gradient(pre, pot)
    dI_smooth = _adaptive_smooth(dI, len(dI))

    # Search in the second half of the pre-peak data for the steepest point
    half = max(len(dI_smooth) // 2, 1)
    search_region = dI_smooth[half:]
    inflection_local = np.argmax(np.abs(search_region))
    inflection_idx = half + inflection_local

    margin = min(5, inflection_idx, len(pre) - inflection_idx - 1)
    if margin < 2:
        return _onset_threshold_fallback(pot, pre, base_coeffs, peak_idx)

    tang_pot = pot[inflection_idx - margin : inflection_idx + margin + 1]
    tang_cur = pre[inflection_idx - margin : inflection_idx + margin + 1]
    tang_coeffs = np.polyfit(tang_pot, tang_cur, 1)

    # --- intersection of the two lines: a1*E + b1 = a2*E + b2 -------------
    a1, b1 = base_coeffs
    a2, b2 = tang_coeffs
    if abs(a2 - a1) < 1e-12:
        return _onset_threshold_fallback(pot, pre, base_coeffs, peak_idx)

    e_onset = (b1 - b2) / (a2 - a1)

    # Sanity: onset must lie within the pre-peak window
    e_min = min(pot[0], pot[-1])
    e_max = max(pot[0], pot[-1])
    if not (e_min <= e_onset <= e_max):
        return _onset_threshold_fallback(pot, pre, base_coeffs, peak_idx)

    return float(e_onset)


def _onset_threshold_fallback(pot, pre, base_coeffs, peak_idx):
    """Fallback: find where net current first crosses 10% of peak height."""
    net = pre - np.polyval(base_coeffs, pot)
    peak_net = net[-1]  # net current at the peak
    if abs(peak_net) < 1e-12:
        return float(pot[0])
    threshold = peak_net * 0.10
    # Find sign-change crossings of (net - threshold)
    diff = net - threshold
    crossings = np.where(np.diff(np.sign(diff)))[0]
    if len(crossings):
        return float(pot[crossings[0]])
    # Last resort: first point exceeding threshold
    if peak_net > 0:
        above = np.where(net >= threshold)[0]
    else:
        above = np.where(net <= threshold)[0]
    if len(above):
        return float(pot[above[0]])
    return float(pot[0])


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
        closest = np.argmin(np.abs(diff))
        return float(potential[closest])

    # Use the last crossing (closest to peak)
    ci = crossings[-1]
    d0, d1 = diff[ci], diff[ci + 1]
    if abs(d1 - d0) > 1e-15:
        frac = -d0 / (d1 - d0)
    else:
        frac = 0.5
    ep2 = potential[ci] + frac * (potential[ci + 1] - potential[ci])
    return float(ep2)


def _detect_peaks_in_segment(potential, current, direction, global_range,
                             min_prominence_frac=0.05, max_peaks=3):
    """Detect peaks in one monotonic segment.

    For anodic segments (E increasing) we look for current maxima (oxidation).
    For cathodic segments (E decreasing) we look for current minima (reduction).

    Uses *global_range* (peak-to-peak of the entire CV current) for
    the minimum prominence threshold so that only peaks significant on the
    full-CV scale are retained.

    Returns at most *max_peaks* results, sorted by prominence.
    """
    n = len(current)
    smoothed = _adaptive_smooth(current, n)

    if global_range == 0:
        return []

    segment_range = np.ptp(smoothed)
    # Use the smaller of 10 % segment-range and 5 % global-range so that
    # real peaks in a small segment aren't overshadowed by a large signal
    # elsewhere (e.g. oxidation near solvent window while reduction dominates
    # the global range).
    min_prom = min(segment_range * 0.10, global_range * min_prominence_frac)
    # … but never below 1.5 % of global range (safety floor against noise)
    min_prom = max(min_prom, global_range * 0.015)
    min_dist = max(n // 15, 5)
    # Require peaks to span at least a few data points (scale with resolution)
    min_width = max(n // 100, 2)

    if direction == "anodic":
        signal = smoothed
        peak_type = "anodic"
    else:
        signal = -smoothed
        peak_type = "cathodic"

    idx, props = find_peaks(signal, prominence=min_prom, distance=min_dist,
                            width=min_width)
    if len(idx) == 0:
        return []

    # Sort by prominence descending and keep only top candidates
    prominences = props["prominences"]
    order = np.argsort(-prominences)
    idx = idx[order]
    prominences = prominences[order]

    results = []
    for i in range(len(idx)):
        pk = int(idx[i])

        # Skip peaks too close to segment endpoints (vertex artifacts).
        # Use a small index guard plus asymmetric voltage margins:
        #   - 200 mV at the segment START (scan just reversed, transient)
        #   - 100 mV at the segment END
        if pk < max(int(n * 0.02), 2) or pk > n - max(int(n * 0.02), 2) - 1:
            continue
        if abs(potential[pk] - potential[0]) < 0.200:
            continue
        if abs(potential[pk] - potential[-1]) < 0.100:
            continue

        # Use a local window (~1 V) for onset and Ep/2 so that
        # baseline fits stay close to the actual peak.
        w_pot, w_cur, w_pk = _local_pre_peak_window(potential, signal, pk)

        ep2 = _compute_half_peak_potential(w_pot, w_cur, w_pk)
        onset = _compute_onset_potential(w_pot, w_cur, w_pk)

        results.append(dict(
            peak_type=peak_type,
            potential=potential[pk],
            current=current[pk],
            net_current=float(prominences[i]),
            half_peak_potential=ep2,
            onset_potential=onset,
        ))

        if len(results) >= max_peaks:
            break

    return results


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def _deduplicate_peaks(peaks, min_separation=0.100):
    """Remove peaks that are closer than *min_separation* V.

    When two peaks are too close, keep the one with the higher prominence
    (net_current).
    """
    if len(peaks) <= 1:
        return peaks

    # Sort by potential
    peaks_sorted = sorted(peaks, key=lambda p: p.potential)
    kept = [peaks_sorted[0]]
    for pk in peaks_sorted[1:]:
        if abs(pk.potential - kept[-1].potential) < min_separation:
            # Keep the more prominent one
            if abs(pk.net_current) > abs(kept[-1].net_current):
                kept[-1] = pk
        else:
            kept.append(pk)
    return kept


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

    # Compute global scale for prominence thresholding
    global_range = np.ptp(current)

    segments = split_segments(potential, current)

    all_peaks: List[CVPeak] = []
    for seg_idx, seg in enumerate(segments):
        raw = _detect_peaks_in_segment(
            seg["potential"], seg["current"], seg["direction"],
            global_range,
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

    # Filter out peaks where the raw current is negligible compared to
    # the global CV scale (catches inflated-prominence artefacts).
    max_abs_current = max(abs(np.max(current)), abs(np.min(current)), 1e-12)
    if global_range > 0:
        min_abs = 0.03 * max_abs_current
        all_peaks = [p for p in all_peaks if abs(p.current) >= min_abs]

    # Remove near-duplicate peaks (within 100 mV)
    all_peaks = _deduplicate_peaks(all_peaks)

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
