"""
Fault model for input-side bit-flip injection in 2MM float32 kernels.

HONESTY DISCLAIMER: This models faults injected into an input matrix B
BEFORE kernel execution (input-side). This is NOT equivalent to register-level
bit-flip injection as performed by SASSIFI or NVBitFI, which inject faults
into destination registers after instruction execution. Adjacent methodology;
Strada Q discipline.

IEEE 754 float32 bit layout:
  bit 31: sign
  bits 30-23: exponent (8 bits)
  bits 22-0: mantissa (23 bits)

We partition the 32 bits into 5 fault classes:
  SIGN          (1 bit  = 31):        P = 1/32  ~ 0.03125
  HIGH_EXP      (4 bits = 27-30):     P = 4/32  = 0.125
  LOW_EXP       (4 bits = 23-26):     P = 4/32  = 0.125
  HIGH_MANTISSA (7 bits = 16-22):     P = 7/32  ~ 0.21875
  LOW_MANTISSA  (16 bits = 0-15):     P = 16/32 = 0.5

Handling correctness:
  SIGN/EXP: the per-(class, bit, v) shift delta is deterministic given v.
            Output D[r,s] | fault_scenario is EXACTLY Gaussian if B is Gaussian.
            These classes return discrete sub-components (exact).
  MANTISSA: shift delta depends on random bit-position p and sign.
            True conditional output is a discrete mixture of Gaussians.
            We APPROXIMATE by moment-matching to a single Gaussian:
              E[delta | v] = 0  (by sign symmetry)
              Var[delta | v] = v^2 * E[2^(2p)]
            This approximation has design target: < 5% relative error on Pr(SDC).
            Enforced by M3.5 cross-validation.

References:
  - plan/2026-05-25-lishan-resilience-poc.md §Fault model
  - gaussian-mixture-expert memo Q4
  - numerical-stability-expert recipe H1, H2, H6
"""

from __future__ import annotations

import enum
import struct
import warnings
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from scipy.stats import norm


# ---------------------------------------------------------------------------
# Warning classes
# ---------------------------------------------------------------------------

class MantissaApproximationWarning(UserWarning):
    """Emitted when mantissa moment-match approximation is used.

    The MANTISSA classes (HIGH_MANTISSA, LOW_MANTISSA) are handled via
    moment-matching to a single Gaussian rather than the true discrete mixture.
    Accuracy target: < 5% relative error on Pr(SDC), validated by M3.5.
    """


class BimodalApproximationWarning(UserWarning):
    """Emitted when Gaussian moment-match of a bimodal Bernoulli prior is used.

    Valid only for DENSE A (m_dense >= 16 nonzero entries per row), where
    CLT applies. For sparse A (including A=I_32), use 2-component GM primary.
    """


# ---------------------------------------------------------------------------
# Fault classes
# ---------------------------------------------------------------------------

class FaultClass(enum.Enum):
    """IEEE 754 float32 fault class partition (5 classes, 32 bits total)."""
    SIGN = "SIGN"                       # bit 31         — EXACT
    HIGH_EXP = "HIGH_EXP"              # bits 27-30     — EXACT
    LOW_EXP = "LOW_EXP"                # bits 23-26     — EXACT
    HIGH_MANTISSA = "HIGH_MANTISSA"    # bits 16-22     — APPROXIMATE (moment-matched)
    LOW_MANTISSA = "LOW_MANTISSA"      # bits 0-15      — APPROXIMATE (moment-matched)


# Class probabilities: uniform over 32 bits
FAULT_CLASS_PROB: dict[FaultClass, float] = {
    FaultClass.SIGN:          1.0 / 32,
    FaultClass.HIGH_EXP:      4.0 / 32,
    FaultClass.LOW_EXP:       4.0 / 32,
    FaultClass.HIGH_MANTISSA: 7.0 / 32,
    FaultClass.LOW_MANTISSA: 16.0 / 32,
}

# Exponent shift magnitudes for EXP classes (bit-flip changes exponent by these factors)
# Flipping bit k in the exponent field changes the value by factor ≈ 2^(2^(k-23)) - 1
# HIGH_EXP bits 27-30: exponent bits 4-7 → factors 2^16, 2^32, 2^64, 2^128 (→ overflow)
HIGH_EXP_K_VALUES = [16, 32, 64, 128]   # these almost always overflow for |v| > 0
# LOW_EXP bits 23-26: exponent bits 0-3 → factors 2^1, 2^2, 2^4, 2^8
LOW_EXP_K_VALUES   = [1, 2, 4, 8]


@dataclass
class ExactSubComponent:
    """One exact sub-component of a fault scenario.

    For SIGN/EXP classes: the shift delta = v * scale is deterministic.
    Output D[r,s] given this sub-component is exactly Gaussian.

    Attributes:
        label: Descriptive name (e.g., "HIGH_EXP_k16")
        P_sub: Probability weight for this sub-component within the class
        scale: Shift scale — delta = v * scale (additive shift on the input cell)
        is_overflow: True if this sub-component always causes OTR (overflow-to-random)
    """
    label: str
    P_sub: float
    scale: float
    is_overflow: bool = False


@dataclass
class MantissaSubComponent:
    """Moment-matched approximation for a mantissa fault class.

    APPROXIMATE: the true distribution is a discrete mixture (14 or 32 components).
    We collapse to E[delta|v]=0, Var[delta|v] = v^2 * E[2^(2p)].

    Attributes:
        label: "HIGH_MANTISSA" or "LOW_MANTISSA"
        P_class: Class probability weight
        E_2p2: E[2^(2p)] for the bit-position range (p = -bit_pos / base_scale)
        var_delta_per_v2: Var[delta|v] / v^2 = E_2p2 (so Var[delta|v] = v^2 * E_2p2)
    """
    label: str
    P_class: float
    E_2p2: float

    def var_delta(self, v: float | np.ndarray) -> float | np.ndarray:
        """Var[delta | v] = v^2 * E[2^(2p)]."""
        return v ** 2 * self.E_2p2

    def std_delta(self, v: float | np.ndarray) -> float | np.ndarray:
        """Std[delta | v] = |v| * sqrt(E[2^(2p)])."""
        return np.abs(v) * np.sqrt(self.E_2p2)


# ---------------------------------------------------------------------------
# Pre-computed mantissa E[2^(2p)] values
# ---------------------------------------------------------------------------
# For HIGH_MANTISSA (bits 16-22), the shift is ±v * 2^(-p) where p in [1,7].
# Actually: flipping bit b (b in 0..22) in the mantissa changes the value by
#   delta = ±v * 2^(b - 23)  (where ± is from sign symmetry of the flip)
# For HIGH_MANTISSA bits 16-22: exponents = bits - 23 = [-7, -6, ..., -1]
# E[2^(2p)] = mean over p in {-7,-6,-5,-4,-3,-2,-1} of 4^p
#           = (4^(-7) + 4^(-6) + ... + 4^(-1)) / 7
_HIGH_MANT_EXPONENTS = np.array([-7, -6, -5, -4, -3, -2, -1])
HIGH_MANTISSA_E2P2 = float(np.mean(4.0 ** _HIGH_MANT_EXPONENTS))

# For LOW_MANTISSA bits 0-15: exponents = bits - 23 = [-23, -22, ..., -8]
# E[2^(2p)] = mean over p in {-23,-22,...,-8} of 4^p
_LOW_MANT_EXPONENTS = np.arange(-23, -7)   # -23 to -8 inclusive = 16 values
LOW_MANTISSA_E2P2 = float(np.mean(4.0 ** _LOW_MANT_EXPONENTS))


# ---------------------------------------------------------------------------
# Public API: get_fault_subcomponents
# ---------------------------------------------------------------------------

def get_sign_subcomponents() -> List[ExactSubComponent]:
    """Return sub-components for the SIGN fault class.

    Flipping bit 31 negates the value: delta = -2*v (new value = -v, shift = -2v).
    Single sub-component; exact.
    """
    return [
        ExactSubComponent(label="SIGN", P_sub=1.0, scale=-2.0)
    ]


def get_high_exp_subcomponents() -> List[ExactSubComponent]:
    """Return sub-components for HIGH_EXP class (bits 27-30).

    Flipping bit 30-k of the 8-bit exponent field changes the stored exponent
    by +/- 2^(bit-23). The value shift is approximately:
        delta_approx = v * (2^k - 1)  where k = 2^(bit - 23) for each bit.

    For k in {16, 32, 64, 128}: 2^k is enormous — these will overflow float32
    for almost all inputs. We pre-classify these as OTR (overflow-to-random).
    The is_overflow flag signals that the tail_gauss computation is bypassed.

    EXACT sub-components (deterministic shift given v).
    """
    subs = []
    for k in HIGH_EXP_K_VALUES:
        scale = float(2**k - 1)
        subs.append(
            ExactSubComponent(
                label=f"HIGH_EXP_k{k}",
                P_sub=1.0 / len(HIGH_EXP_K_VALUES),
                scale=scale,
                is_overflow=True,  # 2^16 - 1 = 65535; v * 65535 >> float32 max for most v
            )
        )
    return subs


def get_low_exp_subcomponents() -> List[ExactSubComponent]:
    """Return sub-components for LOW_EXP class (bits 23-26).

    k in {1, 2, 4, 8}: the scale = 2^k - 1 in {1, 3, 15, 255}.
    No overflow for typical values; these can cause large SDC.
    EXACT sub-components.
    """
    subs = []
    for k in LOW_EXP_K_VALUES:
        scale = float(2**k - 1)
        subs.append(
            ExactSubComponent(
                label=f"LOW_EXP_k{k}",
                P_sub=1.0 / len(LOW_EXP_K_VALUES),
                scale=scale,
                is_overflow=False,
            )
        )
    return subs


def get_high_mantissa_component() -> MantissaSubComponent:
    """Return moment-matched component for HIGH_MANTISSA class (bits 16-22).

    APPROXIMATE: true distribution is 14 sub-components (7 bit positions x 2 signs).
    Collapsed to E[delta|v]=0, Var[delta|v] = v^2 * E[2^(2p)].

    Emits MantissaApproximationWarning on first call.
    """
    warnings.warn(
        "HIGH_MANTISSA is handled via moment-matched Gaussian approximation "
        "(not exact). Accuracy target: < 5% relative error on Pr(SDC). "
        "See M3.5 cross-validation.",
        MantissaApproximationWarning,
        stacklevel=2,
    )
    return MantissaSubComponent(
        label="HIGH_MANTISSA",
        P_class=FAULT_CLASS_PROB[FaultClass.HIGH_MANTISSA],
        E_2p2=HIGH_MANTISSA_E2P2,
    )


def get_low_mantissa_component() -> MantissaSubComponent:
    """Return moment-matched component for LOW_MANTISSA class (bits 0-15).

    APPROXIMATE: true distribution is 32 sub-components.
    Contribution to SDC is negligible for typical eps since delta << eps*|mu_golden|.

    Emits MantissaApproximationWarning on first call.
    """
    warnings.warn(
        "LOW_MANTISSA is handled via moment-matched Gaussian approximation "
        "(not exact). Typically negligible contribution to Pr(SDC) since "
        "delta << eps * |mu_golden|.",
        MantissaApproximationWarning,
        stacklevel=2,
    )
    return MantissaSubComponent(
        label="LOW_MANTISSA",
        P_class=FAULT_CLASS_PROB[FaultClass.LOW_MANTISSA],
        E_2p2=LOW_MANTISSA_E2P2,
    )


# ---------------------------------------------------------------------------
# H1: Overflow pre-classification (log-space)
# ---------------------------------------------------------------------------

# float32 max value is 3.4028235e+38; overflow when log|value| > log(float32_max)
# Use np.finfo(np.float32).max rather than 127*log(2) = 88.03, which
# underestimates by ~0.7 nats (true log(float32_max) = 88.72).
_LOG2 = float(np.log(2.0))
_FLOAT32_MAX_LOG = float(np.log(np.finfo(np.float32).max))  # = 88.7228...


def check_overflow(
    v: float | np.ndarray,
    fault_class: FaultClass,
    A_ri: float | np.ndarray,
) -> Tuple[bool | np.ndarray, float | np.ndarray]:
    """Pre-classify overflow in log-space (H1 recipe).

    For EXP classes, the shifted value = v * (2^k - 1) * A[r,i].
    Overflow condition (IEEE 754 float32):
        log|v| + log(2^k - 1) + log|A[r,i]| > 127 * log(2)

    For MANTISSA classes: no overflow (shifts are tiny relative to v).
    For SIGN class: no overflow (flips sign only; |new value| = |old value|).

    Args:
        v: Input cell value (scalar or array).
        fault_class: Which fault class.
        A_ri: A[r,i] kernel coefficient (scalar or array).

    Returns:
        (is_overflow, log_magnitude): is_overflow is bool/array; log_magnitude is
        log|v * scale * A[r,i]| (or -inf if v=0 or A_ri=0).
    """
    if fault_class in (FaultClass.SIGN, FaultClass.HIGH_MANTISSA, FaultClass.LOW_MANTISSA):
        # SIGN: no magnitude change. MANTISSA: shifts are tiny.
        if np.isscalar(v):
            return False, -np.inf
        return np.zeros(np.asarray(v).shape, dtype=bool), np.full(np.asarray(v).shape, -np.inf)

    # EXP classes: check each k sub-component
    if fault_class == FaultClass.HIGH_EXP:
        k_values = HIGH_EXP_K_VALUES
    else:  # LOW_EXP
        k_values = LOW_EXP_K_VALUES

    v_arr = np.asarray(v, dtype=np.float64)
    A_ri_arr = np.asarray(A_ri, dtype=np.float64)

    # log|v| — handle v=0 explicitly
    with np.errstate(divide="ignore"):
        log_abs_v = np.where(v_arr == 0.0, -np.inf, np.log(np.abs(v_arr)))
        log_abs_A = np.where(A_ri_arr == 0.0, -np.inf, np.log(np.abs(A_ri_arr)))

    # Any sub-component that overflows makes the cell OTR
    any_overflow = np.zeros_like(v_arr, dtype=bool)
    max_log_mag = np.full_like(v_arr, -np.inf)

    for k in k_values:
        scale = float(2**k - 1)
        log_scale = float(np.log(scale))
        log_mag = log_abs_v + log_scale + log_abs_A
        is_overflow_k = log_mag > _FLOAT32_MAX_LOG
        any_overflow = any_overflow | is_overflow_k
        max_log_mag = np.maximum(max_log_mag, log_mag)

    if np.isscalar(v) and np.isscalar(A_ri):
        return bool(any_overflow.item()), float(max_log_mag.item())
    return any_overflow, max_log_mag


# ---------------------------------------------------------------------------
# H1 convenience: per-class overflow check for a full row
# ---------------------------------------------------------------------------

def check_overflow_row(
    v: float,
    fault_class: FaultClass,
    A_row: np.ndarray,
) -> np.ndarray:
    """Check overflow for each (v, fault_class, A[r,i]) combination.

    Args:
        v: Scalar input value.
        fault_class: Fault class.
        A_row: Row r of A matrix (shape m,).

    Returns:
        Boolean array of shape (m,) indicating OTR for each input column i.
    """
    result = np.zeros(len(A_row), dtype=bool)
    for i, a in enumerate(A_row):
        is_ovf, _ = check_overflow(v, fault_class, a)
        result[i] = is_ovf
    return result


# ---------------------------------------------------------------------------
# M1.3: shift_moments
# ---------------------------------------------------------------------------

def shift_moments(
    v: float | np.ndarray,
    fault_class: FaultClass,
) -> Tuple[float | np.ndarray, float | np.ndarray]:
    """Return (E[delta | v], Var[delta | v]) for each fault class.

    For SIGN/EXP: Var = 0 (deterministic shift given v).
    For MANTISSA: moment-matched approximation.
      E[delta | v] = 0  (sign symmetry)
      Var[delta | v] = v^2 * E[2^(2p)]

    Note: The 'delta' here is the shift on the INPUT CELL B[i,j].
    The shift on output D[r,s] is delta * A[r,i] (for the (i,j=s) fault).

    Args:
        v: Input cell value (scalar or array).
        fault_class: Fault class.

    Returns:
        (E_delta, Var_delta): Conditional mean and variance of the shift.
    """
    v_arr = np.asarray(v, dtype=np.float64)
    zeros = np.zeros_like(v_arr)

    if fault_class == FaultClass.SIGN:
        # delta = -2*v (deterministic); E = -2v, Var = 0
        return -2.0 * v_arr, zeros

    elif fault_class == FaultClass.HIGH_EXP:
        # Average over k in {16,32,64,128}: scale = 2^k - 1
        # Each is deterministic given k; average scale^2 = mean over k of (2^k-1)^2
        # E[delta] = mean_k(v*(2^k-1)) = v * mean(scales)
        scales = np.array([float(2**k - 1) for k in HIGH_EXP_K_VALUES])
        e_delta = v_arr * float(np.mean(scales))
        # Var over k (treating k as uniform discrete): Var[delta] = v^2 * Var[scale]
        var_delta = v_arr ** 2 * float(np.var(scales))
        return e_delta, var_delta

    elif fault_class == FaultClass.LOW_EXP:
        scales = np.array([float(2**k - 1) for k in LOW_EXP_K_VALUES])
        e_delta = v_arr * float(np.mean(scales))
        var_delta = v_arr ** 2 * float(np.var(scales))
        return e_delta, var_delta

    elif fault_class == FaultClass.HIGH_MANTISSA:
        # Moment-matched: E=0 (sign symmetry), Var = v^2 * E[2^(2p)]
        warnings.warn(
            "HIGH_MANTISSA shift_moments uses moment-matched approximation.",
            MantissaApproximationWarning,
            stacklevel=2,
        )
        return zeros, v_arr ** 2 * HIGH_MANTISSA_E2P2

    elif fault_class == FaultClass.LOW_MANTISSA:
        warnings.warn(
            "LOW_MANTISSA shift_moments uses moment-matched approximation.",
            MantissaApproximationWarning,
            stacklevel=2,
        )
        return zeros, v_arr ** 2 * LOW_MANTISSA_E2P2

    else:
        raise ValueError(f"Unknown fault class: {fault_class}")


# ---------------------------------------------------------------------------
# H2: Tail probability (vectorized, log-space)
# ---------------------------------------------------------------------------

def tail_gauss(
    mu_post: float | np.ndarray,
    sigma_post: float | np.ndarray,
    threshold: float | np.ndarray,
) -> float | np.ndarray:
    """P(|X - mu_golden| > threshold) where X ~ N(mu_post, sigma_post^2).

    Uses scipy.stats.norm.logsf + logsumexp for numerical stability (H2).
    Handles threshold=0 and sigma=0 edge cases.

    In our model, mu_golden ~ 0 for zero-mean B, so:
    P_SDC = P(|X| > threshold) where X ~ N(mu_post, sigma_post^2)
          = P(X > threshold) + P(X < -threshold)
          = norm.sf(threshold, mu_post, sigma_post) + norm.cdf(-threshold, mu_post, sigma_post)

    We use log-space to avoid underflow for extreme z-scores.

    Args:
        mu_post: Posterior mean of output cell D[r,s] (scalar or array).
        sigma_post: Posterior std dev (scalar or array, >= 0).
        threshold: SDC threshold = eps * |mu_golden| (scalar or array, >= 0).

    Returns:
        P_SDC in [0, 1]. Capped at 1.0.
    """
    mu_post = np.asarray(mu_post, dtype=np.float64)
    sigma_post = np.asarray(sigma_post, dtype=np.float64)
    threshold = np.asarray(threshold, dtype=np.float64)

    # Handle degenerate sigma=0: deterministic outcome
    # If |mu_post| > threshold → SDC=1, else SDC=0
    degenerate = sigma_post < 1e-300
    if np.any(degenerate):
        det_sdc = (np.abs(mu_post) > threshold).astype(np.float64)
    else:
        det_sdc = np.zeros_like(mu_post)

    # Handle threshold=0: any mu != 0 is SDC
    zero_thresh = threshold <= 0.0

    # Main computation for non-degenerate cases
    # P_SDC = P(X > T) + P(X < -T) where X ~ N(mu_post, sigma_post^2)
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma_safe = np.where(degenerate, 1.0, sigma_post)  # avoid /0

        # P(X > T) = norm.sf(T, mu_post, sigma_post) in log-space
        z_upper = (threshold - mu_post) / sigma_safe
        z_lower = (-threshold - mu_post) / sigma_safe

        log_p_upper = norm.logsf(z_upper)
        log_p_lower = norm.logcdf(z_lower)

        # logsumexp to combine
        # log(p_upper + p_lower) = logsumexp([log_p_upper, log_p_lower])
        from scipy.special import logsumexp as sp_logsumexp
        log_p_sdc = sp_logsumexp(
            np.stack([log_p_upper, log_p_lower], axis=-1), axis=-1
        )
        p_sdc = np.exp(np.clip(log_p_sdc, -750.0, 0.0))  # cap at 1.0

    # Apply degenerate and zero-threshold overrides
    result = np.where(degenerate, det_sdc, p_sdc)
    result = np.where(zero_thresh & ~degenerate, 1.0, result)
    result = np.clip(result, 0.0, 1.0)

    # Return scalar if inputs were scalar
    if result.ndim == 0:
        return float(result.item())
    return result


# ---------------------------------------------------------------------------
# Mantissa moment-match validation helper (M1.4 edge case)
# ---------------------------------------------------------------------------

def mantissa_moment_match_validation(
    v: float,
    fault_class: FaultClass,
    A_ri: float,
    eps: float,
    mu_golden: float = 0.0,
    sigma_base: float = 0.0,
) -> dict:
    """Compare moment-matched Pr(SDC) against full discrete-mixture expansion.

    Used for M1.4 sanity check: max relative error < 5% for MANTISSA classes.

    Args:
        v: Input cell value.
        fault_class: Must be HIGH_MANTISSA or LOW_MANTISSA.
        A_ri: Kernel coefficient A[r,i].
        eps: SDC threshold fraction.
        mu_golden: Baseline D[r,s] mean.
        sigma_base: Baseline D[r,s] std dev.

    Returns:
        dict with keys: 'p_sdc_mm' (moment-matched), 'p_sdc_exact' (full mixture),
        'rel_err', 'pass' (bool, rel_err < 0.05).
    """
    if fault_class not in (FaultClass.HIGH_MANTISSA, FaultClass.LOW_MANTISSA):
        raise ValueError("Only MANTISSA classes for moment-match validation")

    threshold = eps * abs(mu_golden) if abs(mu_golden) > 0 else eps * abs(A_ri * v) + 1e-30

    # Moment-matched version
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MantissaApproximationWarning)
        _, var_delta = shift_moments(v, fault_class)
    sigma_delta_post = float(np.sqrt(var_delta)) * abs(A_ri)
    sigma_post_mm = float(np.sqrt(sigma_base**2 + sigma_delta_post**2))
    p_sdc_mm = float(tail_gauss(mu_golden, sigma_post_mm, threshold))

    # Full discrete-mixture version
    if fault_class == FaultClass.HIGH_MANTISSA:
        bit_exponents = _HIGH_MANT_EXPONENTS  # [-7, -6, ..., -1]
    else:
        bit_exponents = _LOW_MANT_EXPONENTS   # [-23, -22, ..., -8]

    n_bits = len(bit_exponents)
    p_sdc_exact = 0.0
    weight = 1.0 / (2 * n_bits)  # uniform over (bit_pos, sign)

    for exp in bit_exponents:
        for sign in [+1.0, -1.0]:
            delta = sign * v * (2.0 ** float(exp)) * A_ri
            mu_scenario = mu_golden + delta
            sigma_scenario = sigma_base
            p_sdc_exact += weight * float(tail_gauss(mu_scenario, sigma_scenario, threshold))

    # Relative error
    denom = max(abs(p_sdc_exact), 1e-10)
    rel_err = abs(p_sdc_mm - p_sdc_exact) / denom

    return {
        "p_sdc_mm": p_sdc_mm,
        "p_sdc_exact": p_sdc_exact,
        "rel_err": rel_err,
        "pass": rel_err < 0.05,
    }
