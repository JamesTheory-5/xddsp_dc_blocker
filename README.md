# xddsp_dc_blocker

MODULE NAME:
**xddsp_dc_blocker**

DESCRIPTION:
First-order, time-varying DC blocker in XDDSP core style. Implements a 1-pole high-pass filter that removes DC and very low-frequency content using a pole near ( z = 1 ). The pole bandwidth is specified in Hz and can vary per sample. The pole coefficient is smoothed over time to avoid zipper noise when modulating parameters.

---

INPUTS:

* **x[n]** : input signal (audio-rate, 1D NumPy array of float32/float64)
* **width_hz[n]** : DC-block bandwidth in Hz (audio-rate 1D array, or broadcasted constant)
* **sr** : sampling rate in Hz (scalar, constant in `params`)
* **param_smooth_ms** : approximate time constant (ms) for smoothing the pole coefficient (scalar, constant in `params`)

OUTPUTS:

* **y[n]** : DC-blocked output signal (same shape as `x`)

---

STATE VARIABLES (as a tuple):
`state = (d1, b1)`

* **d1** : previous internal delay state (corresponds to previous intermediate `t` sample)
* **b1** : current (smoothed) pole feedback coefficient

`params = (sr, alpha, min_width_hz)`

* **sr** : sampling rate (Hz)
* **alpha** : smoothing coefficient for the pole (`0..1`), where larger = faster tracking
* **min_width_hz** : lower bound on bandwidth to keep the pole strictly inside the unit circle

---

EQUATIONS / MATH:

### Core DC-blocking filter

We implement the transposed direct form of the classic DC blocker:

[
y[n] = x[n] - x[n-1] + b_1[n], y[n-1]
]

In the chosen state-space form:

* State:

  * ( d_1[n] = t[n-1] )
  * ( b_1[n] ) = current smoothed feedback coefficient

* Per-sample update:

[
\begin{aligned}
t[n] &= x[n] + b_1[n]; d_1[n] \
y[n] &= t[n] - d_1[n] \
d_1[n+1] &= t[n]
\end{aligned}
]

So:

[
\text{state}[n] = (d_1[n], b_1[n]) \
\text{state}[n+1] = (d_1[n+1], b_1[n+1])
]

### Pole / bandwidth relationship

Given a desired DC-block bandwidth `width_hz[n]` and sample rate `sr`:

[
b_{1,\text{target}}[n] = \exp\left(-2\pi \frac{\max(\text{width_hz}[n], \text{min_width_hz})}{sr}\right)
]

This keeps the pole at radius ( r = b_1 ) near 1 for small `width_hz` and ensures stability by clamping `width_hz` to a minimum positive value.

### Parameter smoothing (time-varying coefficient rule)

The pole coefficient is smoothed over time using a simple one-pole lowpass in the coefficient domain:

[
b_1[n+1] = b_1[n] + \alpha \left( b_{1,\text{target}}[n] - b_1[n] \right)
]

where:

* ( \alpha \in (0, 1] ) is computed from `param_smooth_ms` and `sr`:

[
\tau = \text{param_smooth_ms} \cdot 10^{-3},\quad
\alpha = 1 - \exp\left(-\frac{1}{\tau, sr}\right)
]

This gives an approximate exponential time constant of `param_smooth_ms`.

### through-zero rules:

Not applicable (no through-zero frequency or phase concepts here); the filter is linear and time-varying only via `b1[n]`.

### phase wrapping rules:

Not applicable; this is a real-valued IIR filter, not an oscillator or phasor.

### nonlinearities:

None. The DC blocker is fully linear; parameter smoothing is also linear in `b1`.

### interpolation rules:

* `width_hz[n]` can be:

  * Constant over the block (no modulation), or
  * Linearly or arbitrarily time-varying (supplied as an array).
* Any interpolation from control-rate values to audio-rate `width_hz[n]` is assumed to be done outside the module (e.g., linear ramps). This module then applies an exponential smoothing in the coefficient domain via `alpha`.

---

NOTES:

* **Stable parameter ranges**:

  * `width_hz[n] >= 0`. Internally clamped to `min_width_hz > 0` to avoid `b1 ≈ 1`.
  * Reasonable `width_hz` range: `[0.1 Hz, 50 Hz]` for audio DC blocking.
* Numerical safety:

  * Clamping at `min_width_hz` prevents `b1` from being exactly 1, ensuring the pole is strictly inside the unit circle.
* All shapes and array allocations are done outside jitted functions.
* Inside Numba code:

  * No Python objects, dicts, or dynamic array allocation.
  * Only scalar control-flow branches (not data-dependent on array *vectors*).
* DSP is fully functional: state and params are passed in, new state is returned, no side effects.

---

## FULL PYTHON FILE

```python
"""
xddsp_dc_blocker.py

First-order, time-varying DC blocker in XDDSP core style.

Implements a 1-pole high-pass (DC-blocking) filter with a pole
near z = 1. The effective bandwidth of the DC blocker is specified
in Hz and may vary per sample. Time-varying parameters are smoothed
in the pole coefficient domain to avoid zipper noise.

MATHEMATICAL FORMULATION
------------------------

Classic DC blocker form:

    y[n] = x[n] - x[n-1] + b1[n] * y[n-1]

where 0 < b1[n] < 1 is the feedback coefficient / pole radius.

We implement a numerically stable transposed direct form:

    t[n]      = x[n] + b1[n] * d1[n]
    y[n]      = t[n] - d1[n]
    d1[n+1]   = t[n]

State:
    d1[n]  = internal delay
    b1[n]  = current (smoothed) pole coefficient

TIME-VARYING COEFFICIENTS
-------------------------

Given an audio-rate bandwidth trajectory width_hz[n] and sampling rate sr:

    width_clamped[n]   = max(width_hz[n], min_width_hz)
    b1_target[n]       = exp(-2*pi * width_clamped[n] / sr)

Parameter smoothing in coefficient domain:

    b1[n+1] = b1[n] + alpha * (b1_target[n] - b1[n])

where:

    tau     = param_smooth_ms * 1e-3
    alpha   = 1 - exp(-1.0 / (tau * sr))

This yields an approximate exponential time constant for the pole updates.

API
---

State and params are tuples of scalars / arrays only:

    state  = (d1, b1)
    params = (sr, alpha, min_width_hz)

Public functions:

    xddsp_dc_blocker_init(width_hz_init, sr, param_smooth_ms=10.0, min_width_hz=0.1)
    xddsp_dc_blocker_update_state(width_t, state, params)
    xddsp_dc_blocker_tick(x_t, state, params)
    xddsp_dc_blocker_process(x, width_hz, state, params, y_out)

All Numba-accelerated functions are decorated with:

    @njit(cache=True, fastmath=True)

This module is pure-functional in its DSP core: given inputs, params, and
state tuples, it returns new outputs and new state without side effects.
"""

from __future__ import annotations

import numpy as np
from numba import njit


# ----------------------------------------------------------------------
# Internal helpers (NumPy / Python side)
# ----------------------------------------------------------------------


def _compute_alpha(param_smooth_ms: float, sr: float) -> float:
    """
    Compute coefficient smoothing factor alpha from a time constant.

    Parameters
    ----------
    param_smooth_ms : float
        Approximate coefficient smoothing time constant in milliseconds.
    sr : float
        Sampling rate in Hz.

    Returns
    -------
    alpha : float
        Coefficient smoothing factor in (0, 1].
    """
    param_smooth_ms = max(param_smooth_ms, 0.0)
    if param_smooth_ms <= 1e-9:
        return 1.0  # effectively instantaneous
    tau = param_smooth_ms * 1e-3
    return 1.0 - np.exp(-1.0 / (tau * sr))


def _pole_radius_from_width(width_hz: float, sr: float, min_width_hz: float) -> float:
    """
    Map a bandwidth in Hz to a pole radius (feedback coefficient b1).

    Parameters
    ----------
    width_hz : float
        Desired DC-block bandwidth in Hz.
    sr : float
        Sampling rate in Hz.
    min_width_hz : float
        Minimum bandwidth to ensure strict stability (pole < 1.0).

    Returns
    -------
    b1 : float
        Pole radius / feedback coefficient.
    """
    w = max(width_hz, min_width_hz)
    return float(np.exp(-2.0 * np.pi * w / sr))


# ----------------------------------------------------------------------
# Public API: init / update / tick / process
# ----------------------------------------------------------------------


def xddsp_dc_blocker_init(
    width_hz_init: float,
    sr: float,
    param_smooth_ms: float = 10.0,
    min_width_hz: float = 0.1,
):
    """
    Initialize xddsp_dc_blocker state and params.

    Parameters
    ----------
    width_hz_init : float
        Initial bandwidth of the DC blocker in Hz.
    sr : float
        Sampling rate in Hz.
    param_smooth_ms : float, optional
        Approximate smoothing time constant for the pole coefficient,
        in milliseconds. Default is 10 ms.
    min_width_hz : float, optional
        Minimum allowed bandwidth in Hz to ensure stability
        (pole strictly inside the unit circle). Default is 0.1 Hz.

    Returns
    -------
    state : tuple
        Initial state tuple (d1, b1).
    params : tuple
        Params tuple (sr, alpha, min_width_hz).
    """
    alpha = _compute_alpha(param_smooth_ms, sr)
    b1_init = _pole_radius_from_width(width_hz_init, sr, min_width_hz)
    d1_init = 0.0

    state = (d1_init, b1_init)
    params = (float(sr), float(alpha), float(min_width_hz))
    return state, params


@njit(cache=True, fastmath=True)
def xddsp_dc_blocker_update_state(width_t: float, state, params):
    """
    Time-varying parameter update for one sample.

    This function smooths the pole coefficient b1 towards the target
    defined by the current bandwidth width_t.

    Parameters
    ----------
    width_t : float
        Desired DC-block bandwidth (Hz) at the current sample.
    state : tuple
        Current state (d1, b1).
    params : tuple
        Params (sr, alpha, min_width_hz).

    Returns
    -------
    new_state : tuple
        Updated state (d1, b1_new). Note that d1 is unchanged here.
    """
    d1, b1 = state
    sr, alpha, min_width_hz = params

    # Clamp width to ensure stability
    if width_t < min_width_hz:
        width_clamped = min_width_hz
    else:
        width_clamped = width_t

    # Compute target pole radius
    b1_target = np.exp(-2.0 * np.pi * width_clamped / sr)

    # Exponential smoothing in coefficient domain
    b1_new = b1 + alpha * (b1_target - b1)

    return (d1, b1_new)


@njit(cache=True, fastmath=True)
def xddsp_dc_blocker_tick(x_t: float, state, params):
    """
    Stateless per-sample DC blocker tick, given current state and params.

    Parameters
    ----------
    x_t : float
        Input sample.
    state : tuple
        Current state (d1, b1).
    params : tuple
        Params (sr, alpha, min_width_hz). Only b1 is used here,
        but params is accepted for API consistency.

    Returns
    -------
    y_t : float
        Output sample (DC-blocked).
    new_state : tuple
        Updated state (d1_new, b1) where b1 is unchanged.
        Parameter smoothing is handled in xddsp_dc_blocker_update_state.
    """
    d1, b1 = state

    # Transposed direct form:
    # t = x + b1 * d1
    # y = t - d1
    # d1' = t
    t = x_t + b1 * d1
    y_t = t - d1
    d1_new = t

    return y_t, (d1_new, b1)


@njit(cache=True, fastmath=True)
def xddsp_dc_blocker_process(x, width_hz, state, params, y_out):
    """
    Block-wise DC blocker processing with time-varying bandwidth.

    All arrays must be preallocated outside this function to avoid
    dynamic allocation inside Numba-jitted code.

    Parameters
    ----------
    x : np.ndarray, shape (N,)
        Input signal array.
    width_hz : np.ndarray, shape (N,)
        Bandwidth trajectory in Hz (per-sample). May be constant
        or time-varying.
    state : tuple
        Initial state (d1, b1).
    params : tuple
        Params (sr, alpha, min_width_hz).
    y_out : np.ndarray, shape (N,)
        Preallocated output array. Will be filled with DC-blocked samples.

    Returns
    -------
    y_out : np.ndarray, shape (N,)
        The same array as passed in, for chaining.
    new_state : tuple
        Final state after processing the block.
    """
    N = x.shape[0]
    s = state

    for n in range(N):
        # Update parameters for this sample
        s = xddsp_dc_blocker_update_state(width_hz[n], s, params)
        # Process one sample with the updated state
        y_t, s = xddsp_dc_blocker_tick(x[n], s, params)
        y_out[n] = y_t

    return y_out, s


# ----------------------------------------------------------------------
# Smoke test / example usage
# ----------------------------------------------------------------------


def _example_signal(sr: float, duration_s: float = 2.0):
    """
    Generate a test signal: DC offset + low-frequency sine + noise.
    """
    t = np.linspace(0.0, duration_s, int(sr * duration_s), endpoint=False)
    dc = 0.2
    sine = 0.5 * np.sin(2.0 * np.pi * 2.0 * t)
    noise = 0.05 * np.random.randn(t.shape[0])
    x = dc + sine + noise
    return t, x


def _example_width(sr: float, N: int):
    """
    Generate a time-varying width trajectory: start narrow, widen over time.
    """
    width_start = 0.5   # Hz
    width_end = 20.0    # Hz
    ramp = np.linspace(0.0, 1.0, N)
    return width_start + (width_end - width_start) * ramp


if __name__ == "__main__":
    # Basic smoke test: run the DC blocker on a noisy, DC-offset signal
    import matplotlib.pyplot as plt

    sr = 48000.0
    duration_s = 2.0

    t, x = _example_signal(sr, duration_s)
    width_hz = _example_width(sr, x.shape[0])

    # Initialize
    state, params = xddsp_dc_blocker_init(
        width_hz_init=1.0,
        sr=sr,
        param_smooth_ms=20.0,
        min_width_hz=0.1,
    )

    # Preallocate output
    y = np.empty_like(x)

    # Process
    y, state = xddsp_dc_blocker_process(x, width_hz, state, params, y)

    print("Input mean:", float(np.mean(x)))
    print("Output mean:", float(np.mean(y)))

    # Plot
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.title("xddsp_dc_blocker: Time Domain")
    plt.plot(t, x, label="input", alpha=0.5)
    plt.plot(t, y, label="output (DC-blocked)", alpha=0.9)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Frequency response via FFT
    X = np.fft.rfft(x)
    Y = np.fft.rfft(y)
    freqs = np.fft.rfftfreq(x.shape[0], 1.0 / sr)

    plt.subplot(2, 1, 2)
    plt.title("Magnitude Spectrum")
    plt.semilogx(freqs[1:], 20.0 * np.log10(np.abs(X[1:]) + 1e-12), label="input")
    plt.semilogx(freqs[1:], 20.0 * np.log10(np.abs(Y[1:]) + 1e-12), label="output")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Optional listen example (requires sounddevice)
    try:
        import sounddevice as sd

        print("Playing input (with DC offset)...")
        sd.play(x, int(sr))
        sd.wait()

        print("Playing DC-blocked output...")
        sd.play(y, int(sr))
        sd.wait()
    except Exception as e:
        print("sounddevice not available or playback failed:", e)
```

---

### Things we could improve or extend next

* **Stereo / multichannel version**
  Same API but with shape `(N, C)` and per-channel state tuples (still no classes).

* **Control-rate modulation helper**
  A small XDDSP-style module that takes sparse control points and generates an audio-rate `width_hz[n]` trajectory via linear or exponential interpolation.

* **DC blocker variants**

  * 2nd-order DC block for steeper low-frequency attenuation.
  * “Tilt” / shelving + DC block combo for gentle low-end cleanup.

* **Analysis utilities**
  Separate script to sweep `width_hz` and show magnitude response overlays for design tuning.
