"""Explicit six-axis line-table vibration scenarios (mode ``scenario_lines_v1``).

Each scenario is a builder ``f(seed, **params) -> LineTable`` that writes per-line acceleration
amplitude, frequency and phase directly (no band synthesis, no Gamma calibration). Structure is fixed
per scenario; frequencies / phases / directions that change between episodes are derived from the
episode seed and recorded as ``latents``. ``level`` multiplies every amplitude.

Canonical catalog v2 (only S1/S2/S3/S4; no legacy aliases):
    S1  heavy rolling about a random horizontal axis (table moves a lot, objects mostly ride along)
    S2  rough road: independent broadband kicks on the three translations (objects random-walk)
    S3   orbital shaker, 20 mm orbit (objects slide / swirl continuously)
    S4  press tending with alternating horizontal kicks (objects jump at every stroke)

Constraints of the repository representation: at most ``DEFAULT_MAX_LINES`` (12) lines per axis,
every line below ``CONSERVATIVE_MAX_LINE_FREQUENCY_HZ`` (8.87 Hz). Rotations act about the deck origin,
0.30 m below and 0.11 m behind the tabletop centre.
"""

from __future__ import annotations

import inspect
import math
from dataclasses import dataclass
from typing import Any, Callable, Mapping

import numpy as np

from shakebench.physics import excitation as ex

SCENARIO_MODE = "scenario_lines_v1"
TWO_PI = 2.0 * np.pi
FMAX_HZ = ex.CONSERVATIVE_MAX_LINE_FREQUENCY_HZ
AXIS = {name: index for index, name in enumerate(ex.AXES)}


class LineTable:
    """Collect ``(axis, f_hz, accel_amplitude, phase)`` rows and emit an ``ExcitationProgram``."""

    def __init__(self, name: str, latents: dict):
        self.name, self.latents, self.rows = name, latents, []

    def add(self, axis, f_hz, accel_amp, phase):
        self.rows.append((axis, float(f_hz), float(accel_amp), float(phase)))

    def add_disp(self, axis, f_hz, amplitude, phase):
        """Amplitude in m (translation) or rad (rotation); stored as acceleration A = q * w^2."""
        self.add(axis, f_hz, amplitude * (TWO_PI * f_hz) ** 2, phase)

    def program(
        self,
        level: float = 1.0,
        *,
        t0: float = 0.0,
        seed: int = 0,
        key: str | None = None,
        params: Mapping[str, Any] | None = None,
        nominal_level: float | None = None,
    ) -> ex.ExcitationProgram:
        """Return the program; ``level`` is the total amplitude multiplier actually applied.

        ``t0`` follows the repository convention: stored phases are ``phase + omega * t0`` so that
        ``program(t0).evaluate(t) == program(0).evaluate(t + t0)`` for the carrier.
        """
        level = float(level)
        t0 = float(t0)
        if not math.isfinite(level) or level < 0.0:
            raise ValueError(f"{self.name}: level must be finite and >= 0")
        n_lines = ex.DEFAULT_MAX_LINES
        amp = np.zeros((6, n_lines))
        omega = np.zeros((6, n_lines))
        phase = np.zeros((6, n_lines))
        mask = np.zeros((6, n_lines), bool)
        used = [0] * 6
        for axis, f_hz, accel, ph in self.rows:
            i = AXIS[axis]
            k = used[i]
            used[i] += 1
            if k >= n_lines:
                raise ValueError(f"{self.name}: more than {n_lines} lines on {axis}")
            if not 0.0 < f_hz < FMAX_HZ:
                raise ValueError(f"{self.name}: line {f_hz:.3f} Hz outside (0, {FMAX_HZ}) Hz")
            amp[i, k] = abs(accel) * level
            omega[i, k] = TWO_PI * f_hz
            phase[i, k] = ph + (np.pi if accel < 0 else 0.0) + TWO_PI * f_hz * t0
            mask[i, k] = True
        phase = np.where(mask, (phase + np.pi) % TWO_PI - np.pi, 0.0)
        active = tuple(a for a in ex.AXES if mask[AXIS[a]].any())
        return ex.ExcitationProgram(
            config=ex.DEFAULT_EXCITATION_CONFIG,
            seed=int(seed),
            t0=t0,
            level_scale=level,
            active_axes=active,
            line_accel_amplitude=amp,
            line_omega_rad_s=omega,
            line_phase_at_episode_zero=phase,
            line_mask=mask,
            bands=ex.build_band_table(),
            mode=SCENARIO_MODE,
            mode_params={
                "scenario": key or self.name,
                "level": level if nominal_level is None else float(nominal_level),
                "params": dict(params or {}),
                "latents": self.latents,
            },
        )


def _rng(seed):
    return np.random.default_rng(seed)


def _phase(r):
    return float(r.uniform(-np.pi, np.pi))


def s3_orbital(seed: int, radius_m: float = 0.010) -> LineTable:
    """Orbital shaker / rotating machinery: periodic circular orbit plus small harmonics."""
    r = _rng(seed)
    f0 = float(r.uniform(2.0, 2.9))
    d = 1 if r.uniform() < 0.5 else -1
    p0 = _phase(r)
    L = LineTable("S3_orbital", {"f0_hz": f0, "direction": "cw" if d > 0 else "ccw", "radius_m": radius_m})
    L.add_disp("tx", f0, radius_m, p0)
    L.add_disp("ty", f0, radius_m, p0 + d * np.pi / 2)
    L.add_disp("tx", 2 * f0, 0.06 * radius_m, 2 * p0 + 0.6)
    L.add_disp("ty", 2 * f0, 0.06 * radius_m, 2 * p0 + 0.6 + d * np.pi / 2)
    L.add_disp("tz", 2 * f0, 0.04 * radius_m, 2 * p0 - 0.4)
    if 3 * f0 < 8.7:
        L.add_disp("tx", 3 * f0, 0.02 * radius_m, 3 * p0 + 1.1)
    return L


# ---------------------------------------------------------------------------------------------
# "Response-first" variants: object responses on the table are made deliberately obvious.
# ---------------------------------------------------------------------------------------------


def s1_roll(seed: int, amp_deg: float = 3.5) -> LineTable:
    """Heavy rolling about a random horizontal axis at 1.25-1.5 Hz (pivot below the deck: tilt + sway)."""
    r = _rng(seed)
    fr = float(r.uniform(1.25, 1.5))
    psi = float(r.uniform(0, np.pi))
    p = _phase(r)
    L = LineTable("S1_roll", {"roll_hz": fr, "axis_psi_rad": psi, "amp_deg": amp_deg})
    ax = np.array([np.cos(psi), np.sin(psi)])
    for f, a, ph in ((fr, amp_deg, p), (fr * 1.06, 0.25 * amp_deg, _phase(r))):
        L.add_disp("rx", f, np.radians(a) * ax[0], ph)
        L.add_disp("ry", f, np.radians(a) * ax[1], ph)
    L.add_disp("rz", 0.8 * fr, np.radians(0.15 * amp_deg), _phase(r))
    return L


def s4_press_alt(seed: int, gamma_peak: float = 0.6, rho_h: float = 1.2, zeta: float = 0.10) -> LineTable:
    """Press strokes whose horizontal kick alternates direction (+psi / -psi) every stroke."""
    r = _rng(seed)
    fr = float(r.uniform(0.75, 1.0))
    fn = float(r.uniform(4.5, 5.5))
    psi = float(r.uniform(0, 2 * np.pi))
    p0 = _phase(r)

    def resp(f):
        wr = f / fn
        H = wr**2 / ((1 - wr**2) + 2j * zeta * wr)
        taper = np.where(f > 7.6, np.cos(0.5 * np.pi * (f - 7.6) / 1.0) ** 2, 1.0)
        hp = (f / 2.0) ** 2 / (1 + (f / 2.0) ** 2)
        return np.abs(H) * taper * hp, np.angle(H)

    kv = np.arange(1, 13)
    fv = kv * fr
    kv, fv = kv[fv < 8.6], fv[fv < 8.6]
    kh = np.arange(0, 12) + 0.5
    fh = kh * fr
    kh, fh = kh[fh < 8.6], fh[fh < 8.6]
    av, pv = resp(fv)
    pv = pv + kv * p0 + np.pi / 2
    ah, phh = resp(fh)
    phh = phh + kh * p0 + np.pi / 2
    tt = np.linspace(0, 2 / fr, 6000, endpoint=False)
    wv = (av[:, None] * np.sin(2 * np.pi * fv[:, None] * tt + pv[:, None])).sum(0)
    wh = (ah[:, None] * np.sin(2 * np.pi * fh[:, None] * tt + phh[:, None])).sum(0)
    av = av * gamma_peak * 9.80665 / np.abs(wv).max()
    ah = ah * rho_h * gamma_peak * 9.80665 / np.abs(wh).max()
    L = LineTable(
        "S4_press_alt", {"stroke_hz": fr, "fn_hz": fn, "psi_rad": psi, "gamma_peak": gamma_peak, "rho_h": rho_h}
    )
    for f, a, p in zip(fv, av, pv):
        L.add("tz", f, a, p)
    for f, a, p in zip(fh, ah, phh):
        L.add("tx", f, a * np.cos(psi), p)
        L.add("ty", f, a * np.sin(psi), p)
    return L


def s2_wander(
    seed: int, h_rms: float = 2.2, v_rms: float = 2.6, f_lo: float = 2.0, f_hi: float = 6.0, n: int = 12
) -> LineTable:
    """Rough/gravel road: broadband random kicks in 2-6 Hz on all three translations (objects random-walk).

    Every episode redraws 12 frequencies (log-uniform in [f_lo, f_hi]) and independent phases per axis,
    so the excitation behaves like a Gaussian random process: kicks come at unpredictable times and in
    unpredictable directions. h_rms / v_rms are the horizontal (per axis) / vertical RMS accelerations.
    """
    r = _rng(seed)
    # independent frequency sets per axis: no fixed horizontal/vertical phase relation at a shared
    # frequency, hence no hidden "vibratory conveyor" direction -> a genuine random walk
    fx, fy, fz = (np.sort(np.exp(r.uniform(np.log(f_lo), np.log(f_hi), n))) for _ in range(3))
    L = LineTable(
        "S2_wander",
        {
            "fx_hz": [round(float(x), 4) for x in fx],
            "fy_hz": [round(float(x), 4) for x in fy],
            "fz_hz": [round(float(x), 4) for x in fz],
            "freqs_hz": [round(float(x), 4) for x in fz],
            "h_rms": h_rms,
            "v_rms": v_rms,
        },
    )
    ah = h_rms * np.sqrt(2.0 / n)
    av = v_rms * np.sqrt(2.0 / n)
    for k in range(n):
        L.add("tx", fx[k], ah, _phase(r))
        L.add("ty", fy[k], ah, _phase(r))
        L.add("tz", fz[k], av, _phase(r))
    return L


@dataclass(frozen=True)
class ScenarioSpec:
    builder: Callable[..., LineTable]
    defaults: Mapping[str, Any]
    title: str
    status: str  # recommended | alternative | legacy | rejected


REGISTRY: dict[str, ScenarioSpec] = {
    "S1": ScenarioSpec(s1_roll, {}, "S1 大幅横摇 (heavy rolling)", "canonical"),
    "S2": ScenarioSpec(s2_wander, {"h_rms": 1.98, "v_rms": 2.34}, "S2 粗糙路面 (rough road)", "canonical"),
    "S3": ScenarioSpec(s3_orbital, {"radius_m": 0.010}, "S3 轨道摇床 (orbital shaker, periodic)", "canonical"),
    "S4": ScenarioSpec(
        s4_press_alt, {"gamma_peak": 0.7, "rho_h": 1.5}, "S4 交替冲击 (alternating press shocks)", "canonical"
    ),
}
RECOMMENDED = ("S1", "S2", "S3", "S4")
# convenience views used by the demo tools
SCENARIOS = {key: (lambda seed, _s=spec: _s.builder(seed, **_s.defaults)) for key, spec in REGISTRY.items()}
TITLES = {key: spec.title for key, spec in REGISTRY.items()}


def build_scenario_program(
    scenario: str,
    *,
    seed: int = 0,
    t0: float = 0.0,
    level: float = 1.0,
    params: Mapping[str, Any] | None = None,
    level_scale: float = 1.0,
) -> ex.ExcitationProgram:
    """Build one registered scenario; total amplitude multiplier = ``level * level_scale``."""
    if not isinstance(scenario, str) or scenario not in REGISTRY:
        raise ValueError(f"unknown scenario {scenario!r}; known: {', '.join(sorted(REGISTRY))}")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    for name, value in (("level", level), ("level_scale", level_scale), ("t0", t0)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(f"{name} must be a finite number")
        if name != "t0" and value < 0:
            raise ValueError(f"{name} must be >= 0")
    if not math.isfinite(float(level) * float(level_scale)):
        raise ValueError("level * level_scale must be finite")
    if params is not None and not isinstance(params, Mapping):
        raise ValueError("params must be an object")
    spec = REGISTRY[scenario]
    params = dict(params or {})
    if any(not isinstance(key, str) for key in params):
        raise ValueError("parameter names must be strings")
    allowed = [n for n in inspect.signature(spec.builder).parameters if n != "seed"]
    unknown = sorted(set(params) - set(allowed))
    if unknown:
        raise ValueError(f"unknown parameter(s) for {scenario}: {', '.join(unknown)}; allowed: {', '.join(allowed)}")
    for name, value in params.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(f"parameter {name!r} of {scenario} must be a finite number")
    resolved = {name: p.default for name, p in inspect.signature(spec.builder).parameters.items() if name != "seed"}
    resolved.update(spec.defaults)
    resolved.update(params)
    for name in ("n",):
        if name in resolved:
            count = resolved[name]
            limit = ex.DEFAULT_MAX_LINES
            if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= limit:
                raise ValueError(f"parameter {name!r} of {scenario} must be an integer in [1, {limit}]")
    for name in (
        "amp_deg",
        "radius_m",
        "h_rms",
        "v_rms",
        "gamma_peak",
        "rho_h",
    ):
        if name in resolved and resolved[name] < 0:
            raise ValueError(f"parameter {name!r} of {scenario} must be >= 0")
    if "zeta" in resolved and resolved["zeta"] <= 0:
        raise ValueError(f"parameter 'zeta' of {scenario} must be > 0")
    if "f_lo" in resolved and not 0 < resolved["f_lo"] < resolved["f_hi"] < FMAX_HZ:
        raise ValueError(f"parameters f_lo/f_hi of {scenario} must satisfy 0 < f_lo < f_hi < {FMAX_HZ}")
    table = spec.builder(int(seed), **resolved)
    return table.program(
        float(level) * float(level_scale),
        t0=t0,
        seed=int(seed),
        key=scenario,
        params=params,
        nominal_level=float(level),
    )


def program_from_mode_params(
    mode_params: Mapping[str, Any] | None, *, seed: int = 0, t0: float = 0.0, level_scale: float = 1.0
) -> ex.ExcitationProgram:
    """Entry point used by ``excitation.build_mode_program`` for ``mode == "scenario_lines_v1"``.

    ``mode_params`` = ``{"scenario": str, "level": float = 1.0, "params": {...} = {}}``; a ``latents``
    key written by a previous build (replay) is accepted and ignored because latents follow from the seed.
    """
    if not isinstance(mode_params, Mapping):
        raise ValueError("mode_params must be an object")
    if any(not isinstance(key, str) for key in mode_params):
        raise ValueError("mode parameter names must be strings")
    unknown = sorted(set(mode_params) - {"scenario", "level", "params", "latents"})
    if unknown:
        raise ValueError("unknown mode parameter(s): " + ", ".join(repr(u) for u in unknown))
    scenario = mode_params.get("scenario")
    if not isinstance(scenario, str):
        raise ValueError("mode_params.scenario must name a registered scenario")
    level = mode_params.get("level", 1.0)
    if (
        isinstance(level, bool)
        or not isinstance(level, (int, float))
        or not math.isfinite(float(level))
        or float(level) < 0
    ):
        raise ValueError("mode_params.level must be a finite number >= 0")
    params = mode_params.get("params", {})
    if not isinstance(params, Mapping):
        raise ValueError("mode_params.params must be an object")
    return build_scenario_program(
        scenario, seed=seed, t0=t0, level=float(level), params=params, level_scale=level_scale
    )


__all__ = [
    "SCENARIO_MODE",
    "LineTable",
    "ScenarioSpec",
    "REGISTRY",
    "RECOMMENDED",
    "SCENARIOS",
    "TITLES",
    "build_scenario_program",
    "program_from_mode_params",
    "s3_orbital",
    "s1_roll",
    "s4_press_alt",
    "s2_wander",
]
