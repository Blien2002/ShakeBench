from __future__ import annotations

from pathlib import Path
import struct

import pytest

from vibench.panel import linear_rgb_to_srgb, panel_lamp_linear_rgb


def test_panel_lamps_are_dim_active_progressive_and_latched() -> None:
    for kind in ("knob", "lever", "button"):
        off = panel_lamp_linear_rgb(kind, 0.0, active=False, completed=False)
        selected = panel_lamp_linear_rgb(kind, 0.0, active=True, completed=False)
        halfway = panel_lamp_linear_rgb(kind, 0.5, active=True, completed=False)
        done = panel_lamp_linear_rgb(kind, 0.0, active=False, completed=True)
        assert all(0.0 < a < b < c < d <= 1.0 for a, b, c, d in zip(off, selected, halfway, done))


def test_panel_lamp_progress_is_clamped_and_srgb_encoded() -> None:
    assert panel_lamp_linear_rgb("knob", -1.0, active=True, completed=False) == (
        pytest.approx(0.063),
        pytest.approx(0.301),
        pytest.approx(0.119),
    )
    assert panel_lamp_linear_rgb("knob", 2.0, active=True, completed=False) == (
        0.18,
        0.86,
        0.34,
    )
    assert linear_rgb_to_srgb((0.0, 0.0031308, 1.0)) == (
        pytest.approx(0.0),
        pytest.approx(0.040449936),
        pytest.approx(1.0),
    )
    with pytest.raises(ValueError, match="unknown panel control"):
        panel_lamp_linear_rgb("slider", 0.0, active=False, completed=False)


def test_control_visuals_keep_moving_geometry_clear_of_fixed_hardware() -> None:
    root = Path(__file__).resolve().parents[1]
    controls = (root / "src" / "vibench" / "panel_controls.py").read_text(encoding="utf-8")
    appearance = (root / "src" / "vibench" / "visual_assets.py").read_text(encoding="utf-8")

    assert "_visual_binary_stl_mesh" in controls
    assert "APOLLO_KNOB_STL_PATH" in controls
    assert 'f"{link_path}/ApolloKnob"' in controls
    assert "base_normal_m=0.018" in controls
    assert "_visual_apollo_selector" not in controls
    assert 'f"{link_path}/WitnessBand"' in controls
    assert 'f"{link_path}/Dome"' not in controls
    assert 'f"{prim_path}/LeverPivotCap"' in appearance
    assert 'f"{prim_path}/LeverBoot"' not in appearance


def test_original_apollo_knob_stl_and_attribution_are_packaged() -> None:
    root = Path(__file__).resolve().parents[1]
    stl = root / "assets" / "models" / "apollo_command_module_control_panel_knob.stl"
    license_file = stl.with_suffix(".LICENSE.txt")

    data = stl.read_bytes()
    triangle_count = struct.unpack_from("<I", data, 80)[0]
    assert triangle_count == 50_944
    assert len(data) == 84 + 50 * triangle_count

    attribution = license_file.read_text(encoding="utf-8")
    assert "James / Jamesteam" in attribution
    assert "Smithsonian Institution" in attribution
    assert "CC BY-NC 4.0" in attribution
    assert "NoAI" in attribution
