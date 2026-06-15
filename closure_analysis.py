from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "outputs" / "closure_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT))

import generate_field.potential_boundary_reconstruction as pbr  # noqa: E402


PARENT_LENGTHS = (1.30, 0.85, 1.10)
PARENT_CENTER = (0.65, 0.425, 0.55)
PARENT_GRID = (65, 51, 57)
TAU_MAIN = 2.0e-3

POLY_COEFFS = {
    "u": 0.55,
    "v": -0.30,
    "w": 0.20,
    "uv": 0.08,
    "uw": -0.04,
    "vw": 0.06,
    "u2_minus_v2": 0.10,
    "2w2_minus_u2_minus_v2": -0.08,
    "u_4w2_u2_v2": 0.03,
    "v_4w2_u2_v2": -0.02,
    "w_u2_minus_v2": 0.05,
    "uvw": 0.04,
    "u_u2_minus_3v2": -0.03,
    "v_3u2_minus_v2": 0.02,
    "w_2w2_minus_3u2_minus_3v2": 0.01,
}

FOURIER_TERMS = [
    {
        "amplitude": 0.10,
        "alpha": 2.0,
        "beta": 1.0,
        "phase_u": 0.0,
        "phase_v": 0.7,
        "z_parity": "even",
    },
    {
        "amplitude": -0.07,
        "alpha": 3.0,
        "beta": 2.0,
        "phase_u": 0.8,
        "phase_v": 1.6,
        "z_parity": "odd",
    },
    {
        "amplitude": 0.05,
        "alpha": 1.0,
        "beta": 4.0,
        "phase_u": 1.2,
        "phase_v": 0.2,
        "z_parity": "even",
    },
]


def make_world_model() -> pbr.HarmonicFieldModel:
    cfg = pbr.HarmonicWorldConfig(
        Lx=PARENT_LENGTHS[0],
        Ly=PARENT_LENGTHS[1],
        Lz=PARENT_LENGTHS[2],
        center=PARENT_CENTER,
        E0=1.0,
        seed=12,
        poly_scale=0.18,
        fourier_scale=0.12,
        n_fourier_terms=5,
        use_polynomial=True,
        use_fourier=True,
    )
    return pbr.create_harmonic_field_model(cfg, poly_coefficients=POLY_COEFFS, fourier_terms=FOURIER_TERMS)


def make_variant(seed: int, scale_multiplier: float) -> pbr.HarmonicFieldModel:
    cfg = pbr.HarmonicWorldConfig(
        Lx=PARENT_LENGTHS[0],
        Ly=PARENT_LENGTHS[1],
        Lz=PARENT_LENGTHS[2],
        center=PARENT_CENTER,
        E0=1.0,
        seed=seed,
        poly_scale=0.18 * scale_multiplier,
        fourier_scale=0.12 * scale_multiplier,
        n_fourier_terms=5,
        use_polynomial=True,
        use_fourier=True,
    )
    return pbr.create_harmonic_field_model(
        cfg,
        poly_coefficients=pbr.sample_poly_coefficients(seed=seed, scale=cfg.poly_scale),
        fourier_terms=pbr.sample_fourier_terms(seed=seed, n_terms=cfg.n_fourier_terms, scale=cfg.fourier_scale),
    )


def recompute_tau_selection(shape_sweep: dict[str, object], tau_values: list[float]) -> list[dict[str, float | bool]]:
    rows = []
    for tau in tau_values:
        candidates = []
        for detail in shape_sweep["details"]:
            record = dict(detail["record"])
            local_values = np.asarray(detail["robust_error_values"], dtype=float)
            pass_fraction = float(np.mean(local_values <= tau))
            record["candidate_pass_fraction"] = pass_fraction
            record["eta_tau"] = float(record["volume_fraction"] * pass_fraction)
            record["epsilon_E95"] = float(np.percentile(local_values, 95))
            candidates.append(record)
        feasible = [row for row in candidates if float(row["epsilon_E95"]) <= tau]
        source = feasible if feasible else candidates
        best = max(
            source,
            key=lambda row: (
                float(row["eta_tau"]),
                -float(row["epsilon_E95"]),
                -float(row["C_omega"]),
            ),
        )
        rows.append(
            {
                "tau_E_local": float(tau),
                "num_feasible": float(len(feasible)),
                "used_fallback": len(feasible) == 0,
                "candidate_id": float(best["candidate_id"]),
                "volume_fraction": float(best["volume_fraction"]),
                "eta_tau": float(best["eta_tau"]),
                "candidate_pass_fraction": float(best["candidate_pass_fraction"]),
                "epsilon_E95": float(best["epsilon_E95"]),
                "epsilon_E99": float(best["epsilon_E99"]),
                "C_omega": float(best["C_omega"]),
                "D_omega_E_over_Erms2_mean": float(best["D_omega_E_over_Erms2_mean"]),
                "nu_max": float(best["nu_max"]),
                "alpha1": float(best["alpha1"]),
                "alpha2": float(best["alpha2"]),
                "p": float(best["p"]),
                "rotation_z": float(best["rotation_z"]),
                "box_Lx": float(best["box_Lx"]),
                "box_Ly": float(best["box_Ly"]),
                "box_Lz": float(best["box_Lz"]),
            }
        )
    return rows


def update_candidate_grid(candidate: dict[str, float | str], grid_shape: tuple[int, int, int]) -> dict[str, float | str]:
    out = dict(candidate)
    out["grid_nx"] = float(grid_shape[0])
    out["grid_ny"] = float(grid_shape[1])
    out["grid_nz"] = float(grid_shape[2])
    return out


def candidate_mask_and_fields(
    model: pbr.HarmonicFieldModel,
    candidate: dict[str, float | str],
    backend: str = "auto",
    boundary_grid: np.ndarray | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray]:
    box_lengths = (float(candidate["box_Lx"]), float(candidate["box_Ly"]), float(candidate["box_Lz"]))
    grid_shape = (int(candidate["grid_nx"]), int(candidate["grid_ny"]), int(candidate["grid_nz"]))
    axes = (float(candidate["axis_a"]), float(candidate["axis_b"]), float(candidate["axis_c"]))
    center = (
        float(candidate.get("center_x", PARENT_CENTER[0])),
        float(candidate.get("center_y", PARENT_CENTER[1])),
        float(candidate.get("center_z", PARENT_CENTER[2])),
    )
    target = pbr.sample_model_on_box(model, box_lengths, grid_shape, center)
    used_boundary = target["phi"] if boundary_grid is None else boundary_grid
    solver = pbr.SpectralDirichletBoxSolver(grid_shape, box_lengths, backend=backend)
    reconstructed_phi = solver.solve(np.asarray(used_boundary, dtype=float))
    reconstructed = {
        **target,
        "phi": reconstructed_phi,
        "E": pbr.compute_electric_field_from_potential(reconstructed_phi, target["x"], target["y"], target["z"]),
    }
    full_mask = pbr.superellipsoid_mask(
        np.asarray(target["grid"], dtype=float),
        axes,
        float(candidate["p"]),
        center,
        float(candidate.get("rotation_z", 0.0)),
    )
    return target, reconstructed, pbr.trim_mask_by_layers(full_mask, 1)


def rms(values: np.ndarray, mask: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(values, dtype=float)[mask] ** 2)))


def validation_metrics(target: dict[str, np.ndarray], reconstructed: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, float]:
    x = np.asarray(target["x"], dtype=float)
    y = np.asarray(target["y"], dtype=float)
    z = np.asarray(target["z"], dtype=float)
    E = np.asarray(target["E"], dtype=float)
    Ex, Ey, Ez = E
    dEx_dx = np.gradient(Ex, x, axis=0, edge_order=2)
    dEy_dy = np.gradient(Ey, y, axis=1, edge_order=2)
    dEz_dz = np.gradient(Ez, z, axis=2, edge_order=2)
    div = dEx_dx + dEy_dy + dEz_dz
    curl_x = np.gradient(Ez, y, axis=1, edge_order=2) - np.gradient(Ey, z, axis=2, edge_order=2)
    curl_y = np.gradient(Ex, z, axis=2, edge_order=2) - np.gradient(Ez, x, axis=0, edge_order=2)
    curl_z = np.gradient(Ey, x, axis=0, edge_order=2) - np.gradient(Ex, y, axis=1, edge_order=2)
    curl_mag = np.sqrt(curl_x * curl_x + curl_y * curl_y + curl_z * curl_z)
    laplace = pbr.compute_laplacian(reconstructed["phi"], reconstructed["x"], reconstructed["y"], reconstructed["z"])
    recon_metrics = pbr.compute_masked_reconstruction_metrics(target, reconstructed, mask, error_threshold=TAU_MAIN)
    length_ref = float(max(target["box_lengths"]))
    e_rms = float(recon_metrics["E_rms"])
    phi_centered = np.asarray(target["phi"], dtype=float) - float(np.mean(target["phi"]))
    phi_rms = float(np.sqrt(np.mean(phi_centered[mask] ** 2)))
    return {
        "target_div_rms": rms(div, mask),
        "target_curl_rms": rms(curl_mag, mask),
        "reconstructed_laplace_rms": rms(laplace, mask),
        "target_div_rms_normalized": rms(div, mask) / max(e_rms / length_ref, 1e-15),
        "target_curl_rms_normalized": rms(curl_mag, mask) / max(e_rms / length_ref, 1e-15),
        "reconstructed_laplace_rms_normalized": rms(laplace, mask) / max(phi_rms / (length_ref * length_ref), 1e-15),
        "max_principle_violation": float(
            pbr.compute_reconstruction_metrics(target, reconstructed, compare_trim=1)["max_principle_violation"]
        ),
        "mean_cosine_similarity": float(recon_metrics["mean_cosine_similarity"]),
        "epsilon_E95": float(recon_metrics["epsilon_E95"]),
        "E_relative_l2": float(recon_metrics["E_relative_l2"]),
    }


def quantize_boundary(phi: np.ndarray, bits: int | None = None, step: float | None = None) -> tuple[np.ndarray, float]:
    out = np.array(phi, copy=True, dtype=float)
    mask = pbr.boundary_mask(out.shape)
    values = out[mask]
    vmin = float(values.min())
    vmax = float(values.max())
    if bits is not None:
        levels = 2**int(bits)
        used_step = (vmax - vmin) / max(levels - 1, 1)
    elif step is not None:
        used_step = float(step)
    else:
        raise ValueError("bits or step must be provided.")
    if used_step <= 0.0:
        return out, 0.0
    out[mask] = vmin + np.round((values - vmin) / used_step) * used_step
    return out, used_step


def plot_patch_sweep(records: list[dict[str, float]], path: Path) -> None:
    patch = np.array([row["patch_y"] for row in records], dtype=float)
    e95 = np.array([row["epsilon_E95"] for row in records], dtype=float)
    l2 = np.array([row["E_relative_l2"] for row in records], dtype=float)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(patch, e95, marker="o", label="epsilon_E95")
    ax.plot(patch, l2, marker="s", label="E_relative_l2")
    ax.axhline(TAU_MAIN, color="tab:red", linestyle="--", label="tau_E,local = 2e-3")
    ax.set_yscale("log")
    ax.set_xlabel("Patch resolution per face direction")
    ax.set_ylabel("Error")
    ax.set_title("High-resolution selected-shape patch sweep")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_tau_sensitivity(rows: list[dict[str, float | bool]], path: Path) -> None:
    tau = np.array([row["tau_E_local"] for row in rows], dtype=float)
    eta = np.array([row["eta_tau"] for row in rows], dtype=float)
    e95 = np.array([row["epsilon_E95"] for row in rows], dtype=float)
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(tau, eta, marker="o", color="tab:blue", label="selected eta_tau")
    ax1.set_xscale("log")
    ax1.set_xlabel("tau_E,local")
    ax1.set_ylabel("eta_tau", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax2 = ax1.twinx()
    ax2.plot(tau, e95, marker="s", color="tab:orange", label="selected epsilon_E95")
    ax2.set_ylabel("epsilon_E95", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")
    ax1.grid(True, which="both", alpha=0.3)
    fig.suptitle("Tolerance sensitivity of robust effective volume")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_bar_metric(rows: list[dict[str, float | str]], key: str, label_key: str, title: str, path: Path) -> None:
    labels = [str(row[label_key]) for row in rows]
    vals = np.array([float(row[key]) for row in rows], dtype=float)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(labels, vals, color="#3A6EA5")
    ax.axhline(TAU_MAIN, color="tab:red", linestyle="--", linewidth=1.5)
    ax.set_ylabel(key)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> dict[str, object]:
    base_model = make_world_model()
    robust_family = [base_model, make_variant(21, 0.85), make_variant(35, 1.10)]

    specs = pbr.build_superellipsoid_candidate_specs(
        volume_fractions=[0.75, 0.55, 0.40, 0.28],
        alpha1_values=[0.65, 0.80, 1.00, 1.25],
        alpha2_values=[0.75, 0.90, 1.15],
        p_values=[4.0, 8.0, 16.0],
        rotation_z_values=[0.0, np.pi / 6.0],
    )
    shape_sweep = pbr.run_superellipsoid_shape_sweep(
        robust_family,
        specs,
        parent_lengths=PARENT_LENGTHS,
        parent_center=PARENT_CENTER,
        reference_grid_shape=(25, 21, 23),
        error_threshold=TAU_MAIN,
        backend="auto",
        compare_trim=1,
        min_grid_points=9,
        keep_details=True,
    )
    selection = pbr.select_effective_volume_optimum(shape_sweep["records"], error_threshold=TAU_MAIN)
    best = dict(selection["best_row"])

    tau_sensitivity = recompute_tau_selection(shape_sweep, [1.0e-3, 2.0e-3, 5.0e-3])

    reference_spacing = tuple(PARENT_LENGTHS[i] / (PARENT_GRID[i] - 1) for i in range(3))
    highres_grid = pbr.grid_shape_for_box_lengths(
        (float(best["box_Lx"]), float(best["box_Ly"]), float(best["box_Lz"])),
        reference_spacing,
        min_points=35,
    )
    highres_best = update_candidate_grid(best, highres_grid)

    target, reconstructed, mask = candidate_mask_and_fields(base_model, highres_best, backend="auto")
    highres_baseline = pbr.compute_masked_reconstruction_metrics(target, reconstructed, mask, error_threshold=TAU_MAIN)
    verification = validation_metrics(target, reconstructed, mask)

    patch_shapes = [(n, n) for n in [12, 16, 20, 24, 28, 32, 36, 40, 44, 48] if n < min(highres_grid)]
    highres_patch = pbr.evaluate_shape_patch_requirement(
        base_model,
        highres_best,
        parent_center=PARENT_CENTER,
        patch_shapes=patch_shapes,
        error_threshold=TAU_MAIN,
        backend="auto",
        compare_trim=1,
    )

    highres_noise = pbr.evaluate_shape_noise_gain(
        base_model,
        highres_best,
        parent_center=PARENT_CENTER,
        error_threshold=TAU_MAIN,
        noise_rms_fraction=1.0e-3,
        n_trials=6,
        backend="auto",
        compare_trim=1,
        seed=777,
    )

    quantization_rows = []
    for bits in [8, 10, 12]:
        q_boundary, step = quantize_boundary(target["phi"], bits=bits)
        _, q_reconstructed, q_mask = candidate_mask_and_fields(base_model, highres_best, backend="auto", boundary_grid=q_boundary)
        metrics = pbr.compute_masked_reconstruction_metrics(target, q_reconstructed, q_mask, error_threshold=TAU_MAIN)
        quantization_rows.append(
            {
                "case": f"{bits}-bit",
                "bits": float(bits),
                "voltage_step": float(step),
                "epsilon_E95": float(metrics["epsilon_E95"]),
                "E_relative_l2": float(metrics["E_relative_l2"]),
                "local_pass_fraction": float(metrics["local_pass_fraction"]),
            }
        )
    q_boundary, step = quantize_boundary(target["phi"], step=0.01)
    _, q_reconstructed, q_mask = candidate_mask_and_fields(base_model, highres_best, backend="auto", boundary_grid=q_boundary)
    metrics = pbr.compute_masked_reconstruction_metrics(target, q_reconstructed, q_mask, error_threshold=TAU_MAIN)
    quantization_rows.append(
        {
            "case": "10 mV step",
            "bits": float("nan"),
            "voltage_step": float(step),
            "epsilon_E95": float(metrics["epsilon_E95"]),
            "E_relative_l2": float(metrics["E_relative_l2"]),
            "local_pass_fraction": float(metrics["local_pass_fraction"]),
        }
    )

    ensemble_models = [
        ("base-12", base_model),
        ("seed-21", make_variant(21, 0.85)),
        ("seed-35", make_variant(35, 1.10)),
        ("seed-47", make_variant(47, 0.95)),
        ("seed-63", make_variant(63, 1.05)),
        ("seed-88", make_variant(88, 1.15)),
    ]
    ensemble_rows = []
    for label, model in ensemble_models:
        t, r, m = candidate_mask_and_fields(model, highres_best, backend="auto")
        metrics = pbr.compute_masked_reconstruction_metrics(t, r, m, error_threshold=TAU_MAIN)
        ensemble_rows.append(
            {
                "model": label,
                "epsilon_E95": float(metrics["epsilon_E95"]),
                "E_relative_l2": float(metrics["E_relative_l2"]),
                "local_pass_fraction": float(metrics["local_pass_fraction"]),
                "mean_cosine_similarity": float(metrics["mean_cosine_similarity"]),
            }
        )

    patch_png = OUT_DIR / "highres_patch_sweep.png"
    tau_png = OUT_DIR / "tau_sensitivity.png"
    quant_png = OUT_DIR / "voltage_quantization.png"
    ensemble_png = OUT_DIR / "ensemble_validation.png"
    plot_patch_sweep(highres_patch["records"], patch_png)
    plot_tau_sensitivity(tau_sensitivity, tau_png)
    plot_bar_metric(quantization_rows, "epsilon_E95", "case", "Voltage quantization sensitivity", quant_png)
    plot_bar_metric(ensemble_rows, "epsilon_E95", "model", "Selected-shape ensemble validation", ensemble_png)

    summary = {
        "tau_main": TAU_MAIN,
        "selected_shape_original_grid": {
            key: float(best[key])
            for key in [
                "candidate_id",
                "volume_fraction",
                "eta_tau",
                "epsilon_E95",
                "C_omega",
                "D_omega_E_over_Erms2_mean",
                "nu_max",
                "alpha1",
                "alpha2",
                "p",
                "rotation_z",
                "axis_a",
                "axis_b",
                "axis_c",
                "box_Lx",
                "box_Ly",
                "box_Lz",
                "grid_nx",
                "grid_ny",
                "grid_nz",
            ]
        },
        "selected_shape_highres_grid": {
            "grid_nx": float(highres_grid[0]),
            "grid_ny": float(highres_grid[1]),
            "grid_nz": float(highres_grid[2]),
            "epsilon_E95": float(highres_baseline["epsilon_E95"]),
            "E_relative_l2": float(highres_baseline["E_relative_l2"]),
            "local_pass_fraction": float(highres_baseline["local_pass_fraction"]),
        },
        "tau_sensitivity": tau_sensitivity,
        "highres_patch_sweep": highres_patch["records"],
        "highres_patch_N_tau": highres_patch["N_tau"],
        "highres_patch_shape_tau": highres_patch["patch_shape_tau"],
        "highres_noise": highres_noise,
        "voltage_quantization": quantization_rows,
        "ensemble_validation": ensemble_rows,
        "verification": verification,
        "figures": {
            "highres_patch_sweep": str(patch_png),
            "tau_sensitivity": str(tau_png),
            "voltage_quantization": str(quant_png),
            "ensemble_validation": str(ensemble_png),
        },
    }
    (OUT_DIR / "closure_analysis.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


if __name__ == "__main__":
    main()
