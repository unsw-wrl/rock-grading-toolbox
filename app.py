import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from io import StringIO

# -----------------------------
# CONSTANTS
# -----------------------------
rho_s = 2.65  # g/cm^3


# -----------------------------
# CORE PROCESSING
# -----------------------------
def process_distribution(csv_text, total_mass_kg, shape_factor):
    numeric_rows = []
    for line in str(csv_text).splitlines():
        vals = re.findall(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", line)
        if len(vals) < 2:
            continue
        numeric_rows.append((float(vals[0]), float(vals[1])))

    if not numeric_rows:
        raise ValueError(
            "Could not parse CSV: expected lines containing at least two numeric values "
            "(mass_g, mass_fraction_percent)."
        )

    df = pd.DataFrame(numeric_rows, columns=["mass_g", "mass_fraction_percent"])

    df = df[df["mass_g"] > 0].copy()
    if df.empty:
        raise ValueError("No positive mass rows found in CSV.")

    df["mass_fraction"] = df["mass_fraction_percent"] / 100.0
    frac_sum = df["mass_fraction"].sum()
    if frac_sum <= 0:
        raise ValueError("CSV mass fractions sum to zero or negative.")
    df["mass_fraction"] /= frac_sum

    V = df["mass_g"] / rho_s
    D_sphere_cm = ((6 * V) / np.pi) ** (1 / 3)
    D_sphere_mm = D_sphere_cm * 10.0
    df["Dn_mm"] = shape_factor * D_sphere_mm

    df["actual_mass_kg"] = df["mass_fraction"] * total_mass_kg

    return df[["mass_g", "Dn_mm", "actual_mass_kg"]]


def process_manual_add(total_mass_kg, dn_min_mm, dn_max_mm, shape_factor, n_points=30):
    n_points = max(2, int(n_points))
    dn_min = float(dn_min_mm)
    dn_max = float(dn_max_mm)
    if dn_max < dn_min:
        dn_min, dn_max = dn_max, dn_min

    shape = float(shape_factor)
    if shape <= 0:
        raise ValueError("Manual add shape_factor must be > 0.")

    dn = np.linspace(dn_min, dn_max, n_points)
    mass_each_kg = float(total_mass_kg) / n_points
    actual_mass_kg = np.full(n_points, mass_each_kg, dtype=float)

    # Convert nominal diameter back to mass so mass-based curves can be built.
    d_sphere_cm = (dn / shape) / 10.0
    vol_cm3 = (np.pi / 6.0) * (d_sphere_cm ** 3)
    mass_g = rho_s * vol_cm3

    return pd.DataFrame(
        {
            "mass_g": mass_g,
            "Dn_mm": dn,
            "actual_mass_kg": actual_mass_kg,
        }
    )


def build_passing_curve(df, x_col):
    out = (
        df.groupby(x_col, as_index=False)["actual_mass_kg"]
        .sum()
        .sort_values(x_col)
    )
    total = out["actual_mass_kg"].sum()
    out["mass_fraction"] = out["actual_mass_kg"] / total
    out["pct_passing"] = 100.0 * out["mass_fraction"].cumsum()
    return out


def project_simplex_with_lower_bounds(v, lb):
    v = np.asarray(v, dtype=float)
    lb = np.asarray(lb, dtype=float)
    residual = 1.0 - float(lb.sum())

    if residual <= 0:
        out = lb.copy()
        s = out.sum()
        return out / s if s > 0 else np.full_like(out, 1.0 / len(out))

    z = v - lb
    n = len(z)
    s = np.sort(z)[::-1]
    cssv = np.cumsum(s)
    rho = np.where(s * np.arange(1, n + 1) > (cssv - residual))[0]
    theta = 0.0 if len(rho) == 0 else (cssv[rho[-1]] - residual) / float(rho[-1] + 1)
    u = np.maximum(z - theta, 0.0)
    return u + lb


def build_target_cdf_grid(dn_grid, d1, d50, d99):
    x0 = float(np.min(dn_grid))
    x4 = float(np.max(dn_grid))
    xk = np.array([x0, d1, d50, d99, x4], dtype=float)
    yk = np.array([0.0, 0.01, 0.50, 0.99, 1.0], dtype=float)
    return np.interp(dn_grid, xk, yk, left=0.0, right=1.0)


def fit_source_weights_cdf(
    source_cdf_mat,
    target_cdf,
    min_source_fraction=0.0,
    diversity_penalty=0.0,
    max_iter=1500,  # reduced for browser
    learning_rate=0.2,
):
    S = np.asarray(source_cdf_mat, dtype=float)
    y = np.asarray(target_cdf, dtype=float)

    n_sources = S.shape[1]
    lb = np.full(n_sources, float(min_source_fraction))
    w = np.full(n_sources, 1.0 / n_sources)
    w = project_simplex_with_lower_bounds(w, lb)

    center = np.full(n_sources, 1.0 / n_sources)
    lam = float(diversity_penalty)

    for _ in range(int(max_iter)):
        pred = S @ w
        grad = (2.0 / len(y)) * (S.T @ (pred - y))
        if lam > 0:
            grad += 2.0 * lam * (w - center)

        w_new = project_simplex_with_lower_bounds(w - learning_rate * grad, lb)

        if np.linalg.norm(w_new - w, ord=1) < 1e-9:
            break

        w = w_new

    mse = float(np.mean((S @ w - y) ** 2))
    return w, mse


# -----------------------------
# MAIN ENTRY FUNCTION
# -----------------------------
def run_model(config):
    # Support both full config {"distributions": {...}} and raw distributions dict
    if "distributions" in config:
        distributions = config.get("distributions", {})
    else:
        distributions = config

    target_mix = config.get("target_mix", {})
    manual_adds = config.get("manual_adds", {})
    do_plots = bool(config.get("do_plots", False))

    combined_list = []
    mass_curves_individual = []
    dn_curves_individual = []
    source_rows = []

    # -----------------------------
    # Process CSV Distributions
    # -----------------------------
    for label, cfg in distributions.items():

        df_processed = process_distribution(
            cfg["csv_text"],
            float(cfg["total_mass_kg"]),
            float(cfg["shape_factor"]),
        )

        mass_curves_individual.append(
            (label, build_passing_curve(df_processed, "mass_g"))
        )
        dn_curves_individual.append(
            (label, build_passing_curve(df_processed, "Dn_mm"))
        )

        combined_list.append(df_processed)

        source_rows.append(
            {
                "label": label,
                "df": df_processed.copy(),
                "current_mass_kg": float(cfg["total_mass_kg"]),
            }
        )

    # -----------------------------
    # Process Manual Additions
    # -----------------------------
    for label, cfg in manual_adds.items():
        df_manual = process_manual_add(
            total_mass_kg=float(cfg["total_mass_kg"]),
            dn_min_mm=float(cfg["dn_min_mm"]),
            dn_max_mm=float(cfg["dn_max_mm"]),
            shape_factor=float(cfg["shape_factor"]),
            n_points=int(cfg.get("n_points", 30)),
        )

        mass_curves_individual.append(
            (str(label), build_passing_curve(df_manual, "mass_g"))
        )
        dn_curves_individual.append(
            (str(label), build_passing_curve(df_manual, "Dn_mm"))
        )
        combined_list.append(df_manual)
        source_rows.append(
            {
                "label": str(label),
                "df": df_manual.copy(),
                "current_mass_kg": float(cfg["total_mass_kg"]),
            }
        )

    if len(combined_list) == 0:
        raise ValueError("No valid distributions were provided.")

    combined_df = pd.concat(combined_list)

    mass_df = build_passing_curve(combined_df, "mass_g")
    dn_df = build_passing_curve(combined_df, "Dn_mm")

    # -----------------------------
    # TARGET MIX SOLVER
    # -----------------------------
    final_mix_dn_df = None
    w = None
    fit_err = None

    if target_mix.get("enabled", False) and len(source_rows) > 0:

        dn_all = np.concatenate(
            [src["df"]["Dn_mm"].to_numpy() for src in source_rows]
        )

        dn_grid = np.linspace(np.min(dn_all), np.max(dn_all), 180)

        target_cdf_grid = build_target_cdf_grid(
            dn_grid,
            target_mix["Dn1_mm"],
            target_mix["Dn50_mm"],
            target_mix["Dn99_mm"],
        )

        source_cdf_mat = np.zeros((len(dn_grid), len(source_rows)))

        for j, src in enumerate(source_rows):
            curve = build_passing_curve(src["df"], "Dn_mm")
            x = curve["Dn_mm"].to_numpy()
            y = curve["pct_passing"].to_numpy() / 100.0
            source_cdf_mat[:, j] = np.interp(dn_grid, x, y)

        w, fit_err = fit_source_weights_cdf(
            source_cdf_mat,
            target_cdf_grid,
            min_source_fraction=float(target_mix.get("min_source_fraction", 0.0)),
            diversity_penalty=float(target_mix.get("diversity_penalty", 0.0)),
        )

        print("=== Target Mix Result ===")
        print("Fit MSE:", fit_err)

        for src, weight in zip(source_rows, w):
            print(src["label"], "->", round(100 * weight, 2), "%")

        # Build final recommended distribution
        frames = []
        for src, weight in zip(source_rows, w):
            dfw = src["df"].copy()
            dfw["actual_mass_kg"] *= weight
            frames.append(dfw)

        final_mix_df = pd.concat(frames)
        final_mix_dn_df = build_passing_curve(final_mix_df, "Dn_mm")

    # -----------------------------
    # PLOTS
    # -----------------------------
    if do_plots:
        plt.figure()
        for label, curve in mass_curves_individual:
            plt.plot(curve["mass_g"], curve["pct_passing"], label=label)
        plt.plot(mass_df["mass_g"], mass_df["pct_passing"], linewidth=2)
        plt.xlabel("Mass (g)")
        plt.ylabel("Percentage Passing (%)")
        plt.title("Mass Distribution")
        plt.legend()
        plt.show()

        plt.figure()
        for label, curve in dn_curves_individual:
            plt.plot(curve["Dn_mm"], curve["pct_passing"], label=label)
        plt.plot(dn_df["Dn_mm"], dn_df["pct_passing"], linewidth=2)

        if final_mix_dn_df is not None:
            plt.plot(
                final_mix_dn_df["Dn_mm"],
                final_mix_dn_df["pct_passing"],
                linewidth=3,
            )

        plt.xlabel("Equivalent Nominal Diameter Dn (mm)")
        plt.ylabel("Percentage Passing (%)")
        plt.title("Nominal Diameter Distribution")
        plt.legend()
        plt.show()

    result = {
        "sources": [src["label"] for src in source_rows],
        "mass_curve": mass_df[["mass_g", "pct_passing"]].to_dict(orient="records"),
        "dn_curve": dn_df[["Dn_mm", "pct_passing"]].to_dict(orient="records"),
        "source_mass_curves": {
            label: curve[["mass_g", "pct_passing"]].to_dict(orient="records")
            for label, curve in mass_curves_individual
        },
        "source_dn_curves": {
            label: curve[["Dn_mm", "pct_passing"]].to_dict(orient="records")
            for label, curve in dn_curves_individual
        },
    }

    if w is not None and fit_err is not None:
        result["target_mix_fit_mse"] = float(fit_err)
        result["target_mix_weights"] = {
            src["label"]: float(weight)
            for src, weight in zip(source_rows, w)
        }

    if final_mix_dn_df is not None:
        result["final_mix_dn_curve"] = final_mix_dn_df[
            ["Dn_mm", "pct_passing"]
        ].to_dict(orient="records")

    return result
