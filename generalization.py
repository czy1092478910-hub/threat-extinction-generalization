import numpy as np
import matplotlib.pyplot as plt


def simulate_cs_gs_once(
        n_acq=30, n_ext=40,
        alpha_pos=0.2, alpha_neg=0.02,
        p_us_acq=1.0, p_us_ext=0.0,
        g=0.3
):
    """
    Two-stimulus RW: CS (trained) + GS (generalizes from CS updates)
    r in {0,1}
    g: generalization strength [0,1]
    """
    V_cs = 0.0
    V_gs = 0.0
    hist_cs = []
    hist_gs = []

    # Acquisition (CS paired with US)
    for _ in range(n_acq):
        r = 1.0 if (np.random.rand() < p_us_acq) else 0.0
        # Update CS
        delta_cs = r - V_cs
        alpha = alpha_pos if delta_cs > 0 else alpha_neg
        V_cs += alpha * delta_cs

        # Generalize update to GS (scaled by g)
        delta_gs = r - V_gs
        V_gs += (g * alpha) * delta_gs

        hist_cs.append(V_cs)
        hist_gs.append(V_gs)

    # Extinction (CS no longer paired with US)
    for _ in range(n_ext):
        r = 1.0 if (np.random.rand() < p_us_ext) else 0.0

        delta_cs = r - V_cs
        alpha = alpha_pos if delta_cs > 0 else alpha_neg
        V_cs += alpha * delta_cs

        delta_gs = r - V_gs
        V_gs += (g * alpha) * delta_gs

        hist_cs.append(V_cs)
        hist_gs.append(V_gs)

    return np.array(hist_cs), np.array(hist_gs)


def simulate_mean(n_runs=300, **kwargs):
    cs_runs = []
    gs_runs = []
    for _ in range(n_runs):
        cs, gs = simulate_cs_gs_once(**kwargs)
        cs_runs.append(cs)
        gs_runs.append(gs)
    return np.mean(np.stack(cs_runs, axis=0), axis=0), np.mean(np.stack(gs_runs, axis=0), axis=0)


if __name__ == "__main__":
    n_acq = 30
    n_ext = 40

    # Compare generalization strengths
    g_values = [0.0, 0.2, 0.5, 0.8]

    plt.figure(figsize=(7, 4))
    for g in g_values:
        cs, gs = simulate_mean(
            n_runs=300,
            n_acq=n_acq, n_ext=n_ext,
            alpha_pos=0.2, alpha_neg=0.02,  # PTSD-like safety learning impairment
            p_us_acq=1.0, p_us_ext=0.0,
            g=g
        )
        plt.plot(gs, label=f"GS (g={g})", linewidth=2)

    plt.axvline(n_acq, linestyle="--", color="gray", label="Extinction onset")
    plt.xlabel("Trial")
    plt.ylabel("Threat expectation (GS)")
    plt.title("Threat generalization to GS under impaired safety learning")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Optional: plot CS vs GS for a single g
    g = 0.5
    cs, gs = simulate_mean(
        n_runs=300,
        n_acq=n_acq, n_ext=n_ext,
        alpha_pos=0.2, alpha_neg=0.02,
        p_us_acq=1.0, p_us_ext=0.0,
        g=g
    )
    plt.figure(figsize=(7, 4))
    plt.plot(cs, label="CS+", linewidth=2)
    plt.plot(gs, label=f"GS (g={g})", linewidth=2)
    plt.axvline(n_acq, linestyle="--", color="gray", label="Extinction onset")
    plt.xlabel("Trial")
    plt.ylabel("Threat expectation")
    plt.title("CS+ vs GS (generalization)")
    plt.legend()
    plt.tight_layout()
    plt.show()
