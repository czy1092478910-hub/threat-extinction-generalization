import numpy as np
import matplotlib.pyplot as plt

def simulate_rw_once(n_acq=30, n_ext=40, alpha_pos=0.2, alpha_neg=0.02,
                     p_us_acq=1.0, p_us_ext=0.0):
    V = 0.0
    V_hist = []

    for _ in range(n_acq):
        r = 1.0 if (np.random.rand() < p_us_acq) else 0.0
        delta = r - V
        alpha = alpha_pos if delta > 0 else alpha_neg
        V += alpha * delta
        V_hist.append(V)

    for _ in range(n_ext):
        r = 1.0 if (np.random.rand() < p_us_ext) else 0.0
        delta = r - V
        alpha = alpha_pos if delta > 0 else alpha_neg
        V += alpha * delta
        V_hist.append(V)

    return np.array(V_hist)

def simulate_rw_mean(n_runs=200, **kwargs):
    runs = [simulate_rw_once(**kwargs) for _ in range(n_runs)]
    return np.mean(np.stack(runs, axis=0), axis=0)

if __name__ == "__main__":
    V_normal = simulate_rw_mean(alpha_pos=0.2, alpha_neg=0.2)
    V_ptsd   = simulate_rw_mean(alpha_pos=0.2, alpha_neg=0.02)

    plt.figure(figsize=(7, 4))
    plt.plot(V_normal, label="Normal (α- = 0.2)", linewidth=2)
    plt.plot(V_ptsd,   label="Impaired safety learning (α- = 0.02)", linewidth=2)
    plt.axvline(30, linestyle="--", color="gray", label="Extinction onset")
    plt.xlabel("Trial")
    plt.ylabel("Threat expectation V")
    plt.title("Threat learning & extinction (mean over runs)")
    plt.legend()
    plt.tight_layout()
    plt.show()
