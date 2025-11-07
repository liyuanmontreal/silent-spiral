import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import streamlit as st

# =========================
# Utility functions
# =========================

def build_network(N: int, k: int, p_rewire: float, seed: int = 0):
    G = nx.watts_strogatz_graph(N, k=k, p=p_rewire, seed=seed)
    rng = np.random.default_rng(seed)
    return G, rng

def init_opinions(N: int, minority_ratio: float, rng: np.random.Generator):
    n_min = int(N * minority_ratio)
    n_maj = N - n_min
    maj = rng.normal(0.6, 0.15, n_maj)
    mino = rng.normal(-0.6, 0.15, n_min)
    o = np.concatenate([maj, mino])
    rng.shuffle(o)
    o = np.clip(o, -1.0, 1.0)
    return o

def init_thresholds(N: int, tau_min: float, tau_max: float, rng: np.random.Generator):
    return rng.uniform(tau_min, tau_max, N)

def run_simulation(N, G, o, tau, media_bias, w_media, T, alpha_algo=0.0):
    s = np.zeros(N, dtype=int)
    w_n = 1.0 - w_media
    true_mean = float(o.mean())
    v_hist, bias_hist, silence_hist = [], [], []

    for _ in range(T):
        speak_idx = np.where(s == 1)[0]
        if len(speak_idx) > 0:
            if alpha_algo > 0 and media_bias != 0:
                sign_m = np.sign(media_bias)
                weights = np.exp(alpha_algo * o[speak_idx] * sign_m)
                v = float(np.sum(weights * o[speak_idx]) / np.sum(weights))
            else:
                v = float(o[speak_idx].mean())
        else:
            v = 0.0

        m = media_bias
        hat_m = np.zeros(N)
        for i in range(N):
            neigh = list(G.neighbors(i))
            if not neigh:
                nb_mean = v
            else:
                nb_s = s[neigh]
                sub_idx = np.where(nb_s == 1)[0]
                nb_mean = float(o[neigh][sub_idx].mean()) if len(sub_idx) > 0 else v
            hat_m[i] = (1 - w_media) * nb_mean + w_media * m

        s = (np.abs(o - hat_m) <= tau).astype(int)
        v_hist.append(v)
        bias_hist.append(abs(true_mean - v))
        silence_hist.append(1.0 - s.mean())

    return {
        "s": s,
        "v_hist": np.array(v_hist),
        "bias_hist": np.array(bias_hist),
        "silence_hist": np.array(silence_hist),
        "true_mean": true_mean,
        "final_visible": v_hist[-1] if len(v_hist) > 0 else 0.0
    }

# =========================
# Streamlit layout
# =========================

st.set_page_config(page_title="Spiral of Silence Lab", layout="wide")

st.title("Spiral of Silence Lab ")
st.write(
    "This app simulates the **Spiral of Silence**: when people who perceive themselves "
    "as minorities choose silence, public opinion gradually deviates from the true distribution."
)

tab_sim, tab_exp = st.tabs(["Interactive Simulation", "Experiments"])

# =========================
# Tab 1: Interactive Simulation
# =========================

with tab_sim:
    st.subheader("Interactive Simulation")

    N = st.sidebar.slider("Population Size (N)", 100, 800, 400, 50)
    minority_ratio = st.sidebar.slider("Minority Opinion Ratio", 0.05, 0.5, 0.3, 0.05)
    k = st.sidebar.slider("Average Degree (k)", 4, 40, 10, 2)
    p_rewire = st.sidebar.slider("Network Rewiring Probability", 0.0, 0.5, 0.15, 0.01)

    media_bias = st.sidebar.slider("Media Bias (toward +1)", -1.0, 1.0, 0.3, 0.05)
    w_media = st.sidebar.slider("Weight of Media vs Neighbors", 0.0, 1.0, 0.2, 0.05)

    tau_min = st.sidebar.slider("Tolerance τ_min", 0.0, 0.5, 0.05, 0.01)
    tau_max = st.sidebar.slider("Tolerance τ_max", 0.05, 0.8, 0.35, 0.01)

    T = st.sidebar.slider("Iterations (T)", 5, 100, 40, 5)
    seed = st.sidebar.slider("Random Seed", 0, 9999, 42, 1)
    alpha_algo = st.sidebar.slider("Algorithmic Amplification (α)", 0.0, 3.0, 0.0, 0.1)

    run_button = st.sidebar.button("Run Simulation", type="primary")

    if run_button:
        G, rng = build_network(N, k, p_rewire, seed)
        o = init_opinions(N, minority_ratio, rng)
        tau = init_thresholds(N, tau_min, tau_max, rng)
        result = run_simulation(N, G, o, tau, media_bias, w_media, T, alpha_algo)

        s, v_hist, bias_hist, silence_hist = result["s"], result["v_hist"], result["bias_hist"], result["silence_hist"]
        true_mean, final_visible = result["true_mean"], result["final_visible"]

        col1, col2, col3 = st.columns(3)
        col1.metric("True Mean Opinion", f"{true_mean:.3f}")
        col2.metric("Final Visible Opinion", f"{final_visible:.3f}")
        col3.metric("Final Silence Rate", f"{silence_hist[-1]:.2%}")

        st.markdown("---")
        fig1, ax1 = plt.subplots()
        ax1.plot(bias_hist, label="Bias |true - visible|")
        ax1.plot(silence_hist, label="Silence Rate")
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("Value")
        ax1.legend()
        st.pyplot(fig1)

        fig2, (ax21, ax22) = plt.subplots(1, 2, figsize=(10, 4))
        ax21.hist(o, bins=20)
        ax21.axvline(true_mean, linestyle="--")
        ax21.set_title("True Opinion Distribution")
        speak_idx = np.where(s == 1)[0]
        if len(speak_idx) > 0:
            vis_mean = float(o[speak_idx].mean())
            ax22.hist(o[speak_idx], bins=20)
            ax22.axvline(vis_mean, linestyle="--")
            ax22.set_title("Visible (Speaking) Opinion Distribution")
        else:
            ax22.text(0.5, 0.5, "No one speaks", ha="center", va="center")
            ax22.set_title("Visible Opinion Distribution")
        st.pyplot(fig2)

        if N <= 300:
            st.markdown("**Final Network (blue = speaking, red = silent)**")
            fig3, ax3 = plt.subplots(figsize=(6, 6))
            pos = nx.spring_layout(G, seed=seed)
            colors = ["tab:blue" if s[i] == 1 else "tab:red" for i in range(N)]
            nx.draw(G, pos=pos, node_color=colors, node_size=30, ax=ax3, with_labels=False)
            st.pyplot(fig3)
    else:
        st.info("Adjust parameters and click 'Run Simulation' to start.")

# =========================
# Tab 2: Experiments (simplified for English demo)
# =========================

with tab_exp:
    st.subheader("Experiment Menu")
    st.write("This tab will include automated experiments as described in design document.")
    st.info("Run pre-defined experiment sets (media bias, tolerance, network topology, etc.) coming soon.")
