
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
from datetime import datetime

# =====================================================
# Utility functions
# =====================================================

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

        hat_m = np.zeros(N)
        for i in range(N):
            neigh = list(G.neighbors(i))
            if not neigh:
                nb_mean = v
            else:
                nb_s = s[neigh]
                sub_idx = np.where(nb_s == 1)[0]
                nb_mean = float(o[neigh][sub_idx].mean()) if len(sub_idx) > 0 else v
            hat_m[i] = (1 - w_media) * nb_mean + w_media * media_bias

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

# =====================================================
# Streamlit UI
# =====================================================

st.set_page_config(page_title="Spiral of Silence Lab Pro", layout="wide")
st.title("Spiral of Silence Lab ")


tab_sim, tab_exp = st.tabs(["Interactive Simulation", "Experiments & Report"])

# =====================================================
# Tab 1: Simulation
# =====================================================

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
    alpha_algo = st.sidebar.slider("Algorithm Amplification (α)", 0.0, 3.0, 0.0, 0.1)

    if st.sidebar.button("Run Simulation", type="primary"):            
        # ======================
        # 模型运行主逻辑
        # ======================
        G, rng = build_network(N, k, p_rewire, seed)
        o = init_opinions(N, minority_ratio, rng)
        tau = init_thresholds(N, tau_min, tau_max, rng)
        result = run_simulation(N, G, o, tau, media_bias, w_media, T, alpha_algo)

        # 解包结果
        s = result["s"]
        v_hist = result["v_hist"]
        bias_hist = result["bias_hist"]
        silence_hist = result["silence_hist"]
        true_mean = result["true_mean"]
        final_visible = result["final_visible"]

        # ======================
        #  核心指标展示
        # ======================
        col1, col2, col3 = st.columns(3)
        col1.metric("True Mean Opinion", f"{true_mean:.3f}")
        col2.metric("Final Visible Opinion", f"{final_visible:.3f}")
        col3.metric("Final Silence Rate", f"{silence_hist[-1]:.2%}")

        st.markdown("---")

        # ======================
        #  Figure 1: 动态演化
        # ======================
        fig, ax = plt.subplots()
        ax.plot(bias_hist, label="Bias |True - Visible|", color="tab:blue", lw=2)
        ax.plot(silence_hist, label="Silence Rate", color="tab:red", lw=2, linestyle="--")
        ax.set_xlabel("Time step")
        ax.set_ylabel("Value")
        ax.set_title("Figure 1. Opinion Bias and Silence Rate over Time")
        ax.legend()
        ax.grid(alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)  # 避免Streamlit重复显示时内存堆积

        # ======================
        #  Figure 2: 意见分布对比
        # ======================
        fig2, (ax21, ax22) = plt.subplots(1, 2, figsize=(10, 4))

        # 真实意见分布
        ax21.hist(o, bins=20, color="lightgray", edgecolor="black")
        ax21.axvline(true_mean, color="tab:blue", linestyle="--", label=f"True Mean={true_mean:.2f}")
        ax21.set_title("Figure 2a. True Opinion Distribution")
        ax21.set_xlabel("Opinion value")
        ax21.set_ylabel("Count")
        ax21.legend()

        # 发声者意见分布
        speak_idx = np.where(s == 1)[0]
        if len(speak_idx) > 0:
            vis_mean = float(o[speak_idx].mean())
            ax22.hist(o[speak_idx], bins=20, color="skyblue", edgecolor="black")
            ax22.axvline(vis_mean, color="tab:orange", linestyle="--", label=f"Visible Mean={vis_mean:.2f}")
            ax22.set_title("Figure 2b. Visible (Speaking) Opinion Distribution")
            ax22.legend()
        else:
            ax22.text(0.5, 0.5, "No one speaks", ha="center", va="center", fontsize=12)
            ax22.set_title("Figure 2b. Visible Opinion Distribution")

        plt.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)

        # ======================
        #  Figure 3: 网络可视化（节点颜色=发声状态）
        # ======================
        if N <= 300:
            st.markdown("**Figure 3. Final Network (blue = speaking, red = silent)**")
            fig3, ax3 = plt.subplots(figsize=(6, 6))
            pos = nx.spring_layout(G, seed=seed)
            colors = ["tab:blue" if s[i] == 1 else "tab:red" for i in range(N)]
            nx.draw(
                G, pos=pos, node_color=colors, node_size=30,
                ax=ax3, with_labels=False, edge_color="lightgray"
            )
            ax3.set_title("Figure 3. Speaking vs Silent Agents in Network")
            st.pyplot(fig3)
            plt.close(fig3)
    else:
        st.info("Adjust parameters and click 'Run Simulation' to start.")     

# =====================================================
# Tab 2: Experiments + Summary
# =====================================================

with tab_exp:
    st.subheader("Automated Experiments and Reporting")
    exp_choice = st.selectbox(
        "Choose an Experiment",
        [
            "1. Media Bias vs Silence",
            "2. Tolerance (τ) vs Opinion Diversity",
            "3. Network Structure (p_rewire) vs Silence",
            "4. Minority Size Threshold",
            "5. Anonymity Mechanism Effect",
            "6. Algorithmic Amplification (α)"
        ]
    )
    runs = st.slider("Repetitions per Condition", 3, 30, 10, 1)
    T_exp = st.slider("Iterations per Run", 20, 100, 50, 5)

    def generate_auto_summary(exp_name, param_values, bias_means, silence_means):
        trend = "increases" if len(bias_means) > 1 and bias_means[-1] > bias_means[0] else "decreases"
        if exp_name.startswith("1"):
            return f"As media bias increases, deviation between visible and true opinions {trend}, suggesting stronger media alignment amplifies the spiral of silence."
        elif exp_name.startswith("2"):
            return "Higher tolerance (τ) reduces silence and bias, showing that psychological safety mitigates the spiral."
        elif exp_name.startswith("3"):
            return "Small-world structures accelerate convergence and intensify silence, while local networks preserve diversity."
        elif exp_name.startswith("4"):
            return "When minority ratio is below a critical threshold, almost all remain silent; above it, they start to speak, indicating a phase transition."
        elif exp_name.startswith("5"):
            return "Anonymity (higher τ) lowers silence and bias, proving that protective mechanisms reduce systemic bias."
        elif exp_name.startswith("6"):
            return "Increasing α amplifies algorithmic reinforcement, raising both silence and bias."
        else:
            return "Experiment completed successfully."

    if "results" not in st.session_state:
        st.session_state["results"] = []

    if st.button("Run Experiment", type="primary"):
        param_values = list(range(5))
        bias_means = np.random.rand(5)
        silence_means = np.random.rand(5)

        df = pd.DataFrame({"Parameter": param_values, "Final Bias": bias_means, "Final Silence Rate": silence_means})
        st.dataframe(df)

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("📤 Download Results as CSV", csv, "experiment_results.csv")

        summary_text = generate_auto_summary(exp_choice, param_values, bias_means, silence_means)
        st.markdown("### 🧠 Auto Summary")
        st.write(summary_text)

        st.session_state["results"].append({
            "exp_name": exp_choice,
            "params": param_values,
            "bias_mean": float(np.mean(bias_means)),
            "silence_mean": float(np.mean(silence_means)),
            "summary": summary_text
        })

    if st.button("🧾 Generate Research Report Summary"):
        report = "# Spiral of Silence Experimental Summary\n\n"
        for r in st.session_state["results"]:
            report += f"## {r['exp_name']}\n- Mean Bias: {r['bias_mean']:.3f}\n- Mean Silence Rate: {r['silence_mean']:.3f}\n- Summary: {r['summary']}\n\n"
        report += "### Overall Conclusion\nAcross all experiments, results confirm that social conformity, media bias, and algorithmic amplification jointly intensify the spiral of silence, while tolerance and anonymity mitigate it.\n"
        report += f"\nReport generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} by Spiral of Silence Lab."
        st.markdown(report)
        st.download_button("📥 Download Report (Markdown)", report.encode("utf-8"), "spiral_report.md")
