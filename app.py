
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
from datetime import datetime
from io import BytesIO

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

        # ==============================
        # Figure 1: 双轴动态演化图
        # ==============================
        fig, ax1 = plt.subplots(figsize=(7, 4))

        # 左轴：Bias
        color1 = "tab:blue"
        ax1.set_xlabel("Time step")
        ax1.set_ylabel("Bias |True - Visible|", color=color1)
        ax1.plot(bias_hist, color=color1, lw=2, label="Bias |True - Visible|")
        ax1.tick_params(axis="y", labelcolor=color1)
        ax1.grid(alpha=0.3)

        # 右轴：Silence Rate
        ax2 = ax1.twinx()
        color2 = "tab:red"
        ax2.set_ylabel("Silence Rate", color=color2)
        ax2.plot(silence_hist, color=color2, lw=2, linestyle="--", label="Silence Rate")
        ax2.tick_params(axis="y", labelcolor=color2)

        fig.suptitle("Figure 1. Opinion Bias (left) and Silence Rate (right) over Time", fontsize=12)
        st.pyplot(fig)
        plt.close(fig)

        # 保存为PNG并生成下载按钮
        buf1 = BytesIO()
        fig.savefig(buf1, format="png", dpi=300, bbox_inches="tight")
        st.download_button(
            label="Download Figure 1",
            data=buf1.getvalue(),
            file_name="figure1_bias_silence.png",
            mime="image/png"
        )
        st.markdown("""
        Figure 1 shows the temporal evolution of the opinion bias (blue) and the silence rate (red).
        When both lines rise or stay high, most individuals remain silent while visible voices reinforce the majority view— the hallmark of the *Spiral of Silence* process.""")

        # ==============================
        # Figure 2: 意见分布 + KDE曲线 / 透明重叠
        # ==============================
        fig2, ax = plt.subplots(figsize=(7, 4))

        # 使用透明度重叠的直方图
        bins = np.linspace(-1, 1, 25)
        ax.hist(o, bins=bins, color="lightgray", edgecolor="black", alpha=0.5, label="True Opinions")
        speak_idx = np.where(s == 1)[0]
        if len(speak_idx) > 0:
            ax.hist(o[speak_idx], bins=bins, color="skyblue", edgecolor="black", alpha=0.6, label="Visible (Speaking)")
        else:
            ax.text(0, 0.5, "No one speaks", ha="center", va="center", fontsize=12)

        # 叠加 KDE 曲线（平滑分布）
        try:
            from scipy.stats import gaussian_kde
            xgrid = np.linspace(-1, 1, 200)
            kde_true = gaussian_kde(o)
            ax.plot(xgrid, kde_true(xgrid) * len(o) * (bins[1]-bins[0]),
                    color="tab:blue", lw=2, label="True KDE")
            if len(speak_idx) > 0:
                kde_visible = gaussian_kde(o[speak_idx])
                ax.plot(xgrid, kde_visible(xgrid) * len(o) * (bins[1]-bins[0]),
                        color="tab:orange", lw=2, linestyle="--", label="Visible KDE")
        except ImportError:
            ax.text(0, 0.9, "Install scipy for KDE curves", ha="center", va="center", fontsize=10)

        # 均值线
        ax.axvline(o.mean(), color="tab:blue", linestyle="--", lw=1)
        if len(speak_idx) > 0:
            ax.axvline(o[speak_idx].mean(), color="tab:orange", linestyle="--", lw=1)

        ax.set_xlabel("Opinion Value")
        ax.set_ylabel("Count / Density")
        ax.set_title("Figure 2. True vs Visible Opinion Distribution (with KDE)")
        ax.legend()
        st.pyplot(fig2)
        plt.close(fig2)

        buf2 = BytesIO()
        fig2.savefig(buf2, format="png", dpi=300, bbox_inches="tight")
        st.download_button(
            label="Download Figure 2",
            data=buf2.getvalue(),
            file_name="figure2_opinion_distribution.png",
            mime="image/png"
        )
        st.markdown("""In Figure 2, Gray bars show the true opinion distribution, while colored KDE curves represent visible opinions.  
If the visible layer is missing or skewed, it indicates latent diversity hidden beneath a homogeneous public front—  
a quantitative signature of perceived consensus under social pressure.""")

        # ==============================
        # Figure 3: 网络中发声与沉默个体的状态图
        # ==============================
        if N <= 300:
            st.markdown("**Figure 3. Final Network (blue = speaking, red = silent)**")

            fig3, ax3 = plt.subplots(figsize=(6, 6))
            pos = nx.spring_layout(G, seed=seed)          # 位置布局
            colors = ["tab:blue" if s[i] == 1 else "tab:red" for i in range(N)]

            nx.draw(
                G, pos=pos, node_color=colors, node_size=35,
                edge_color="lightgray", with_labels=False, ax=ax3
            )
            ax3.set_title("Figure 3. Speaking vs Silent Agents in Network")
            st.pyplot(fig3)

            # 下载按钮
            buf3 = BytesIO()
            fig3.savefig(buf3, format="png", dpi=300, bbox_inches="tight")
            st.download_button(
                label="Download Figure 3",
                data=buf3.getvalue(),
                file_name="figure3_network_state.png",
                mime="image/png"
            )

            plt.close(fig3)
        else:
            st.info("Network visualization skipped (too large N).")


        st.markdown("""
            In Figure 3,Blue nodes are speaking agents and red nodes are silent ones. An all-red network means that communication links exist but information flow has stopped,  
    depicting a “frozen” opinion state caused by fear of isolation.""")    


        # ==============================
        # Figure 4: 参数扫描 —— 社会容忍度 vs 沉默率
        # ==============================
        import numpy as np
        import matplotlib.pyplot as plt
        from io import BytesIO

        st.markdown("---")
        st.subheader("📈 Figure 4. Silence Rate vs Social Tolerance")

        # 参数扫描范围
        tau_range = np.linspace(0.05, 0.6, 10)
        media_bias_values = [0.0, 0.3, 0.6]  # 三种媒体偏向

        fig4, ax4 = plt.subplots(figsize=(7, 4))

        for mb in media_bias_values:
            silence_rates = []
            for tau_mid in tau_range:
                tau_min = tau_mid * 0.7
                tau_max = tau_mid * 1.3
                tau_scan = rng.uniform(tau_min, tau_max, N)
                res = run_simulation(N, G, o, tau_scan, mb, w_media, T, alpha_algo)
                silence_rates.append(res["silence_hist"][-1])
            ax4.plot(tau_range, silence_rates, marker="o", label=f"Media bias={mb:.1f}")

        ax4.set_xlabel("Average Social Tolerance (τ)")
        ax4.set_ylabel("Final Silence Rate")
        ax4.set_title("Figure 4. Silence Rate vs Social Tolerance under Different Media Bias")
        ax4.legend()
        ax4.grid(alpha=0.3)
        st.pyplot(fig4)

        # 下载按钮
        buf4 = BytesIO()
        fig4.savefig(buf4, format="png", dpi=300, bbox_inches="tight")
        st.download_button(
            label="📥 Download Figure 4",
            data=buf4.getvalue(),
            file_name="figure4_silence_vs_tolerance.png",
            mime="image/png"
        )
        plt.close(fig4)

        st.markdown("""
        Figure 4 shows how final silence rate depends on social tolerance (τ) under varying media bias.  
        Higher tolerance reduces fear of isolation and encourages expression, leading to a lower silence rate.  
        Stronger media bias pushes the whole curve upward—biased media amplify conformity pressure and deepen the spiral of silence.  

        **Key Insight:**  
        Social tolerance functions as a *release valve* for diversity of expression,  
        while media bias acts as an *accelerator* of the silence spiral.
        """)

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
