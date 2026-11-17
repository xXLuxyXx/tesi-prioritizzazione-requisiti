import os
from datetime import datetime
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import torch
from sentence_transformers import SentenceTransformer

from ai_analysis import analyze_requirement, plantuml_encode
from gnn_predictor import GNNRegressor, predict_dependencies
from preprocess_input import load_optimized_artifacts, predict_priority_optimized

st.set_page_config(page_title="Tool of Requirements Prioritization and Analysis", layout="wide")
st.title("⭐ Tool of Requirements Prioritization")

FEEDBACK_STORE_PATH = "dataset/operational_feedback.csv"


@st.cache_resource
def load_all_models():
    """Caricamento persistente in cache delle risorse pesanti (Sez. 4.4.2)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clf_opt, target_enc, opt_cfg = load_optimized_artifacts()
    sbert = SentenceTransformer("all-MiniLM-L6-v2", device=str(device))

    scaler_legacy = joblib.load("models/scaler.pkl")
    pca_legacy = joblib.load("models/pca.pkl")
    regressor = joblib.load("models/regressor_cost_time.pkl")

    model_gnn = GNNRegressor(in_channels=388, hidden_channels=64).to(device)
    state_dict = torch.load("models/gnn_model.pt", map_location=device)
    model_gnn.load_state_dict(state_dict)
    model_gnn.eval()

    df_existing = pd.read_csv("dataset/full_requirements.csv")
    return device, clf_opt, target_enc, opt_cfg, sbert, scaler_legacy, pca_legacy, regressor, model_gnn, df_existing


def save_operational_feedback(row_dict: dict, predicted_prio: str, human_prio: str, entropy: float):
    """Salva il feedback validato dal Product Owner nell'Operational Feedback Store (Sez. 3.2.3)."""
    record = {
        "timestamp": datetime.now().isoformat(),
        "Type": row_dict.get("Type", ""),
        "Requirement": row_dict.get("Requirement", ""),
        "Change": row_dict.get("Change", ""),
        "Urgency": row_dict.get("Urgency", ""),
        "Origin": row_dict.get("Origin", ""),
        "N_dependencies": row_dict.get("N_dependencies", 0),
        "Cost": row_dict.get("Cost", 0),
        "Time": row_dict.get("Time", 0),
        "predicted_priority": predicted_prio,
        "human_priority": human_prio,
        "is_override": int(predicted_prio.lower() != human_prio.lower()),
        "entropy": round(entropy, 4),
        "alpha_weight": 5.0,
    }
    df_new = pd.DataFrame([record])
    if os.path.exists(FEEDBACK_STORE_PATH):
        df_new.to_csv(FEEDBACK_STORE_PATH, mode="a", header=False, index=False)
    else:
        df_new.to_csv(FEEDBACK_STORE_PATH, index=False)


(
    device,
    clf_opt,
    target_enc,
    opt_cfg,
    sbert_model,
    scaler,
    pca_emb,
    regressor,
    model_gnn,
    df_existing,
) = load_all_models()


########## Manual Input ##########

st.header("✚ Input")

predicted_dependencies = st.session_state.get("dependencies", "-")
predicted_cost = st.session_state.get("cost", "-")
predicted_time = st.session_state.get("time", "-")

with st.form("manual_input"):
    col1, col2, col3 = st.columns(3)
    with col1:
        req_type = st.selectbox("Type", ["Functional", "Non functional"])
        change = st.selectbox("Change", ["Addition", "Modification", "Deletion"])
    with col2:
        urgency = st.selectbox("Urgency", ["Soon", "Later"])
        origin = st.selectbox("Origin", ["Organization", "Market", "Customer", "Developer knowledge", "Project vision"])
    with col3:
        st.markdown(f"Cost (predicted): `{predicted_cost}`")
        st.markdown(f"Time (predicted): `{predicted_time}`")
        st.markdown(f"N.dependencies (predicted): `{predicted_dependencies}`")

    requirement = st.text_area("Description of requirement")
    submitted = st.form_submit_button("Predict Priority")

if submitted:
    if requirement.strip() == "":
        st.error("❗Error: Please insert a requirement description")
    else:
        row = {
            "Type": req_type,
            "Requirement": requirement,
            "Change": change,
            "Urgency": urgency,
            "Origin": origin,
            "N_dependencies": "",
            "Cost": "",
            "Time": "",
        }

        # 1. Predizione dipendenze tramite GNN
        dependencies, new_embedding, new_cat = predict_dependencies(row, model_gnn, df_existing, device=device)
        st.session_state.dependencies = dependencies
        row["N_dependencies"] = dependencies

        # 2. Stima Cost & Time
        scaled_all = scaler.transform([[dependencies, 0.0, 0.0]])
        scaled_num = scaled_all[:, 0].reshape(1, -1)
        emb_norm = new_embedding.cpu().numpy() / np.linalg.norm(new_embedding.cpu().numpy(), axis=1, keepdims=True)
        emb_reduced = pca_emb.transform(emb_norm)
        x_reg_input = np.hstack([emb_reduced * 0.5, new_cat.cpu().numpy().astype(float), scaled_num])

        predicted_cost, predicted_time = regressor.predict([x_reg_input[0]])[0]
        predicted_cost = int(round(predicted_cost))
        predicted_time = int(round(predicted_time))

        row["Cost"] = predicted_cost
        row["Time"] = predicted_time
        st.session_state.cost = predicted_cost
        st.session_state.time = predicted_time

        # 3. Classificazione calibrata + Entropia di Shannon (Config. D)
        graph_metrics = {
            "in_degree": float(dependencies),
            "out_degree": 0.0,
            "betweenness": 0.0,
            "impact_score": float(dependencies) * 0.1,
        }
        res = predict_priority_optimized(
            row_dict=row,
            sbert_model=sbert_model,
            clf_opt=clf_opt,
            target_enc=target_enc,
            opt_cfg=opt_cfg,
            graph_metrics=graph_metrics,
        )

        st.session_state.last_row = row
        st.session_state.prediction_details = res
        st.session_state.prediction_result = res["priority"]

        # 4. Analisi qualitativa & PlantUML via Groq API
        with st.spinner("Analysis of requirement with AI..."):
            full_text, plantuml_code = analyze_requirement(requirement, row["Type"])
            st.session_state.text_output = full_text
            st.session_state.plantuml_code = plantuml_code

        st.rerun()

if "prediction_result" in st.session_state:
    pred = st.session_state.prediction_result
    details = st.session_state.get("prediction_details", {})
    proba_map = details.get("proba", {})
    entropy = details.get("entropy", 0.0)
    needs_review = details.get("needs_review", False)
    theta_high = details.get("theta_high", 0.44)

    st.subheader("🎯 Result of prediction")
    emoji_map = {"high": "🔴", "medium": "🟡", "low": "🟢"}

    c_res, c_prob, c_ent = st.columns([1.2, 1.5, 1.3])
    with c_res:
        st.markdown(
            f'<p style="padding-left: 10px; font-size:24px;"> ▶️ Priority: {emoji_map[pred]} <strong>{pred.upper()}</strong></p>',
            unsafe_allow_html=True,
        )
        st.caption(f"Soglia decisionale calibrata $\\theta^*_{{High}} = {theta_high:.2f}$")

    with c_prob:
        st.markdown("**Probabilità Posteriori Calibrate:**")
        p_low = proba_map.get("low", 0.0)
        p_med = proba_map.get("medium", 0.0)
        p_high = proba_map.get("high", 0.0)
        st.write(f"🟢 Low: `{p_low:.1%}` | 🟡 Medium: `{p_med:.1%}` | 🔴 High: `{p_high:.1%}`")
        st.progress(min(max(p_high, 0.0), 1.0))

    with c_ent:
        st.metric(label="Incertezza (Entropia di Shannon H)", value=f"{entropy:.3f} bit")
        if needs_review:
            st.warning("⚠️ Alta incertezza: consigliata revisione Human-in-the-Loop.")
        else:
            st.success("✅ Confidenza elevata.")

    # Pannello Human-in-the-Loop (HITL)
    with st.expander("🧑‍💻 Validazione Human-in-the-Loop (Product Owner Feedback)", expanded=needs_review):
        col_h1, col_h2 = st.columns([2, 1])
        with col_h1:
            human_choice = st.selectbox(
                "Conferma o correggi la priorità assegnata:",
                ["low", "medium", "high"],
                index=["low", "medium", "high"].index(pred),
                key="hitl_selectbox",
            )
        with col_h2:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("💾 Registra Feedback HITL"):
                save_operational_feedback(
                    st.session_state["last_row"],
                    pred,
                    human_choice,
                    entropy,
                )
                st.success(f"Feedback salvato ({human_choice.upper()}) nell'Operational Feedback Store.")

if "text_output" in st.session_state:
    st.subheader("📄 AI Analysis")
    st.markdown(st.session_state.text_output)

if "plantuml_code" in st.session_state and st.session_state.plantuml_code:
    st.subheader("🏗️ Architectural Diagram")
    encoded = plantuml_encode(st.session_state.plantuml_code)
    plantuml_url = f"https://www.plantuml.com/plantuml/svg/{encoded}"
    st.image(plantuml_url)

st.markdown("<br><br>", unsafe_allow_html=True)


########## CSV Loading ##########

st.header("📁 Requirement upload from CSV")

uploaded_file = st.file_uploader("Upload file .csv", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("✅ CSV uploaded:")
    st.dataframe(df.head())

    n_deps_list, costs_list, times_list = [], [], []
    prios_list, p_high_list, entropies_list, alerts_list = [], [], [], []
    ai_text_list, ai_plantuml_list = [], []

    for _, row in df.iterrows():
        row_dict = row.to_dict()
        dep, emb, cat = predict_dependencies(row_dict, model_gnn, df_existing, device=device)
        n_deps_list.append(dep)
        row_dict["N_dependencies"] = dep

        scaled_all = scaler.transform([[dep, 0.0, 0.0]])
        scaled_num = scaled_all[:, 0].reshape(1, -1)
        emb_norm = emb.cpu().numpy() / np.linalg.norm(emb.cpu().numpy(), axis=1, keepdims=True)
        emb_reduced = pca_emb.transform(emb_norm)
        x_reg_input = np.hstack([emb_reduced * 0.5, cat.cpu().numpy().astype(float), scaled_num])

        predicted_cost, predicted_time = regressor.predict([x_reg_input[0]])[0]
        c_val = int(round(predicted_cost))
        t_val = int(round(predicted_time))
        costs_list.append(c_val)
        times_list.append(t_val)
        row_dict["Cost"] = c_val
        row_dict["Time"] = t_val

        res = predict_priority_optimized(
            row_dict=row_dict,
            sbert_model=sbert_model,
            clf_opt=clf_opt,
            target_enc=target_enc,
            opt_cfg=opt_cfg,
            graph_metrics={"in_degree": float(dep), "impact_score": float(dep) * 0.1},
        )
        prios_list.append(res["priority"])
        p_high_list.append(round(res["proba"].get("high", 0.0), 3))
        entropies_list.append(round(res["entropy"], 3))
        alerts_list.append("⚠️ Review" if res["needs_review"] else "OK")

        ai_text, plantuml_code = analyze_requirement(row["Requirement"], row["Type"])
        ai_text_list.append(ai_text)
        ai_plantuml_list.append(plantuml_code)

    df["N_dependencies"] = n_deps_list
    df["Cost"] = costs_list
    df["Time"] = times_list
    df["Priority"] = prios_list
    df["P_High_Calibrated"] = p_high_list
    df["Entropy_H"] = entropies_list
    df["HITL_Alert"] = alerts_list
    df["AI_Analysis"] = ai_text_list
    df["AI_PlantUML"] = ai_plantuml_list

    st.subheader("🔍 Predictions on file requirements")
    st.dataframe(df)

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download CSV with Priority",
        data=csv,
        file_name="prioritized_requirements.csv",
        mime="text/csv",
    )