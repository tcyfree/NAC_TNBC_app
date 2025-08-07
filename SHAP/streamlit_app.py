import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import tempfile
import os

# === 初始化 ===
st.set_page_config(page_title="TNBC pCR Prediction Tool")
st.title("TNBC pCR Prediction Tool")
st.markdown("### Enter values for 6 phenotype-based features")

# === 加载模型 ===
model_path = '/home/network/Desktop/Project/NAC_TNBC_app/SHAP/Ensemble_voting_no_split_selected.pkl'
ensemble = joblib.load(model_path)

# =======================================
# STAR: Background数据读取
# =======================================
df = pd.read_excel("/home/network/Desktop/Project/NAC_TNBC_app/SHAP/cluster_factors_pearson_6.xlsx")
feature_cols = [c for c in df.columns if c not in ["ID", "pCR"]]
# 去掉非数值列
non_num = df[feature_cols].select_dtypes(exclude="number").columns
df = df.drop(columns=non_num)

# Pearson 相关性筛选
corr = df.corr()["pCR"].drop(["pCR", "ID"], errors="ignore")
sel_feats = corr[corr.abs() > 0].index.tolist()

X = df[sel_feats].reset_index(drop=True)
y = df["pCR"].reset_index(drop=True)

# 1. 背景样本：这里直接用全部 X（6 个特征，样本量一般也不会太大）
background = X.values

# 2. 定义 ensemble 的预测函数
def ensemble_predict_proba(data):
    # data: numpy array 或者 DataFrame
    df = pd.DataFrame(data, columns=X.columns) if not isinstance(data, pd.DataFrame) else data
    return ensemble.predict_proba(df)[:, 1]

# 3. 创建 KernelExplainer（对 6 维输入，用 logit link）
explainer = shap.KernelExplainer(
    model=ensemble_predict_proba,
    data=background,
    link="logit"
)
# =======================================
# END: Background数据读取
# =======================================
# === 特征名 ===
factor_names = ["Factor 1 Score", "Factor 2 Score", "Factor 3 Score",
                "Factor 4 Score", "Factor 5 Score", "Factor 6 Score"]
display_names = [f"Phenotype {i+1}" for i in range(6)]

# === 创建输入 ===
input_data = {}
# 获取默认值（用 X 的第36行）
default_values = X.iloc[35].to_dict()

for fname, display in zip(factor_names, display_names):
    default_val = default_values.get(fname, 0.0)  # 如果找不到默认值，填 0.0
    input_data[fname] = st.number_input(display, value=float(default_val), step=0.01, format="%.3f")
# === 用户点击预测 ===
if st.button("Predict"):
    input_df = pd.DataFrame([input_data])
    prob = ensemble.predict_proba(input_df)[0, 1]
    st.success(f"Predicted probability of TNBC pCR is **{prob:.2f}**")

    print(input_df)
    shap_values = explainer.shap_values(input_df)

    # ======================================
    # Force Plot (HTML, interactive)
    # ======================================
    st.markdown("### SHAP Force Plot")
    force_plot = shap.force_plot(
        base_value=explainer.expected_value,
        shap_values=shap_values[0],
        features=input_df.iloc[0],
        feature_names=display_names,
        matplotlib=False,
        show=False
    )

    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmpfile:
        shap.save_html(tmpfile.name, force_plot)
        tmp_path = tmpfile.name

    with open(tmp_path, 'r', encoding='utf-8') as f:
        html_content = f.read()
    st.components.v1.html(html_content, height=300)

    # ======================================
    # Waterfall Plot (matplotlib)
    # ======================================
    st.markdown("### SHAP Waterfall Plot")

    # 构建 Explanation 对象以绘图
    explanation = shap.Explanation(
        values=shap_values[0],
        base_values=explainer.expected_value,
        data=input_df.values[0],
        feature_names=display_names
    )

    fig, ax = plt.subplots(figsize=(6, 4))
    shap.plots.waterfall(explanation, max_display=6, show=False)
    st.pyplot(fig)

    # 保存 PDF 到临时文件
    pdf_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    fig.savefig(pdf_temp.name, bbox_inches='tight', dpi=300)
    pdf_temp.close()

    # 提供下载按钮
    with open(pdf_temp.name, "rb") as f:
        st.download_button(
            label="📥 Download Waterfall Plot as PDF",
            data=f,
            file_name="shap_waterfall_plot.pdf",
            mime="application/pdf"
        )
