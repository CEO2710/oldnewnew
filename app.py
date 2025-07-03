import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import xgboost as xgb
from sklearn.model_selection import train_test_split

# 设置中文字体
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

# 页面配置
st.set_page_config(
    page_title="Unplanned Reoperation Risk Prediction",
    page_icon="⚕️",
    layout="wide"
)

# 加载数据
@st.cache_data
def load_data():
    try:
        df = pd.read_excel("data/2222222.xlsx")
        return df
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        st.stop()

# 训练模型
@st.cache_data
def train_model(df):
    # 提取特征和目标变量
    X = df.drop("Unplanned reoperation", axis=1)
    y = df["Unplanned reoperation"]
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # 训练XGBoost模型
    model = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    return model, X.columns

# 主应用
def main():
    st.title("❤️ Unplanned Reoperation Risk Prediction")
    st.markdown("This application uses XGBoost to predict the risk of unplanned reoperation based on patient characteristics.")
    
    # 加载数据
    df = load_data()
    
    # 训练模型
    model, feature_names = train_model(df)
    
    # 侧边栏 - 变量定义说明
    with st.sidebar.expander("Variable Definitions", expanded=False):
        st.markdown("### Categorical Variable Definitions")
        st.markdown("""
        - **SEX**:
          - 0 = Female
          - 1 = Male
          
        - **ASA scores**:
          - 0 = ASA < 3
          - 1 = ASA ≥ 3
          
        - **tumor location**:
          - 1 = off-axis AND subcerebellar
          - 2 = intra-axis AND subcerebellar
          - 3 = off-axis AND supracerebellar
          - 4 = intra-axis AND supracerebellar
          
        - **Benign or malignant**:
          - 0 = Benign
          - 1 = Malignant
          
        - **Admitted to NICU**:
          - 0 = No NICU
          - 1 = Admitted to NICU
          
        - **Duration of surgery**:
          - 0 = ≤ 4 hours
          - 1 = > 4 hours
          
        - **diabetes**:
          - 0 = No diabetes
          - 1 = Diabetes
          
        - **CHF**:
          - 0 = No CHF
          - 1 = CHF
          
        - **Functional dependencies**:
          - 0 = No functional dependencies
          - 1 = Functional dependencies
          
        - **mFI-5**:
          - 0 = Robust
          - 1 = Pre-frail
          - 2 = Frail
          - ≥3 = Severely frail
          
        - **Type of tumor**:
          - 1 = Meningiomas
          - 2 = Primary malignant brain tumors
          - 3 = Metastatic brain tumor
          - 4 = Acoustic neuroma
          - 5 = Other
        """)
    
    # 用户预测界面
    st.subheader("🔍 Patient Risk Prediction")
    
    with st.form("prediction_form"):
        st.markdown("### Enter Patient Characteristics")
        
        # 创建输入表单（整数选项）
        input_data = {}
        for feature in feature_names:
            # 获取变量的最小值和最大值
            min_val = int(df[feature].min())
            max_val = int(df[feature].max())
            mean_val = int(df[feature].mean())
            
            # 为分类变量添加标注
            help_text = None
            if feature == "SEX":
                help_text = "0 = Female, 1 = Male"
            elif feature == "ASA scores":
                help_text = "0 = ASA < 3, 1 = ASA ≥ 3"
            elif feature == "tumor location":
                help_text = "1 = off-axis AND subcerebellar, 2 = intra-axis AND subcerebellar, 3 = off-axis AND supracerebellar, 4 = intra-axis AND supracerebellar"
            elif feature == "Benign or malignant":
                help_text = "0 = Benign, 1 = Malignant"
            elif feature == "Admitted to NICU":
                help_text = "0 = No NICU, 1 = Admitted to NICU"
            elif feature == "Duration of surgery":
                help_text = "0 = ≤ 4 hours, 1 = > 4 hours"
            elif feature == "diabetes":
                help_text = "0 = No diabetes, 1 = Diabetes"
            elif feature == "CHF":
                help_text = "0 = No CHF, 1 = CHF"
            elif feature == "Functional dependencies":
                help_text = "0 = No functional dependencies, 1 = Functional dependencies"
            elif feature == "mFI-5":
                help_text = "0 = Robust, 1 = Pre-frail, 2 = Frail, ≥3 = Severely frail"
            elif feature == "Type of tumor":
                help_text = "1 = Meningiomas, 2 = Primary malignant brain tumors, 3 = Metastatic brain tumor, 4 = Acoustic neuroma, 5 = Other"
            
            # 使用整数滑块
            input_data[feature] = st.slider(
                feature,
                min_value=min_val,
                max_value=max_val,
                value=mean_val,
                step=1,  # 确保步长为1（整数）
                help=help_text
            )
        
        # 提交预测
        submitted = st.form_submit_button("Predict Risk")
        
        if submitted:
            # 创建预测数据
            input_df = pd.DataFrame([input_data])
            
            # 预测
            prediction = model.predict(input_df)[0]
            proba = model.predict_proba(input_df)[0][1]
            
            # 显示预测结果
            st.subheader("📊 Prediction Result")
            
            if prediction == 1:
                st.error(f"Prediction: **High Risk of Unplanned Reoperation**")
                st.warning(f"Risk Probability: {proba:.2%}")
            else:
                st.success(f"Prediction: **Low Risk of Unplanned Reoperation**")
                st.info(f"Risk Probability: {proba:.2%}")
            
            # 生成SHAP解释
            try:
                st.subheader("🔍 Prediction Explanation")
                
                # 创建SHAP解释器
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(input_df)
                
                # 显示SHAP force plot
                fig, ax = plt.subplots(figsize=(12, 4))
                shap.force_plot(
                    explainer.expected_value,
                    shap_values[0],
                    input_df.iloc[0],
                    feature_names=feature_names,
                    matplotlib=True,
                    show=False
                )
                plt.tight_layout()
                st.pyplot(fig)
                
                # 显示特征重要性排序
                st.subheader("Feature Importance")
                feature_importance = pd.Series(
                    np.abs(shap_values[0]),
                    index=feature_names
                ).sort_values(ascending=False)
                
                st.dataframe(feature_importance.reset_index().rename(
                    columns={"index": "Feature", 0: "SHAP Value (Impact)"}
                ))
                
            except Exception as e:
                st.warning(f"Failed to generate SHAP explanation: {e}")
                st.write("SHAP explanation is not available for this model.")

if __name__ == "__main__":
    main()
