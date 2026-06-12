import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle
import io

st.set_page_config(page_title="Student Pass/Fail Predictor", layout="wide")
st.title("🎓 Student Performance – Decision Tree Classifier")

# ── 1. Upload CSV ──────────────────────────────────────────────────────────────
st.header("student-mat.csv")
uploaded_file = st.file_uploader("Upload student-mat.csv (semicolon-separated)", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file, sep=';')
    st.success(f"Loaded {df.shape[0]} rows × {df.shape[1]} columns")

    with st.expander("Preview raw data"):
        st.dataframe(df.head())

    # ── 2. EDA ─────────────────────────────────────────────────────────────────
    st.header("2. Exploratory Data Analysis")

    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", df.shape[0])
    col2.metric("Duplicates", int(df.duplicated().sum()))
    col3.metric("Missing values", int(df.isnull().sum().sum()))

    with st.expander("Descriptive statistics"):
        st.dataframe(df.describe())

    # ── 3. Target column ───────────────────────────────────────────────────────
    st.header("3. Create Target Column")
    threshold = st.slider("Pass threshold (G3 ≥ this value → Pass)", 1, 20, 10)
    df['Result'] = df['G3'].apply(lambda x: 'Pass' if x >= threshold else 'Fail')

    counts = df['Result'].value_counts()
    st.bar_chart(counts)

    # ── 4. Encoding ────────────────────────────────────────────────────────────
    # Save target BEFORE encoding so it stays as 'Pass'/'Fail' strings
    y = df['Result'].copy()

    encoders = {}
    # Encode only feature columns, skip G3 and Result
    feature_df = df.drop(columns=['G3', 'Result'])
    categorical_columns = feature_df.select_dtypes(include='object').columns
    for col in categorical_columns:
        le = LabelEncoder()
        feature_df[col] = le.fit_transform(feature_df[col])
        encoders[col] = le

    X = feature_df

    # ── 5. Train ───────────────────────────────────────────────────────────────
    st.header("4. Train Decision Tree")

    col_a, col_b = st.columns(2)
    criterion = col_a.selectbox("Criterion", ["gini", "entropy"])
    test_size  = col_b.slider("Test size", 0.1, 0.4, 0.2, step=0.05)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    model = DecisionTreeClassifier(criterion=criterion, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    st.success(f"✅ Accuracy: **{acc:.2%}**")

    # Confusion matrix
    st.subheader("Confusion Matrix")
    labels = sorted(y.unique())
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=[f'Actual {l}' for l in labels],
                             columns=[f'Predicted {l}' for l in labels])
    st.dataframe(cm_df)

    # Classification report
    with st.expander("Classification Report"):
        report = classification_report(y_test, y_pred, output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose().round(2))

    # ── 6. Feature Importance ──────────────────────────────────────────────────
    st.header("5. Feature Importance")
    importance = (
        pd.DataFrame({'Feature': X.columns, 'Importance': model.feature_importances_})
        .sort_values('Importance', ascending=False)
    )

    top_n = st.slider("Show top N features", 5, len(importance), 10)
    fig1, ax1 = plt.subplots(figsize=(10, 4))
    top = importance.head(top_n)
    ax1.bar(top['Feature'], top['Importance'])
    ax1.set_xlabel("Feature")
    ax1.set_ylabel("Importance")
    ax1.set_title("Top Feature Importances")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig1)

    # ── 7. Decision Tree Plot ─────────────────────────────────────────────────
    st.header("6. Decision Tree Visualization")
    max_depth_vis = st.slider("Max depth to display", 1, 5, 3)
    fig2, ax2 = plt.subplots(figsize=(20, 8))
    plot_tree(model, feature_names=X.columns, class_names=['Fail', 'Pass'],
              filled=True, max_depth=max_depth_vis, ax=ax2)
    plt.tight_layout()
    st.pyplot(fig2)

    # ── 8. Download model ─────────────────────────────────────────────────────
    st.header("7. Download Trained Model")
    model_bytes = pickle.dumps(model)
    st.download_button("⬇ Download model (.pkl)", model_bytes,
                       file_name="decision_tree_model.pkl")

    # ── 9. Predict on new student ─────────────────────────────────────────────
    st.header("8. Predict for a New Student")
    st.info("Fill in the student details below and click Predict.")

    input_data = {}
    cols = st.columns(4)
    for i, col_name in enumerate(X.columns):
        with cols[i % 4]:
            if col_name in encoders:
                options = list(encoders[col_name].classes_)
                chosen = st.selectbox(col_name, options)
                input_data[col_name] = encoders[col_name].transform([chosen])[0]
            else:
                input_data[col_name] = st.number_input(col_name, value=0)

    if st.button("🔍 Predict"):
        input_df = pd.DataFrame([input_data])
        prediction = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0]
        classes = model.classes_
        st.subheader(f"Prediction: **{prediction}**")
        for cls, p in zip(classes, proba):
            st.write(f"  {cls}: {p:.1%}")

else:
    st.info("👆 Please upload your **student-mat.csv** file to get started.")
