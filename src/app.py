"""Streamlit UI for AI-Based Smart Bug Prediction System."""
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

ROOT=Path(__file__).resolve().parents[1]
MODEL_FILE=ROOT/"results"/"bug_prediction_model.joblib"

st.set_page_config(page_title="AI-Based Smart Bug Prediction System",page_icon="🐞",layout="wide")
st.title("AI-Based Smart Bug Prediction System")
st.write("Enter software metrics to predict whether a module is defective.")

@st.cache_resource
def load_artifact():
    if not MODEL_FILE.exists():
        raise FileNotFoundError("Trained model not found. Run: python src/train_model.py")
    return joblib.load(MODEL_FILE)

try:
    artifact=load_artifact()
    model=artifact["model"]; scaler=artifact["scaler"]; names=artifact["feature_names"]
    st.subheader("Software Metrics")
    values={}
    left,right=st.columns(2)
    for i,name in enumerate(names):
        with (left if i%2==0 else right):
            values[name]=st.number_input(name,value=0.0,format="%.6f")
    if st.button("Predict Bug Risk",type="primary"):
        X=pd.DataFrame([[values[n] for n in names]],columns=names)
        Xs=scaler.transform(X); pred=int(model.predict(Xs)[0])
        if pred: st.error("Prediction: DEFECTIVE")
        else: st.success("Prediction: NON-DEFECTIVE")
        if hasattr(model,"predict_proba"):
            p=model.predict_proba(Xs)[0]
            st.metric("Defective Probability",f"{p[1]*100:.2f}%")
            st.metric("Non-Defective Probability",f"{p[0]*100:.2f}%")
        st.caption(f"Model used: {artifact.get('best_model','Trained model')}")
except Exception as e:
    st.error(str(e))
