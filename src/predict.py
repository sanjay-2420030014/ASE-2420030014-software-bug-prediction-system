"""Bug prediction module for the AI-Based Smart Bug Prediction System."""
from pathlib import Path
import joblib, pandas as pd
ROOT=Path(__file__).resolve().parents[1]
MODEL_FILE=ROOT/"results"/"bug_prediction_model.joblib"

def load_model(model_path=MODEL_FILE):
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Trained model not found: {model_path}")
    return joblib.load(model_path)

def predict_module(feature_values, model_path=MODEL_FILE):
    artifact=load_model(model_path)
    names=artifact["feature_names"]; scaler=artifact["scaler"]; model=artifact["model"]
    if isinstance(feature_values,dict):
        missing=[n for n in names if n not in feature_values]
        if missing: raise ValueError(f"Missing feature values: {missing}")
        values=[feature_values[n] for n in names]
    else:
        values=list(feature_values)
    if len(values)!=len(names):
        raise ValueError(f"Expected {len(names)} feature values, got {len(values)}.")
    X=pd.DataFrame([[float(v) for v in values]],columns=names)
    Xs=scaler.transform(X)
    pred=int(model.predict(Xs)[0])
    result={"prediction":pred,"label":"Defective" if pred else "Non-Defective"}
    if hasattr(model,"predict_proba"):
        prob=model.predict_proba(Xs)[0]
        result["non_defective_probability"]=float(prob[0])
        result["defective_probability"]=float(prob[1])
    return result

def predict_from_dataset_row(row_index=0,dataset_path=None):
    dataset_path=dataset_path or ROOT/"data"/"bug_dataset.csv"
    df=pd.read_csv(dataset_path); artifact=load_model()
    if not 0<=row_index<len(df): raise IndexError("Invalid row index.")
    names=artifact["feature_names"]
    result=predict_module(df.iloc[row_index][names].tolist())
    result["actual_label"]="Defective" if bool(df.iloc[row_index]["defects"]) else "Non-Defective"
    result["row_index"]=row_index
    return result

if __name__=="__main__":
    x=predict_from_dataset_row(0)
    print("Prediction test completed successfully.")
    print("Actual:",x["actual_label"])
    print("Predicted:",x["label"])
    print(f"Defective probability: {x['defective_probability']:.4f}")
