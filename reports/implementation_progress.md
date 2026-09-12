# Implementation Progress

## Completed
- Dataset and preprocessing completed.
- ML models trained and compared.
- Trained model saved in `results/bug_prediction_model.joblib`.
- Prediction module implemented and tested.
- Streamlit UI implemented.

## UI Features
- 21 software metric inputs.
- Predict Bug Risk button.
- Defective / Non-Defective result.
- Probability display when supported.
- Shows the model used.

## Run
From the repository root:

```bash
pip install streamlit pandas scikit-learn joblib
streamlit run src/app.py
```

Take screenshots only after running the real application.
