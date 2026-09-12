# Prediction Module Testing

Run from the repository root:

```bash
python src/predict.py
```

The module loads `results/bug_prediction_model.joblib`, uses the saved 21-feature order and scaler, and predicts **Defective** or **Non-Defective**.

Record only the actual output produced by the program. Do not manually invent test results.


## Actual Test Output

Prediction test completed successfully.

Actual: Non-Defective
Predicted: Defective
Defective probability: 0.9993
