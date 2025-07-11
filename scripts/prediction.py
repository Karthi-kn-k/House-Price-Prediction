from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd

def predict_and_evaluate(model, x_test, y_test):
    y_pred = model.predict(x_test)
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\n📊 Evaluation:")
    print("Mean Squared Error:", mse)
    print("R-squared:", r2)

    return y_pred, mse, r2

def save_predictions(y_test, y_pred, path="results/predictions.csv"):
    output = pd.DataFrame({
        "Actual": y_test.values,
        "Predicted": y_pred
    })
    output.to_csv(path, index=False)
    print(f"💾 Saved predictions to '{path}'")
