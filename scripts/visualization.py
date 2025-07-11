import matplotlib.pyplot as plt
import os

def plot_results(y_test, y_pred, output_path="results/actual_vs_predicted.png"):
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.xlabel("Actual Prices")
    plt.ylabel("Predicted Prices")
    plt.title("Actual vs Predicted Prices")
    plt.grid(True)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.show()
    print(f"📉 Saved plot to '{output_path}'")
