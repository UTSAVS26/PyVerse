import matplotlib.pyplot as plt
import numpy as np

def overlay_prediction(actual, predicted, label="Metric", export_path=None):
    plt.figure(figsize=(10, 5))
    plt.plot(actual, label=f"Actual {label}")
    pred_x = np.arange(len(actual), len(actual) + len(predicted))
    plt.plot(pred_x, predicted, label=f"Predicted {label}", linestyle='--')
    plt.xlabel('Time (s)')
    plt.ylabel(label)
    plt.legend()
    plt.title(f"Actual vs Predicted {label}")
    if export_path:
        plt.savefig(export_path)
    plt.show()
    plt.close()
