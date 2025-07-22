import matplotlib.pyplot as plt
import numpy as np

def overlay_prediction(actual, predicted, label="Metric", export_path=None):
    # Input validation
    if not isinstance(actual, (list, np.ndarray)) or not isinstance(predicted, (list, np.ndarray)):
        raise TypeError("actual and predicted must be arrays or lists")
    if len(actual) == 0 or len(predicted) == 0:
        raise ValueError("actual and predicted arrays cannot be empty")

    actual = np.asarray(actual)
    predicted = np.asarray(predicted)

    plt.figure(figsize=(10, 5))
    plt.plot(actual, label=f"Actual {label}")
    pred_x = np.arange(len(actual), len(actual) + len(predicted))
    plt.plot(pred_x, predicted, label=f"Predicted {label}", linestyle='--')
    plt.xlabel('Time (s)')
    plt.ylabel(label)
    plt.legend()
    plt.title(f"Actual vs Predicted {label}")
    if export_path:
        try:
            plt.savefig(export_path)
        except Exception as e:
            print(f"Failed to export plot to {export_path}: {e}")
            return False
    plt.show()
    plt.close()
    return True
