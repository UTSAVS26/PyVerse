import matplotlib.pyplot as plt
import numpy as np

def live_plot(data, label="Metric", export_path=None):
    plt.ion()
    fig, ax = plt.subplots()
    line, = ax.plot(data, label=label)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel(label)
    ax.legend()
    plt.show()
    for i in range(len(data), len(data) + 60):
        # Simulate real-time update (replace with actual data in real use)
        new_val = np.random.random() * 100
        data = np.append(data, new_val)
        line.set_ydata(data)
        line.set_xdata(np.arange(len(data)))
        ax.relim()
        ax.autoscale_view()
        plt.pause(1)
    if export_path:
        plt.savefig(export_path)
    plt.ioff()
    plt.close()
