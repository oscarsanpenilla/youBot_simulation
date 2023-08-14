import numpy as np
import matplotlib.pyplot as plt


def plot_columns_from_csv(filename, save_path="output_plots.png"):
    """
    Read the matrix from a CSV file, plot each column in its own subplot, save the plots as a PNG file, and show them.

    Args:
    - filename (str): Path to the CSV file.
    - save_path (str): Path to save the PNG plot. Defaults to 'output_plots.png' in the current directory.

    Returns:
    - None
    """

    # Load matrix from CSV
    matrix = np.loadtxt(filename, delimiter=",")

    # Ensure that the matrix has the correct number of columns
    assert matrix.shape[1] == 6, "Matrix should have 6 columns to match given labels"

    labels = ["Wz", "Wy", "Wx", "Vx", "Vy", "Vz"]

    # Get number of rows in the matrix which determines the length of the time sequence
    num_rows, _ = matrix.shape
    time = np.arange(0, num_rows * 0.01, 0.01)

    # Create 2x3 grid of subplots
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    axs = axs.ravel()  # flatten the 2x3 array to make indexing easier

    for i in range(matrix.shape[1]):
        axs[i].plot(time, matrix[:, i], label=labels[i])
        axs[i].set_title(labels[i])
        axs[i].set_xlabel('Time (s)')
        axs[i].set_ylabel('Value')
        axs[i].grid(True)
        axs[i].legend()

    plt.tight_layout()

    # Save the figure as a PNG file
    plt.savefig(save_path, format="png")

    # Display the plot (optional)
    plt.show()


# Test
if __name__ == "__main__":
    plot_columns_from_csv("error.csv", "error_plot.png")