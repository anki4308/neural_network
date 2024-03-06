import os
import numpy as np
import matplotlib.pyplot as plt

def plot_mean_weights(csv_path, layer_name):
    # Load CSV file into a NumPy array
    data = np.genfromtxt(csv_path, delimiter=',')

    # Calculate mean weights across neurons
    mean_weights = np.mean(data, axis=0)

    # Plot mean weights
    plt.plot(mean_weights, label=f'{layer_name} Mean Weights')

# Get the script's directory
script_dir = os.path.dirname(os.path.realpath(__file__))

# Specify the full paths to the CSV files
hidden_layer_csv = os.path.join(script_dir, 'hidden_weights.csv')
output_layer_csv = os.path.join(script_dir, 'output_weights.csv')

# Plot mean weights for both hidden and output layers on the same graph
plot_mean_weights(hidden_layer_csv, 'Hidden')
plot_mean_weights(output_layer_csv, 'Output')

plt.xlabel('Neuron Index')
plt.ylabel('Mean Weight Value')
plt.title('Comparison of Mean Weights for Hidden and Output Layers')
plt.legend()
plt.show()

