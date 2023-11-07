import matplotlib.pyplot as plt
import numpy as np
import sys

def main(file_path):
    # Initialize lists to store the iterations, loss, and slope values
    iterations = []
    loss_values = []
    slope_values = []

    # Process the file line by line to extract the iterations, loss, and slope values
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            parts = line.split(',')
            # Ensure that the line has enough columns
            if len(parts) >= 5:
                # Append the iteration value from the first column
                iterations.append(int(parts[0]))
                # Append the loss value from the fourth column
                loss_values.append(float(parts[3]))
                # Append the slope value from the fifth column
                slope_values.append(float(parts[4]))

    # Convert the slope values to the log10 of their absolute values
    # Handle any zero values to avoid log10 issues
    slope_log_values = [np.log10(abs(slope)) if slope != 0 else float('-inf') for slope in slope_values]

    # loss_log_values = [np.log10(abs(loss)) if loss != 0 else float('-inf') for loss in loss_values]

    # Plot the loss values on the left y-axis
    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('Loss', color=color)
    ax1.plot(iterations, loss_values, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    # Create a second y-axis for the log10 of the slope values
    ax2 = ax1.twinx()  
    color = 'tab:blue'
    ax2.set_ylabel('Log10 |Loss Slope|', color=color)
    ax2.plot(iterations, slope_log_values, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    # Show the plot with a title
    plt.title('Loss and Log10 |Loss Slope| per Iteration')
    plt.show()

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python script.py <file_path>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    main(file_path)
