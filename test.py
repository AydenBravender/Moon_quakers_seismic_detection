import numpy as np
import matplotlib.pyplot as plt

# Simple test data
time = np.linspace(0, 10, 100)  # Time from 0 to 10 seconds
data = np.sin(time)  # Example data: sine wave

# Create a plot
plt.figure(figsize=(10, 6))
plt.plot(time, data, label='Sine Wave', color='b')
plt.title('Test Plot: Sine Wave')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.axhline(0, color='k', linestyle='--', lw=0.5)  # Horizontal line at y=0
plt.grid(True)
plt.legend()

# Show the plot
plt.show(block=True)  # Block until the plot is closed
