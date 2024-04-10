import matplotlib.pyplot as plt



# Initialize empty lists to store the data
time_points = []
memory_values = []

# Read the data from the text file
with open('profile1.txt', 'r') as file:
    for line in file:
        parts = line.split()
        if parts[0] == 'MEM':
            time_points.append(float(parts[1]))
            memory_values.append(float(parts[2]))

# Create a line plot
plt.plot(time_points, memory_values, label="Memory Usage")
plt.xlabel("Time")
plt.ylabel("Memory Value")
plt.title("Memory Usage Over Time")
plt.legend()
plt.grid(True)

plt.savefig("memory_plot.png")