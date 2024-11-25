import matplotlib.pyplot as plt
import sys

if len(sys.argv) == 1:
    output_path = "."
elif len(sys.argv) == 2:
    output_path = sys.argv[1]
else:
    print("Usage: python test_graph_saving.py [output_path]")
    sys.exit(1)

# Sample data
x = [0, 1, 2, 3, 4, 5]
y = [0, 1, 4, 9, 16, 25]  # y = x^2

# Output the values used in the plot
print("x values:", x)
print("y values:", y)

# Create the plot
plt.figure(figsize=(8, 6))  # Set figure size
plt.plot(x, y, label="y = x^2", marker="o")  # Plot with markers
plt.title("Sample Graph: y = x^2")
plt.xlabel("x-axis")
plt.ylabel("y-axis")
plt.legend()

# Save the plot as a PNG file
print("Saving graphs to disk")
plt.savefig(f"{output_path}/sample_graph_settings.png", dpi=300, bbox_inches="tight")

# Optional: Display the plot
# plt.show()
