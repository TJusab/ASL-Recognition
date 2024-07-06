import pickle
import matplotlib.pyplot as plt

# Load the data from the pickle file
with open('data.pickle', 'rb') as f:
    data_dict = pickle.load(f)

# Extract data and labels
data = data_dict['data']
labels = data_dict['labels']

ranges = {'a': [], 'b': [], 'c': []}

current_label = None
start_index = None

for i, label in enumerate(labels):
    if label != current_label:
        if current_label is not None:
            ranges[current_label].append((start_index, i - 1))
        current_label = label
        start_index = i

# Add the final range
if current_label is not None:
    ranges[current_label].append((start_index, len(labels) - 1))

# Print the ranges for each label
for label, ranges_list in ranges.items():
    print(f"Ranges for label '{label}':")
    for start, end in ranges_list:
        print(f"    Start: {start}, End: {end}, Count: {end - start + 1}")


# Plot the landmarks of the letter 'a'
def plot_landmarks(landmarks, title, color):
    x = landmarks[0::2]
    y = landmarks[1::2]
    plt.scatter(x, y, c=color, label=title)
    plt.title('Landmarks for Label "a"')
    plt.gca().invert_yaxis()


label_to_plot = 'a'
colors = ['red', 'blue', 'green', 'yellow', 'orange', 'pink', 'purple']

plt.figure(figsize=(8, 8))

# Plot the data of the letter 'a'
for i in range(len(labels)):
    if labels[i] == label_to_plot:
        plot_landmarks(data[i], label_to_plot, colors[i % len(colors)])

# Save the plot as PNG after plotting and showing
plt.savefig('a_plot.png')
plt.show()
