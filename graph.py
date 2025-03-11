#https://matplotlib.org/stable/tutorials/pyplot.html

from os import replace
import sys

import pandas as pd
import matplotlib.pyplot as plt

def generate_graph(data_file:str, img_folder:str = "") -> str:
    # Read CSV file
    df = pd.read_csv(data_file)

    # Prepare data
    x_data = df['Nm']
    y_data = df['Reward']

    # Create plot
    plt.plot(x_data, y_data)
    # plt.scatter(x_data, y_data) # Scatter plot
    # plt.bar(x_data, y_data)     # Bar plot

    # Customize plot
    plt.xlabel('Attempt')
    plt.ylabel('Reward')
    plt.title('Graph Reward for attempt')
    plt.grid(True)

    img_file = data_file.replace(".csv", ".png")

    if img_folder:
        f_parts = img_file.split("/")
        img_file = img_folder + "/" + f_parts[-1]

    # Save image
    plt.savefig(img_file)

    return img_file

# Show plot
#plt.show()

if __name__ == '__main__':
    """Generate graph"""
    if len(sys.argv) <= 1:
        print("No data file. Usage data_file.csv [image folder]")
        exit()

    data_file = sys.argv[1]
    image_folder = ""

    if len(sys.argv) > 2:
        image_folder = sys.argv[2]

    img = generate_graph(data_file, image_folder)
    print(img)
