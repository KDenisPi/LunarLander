#https://matplotlib.org/stable/tutorials/pyplot.html

from os import replace
import sys
from math import sqrt

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class GrGenerator:
    """Graph generator"""
    def __init__(self):
        self._img_folder = None
        self.df = None

    @property
    def data_file(self) -> str:
        return self._data_file

    @data_file.setter
    def data_file(self, in_file) -> None:
        plt.close()

        self._data_file = in_file
        # Read CSV file
        self.df = pd.read_csv(self._data_file)

    @property
    def img_folder(self) -> str:
        return self._img_folder

    @img_folder.setter
    def img_folder(self, n_val:str) -> None:
        self._img_folder = n_val

    def gen_filename_by_func(self, file_in:str, func_name:str = "reward") -> str:
        """Generate output filename from the input based on function name"""
        img_file = file_in.replace(".csv", ".png")
        f_parts = img_file.split("/")

        if self.img_folder:
            img_file = "{0}/{1}_{2}".format(self.img_folder, func_name, f_parts[-1])
        else:
            img_file = "{0}/{1}_{2}".format("/".join(f_parts[0,-1]), func_name, f_parts[-1])

        return img_file

    def distance(self, row):
        return sqrt(row[1]*row[1]+row[2]*row[2])

    def generate_distance_graph(self) -> str:
        """Generate deviation graph from landing point"""
        x_data = self.df['Nm']
        np_arr = np.asarray(self.df)
        result = np.apply_along_axis(self.distance, 1, np_arr)

        plt.xlabel('Attempt')
        plt.ylabel('Distance')
        plt.title('Graph deviation from landing point')
        plt.plot(x_data, result)
        plt.grid(True)

        img_file = self.gen_filename_by_func(self.data_file, "distance")

        plt.savefig(img_file)
        plt.cla()
        return img_file


    def generate_reward_graph(self) -> str:
        """Generate reward graph"""
        # Select data
        x_data = self.df['Nm']
        y_data = self.df['Reward']

        # Create plot
        plt.plot(x_data, y_data)

        # plt.scatter(x_data, y_data) # Scatter plot
        # plt.bar(x_data, y_data)     # Bar plot

        # Customize plot
        plt.xlabel('Attempt')
        plt.ylabel('Reward')
        plt.title('Graph Reward for attempt')
        plt.grid(True)

        img_file = self.gen_filename_by_func(self.data_file)

        # Save image
        plt.savefig(img_file)
        plt.cla()

        return img_file

        # Show plot
        #plt.show()

if __name__ == '__main__':
    """Generate graph"""

    gen = GrGenerator()

    for prm in sys.argv:
        if prm == "--help" or len(sys.argv) == 1:
            print("Usage: python3 graph.py [iamages=output] file1.csv [file2.csv] ... [filen.csv]")
            exit()
        if prm.startswith("images="):
            _, image_folder = prm.split("=")
            if image_folder:
                gen.img_folder = image_folder


    for prm in sys.argv[1:]:
        if not prm.startswith("images="):
            data_file = prm
            gen.data_file = data_file
            img_rwd = gen.generate_reward_graph()
            img_dist = gen.generate_distance_graph()
            #print(img)


