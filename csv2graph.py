#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

from os import replace
import sys
from math import sqrt

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Csv2ImageGenerator:
    """
    Docstring for Csv2ImageGenerator
    """
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

    def load_labels(self, labels_file : str) -> list:
        """Load list of labels from file"""
        result = []
        if os.path.exists(labels_file):
            with open(labels_file, 'r') as file:
                result = file.readline().split(",")
            file.close()

        return result

    def csv2img_filename(self, csv_file : str) -> str:
        """
        Docstring for csv2img_filename

        :param self: Description
        :param csv_file: Description
        :type csv_file: str
        :return: Description
        :rtype: str
        """
        img_file = csv_file.replace(".csv", ".png")
        if not self.img_folder:
            return img_file

        f_parts = img_file.split("/")
        return "{0}/{1}".format(self.img_folder, f_parts[-1])

    def generate_img(self, img_file : str, labels:list = []) -> None:
        """
        Docstring for generate_img

        :param self: Description
        :param img_file: Description
        :type img_file: str
        """
        plt.xlabel('Ckpnt')
        plt.grid(True)

        x_data = self.df['Idx']
        np_arr = np.asarray(self.df)

        for y_idx in range(1, len(np_arr[0]-1)):
            y_data = np_arr[:, y_idx]
            label = labels[y_idx-1] if len(labels)>0 and y_idx-1 < len(labels) else "Val{}".format(y_idx)
            plt.plot(x_data, y_data, label=label)

        plt.legend()
        # Save image
        plt.savefig(img_file, dpi=300)

        plt.cla()

if __name__ == '__main__':
    """Generate graph"""
    gen = Csv2ImageGenerator()

    csv_file = sys.argv[1]
    labels = []

    if len(sys.argv) > 2:
        """Load labels if it is presented"""
        labels = gen.load_labels(sys.argv[2])

    gen.data_file = csv_file
    img_file = gen.csv2img_filename(sys.argv[1])
    gen.generate_img(img_file, labels)