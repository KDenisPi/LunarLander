#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

from os import replace
import sys
from math import sqrt
import glob

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
        self._labels = self.df.columns

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
        if len(labels)>0 :
            self._labels = labels

        plt.xlabel(self._labels[0])
        plt.grid(True)

        x_data = self.df[self._labels[0]]
        np_arr = np.asarray(self.df)

        for y_idx in range(1, len(np_arr[0]-1)):
            y_data = np_arr[:, y_idx]
            label = self._labels[y_idx] if len(self._labels)>0 and y_idx < len(self._labels) else "Val{}".format(y_idx)
            plt.plot(x_data, y_data, label=label)

        plt.legend()
        # Save image
        plt.savefig(img_file, dpi=300) #300

        plt.cla()

if __name__ == '__main__':
    """Generate graph"""
    gen = Csv2ImageGenerator()
    labels = []
    img_file = ""

    if len(sys.argv) == 1:
        print("Missing CSV file.\nUsage csv2graph.py file.csv [--output=filename.png] [--labels=custon.csv]")
        exit()

    csv_files = glob.glob(sys.argv[1])

    img_file_custom = ""

    for pcmd in sys.argv[1:]:
        prms = pcmd.split("=")
        print(pcmd)
        if prms[0] == "--labels":
            """Load labels if it is presented"""
            labels = gen.load_labels(prms[1])
        #do not select name for multiple files
        if prms[0] == "--output" and len(csv_files)==1:
            img_file_custom=prms[1]

    for csv_file in csv_files:
        img_file = img_file_custom if len(img_file_custom)>0 else gen.csv2img_filename(csv_file)

        gen.data_file = csv_file
        gen.generate_img(img_file, labels)