# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 16:42:45 2023

@author: petersdorf
"""

from predicter_xrr_ml import prediction_sample
from matplotlib import pyplot as plt
import time
import pandas as pd
from make_refl2 import make_xrr

settings = {"file" :"nexus"}

if settings["file"] == "data":
# Choose from a dat file
    path_name = "C:/Users/Petersdorf/Desktop/Gordon_conference/water/"
    file_name = "water_sample_1_01.dat"
    file = path_name + file_name
    refl = pd.read_csv(file,  delimiter= "\t")
    qz = refl["//qz"][0:94]
    intensity = abs(refl["intensity_normalized"])[0:94]
    intensity_err = abs(refl["e_intensity_normalized"])[0:94]
elif settings["file"] == "nexus":
    # Choose from a set of nexus file
    xrr_scan = make_xrr()
    xrr_scan.scan_numbers = range(497,507+1)
    xrr_scan.data_directory = "K:/SYNCHROTRON/Murphy/2019-05_P08_11005943_giri_warias/raw_data/raw"
    xrr_scan.experiment = "align_liso63"
    xrr_scan()
    qz, intensity, intensity_err = xrr_scan.get_xrr()

prediction_water = prediction_sample(qz, intensity, intensity_err, 1000, pathfile = "trained_0_layer_test_new.h5")
prediction_water()

