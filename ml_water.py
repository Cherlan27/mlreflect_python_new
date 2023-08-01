# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 16:42:45 2023

@author: petersdorf
"""

from predicter_xrr_ml import prediction_sample
from matplotlib import pyplot as plt
import time
import pandas as pd



path_name = "C:/Users/Petersdorf/Desktop/Gordon_conference/water/"
file_name = "water_sample_1_01.dat"
file = path_name + file_name
refl = pd.read_csv(file,  delimiter= "\t")

qz = refl["//qz"][0:94]
intensity = abs(refl["intensity_normalized"])[0:94]
intensity_err = abs(refl["e_intensity_normalized"])[0:94]

prediction_water = prediction_sample(qz, intensity, intensity_err, 1000, pathfile = "trained_1_layer_test_new.h5")
prediction_water()