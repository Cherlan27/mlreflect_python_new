# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 15:36:58 2023

@author: petersdorf
"""

from training_water import water_training

# roughness_layers = []
# thickness_layers = []
# sld_layers = []

sld_layers = [(0, 12)]
thickness_layers = [(0,18)]
roughness_layers = [(1,3.9)]

# roughness_layers = [(0, 12), (0, 12)]
# thickness_layers = [(0,18), (0,18)]
# sld_layers = [(1,3.9), (1,3.9)]

train_h2o = water_training((0, 15), (1.2,3.9), thickness_layers, roughness_layers, sld_layers)
train_h2o.print_structure()
train_h2o.training_generator(2**18)
train_h2o.get_loss(20)
train_h2o.fit_model()
train_h2o.get_prediction_tests()

train_h2o.saving_model("trained_1_layer_test_new.h5")