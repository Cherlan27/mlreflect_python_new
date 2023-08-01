# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 14:34:50 2023

@author: petersdorf
"""

from mlreflect.data_generation import Layer, Substrate, AmbientLayer, MultilayerStructure

class structure_creater():
    def __init__(self, number_layer, sld_bulk, roughness_bulk, thickness_layer, roughness_layer, sld_layer):
        self.roughness = roughness_bulk
        self.sld = sld_bulk
        
        self.thickness_layer = []
        self.roughness_layer = []
        self.sld_layer = []
        
        for i in range(0, number_layer):
            self.thickness_layer.append(thickness_layer[i])
            self.roughness_layer.append(roughness_layer[i])
            self.sld_layer.append(sld_layer[i])
        
        self.substrate = Substrate('H2O', self.roughness, self.sld) 
        self.ambient = AmbientLayer('ambient', 0)
        self.sample = MultilayerStructure()
        self.sample.set_substrate(self.substrate)
        self.sample.set_ambient_layer(self.ambient)
        self.add_layering(number_layer)
    
    def add_layering(self, number_layer):
        for i in range(number_layer):
            str_name = "Layer" + str(i)
            layer_gen = Layer(str_name, self.thickness_layer[i], self.roughness_layer[i], self.sld_layer[i])
            self.sample.add_layer(layer_gen)
        
    def get_sample(self):
        return self.sample
        