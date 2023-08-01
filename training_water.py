import mlreflect
from mlreflect.utils import check_gpu
import matplotlib.pyplot as plt
import numpy as np
from mlreflect.data_generation import Layer, Substrate, AmbientLayer, MultilayerStructure
from mlreflect.training import Trainer
from mlreflect.data_generation import ReflectivityGenerator
from mlreflect.curve_fitter import CurveFitter
import pandas as pd
from structure_creater import structure_creater


class water_training():
    def __init__(self, sld_bulk, roughness_bulk, thickness_layers, roughness_layers, sld_layers):
        self.q = np.linspace(0.001, 0.7, 100)
        self.layer_num = len(thickness_layers)
        self.structure = structure_creater(self.layer_num, sld_bulk, roughness_bulk, thickness_layers, roughness_layers, sld_layers)
        self.sample = self.structure.get_sample()
    
    def print_structure(self):
        print(self.sample)
        
    def training_generator(self, number_generated: int):
        self.trainer = Trainer(self.sample, self.q, random_seed = 10)
        self.trainer.generate_training_data(number_generated)
        self.trainer.training_data['labels'].head(5)
        self.generator = ReflectivityGenerator(self.q, self.sample)
        self.sld_profiles = self.generator.simulate_sld_profiles(self.trainer.training_data['labels'].head(5))

    def get_sld_profiles(self, number: int):
        fig = plt.figure(dpi = 300)
        for i in range(number):
            plt.plot(self.sld_profiles[i][0], self.sld_profiles[i][1])
        plt.xlabel('Depth [A]')
        plt.ylabel('SLD')
        plt.title('SLD profiles')
        plt.show()

    def get_reflectivities(self, number: int):
        fig2 = plt.figure(dpi = 300)
        for i in range(number):
            plt.semilogy(self.q, self.trainer.training_data["reflectivity"][i, :], label = str(i))
        plt.xlabel("q [1/A]")
        plt.ylabel("Reflectivity")
        plt.legend()
        plt.show()

    def get_loss(self, number_epochs: int):
        self.trained_model, self.hist = self.trainer.train(n_epochs = number_epochs, batch_size = 512, verbose = 1)
        
    def plot_loss(self):
        plt.plot(self.hist.history['loss'], label = "loss")
        plt.plot(self.hist.history["val_loss"], label = "val_loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss (linear)")
        plt.legend()
        plt.show()
        
        plt.semilogy(self.hist.history["loss"], label = "loss")
        plt.semilogy(self.hist.history["val_loss"], label = "val_loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Loss (log")
        plt.legend()
        plt.show()
        
    def fit_model(self):
        self.curve_fitter = CurveFitter(self.trained_model)
        self.test_labels = self.generator.generate_random_labels(2**12)
        self.test_reflectivity = self.generator.simulate_reflectivity(self.test_labels)
        self.fit_output = self.curve_fitter.fit_curve(self.test_reflectivity, self.q, polish = False, optimize_q = False)

        self.predicted_test_labels = self.fit_output["predicted_parameters"]
        self.predicted_test_reflectivity = self.fit_output["predicted_reflectivity"]


    def saving_model(self, path: str):
        self.trained_model.save_model(path)

    def get_prediction_tests(self):
        if self.layer_num > 0:
            for i in range(self.layer_num):
                label_name = "Layer" + str(i) + "_thickness"
                fig = plt.figure(dpi = 300)
                plt.plot(self.predicted_test_labels[label_name], self.test_labels[label_name], '.', label = "predicted", alpha = 0.2)
                plt.plot(self.predicted_test_labels[label_name], self.predicted_test_labels[label_name], label = "ground truth")
                plt.legend()
                plt.xlabel("Ground truth")
                plt.ylabel("Thickness")
                plt.show()
        
        fig = plt.figure(dpi = 300)
        plt.plot(self.predicted_test_labels["H2O_roughness"], self.test_labels["H2O_roughness"], '.', label = "predicted", alpha = 0.2)
        plt.plot(self.predicted_test_labels["H2O_roughness"], self.predicted_test_labels["H2O_roughness"], label = "ground truth")
        plt.legend()
        plt.xlabel("Ground truth")
        plt.ylabel("Roughness")
        plt.show()
        
        fig = plt.figure(dpi = 300)
        plt.plot(self.predicted_test_labels["H2O_sld"], self.test_labels["H2O_sld"], '.', label = "predicted", alpha = 0.2)
        plt.plot(self.predicted_test_labels["H2O_sld"], self.predicted_test_labels["H2O_sld"], label = "ground truth")
        plt.legend()
        plt.xlabel("Ground truth")
        plt.ylabel("SLD")
        plt.show()

        # abs(predicted_test_labels - test_labels).mean()

        # test_plotted_index = []
        # fig = plt.figure(dpi = 300)
        # for j,i in enumerate(np.random.uniform(low = 0, high = 4000, size = 5).astype(int)):
        #     plt.semilogy(q, test_reflectivity[i, :]*10**j, 'o', markerfacecolor = "white", label='_nolegend_')
        #     plt.xlabel(r"q ($\AA^{1}$)")
        #     plt.ylabel("Reflectivity [norm.]")
        #     test_plotted_index.append(i)
        # plt.legend(["Ground truth", "Prediction"])
        # plt.show()
    

