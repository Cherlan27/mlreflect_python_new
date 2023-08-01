# -*- coding: utf-8 -*-
"""
Created on Thu Jan  5 11:07:38 2023

@author: petersdorf
"""

# Test
# Test feature
# Test2

import mlreflect
from mlreflect.utils import check_gpu
import matplotlib.pyplot as plt
import numpy as np
from mlreflect.data_generation import Layer, Substrate, AmbientLayer, MultilayerStructure
from mlreflect.training import Trainer
from mlreflect.data_generation import ReflectivityGenerator
from mlreflect.curve_fitter import CurveFitter
import pandas as pd

q = np.linspace(0.001, 1, 100)

substrate = Substrate('H2O', (2.1, 2.8), (5, 13))
#layer = Layer("Test", 10, 2.5, 17)
ambient = AmbientLayer('ambient', 0)

sample = MultilayerStructure()
sample.set_substrate(substrate)
#sample.add_layer(layer)
sample.set_ambient_layer(ambient)

print(sample)

trainer = Trainer(sample, q, random_seed = 10)
trainer.generate_training_data(2**19)
trainer.training_data['labels'].head(5)

generator = ReflectivityGenerator(q, sample)
sld_profiles = generator.simulate_sld_profiles(trainer.training_data['labels'].head(5))

fig = plt.figure(dpi = 300)
for i in range(5):
    plt.plot(sld_profiles[i][0], sld_profiles[i][1])
plt.xlabel('Depth [A]')
plt.ylabel('SLD')
plt.title('SLD profiles')
plt.show()

fig2 = plt.figure(dpi = 300)
for i in range(5):
    plt.semilogy(q, trainer.training_data["reflectivity"][i, :], label = str(i))
plt.xlabel("q [1/A]")
plt.ylabel("Reflectivity")
plt.legend()
plt.show()

trained_model, hist = trainer.train(n_epochs = 100, batch_size = 512, verbose = 1)
plt.plot(hist.history['loss'], label = "loss")
plt.plot(hist.history["val_loss"], label = "val_loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss (linear)")
plt.legend()
plt.show()

plt.semilogy(hist.history["loss"], label = "loss")
plt.semilogy(hist.history["val_loss"], label = "val_loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss (log")
plt.legend()
plt.show()

curve_fitter = CurveFitter(trained_model)
test_labels = generator.generate_random_labels(2**12)
test_reflectivity = generator.simulate_reflectivity(test_labels)
fit_output = curve_fitter.fit_curve(test_reflectivity, q, polish = False, optimize_q = False)

predicted_test_labels = fit_output["predicted_parameters"]
predicted_test_reflectivity = fit_output["predicted_reflectivity"]

fig3 = plt.figure(dpi = 300)
plt.plot(predicted_test_labels["H2O_roughness"], test_labels["H2O_roughness"], '.', label = "predicted", alpha = 0.2)
plt.plot(predicted_test_labels["H2O_roughness"], predicted_test_labels["H2O_roughness"], label = "ground truth")
plt.legend()
plt.xlabel("Ground truth")
plt.ylabel("Roughness")
plt.show()

fig4 = plt.figure(dpi = 300)
plt.plot(predicted_test_labels["H2O_sld"], test_labels["H2O_sld"], '.', label = "predicted", alpha = 0.2)
plt.plot(predicted_test_labels["H2O_sld"], predicted_test_labels["H2O_sld"], label = "ground truth")
plt.legend()
plt.xlabel("Ground truth")
plt.ylabel("SLD")
plt.show()

abs(predicted_test_labels - test_labels).mean()

# test_plotted_index = []
# fig = plt.figure(dpi = 300)
# for j,i in enumerate(np.random.uniform(low = 0, high = 4000, size = 3).astype(int)):
#     print(i)
#     if j != 0:
#         plt.semilogy(q, test_reflectivity[i, :]*10**j, 'o', markerfacecolor = "white", markeredgecolor = "blue", label='_nolegend_')
#     else:
#         plt.semilogy(q, test_reflectivity[i, :]*10**j, 'o', markerfacecolor = "white", markeredgecolor = "blue")
#     plt.xlabel("q [1/A]")
#     plt.ylabel("Reflectivity [norm.]")
#     test_plotted_index.append(i)
# for j,i in enumerate(test_plotted_index):
#     plt.semilogy(q, predicted_test_reflectivity[i, :]*10**j, label = "predicted", alpha = 1, color="red")
# plt.legend(["Ground truth", "Prediction"])
# plt.show()

test_plotted_index = []
fig = plt.figure(dpi = 300)
for j,i in enumerate(np.random.uniform(low = 0, high = 4000, size = 5).astype(int)):
    plt.semilogy(q, test_reflectivity[i, :]*10**j, 'o', markerfacecolor = "white", label='_nolegend_')
    plt.xlabel(r"q ($\AA^{1}$)")
    plt.ylabel("Reflectivity [norm.]")
    test_plotted_index.append(i)
plt.legend(["Ground truth", "Prediction"])
plt.show()



# Own experimental data
df= pd.read_csv('Data/rbbr_Reflectivity_rbbr_3mol_s1_3_off.dat', sep = "\t")
#q_exp = np.concatenate((np.array(df["//qz"])[1:-6], np.array(df["//qz"])[-5:]))
#intensity_exp = np.concatenate((np.array(df["intensity_normalized"])[1:-6], np.array(df["intensity_normalized"])[-5:]))
q_exp = np.array(df["//qz"])[:-3]
intensity_exp = np.array(df["intensity_normalized"])[:-3]/7.58493186e-01


experimental_fit_output = curve_fitter.fit_curve(intensity_exp, q_exp, polish = True, optimize_q = True)
pred_experimental_reflectivity = experimental_fit_output["predicted_reflectivity"]
pred_experimental_test_labels = experimental_fit_output["predicted_parameters"]

fig = plt.figure(dpi = 300)
plt.semilogy(q_exp, intensity_exp, 'o', markerfacecolor = "white", markeredgecolor = "blue", label = "Experiment")
plt.semilogy(q_exp, pred_experimental_reflectivity[0], label = "Prediction", color="red")
plt.legend()
plt.xlabel("q [1/A]")
plt.ylabel("Reflectivity [norm]")
plt.show()

print("Nices Features")
print("Last but not least")
print("Super test")
print(pred_experimental_test_labels)
print("Super Text hier")
print("Second Test")
