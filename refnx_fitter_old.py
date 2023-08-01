"""
Created on Sun Aug 29 16:09:47 2021

@author: Lukas
"""

import os.path
import numpy as np
import matplotlib.pyplot as plt
import scipy
import pickle
from matplotlib import rcParams

import refnx
from refnx.dataset import ReflectDataset, Data1D
from refnx.analysis import Transform, CurveFitter, Objective, Model, Parameter
from refnx.reflect import SLD, Slab, ReflectModel
from refnx.reflect import Linear, Tanh, Interface, Erf, Sinusoidal, Exponential
from eigene.refnx_transform_fresnel import Transform_Fresnel

save_file = ()


def fresnel(qc, qz, roughness=0):
    return (np.exp(-qz**2 * roughness**2) *
            abs((qz - np.sqrt((qz**2 - qc**2) + 0j)) /
            (qz + np.sqrt((qz**2 - qc**2)+0j)))**2)

settings = dict(file_path = "./processed/reflectivity/",
                file_name = ["h2o_sample_01"],
                save_figures = True,
                save_obj_struc = True,
                file_save = "./processed/reflectivity/",
                figure_save = "./processed/reflectivity/",
                objective_struc_subname = None,
                q_c = 0.0216,    # q_c h2o
                            )
file = settings["file_path"] + settings["file_name"][0] + ".dat"
data_unpolished = ReflectDataset(file)
x_unpolished = data_unpolished.x
y_err_unpolished = data_unpolished.y_err
y_unpolished = data_unpolished.y

masky = np.logical_and(data_unpolished.x < 0.8, data_unpolished.x > 0.06)
data = ReflectDataset(file, mask = masky)

# Fit no layer

air = SLD(0, name='air')
h2o_bulk_sld = SLD(9.7, name='h2o_bulk')
h2o_bulk = h2o_bulk_sld(0, 2.55)

structure = air | h2o_bulk

#structure[1].interfaces = Tanh()
#structure[1].interfaces = Erf()
#structure[1].interfaces = Sinusoidal()
#structure[1].interfaces = Exponential()

#h2o_layer.thick.setp(bounds=(15, 5000), vary=True)
h2o_bulk.rough.setp(bounds=(2.2, 3.8), vary=True)
h2o_bulk.sld.real.setp(bounds=(8.0, 15.5), vary=True)

model = ReflectModel(structure, bkg=2e-11, dq=5.0, scale=1)
model.scale.setp(bounds=(0.7, 1.85), vary=False)
model.bkg.setp(bounds=(1e-17, 9e-8), vary=True)

objective = Objective(model, data, transform=Transform_Fresnel('fresnel', use_weights = True, qc = settings["q_c"], roughness = 0))
print(objective.chisqr(), objective.logp(), objective.logl(), objective.logpost())
fitter = CurveFitter(objective)
fitter.fit('differential_evolution')
qc_fit = np.sqrt(16*np.pi*objective.parameters[1]["h2o_bulk"][1].value*10**(-6))

rcParams["figure.figsize"] = 12, 8
rcParams["font.size"] = 20

fig1 = plt.figure()
plt.plot(*structure.sld_profile())
plt.ylabel('SLD in $10^{-6} \AA^{-2}$')
plt.xlabel(r'z in $\AA$')
h2o_half = objective.parameters[1]["h2o_bulk"][1].value/2
if len(objective.parameters[1]) == 4:
    h2o_half = objective.parameters[1][1][1].value/2
    sld_min = np.abs(structure.sld_profile()[1]-h2o_half)
    start = np.where(sld_min == sld_min.min())
    step_sld = np.heaviside(structure.sld_profile()[0] - structure.sld_profile()[0][start[0][0]], 1)*h2o_half*2*np.heaviside(-(structure.sld_profile()[0] - structure.sld_profile()[0][start[0][0]]) + objective.parameters[1][1][0].value, 1)
    h2o_half = objective.parameters[1][2][1].value/2
    step_sld += np.heaviside(structure.sld_profile()[0] - structure.sld_profile()[0][start[0][0]] - objective.parameters[1][1][0].value, 1)*h2o_half*2
    
if len(objective.parameters[1]) == 3:
    h2o_half = objective.parameters[1][1][1].value/2
    sld_min = np.abs(structure.sld_profile()[1]-h2o_half)
    start = np.where(sld_min == sld_min.min())
    step_sld = np.heaviside(structure.sld_profile()[0] - structure.sld_profile()[0][start[0][0]], 1)*h2o_half*2*np.heaviside(-(structure.sld_profile()[0] - structure.sld_profile()[0][start[0][0]]) + objective.parameters[1][1][0].value, 1)
    h2o_half = objective.parameters[1][2][1].value/2
    step_sld += np.heaviside(structure.sld_profile()[0] - structure.sld_profile()[0][start[0][0]] - objective.parameters[1][1][0].value, 1)*h2o_half*2
    plt.plot(structure.sld_profile()[0], step_sld,linestyle = "--")
elif len(objective.parameters[1]) == 2:
    h2o_half = objective.parameters[1]["h2o_bulk"][1].value/2
    sld_min = np.abs(structure.sld_profile()[1]-h2o_half)
    start = np.where(sld_min == sld_min.min())
    step_sld = np.heaviside(structure.sld_profile()[0] - structure.sld_profile()[0][start[0][0]], 1)*h2o_half*2
    sld_min = np.abs(structure.sld_profile()[1]-h2o_half)
    start = np.where(sld_min == sld_min.min())
    step_sld = np.heaviside(structure.sld_profile()[0] - structure.sld_profile()[0][start[0][0]], 1)*h2o_half*2
    plt.plot(structure.sld_profile()[0], step_sld,linestyle = "--")
elif len(objective.parameters[1]) == 1:
    plt.plot(structure.sld_profile()[0], step_sld,linestyle = "--")

plt.show()
plt.close()
if settings["save_figures"] == True:
    if os.path.exists(settings["file_save"]) == False:
        os.mkdir(settings["file_save"])

if settings["save_figures"] == True:
    if settings["save_obj_struc"] == True:
        if settings["objective_struc_subname"] != None:
            fig1.savefig(settings["file_save"] + settings["file_name"][0] + "_fitted_sld" +  "_" + settings["objective_struc_subname"])
        else:
            n = 1
            while os.path.exists(settings["file_save"] + settings["file_name"][0] + "_fitted_sld" +  "_" + str(n) + ".png") == True:
                n = n + 1
            fig1.savefig(settings["file_save"] + settings["file_name"][0] + "_fitted_sld" +  "_" + str(n))

fig2 = plt.figure()
ax = fig2.gca()
ax.set_yscale("log")
x,y,y_err,model = (data.x, data.y, data.y_err, objective.model(data_unpolished.x))
ax.errorbar(x_unpolished, fresnel(qc_fit, x_unpolished, roughness = 0), c="#424242", ls="--")
ax.errorbar(x_unpolished,y_unpolished,y_err_unpolished,color="blue",marker="o",ms=3,lw=0,elinewidth=2)
ax.errorbar(x_unpolished, model, color="red")
ax.legend(["Fresnel", "Mes. points", "Fit"])
plt.xlabel(r'$q_z\;in\; \AA^{-1}$')
plt.ylabel(r'$R$')
plt.show()
plt.close()

if settings["save_figures"] == True:
    if settings["save_obj_struc"] == True:
        if settings["objective_struc_subname"] != None:
            fig2.savefig(settings["file_save"] + settings["file_name"][0] + "_fitted_reflectivtiy" +  "_" + settings["objective_struc_subname"])
        else:
            n = 1
            while os.path.exists(settings["file_save"] + settings["file_name"][0] + "_fitted_reflectivtiy" +  "_" + str(n) + ".png") == True:
                n = n + 1
            fig2.savefig(settings["file_save"] + settings["file_name"][0] + "_fitted_reflectivtiy" +  "_" + str(n))

fig3 = plt.figure()
ax = fig3.gca()
#ax.set_ylim([-0.2,1.5])
ax.set_yscale("log")
roughness_fit = objective.parameters[1][1][3].value
x,y,y_err,model = (data.x, data.y, data.y_err, objective.model(data_unpolished.x))
ax.errorbar(x_unpolished,y_unpolished/fresnel(qc_fit, x_unpolished, roughness = 0),y_err_unpolished/fresnel(qc_fit, x_unpolished, roughness = 0),color="blue",marker="o",ms=3,lw=0,elinewidth=2)
ax.errorbar(x_unpolished, model/fresnel(qc_fit, x_unpolished, roughness = 0), color="red")
ax.legend(["Mes. points", "Fit"])
plt.xlabel(r'$q_z\;in\; \AA^{-1}$')
plt.ylabel(r'$R/R_F$')
plt.ylim(-0.5,5)
plt.show()
plt.close()

if settings["save_figures"] == True:
    if settings["save_obj_struc"] == True:
        if settings["objective_struc_subname"] != None:
            fig3.savefig(settings["file_save"] + settings["file_name"][0] + "_fitted_r_rf" +  "_" + settings["objective_struc_subname"])
        else:
            n = 1
            while os.path.exists(settings["file_save"] + settings["file_name"][0] + "_fitted_r_rf" +  "_" + str(n) + ".png") == True:
                n = n + 1
            fig3.savefig(settings["file_save"] + settings["file_name"][0] + "_fitted_r_rf" +  "_" + str(n))


fig4 = plt.figure()
ax = fig4.gca()
ax.set_xlim([0,0.05])
ax.set_yscale("log")
x,y,y_err,model = (data.x, data.y, data.y_err, objective.model(data_unpolished.x))
ax.errorbar(x_unpolished,y_unpolished/fresnel(qc_fit, x_unpolished, roughness = 0),y_err_unpolished/fresnel(qc_fit, x_unpolished, roughness = 0),color="blue",marker="o",ms=3,lw=0,elinewidth=2)
ax.errorbar(x_unpolished, model/fresnel(qc_fit, x_unpolished, roughness = 0), color="red")
ax.legend(["Mes. points", "Fit"])
plt.xlabel(r'$q_z\;in\; \AA^{-1}$')
plt.ylabel(r'$R/R_F$')
plt.show()
plt.close()
if settings["save_figures"] == True:
    if settings["save_obj_struc"] == True:
        if settings["objective_struc_subname"] != None:
            fig4.savefig(settings["file_save"] + settings["file_name"][0] + "_fitted_lowqz_reflectivtiy" +  "_" + settings["objective_struc_subname"])
        else:
            n = 1
            while os.path.exists(settings["file_save"] + settings["file_name"][0] + "_fitted_lowqz_reflectivtiy" +  "_" + str(n) + ".png") == True:
                n = n + 1
            fig4.savefig(settings["file_save"] + settings["file_name"][0] + "_fitted_lowqz_reflectivtiy" +  "_" + str(n))

if settings["save_obj_struc"] == True:
    if settings["objective_struc_subname"] != None:
        objetive_file = settings["file_save"] + settings["file_name"][0] + "_" + settings["objective_struc_subname"] + "_object" + ".txt"
        structure_file = settings["file_save"] + settings["file_name"][0] + "_" + settings["objective_struc_subname"] + "_structure" + ".txt"
        data_file = settings["file_save"] + settings["file_name"][0] + "_" + settings["objective_struc_subname"] + "_data" + ".txt"
    else:
        n = 1
        while os.path.exists(settings["file_save"] + settings["file_name"][0] + "_" + str(n) + "_object" + ".txt") == True:
            n = n + 1
        objetive_file = settings["file_save"] + settings["file_name"][0] + "_" + str(n) + "_object" + ".txt"
        structure_file = settings["file_save"] + settings["file_name"][0] + "_" + str(n) + "_structure" + ".txt"
        data_file = settings["file_save"] + settings["file_name"][0] + "_" + str(n) + "_data" + ".txt"
        
    filehandler = open(objetive_file, 'wb')
    pickle.dump(objective, filehandler)
    filehandler.close()
    
    filehandler = open(structure_file, 'wb')
    pickle.dump(structure, filehandler)
    filehandler.close()
    
    filehandler = open(data_file, 'wb')
    pickle.dump(data_unpolished, filehandler)
    filehandler.close()
    
    parameter_file = settings["file_save"] + settings["file_name"][0] + "_parameter" + str(n) + ".txt"
    f = open(parameter_file,'w')
    for i in range(len(objective.parameters[0])):
        f.write(str(objective.parameters[0][i]) + "\n")
    for i in range(len(objective.parameters[1])):
        for j in range(len(objective.parameters[1][i])):
            f.write(str(objective.parameters[1][i][j]) + "\n")
    f.write("Calculation of the critical angle based on SLD: " + str(qc_fit) + "\n")
    f.close()

print(objective)
print("Calculation of the critical angle based on SLD: " + str(qc_fit))