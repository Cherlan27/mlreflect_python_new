"""
refl_plot.py - Plot XRR curves.

Author: Sven Festersen <festersen@physik.uni-kiel.de>

Use this script to plot multiple reflectivity curves produced by the
refl_make.py script.
"""

import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.ticker import MultipleLocator
import pandas
import numpy
import os
import matplotlib
import tkinter

# == settings
settings = dict(filenames =[#"./processed/reflectivity/rbbr_test_off_alk.dat",
                            #"./processed/reflectivity/rbbr_test_on_alk.dat",
                            #"./processed/reflectivity/rbbr_test_1mol_off.dat",
                            #"./processed/reflectivity/rbbr_test_1mol_on.dat",
                            #"./processed/reflectivity/h2o_Reflectivity_h2o_s1_1_off.dat",
                            #"./processed/reflectivity/h2o_Reflectivity_h2o_s1_2_on.dat",
                            #"./processed/reflectivity/rbbr_Reflectivity_rbbr_3mol_s1_1_off.dat",
                            #"./processed/reflectivity/rbbr_Reflectivity_rbbr_3mol_s1_2_on.dat",
                            #"./processed/reflectivity/rbbr_Reflectivity_rbbr_3mol_s1_1_off.dat",
                            #"./processed/reflectivity/rbbr_Reflectivity_rbbr_3mol_s1_2_on.dat",
                            #"./processed/reflectivity/rbbr_Reflectivity_rbbr_3mol_s1_3_on.dat",
                            #"./processed/reflectivity/rbbr_Reflectivity_rbbr_3mol_s1_4_off.dat",
                            # "./processed/reflectivity/rbbr_Reflectivity_rbbr_3mol_ac_s1_1_off.dat",
                             #"./processed/reflectivity/rbbr_Reflectivity_rbbr_3mol_ac_s1_2_on.dat",
                             #"./processed/reflectivity/rbbr_Reflectivity_rbbr_3mol_ac_s1_3_off.dat",
                             #"./processed/reflectivity/rbbr_Reflectivity_rbbr_3mol_ac_s1_4_off.dat",
                             #"./processed/reflectivity/rbbr_Reflectivity_rbbr_3mol_ac_s2_1_off.dat",
                             #"./processed/reflectivity/rbbr_Reflectivity_rbbr_3mol_ac_s2_2_off.dat",
                             #"./processed/reflectivity/rbbr_Reflectivity_rbbr_3mol_ac_s2_3_on.dat",
                             #"./processed/reflectivity/rbbr_Reflectivity_rbbr_3mol_alk_s1_1_off.dat",
                             #"./processed/reflectivity/rbbr_Reflectivity_rbbr_3mol_alk_s1_2_on.dat",
                             #"./processed/reflectivity/rbbr_Reflectivity_rbbr_3mol_alk_s1_3_off.dat",
                             "./processed/reflectivity/rbbr_Reflectivity_rbbr_5mol_s1_1_off.dat",
                             #"./processed/reflectivity/rbbr_Reflectivity_rbbr_5mol_s1_2_on.dat",
                             #"./processed/reflectivity/rbbr_Reflectivity_rbbr_5mol_s1_3_off.dat",
                             #"./processed/reflectivity/rbbr_Reflectivity_rbbr_1mol_s1_1_off.dat",
                             #"./processed/reflectivity/rbbr_Reflectivity_rbbr_1mol_s1_2_on.dat",
                             #"./processed/reflectivity/rbbr_Reflectivity_srcl2_3mol_s1_1_off.dat",
                             #"./processed/reflectivity/rbbr_Reflectivity_srcl2_3mol_s1_2_on.dat",
                             #"./processed/reflectivity/rbbr_Reflectivity_srcl2_3mol_s1_3_off.dat",
                            #"./processed/reflectivity/rbbr_Reflectivity_ercl3_1mol_s1_1_off.dat",
                            #"./processed/reflectivity/rbbr_Reflectivity_ercl3_1mol_s1_2_on.dat",
                            #"./processed/reflectivity/rbbr_Reflectivity_ercl3_1mol_s1_3_off.dat",
                            #"./processed/reflectivity/rbbr_Reflectivity_ercl3_1mol_s1_4_off.dat",
                            #"./processed/reflectivity/rbbr_Reflectivity_rbbr_5mol.dat",
                            #"./processed/reflectivity/rbbr_test.dat"
                            ],
                primary_intensities = [
                                        "auto",
                                        "auto",
                                        "auto",
                                        "auto",
                                        "auto",
                                        "auto",
                                        "auto",
                                      ],
                colors = [
                          "blue",
                          "red",
                          "cyan",
                          "green",
                          "orange",
                          "magenta",
                          "black",
                          "blue",
                          "red"   
                          ],
                qc= [0.025, 0.025], # rbbr 3 mol
                #qc = 0.0257,    # rbbr 5 mol  
                #qc = 0.023,    # rbbr 1 mol 
                #qc = 0.0248,    # srcl 3 mol
                #qc = 0.0241,    # ercl 3 mol  
                #fresnel_roughness = 2.55, # rbbr 3 mol 
                # fresnel_roughness = 2.8, # rbbr 3 mol ac
                #fresnel_roughness = 2.55, # rbbr 5 mol
                #fresnel_roughness = 2.75, # srcl
                fresnel_roughness = 2.55, # ercl3
                save_file = "./processed/reflectivity/rbbr_Reflectivity_ercl3_1mol_s1_1_off")
                # roughness for hg is 0.8)


# == helper functions

def fresnel(qc, qz, roughness):
    """
    Calculate the Fresnel curve for critical q value qc on qz assuming
    roughness.
    """
    return (numpy.exp(-qz**2 * roughness**2) *
            abs((qz - numpy.sqrt((qz**2 - qc**2) + 0j)) /
            (qz + numpy.sqrt((qz**2 - qc**2)+0j)))**2)

# == plot
rcParams["figure.figsize"] = 8, 8
rcParams["font.size"] = 16
#rcParams["text.usetex"] = True
#rcParams["text.latex.preamble"] = r"\usepackage{sfmath}"


fig = plt.figure()
fig.patch.set_color("white")
ax1 = fig.add_subplot(211)
ax1.set_yscale("log", nonposy="clip")

qz_min, qz_max = -1, -1

for n in range(len(settings["filenames"])):
    print(settings["filenames"][n])
    filename = settings["filenames"][n]
    df = pandas.read_csv(filename, sep="\t")
    
    qz = numpy.array(df["//qz"])
    intensity = numpy.array(df["intensity_normalized"])
    e_intensity = numpy.array(df["e_intensity_normalized"])
    
    if n == 0:
        intensity_prim = numpy.array(df["intensity_normalized"])
    
    if settings["primary_intensities"][n] == "auto":
        primary = intensity_prim[(qz > 0.020) & (qz<settings["qc"][n])].mean()
    else:
        primary = settings["primary_intensities"][n]
    
    ax1.errorbar(df["//qz"], df["intensity_normalized"]/primary, yerr=df["e_intensity_normalized"]/primary, ls='none',
            marker='o', mec=settings["colors"][n], mfc='white', color=settings["colors"][n],
            mew=1.2, label=os.path.basename(filename).replace("_", r"\_"))
    if qz_min == -1:
        qz_min = min(qz)
    else:
        qz_min = min(qz_min, min(qz))
    if qz_max == -1:
        qz_max = max(qz)
    else:
        qz_max = max(qz_max, max(qz))
        
qz = numpy.arange(qz_min, qz_max, 0.01)
qz = numpy.array(df["//qz"])
fr = fresnel(settings["qc"][n], qz, settings["fresnel_roughness"])
ax1.errorbar(qz, fr, c="#424242", ls="--")
ax1.set_ylabel('$R$')
ax1.xaxis.set_minor_locator(MultipleLocator(0.05))
ax1.set_xticklabels([])
ax1.legend(fontsize=10)

# ==== R/RF
ax2 = fig.add_subplot(212)
for n in range(len(settings["filenames"])):
    filename = settings["filenames"][n]
    df = pandas.read_csv(filename, sep="\t")
    
    qz = numpy.array(df["//qz"])
    intensity = numpy.array(df["intensity_normalized"])
    e_intensity = numpy.array(df["e_intensity_normalized"])
    
    if n == 0:
        intensity_prim = numpy.array(df["intensity_normalized"])
    
    if settings["primary_intensities"][n] == "auto":
        primary = intensity_prim[(qz > 0.015) & (qz<settings["qc"][n])].mean()
    else:
        primary = settings["primary_intensities"][n]
        
    fr = fresnel(settings["qc"][n], qz, settings["fresnel_roughness"])
    
    # ax2.errorbar(df["//qz"], df["intensity_normalized"]/primary/fr, yerr=df["e_intensity_normalized"]/primary/fr, ls='none',
    #         marker='o', mec=settings["colors"][n], mfc='white', color=settings["colors"][n],
    #         mew=1.2)
    ax2.errorbar(df["//qz"], df["intensity_normalized"]/primary/fr, yerr=df["e_intensity_normalized"]/primary/fr, ls='none',
        marker='o', mec=settings["colors"][n], mfc='white', color=settings["colors"][n],
        mew=1.2)

ax2.xaxis.set_minor_locator(MultipleLocator(0.05))
ax2.set_ylabel('$R/R_F$')
ax2.set_xlabel('q$_z$')
ax2.set_xlim(*ax1.get_xlim())
#ax2.legend(fontsize=10)


plt.ylim([-1,4])
plt.subplots_adjust(bottom=0.12)
plt.tight_layout()
fig.savefig("{save_file}".format(**settings))
plt.show()





fig = plt.figure(figsize=(12,8), dpi=600)
fig.patch.set_color("white")
ax = fig.gca()
for n in range(len(settings["filenames"])):
    print(settings["filenames"][n])
    filename = settings["filenames"][n]
    df = pandas.read_csv(filename, sep="\t")
    
    qz = numpy.array(df["//qz"])
    intensity = numpy.array(df["intensity_normalized"])
    e_intensity = numpy.array(df["e_intensity_normalized"])
    
    if n == 0:
        intensity_prim = numpy.array(df["intensity_normalized"])
    
    if settings["primary_intensities"][n] == "auto":
        primary = intensity_prim[(qz > 0.020) & (qz<settings["qc"][n])].mean()
    else:
        primary = settings["primary_intensities"][n]
    
    ax.errorbar(df["//qz"], df["intensity_normalized"]/primary, yerr=df["e_intensity_normalized"]/primary, ls='none',
            marker='o', mec=settings["colors"][n], mfc='white', color=settings["colors"][n],
            mew=1.2, label=os.path.basename(filename).replace("_", r"\_"))
    if qz_min == -1:
        qz_min = min(qz)
    else:
        qz_min = min(qz_min, min(qz))
    if qz_max == -1:
        qz_max = max(qz)
    else:
        qz_max = max(qz_max, max(qz))
    
    
ax.xaxis.set_minor_locator(MultipleLocator(0.05))
qz = numpy.arange(qz_min, qz_max, 0.01)
qz = numpy.array(df["//qz"])
fr = fresnel(0.025, qz, settings["fresnel_roughness"])
ax.errorbar(qz, fr, c="#424242", ls="--")
ax.set_ylabel('$R/R_F$')
#ax.set_yscale("log")
ax.set_xlabel('q$_z$')
ax.set_xlim([0.015,0.05])
plt.show()















fig = plt.figure(figsize=(12,8), dpi=600)
fig.patch.set_color("white")
ax = fig.gca()
for n in range(len(settings["filenames"])):
    filename = settings["filenames"][n]
    df = pandas.read_csv(filename, sep="\t")
    
    qz = numpy.array(df["//qz"])
    intensity = numpy.array(df["intensity_normalized"])
    e_intensity = numpy.array(df["e_intensity_normalized"])
    
    if n == 0:
        intensity_prim = numpy.array(df["intensity_normalized"])
    
    if settings["primary_intensities"][n] == "auto":
        primary = intensity_prim[(qz > 0.015) & (qz<settings["qc"][n])].mean()
    else:
        primary = settings["primary_intensities"][n]
        
    fr = fresnel(settings["qc"][n], qz, 0)
    
    # ax.errorbar(df["//qz"], df["intensity_normalized"]/primary/fr, yerr=df["e_intensity_normalized"]/primary/fr, ls='none',
    #         marker='o', mec=settings["colors"][n], mfc='white', color=settings["colors"][n],
    #         mew=1.2)
    # ax.errorbar(df["//qz"], df["intensity_normalized"]/primary/fr, yerr=df["e_intensity_normalized"]/primary/, ls='none',
    #     marker='o', mec=settings["colors"][n], mfc='white', color=settings["colors"][n],
    #     mew=1.2)

    ax.errorbar(df["//qz"], df["intensity_normalized"]/primary/fr, yerr=df["e_intensity_normalized"]/primary/fr, ls='none',
        marker='o', mec=settings["colors"][n], mfc='white', color=settings["colors"][n],
        mew=1.2)
    
    
ax.set_ylabel('$R/R_F$')
ax.set_xlabel('q$_z$')
ax.set_yscale("log")

fig = plt.figure(figsize=(12,8), dpi=600)
ax = fig.gca()
for n in range(len(settings["filenames"])):
    filename = settings["filenames"][n]
    df = pandas.read_csv(filename, sep="\t")
    
    qz = numpy.array(df["//qz"])
    intensity = numpy.array(df["intensity_normalized"])
    e_intensity = numpy.array(df["e_intensity_normalized"])
    
    if n == 0:
        intensity_prim = numpy.array(df["intensity_normalized"])
    
    if settings["primary_intensities"][n] == "auto":
        primary = intensity_prim[(qz > 0.015) & (qz<settings["qc"][n])].mean()
    else:
        primary = settings["primary_intensities"][n]
        
    fr = fresnel(settings["qc"][n], qz, 0)
    ax.errorbar(df["//qz"]**2, df["intensity_normalized"]/primary/fr, yerr=df["e_intensity_normalized"]/primary/fr, ls='none',
        marker='o', mec=settings["colors"][n], mfc='white', color=settings["colors"][n],
        mew=1.2)
ax.set_xlim(0.05,0.4)
ax.set_ylim(0,1)
plt.show()
