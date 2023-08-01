# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 12:57:34 2023

@author: petersdorf
"""

import numpy
import h5py
import sys
from PIL import Image
import pandas
import os
import itertools
import re

from matplotlib import pyplot as plt
from matplotlib import rcParams
from matplotlib.ticker import MultipleLocator

from eigene.abs_overlap_fit_poly import Absorber
from eigene.fio_reader import read
from eigene.p08_detector_read import p08_detector_read

class make_xrr():
    def __init__(self,
                 data_directory = "K:/SYNCHROTRON/Murphy/XRR-data_example/Rohdaten",
                 use_flatfield = True,
                 use_mask = None,
                 experiment = "test",
                 detector = "lambda",
                 
                 scan_numbers = list(range(1535,1546+1)),
                 
                 detector_orientation = "vertical",
                 footprint_correct = True,
                 beam_width = 20e-6, 
                 sample_length = 81.4e-3,
                 
                 roi = (14, 70, 40, 22),
                 roi_offset = 30,
                 
                 calculate_abs = True,
                 
                 monitor = "ion2",
                 primary_intensity = "auto",
                 auto_cutoff = [0.015, 0.003],
                 auto_cutoff_nom = [0.0184, 0.0013],
                 scan_number_primary = 1137,
                 
                 qc = 0.0216,    
                 roughness = 2.55, 
                 
                 save_results = True,
                 show_plot = True,
                 save_plot = True,
                 
                 out_data_directory = "./processed/",
                 out_user_run = "2021-04",
                 out_experiment = "test",
                 out_typ_experiment = "sample_01", 
                 ):
        
        self.qz = []
        self.intensity = []
        self.e_intensity = []
        
        self.data_directory = data_directory
        self.use_flatfield = use_flatfield
        self.use_mask = use_mask
        self.flatfield = "./Module_2017-004_GaAs_MoFluor_Flatfielddata.tif"
        self.pixel_mask = "./Module_2017-004_GaAs_mask.tif"
        self.experiment = experiment
        self.detector = detector
        self.file_finding = "auto"
        self.roi_finding = "auto"
        self.roi_auto_set = False
        
        self.scan_numbers = scan_numbers
        
        self.detector_orientation = detector_orientation
        self.footprint_correct = footprint_correct
        self.beam_width = beam_width
        self.sample_length = sample_length
        self.wl = 12.38/18 * 1e-10
        
        self.roi = roi
        self.roi_offset = roi_offset
        
        # absorber factors:
        self.absorbers = Absorber()
        self.calculate_abs = calculate_abs           
        self.absorber_factors = {1: 12.617,
                            2: 11.0553,
                            3: 11.063,
                            4: 11.048,
                            5: 11.7,
                            6: 12.813},
        
        self.monitor = monitor
        self.primary_intensity = primary_intensity
        self.nom_scan_numbers = scan_numbers
        self.auto_cutoff = auto_cutoff
        self.auto_cutoff_nom = auto_cutoff_nom
        self.scan_number_primary = scan_number_primary
        
        self.qc = qc    
        self.roughness = roughness
        
        # output data:
        self.save_results = save_results
        self.show_plot = show_plot
        self.save_plot = save_plot
        
        # results are written to this file:
        self.out_data_directory = out_data_directory
        self.out_user_run = out_user_run
        self.out_experiment = out_experiment
        self.out_typ_experiment = out_typ_experiment
      
    def __call__(self):
        self.xrr_calculate()
        
    def set_roi(self, roi, roi_offset = 30):
        self.roi = roi
        self.roi_offset = roi_offset
    
    def roi_getter(self, img, width = 60, height = 20, corner_detection = True):
        pix_list = numpy.array(([],[]))
        int_limit = 1000
        img2 = img
        if corner_detection == True:
            for i in range(len(img[0])-1):
                for j in range(len(img[1])-1):
                    if (i > 100 and j > 100) and (i < 1400 and j < 450):
                        img2[j,i] = 0
        while len(pix_list[0]) < 50:
            pix_list = numpy.where(img2 > int_limit)
            int_limit /= 2
        y_max = []
        y = numpy.bincount(pix_list[0]) 
        maximum = max(y) 
        for i in range(len(y)):
            if y[i] == maximum: 
                y_max.append(i)
                
        if len(y_max) > 1:
            y_max  = int(sum(y_max)/len(y_max))
        else:
            y_max = y_max[0]
         
        x_max = []
        x = numpy.bincount(pix_list[1]) 
        maximum = max(x) 
        for i in range(len(x)):
            if x[i] == maximum: 
                x_max.append(i)
        
        if len(x_max) > 1:
            x_max  = int(sum(x_max)/len(x_max))
        else:
            x_max = x_max[0]
            
        return (int(x_max-width/2), int(y_max-height/2), int(width), int(height))
        

    def file_finder(self, scan_number):
        file_search = self.data_directory
        matching = [s for s in os.listdir(file_search) if str(scan_number).rjust(5, "0") in s]
        if matching != []:
            file_searched = [s for s in matching if not "." in s][0]
            return file_searched
        else:
            return None


    def abs_fac(self, abs_val):
        abs_val = int(abs_val + 0.2)
        if abs_val == 0:
            return 1.0
        else:
            return self.absorber_factors[abs_val] * self.abs_fac(abs_val - 1)

    def fresnel(self, qc, qz, roughness=2.5):
        """
        Calculate the Fresnel curve for critical q value qc on qz assuming
        roughness.
        """
        return (numpy.exp(-qz**2 * roughness**2) *
                abs((qz - numpy.sqrt((qz**2 - qc**2) + 0j)) /
                (qz + numpy.sqrt((qz**2 - qc**2)+0j)))**2)

    def footprint_correction(self, q, intensity, b, L, wl = 68.88e-12):
        """ 
        Input:
            q [A^(-1)]: Inverse wavevector q_z
            intensity [a.u.]: reflected intensity
            b [mm]: beam width
            L [mm]: sample length
            wl [A^(-1)]: wavelength of X-ray             
        Output:
            intensity2[a.u.]: Corrected intensity
        """
        q_b = (4*numpy.pi/wl*b/L)*10**(-10)
        intensity2 = intensity
        i = 0
        for i in range(0,len(q),1):
            if q[i] < q_b:
                intensity2[i] = intensity2[i]/(q[i]/q_b)
                i += 1
        else:
            None
        return intensity2
    
    def get_flatfield(self):
        """ 
        Flatfield correction for the lambda detector.
        Output:
            Flatfield: Array
        """
        flatfield_2 = numpy.ones((516,1556))
        flatfield = numpy.array(Image.open(self.flatfield))
        return flatfield
        
    def get_mask(self ):
        """ 
        Mask for the detector.
        Output:
            Mask: Array
        """
        file_mask = h5py.File(self.mask, "r")
        img_mask = numpy.array(file_mask["/entry/instrument/detector/data"])[0]
        file_mask.close()
        mask = numpy.zeros_like(img_mask)
        mask[(img_mask > 1)] = 1
        mask = (mask == 1)
        mask_value = 0
        return mask
    
    def xrr_calculate(self):
        """ 
        Calculation of the XRR intensity and sort the qz-values of all XRR scans.
        """
        qz_temp, s_intens_temp, s_e_intens = self.get_scans(self.scan_numbers)
        intensity, e_intensity = self.absorber_calculation(s_intens_temp, s_e_intens)
        
        qz = []
        for _, qz_s in qz_temp.items():
            for qz_item in qz_s:
                 qz.append(qz_item)
        
        m = numpy.argsort(qz)
        qz = numpy.array(qz)[m]
        intensity = intensity[m]
        e_intensity = e_intensity[m]
        
        self.qz = qz
        self.intensity = numpy.array(intensity)
        self.e_intensity = numpy.array(e_intensity)
        
        primary = self.normalisation()
        self.primary = primary
        
        self.intensity_norm = intensity/primary
        self.e_intensity_norm = e_intensity/primary
        
    def get_scans(self, scan_numbers):
        """ 
        Load the fio file for header and monitor data and angles (qz-values).
        Get the detector image from the nexus file (p08_detector_read).
        Output:
            qz: qz-values [Dictionary]
            temp_intens: intensities of the scans [Dictionary]
            temp_e_intens: intensity errors of the scans [Dictionary]
        """
        temp_intens = {}
        temp_e_intens = {}
        qz = {}
        
        for scan_number in scan_numbers:
            if self.file_finding == "auto":
                file_search = self.file_finder(scan_number)
                temp = re.compile("_[0-9][0-9][0-9][0-9][0-9]")
                experimental_part = re.split(temp, file_search)[0]
                fio_filename = self.data_directory + "/" + experimental_part + "_" + str(scan_number).rjust(5, "0") + ".fio"
                detector_images = p08_detector_read(self.data_directory, experimental_part, scan_number, self.detector)()
            else: 
                fio_filename = "{0}/{1}_{2:05}.fio".format(self.data_directory, self.experiment, scan_number)
                
                detector_images = p08_detector_read(self.data_directory, self.experiment, scan_number, self.detector)()
                
            
            self.header, self.column_names, self.data, self.scan_cmd = read(fio_filename)
            self.s_moni = self.data[self.monitor]
            s_alpha = self.data["alpha_pos"]
            s_beta = self.data["beta_pos"]
            s_qz = ((4 * numpy.pi / self.wl) *
                    numpy.sin(numpy.radians(s_alpha + s_beta) / 2.0) * 1e-10)

            n_images = detector_images.shape[0]
            n_points = min(n_images, len(s_alpha))
            images_intens, images_e_intens = self.get_images(detector_images, n_points)
            if self.footprint_correct == True:
                images_intens = self.footprint_correction(s_qz, images_intens, self.beam_width, self.sample_length)
            temp_int_values, temp_int_e_values = self.prep_absorber_calculation(s_qz, images_intens, images_e_intens, scan_number)
            temp_intens[scan_number] = temp_int_values
            temp_e_intens[scan_number] = temp_int_e_values
            qz[scan_number] = s_qz
        return qz, temp_intens, temp_e_intens
            
    def get_images(self, detector_images, n_points):
        """ 
        Get the intensities of the single images of the scans.
        Applay flatfield and mask if necessary.
        Output:
            images_intens: summed intensities of the image 
            images_e_intens: summed intensity errors of the image 
        """
        images_intens = []
        images_e_intens = []
        for n in range(n_points):
            img = detector_images[n]
        
            # flatfield correction
            if self.use_flatfield == True:
                flatfield = self.get_flatfield()
                img = img / flatfield
            
            if self.use_mask == True:
                mask, mask_value = self.use_mask()
                img[mask] = mask_value
            
            p_intens, p_e_intens = self.specular_calculation(img, n = n)
            images_intens.append(p_intens)
            images_e_intens.append(p_e_intens)
        return images_intens, images_e_intens
    
    def specular_calculation(self, img, n = 0):
        """ 
        Calculation of the specular intensity due to choosen ROI.
        Output:
            p_intens: intensity sum within ROI
            p_e_intens: intensity sum error within ROI
        """
        if self.roi_finding == "auto" and self.roi_auto_set == False:
            self.roi = self.roi_getter(img)
            self.roi_auto_set = True
        
        p_specular = img[self.roi[1]:(self.roi[1]+self.roi[3]),self.roi[0]:(self.roi[0]+self.roi[2])].sum() 
        p_bg0, p_bg1 = self.background_calculation(img)
        p_intens = ((p_specular - (p_bg0 + p_bg1) / 2.0) / self.s_moni[n])
        
        if self.monitor == "Seconds":
            p_e_intens = ((numpy.sqrt(p_specular) + (numpy.sqrt(p_bg0) + numpy.sqrt(p_bg1)) / 2.0) / self.s_moni[n])
        else:    
            p_e_intens = ((numpy.sqrt(p_specular) + (numpy.sqrt(p_bg0) + numpy.sqrt(p_bg1)) / 2.0) / self.s_moni[n] 
                         + abs (0.1 * (p_specular - (p_bg0 + p_bg1) / 2.0) / self.s_moni[n]))
        return p_intens, p_e_intens
    
    def background_calculation(self, img):
        """ 
        Calculation of the background intensity due to choosen ROI.
        Output:
            p_bg0: background next to the ROI
            p_bg1: background next to the ROI (other side)
        """
        if self.detector_orientation == "horizontal":            
            p_bg0 = img[self.roi[1]:(self.roi[1]+self.roi[3]),
                        (self.roi[0]+self.roi[2]+self.roi_offset):(self.roi[0]+2*self.roi[2]+self.roi_offset)].sum()
            p_bg1 = img[self.roi[1]:(self.roi[1]+self.roi[3]),
                        (self.roi[0]-self.roi[2]-self.roi_offset):(self.roi[0]-self.roi_offset)].sum()            
        elif self.detector_orientation == "vertical":            
            p_bg0 = img[(self.roi[1]+self.roi[3]+self.roi_offset):(self.roi[1]+2*self.roi[3]+self.roi_offset),
                        (self.roi[0]):(self.roi[0]+self.roi[2])].sum()
            p_bg1 = img[(self.roi[1]-self.roi[3]-self.roi_offset):(self.roi[1]-self.roi_offset),
                        (self.roi[0]):(self.roi[0]+self.roi[2])].sum()  
        return p_bg0, p_bg1
    
    def prep_absorber_calculation(self, s_qz, s_intens, s_e_intens, scan_number):
        """ 
        Sort the scans with the responding intensities to the absorber values
        Output:
            temp_int_value: intensities with the absorber value (due to changed absorber wheel)
            temp_int_e_value: intensity error with the absorber value (due to changed absorber wheel)
        """
        if self.calculate_abs == True:
            self.absorbers.add_dataset(self.header["abs"], s_qz, s_intens)
            temp_int_value = int(self.header["abs"]+0.1), s_intens
            temp_int_e_value = int(self.header["abs"]+0.1), s_e_intens
        elif self.calculate_abs == None or False:
            temp_int_value = s_intens * self.abs_fac(self.header["abs"])
            temp_int_e_value = s_intens * self.abs_fac(self.header["abs"])
        return temp_int_value, temp_int_e_value
            
    def absorber_calculation(self, temp_intens, temp_e_intens):
        """ 
        Calculate and fit the overlap between the scans with different absorber values.
        Concatenate the intensities to one array.
        Output:
            intensity: Intensity of the XRR in one array
            e_intensity: Intensity error of the XRR in one array
        """
        if self.calculate_abs == True:
            self.absorbers.calculate_from_overlaps()
            intensity = numpy.concatenate([self.absorbers(x[0])*numpy.array(x[1]) for x in list(temp_intens.values())])
            e_intensity = numpy.concatenate([self.absorbers(x[0])*numpy.array(x[1]) for x in list(temp_e_intens.values())])
        else:
            intensity = list(self.temp_intens.values())
            e_intensity = list(self.temp_e_intens.values())
        return intensity, e_intensity
    
    def normalisation(self):
        """ 
        Normalize the XRR scan in different ways.
        Auto: Normalization by crit qz area (own scan)
        Scan: Normalization by one scan (primary beam scan for example)
        Normalized: Normalization by other scan
        Output:
            primary: normalization intensity
        """
        if self.primary_intensity == "auto":
            primary = self.intensity[(self.qz > self.auto_cutoff[0]) & (self.qz<(self.qc - self.auto_cutoff[1]))].mean()
        elif self.primary_intensity == "scan":
            # data for normalization 
            # load scan
            fio_filename = "{0}/{1}_{2:05}.fio".format(self.data_directory, self.experiment, self.nom_scan_numbers)
            header, column_names, data, scan = read(fio_filename)
            # load monitor
            s_moni = data[self.monitor]
            s_alpha = data["alpha_pos"]
            s_beta = data["beta_pos"]
            s_qz = ((4 * numpy.pi / self.wl) *
                    numpy.sin(numpy.radians(s_alpha + s_beta) / 2.0) * 1e-10)
            # prepare data structures
            s_intens = []
            s_e_intens = []
            # load detector data
            detector_images = p08_detector_read(self.data_directory, self.experiment, self.scan_number_primery, self.detector)()
            n_images = detector_images.shape[0]
            
            img = detector_images[0]
            s_intens, s_e_intens = self.specular_calculation(img)
            
            norm_intens = numpy.array(s_intens)*Absorber.absorbers(list(self.temp_e_intens.values())[0][0])
            norm_e_intens = numpy.array(s_e_intens)*Absorber.absorbers(list(self.temp_e_intens.values())[0][0])
            primary = norm_intens[0]
        elif self.primary_intensity == "normalized":
            xrr_scan_nom = make_xrr()
            xrr_scan_nom.primary_intensity = "auto"
            xrr_scan_nom.scan_numbers = self.nom_scan_numbers
            xrr_scan_nom()
            primary = xrr_scan_nom.primary
        else:
            primary = self.primary_intensity
        return primary
        
    def saving_xrr_data(self):
        """ 
        Save the the qz, intensity and intensity_err data as dat file.
        """
        if self.save_plot or self.save_results:
            file_path = "{0}/reflectivity/".format(self.out_data_directory)
            try:
                os.makedirs(file_path)
            except OSError:
                if not os.path.isdir(file_path):
                    raise
          
        # == save data to file
        if self.save_results:
            out_filename = file_path + "/{0}_{1}.dat".format(self.out_experiment, self.out_typ_experiment)
            df = pandas.DataFrame()
            df["//qz"] = self.qz
            df["intensity_normalized"] = self.intensity / self.primary
            df["e_intensity_normalized"] = self.e_intensity / self.primary
            if os.path.exists(out_filename):
                self.save_results = input("Results .dat output file already exists. Overwrite? [y/n] ") == "y"
            if self.save_results:
                df.to_csv(out_filename, sep="\t", index=False)
    
    def get_xrr(self):
        return [self.qz, self.intensity/self.primary, self.e_intensity/self.primary]
    
    def plot_xrr(self):
        """ 
        Plot the XRR data.
        """
        if self.save_plot or self.save_results:
            file_path = "{0}/reflectivity/".format(self.out_data_directory)
            try:
                os.makedirs(file_path)
            except OSError:
                if not os.path.isdir(file_path):
                    raise
                    
        fig = plt.figure()
        fig.patch.set_color("white")
        ax = fig.gca()
        ax.set_yscale('log')
        ax.errorbar(self.qz, self.intensity/self.primary, yerr=self.e_intensity/self.primary, ls='none',
                    marker='o', mec='#cc0000', mfc='white', color='#ee0000',
                    mew=1.2)
        ax.errorbar(self.qz, self.fresnel(self.qc, numpy.array(self.qz), self.roughness), ls='--', c='#424242')
        ax.set_xlabel('q$_z$')
        ax.set_ylabel('R')
        ax.xaxis.set_minor_locator(MultipleLocator(0.05))
        plt.subplots_adjust(bottom=0.12)
        if self.show_plot:
            plt.show()
        if self.save_plot:
            out_plotname = file_path + "/{0}_{1}.png".format(self.out_experiment, self.out_typ_experiment)
            if os.path.exists(out_plotname):
                self.save_plot = input("plot output file already exists. Overwrite? [y/n] ") == "y"
            if self.save_plot:
                plt.savefig(out_plotname, dpi=300)
              
    def plot_xrr_crit(self):
        """ 
        Plot the XRR data as close shot for critical angle.
        """
        if self.save_plot or self.save_results:
            file_path = "{0}/reflectivity/".format(self.out_data_directory)
            try:
                os.makedirs(file_path)
            except OSError:
                if not os.path.isdir(file_path):
                    raise
        
        fig2 = plt.figure()
        fig2.patch.set_color("white")
        ax2 = fig2.gca()
        plt.xlim([0.01,0.04])
        plt.ylim([0.05,1.4])
        #ax2.set_yscale('log', nonposy='clip')
        ax2.errorbar(self.qz, self.intensity/self.primary, yerr=self.e_intensity/self.primary, ls='none',
                    marker='o', mec='#cc0000', mfc='white', color='#ee0000',
                    mew=1.2)
        ax2.errorbar(self.qz, self.fresnel(self.qc, self.qz, self.roughness), ls='--', c='#424242')
        ax2.set_xlabel('q$_z$')
        ax2.set_ylabel('R')
        ax2.xaxis.set_minor_locator(MultipleLocator(0.05))
        if self.show_plot:
            plt.show()
        if self.save_plot:
            out_plotname = file_path + "/{0}_{1}_crit.png".format(self.out_experiment, self.out_typ_experiment)
            if os.path.exists(out_plotname):
                self.save_plot = input("plot output file already exists. Overwrite? [y/n] ") == "y"
            if self.save_plot:
                plt.savefig(out_plotname, dpi=300)
        
