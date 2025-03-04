import sys
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, './encoding')
import encoding
import argparse
import os
from pathlib import Path
import glob
import csv

import helpers
import encoding

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import colors
import nibabel
import nilearn
from nilearn import plotting
from nilearn import surface
from nilearn.maskers import NiftiSpheresMasker
from bids import BIDSLayout

from joblib import Parallel, delayed
from statsmodels.stats.multitest import fdrcorrection

from matplotlib.colors import LinearSegmentedColormap


class SecondLevelGroup(encoding.EncodingModel):
	#class attributes go here and must be initialized to something
	# process = 'EncodingModel'

	def __init__(self, args):
		self.process = 'SecondLevelGroup'
		self.dir = args.dir
		self.data_dir = args.dir + '/data'
		self.in_dir = args.out_dir + '/EncodingModel'
		self.in_dir_GLM = args.out_dir + '/SecondLevelIndividual'
		self.out_dir = args.out_dir + "/" + self.process
		self.subjects = []
		self.population = args.population
		self.sid = 'sub-'+self.population
		self.task = args.task
		self.mask = args.mask
		self.mask_name = self.mask
		self.model = args.model
		self.feature_of_interest = args.feature_of_interest
		self.smoothing_fwhm = args.smoothing_fwhm #change?
		self.chunklen = args.chunklen
		self.fMRI_data = []
		self.brain_shape = []
		self.affine = []
		self.feature_names = []
		self.features = []
		self.features1 = [] #will be high level labels
		self.features2 = [] #will be DNN layer
		self.features3 = []
		self.separate_dimensions = False
		self.weights = []
		self.save_weights = False
		self.model1_performance = []
		self.model1_performance_null_distribution = []
		self.model2_performance = []
		self.model2_performance_null_distribution = []
		self.n_top_voxels = '100'
		self.figure_dir = args.figure_dir + "/" + self.process
		Path(f'{self.out_dir}/{"weights"}').mkdir(exist_ok=True, parents=True)
		Path(f'{self.out_dir}/{"performance"}').mkdir(exist_ok=True, parents=True)
		Path(f'{self.out_dir}/{"ind_feature_performance"}').mkdir(exist_ok=True, parents=True)
		Path(f'{self.out_dir}/{"ind_product_measure"}').mkdir(exist_ok=True, parents=True)
		Path(f'{self.out_dir}/{"unique_variance"}').mkdir(exist_ok=True, parents=True)
		Path(f'{self.out_dir}/{"preference_map"}').mkdir(exist_ok=True, parents=True)
		Path(f'{self.out_dir}/{"features"}').mkdir(exist_ok=True, parents=True)
		Path(f'{self.out_dir}/{"perf_p_unc"}').mkdir(exist_ok=True, parents=True)
		Path(f'{self.out_dir}/{"perf_p_fdr"}').mkdir(exist_ok=True, parents=True)

		Path(f'{self.figure_dir}/{"weights"}').mkdir(exist_ok=True, parents=True)
		Path(f'{self.figure_dir}/{"performance"}').mkdir(exist_ok=True, parents=True)
		Path(f'{self.figure_dir}/{"ind_feature_performance"}').mkdir(exist_ok=True, parents=True)
		Path(f'{self.figure_dir}/{"ind_product_measure"}').mkdir(exist_ok=True, parents=True)
		Path(f'{self.figure_dir}/{"unique_variance"}').mkdir(exist_ok=True, parents=True)
		Path(f'{self.figure_dir}/{"performance/surface"}').mkdir(exist_ok=True, parents=True)
		Path(f'{self.figure_dir}/{"ind_feature_performance/surface"}').mkdir(exist_ok=True, parents=True)
		Path(f'{self.figure_dir}/{"ind_product_measure/surface"}').mkdir(exist_ok=True, parents=True)
		Path(f'{self.figure_dir}/{"unique_variance/surface"}').mkdir(exist_ok=True, parents=True)

		Path(f'{self.figure_dir}/{"preference_map"}').mkdir(exist_ok=True, parents=True)

		self.beta_weights = []
		self.predicted_time_series = []
		self.performance = []
		self.performance_null_distribution = []
		self.perf_p_fdr = []
		self.ind_feature_performance = []
		self.ind_product_measure = []
		self.ind_product_measure_proportion = []
		self.ind_feature_perf_p_fdr = []
		self.ind_feature_performance_null_distribution =[]
		self.preference1_map = []
		self.preference2_map = []
		self.preference3_map = []
		self.model_features = []
		self.perf_p_unc = []
		self.perf_fdr_reject = []
		self.perf_p_fdr = []
		self.ind_perf_p_unc = []
		self.ind_perf_fdr_reject = []
		self.ind_perf_p_fdr = []
		self.ind_prod_p_unc = []
		self.ind_prod_fdr_reject = []
		self.ind_prod_p_fdr = []
		self.performance_null = []
		self.features_preferred_delay=[]
		self.final_weight_feature_names=[]

		self.subjects = helpers.get_subjects(self.population)
		self.cmaps = helpers.get_cmaps()
		self.models_dict = helpers.get_models_dict()
		self.colors_dict = helpers.get_colors_dict()

		self.file_label = '_encoding_model-'+self.model + '_smoothingfwhm-'+str(self.smoothing_fwhm)+'_chunklen-'+str(self.chunklen)+'_mask-'+self.mask
		self.extra_label = '_smoothingfwhm-'+str(self.smoothing_fwhm)+'_chunklen-'+str(self.chunklen)+'_mask-'+self.mask
		self.file_label_glm = '_smoothingfwhm-'+str(self.smoothing_fwhm)+'_mask-'+self.mask

		self.models_dict = helpers.get_models_dict()
		self.combined_features = helpers.get_combined_features()


	def get_feature_index(self, subject,feature, selection_model=''):
		if(selection_model==''):
			file_label = subject+'_encoding_model-'+self.model+self.extra_label
		else:
			file_label = subject+'_encoding_model-'+selection_model+self.extra_label
		
		filename = self.in_dir+'/features/'+file_label+'_features.csv' 

		file = open(filename, "r")
		data = list(csv.reader(file, delimiter=','))[0]
		file.close()
		# print(data.index(feature))
		return data.index(feature)

	def compile_data(self,smooth=3.0):

			model_features = self.model.split('_')
			if(self.feature_of_interest!=None):
				sub_model_features = [feature for feature in model_features if feature!=self.feature_of_interest]
				sub_model_features = sorted(sub_model_features, key=str.casefold) #always alphabetical, ignore the case
				sub_model = '_'.join(sub_model_features)
			else:
				sub_model=self.model

			label = 'perf_raw'
			all_data_performance = []
			print(self.subjects)
			for subject in self.subjects[self.task]:
				try:
					print(subject)
					#load the raw performance data for each subject
					nii = nibabel.load(self.in_dir+'/performance/'+subject+self.file_label+'_measure-'+label+'.nii.gz')
					##smooth with 3.0 fwhm gaussian
					nii = nibabel.processing.smooth_image(nii,fwhm=smooth,mode='nearest')
					# view = plotting.view_img(
					#     nii, title=subject, cut_coords=[36, -27, 66],vmin=0,vmax=0.5, symmetric_cmap=False,opacity=0.5,
					# )
					# view.open_in_browser()
					performance = nii.get_fdata()
					affine = nii.affine
					all_data_performance.append(performance)
				except Exception as e:
					print(e)
					pass

			all_data_performance = np.array(all_data_performance)
			model1_performance = np.nanmean(all_data_performance,axis=0)
			self.brain_shape = model1_performance.shape
			self.affine = affine

			img = nibabel.Nifti1Image(model1_performance, affine)
			nibabel.save(img, self.out_dir+'/performance/'+self.sid+self.file_label+'_measure-'+label+'.nii.gz')

			
			if(self.do_stats):
				label = 'perf_null_distribution'
				all_data_performance = []
				for subject in self.subjects[self.task]:
					try:
						#load the raw performance data for each subject
						nii = nibabel.load(self.in_dir+'/performance/'+subject+self.file_label+'_measure-'+label+'.nii.gz')
						performance = nii.get_fdata()
						affine = nii.affine
						all_data_performance.append(performance)
					except Exception as e:
						print(e)
						pass

				all_data_performance = np.array(all_data_performance)
				model1_performance_null_distribution = np.nanmean(all_data_performance,axis=0)

				img = nibabel.Nifti1Image(model1_performance_null_distribution, affine)
				nibabel.save(img, self.out_dir+'/performance/'+self.sid+self.file_label+'_measure-'+label+'.nii.gz') 

			#if there is another model for calculating unique variance of a feature of interest, also load that
			if(self.feature_of_interest!='None'):
				label = 'perf_raw'
				all_data_performance = []
				for subject in self.subjects[self.task]:
					try:
						print(subject)
						#load the raw performance data for each subject
						file_label = '_encoding_model-'+sub_model+ '_smoothingfwhm-'+str(self.smoothing_fwhm)+'_chunklen-'+str(self.chunklen)+'_mask-'+self.mask
						nii = nibabel.load(self.in_dir+'/performance/'+subject+file_label+'_measure-'+label+'.nii.gz')
						performance = nii.get_fdata()
						affine = nii.affine
						all_data_performance.append(performance)
					except Exception as e:
						print(e)
						pass

				all_data_performance = np.array(all_data_performance)
				model2_performance = np.nanmean(all_data_performance,axis=0)

				img = nibabel.Nifti1Image(model2_performance, affine)
				nibabel.save(img, self.out_dir+'/performance/'+self.sid+file_label+'_measure-'+label+'.nii.gz')  

				if(self.do_stats):
					label = 'perf_null_distribution'
					all_data_performance = []
					for subject in self.subjects[self.task]:
						try:
							#load the raw performance data for each subject
							file_label = '_encoding_model-'+sub_model+ '_smoothingfwhm-'+str(self.smoothing_fwhm)+'_chunklen-'+str(self.chunklen)+'_mask-'+self.mask
							nii = nibabel.load(self.in_dir+'/performance/'+subject+file_label+'_measure-'+label+'.nii.gz')
							performance = nii.get_fdata()
							affine = nii.affine
							all_data_performance.append(performance)
						except Exception as e:
							print(e)
							pass

					all_data_performance = np.array(all_data_performance)
					model2_performance_null_distribution = np.nanmean(all_data_performance,axis=0)

					img = nibabel.Nifti1Image(model2_performance_null_distribution, affine)
					nibabel.save(img, self.out_dir+'/performance/'+self.sid+file_label+'_measure-'+label+'.nii.gz')  

			# all_weights = []
			# # if(self.model=='DNN'):
			# # 	self.feature_names = ['DNN_'+str(dim+1) for dim in range(0,11)]

			# for feature_name in self.feature_names:
			# 	all_data_weights = []
			# 	for subject in self.subjects:
			# 		nii = nibabel.load(self.in_dir+'/weights/'+subject+self.file_label+'_measure-weights_raw.nii.gz')
			# 		weights = nii.get_fdata()
			# 		feature_index = self.get_feature_index(subject,feature_name)
			# 		all_data_weights.append(weights[feature_index])
			# 	all_data_weights = np.array(all_data_weights)
			# 	avg_weights = np.nanmean(all_data_weights,axis=0)
			# 	all_weights.append(avg_weights)
			# weights = np.array(all_weights)
			# img = nibabel.Nifti1Image(weights, affine)
			# nibabel.save(img, self.out_dir+'/weights/'+self.sid+self.file_label+'_measure-weights_raw.nii.gz' )

			# all_ind_feature_perf = []
			# # if(self.model=='DNN'):
			# # 	self.feature_names = ['DNN_'+str(dim+1) for dim in range(0,11)]

			# for feature_name in self.feature_names:
			# 	all_data_ind_feature_perf = []
			# 	for subject in self.subjects:
			# 		try:
			# 			nii = nibabel.load(self.in_dir+'/ind_feature_performance/'+subject+self.file_label+'_measure-ind_perf_raw.nii.gz')
			# 			performance = nii.get_fdata()
			# 			feature_index = self.get_feature_index(subject,feature_name)
			# 			all_data_ind_feature_perf.append(performance[feature_index])
			# 		except Exception as e:
			# 			print(e)
			# 			pass
			# 	all_data_ind_feature_perf = np.array(all_data_ind_feature_perf)
			# 	avg_ind_feature_perf = np.nanmean(all_data_ind_feature_perf,axis=0)
			# 	all_ind_feature_perf.append(avg_ind_feature_perf)
			# ind_feature_performance = np.array(all_ind_feature_perf) #mask it for computing the preference maps, which need a flat array
			# img = nibabel.Nifti1Image(ind_feature_performance, affine)
			# nibabel.save(img, self.out_dir+'/ind_feature_performance/'+self.sid+self.file_label+'_measure-ind_perf_raw.nii.gz' )

			all_ind_product_measure= []
			# if(self.model=='DNN'):
			# 	self.feature_names = ['DNN_'+str(dim+1) for dim in range(0,11)]

			for feature_name in self.feature_names:
				# print(feature_name)
				all_data_ind_product_measure = []
				for subject in self.subjects[self.task]:
					try:
						nii = nibabel.load(self.in_dir+'/ind_product_measure/'+subject+self.file_label+'_measure-ind_product_measure_raw.nii.gz')
						data = nii.get_fdata()
						if(self.scale_by=='total_variance'):
						    data[data<0] = 0 #clip response values to 0
						    data = data/data.sum(axis=0,keepdims=1)
						if(feature_name in self.combined_features):
							for (ind,sub_feature_name) in enumerate(self.models_dict[feature_name]):
								feature_ind = self.get_feature_index(subject,sub_feature_name)

								sub_data = data[feature_ind]
								if(ind==0):
									overall = sub_data
								else:
									overall = overall+sub_data
							data = overall
						else:
							feature_index = self.get_feature_index(subject,feature_name)
							data = data[feature_index]
						# print(feature_index)
						nii = nibabel.Nifti1Image(data,nii.affine)
						#smooth after adding together layers if nec
						nii = nibabel.processing.smooth_image(nii,fwhm=smooth,mode='nearest')
						all_data_ind_product_measure.append(nii.get_fdata())
					except Exception as e:
						print(e)
						pass
				all_data_ind_product_measure = np.array(all_data_ind_product_measure)
				avg_ind_product_measure = np.nanmean(all_data_ind_product_measure,axis=0)
				all_ind_product_measure.append(avg_ind_product_measure)
			ind_product_measure = np.array(all_ind_product_measure) #mask it for computing the preference maps, which need a flat array
			img = nibabel.Nifti1Image(ind_product_measure, affine)
			nibabel.save(img, self.out_dir+'/ind_product_measure/'+self.sid+self.file_label+'_measure-ind_product_measure_raw.nii.gz' )


			# all_ind_feature_perf = []
			# for (ind,feature_name) in enumerate(self.feature_names[0:self.DNN_index]):
			# 	all_data_ind_feature_perf = []
			# 	for subject in self.subjects:
			# 		nii = nibabel.load(self.in_dir+'/ind_feature_performance/'+subject+self.file_label+'_measure-ind_perf_null_distribution_masked.nii.gz')
			# 		performance = nii.get_fdata()
			# 		all_data_ind_feature_perf.append(performance[:,ind])
			# 	all_data_ind_feature_perf = np.array(all_data_ind_feature_perf)
			# 	avg_ind_feature_perf = np.nanmean(all_data_ind_feature_perf,axis=0)
			# 	all_ind_feature_perf.append(avg_ind_feature_perf)
			# self.ind_feature_performance_null_distribution = np.array(all_ind_feature_perf) #already masked
			# img = nibabel.Nifti1Image(self.ind_feature_performance_null_distribution, self.affine)
			# nibabel.save(img, self.out_dir+'/ind_feature_performance/'+self.sid+self.file_label+'_measure-ind_perf_null_distribution_masked.nii.gz' )

			#MASK all data
			self.load_mask()
			self.model1_performance = model1_performance[self.mask==1]
			#null distribution already masked
			if(self.do_stats):
				self.model1_performance_null_distribution = model1_performance_null_distribution
			print(type(self.ind_feature_performance))
			if(not (type(self.ind_feature_performance)==list)):
				self.ind_feature_performance = ind_feature_performance[:,self.mask==1]
			if(not (type(self.ind_product_measure)==list)):
				self.ind_product_measure = ind_product_measure[:,self.mask==1]
			# self.weights = weights[:,self.mask==1]

			if(self.feature_of_interest!='None'):
				self.model2_performance = model2_performance[self.mask==1]
				#null distribution already masked
				if(self.do_stats):
					self.model2_performance_null_distribution = model2_performance_null_distribution

	def unique_variance(self):
		#load the group level model1 and model2 performances and null performances and mask them
		self.load_mask()

		model_features = self.model.split('_')
		sub_model_features = [feature for feature in model_features if feature!=self.feature_of_interest]
		sub_model_features = sorted(sub_model_features, key=str.casefold) #always alphabetical, ignore the case
		sub_model = '_'.join(sub_model_features)

		label = 'perf_raw'
		model1_filepath = self.out_dir+'/performance/'+self.sid+self.file_label+'_measure-'+label+'.nii.gz'
		model1_img = nibabel.load(model1_filepath)
		affine = model1_img.affine
		model1_data = model1_img.get_fdata()[self.mask==1]

		self.brain_shape = model1_img.get_fdata().shape
		self.affine = affine

		if(self.do_stats):
			label = 'perf_null_distribution'
			model1_null_filepath = self.out_dir+'/performance/'+self.sid+self.file_label+'_measure-'+label+'.nii.gz'
			model1_null_img = nibabel.load(model1_null_filepath)
			affine = model1_null_img.affine
			model1_null_data = model1_null_img.get_fdata()#[:,self.mask==1] null already masked!
 
		file_label = '_encoding_model-'+sub_model+ '_smoothingfwhm-'+str(self.smoothing_fwhm)+'_chunklen-'+str(self.chunklen)+'_mask-'+self.mask_name
		
		label = 'perf_raw'
		model2_filepath = self.out_dir+'/performance/'+self.sid+file_label+'_measure-'+label+'.nii.gz'
		model2_img = nibabel.load(model2_filepath)
		model2_data = model2_img.get_fdata()[self.mask==1]

		if(self.do_stats):
			label = 'perf_null_distribution'
			model2_null_filepath = self.out_dir+'/performance/'+self.sid+file_label+'_measure-'+label+'.nii.gz'
			model2_null_img = nibabel.load(model2_null_filepath)
			model2_null_data = model2_null_img.get_fdata()#[:,self.mask==1] null already masked!

		#calculate and save unique variance 
		self.unique_variance = model1_data - model2_data #model1_data**2 - model2_data**2
		print(self.unique_variance)
		if(self.do_stats):
			self.null_unique_variance = model1_null_data - model2_null_data

		file_label = self.file_label +'_feature-'+self.feature_of_interest
		img = nibabel.Nifti1Image(self.unmask_reshape(self.unique_variance),affine)
		nibabel.save(img, self.out_dir+'/unique_variance/'+self.sid+file_label+'_measure-unique_var_raw.nii.gz') 

	def plot_weights(self,label):
		filepath = self.out_dir+'/weights/'+self.sid+self.file_label

		if(label=='raw'):
			img = nibabel.load(filepath+'_measure-weights_raw.nii.gz')
			img_data = img.get_fdata()
			affine = img.affine
		for (ind,feature_name) in enumerate(self.feature_names):
			# if(feature_name=='DNN_6'): #only plot up to DNN 5
			# 	break
			print(feature_name)
			fsaverage = nilearn.datasets.fetch_surf_fsaverage(mesh='fsaverage')
			hemispheres = ["L","R"]

			weights = img_data[ind]
			weights = nibabel.Nifti1Image(weights,affine)

			filepath = self.figure_dir + "/weights/" +self.sid+self.file_label+'_feature-'+feature_name+'_measure-weights_'+label+'.png'
			threshold=None
			vmax=None
			title=feature_name
			helpers.plot_img_volume(weights,filepath,threshold,vmax)
			helpers.plot_surface(weights,self.figure_dir + "/weights/surface/" +self.sid+self.file_label+'_feature-'+feature_name+'_measure-weights_'+label+'_surf.png',threshold=threshold,vmax=vmax,title=title,symmetric_cbar=False)

	def plot_ind_feature_performance(self,label,threshold=0.000001,vmin=0,vmax=1):
		filepath = self.out_dir+'/ind_feature_performance/'+self.sid+self.file_label

		if(label=='raw'):
			img = nibabel.load(filepath+'_measure-ind_perf_raw.nii.gz')
			img_data = img.get_fdata()
			affine = img.affine
		for (ind,feature_name) in enumerate(self.feature_names):
			print(feature_name)
			fsaverage = nilearn.datasets.fetch_surf_fsaverage(mesh='fsaverage')
			hemispheres = ["L","R"]

			ind_perf = img_data[ind]
			ind_perf = nibabel.Nifti1Image(ind_perf,affine)

			filepath = self.figure_dir + "/ind_feature_performance/" +self.sid+self.file_label+'_feature-'+feature_name+'_measure-ind_perf_'+label+'.png'
			title=feature_name
			helpers.plot_img_volume(ind_perf,filepath,threshold,vmin,vmax,cmap='Greens')
			helpers.plot_surface(ind_perf,self.figure_dir + "/ind_feature_performance/surface/" + self.sid+self.file_label+'_feature-'+feature_name+'_measure-ind_perf_'+label+'_surf.png',threshold=threshold,vmin=vmin,vmax=vmax,title=title,symmetric_cbar=False,cmap='Greens')
	def plot_difference(self,label,threshold=0.000001,vmin=None,vmax=None,group='',cmap='coolwarm'):
		filepath = self.out_dir+'/ind_product_measure/'+self.sid+self.file_label


		if(label=='raw'):
			img = nibabel.load(filepath+'_measure-ind_product_measure_raw.nii.gz')
			img_data = img.get_fdata()
			affine = img.affine

		# for (ind,feature_name) in enumerate(self.feature_names):
		# print(feature_name)
		fsaverage = nilearn.datasets.fetch_surf_fsaverage(mesh='fsaverage')
		hemispheres = ["L","R"]

		ind_perf_1 = img_data[0] #get first feature

		ind_perf_2 = img_data[1] #get second feature

		ind_perf = ind_perf_1-ind_perf_2
		ind_perf = nibabel.Nifti1Image(ind_perf,affine)

		# filepath = self.figure_dir + "/ind_product_measure/" +self.sid+self.file_label+'_feature-'+feature_name+'_measure-ind_product_measure_'+label+'.png'
		# helpers.plot_img_volume(ind_perf,filepath,threshold,vmin,vmax,cmap='inferno')
		title = ''
		helpers.plot_surface(ind_perf,self.figure_dir + "/ind_product_measure/surface/" + self.sid+self.file_label+'_diff-'+self.feature_names[0]+'-'+self.feature_names[1]+'_measure-ind_product_measure_'+label+'_surf',threshold=threshold,vmin=vmin,vmax=vmax,title=title,symmetric_cbar=True,cmap=cmap,colorbar_label='difference (proportion of total $R^2$)')
	
	def plot_model_performance_difference(self,model1,model2,threshold=0.000001,vmin=None,vmax=None,group='',cmap='coolwarm'):

		file_label = '_encoding_model-'+ model1 + '_smoothingfwhm-'+str(self.smoothing_fwhm)+'_chunklen-'+str(self.chunklen)+'_mask-'+self.mask_name
		filepath = self.out_dir+'/performance/'+self.sid+file_label+'_measure-perf_raw.nii.gz'
		img = nibabel.load(filepath)
		ind_perf_1 = img.get_fdata() #first model
		affine = img.affine

		file_label = '_encoding_model-'+model2 + '_smoothingfwhm-'+str(self.smoothing_fwhm)+'_chunklen-'+str(self.chunklen)+'_mask-'+self.mask_name
		filepath = self.out_dir+'/performance/'+self.sid+file_label+'_measure-perf_raw.nii.gz'
		img = nibabel.load(filepath)
		ind_perf_2 = img.get_fdata() #second model
		affine = img.affine

		fsaverage = nilearn.datasets.fetch_surf_fsaverage(mesh='fsaverage')
		hemispheres = ["L","R"]

		ind_perf = ind_perf_1-ind_perf_2
		ind_perf = nibabel.Nifti1Image(ind_perf,affine)

		# filepath = self.figure_dir + "/ind_product_measure/" +self.sid+self.file_label+'_feature-'+feature_name+'_measure-ind_product_measure_'+label+'.png'
		# helpers.plot_img_volume(ind_perf,filepath,threshold,vmin,vmax,cmap='inferno')
		title = ''
		helpers.plot_surface(ind_perf,self.figure_dir + "/performance/surface/" + self.sid+self.file_label+'_diff-'+model1+'-'+model2+'_measure-performance_surf',threshold=threshold,vmin=vmin,vmax=vmax,title=title,symmetric_cbar=True,cmap=cmap,colorbar_label='difference ($R^2$)')
	
	def plot_ind_product_measure(self,label,threshold=0.000001,vmin=None,vmax=None,group=''):
		filepath = self.out_dir+'/ind_product_measure/'+self.sid+self.file_label

		group_dict = {'social':['social','valence','arousal','speaking','turn_taking','mentalization'],
					  'visual_perceptual':['pixel','hue','alexnet','indoor_outdoor','face','num_agents','motion','written_text'],
					  'auditory_perceptual':['pitch','amplitude'],
					  'language':['sbert','word2vec'],
					  'alexnet_layers':['alexnet_layer'+str(layer) for layer in [1,2,3,4,5,6,7]],
					  'sbert_layers':['sbert_layer'+str(layer) for layer in [1,2,3,4,5,6,7,8,9,10,11,12]]}
		group_vmax = {'social':0.04,
					  'visual_perceptual':None,
					  'auditory_perceptual':None,
					  'language':0.07,
					  'alexnet_layers':None,
					  'sbert_layers':None}

		if(label=='raw'):
			img = nibabel.load(filepath+'_measure-ind_product_measure_raw.nii.gz')
			img_data = img.get_fdata()
			affine = img.affine

		if(group==''):
			for (ind,feature_name) in enumerate(self.feature_names):
				print(feature_name)
				fsaverage = nilearn.datasets.fetch_surf_fsaverage(mesh='fsaverage')
				hemispheres = ["L","R"]

				ind_perf = img_data[ind]
				ind_perf = nibabel.Nifti1Image(ind_perf,affine)

				filepath = self.figure_dir + "/ind_product_measure/" +self.sid+self.file_label+'_feature-'+feature_name+'_measure-ind_product_measure_'+label+'.png'
				title=feature_name
				# helpers.plot_img_volume(ind_perf,filepath,threshold,vmin,vmax,cmap='inferno')
				cmap = self.cmaps['yellow_hot']
				# cmap=self.cmaps['SLIPtext']
				helpers.plot_surface(ind_perf,self.figure_dir + "/ind_product_measure/surface/" + self.sid+self.file_label+'_feature-'+feature_name+'_measure-ind_product_measure_'+label+'_surf',threshold=threshold,vmin=vmin,vmax=vmax,title=title,symmetric_cbar=False,cmap=cmap,colorbar_label='proportion of total explained variance $R^2$')
		else:
			overall = []
			for (ind,feature_name) in enumerate(group_dict[group]):
				print(feature_name)
				fsaverage = nilearn.datasets.fetch_surf_fsaverage(mesh='fsaverage')
				hemispheres = ["L","R"]
				feature_ind = self.get_feature_index('sub-06',feature_name)

				ind_perf = img_data[feature_ind]
				if(ind==0):
					overall = ind_perf
				else:
					overall = overall+ind_perf
			ind_perf = nibabel.Nifti1Image(overall,affine)
			filepath = self.figure_dir + "/ind_product_measure/" +self.sid+self.file_label+'_group-'+group+'_measure-ind_product_measure_'+label+'.png'
			title=group+' features'
			vmax = group_vmax[group]
			cmap = self.cmaps['yellow_hot']
			# cmap=self.cmaps['SLIPtext']
			# helpers.plot_img_volume(ind_perf,filepath,threshold,vmin,vmax,cmap='plasma')
			helpers.plot_surface(ind_perf,self.figure_dir + "/ind_product_measure/surface/" + self.sid+self.file_label+'_group-'+group+'_measure-ind_product_measure_'+label+'_surf',threshold=threshold,vmin=vmin,vmax=vmax,title=title,symmetric_cbar=True,cmap=cmap,colorbar_label='proportion of total explained variance $R^2$')

	def plot_unique_variance(self,label,threshold=0.05):
		fsaverage = nilearn.datasets.fetch_surf_fsaverage(mesh='fsaverage')
		hemispheres = ['L','R']

		file_label = '_encoding_model-'+self.model+ '_smoothingfwhm-'+str(self.smoothing_fwhm)+'_chunklen-'+str(self.chunklen)+'_mask-'+self.mask_name+'_feature-'+self.feature_of_interest

		filepath = self.out_dir+'/unique_variance/'+self.sid+file_label
		title = self.sid+', (n=' + str(len(self.subjects))+ '), unique variance explained by ' + self.feature_of_interest 
		if(label=='raw'):
			img = nibabel.load(filepath+'_measure-unique_var_raw.nii.gz')
			cmap='cold_hot'
			threshold=0

		elif(label=='stats'):
			img = nibabel.load(filepath+'_measure-unique_var_p_fdr.nii.gz')
			title = title+', fdr corrected pvalue<'+str(threshold)
			p = img.get_fdata()
			threshold = 1-threshold
			p[self.mask==1] = 1-p[self.mask==1] #turn all significant voxels into high values for plotting
			affine = img.affine
			img = nibabel.Nifti1Image(p, affine)
			cmap = 'Greys'

		plot_label = 'average of all subjects (n='+ str(len(self.subjects))+')' + label + ', smooth ' + str(self.smoothing_fwhm) + ', chunklen ' + str(self.chunklen)
		filepath = self.figure_dir + "/unique_variance/" + self.sid+self.file_label+'_measure-unique_var_'+label+'_feature-'+self.feature_of_interest+'.png'
		# helpers.plot_img_on_surface(img,plot_mesh,bg_map,hemi,plot_label,threshold,html_filepath,vmax)
		helpers.plot_img_volume(img,filepath,threshold=0.000000001,title=title,cmap=cmap,vmin=0,vmax=0.06)
		filename = self.figure_dir + "/unique_variance/surface/" + self.sid+self.file_label+'_measure-unique_var_'+label+'_feature-'+self.feature_of_interest+'_surface'
		helpers.plot_surface(img,filename,title=title,threshold=0.000000001,vmin=0,vmax=0.04,symmetric_cbar=False,cmap='plasma')

		# for hemi in hemispheres:
		# 	if (hemi=='L'):
		# 		plot_mesh = fsaverage.pial_left
		# 		bg_map = fsaverage.sulc_left
		# 		plot_hemi = "left" 
		# 	if(hemi=='R'):
		# 		plot_mesh = fsaverage.pial_right
		# 		bg_map = fsaverage.sulc_right
		# 		plot_hemi = "right"

		# 	plot_label = 'average of all subjects (n='+ str(len(self.subjects))+')' + label + ', smooth ' + str(self.smoothing_fwhm) + ', chunklen ' + str(self.chunklen)
		# 	html_filepath = self.figure_dir + "/unique_variance/" + self.sid+self.file_label+'_measure-unique_var_'+label+'_hemi-'+hemi+'_feature-'+self.feature_of_interest+'.html'
		# 	helpers.plot_img_on_surface(img,plot_mesh,bg_map,hemi,plot_label,threshold,html_filepath)

	def plot_performance(self, label, threshold=None,vmin=None,vmax=None,symmetric_cbar=True):
	    fsaverage = nilearn.datasets.fetch_surf_fsaverage(mesh='fsaverage')
	    hemispheres = ["L","R"]

	    file_label = self.sid+'_encoding_model-'+self.model + '_smoothingfwhm-'+str(self.smoothing_fwhm)+'_chunklen-'+str(self.chunklen)
	    if(self.mask_name!=None):
	        file_label = file_label + '_mask-'+self.mask_name

	    filepath = self.out_dir+'/performance/'+file_label
	    if(label=='raw'):
	        img = nibabel.load(filepath+'_measure-perf_raw.nii.gz')
	        # cmap='inferno'
	        title = self.sid
	    elif(label=='stats'):
	        img = nibabel.load(filepath+'_measure-perf_p_fdr.nii.gz')
	        title = self.sid+', pvalue<'+str(threshold)
	        # print(filepath+'_measure-perf_p_binary.nii.gz')
	        #add a small number to each value so that zeroes are plotted!
	        performance_p = img.get_fdata()
	        threshold = 1-threshold
	        # #mask the brain with significant pvalues
	        # performance_p[performance_p>0.05]=-1
	        performance_p[self.mask==1] = 1-performance_p[self.mask==1] #turn all significant voxels into high values for plotting
	        affine = img.affine
	        img = nibabel.Nifti1Image(performance_p, affine)
	        # cmap = 'Greys'
	    # vmin=None

	    vmax = np.max(img.get_fdata()-0.03)
	    cmap = self.cmaps['yellow_hot']
	    title=''
	    helpers.plot_img_volume(img,self.figure_dir + "/performance/" + file_label+'_measure-perf_'+label+'.png',threshold,vmin=vmin,vmax=vmax,cmap=cmap,title=title,symmetric_cbar=False)
	    helpers.plot_surface(img,self.figure_dir + "/performance/surface/" + file_label+'_measure-perf_'+label+'_surf',threshold=threshold,vmin=vmin,vmax=vmax,cmap=cmap,title=title,symmetric_cbar=symmetric_cbar,colorbar_label='Explained Variance $R^2$')

	def stats(self):

		#performance stats
		def process(voxel_performance,voxel_null_distribution):
				#one-tailed t test for performance
				null_n = voxel_null_distribution.shape[0]
				null_n_over_sample = sum((voxel_null_distribution>voxel_performance).astype(int))
				p = null_n_over_sample/null_n
				self.iterations = null_n
				return p

		self.perf_p_unc = np.array(Parallel(n_jobs=-1)(delayed(process)(voxel_performance,voxel_null_distribution) for (voxel_performance,voxel_null_distribution) in zip(self.model1_performance, self.model1_performance_null_distribution.T)))
		self.perf_fdr_reject,self.perf_p_fdr = fdrcorrection(self.perf_p_unc, alpha=0.05, method='n', is_sorted=False) #method is BY for dependence between variables
		file_label = '_encoding_model-'+self.model + '_smoothingfwhm-'+str(self.smoothing_fwhm)+'_chunklen-'+str(self.chunklen)+'_mask-'+self.mask_name


		img = nibabel.Nifti1Image(self.unmask_reshape(self.perf_p_fdr),self.affine)
		nibabel.save(img, self.out_dir+'/performance/'+self.sid+file_label+'_measure-perf_p_fdr.nii.gz') 

		#ind feature performance stats

		if(len(self.ind_feature_performance_null_distribution)>0):
			ind_features = self.feature_names 
			ind_feature_perf_p_unc = []
			ind_feature_perf_p_fdr = []
			ind_feature_perf_fdr_reject = []

			results = self.ind_feature_performance_null_distribution
			feature_null_distributions = np.reshape(results,(results.shape[1],results.shape[2],results.shape[0])) #reshape to #features,#voxels,#iterations

			for (ind,feature) in enumerate(ind_features):
				voxelwise_null_distribution = feature_null_distributions[ind]
				performance = self.ind_feature_performance[ind]
				def process(voxel_performance,voxel_null_distribution):
					null_n = voxel_null_distribution.shape[0]
					null_n_over_sample = sum((voxel_null_distribution>voxel_performance).astype(int))
					p = null_n_over_sample/null_n
					return p
				perf_p_unc = np.array(Parallel(n_jobs=-1)(delayed(process)(voxel_performance,voxel_null_distribution) for (voxel_performance,voxel_null_distribution) in zip(performance, voxelwise_null_distribution)))
				perf_fdr_reject,perf_p_fdr = fdrcorrection(perf_p_unc, alpha=0.05, method='n', is_sorted=False)

				ind_feature_perf_p_unc.append(perf_p_unc)
				ind_feature_perf_p_fdr.append(perf_p_fdr)
				ind_feature_perf_fdr_reject.append(perf_fdr_reject)

			self.ind_feature_perf_p_unc = np.array(ind_feature_perf_p_unc)
			self.ind_feature_perf_p_fdr = np.array(ind_feature_perf_p_fdr)
			self.ind_feature_perf_fdr_reject = np.array(ind_feature_perf_p_fdr)

			file_label = '_encoding_model-'+self.model + '_smoothingfwhm-'+str(self.smoothing_fwhm)+'_chunklen-'+str(self.chunklen)+'_mask-'+self.mask_name


			img = nibabel.Nifti1Image(self.unmask_reshape(self.ind_feature_perf_p_fdr),self.affine)
			nibabel.save(img, self.out_dir+'/ind_feature_performance/'+self.sid+file_label+'_measure-ind_perf_p_fdr.nii.gz') 


		# do unique variance permutation testing
		if(self.feature_of_interest!='None'):
			def process(voxel_performance,voxel_null_distribution):
				#one-tailed t test for performance
				null_n = voxel_null_distribution.shape[0]
				null_n_over_sample = sum((voxel_null_distribution>voxel_performance).astype(int))
				p = null_n_over_sample/null_n
				self.iterations = null_n
				return p

			self.unique_var_p_unc = np.array(Parallel(n_jobs=-1)(delayed(process)(voxel_performance,voxel_null_distribution) for (voxel_performance,voxel_null_distribution) in zip(self.unique_variance, self.null_unique_variance.T)))
			self.unique_var_fdr_reject,self.unique_var_p_fdr = fdrcorrection(self.unique_var_p_unc, alpha=0.05, method='n', is_sorted=False) #method is BY for dependence between variables
			file_label = '_encoding_model-'+self.model + '_smoothingfwhm-'+str(self.smoothing_fwhm)+'_chunklen-'+str(self.chunklen)+'_mask-'+self.mask_name +'_feature-'+self.feature_of_interest


			img = nibabel.Nifti1Image(self.unmask_reshape(self.unique_var_p_fdr),self.affine)
			nibabel.save(img, self.out_dir+'/unique_variance/'+self.sid+file_label+'_measure-unique_var_p_fdr.nii.gz') 


	def run(self):
		self.explainable_variance_mask = False
		self.do_stats=False
		# self.feature_names = ['social','num_agents','speaking','turn_taking','mentalization','word2vec','sbert','GPT2_1sent','valence','arousal','motion','face','indoor_outdoor','written_text','pixel','hue','pitch','amplitude','music','alexnet']
		# self.feature_names.reverse()
		
		# self.load_features() #inherited from EncodingModel
		# print(self.feature_names)
		# print(len(self.feature_names))
		print(self.subjects)
		self.compile_data()
		print('compiled data')
		# self.unique_variance()
		print('computed unique variance')
		self.compute_preference_maps(restricted=True)#restrict to voxels with positive explained variance
		# print('computed preference maps')
		if(self.do_stats):
			self.stats()
		self.save_results()

		self.plot_ind_product_measure('raw',threshold=0.000001,vmax=None,group='social')
		self.plot_ind_product_measure('raw',threshold=0.000001,vmax=None,group='visual_perceptual')
		self.plot_ind_product_measure('raw',threshold=0.000001,vmax=None,group='auditory_perceptual')
		self.plot_ind_product_measure('raw',threshold=0.000001,vmax=None,group='language')



		if(self.feature_of_interest!='None'):
			self.plot_unique_variance('raw',threshold=0.005)
			if(self.do_stats):
				self.plot_unique_variance('stats',threshold=0.05)
		self.plot_preference_maps(label='')

		self.plot_performance('raw',threshold=0.0000001,vmin = 0, vmax=None)
		self.plot_ind_feature_performance('raw',threshold=0.000001,vmax=None)
		self.plot_ind_product_measure('raw',threshold=0.0000001,vmax=None)
		if(self.do_stats):
			self.plot_performance('stats',threshold=0.00001)
		# self.plot_ind_feature_performance('stats',threshold=0.05)
		# self.plot_weights('raw')



def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--task','-task',type=str,default='sherlock')
	parser.add_argument('--mask','-mask',type=str, default='ISC')
	parser.add_argument('--model','-model',type=str,default='full')
	parser.add_argument('--smoothing-fwhm','-smoothing-fwhm',type=float,default=3.0)
	parser.add_argument('--chunklen','-chunklen',type=int,default=30)
	parser.add_argument('--feature-of-interest','-feature-of-interest',type=str,default='None')
	parser.add_argument('--population','-population',type=str,default='NT')


	parser.add_argument('--dir', '-dir', type=str,
						default='/Users/hsmall2/Documents/GitHub/deep_nat_lat')
	parser.add_argument('--out_dir', '-output', type=str,
						default='/Users/hsmall2/Documents/GitHub/deep_nat_lat/analysis')
	parser.add_argument('--figure_dir', '-figures', type=str,
						default='/Users/hsmall2/Documents/GitHub/deep_nat_lat/figures')
	args = parser.parse_args()
	SecondLevelGroup(args).run()

if __name__ == '__main__':
	main()