import sys
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, './glm')
import glm
import argparse
import os
from pathlib import Path
import glob
import csv
import json

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
from nilearn.glm.second_level import make_second_level_design_matrix
from nilearn.glm.second_level import SecondLevelModel


class SecondLevelGroup(glm.GLM):
	#class attributes go here and must be initialized to something
	# process = 'EncodingModel'

	def __init__(self, args):
		self.process = 'SecondLevelGroup'
		self.dir = args.dir
		self.data_dir = args.dir + '/data'
		self.in_dir = args.out_dir + '/GLM'
		self.out_dir = args.out_dir + "/" + self.process
		self.subjects = []
		self.population = args.population
		self.sid = 'sub-'+self.population
		self.task = args.task
		self.space = args.space
		self.smoothing_fwhm = args.smoothing_fwhm #change?
		self.fMRI_data = []
		self.brain_shape = []
		self.affine = []
		# self.weights = []
		# self.contrast_z_scores = []
		self.figure_dir = args.figure_dir + "/" + self.process
		Path(f'{self.out_dir}/{"glm_weights"}').mkdir(exist_ok=True, parents=True)
		Path(f'{self.out_dir}/{"glm_zscores"}').mkdir(exist_ok=True, parents=True)
		
		Path(f'{self.figure_dir}/{"glm_weights"}').mkdir(exist_ok=True, parents=True)
		Path(f'{self.figure_dir}/{"glm_zscores"}').mkdir(exist_ok=True, parents=True)
		self.file_label ='_smoothingfwhm-'+str(self.smoothing_fwhm)

		#load the parameter file for the specified glm
		params_filepath = 'glm/glm_'+self.task+'.json'
		with open(params_filepath) as json_file:
			glm_params = json.load(json_file)
		json_file.close()

		run_groups = glm_params['run_groups']
		self.run_group = next(iter(run_groups)) #the first run group specified should be using all of the runs
		self.contrasts = glm_params['contrasts']
		self.subjects = helpers.get_subjects(self.population)
		self.cmaps = helpers.get_cmaps()


	def compile_data(self):
		# self.load_mask()
		#weights and contrast z-scores
		print(self.subjects)
		
		label = 'zscore'
		zscore_dict = {}
		for contrast in self.contrasts:
			all_data= []
			subjects_included = []
			for subject in self.subjects[self.task]:
				try:
					# print(subject)
					filepath = self.in_dir + '/'+ subject+ '/' +subject+ "_task-"+ self.task+'_space-'+self.space+ "_run-"+ self.run_group + "_contrast-"+contrast+ "_measure-"+label+".nii.gz"
					nii = nibabel.load(filepath)
					affine = nii.affine
					all_data.append(nii)
					subjects_included.append(subject)
					#
				except Exception as e:
					print(e)
					pass

			# all_data = np.array(all_data)
			## TODO use nilearn to do a group level analysis with stats!!
			#create confounds for the second level group analysis
			design_matrix = make_second_level_design_matrix(
			    subjects_included, ##could include extra info here like age, https://nilearn.github.io/dev/auto_examples/05_glm_second_level/plot_second_level_design_matrix.html#sphx-glr-auto-examples-05-glm-second-level-plot-second-level-design-matrix-py
			)
			second_level_model = SecondLevelModel(smoothing_fwhm=3.0, n_jobs=2)
			second_level_model = second_level_model.fit(
			    all_data, design_matrix=design_matrix
			)
			z_map = second_level_model.compute_contrast(output_type="z_score")
			zscore_dict[contrast] = z_map
			self.brain_shape = z_map.shape
			self.affine = affine

			# img = nibabel.Nifti1Image(all_data_mean, affine)
			nibabel.save(z_map, self.out_dir+'/glm_zscores/'+self.sid+self.file_label+'_contrast-'+contrast+'_measure-'+label+'.nii.gz')

		label = 'weights'
		weights_dict = {}
		for contrast in self.contrasts:
			all_data= []
			for subject in self.subjects[self.task]:
				try:
					# print(subject)
					filepath = self.in_dir + '/'+ subject+ '/' + subject+ "_task-"+ self.task+'_space-'+self.space+ "_run-"+ self.run_group + "_contrast-"+contrast+ "_measure-"+label+".nii.gz"
					nii = nibabel.load(filepath)
					data = nii.get_fdata()
					affine = nii.affine
					all_data.append(data)
				except Exception as e:
					print(e)
					pass

			all_data = np.array(all_data)
			all_data_mean = np.nanmean(all_data,axis=0)
			weights_dict[contrast]=all_data_mean
			self.brain_shape = all_data_mean.shape
			self.affine = affine

			img = nibabel.Nifti1Image(all_data_mean, affine)
			nibabel.save(img, self.out_dir+'/glm_weights/'+self.sid+self.file_label+'_contrast-'+contrast+'_measure-'+label+'.nii.gz')

		self.zscore_dict = zscore_dict
		self.weights_dict = weights_dict

	def plot_weights(self,threshold=0.01,vmin=None,vmax=None):
		
		for contrast in self.contrasts:
			filepath = self.out_dir+'/glm_weights/'+self.sid+self.file_label+'_contrast-'+contrast
			img = nibabel.load(filepath+'_measure-weights.nii.gz')
				
			# if(feature_name=='DNN_6'): #only plot up to DNN 5
			# 	break
			print(contrast)

			plot_filepath = self.figure_dir + "/glm_weights/" +self.sid+self.file_label+'_contrast-'+contrast+'_measure-weights_surf'
			# threshold=None
			vmax=None
			title=contrast
			cmap = self.cmaps['yellow_hot']
			# helpers.plot_img_volume(weights,filepath,threshold,vmax)
			helpers.plot_surface(img,plot_filepath,threshold=threshold,vmax=vmax,title=title,symmetric_cbar=False,cmap=cmap,colorbar_label='weight')

	def plot_zscores(self,threshold=0.01,vmin=None,vmax=None,symmetric_cbar=True):

	    for contrast in self.contrasts:
	    	print(contrast)
	    	filepath = self.out_dir+'/glm_zscores/'+self.sid+self.file_label+'_contrast-'+contrast
	    	img = nibabel.load(filepath+'_measure-zscore.nii.gz')
	    	title=contrast
	    	cmap = self.cmaps['yellow_hot']
	    	plot_filepath = self.figure_dir + "/glm_zscores/" +self.sid+self.file_label+'_contrast-'+contrast+'_measure-zscore_surf'
	    	helpers.plot_surface(img,plot_filepath,threshold=threshold,vmin=vmin,vmax=vmax,cmap=cmap,title=title,symmetric_cbar=symmetric_cbar,colorbar_label='z-score')

	def stats(self):

		### TODO update for GLM!!!

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
		self.do_stats=False
		# self.feature_names = ['social','num_agents','speaking','turn_taking','mentalization','word2vec','sbert','GPT2_1sent','valence','arousal','motion','face','indoor_outdoor','written_text','pixel','hue','pitch','amplitude','music','alexnet']
		# self.feature_names.reverse()
		
		# self.load_features() #inherited from EncodingModel
		# print(self.feature_names)
		# print(len(self.feature_names))
		print(self.subjects)
		self.compile_data()
		print('compiled data')
		if(self.do_stats):
			self.stats()
		# self.save_results()

		self.plot_weights('raw',threshold=0.000001,vmax=None)
		self.plot_zscores('raw',threshold=0.000001,vmax=None)

		if(self.do_stats):
			self.plot_performance('stats',threshold=0.00001)
		# self.plot_ind_feature_performance('stats',threshold=0.05)
		# self.plot_weights('raw')



def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--task','-task',type=str,default='SIpointlights')
	parser.add_argument('--mask','-mask',type=str, default=None)
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