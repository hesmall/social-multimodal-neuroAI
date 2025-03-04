import sys
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, './encoding')
import helpers
import encoding
import argparse
from pathlib import Path
import os
from os.path import exists
from nilearn import surface
import nibabel
import nilearn
import numpy as np

from statsmodels.stats.multitest import fdrcorrection

import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
from tqdm import tqdm
from joblib import Parallel, delayed

from matplotlib.colors import LinearSegmentedColormap
from random import randrange
from statsmodels.stats.multitest import fdrcorrection


class IntersubjectCorrelation(encoding.EncodingModel):
	#class attributes go here and must be initialized to something
	# process = 'EncodingModel'

	def __init__(self, args):
		self.process = 'IntersubjectCorrelation'
		self.dir = args.dir
		self.data_dir = args.dir + '/data'
		self.out_dir = args.out_dir + "/" + self.process
		self.sid = 'sub-'+args.population
		self.task = args.task
		self.mask_name='None'
		self.mask_id = args.mask
		self.ISC_type = args.ISC_type
		self.mask_affine = None
		self.smoothing_fwhm = args.smoothing_fwhm #change?
		self.chunklen = args.chunklen
		self.population = args.population
		self.brain_shape = (97,115,97)
		self.mask = np.ones(self.brain_shape) #default is full brain
		self.affine = []
		self.figure_dir = args.figure_dir + "/" + self.process
		# self.feature_of_interest = args.feature_of_interest#['social','mentalization']
		Path(f'{self.out_dir}/{"intersubject_correlation"}').mkdir(exist_ok=True, parents=True)
		Path(f'{self.figure_dir}/{"intersubject_correlation"}').mkdir(exist_ok=True, parents=True)
		self.time_series = []
		self.ISC = []
		# self.base_model_predicted_time_series = []
		# self.base_model_predicted_time_series_null = []
		# self.modified_model_predicted_time_series = []
		# self.modified_model_predicted_time_series_null = []
		# self.selected_voxels_mask = []
		self.cmaps = {

        'yellow_hot': LinearSegmentedColormap.from_list('yellow_hot', (
                    # Edit this gradient at https://eltos.github.io/gradient/#4C71FF-0025B3-000000-C73A03-FCEB4A
                    (0.000, (0.298, 0.443, 1.000)),
                    (0.250, (0.000, 0.145, 0.702)),
                    (0.500, (0.000, 0.000, 0.000)),
                    (0.750, (0.780, 0.227, 0.012)),
                    (1.000, (0.988, 0.922, 0.290))))
        }
	def load_subjects(self):
		if(self.population=='NT'):
			self.subjects = ['sub-05','sub-06','sub-07','sub-10','sub-14','sub-15','sub-16','sub-21','sub-23','sub-25','sub-26','sub-28','sub-32','sub-33','sub-35','sub-36']
			#leave 13 out because it is not time-locked with the rest, 
			# 9 and 19 have too much motion
			self.bad_subjects = ['sub-11','sub-12']

		if(self.population=='ASD'):
			self.subjects = ['sub-04','sub-17','sub-18','sub-20','sub-22','sub-24','sub-27','sub-34'] #sub-04 <- check on this one
			self.bad_subjects = []
		
		mask_computed = False
		subject_data = []
		for i,subject1 in enumerate(self.subjects):
				print(subject1)
			# a_tqdm.set_description('loading '+subject1)
			# try:
				##load each subject's time series into a dict
				file_label = subject1 +'_smoothingfwhm-'+str(self.smoothing_fwhm)
				filepath = self.dir+'/analysis/TimeSeries/'+file_label+'_mask-'+self.mask_id+'_measure-actual_time_series.nii.gz'
				subject1_timeseries_nii = nibabel.load(filepath,mmap=True)#mmap to save RAM
				self.affine = subject1_timeseries_nii.affine
				
				if(self.mask_name=='None'):
					if(~mask_computed):
						img = nibabel.Nifti1Image(np.transpose(subject1_timeseries_nii.get_fdata(),(1,2,3,0)),self.affine)
						mask = nilearn.masking.compute_brain_mask(img, mask_type='whole-brain')
						self.mask = mask.get_fdata()
						self.affine = mask.affine
						mask_computed=True
						# print('mask',self.mask.shape)
				subject1_timeseries = subject1_timeseries_nii.get_fdata()[:,self.mask==1]
				subject1_timeseries = subject1_timeseries.reshape(subject1_timeseries.shape[0],-1).T
				subject_data.append(subject1_timeseries.astype('float32'))
				del subject1_timeseries
		self.subject_data = np.array(subject_data)
		print(self.subject_data.shape)

	def leave_one_out_correlation(self,circle_shift=False):
		# for each subject, average all other subjects time series and then correlate with the subject
		correlations = []
		explained_variances = []
		# print(subject_data)
		print('computing leave one out intersubject correlation')
		# def process(i, subject1,subject_data):
		for i,subject1 in enumerate(tqdm(self.subjects)):
			indices = [i for (i,subject) in enumerate(self.subjects) if subject!=subject1]
			
			if(circle_shift):
				#get random amount to shift by
				shift_values = np.random.randint(0, self.subject_data.shape[2], size=self.subject_data.shape[0])
				print(shift_values)
				#shift each subject by a different shift value, replacing the original data to be memory efficient
				for i in range(self.subject_data.shape[0]):
				    self.subject_data[i] = np.roll(self.subject_data[i], shift_values[i])
			
			mat1 = self.subject_data[i]
			mat2 = np.nanmean(self.subject_data[indices],axis=0)

			## standardize the time series
			mat1_standardized = mat1#already z-scored(mat1 - np.mean(mat1, axis=1, keepdims=True)) / np.std(mat1, axis=1, ddof=1, keepdims=True)
			mat2_standardized = (mat2 - np.mean(mat2, axis=1, keepdims=True)) / np.std(mat2, axis=1, keepdims=True)
			# Calculate correlation using matrix multiplication
			correlation_matrix = np.einsum('ij,ij->i', mat1_standardized, mat2_standardized) / (mat1.shape[1] - 1)
			
			correlations.append(correlation_matrix)
			explained_variances.append(correlation_matrix**2 * np.sign(correlation_matrix)) #r squared, maintain sign
			del mat1,mat2,mat1_standardized,mat2_standardized
			
		print('correlations: ',len(correlations))
		# print(np.array(explained_variances).shape)
		self.ISC_corr_subjects = np.array(correlations.copy())
		
		#Fisher z transform before averaging!!!
		explained_variances = np.arctanh(explained_variances)
		correlations = np.arctanh(correlations)

		#average
		explained_variances = np.nanmean(np.array(explained_variances),axis=0)
		correlations = np.nanmean(np.array(correlations),axis=0)

		#inverse fisher z transform!
		explained_variances = np.tanh(explained_variances)
		correlations = np.tanh(correlations)

		self.ISC_explained_variance_mean = explained_variances
		self.ISC_corr_mean = correlations

		# self.ISC_corr_std = np.nanstd(np.array(correlations))

	def pairwise_correlation(self,circle_shift=False):
		mask_computed=False

		#for each pair of subjects for each voxel, get the time series correlation 
		subject_data = {}
		done = []
		correlations = []
		explained_variances = []
		a_tqdm = tqdm(self.subjects)

		results = np.array((len(self.subjects),len(self.subjects)))
		print('computing pairwise correlation')
		for i,subject1 in enumerate(tqdm(self.subjects)):
			for j,subject2 in enumerate(self.subjects):
				try:
					if(subject1==subject2):
						pass 
					elif(subject1+'_'+subject2 in done):
						pass 
					elif(subject2+'_'+subject1 in done):
						pass 
					else:
						done.append(subject1+'_'+subject2)

						if(circle_shift):
							#get random amount to shift by
							shift_values = np.random.randint(0, self.subject_data.shape[2], size=2)
							print(shift_values)
							#shift each subject by a different shift value, replacing the original data to be memory efficient
							for x in [i,j]:
							    self.subject_data[i] = np.roll(self.subject_data[x], shift_values[x])

						mat1_standardized = self.subject_data[subject1] #already z-scored
						mat2_standardized = self.subject_data[subject2] #already z-scored

						# Calculate correlation using matrix multiplication
						correlation_matrix = np.einsum('ij,ij->i', mat1_standardized, mat2_standardized) / (mat1.shape[1] - 1)
						correlations.append(correlation_matrix)
						explained_variances.append(correlation_matrix**2 * np.sign(correlation_matrix)) #r squared, maintain sign
				except Exception as e:
					print(e)
					pass

		print('correlations: ',len(correlations))

		self.ISC_corr_subjects = correlations.copy()

		#Fisher z transform before averaging!!!
		explained_variances = np.arctanh(explained_variances)
		correlations = np.arctanh(correlations)

		#average
		explained_variances = np.nanmean(np.array(explained_variances),axis=0)
		correlations = np.nanmean(np.array(correlations),axis=0)

		#inverse fisher z transform!
		explained_variances = np.tanh(explained_variances,axis=0)
		correlations = np.tanh(correlations,axis=0)

		self.ISC_explained_variance_mean = explained_variances
		self.ISC_corr_mean = correlations
		


	def permutation_brainiak(self,iterations):
		from brainiak.isc import permutation_isc
		print('starting permutation testing')
		import datetime
		now = datetime.datetime.now()
		print(now)
		group_assignment = [1 for subjects in self.subjects]
		print(self.ISC_corr_subjects.shape)
		observed,p,distribution = permutation_isc(self.ISC_corr_subjects, group_assignment=None, pairwise=self.ISC_type=='pairwise',  # noqa: C901
                    summary_statistic='median', n_permutations=iterations,
                    side='right', random_state=None)
		print('ended permutation testing')
		now = datetime.datetime.now()
		print(now)
		self.ISC_corr_median = np.squeeze(observed)
		print('observed',observed.shape)
		print(observed)
		self.p_unc = p
		print(self.p_unc.shape)
		print(self.p_unc)
		print('distribution',distribution.shape)
		print(distribution)
		#perform FDR correction
		self.fdr_reject,self.p_fdr = fdrcorrection(self.p_unc, alpha=0.05, method='n', is_sorted=False)

	def generate_null_distribution(self,iterations=5000):
		
		null_distribution_corr = []
		null_distribution_exp_var = []
		
		## do ISC over and over, with circle shifting the time series
		for iteration in np.arange(iterations):
			self.leave_one_out_correlation(circle_shift=True)
			null_distribution_corr.append(self.ISC_corr)
			null_distribution_exp_var.append(self.ISC)

		self.null_distribution_corr = np.array(null_distribution_corr)
		self.null_distribution_exp_var = np.array(null_distribution_exp_var)

	def permutation_statistics(self):
		## load the non-shuffled ISC
		file_label = self.sid +'_smoothingfwhm-'+str(self.smoothing_fwhm)
		if(self.mask_id!=None):
		    file_label = file_label + '_mask-'+self.mask_id
		filepath = self.out_dir+'/intersubject_correlation/'+file_label+'_measure-intersubject_correlation.nii.gz'
		ISC = nibabel.load(filepath).get_fdata()


		## get p-values and FDR correct them
		def process(voxel_ISC,voxel_null_distribution):
		    #one-tailed t test for performance
		    null_n = voxel_null_distribution.shape[0]
		    null_n_over_sample = sum((voxel_null_distribution>voxel_ISC).astype(int))
		    p = null_n_over_sample/null_n
		    return p

		self.p_unc = np.array(Parallel(n_jobs=-1)(delayed(process)(voxel_ISC,voxel_null_distribution) for (voxel_ISC,voxel_null_distribution) in zip(ISC, self.null_distribution_corr.T)))

		#perform FDR correction
		self.fdr_reject,self.p_fdr = fdrcorrection(self.p_unc, alpha=0.05, method='n', is_sorted=False)


	def plot_intersubject_correlation(self,threshold=0.001,vmin=None,vmax=None):
		file_label = self.sid +'_smoothingfwhm-'+str(self.smoothing_fwhm)+'_type-'+self.ISC_type
		if(self.mask_id!=None):
		    file_label = file_label + '_mask-'+self.mask_id
		
		labels = ['correlation']#,'explained_variance','correlation_median']
		for label in labels:
			filepath = self.out_dir+'/intersubject_correlation/'+file_label+'_measure-intersubject_'+label+'.nii.gz'
			img = nibabel.load(filepath)
			cmap = 'inferno'#self.cmaps['yellow_hot']
			title = self.population + ' '+label
			helpers.plot_surface(img,self.figure_dir + "/intersubject_correlation/" + file_label+'_measure-intersubject_'+label+'_surf',threshold=threshold,vmin=vmin,vmax=vmax,cmap=cmap,title=title,symmetric_cbar=False,colorbar_label='mean ISC')

			data = img.get_fdata()**2
			img = nibabel.Nifti1Image(data,img.affine)
			helpers.plot_surface(img,self.figure_dir + "/intersubject_correlation/" + file_label+'_measure-intersubject_'+label+'_squared_surf',threshold=threshold,vmin=vmin,vmax=vmax,cmap=cmap,title=title,symmetric_cbar=False,colorbar_label='mean ISC ($r^2$)')

	def plot_intersubject_correlation_stats(self,threshold=0.001,vmin=None,vmax=None):
		file_label = self.sid +'_smoothingfwhm-'+str(self.smoothing_fwhm)+'_type-'+self.ISC_type
		if(self.mask_id!=None):
		    file_label = file_label + '_mask-'+self.mask_id
		
		labels = ['p_fdr']
		for label in labels:
			filepath = self.out_dir+'/intersubject_correlation/'+file_label+'_measure-intersubject_correlation_'+label+'.nii.gz'
			img = nibabel.load(filepath)
			p = img.get_fdata()
			threshold = 1-threshold
			# #mask the brain with significant pvalues
			p[self.mask==1] = 1-p[self.mask==1] #turn all significant voxels into high values for plotting
			affine = img.affine
			img = nibabel.Nifti1Image(p, affine)
			cmap = 'Greys'
			title = self.population + ' '+label
			helpers.plot_surface(img,self.figure_dir + "/intersubject_correlation/" + file_label+'_measure-intersubject_correlation_'+label+'_surf',threshold=0,vmin=vmin,vmax=vmax,cmap=cmap,title=title,symmetric_cbar=True)

	def save_results(self):
		#unmask and save results
		file_label = self.sid +'_smoothingfwhm-'+str(self.smoothing_fwhm)+'_type-'+self.ISC_type
		if(self.mask_id!=None):
		    file_label = file_label + '_mask-'+self.mask_id
		# if(len(self.ISC)>0):
		ISC = self.unmask_reshape(self.ISC_explained_variance_mean)
		img = nibabel.Nifti1Image(ISC,self.affine)
		nibabel.save(img, self.out_dir+'/intersubject_correlation/'+file_label+'_measure-intersubject_explained_variance.nii.gz') 

		ISC_corr = self.unmask_reshape(self.ISC_corr_mean)
		img = nibabel.Nifti1Image(ISC_corr,self.affine)
		nibabel.save(img, self.out_dir+'/intersubject_correlation/'+file_label+'_measure-intersubject_correlation.nii.gz') 

	def save_stats_results(self):
		#unmask and save results
		file_label = self.sid +'_smoothingfwhm-'+str(self.smoothing_fwhm)+'_type-'+self.ISC_type
		if(self.mask_id!=None):
		    file_label = file_label + '_mask-'+self.mask_name
		# if(len(self.ISC)>0):
		ISC = self.unmask_reshape(self.ISC_corr_median)
		img = nibabel.Nifti1Image(ISC,self.affine)
		nibabel.save(img, self.out_dir+'/intersubject_correlation/'+file_label+'_measure-intersubject_correlation_median.nii.gz') 

		ISC = self.unmask_reshape(self.p_unc)
		img = nibabel.Nifti1Image(ISC,self.affine)
		nibabel.save(img, self.out_dir+'/intersubject_correlation/'+file_label+'_measure-intersubject_correlation_p_unc.nii.gz') 

		ISC = self.unmask_reshape(self.p_fdr)
		img = nibabel.Nifti1Image(ISC,self.affine)
		nibabel.save(img, self.out_dir+'/intersubject_correlation/'+file_label+'_measure-intersubject_correlation_p_fdr.nii.gz') 

	def run(self):
		
		# self.load_subjects() 
		# self.load_mask()
		# self.explainable_variance_mask = False
		# if(self.ISC_type=='pairwise'):
		# 	self.pairwise_correlation()
		# elif(self.ISC_type=='leave_one_out'):
		# 	self.leave_one_out_correlation()
		# self.permutation_brainiak(iterations=10000)

		# self.save_results()
		# self.save_stats_results()

		self.plot_intersubject_correlation(threshold=0.01,vmin=0,vmax=0.15)
		self.plot_intersubject_correlation_stats(threshold=0.0001,vmin=None,vmax=None) #requires self.mask to be computed
		# elif(self.pipeline=='permutation_testing'):
		# 	self.generate_null_distribution(iterations=3)
		# 	self.permutation_statistics()

		# 	self.save_results()
			


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--task','-task',type=str,default='sherlock')
	parser.add_argument('--mask','-mask',type=str, default='None')
	parser.add_argument('--space','-space',type=str,default='MNI152NLin2009cAsym')
	parser.add_argument('--smoothing-fwhm','-smoothing-fwhm',type=float,default=3.0)
	parser.add_argument('--chunklen','-chunklen',type=int,default=20)
	parser.add_argument('--population','-population',type=str,default='NT')
	parser.add_argument('--ISC-type','-ISC-type',type=str,default='leave_one_out')

	parser.add_argument('--dir', '-dir', type=str,
						default='/Users/hsmall2/Documents/GitHub/deep_nat_lat')
	parser.add_argument('--out_dir', '-output', type=str,
						default='/Users/hsmall2/Documents/GitHub/deep_nat_lat/analysis')
	parser.add_argument('--figure_dir', '-figures', type=str,
						default='/Users/hsmall2/Documents/GitHub/deep_nat_lat/figures')
	args = parser.parse_args()
	IntersubjectCorrelation(args).run()

if __name__ == '__main__':
	main()