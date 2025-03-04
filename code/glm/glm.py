import sys
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, '../')
import helpers
# import surface_helpers

import argparse
import os
from pathlib import Path
# import glob
import nibabel as nib
import nilearn
import nilearn.datasets
import pandas as pd
import numpy as np

import os
from os.path import exists
from nilearn import surface
import nibabel
from nibabel import processing

# from bids import BIDSLayout
import json

from nilearn.glm.first_level import first_level_from_bids
from nilearn.glm.first_level import make_first_level_design_matrix

from nilearn import plotting
from nilearn.plotting import plot_design_matrix

from nilearn.glm.first_level import run_glm, mean_scaling
from nilearn.glm.contrasts import compute_contrast
from nilearn.reporting import make_glm_report

from nilearn.masking import compute_brain_mask
from nilearn.masking import apply_mask
from nilearn.masking import unmask

from statsmodels.stats.multitest import fdrcorrection

import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns

class GLM:
    #class attributes go here
    
    def __init__(self, args):
        self.process = 'GLM'
        self.dir = args.dir
        self.data_dir = args.data_dir
        self.s_num = str(int(args.s_num)).zfill(2)
        self.sid = 'sub-'+self.s_num
        self.task = args.task
        self.space = args.space
        self.brain_shape = (97,115,97)
        self.smoothing_fwhm = args.smoothing_fwhm
        self.slice_time_ref = 0.5
        self.mask_name = args.mask #NOTE: GLM runs in whole brain but results are masked by this parameter
        self.events = []
        self.imgs = []
        self.confounds = []
        self.metadata = []
        self.design_matrix = []
        self.subject_label = self.sid+ "_task-"+ self.task +'_space-'+self.space
        self.out_dir = args.out_dir + '/'+self.process + '/' + self.sid + '/' 
        self.figure_dir = args.figure_dir + '/'+self.process + '/' + self.sid + '/'
        Path(self.out_dir).mkdir(exist_ok=True, parents=True)
        Path(self.figure_dir).mkdir(exist_ok=True, parents=True)

        #load the parameter file for the specified glm
        params_filepath = 'glm_'+self.task+'.json'
        with open(params_filepath) as json_file:
            glm_params = json.load(json_file)
        json_file.close()

        self.run_groups = glm_params['run_groups']
        self.contrasts = glm_params['contrasts']

        self.n_runs = glm_params['n_runs']

    #what runs when you print(class_name)
    def __str__(self):
        pass

    def load_subject_data(self):
        print('loading subject data...')
        #assumes BIDS layout

        search_str_base = self.sid + '*' + 'task-'+self.task+'*'

        base_file = self.sid + '_task-'+self.task

        #explicit paths is the fastest on the cluster...
        event_files = []
        img_files = []
        confound_files = []
        metadata_files = []
        for run in range(1,self.n_runs+1):
            event_files.append(self.data_dir+'/'+self.sid+'/func/'+base_file+'_run-0'+str(run)+'_events.tsv')
            img_files.append(self.data_dir+'/derivatives/'+self.sid+'/func/'+base_file+'_run-'+str(run)+'_space-'+self.space+'_res-2_desc-preproc_bold.nii.gz')
            confound_files.append(self.data_dir+'/derivatives/'+self.sid+'/func/'+base_file+'_run-'+str(run)+'_desc-confounds_timeseries.tsv')
            metadata_files.append(self.data_dir+'/derivatives/'+self.sid+'/func/'+base_file+'_run-'+str(run)+'_space-'+self.space+'_res-2_desc-preproc_bold.json')
        
        events = []
        imgs = []
        confounds = []
        metadata = []
        for (event,img,confound,metadatum) in zip(event_files,img_files,confound_files,metadata_files):
            events.append(pd.read_csv(event,sep='\t'))
            imgs.append(str(img))
            confounds.append(pd.read_csv(confound,sep='\t'))
            with open(metadatum) as json_file:
                metadata.append(json.load(json_file))

        self.events = events
        self.imgs = imgs
        self.confounds = confounds
        self.metadata = metadata

        print(events)
        print(imgs)
        print(confounds)
        print(metadata)

    def load_subject_data_old(self):
        print('loading subject data...')
        #assumes BIDS layout

        layout = BIDSLayout(self.data_dir, derivatives=True)
        subject = self.s_num
        task_label = self.task
        space_label = self.space

        events = []
        imgs = []
        confounds = []
        metadata = []
        f_events = layout.get(subject=subject, task=task_label, extension='tsv', suffix='events')
        f_imgs = layout.get(subject=subject, task=task_label, space=space_label, extension='nii.gz', suffix='bold', scope='derivatives', return_type='filename')
        f_confounds = layout.get(subject=subject, task=task_label, extension='tsv', scope='derivatives')
        f_metadata = layout.get(subject=subject, task=task_label, space=space_label, extension='json', suffix='bold', scope='derivatives', return_type='filename')
        args = zip(f_events, f_imgs, f_confounds, f_metadata)
        for (event, img, confound, metadatum) in args:
            events.append(event.get_df())
            imgs.append(img)
            confounds.append(confound.get_df())
            with open(metadatum) as json_file:
                metadata.append(json.load(json_file))

        self.events = events
        self.imgs = imgs
        self.confounds = confounds
        self.metadata = metadata

        print(events)
        print(imgs)
        print(confounds)
        print(metadata)

    def get_contrast_values(self, contrast_id):
        # get the basic contrasts of the design first
        contrast = self.contrasts[contrast_id]
        contrast_matrix = np.eye(self.design_matrix.shape[1])
        basic_contrasts = dict(
            [(column, contrast_matrix[i]) for i, column in enumerate(self.design_matrix.columns)]
        )

        # get the first contrast, assumes that the contrasts are subtraction at the lowest level, connected by addition at higher level
        split = contrast.split("&")
        full_split = [x.split("-") for x in split]
        # initialize with first contrast
        contrast_values = basic_contrasts[full_split[0][0]]
        if len(full_split[0]) > 1:
            # on lowest level, subtraction
            contrast_values = contrast_values - basic_contrasts[full_split[0][1]]
        if len(full_split) > 1:
            for x in full_split[1:]:
                curr_contrast_values = basic_contrasts[x[0]]
                if len(x) > 1:
                    # on lowest level, subtraction
                    curr_contrast_values = curr_contrast_values - basic_contrasts[x[1]]
                # on higher level, addition
                contrast_values = contrast_values + curr_contrast_values

        return contrast_values

    def load_mask(self,mask_name):
        #get relevant masks
        if(mask_name =='STS'):
            atlas = nilearn.datasets.fetch_atlas_destrieux_2009(lateralized=True)
            atlas = nibabel.load(atlas['maps'])
            self.affine = atlas.affine
            self.mask_affine = atlas.affine
            atlas = nilearn.image.resample_img(atlas, target_affine=self.affine, target_shape=self.brain_shape,interpolation='nearest')
            atlas = atlas.get_fdata()
            
            mask = atlas.copy()
            mask[:] = 0
            mask[(atlas==74)|(atlas==149)] = 1
            #74 is left STS, 149 is right STS
            self.mask = mask

        if(mask_name =='ISC'):
            mask = nibabel.load(self.dir + '/analysis/parcels/isc_bin.nii')
            self.affine = mask.affine
            mask = nilearn.image.resample_img(mask, target_affine=self.affine, target_shape=self.brain_shape,interpolation='nearest')
            mask = mask.get_fdata()
            self.mask = mask
            self.mask_affine = self.affine

        if(self.mask_name =='STS_and_MT'):
            atlas = nilearn.datasets.fetch_atlas_destrieux_2009(lateralized=True)
            atlas = nibabel.load(atlas['maps'])
            self.affine = atlas.affine
            self.mask_affine = self.affine
            atlas = nilearn.image.resample_img(atlas, target_affine=self.affine, target_shape=self.brain_shape,interpolation='nearest')
            atlas = atlas.get_fdata()
            
            mask = atlas.copy()
            mask[:] = 0
            mask[(atlas==74)|(atlas==149)] = 1 #boolean for the conjunction below
            
            STS_mask = mask
            MT_mask = nibabel.load(self.dir+'/analysis/parcels/MT.nii')
            MT_mask = nilearn.image.resample_img(MT_mask, target_affine=self.affine, target_shape=self.brain_shape,interpolation='nearest')

            MT_mask=MT_mask.get_fdata()

            mask = ((STS_mask==1) | (MT_mask==1) )*1

            #74 is left STS, 149 is right STS
            self.mask = mask
        
        if(self.mask_name =='lateral'):
            mask = nibabel.load(self.dir + '/analysis/parcels/lateral_STS_mask.nii')
            self.mask_affine = mask.affine
            mask = nilearn.image.resample_img(mask, target_affine=self.affine, target_shape=self.brain_shape,interpolation='nearest')
            self.mask = mask.get_fdata()
            

    def run_glm(self):
        fsaverage = nilearn.datasets.fetch_surf_fsaverage(mesh='fsaverage')
        hemispheres = ['L','R']

        data = pd.DataFrame(columns=[
            "subject",
            "hemisphere",
            "run",
            "contrast",
            "z_scores",
            "effect_sizes",
            "mask"
            ], dtype='object')


        for group_id in self.run_groups.keys():
            textures = []
            design_matrices = []

            ##### MASKING imgs -- MNI template, whole_brain
            
            # nilearn.masking.apply_mask(self.imgs,self.mask)
            for run in self.run_groups[group_id]:  # runs
                run = run - 1
                img = self.imgs[run]
                event = self.events[run]
                confound = self.confounds[run]
                metadata = self.metadata[run]
                
                img = nibabel.load(img) #load the numpy array
                print(img.shape)

                #resample to the specified mask
                if(self.mask_name!='None'):
                    self.load_mask(self.mask_name)
                    print('...resampling to '+self.mask_name+' mask...')
                    img = nilearn.image.resample_img(img, target_affine=self.affine, target_shape=self.brain_shape)
                    print(mg.shape)
                #masking and smoothing
                print('...whole brain masking and smoothing with gaussian fwhm='+str(self.smoothing_fwhm)+'...')
                whole_brain_mask = nilearn.masking.compute_brain_mask(img,mask_type='whole-brain')
                masked_smoothed_data = nilearn.masking.apply_mask([img],whole_brain_mask,smoothing_fwhm=self.smoothing_fwhm)
                print(masked_smoothed_data.shape)
                print('...mean scaling....')
                fMRI_data = np.array(mean_scaling(masked_smoothed_data)[0])
                # fMRI_data = np.array(masked_smoothed_data)
                #mean scaling
                print(fMRI_data.shape)
                
                textures.append(fMRI_data)
                ### FILTER CONFOUNDS from fmriprep preprocessing ###############################################

                # 6 rigid-body transformations, FD, and aCompCor components
                confounds_to_use = ['rot_x', 
                                    'rot_y',
                                    'rot_z',
                                    'trans_x',
                                    'trans_y',
                                    'trans_z',
                                    'framewise_displacement',
                                    'a_comp_cor_00',
                                    'a_comp_cor_01',
                                    'a_comp_cor_02',
                                    'a_comp_cor_03',
                                    'a_comp_cor_04',
                                    'cosine00'
                                    ]
                confound = confound[confounds_to_use].fillna(0)
                confounds_matrix = confound.values
                confounds_names = confound.columns.tolist()

                ### CREATE DESIGN MATRIX with events and confound noise regressors #############################
                # n_scans = texture.shape[1]
                n_scans = fMRI_data.shape[0]
                # need to shift fram times because of slice timing correction in fmriprep preproc
                t_r = metadata["RepetitionTime"]
                frame_times = (t_r * ( np.arange(n_scans) + 0.7))
                
                #make sure trial_type is not read as an integer
                event = event.astype({'trial_type': '<U11'})
                
                design_matrix = make_first_level_design_matrix(
                    frame_times,
                    events=event,
                    hrf_model="glover + derivative",
                    add_regs=confounds_matrix,
                    add_reg_names=confounds_names
                )
                design_matrices.append(design_matrix)

            full_design_matrix = pd.concat(design_matrices)
            full_design_matrix.fillna(0, inplace=True)

             # plot design matrix for debugging later
            fig, ax1 = plt.subplots(figsize=(10, 6), nrows=1, ncols=1)
            plot_design_matrix(full_design_matrix, ax=ax1)
            plt.savefig(self.out_dir + self.sid + "_task-"+ self.task+ "_run-"+ group_id
                            + "_glm_design_matrix.svg")
            plt.close()

            self.design_matrix = full_design_matrix

            ### FIT THE GLM on the data  ###########################################
            labels, estimates = run_glm(np.concatenate(textures), full_design_matrix.values)

            ### COMPUTE CONTRASTS #########################################################################
            for contrast_id in self.contrasts.keys():
                  
                contrast_values = self.get_contrast_values(contrast_id)
                print(contrast_id)
                print(contrast_values)
                contrast = compute_contrast(
                    labels, estimates, contrast_values, contrast_type="t"
                )

                if(self.mask_name!='None'):
                    self.load_mask(self.mask_name)
                    mask_img = nibabel.Nifti1Image(self.mask, self.mask_affine)
                    resampled_mask = nilearn.image.resample_img(mask_img, target_affine=self.affine, target_shape=self.brain_shape,interpolation='nearest')
                    mask = nilearn.masking.apply_mask([resampled_mask],whole_brain_mask)

                    #only use the mask inside the whole brain mask for this subject
                    flattened_mask = mask.astype(int).flatten()
                    print(flattened_mask)

                    z_scores = np.zeros(contrast.z_score().shape)
                    weights = np.zeros(contrast.effect.flatten().shape)
                    p_values = np.zeros(contrast.p_value().shape)

                    print(z_scores.shape)
                    print(flattened_mask.shape)

                    z_scores[flattened_mask==1] = contrast.z_score()[flattened_mask==1]
                    weights[flattened_mask==1] = contrast.effect.flatten()[flattened_mask==1]

                    #fdr correction within the mask
                    unc_p_values_masked = contrast.p_value()[flattened_mask==1]
                    reject, p_values_masked = fdrcorrection(unc_p_values_masked,alpha=0.05, method='n', is_sorted=False)
                    p_values[flattened_mask==1] = p_values_masked


                else:

                    z_scores = contrast.z_score()
                    weights = contrast.effect.flatten()
                    unc_p_values = contrast.p_value()
                    #fdr correction
                    reject, p_values = fdrcorrection(unc_p_values, alpha=0.05, method='n', is_sorted=False)

                #put back into 3D space!!
                z_scores_img = nilearn.masking.unmask(z_scores,whole_brain_mask)
                weights_img = nilearn.masking.unmask(weights,whole_brain_mask)
                p_values_img = nilearn.masking.unmask(p_values,whole_brain_mask)

                print('SHAPE')
                print(z_scores_img)
                
                # z_scores = np.reshape(z_scores,self.brain_shape)
                # weights = np.reshape(weights,self.brain_shape)
                # p_values = np.reshape(p_values,self.brain_shape)

                ##### SAVE DATA IMG ########
                z_scores_name = self.out_dir + '/'+ self.sid+ "_task-"+ self.task+'_space-'+self.space+ "_run-"+ group_id + "_contrast-"+contrast_id+ "_measure-zscore.nii.gz"
                # z_scores_img = nibabel.Nifti1Image(z_scores, self.affine)
                nibabel.save(z_scores_img, z_scores_name)

                weights_name = self.out_dir + '/'+ self.sid+ "_task-"+ self.task+'_space-'+self.space+ "_run-"+ group_id + "_contrast-"+contrast_id+ "_measure-weights.nii.gz"
                # weights_img = nibabel.Nifti1Image(weights, self.affine)
                nibabel.save(weights_img, weights_name)

                p_values_name = self.out_dir + '/'+ self.sid+ "_task-"+ self.task+'_space-'+self.space+ "_run-"+ group_id + "_contrast-"+contrast_id+ "_measure-pvalue.nii.gz"
                # p_values_img = nibabel.Nifti1Image(p_values, self.affine)
                nibabel.save(p_values_img, p_values_name)

                # # threshold, num_voxels = helpers.get_top_percent(weights, 10) #getting top 10% of voxels
                # brain_html_filepath = self.figure_dir + self.subject_label+"_run-"+ group_id+ "_weight-"+contrast_id+ "_smoothfwhm-"+str(self.smoothing_fwhm) + '.png'
                # helpers.plot_img_volume(weights_img, brain_html_filepath,symmetric_cbar=True,threshold=0)#,threshold=0)

                #plot top 10% of voxels (whole brain)
                # threshold, num_voxels = helpers.get_top_percent(z_scores, 10) #getting top 10% of voxels
                brain_html_filepath = self.figure_dir + self.subject_label+"_run-"+ group_id+ "_contrast-"+contrast_id+ "_smoothfwhm-"+str(self.smoothing_fwhm) + '.png'
                brain_html_title = self.sid + ', ' + contrast_id + ', runs:' + group_id
                print(brain_html_filepath)
                # 
                helpers.plot_img_volume(z_scores_img, brain_html_filepath,symmetric_cbar=True,threshold=0)#,threshold=threshold)

    def run(self):
        self.load_subject_data()

        self.run_glm()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--s_num', '-s', type=str, default='1')
    parser.add_argument('--task','-task',type=str,default='SIpointlights')
    parser.add_argument('--space','-space',type=str,default='MNI152NLin2009cAsym')
    parser.add_argument('--smoothing-fwhm','-smoothing-fwhm',type=float,default=0)
    parser.add_argument('--mask','-mask',type=str,default='None')
    parser.add_argument('--dir', '-dir', type=str,
                        default='/Users/hsmall2/Documents/GitHub/deep_nat_lat')
    parser.add_argument('--data_dir', '-data_dir', type=str,
                        default='/Users/hsmall2/Documents/GitHub/Sherlock_ASD/data') #where the derivatives folder for the fMRI data is 
    parser.add_argument('--out_dir', '-output', type=str,
                        default='/Users/hsmall2/Documents/GitHub/deep_nat_lat/analysis')
    parser.add_argument('--figure_dir', '-figures', type=str,
                        default='/Users/hsmall2/Documents/GitHub/deep_nat_lat/figures')
    args = parser.parse_args()
    print(args)
    GLM(args).run()

if __name__ == '__main__':
    main()
