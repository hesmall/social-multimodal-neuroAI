import sys
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, './encoding')
import encoding
import argparse
import os
import csv
from pathlib import Path
import glob

import helpers
import encoding
from sklearn.manifold import MDS 

import numpy as np
import scipy.stats
from statsmodels.stats.multitest import fdrcorrection
import statsmodels.formula.api as smf #for mixed effects modeling
from scipy.stats import wilcoxon
from statannotations.Annotator import Annotator
import nibabel
import nilearn
from nilearn import plotting
from nilearn import surface
import math

from nilearn.maskers import NiftiSpheresMasker
from bids import BIDSLayout

import pandas as pd
import seaborn as sns
from scipy import stats
from matplotlib import pyplot as plt
from matplotlib import colors
import matplotlib
from matplotlib_venn import venn2
from matplotlib.patches import Patch

from joblib import Parallel, delayed
import pickle
from tqdm import tqdm

from matplotlib.colors import LinearSegmentedColormap

import warnings

plt.rcParams.update({'font.size': 16,'font.family': 'Arial'})

warnings.filterwarnings("ignore", message="`legacy_format` will default to `False` in release 0.11. Dataset fetchers will then return pandas dataframes by default instead of recarrays.")
warnings.filterwarnings("ignore", message="Mean of empty slice.")
warnings.filterwarnings("ignore", message="invalid value encountered in scalar divide")

def is_picklable(obj):
        try:
            pickle.dumps(obj)
            return True
        except (pickle.PicklingError, TypeError):
            return False

class SecondLevelIndividual(encoding.EncodingModel):

    def __init__(self, args):
        self.process = 'SecondLevelIndividual'
        self.dir = args.dir
        self.data_dir = args.dir + '/data'
        self.enc_dir = args.out_dir + '/EncodingModel'
        self.glm_dir = args.out_dir + '/GLM'
        self.out_dir = args.out_dir + "/" + self.process
        self.population = args.population
        self.sid = 'sub-'+self.population
        self.subjects = []
        self.glm_task = ['SIpointlights','language']
        self.enc_task = 'sherlock'
        self.mask_name = args.mask
        self.space = args.space
        self.model = args.model
        self.models = [self.model]#,'sbert']#,'SLIP',sbert']#['SimCLR','SLIP']'SimCLR_SLIP','CLIP_SimCLR_SLI[self.model,'full_w',
        self.smoothing_fwhm = args.smoothing_fwhm #change?
        self.chunklen = args.chunklen
        self.fMRI_data = []
        self.included_data_features = [(26,946),(975,1976)] #Haemy had 27:946, and 973:1976, have to subtract one for python 0-indexing and add one to end bc exclusive slicing in python

        self.brain_shape = (97,115,97)
        self.affine = []
        self.feature_names = []
        self.run_groups = {'SIpointlights':[('12','3'),('23','1'),('13','2'),('123','123')],'language':[('1','2'),('2','1'),('12','12')]} #localizer, response
        self.all_runs = {'SIpointlights':'123','language':'12'}
        # self.localizer_contrasts = {'SIpointlights':{'interact-no_interact','interact&no_interact'},'language':{'intact-degraded','degraded-intact'}}
        self.localizer_contrasts = {'SIpointlights':{'interact&no_interact','interact-no_interact'},'language':{'intact-degraded'}}
        self.MT = ['MT']#['MT']#['pMT','aMT']
        self.ISC = ['ISC']#['pISC','aISC']#['lateral']
        self.STS = ['pSTS','aSTS']#
        self.language =['language']#'temporal_language','frontal_language']
        self.language_ROI_names = ['language']#['temporal language','frontal language']#['pSTS','aSTS'] #label names!
        self.localizer_masks = {'interact&no_interact':self.MT,'interact-no_interact':self.STS,'intact-degraded':self.language,
                                'motion pointlights':self.MT,'SI pointlights':self.STS, 'language':self.language,
                                'motion':self.MT,'num_agents':self.STS, 'alexnet':self.ISC,
                                'social':self.STS,'valence':self.STS,'face':self.STS,'mentalization':self.ISC, 'arousal':self.ISC,
                                'SLIP':self.ISC,'SimCLR':self.ISC,'CLIP':self.language, 'GPT2':self.language,
                                'SLIP_attention':self.ISC,'SimCLR_attention':self.ISC,'SLIP_embedding':self.ISC,'SimCLR_embedding':self.ISC,
                                'glove':self.language,'sbert':self.ISC,'word2vec':self.language,
                                'speaking':self.ISC,'indoor_outdoor':self.ISC,'pitch':self.ISC,'amplitude':self.ISC,
                                'turn_taking':self.ISC,'written_text':self.ISC,'music':self.ISC,'pixel':self.ISC,'hue':self.ISC,'none':self.ISC}
        self.perc_top_voxels = str(args.perc_top_voxels)
        self.n_voxels_all = {
                '1': {'MT':34,'STS':85,'temporal_language':85,'pSTS':60,'aSTS':25,'pTemp':60,'aTemp':25},
                '2.5':{'MT':86,'STS':212,'temporal_language':212,'pSTS':150,'aSTS':62,'pTemp':150,'aTemp':62},
                '5': {'MT':172,'STS':425,'temporal_language':425,'pSTS':300,'aSTS':125,'pTemp':300,'aTemp':125},
                '7.5': {'MT':259,'STS':637,'temporal_language':637,'pSTS':450,'aSTS':187,'pTemp':450,'aTemp':187},
                '10': {'MT':345,'STS':850,'temporal_language':850,'pSTS':600,'aSTS':250,'pTemp':600,'aTemp':250},
                '12.5':{'MT':432,'STS':1062,'temporal_language':1062,'pSTS':750,'aSTS':312,'pTemp':750,'aTemp':312},
                '15':{'MT':518,'STS':1275,'temporal_language':1275,'pSTS':900,'aSTS':375,'pTemp':900,'aTemp':375},
                '20':{'MT':690,'STS':1700,'temporal_language':1700,'pSTS':1200,'aSTS':500,'pTemp':1200,'aTemp':500}
        }
        self.response_contrasts = {'SIpointlights':{'interact','no_interact'},'language':{'intact','degraded'}}
        self.save_weights = False
        self.scale_by = None
        self.group_encoding_weights = []
        self.group_encoding_performance = []
        self.subj_encoding_localizer_masks = []
        self.subj_glm_localizer_masks = []
        self.glm_results = None
        self.performance_stats = []
        self.figure_dir = args.figure_dir + "/" + self.process
        Path(f'{self.out_dir}/{"weights"}').mkdir(exist_ok=True, parents=True)
        Path(f'{self.out_dir}/{"performance"}').mkdir(exist_ok=True, parents=True)
        Path(f'{self.out_dir}/{"localizer_masks"}').mkdir(exist_ok=True, parents=True)
        Path(f'{self.out_dir}/{"all_significant_voxels"}').mkdir(exist_ok=True, parents=True)

        Path(f'{self.figure_dir}/{"localizer_masks"}').mkdir(exist_ok=True, parents=True)
        Path(f'{self.figure_dir}/{"glm_zscores"}').mkdir(exist_ok=True, parents=True)
        Path(f'{self.figure_dir}/{"scatter"}').mkdir(exist_ok=True, parents=True)
        Path(f'{self.figure_dir}/{"localizer_overlap_maps"}').mkdir(exist_ok=True, parents=True)
        Path(f'{self.figure_dir}/{"difference_maps"}').mkdir(exist_ok=True, parents=True)
        Path(f'{self.figure_dir}/{"difference_maps/surface"}').mkdir(exist_ok=True, parents=True)
        Path(f'{self.figure_dir}/{"response_similarity"}').mkdir(exist_ok=True, parents=True)
        Path(f'{self.figure_dir}/{"map_and_localizer"}').mkdir(exist_ok=True, parents=True)
        Path(f'{self.figure_dir}/{"preference_map"}').mkdir(exist_ok=True, parents=True)
        Path(f'{self.figure_dir}/{"map_and_localizer/surface"}').mkdir(exist_ok=True, parents=True)
        Path(f'{self.figure_dir}/{"MDS"}').mkdir(exist_ok=True, parents=True)
        Path(f'{self.figure_dir}/{"overlap"}').mkdir(exist_ok=True, parents=True)
        Path(f'{self.figure_dir}/{"check_null_distribution"}').mkdir(exist_ok=True, parents=True)
        Path(f'{self.figure_dir}/{"features_preferred_delay"}').mkdir(exist_ok=True, parents=True)

        self.enc_file_label = '_smoothingfwhm-'+str(self.smoothing_fwhm)+'_chunklen-'+str(self.chunklen) 
        self.glm_file_label = '_smoothingfwhm-'+str(self.smoothing_fwhm)

        if(self.mask_name!=None):
            self.enc_file_label = self.enc_file_label + '_mask-'+self.mask_name
            self.glm_file_label = self.glm_file_label + '_mask-' +self.mask_name
            self.load_mask(self.mask_name)

        self.labels_dict = {
            # 'SimCLR':'SimCLR only',
            # 'SLIP':'SLIP only',
            'GPT2_1sent':'GPT2',
            'SimCLR_SLIP':'SimCLR+SLIP',
            'SimCLR_SLIP_100ep':'SimCLR+SLIP+100ep',
            'SimCLR_SLIP_SLIPtext':'SimCLR+SLIP+SLIPtext',
            'SimCLR_SLIP_SLIPtext_100ep':'SimCLR+SLIP+SLIPtext+100ep',
            'SimCLR_SLIP_SLIPtext_word2vec':'SimCLR+SLIP+SLIPtext+word2vec',
            'SimCLR_SLIP_SLIPtext_GPT2_word2vec':'full model with SLIP trained 25 epochs',
            # 'SimCLR_SLIP_SLIPtext_GPT2_word2vec':'SimCLR+SLIP+SLIPtext+word2vec+GPT2',
            'SimCLR_SLIP_SLIPtext_100ep_word2vec_GPT2':'full model with SLIP trained 100 epochs',
            # 'SimCLR_SLIP_SLIPtext_GPT2_word2vec':'SimCLR+SLIP+SLIPtext+word2vec+GPT2',
            'interact-no_interact':'social interaction',
            'interact&no_interact':'motion',
            'intact-degraded':'language',
            'interact':'interacting pointlights',
            'no_interact':'non-interacting pointlights',
            'intact':'intact speech',
            'degraded':'degraded speech',
            'social':'social',
            'sbert+word2vec':'sbert+word2vec',
            'alexnet':'alexnet',
            'L':'left',
            'R':'right',
            'SimCLR_SLIP':'SimCLR+SLIP',
            'frontal_language':'frontal language',
            'temporal_language':'temporal language',
            'post_temporal_language':'pTemp',
            'ant_temporal_language':'aTemp',
            'SimCLR_SLIP':'+SLIP',
            'SimCLR_SLIP_SLIPtext':'+SLIPtext',
            'SimCLR_SLIP_SLIPtext_word2vec':'+word2vec',
            # 'SimCLR_SLIP_SLIPtext_GPT2_word2vec':'+word2vec+GPT2',
            }

        self.model_features_dict = helpers.get_models_dict()
        self.combined_features = helpers.get_combined_features()

        self.feature_names = self.model_features_dict[self.model]
        self.plot_features_dict = self.model_features_dict ## Default is to plot all features
        self.plot_features = self.plot_features_dict[self.model]

        self.cmaps = helpers.get_cmaps()

        self.colors_dict = helpers.get_colors_dict()
        self.subjects = helpers.get_subjects(self.population)

    def load_mask(self, mask_name):
        def get_hemi(hemi,mask,radiological):
            fullway = mask.shape[0]
            halfway = int(fullway/2)

            if(radiological):
                #if radiological, swap the left and right
                if(hemi=='right'):
                    mask[halfway+1:fullway] = 0 
                elif(hemi=='left'):
                    mask[0:halfway]=0 
                elif(hemi=='both'):
                    pass
            else:
                if(hemi=='left'):
                    mask[halfway+1:fullway] = 0 #remove right hemi
                elif(hemi=='right'):
                    mask[0:halfway]=0 #remove left hemi
                elif(hemi=='both'):
                    pass


            return mask

        def get_sagittal(sagittal,mask): #if also specifying hemi, hemi should be first for this to be accurate since their might be diff in left and right!!
            Y_index_mask = np.zeros(mask.shape)
            for i in range(mask.shape[0]):
                for j in range(mask.shape[1]):
                    for k in range(mask.shape[2]):
                        Y_index_mask[i, j, k] = j
            Y_mask_indices = Y_index_mask[mask==1]
            posteriorY = np.min(Y_mask_indices)
            anteriorY = np.max(Y_mask_indices)
            midY = int((anteriorY+posteriorY)/2 )
            if(sagittal=='a'): #anterior
                mask[Y_index_mask<midY] = 0 #take out posterior
            elif(sagittal=='p'):#posterior
                mask[Y_index_mask>midY] = 0 #take out anterior
            elif(sagittal=='both'):
                pass
            
            return mask
        #should always be 'hemi'-'sagittal''mask'
        #hemi = 'left','right'
        #sagittal = 'a','p' (anterior,posterior)
        #example: right-aSTS
        hemi = 'both' #default is both hemispheres
        sagittal = ''
        og_mask_name = mask_name
        split = mask_name.split('-')
        if(len(split)>1):
            hemi = split[0]
            mask_name = split[1]
        else:
            mask_name=split[0]

        #check for a sagittal specifier based on mask
        possible_masks = ['STS','MT','MT_STS','STS_and_language','lateral','ventral','ISC','language','frontal_language','temporal_language','pTemp','aTemp']
        if(mask_name not in possible_masks):

            sagittal = mask_name[0] #take the sagittal specifier from the front
            mask_name = mask_name[1:] #keep only the mask

            if(mask_name not in possible_masks):
                print('ERROR! mask not specified correctly! mask = ' + mask_name)

        #load mask
        radiological=False
        if(mask_name=='STS'):
            # atlas = nilearn.datasets.fetch_atlas_destrieux_2009(lateralized=True,legacy_format=False)
            # atlas = nibabel.load(atlas['maps'])
            # self.mask_affine = atlas.affine
            # atlas = atlas.get_fdata()
            
            # mask = atlas.copy()
            # mask[:] = 0
            # mask[(atlas==74)|(atlas==149)] = 1
            left_mask = nibabel.load(self.dir+'/analysis/parcels/lSTS.nii.gz') #Ben Deen's parcels (STS+TPJ)
            right_mask = nibabel.load(self.dir+'/analysis/parcels/rSTS.nii.gz')
            self.mask_affine = left_mask.affine
            mask = ((left_mask.get_fdata() + right_mask.get_fdata())>0)*1
            radiological=True
            #74 is left STS, 149 is right STS
        elif(mask_name=='MT'):
            mask = nibabel.load(self.dir+'/analysis/parcels/MT.nii.gz')
            self.mask_affine = mask.affine
            mask = mask.get_fdata()
        elif(mask_name=='MT_STS'):
            atlas = nilearn.datasets.fetch_atlas_destrieux_2009(lateralized=True,legacy_format=False)
            atlas = nibabel.load(atlas['maps'])
            self.mask_affine = atlas.affine
            atlas = atlas.get_fdata()
            
            mask = atlas.copy()
            mask[:] = 0
            mask[(atlas==74)|(atlas==149)] = 1

            STS_mask = mask
            MT_mask = nibabel.load(self.dir+'/analysis/parcels/MT.nii.gz').get_fdata()
            mask = ((STS_mask==1) | (MT_mask==1) )*1
        elif(mask_name=='STS_and_language'):
            atlas = nilearn.datasets.fetch_atlas_destrieux_2009(lateralized=True,legacy_format=False)
            atlas = nibabel.load(atlas['maps'])
            self.mask_affine = atlas.affine
            atlas = atlas.get_fdata()
            
            STS_mask = atlas.copy()
            STS_mask[:] = 0
            STS_mask[(atlas==74)|(atlas==149)] = 1
            mask=nibabel.Nifti1Image(STS_mask.astype('int32'), self.mask_affine)
            STS_mask = nilearn.image.resample_img(mask, target_affine=self.mask_affine, target_shape=self.brain_shape,interpolation='nearest')

            mask = nibabel.load(self.dir + '/analysis/parcels/langloc_n806_top10%_atlas.nii')
            self.mask_affine = mask.affine
            language_mask = (mask.get_fdata()>0.3)*1 #require 10% overlap across 806 participants for this map
            mask=nibabel.Nifti1Image(language_mask.astype('int32'), self.mask_affine)
            language_mask = nilearn.image.resample_img(mask, target_affine=self.mask_affine, target_shape=self.brain_shape,interpolation='nearest')
            
            mask = ((STS_mask.get_fdata()==1) | (language_mask.get_fdata()==1) )*1
            radiological=True

        elif(mask_name=='lateral'):
            mask=nibabel.load(self.dir+'/analysis/parcels/lateral_STS_mask.nii.gz')
            self.mask_affine = mask.affine 
            mask = mask.get_fdata()
        elif(mask_name=='ventral'):
            mask=nibabel.load(self.dir+'/analysis/parcels/MNI152_T1_1mm.nii.gz')
            self.mask_affine = mask.affine
            mask = mask.get_fdata()
            print(mask[mask>0])
            mask = ((mask==1)|(mask==3)|(mask==5)|(mask==7)|(mask==8)|(mask==9)|(mask==10)|(mask==1))*1
            print(len(mask[mask>0]))
        elif(mask_name =='ISC'):
            mask = nibabel.load(self.dir + '/analysis/parcels/isc_bin.nii')
            self.mask_affine = mask.affine
            mask = mask.get_fdata()
        elif(mask_name =='language'):
            # mask = nibabel.load(self.dir + '/analysis/parcels/allParcels_language_SN220.nii')
            mask = nibabel.load(self.dir + '/analysis/parcels/langloc_n806_top10%_atlas.nii')
            self.mask_affine = mask.affine
            mask = (mask.get_fdata()>0.3)*1 #require 30% overlap across 806 participants for this map
            radiological=True
        elif(mask_name =='frontal_language'):
            mask = nibabel.load(self.dir + '/analysis/parcels/allParcels_language_SN220.nii')
            self.mask_affine = mask.affine
            mask = mask.get_fdata() #frontal regions
            mask = ( (mask==1) | (mask==2) | (mask==3) | (mask==7) | (mask==8) | (mask==9) )*1
            radiological=True

        elif(mask_name =='temporal_language'):
            mask = nibabel.load(self.dir + '/analysis/parcels/allParcels_language_SN220.nii')
            self.mask_affine = mask.affine
            mask = mask.get_fdata() #temporal regions
            mask = ( (mask==4) | (mask==5) | (mask==6) | (mask==10) | (mask==11) | (mask==12) )*1
            radiological=True
        
        elif(mask_name =='pTemp'):
            mask = nibabel.load(self.dir + '/analysis/parcels/allParcels_language_SN220.nii')
            self.mask_affine = mask.affine
            mask = mask.get_fdata() #temporal regions
            mask = ( (mask==5) | (mask==6) | (mask==11) | (mask==12) )*1
            radiological=True

        elif(mask_name =='aTemp'):
            mask = nibabel.load(self.dir + '/analysis/parcels/allParcels_language_SN220.nii')
            self.mask_affine = mask.affine
            mask = mask.get_fdata() #temporal regions
            mask = ( (mask==4) | (mask==10)  )*1
            radiological=True


        #need to do both hemis to get accurate sagittal splits (because left and right hemis might be different)
        mask_dict = {}
        for hemi_label in ['left','right']:
            temp_mask = get_hemi(hemi_label,mask.copy(),radiological)
            
            if(len(sagittal)>0):
                temp_mask = get_sagittal(sagittal,temp_mask)
            mask_dict[hemi_label]=temp_mask

        if(hemi=='both'):
            mask = mask_dict['left'] + mask_dict['right']
            mask = (mask>0)*1
        else:
            mask = mask_dict[hemi]
        # print(og_mask_name, np.sum(mask))
        mask = nibabel.Nifti1Image(mask.astype('int32'), self.mask_affine)
        

        ##intersect the mask with whatever the overarching mask is to account for lack of coverage in the ISC... 
        ## todo: try again with the results from no smoothing
        if(mask_name!=self.mask_name):
            self.load_mask(self.mask_name)
            whole_mask = self.mask.get_fdata()
            # print('ISC mask, ', len(whole_mask[(whole_mask==1)]))
            #now self.mask is set to the overarching mask and self.mask_affine is set to mask
            #resample mask so we can overlay the whole_mask and the mask
            mask = nilearn.image.resample_img(mask, target_affine=self.mask_affine, target_shape=whole_mask.shape,interpolation='nearest')
        
            #only take voxels that are also in the overarching mask
            mask = (mask.get_fdata()==1)&(whole_mask==1)
            self.mask = nibabel.Nifti1Image(mask.astype('int32'), self.mask_affine)

            # print(mask_name,',',len(mask[(mask==1)]))
        else:
            self.mask = mask

        # print(mask_name)
        # print(len(self.mask.get_fdata()[self.mask.get_fdata()==1]))

        #update the self.mask with cleaned mask
        

    def get_feature_index(self, subject,feature, selection_model=''):
        if(selection_model==''):
            file_label = subject+'_encoding_model-'+self.model+self.enc_file_label
        else:
            file_label = subject+'_encoding_model-'+selection_model+self.enc_file_label
        filename = self.enc_dir+'/features/'+file_label+'_features.csv' 

        file = open(filename, "r")
        data = list(csv.reader(file, delimiter=','))[0]
        file.close()
        # print(data.index(feature))
        return data.index(feature)

    def remove_subjects(self, data, subjects):
        for subject in subjects:
            for hemi in ['left','right']:
                data = data[data.subject!=subject]

        return data
    def significance_summary(self,load=False,threshold=0.05,response_label='perf_p_fdr'):
        if(not load):
            params_list = []
            for subject in self.subjects['SIpointlights']:
                for hemi in ['left','right']:
                    params_list.append((subject,hemi))

            def process(subject,hemi):
                enc_file_label = '_encoding_model-'+self.model+'_smoothingfwhm-'+str(self.smoothing_fwhm)+'_chunklen-'+str(self.chunklen)
                if(self.mask_name!=None):
                    enc_file_label = enc_file_label + '_mask-'+self.mask_name
                enc_file_label = subject+enc_file_label

                filepath = self.enc_dir+'/'+response_label+'/'+enc_file_label+'_measure-'+response_label+'.nii.gz'
                nii = nibabel.load(filepath)
                data = nii.get_fdata()

                if(self.mask_name!=None):
                    self.load_mask(hemi+'-'+self.mask_name) #sets self.mask to the mask specified by mask_name, only in the specified hemisphere
                    resampled_mask = nilearn.image.resample_img(self.mask, target_affine=nii.affine, target_shape=nii.shape,interpolation='nearest')
                    self.mask = resampled_mask.get_fdata()
                    data = data[self.mask==1]

                print(data[data>0])

                n_sig_voxels = len(data[data<threshold])
                percent_sig_voxels = n_sig_voxels/len(data)

                return (subject,self.mask_name,hemi,n_sig_voxels,percent_sig_voxels)

            results = Parallel(n_jobs=-1)(delayed(process)(subject,hemi) for (subject,hemi) in tqdm(params_list))
            results = np.array(results)
            results = pd.DataFrame(results,columns =['subject','hemi','mask_name','n_sig_voxels','percent_sig_voxels'])

            results.to_csv(self.out_dir + '/'+self.sid+self.enc_file_label+ '_model-' + self.model+'_measure-'+response_label+'.csv')

        results = pd.read_csv(self.out_dir + '/'+self.sid+self.enc_file_label+ '_model-' + self.model+'_measure-'+response_label+'.csv')

        results.replace(self.labels_dict,inplace=True)

        results['hemi_mask'] = [hemi+ ' ' +mask_name for (hemi,mask_name) in zip(results['hemi'],results['mask_name'])]

        fig = sns.catplot(kind='strip',data=results, 
            x='hemi_mask',y='percent_sig_voxels',
            # errorbar='se',errcolor='black',
            edgecolor="black",linewidth=1)

        for axes in fig.axes:
            for ax in axes:
                # value = ax.title.get_text()
                # if(value=='1'):
                #   title = 'first preference'
                # elif(value=='2'):
                #   title = 'second preference'
                # ax.set_title(title)
                for container in ax.containers:
                    ax.bar_label(container,padding=20, rotation=90,size=6)
        plt.savefig(self.figure_dir + '/'+self.sid+self.enc_file_label+ '_model-' + self.model+'_'+response_label+'_summary.pdf',bbox_inches='tight')
        plt.close()
    def check_null_distribution(self):
        import h5py
        #for each subject, plot the histogram of the voxelwise average null performance score and the histogram of actual performance
        for subject in self.subjects['SIpointlights']:
            #load the subjects null distribution and actual
            try:
                filepath = self.enc_dir + '/performance/'+subject+'_encoding_model-'+self.model+self.enc_file_label+'_measure-perf_raw.nii.gz'
                actual = nibabel.load(filepath).get_fdata().flatten()

                filepath = self.enc_dir + '/performance/'+subject+'_encoding_model-'+self.model+self.enc_file_label+'_measure-perf_null_distribution.h5'
                f = h5py.File(filepath,'r+')
                null = np.mean(f['null_performance'][()],axis=0)

                #plot histograms of both on same figure
                bins = np.linspace(-1,1,100)
                plt.hist(actual, bins, alpha=0.5,label='actual',density=False)
                plt.hist(null,bins,alpha=0.5,label='null',density=False)
                plt.legend(loc='upper right')
                plt.ylim((0,10000))
                plt.title(subject+', '+self.model)
                plt.savefig(self.figure_dir+'/check_null_distribution/'+subject+'_model-'+self.model+self.enc_file_label+'.png')
                plt.close()

            except Exception as e:
                print(e)
                pass
    def plot_preferred_time_delay(self,features,hemis=['left','right'],extraction_threshold=0.05):
        fsaverage = nilearn.datasets.fetch_surf_fsaverage(mesh='fsaverage')
        #for each specified feature, plot a voxelwise preferred time delay brain map
        for subject in self.subjects['SIpointlights']:
            #load subject data
            filepath = self.enc_dir + '/ind_product_measure/'+subject+'_encoding_model-'+self.model+self.enc_file_label+'_measure-ind_product_measure_raw.nii.gz'
            prod_measure = nibabel.load(filepath).get_fdata()
            filepath = self.enc_dir + '/features_preferred_delay/'+subject+'_encoding_model-'+self.model+self.enc_file_label+'_measure-features_preferred_delay.nii.gz'
            nii = nibabel.load(filepath)
            data = nii.get_fdata()
            for feature in features:
                if(feature in self.combined_features):
                    for (ind,sub_feature_name) in enumerate(self.model_features_dict[feature]):
                        feature_ind = self.get_feature_index(subject,sub_feature_name)
                        sub_data = data[feature_ind]
                        sub_prod_data = prod_measure[feature_ind]
                        if(ind==0):
                            overall = sub_data
                            overall_prod = sub_prod_data
                        else:
                            overall = overall+sub_data
                            overall_prod = overall_prod + sub_prod_data
                    curr_data = overall/len(self.model_features_dict[feature]) #average over all the included layers
                    curr_prod_data = overall_prod
                    curr_data[curr_prod_data<extraction_threshold] = 0
                else:
                    feature_index = self.get_feature_index(subject,feature)
                    curr_data = data[feature_index]
                    curr_data[prod_measure[feature_index]<extraction_threshold] = 0
                #plot a preference map
                curr_nii = nibabel.Nifti1Image(curr_data,affine=nii.affine)
                surfaces = []
                for hemi in hemis:
                    transform_mesh = fsaverage['pial_'+hemi]
                    plot_mesh = fsaverage['infl_'+hemi]
                    bg_map = fsaverage['sulc_'+hemi]
                    inner_mesh = fsaverage['white_'+hemi]
                    n_points_to_sample = 50
                    surf = surface.vol_to_surf(curr_nii, transform_mesh,inner_mesh=inner_mesh,depth = np.linspace(0, 1, n_points_to_sample),interpolation='nearest')
                    surfaces.append(surf)
                surf_cmap = 'cool'
                features_list = []
                title = subject
                helpers.plot_preference_surf(surfaces,self.figure_dir + "/features_preferred_delay/" +subject+'_encoding_model-'+self.model+self.enc_file_label+'_measure-features_preferred_delay-'+feature,self.colors_dict,features_list,threshold=0.99,cmap=surf_cmap,make_cmap=False,title=title,vmax = 6,colorbar_label='preferred time delay (TR)')

    def get_preference_map(self,temp_ind_product_measure):
        temp_transposed = temp_ind_product_measure.T
        # print(temp_transposed)
        nan_col = np.nanmax(temp_transposed, axis=1)
        temp = np.zeros(temp_ind_product_measure.shape[1])
        temp[~np.isnan(nan_col)] = np.nanargmax(temp_transposed[~np.isnan(nan_col),:], axis=1)
        return temp.astype(int)

    def generate_preference_maps(self,load=False,restricted=False,threshold=0,features=[],file_tag=''):
        fsaverage = nilearn.datasets.fetch_surf_fsaverage(mesh='fsaverage')
        enc_file_label = '_encoding_model-'+self.model + '_smoothingfwhm-'+str(self.smoothing_fwhm)+'_chunklen-'+str(self.chunklen)
        if(self.mask_name!=None):
            enc_file_label = enc_file_label + '_mask-'+self.mask_name

        for subject in self.subjects['sherlock']:
            #load the feature product measure data
            enc_response_path = self.enc_dir+'/ind_product_measure/'+subject+enc_file_label+'_measure-ind_product_measure_raw.nii.gz' #'raw'
            enc_response_img = nibabel.load(enc_response_path)
            
            hemis= ['left','right']
            preference1_map_surf = []
            preference2_map_surf = []
            for hemi in hemis:
                transform_mesh = fsaverage['pial_'+hemi]
                plot_mesh = fsaverage['infl_'+hemi]
                bg_map = fsaverage['sulc_'+hemi]
                inner_mesh = fsaverage['white_'+hemi]

                temp = enc_response_img.get_fdata()
                temp[temp<0] = 0 #clip response values to 0
                temp = temp/temp.sum(axis=0,keepdims=1) #get the proportion of total variance explained
                final_temp = []
                for feature in features:
                    if(feature in self.combined_features):
                        for (ind,sub_feature_name) in enumerate(self.model_features_dict[feature]):
                            feature_ind = self.get_feature_index(subject,sub_feature_name)
                            sub_data = temp[feature_ind]
                            if(ind==0):
                                overall = sub_data
                            else:
                                overall = overall+sub_data
                        data = overall
                    else:
                        feature_index = self.get_feature_index(subject,feature)
                        data = temp[feature_index]
                    final_temp.append(data)
                temp = np.array(final_temp)
                #add a dimension to begininning that is just 0's (so it will never be the max )
                placeholder = np.reshape(np.zeros(temp[0].shape),(1,temp[0].shape[0],temp[0].shape[1],temp[0].shape[2]))
                temp = np.concatenate((placeholder,temp))
                # print('temp',temp.shape)
                nii = nibabel.Nifti1Image(np.transpose(temp, (1, 2, 3, 0)),enc_response_img.affine)

                n_points_to_sample = 50
                temp_ind_product_measure = np.transpose(surface.vol_to_surf(nii, transform_mesh,inner_mesh=inner_mesh,depth = np.linspace(0, 1, n_points_to_sample),interpolation='nearest'),(1,0))
                # print('temp_ind_product_measure',temp_ind_product_measure.shape)
                if(restricted):
                    temp_ind_product_measure[temp_ind_product_measure<threshold] = np.nan

                preference1_map = self.get_preference_map(temp_ind_product_measure)
                preference1_map_surf.append(preference1_map)

                one_hot_encoded_preference_map = np.eye(temp_ind_product_measure.shape[0])[preference1_map.copy()].T.astype(bool)
                temp_ind_product_measure[one_hot_encoded_preference_map] = 0

                preference2_map = self.get_preference_map(temp_ind_product_measure)
                preference2_map_surf.append(preference2_map)

            #plot on surface
            features_list = ['blank'] + features
            cmap = colors.ListedColormap([self.colors_dict[feature_name] for it in [0,1] for feature_name in features_list ])
            surf_cmap = colors.ListedColormap([self.colors_dict[feature_name] for feature_name in features_list ])
            file_label = subject+'_encoding_model-'+self.model + '_smoothingfwhm-'+str(self.smoothing_fwhm)+'_chunklen-'+str(self.chunklen)+'_'+file_tag
            title = 'first preference'
            helpers.plot_preference_surf(preference1_map_surf,self.figure_dir + "/preference_map/" + file_label+'_measure-preference1_map_surf.png',self.colors_dict,features_list,threshold=0.001,cmap=surf_cmap,title=title,vmax = len(features))
        
    def preference_map_summary(self,load=False,ROIs=['ISC'],ROI_type='anatomical'):
        
        if(not load):
            params_list = []
            for subject in self.subjects['SIpointlights']:
                for preference_map_id in ['1','2']:
                    for feature_name in ['none'] + self.feature_names:
                        if(ROI_type=='anatomical'):
                            for mask_name in ROIs:#self.localizer_masks[feature_name]:
                                for hemi in ['left','right']:
                                    params_list.append((subject,hemi,mask_name,feature_name,preference_map_id))
                        elif(ROI_type=='functional'):
                            for mask_name in ROIs:#self.localizer_masks[feature_name]:
                                for anat_mask in self.localizer_masks[mask_name]:
                                    final_mask = mask_name + ':' + anat_mask
                                    for hemi in ['left','right']:
                                        params_list.append((subject,hemi,final_mask,feature_name,preference_map_id))

            def process(subject,hemi,mask_name,feature_name,preference_map_id):
                try:
                    enc_file_label = '_encoding_model-'+self.model+'_smoothingfwhm-'+str(self.smoothing_fwhm)+'_chunklen-'+str(self.chunklen)
                    if(self.mask_name!=None):
                        enc_file_label = enc_file_label + '_mask-'+self.mask_name
                    enc_file_label = subject+enc_file_label

                    filepath = self.enc_dir+'/preference_map/'+enc_file_label+'_measure-preference'+preference_map_id+'_map.nii.gz'
                    nii = nibabel.load(filepath)
                    data = nii.get_fdata()

                    if(ROI_type=='anatomical'):
                        if(mask_name!=None):
                            self.load_mask(hemi+'-'+mask_name) #sets self.mask to the mask specified by mask_name, only in the specified hemisphere
                            resampled_mask = nilearn.image.resample_img(self.mask, target_affine=nii.affine, target_shape=nii.shape,interpolation='nearest')
                            self.mask = resampled_mask.get_fdata()
                            ### TODO do functional maps??
                            data = data[self.mask==1]
                    elif(ROI_type=='functional'):
                        #load the binary functional localizer mask for this subject, using the localizer within an anatomical mask
                        localizer_contrast = mask_name.split(':')[0]
                        mask = mask_name.split(':')[1]
                        filepath = self.out_dir + '/localizer_masks/'+subject+'_smoothingfwhm-'+str(self.smoothing_fwhm)+'_mask-'+mask+'_glm_loc-'+localizer_contrast+'_run-all_binary.nii.gz'
                        mask = nibabel.load(filepath).get_fdata()
                        data = data[mask==1]
                        mask_name=localizer_contrast

                    data = data.flatten()-1 
                    ### TODO --  fix when rockfish is back1!!!!@
                    # print(data)
                    if(feature_name in self.combined_features):
                        for (ind,sub_feature_name) in enumerate(self.model_features_dict[feature_name]):
                            feature_ind = self.get_feature_index(subject,sub_feature_name)
                            sub_data = enc_perf[feature_ind]
                            if(ind==0):
                                overall = sub_data
                            else:
                                overall = overall+sub_data
                        data = overall
                    if(feature_name!='none'):
                        # print(feature_name)
                        feature_index = self.get_feature_index(subject, feature_name)
                        # print(feature_index)
                    else:
                        feature_index = 0
                    # print(feature_name)
                    # print(feature_index+1)

                    total_n = len(data)
                    feature_n = len(data[data==feature_index]) #shift the feature index by one to match the preference map
                    # print(feature_n)
                    feature_percent = feature_n/total_n *100

                    return (subject,hemi,mask_name,feature_name,preference_map_id,total_n,feature_n,feature_percent)
                except Exception as e:
                    # print(e)
                    return([np.nan]*8)

            results = Parallel(n_jobs=-1)(delayed(process)(subject,hemi,mask_name,feature_name,preference_map_id) for (subject,hemi,mask_name,feature_name,preference_map_id) in tqdm(params_list))
            results = np.array(results)
            results = pd.DataFrame(results,columns =['subject','hemi','mask_name','feature_name','preference_map_id','total_n','feature_n','feature_percent'])

            results.to_csv(self.out_dir + '/'+self.sid+self.enc_file_label+ '_model-' + self.model+'_measure-preference_map.csv')

        results = pd.read_csv(self.out_dir + '/'+self.sid+self.enc_file_label+ '_model-' + self.model+'_measure-preference_map.csv')

        results.replace(self.labels_dict,inplace=True)
        print(results)
        results.dropna(inplace=True)

        results['feature_mask'] = [feature+ ' ' +mask_name for (feature,mask_name) in zip(results['feature_name'],results['mask_name'])]
        results['hemi_mask'] = [hemi+ ' ' +mask_name for (hemi,mask_name) in zip(results['hemi'],results['mask_name'])]

        # fig = sns.catplot(kind='strip',data=results, 
        #   x='feature_mask',y='feature_percent',
        #   # errorbar='se',errcolor='black',
        #   # edgecolor="black",linewidth=1,
        #   dodge=True,
        #   hue='hemi',#hue_order=self.feature_names,palette=self.colors_dict,
        #   col='preference_map_id')
# 
        # order=['left pISC','right pISC','left aISC','right aISC']
        # order = ['left ISC','right ISC']
        order = [hemi+ ' '+ROI.replace('_',' ') for hemi in ['left','right'] for ROI in ROIs]
        if(ROI_type=='functional'):
            order = [hemi+ ' '+self.labels_dict[ROI] for hemi in ['left','right'] for ROI in ROIs]
        # order = ['left MT','right MT']
        # order = ['left lateral','right lateral']
        # order = ['left STS','right STS']
        results.hemi_mask = pd.Categorical(values=results.hemi_mask, categories=order) ## Set the order for the column as you want

        # order=['left pISC','right pISC','left aISC','right aISC']

        fig = sns.relplot(kind='line',data=results[results.preference_map_id==1], 
            x='hemi',y='feature_percent',
            # errorbar='se',errcolor='black',
            # edgecolor="black",linewidth=1,
            # dodge=True,
            # hue='hemi',
            palette='tab20',
            facet_kws={'sharey': False, 'sharex': True},
            estimator=None,
            hue='subject',
            units='subject',#hue_order=self.feature_names,palette=self.colors_dict,
            col='feature_name',
            row='mask_name')

        # fig.set_titles("{col_name}")
        # fig.set_axis_labels("","percentage (%)")
        # for axes in fig.axes:
        #   for ax in axes:
        #       value = ax.title.get_text()
        #       if(value=='1'):
        #           title = 'first preference'
        #       elif(value=='2'):
        #           title = 'second preference'
        #       ax.set_title(title)
        #       for container in ax.containers:
        #           ax.bar_label(container,padding=20, rotation=90,size=6)
        plt.savefig(self.figure_dir + '/'+self.sid+self.enc_file_label+ '_model-' + self.model+'_preference_map_summary.pdf',bbox_inches='tight')
        plt.close()

        fig, (ax1,ax2) = plt.subplots(1,2,figsize=(7,10))
        axes = [ax1,ax2]
        pref_maps = ['1','2']
        palette = self.colors_dict

        fig, (ax1) = plt.subplots(1,figsize=(len(ROIs)*2,10))
        axes = [ax1]
        pref_maps = ['1']#,'2']
        palette = self.colors_dict


        averaged_results = pd.pivot_table(data=results,values='feature_percent', columns=['feature_name','hemi_mask','preference_map_id'],aggfunc='mean')
        averaged_results = pd.melt(averaged_results) #'value' is now the average
        print(averaged_results)
        for ind,(ax,pref_map) in enumerate(zip(axes,pref_maps)):
            temp_averaged_results = averaged_results[averaged_results['preference_map_id']==int(pref_map)].copy()
            print(temp_averaged_results)

            # temp_averaged_results.hemi_mask = pd.Categorical(values=temp_averaged_results.hemi_localizer_contrast_mask, categories=order) ## Set the order for the column as you want

            features = [x for x in self.feature_names]# if x!=localizer]
            p = sns.histplot(data=temp_averaged_results, x='hemi_mask', hue='feature_name',hue_order=features,palette=palette,multiple='stack',weights='value',shrink=0.8,ax=ax,stat='count',linewidth=1.5)
            
            if(pref_map=='1'):
                title = 'first preference'
                legend_elements = [Patch(facecolor=palette[item], edgecolor='k',
                                         label=item) for item in features]
                ax.legend(handles=legend_elements, loc='center')
                sns.move_legend(ax, loc='center left', bbox_to_anchor=(1, 0.5))
            elif(pref_map=='2'):
                title = 'second preference'
                legend_elements = [Patch(facecolor=palette[item], edgecolor='k',
                                         label=item) for item in features]
                ax.legend(handles=legend_elements, loc='center')
                sns.move_legend(ax, loc='center left', bbox_to_anchor=(1, 0.5))
                ax.yaxis.set_visible(False)
            ax.set_title(title)
            ax.set_ylabel('percentage (%)')
            ax.set_xlabel('')
            # ax.set_ylim(0,0.31)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            # if(localizer=='SI pointlights'):
            #   new_xticktexts = ['pSTS','aSTS','pSTS','aSTS']
            # elif(localizer=='language'):
            #   new_xticktexts = ['frontal','temporal','frontal','temporal']
            # new_xticktexts = ['left','right']
            plt.xticks(rotation=90)
            # ax.set_xticklabels(new_xticktexts)

        plt.savefig(self.figure_dir + '/'+self.sid+self.enc_file_label+ '_model-' + self.model+'_preference_map_summary_stacked.pdf',bbox_inches='tight')
        plt.close()


    def encoding_voxel_selection(self,load=False,pvalue=None,selection_model='full',selection_label='ind_feature_performance',response_label='ind_feature_performance',localizers_to_plot=[],localizer_label_dict={},plot_noise_ceiling=False,stats_to_do=None,filepath_tag=''):
        print('encoding voxel selection:')

        fsaverage = nilearn.datasets.fetch_surf_fsaverage(mesh='fsaverage')
        if(self.mask_name!=None):
            # masks = helpers.get_masks('STS','fsaverage')
            self.load_mask(self.mask_name)

        if(selection_label=='performance'):
            measure_label = 'perf_raw'
            features_loc = [selection_model]
        if(selection_label=='ind_feature_performance'):
            measure_label = 'ind_perf_raw'
            # features_loc = self.feature_names
            features_loc = self.plot_features_dict[selection_model]#self.plot_features#['motion','alexnet','social','valence','face','num_agents']
        elif(selection_label=='unique_variance'):
            measure_label = 'unique_var_raw'
            features_loc = self.plot_features_dict[selection_model]
        elif(selection_label=='shared_variance'):
            measure_label = 'shared_var_raw'
            features_loc = ['SimCLR-SLIP']#'speaking','glove','DNN']#,'mentalization']
        elif(selection_label=='ind_product_measure'):
            measure_label = 'ind_product_measure_raw'
            features_loc = self.plot_features_dict[selection_model]
        
        #encoding measure response
        if(response_label=='weights'):
            response_folder = 'weights'
            resp_label = 'weights_raw'
            feature_names = self.plot_features
        elif(response_label=='performance'):
            response_folder = 'performance'
            resp_label='perf_raw'
            feature_names = self.models
        elif(response_label=='ind_feature_performance'):
            response_folder = 'ind_feature_performance'
            resp_label='ind_perf_raw'
            feature_names = self.plot_features#['social','num_agents','face','motion','valence','alexnet']
        elif(response_label=='ind_product_measure'):
            response_folder = 'ind_product_measure'
            resp_label = 'ind_product_measure_raw'
            feature_names = self.plot_features
        elif(response_label=='unique_variance'):
            response_folder='unique_variance'
            resp_label = 'unique_var_raw'
            feature_names = self.plot_features
        elif(response_label=='shared_variance'):
            response_folder='shared_variance'
            resp_label = 'shared_var_raw'
            feature_names = ['SimCLR-SLIP']


        if(not load):

            glm_params_list = []

            for glm_task in self.glm_task:
                response_contrasts = self.response_contrasts[glm_task]
                run_groups = self.run_groups[glm_task]
                for subject in self.subjects[glm_task]:
                    for response_contrast in response_contrasts:
                        for hemi in ['L','R']:
                            for feature_name_loc in features_loc:
                                for mask_name in self.localizer_masks[feature_name_loc]:
                                    for feature_name_resp in feature_names:#[0:self.features_idx]:
                                        # if(feature_name_loc!=feature_name_resp):
                                        glm_params_list.append((subject,glm_task,response_contrast,hemi,feature_name_loc,feature_name_resp,mask_name))


            def process(subject,glm_task,response_contrast,hemi,feature_name_loc,feature_name_resp,pvalue,mask_name):   
                
                enc_file_label = '_encoding_model-'+selection_model+'_smoothingfwhm-'+str(self.smoothing_fwhm)+'_chunklen-'+str(self.chunklen)
                if(self.mask_name!=None):
                    enc_file_label = enc_file_label + '_mask-'+self.mask_name
                enc_file_label = subject+enc_file_label
                glm_file_label=subject+self.glm_file_label

                glm_responses = []
                enc_responses = []
                num_voxels = [] 
                prop_voxels = []
                #load encoding performance
                # enc_weights_path = self.enc_dir+'/weights/'+subject+'_encoding_model-'+self.model+'_feature-'+feature_name+'_measure-weights_raw.nii.gz'
                
                # enc_perf_path = self.enc_dir+'/ind_feature_performance/'+file_label+'_feature-'+feature_name_loc+'_measure-performance_raw.nii.gz'
                if((selection_label=='unique_variance')|(selection_label=='shared_variance')):
                    enc_perf_path = self.enc_dir+'/'+selection_label+'/'+enc_file_label+'_feature-'+feature_name_loc+'_measure-'+measure_label+'.nii.gz'
                    # enc_perf_path = '/Users/hsmall2/Documents/GitHub/deep_nat_lat/analysis/'+selection_label+'/'+enc_file_label+'_feature-'+feature_name_loc+'_measure-'+measure_label+'_raw.nii.gz'
                else:
                    enc_perf_path = self.enc_dir+'/'+selection_label+'/'+enc_file_label+'_measure-'+measure_label+'.nii.gz' #'raw'

                        
                
                # if(pvalue!=None):
                #   enc_perf_p_path = self.enc_dir+'/ind_feature_performance/'+enc_file_label+'_measure-ind_perf_p_fdr.nii.gz'
                #   enc_perf_p_img = nibabel.load(enc_perf_p_path)
                #   enc_perf_p = enc_perf_p_img.get_fdata()
                #   enc_perf_p = enc_perf_p[feature_index]

                enc_perf_img = nibabel.load(enc_perf_path)
                enc_perf = enc_perf_img.get_fdata()
                #get performance of the specified feature only
                if((selection_label=='ind_feature_performance')|(selection_label=='ind_product_measure')|(selection_label=='weights')):
                    if(feature_name_loc in self.combined_features): #load all of the layers for this feature
                        for (ind,sub_feature_name) in enumerate(self.model_features_dict[feature_name_loc]):
                            feature_ind = self.get_feature_index(subject,sub_feature_name,selection_model)
                            sub_data = enc_perf[feature_ind]
                            if(ind==0):
                                overall = sub_data
                            else:
                                overall = overall+sub_data
                        enc_perf = overall
                    else:
                        feature_index = self.get_feature_index(subject, feature_name_loc,selection_model)
                        enc_perf = enc_perf[feature_index]
                perf_img = nibabel.Nifti1Image(enc_perf, enc_perf_img.affine)

                if(mask_name!=None):
                    self.load_mask(mask_name) #sets self.mask to the mask specified by mask_name
                    resampled_mask = nilearn.image.resample_img(self.mask, target_affine=enc_perf_img.affine, target_shape=enc_perf.shape,interpolation='nearest')
                    self.mask = resampled_mask.get_fdata()

                #if using pvalues, mask the performance image with p_binary mask (for significant voxels)
                if(pvalue==None):
                    bilateral = enc_perf[self.mask==1]
                else:
                    bilateral = enc_perf[(self.mask==1)&(enc_perf_p<pvalue)] #within this mask, pvalues of 0 are extremely significant. outside of the mask, 0's just mean not tested

                # threshold, num_voxels_localized = helpers.get_top_n(bilateral, int(self.perc_top_voxels))

                threshold, num_voxels_localized = helpers.get_top_n(bilateral, self.n_voxels_all[self.perc_top_voxels][mask_name])

                enc_file_label = '_encoding_model-'+self.model+'_smoothingfwhm-'+str(self.smoothing_fwhm)+'_chunklen-'+str(self.chunklen)
                if(response_label=='performance'):
                    enc_file_label = '_encoding_model-'+feature_name_resp+'_smoothingfwhm-'+str(self.smoothing_fwhm)+'_chunklen-'+str(self.chunklen)
                if(self.mask_name!=None):
                    enc_file_label = enc_file_label + '_mask-'+self.mask_name
                enc_file_label = subject+enc_file_label

                if((response_label=='unique_variance')|(response_label=='shared_variance')):
                    enc_response_path = self.enc_dir+'/'+response_folder+'/'+enc_file_label+'_feature-'+feature_name_resp+'_measure-'+resp_label+'.nii.gz'
                    enc_response_img = nibabel.load(enc_response_path)
                    enc_response = enc_response_img.get_fdata()
                else:
                    
                    enc_response_path = self.enc_dir+'/'+response_folder+'/'+enc_file_label+'_measure-'+resp_label+'.nii.gz'

                    enc_response_img = nibabel.load(enc_response_path)
                    enc_response = enc_response_img.get_fdata()
                    if(self.scale_by=='total_variance'):
                        enc_response[enc_response<0] = 0 #clip response values to 0
                        enc_response = enc_response/enc_response.sum(axis=0,keepdims=1)
                    if((response_label=='ind_feature_performance')|(response_label=='ind_product_measure')|(response_label=='weights')):
                        if(feature_name_resp in self.combined_features): #if a _layers feature, load all of the layers for this feature
                            for (ind,sub_feature_name) in enumerate(self.model_features_dict[feature_name_resp]):
                                feature_ind = self.get_feature_index(subject,sub_feature_name)
                                sub_data = enc_response[feature_ind]
                                if(ind==0):
                                    overall = sub_data
                                else:
                                    overall = overall+sub_data
                            enc_response = overall
                        else:
                            feature_index = self.get_feature_index(subject, feature_name_resp)
                            enc_response = enc_response[feature_index]
                        
                        if((response_label=='ind_product_measure')&(self.scale_by!=None)):
                            plot_noise_ceiling=False
                            #get the proportion of the noise ceiling
                            if(self.scale_by=='noise_ceiling'):
                                temp_file_label = self.sid +'_smoothingfwhm-'+str(self.smoothing_fwhm)+'_type-leave_one_out'+'_mask-None'
                                filepath = self.dir+'/analysis/IntersubjectCorrelation/intersubject_correlation/'+temp_file_label+'_measure-intersubject_explained_variance.nii.gz'
                                noise_ceiling_img = nibabel.load(filepath)
                                scaling_data = noise_ceiling_img.get_fdata()

                                enc_response[enc_response<0] = 0 #clip response values to 0
                                scaling_data[scaling_data<0]= np.nan #get rid of voxels with negative performance
                                enc_response = enc_response/scaling_data #get the proportion of the noise ceiling

                remove_overlap = False
                if(remove_overlap):
                    filepath = self.out_dir+'/localizer_masks/'+subject+'_overlap-sbert_social_binary.nii.gz'
                    overlap_mask = nibabel.load(filepath).get_fdata()
                    if(pvalue==None):
                        localizer_mask = ( (self.mask==1) &(enc_perf>threshold) &(overlap_mask==0))
                    else:
                        localizer_mask = ((self.mask==1) &(enc_perf>threshold) & (enc_perf_p<pvalue) &(overlap_mask==0)) #less than or equal to account for very low pvalues 
                        # print('localizer_mask')
                        # print(np.sum(localizer_mask))
                if(pvalue==None):
                    localizer_mask = ( (self.mask==1) &(enc_perf>threshold))
                else:
                    localizer_mask = ((self.mask==1) &(enc_perf>threshold) & (enc_perf_p<pvalue))
                vmax=None
                save_threshold = threshold
                plot_label = 'localizer - '+ feature_name_loc +', top'+str(self.perc_top_voxels)+'% bilateral STS threshold='+str(threshold)
                file_label = subject + '_encoding_model-'+selection_model+'_smoothingfwhm-'+str(self.smoothing_fwhm)+'_chunklen-'+str(self.chunklen) + '_mask-'+mask_name
                # filepath = self.figure_dir + '/localizer_masks/'+file_label+'_measure-'+measure_label+'_enc_feature_loc-'+feature_name_loc+'_hemi-'+hemi+'.html'
                img = nibabel.Nifti1Image(localizer_mask.astype('uint8'), enc_perf_img.affine)

                if(hemi=='L'):
                    filepath = self.out_dir + '/localizer_masks/'+file_label+'_measure-'+measure_label+'_enc_feature_loc-'+feature_name_loc+'.nii.gz'
                    nibabel.save(img, filepath)  
                
                if(hemi=='L'):
                    mask_name_ = 'left-'+self.mask_name
                    self.load_mask(mask_name_)
                    resampled_mask = nilearn.image.resample_img(self.mask, target_affine=enc_perf_img.affine, target_shape=enc_perf.shape,interpolation='nearest')
                    mask = resampled_mask.get_fdata()
                if(hemi=='R'):
                    mask_name_ = 'right-'+self.mask_name
                    self.load_mask(mask_name_)
                    resampled_mask = nilearn.image.resample_img(self.mask, target_affine=enc_perf_img.affine, target_shape=enc_perf.shape,interpolation='nearest')
                    mask = resampled_mask.get_fdata()

                #get glm weights averaged across the specified response runs
                for run in self.all_runs[glm_task]:#['1','2','3']:
                    glm_weights_path = self.glm_dir + '/'+ subject+'/'+ subject + "_task-"+ glm_task+'_space-'+self.space+ "_run-"+run+"_contrast-"+response_contrast+ "_measure-weights.nii.gz"
                    glm_weights_img = nibabel.load(glm_weights_path)
                    glm_weights = glm_weights_img.get_fdata()

                    glm_data = glm_weights[ (mask==1) & localizer_mask ]
                    enc_data = enc_response[ (mask==1) & localizer_mask ]

                    num_voxel = len(enc_data)

                    if(np.isnan(glm_data).all()):
                            glm_responses.append(np.nan)
                    else:
                        glm_responses.append(np.nanmean(glm_data))
                        
                    if(np.isnan(enc_data).all()):
                        enc_responses.append(np.nan)
                    else:
                        enc_responses.append(np.nanmean(enc_data))
                    # num_voxels.append(num_voxel/int(self.perc_top_voxels))
                    num_voxels.append(num_voxel)
                    prop_voxels.append(num_voxel/self.n_voxels_all[self.perc_top_voxels][mask_name])
                num_voxels = np.mean(num_voxels)
                prop_voxels = np.mean(prop_voxels)
                if(np.isnan(glm_responses).all()):
                    glm_responses = np.nan
                else:
                    glm_responses = np.nanmean(glm_responses)
                if(np.isnan(enc_responses).all()):
                    enc_responses = np.nan
                else:
                    enc_responses = np.nanmean(enc_responses)
                return (subject,glm_task,response_contrast, hemi,mask_name, feature_name_loc, feature_name_resp, glm_responses, enc_responses, num_voxels, prop_voxels, save_threshold)
            results = Parallel(n_jobs=-1)(delayed(process)(subject,glm_task,response_contrast,hemi,feature_name_loc,feature_name_resp, pvalue,mask_name) for (subject,glm_task,response_contrast,hemi,feature_name_loc,feature_name_resp,mask_name) in tqdm(glm_params_list))
            results = np.array(results)
            results = pd.DataFrame(results,columns =['subject','glm_task','glm_response_contrast','hemisphere','mask','enc_feature_name_loc','enc_feature_name_resp','glm_weight','encoding_response','num_voxels','proportion_voxels','threshold'])

            results.to_csv(self.out_dir + '/'+self.sid+self.enc_file_label+ '_model-' + self.model+'_measure-'+measure_label+'_perc_top_voxels-' + self.perc_top_voxels+'_subject_level_encoding_localizer_'+filepath_tag+'.csv')

        results = pd.read_csv(self.out_dir + '/'+self.sid+self.enc_file_label+ '_model-' + self.model+'_measure-'+measure_label+'_perc_top_voxels-' + self.perc_top_voxels+'_subject_level_encoding_localizer_'+filepath_tag+'.csv')

        results.replace(self.labels_dict,inplace=True)

        localizer_contrasts = [localizer + ' (' + localizer_label_dict[localizer]+')' for localizer in localizers_to_plot]
        width_ratios = [len(self.localizer_masks[localizer]) for localizer in localizers_to_plot]

        self.plot_enc_response(results, column_group='localizer_contrast_label',col_order=localizer_contrasts,label_dict=localizer_label_dict,width_ratios=width_ratios,response_label=response_label,hue='enc_feature_name_resp',loc_name='enc_feature_name_loc',model_label='enc_feature_name_resp',file_label='enc_localizer_'+measure_label,plot_noise_ceiling=plot_noise_ceiling,stats_to_do=stats_to_do,filepath_tag=filepath_tag)

        self.plot_glm_response(results, column_group='localizer_contrast_label',col_order=localizer_contrasts,label_dict=localizer_label_dict,width_ratios=width_ratios,loc_name='enc_feature_name_loc',model_label='enc_feature_name_resp',selection_model=selection_model,file_label='enc_localizer_'+measure_label,filepath_tag=filepath_tag)


    def glm_voxel_selection(self,load=False,plot_ind=True,plot_stacked=True,response_label='ind_feature_performance',model_to_measure_proportion_from='',pvalue=None,localizers_to_plot=[],localizer_label_dict={},plot_noise_ceiling=False,stats_to_do=None,filepath_tag='',extraction_threshold=None):
        '''For GLM, get the localizer mask of the top % beta weight in mask for each subject.
        For encoding, get the localizer mask of the top % beta weight in mask where perf>0.1 for each subject'''
        print('glm voxel selection:')
        fsaverage = nilearn.datasets.fetch_surf_fsaverage(mesh='fsaverage')
        
        if(response_label=='weights'):
            response_folder = 'weights'
            resp_label = 'weights_raw'
            feature_names = self.feature_names
            models = [self.model]
        elif(response_label=='performance'):
            response_folder = 'performance'
            resp_label = 'perf_raw'
            feature_names = ['NA']
            models = self.models #+ ['CLIP','SimCLR_SLIP','social','glove']
        elif(response_label=='ind_feature_performance'):
            response_folder = 'ind_feature_performance'
            resp_label = 'ind_perf_raw'
            feature_names = self.plot_features#['social','num_agents','turn_taking','speaking','mentalization','valence','arousal','glove','motion','face','indoor_outdoor','written_text','pixel','hue','amplitude','pitch','music','alexnet']
            # feature_names = ['social','num_agents','face']#,'speaking','valence','arousal','face','mentalization']#,'valence','arousal','glove','alexnet']
            # feature_names = ['social','motion','valence','alexnet']
            # feature_names = ['social','num_agents','alexnet','motion','valence','arousal','mentalization','face','indoor_outdoor']
            models = [self.model]
        elif(response_label=='ind_product_measure'):
            response_folder = 'ind_product_measure'
            resp_label = 'ind_product_measure_raw'
            feature_names = self.plot_features#['social','num_agents','turn_taking','speaking','mentalization','valence','arousal','glove','motion','face','indoor_outdoor','written_text','pixel','hue','amplitude','pitch','music','alexnet']
            # feature_names = ['social','num_agents','face']#,'speaking','valence','arousal','face','mentalization']#,'valence','arousal','glove','alexnet']
            # feature_names = ['social','motion','valence','alexnet']
            # feature_names = ['social','num_agents','alexnet','motion','valence','arousal','mentalization','face','indoor_outdoor']
            models = [self.model]
        elif(response_label=='features_preferred_delay'):
            response_folder = 'features_preferred_delay'
            resp_label = 'features_preferred_delay'
            feature_names = self.plot_features
            models = [self.model]
        elif(response_label=='unique_variance'):
            response_folder='unique_variance'
            resp_label = 'unique_var_raw'
            models = [self.model]
            feature_names = self.plot_features
        elif(response_label=='shared_variance'):
            response_folder='shared_variance'
            resp_label = 'shared_var_raw'
            models = [self.model]
            feature_names = ['SimCLR-SLIP']
        elif(response_label=='ISC'):
            response_folder='intersubject_correlation'
            feature_names = ['n/a']
            models = [self.model]

        if(not load):
            glm_params_list = []

            for glm_task1 in self.glm_task:
                localizer_contrasts = self.localizer_contrasts[glm_task1]
                for glm_task2 in self.glm_task:
                    if(glm_task1!=glm_task2):
                        response_contrasts = self.response_contrasts[glm_task2]
                        run_groups = [(self.all_runs[glm_task1],self.all_runs[glm_task2])]
                    else:
                        response_contrasts = self.response_contrasts[glm_task1]
                        run_groups = self.run_groups[glm_task1]

                    for subject in self.subjects[glm_task1]:
                        for localizer_contrast in localizer_contrasts:
                            for response_contrast in response_contrasts:
                                for model in models:
                                    for hemi in ['L','R']:
                                        for mask_name in self.localizer_masks[localizer_contrast]:#,'pMT','aMT']:#,'pSTS','aSTS']:#self.localizer_masks[localizer_contrast]:
                                            for feature_name in feature_names:#[0:self.features_idx]:
                                                if(subject in self.subjects[glm_task2]):
                                                    glm_params_list.append((subject,glm_task1,glm_task2,localizer_contrast,response_contrast,run_groups,hemi,feature_name,mask_name,model))
            def process(subject,glm_task1,glm_task2, localizer_contrast,response_contrast,run_groups,hemi,feature_name,mask_name,model):
                file_label = subject+'_smoothingfwhm-'+str(self.smoothing_fwhm)+'_mask-'+mask_name
                enc_file_label = subject+self.enc_file_label
                glm_responses = []
                enc_responses = []
                ISC_responses = []
                num_voxels = []
                prop_voxels =[]
                thresholds = []
                localizer_runs = []
                response_runs = []
                # print(glm_task1)
                # print(glm_task2)
                # print(run_groups)
                for localize_run,response_run in run_groups: #average across the different run groups 

                    #### TODO -- do in same way as encoding! pvalue mask and then highest weights? or zscores?

                    #load GLM localizer data
                    if(pvalue == None):
                        z_scores_path = self.glm_dir + '/'+ subject+'/'+subject + "_task-"+ glm_task1+'_space-'+self.space+ "_run-"+ localize_run + "_contrast-"+localizer_contrast+ "_measure-zscore.nii.gz"
                    else:
                        z_scores_path = self.glm_dir + '/'+ subject+'/'+subject + "_task-"+ glm_task1+'_space-'+self.space+ "_run-"+ localize_run + "_contrast-"+localizer_contrast+ "_measure-pvalue.nii.gz"

                    z_scores_img = nibabel.load(z_scores_path)
                    z_scores = z_scores_img.get_fdata()
                    # p_values_path = z_scores_path = self.glm_dir + '/'+ subject+'/'+subject + "_task-"+ glm_task1+'_space-'+self.space+ "_run-"+ localize_run + "_contrast-"+localizer_contrast+ "_measure-pvalue.nii.gz"
                    # p_values = nibabel.load(p_values_path).get_fdata()

                    # #get mask of p_values<0.05 before doing the top n voxels selection
                    # p_value_threshold = 1-0.05
                    # temp_zscores = 1-z_scores
                    # p_value_mask = (temp_zscores>p_value_threshold)
                
                    if(mask_name!=None):
                        self.load_mask(mask_name) #sets self.mask to the mask specified by mask_name
                        resampled_mask = nilearn.image.resample_img(self.mask, target_affine=z_scores_img.affine, target_shape=z_scores.shape,interpolation='nearest')
                        self.mask = resampled_mask.get_fdata()
                        localizer_mask = self.mask #temporary, use this later


                    #combine bilateral z-scores and get top voxels within the mask
                    bilateral = z_scores[((self.mask==1))]#&(p_value_mask))]
                    # print(z_scores[self.mask==1])

                    n_top_voxels = self.n_voxels_all[self.perc_top_voxels][mask_name]#get the number of voxels for this percentage of top voxels
                    threshold, num_voxels_localized = helpers.get_top_n(bilateral, int(n_top_voxels))
                    save_threshold = threshold
                    if(pvalue!=None):
                        # threshold, num_voxels_localized = helpers.get_bottom_n(bilateral, 100)
                        # #restrict to 100 voxels
                        # print('here')
                        # print(num_voxels_localized)
                        # print(threshold)
                        # if(threshold>pvalue):
                        threshold=1-pvalue
                        z_scores = 1-z_scores #z_scores are actually p_values!! need to subtract from 1 to threshold correctly

                    if(hemi=='L'):
                        mask_name_ = 'left-'+mask_name
                        self.load_mask(mask_name_)
                        resampled_mask = nilearn.image.resample_img(self.mask, target_affine=z_scores_img.affine, target_shape=z_scores.shape,interpolation='nearest')
                        mask = resampled_mask.get_fdata()   
                    if(hemi=='R'):
                        mask_name_ = 'right-'+mask_name
                        self.load_mask(mask_name_)
                        resampled_mask = nilearn.image.resample_img(self.mask, target_affine=z_scores_img.affine, target_shape=z_scores.shape,interpolation='nearest')
                        mask = resampled_mask.get_fdata()

                    glm_weights_path = self.glm_dir + '/'+ subject+'/'+ subject + "_task-"+glm_task2+'_space-'+self.space+ "_run-"+ response_run + "_contrast-"+response_contrast+ "_measure-weights.nii.gz"
                    glm_weights_img = nibabel.load(glm_weights_path)
                    glm_weights = glm_weights_img.get_fdata()

                    enc_file_label = '_encoding_model-'+model + '_smoothingfwhm-'+str(self.smoothing_fwhm)+'_chunklen-'+str(self.chunklen)
                    enc_file_label_to_get_proportion_from = '_encoding_model-'+model_to_measure_proportion_from+ '_smoothingfwhm-'+str(self.smoothing_fwhm)+'_chunklen-'+str(self.chunklen)

                    if(self.mask_name!=None):
                        enc_file_label = enc_file_label + '_mask-'+self.mask_name
                        enc_file_label_to_get_proportion_from = enc_file_label_to_get_proportion_from + '_mask-'+self.mask_name
                    
                    ISC_file_label = 'sub-NT' +'_smoothingfwhm-'+str(self.smoothing_fwhm)+'_type-leave_one_out'+'_mask-None'
                    ISC_filepath = self.dir+'/analysis/IntersubjectCorrelation/intersubject_correlation/'+ISC_file_label+'_measure-intersubject_explained_variance.nii.gz'
                    ISC = nibabel.load(ISC_filepath).get_fdata()

                    # enc_weights_path = self.enc_dir+'/weights/'+file_label+'_feature-'+feature_name+'_measure-weights_raw.nii.gz'
                    if((response_label=='unique_variance')|(response_label=='shared_variance')):
                        enc_response_path = self.enc_dir+'/'+response_folder+'/'+subject+enc_file_label+'_feature-'+feature_name+'_measure-'+resp_label+'.nii.gz'
                        enc_response_img = nibabel.load(enc_response_path)
                        # enc_response_img = nilearn.image.resample_img(enc_response_img, target_affine=z_scores_img.affine, target_shape=z_scores.shape,interpolation='nearest')
                        enc_response = enc_response_img.get_fdata()
                    elif(response_label=='ISC'):
                        temp_file_label = self.sid +'_smoothingfwhm-3.0_type-leave_one_out'+'_mask-None'
                        enc_response_path = self.dir+'/analysis/IntersubjectCorrelation/intersubject_correlation/'+temp_file_label+'_measure-intersubject_explained_variance.nii.gz'
                        enc_response_img = nibabel.load(enc_response_path)
                        enc_response = enc_response_img.get_fdata()
                    else:
                        enc_response_path = self.enc_dir+'/'+response_folder+'/'+subject+enc_file_label+'_measure-'+resp_label+'.nii.gz' #'raw'
                        enc_response_img = nibabel.load(enc_response_path)
                        if(model_to_measure_proportion_from!=''):
                            enc_response_to_get_proportion_from = nibabel.load(self.enc_dir+'/'+response_folder+'/'+subject+enc_file_label_to_get_proportion_from+'_measure-'+resp_label+'.nii.gz').get_fdata()
                        # enc_response_img = nilearn.image.resample_img(enc_response_img, target_affine=z_scores_img.affine, target_shape=z_scores.shape,interpolation='nearest')
                        enc_response = enc_response_img.get_fdata()
                        
                        temp = nibabel.load(self.enc_dir+'/ind_product_measure/'+subject+enc_file_label+'_measure-ind_product_measure_raw.nii.gz').get_fdata()
                        # print(enc_response)
                        if((response_label=='ind_feature_performance')|(response_label=='ind_product_measure')|(response_label=='weights')|(response_label=='features_preferred_delay')):
                            #if scale_by total variance, normalize the variance decomposition vector, after clipping negative values to 0
                            
                            if(self.scale_by=='total_variance'):
                                enc_response[enc_response<0] = 0 #clip response values to 0
                                # enc_response[enc_response<0] = 0 #clip response values to 0
                                if(model_to_measure_proportion_from==''):
                                    enc_response = enc_response/enc_response.sum(axis=0,keepdims=1)
                                else:
                                    enc_response = enc_response/enc_response_to_get_proportion_from.sum(axis=0,keepdims=1)
                            if(feature_name in self.combined_features): #if a _layers feature, load all of the layers for this feature
                                for (ind,sub_feature_name) in enumerate(self.model_features_dict[feature_name]):
                                    feature_ind = self.get_feature_index(subject,sub_feature_name)
                                    sub_data = enc_response[feature_ind]
                                    temp_data = temp[feature_ind]
                                    if(ind==0):
                                        overall = sub_data
                                        overall_temp = temp_data
                                    else:
                                        overall = overall+sub_data
                                        overall_temp = overall_temp+temp_data
                                enc_response = overall
                                if(response_label=='features_preferred_delay'):
                                    enc_response = enc_response/len(self.model_features_dict[feature_name]) #average over the layers
                                    mask[overall_temp<extraction_threshold] = 0 #take out any voxels that don't have sufficient performance
                            else:
                                feature_index = self.get_feature_index(subject, feature_name)
                                enc_response = enc_response[feature_index] #use the relevant feature only
                                if(response_label=='features_preferred_delay'):
                                    # print('mask before', np.sum(mask))
                                    mask[temp[feature_index]<extraction_threshold] = 0 #take out any voxels that don't have sufficient performance
                                    # print('mask after',np.sum(mask))

                            if((response_label=='ind_product_measure')&(self.scale_by!=None)):
                                plot_noise_ceiling=False
                                if(self.scale_by=='noise_ceiling'):
                                    temp_file_label = self.sid +'_smoothingfwhm-'+str(self.smoothing_fwhm)+'_type-leave_one_out'+'_mask-None'
                                    filepath = self.dir+'/analysis/IntersubjectCorrelation/intersubject_correlation/'+temp_file_label+'_measure-intersubject_explained_variance.nii.gz'
                                    noise_ceiling_img = nibabel.load(filepath)
                                    scaling_data = noise_ceiling_img.get_fdata()
                                
                                    enc_response[enc_response<0] = 0 #clip response values to 0
                                    scaling_data[scaling_data<0]= np.nan #get rid of voxels with negative performance
                                    enc_response = enc_response/scaling_data #get the proportion of the noise ceiling
                                

                    localizer_mask = ((localizer_mask==1) & (z_scores>threshold))

                    plot_label = 'localizer - '+ localizer_contrast +', top'+str(self.perc_top_voxels)+'% bilateral STS threshold='+str(threshold)

                    filepath = self.figure_dir + '/localizer_masks/'+file_label+'_glm_loc-'+localizer_contrast+'_run-'+localize_run+'_hemi-'+hemi+'.html'
                    img = nibabel.Nifti1Image(localizer_mask.astype('uint8'), z_scores_img.affine)
                    # threshold = 0.1

                    if(hemi=='L'): #only need to do this once
                        filepath = self.out_dir + '/localizer_masks/'+file_label+'_glm_loc-'+localizer_contrast+'_run-'+localize_run+'.nii.gz'
                        nibabel.save(img, filepath)  
                    
                    if(glm_task1!=glm_task2)|(localize_run!=self.all_runs[glm_task1]): #don't use localizer from all runs in compiling data
                        
                        remove_overlap = False 
                        if(remove_overlap):
                            filepath = self.out_dir+'/localizer_masks/'+subject+'_overlap-intact-degraded_interact-no_interact_binary.nii.gz'
                            overlap_mask = nibabel.load(filepath).get_fdata()

                            glm_data = glm_weights[ ((mask==1) & localizer_mask) &(overlap_mask==0) ]
                            enc_data = enc_response[ ((mask==1) & localizer_mask) &(overlap_mask==0)]
                            ISC_data = ISC[ ((mask==1) & localizer_mask) &(overlap_mask==0)]
                        else:
                            glm_data = glm_weights[ ((mask==1) & localizer_mask) ]
                            enc_data = enc_response[ ((mask==1) & localizer_mask) ]
                            ISC_data = ISC[ ((mask==1) & localizer_mask) ]
                            # print(enc_data.shape)
                            # print(enc_data)
                        # print(enc_data)
                        num_voxel = len(glm_data)
                        localizer_runs.append(str(localize_run))
                        response_runs.append(str(response_run))

                        if(np.isnan(glm_data).all()):
                            glm_responses.append(np.nan)
                        else:
                            glm_responses.append(np.nanmean(glm_data))
                        
                        if(np.isnan(enc_data).all()):
                            enc_responses.append(np.nan)
                        else:
                            enc_responses.append(np.nanmean(enc_data))
                        if(np.isnan(ISC_data).all()):
                            ISC_responses.append(np.nan)
                        else:
                            ISC_responses.append(np.nanmean(ISC_data))
                        prop_voxels.append(num_voxel/self.n_voxels_all[self.perc_top_voxels][mask_name])
                        num_voxels.append(num_voxel)
                        thresholds.append(save_threshold)

                num_voxels = np.mean(num_voxels)
                prop_voxels = np.mean(prop_voxels)
                #remove all inf values and nan values
                glm_responses = [v for v in glm_responses if not math.isnan(v) and not math.isinf(v)]
                enc_responses = [v for v in enc_responses if not math.isnan(v) and not math.isinf(v)]
                ISC_responses = [v for v in ISC_responses if not math.isnan(v) and not math.isinf(v)]
                # if(np.isnan(glm_responses).all()):
                #   glm_responses = np.nan
                # else:
                glm_responses = np.nanmean(glm_responses)
                # if(np.isnan(enc_responses).all()):
                #   enc_responses = np.nan
                # else:
                # print(enc_responses)
                enc_responses = np.nanmean(enc_responses)
                ISC_responses = np.nanmean(ISC_responses)
                # print(enc_responses)
                thresholds = np.mean(thresholds)

                return (subject, glm_task1, glm_task2, localizer_contrast, response_contrast, hemi, mask_name,model,feature_name, glm_responses, enc_responses, ISC_responses,num_voxels,prop_voxels, thresholds,','.join(localizer_runs),','.join(response_runs))
            
            results = Parallel(n_jobs=-1)(delayed(process)(subject,glm_task1,glm_task2,localizer_contrast,response_contrast,run_groups,hemi,feature_name,mask_name,model) for (subject, glm_task1,glm_task2,localizer_contrast,response_contrast,run_groups,hemi,feature_name,mask_name,model) in tqdm(glm_params_list))
            results = np.array(results)
            results = pd.DataFrame(results,columns = ['subject','glm_task_localizer','glm_task_response','localizer_contrast','glm_response_contrast','hemisphere','mask','model','enc_feature_name','glm_weight','encoding_response','ISC','num_voxels','proportion_voxels','threshold','averaged_localizer_runs','averaged_response_runs'])

            self.glm_results = results
            results.to_csv(self.out_dir + '/'+self.sid+self.enc_file_label+ '_model-' + self.model+'_perc_top_voxels-' + self.perc_top_voxels+'_subject_level_glm_localizer_enc_response-'+response_label+'_'+filepath_tag+'.csv')

        results = pd.read_csv(self.out_dir + '/'+self.sid+self.enc_file_label+ '_model-' + self.model+'_perc_top_voxels-' + self.perc_top_voxels+'_subject_level_glm_localizer_enc_response-'+response_label+'_'+filepath_tag+'.csv')
        

        if(response_label=='performance'):
            #put ISC as a model
            temp_results = results.copy()
            temp_results['model'] = ['ISC' for item in results['model']]
            temp_results['encoding_response'] = temp_results['ISC']

            results = pd.concat([results,temp_results])
            print(results)

        # localizers_to_plot = ['motion pointlights','SI pointlights','language']
        # localizer_label_dict = {'motion pointlights':'MT','SI pointlights':'STS','language':'language'}
        localizer_contrasts = [localizer + ' (' + localizer_label_dict[localizer]+')' for localizer in localizers_to_plot]
        # if (average_posterior_anterior):
        width_ratios = [1 for localizer in localizers_to_plot]
        # else:
        width_ratios = [len(self.localizer_masks[localizer]) for localizer in localizers_to_plot]

        # self.plot_glm_response(results, average_posterior_anterior=False,column_group='localizer_contrast_label',col_order=localizer_contrasts,label_dict=localizer_label_dict,width_ratios=width_ratios,loc_name='localizer_contrast',model_label='model',selection_model='',file_label='glm_localizer',filepath_tag=filepath_tag)
        self.plot_enc_response(results, average_posterior_anterior=False,column_group='localizer_contrast_label',col_order=localizer_contrasts,label_dict=localizer_label_dict,width_ratios=width_ratios,response_label=response_label,hue='enc_feature_name',loc_name='localizer_contrast',model_label='model',file_label='glm_localizer',plot_stacked=plot_stacked,plot_noise_ceiling=plot_noise_ceiling,stats_to_do=stats_to_do,filepath_tag=filepath_tag)

    def plot_glm_response(self,results,average_posterior_anterior,column_group,col_order,width_ratios,label_dict,loc_name,model_label,selection_model,file_label,filepath_tag):
        plt.rcParams.update({'font.size': 14,'font.family': 'Arial'})
        
        results = results.dropna(axis='index')
        results = results.drop(columns=['Unnamed: 0'])

        
        if (average_posterior_anterior):
            for prev_mask in self.STS:
                results['mask'].replace(prev_mask, 'STS', inplace=True) 
            for prev_mask in self.language:
                results['mask'].replace(prev_mask, 'temporal_language', inplace=True)
            #average across the masks 
            average_these = ['glm_weight','encoding_response','proportion_voxels']
            columns = [ind for ind in results.columns if ind not in average_these]
            results = pd.pivot_table(data=results,values=average_these, index = columns, aggfunc='mean').reset_index()

        results.replace(self.labels_dict,inplace=True)
        localizer_label_dict = label_dict
        results['localizer_contrast_label'] = [localizer_contrast + ' (' + localizer_label_dict[localizer_contrast]+')' for localizer_contrast in results[loc_name]]
        results['hemi_mask'] = [hemi + ' ' + mask for (hemi,mask) in zip(results['hemisphere'],results['mask'])]
        results['Condition'] =  [x for x in results['glm_response_contrast']]
        
        
        # Pivot the table to get right and left hemisphere responses for each group
        group_columns = ['subject',loc_name,'mask']
        df_pivot = results.pivot_table(index=group_columns, 
                                           columns='hemisphere', 
                                           values='proportion_voxels',
                                           aggfunc='mean').reset_index()

        # Calculate the 'unilaterality'
        df_pivot['unilaterality of glm localizer'] = ( (df_pivot['right'] - df_pivot['left']) ).abs()/((df_pivot['right'] + df_pivot['left']))

        # Merging the 'unilaterality' back to the original dataframe
        results = pd.merge(results, df_pivot[group_columns + ['unilaterality of glm localizer']], on=group_columns, how='left')

        ## REORDER the results dataframe so it is plotted correctly

        hemi_mask_ID_dict = {
                            'left lateral':-4,
                            'right lateral':-3,
                            'left ISC':-2,
                            'right ISC':-1,
                            'left MT':0,
                             'right MT':1,
                             'left STS':2,
                             'right STS':3,
                             'left pSTS':2,
                             'left aSTS':3,
                             'right pSTS':4,
                             'right aSTS':5,
                             'left temporal language':6,
                             'left frontal language':7,
                             'right temporal language':8,
                             'right frontal language':9,
                             'left language':10,
                             'right language':11,
                             'left post temporal language':6,
                             'left ant temporal language':7,
                             'right post temporal language':8,
                             'right ant temporal language':9,
                             'left pTemp':6,
                             'left aTemp':7,
                             'right pTemp':8,
                             'right aTemp':9
                                        }
        results['hemi_mask_ID'] = [hemi_mask_ID_dict[x] for x in results.hemi_mask]
        results = results.sort_values(by='hemi_mask_ID',ascending=True)

        #remove repeats
        results = results.drop_duplicates(subset=['subject','hemi_mask','Condition','localizer_contrast_label'])

        print(results)
        params = {
                    'x':       'hemi_mask',
                    'y':       'glm_weight',
                    'hue':      'Condition',
                    'hue_order':[glm for glm in ['interacting pointlights','non-interacting pointlights','intact speech','degraded speech']],
                    'palette':'plasma'  }

        f = sns.catplot(kind="bar",data=results,col=column_group, col_order=col_order, 
            alpha=None,
            # legend=False,
            edgecolor="black",linewidth=1.5,errorbar='se', errcolor="black",
            height=4, aspect=1, sharex=False, **params,facet_kws={'gridspec_kws':{'width_ratios':width_ratios}})
        plt.subplots_adjust(wspace=0.2)

        g = sns.catplot(kind="strip",data=results,col=column_group, col_order=col_order, 
            alpha=1,
            # legend=False,
            edgecolor="black",linewidth=1.5,errorbar='se', dodge=True, jitter=False,
            height=4, aspect=1, sharex=False,**params,facet_kws={'gridspec_kws':{'width_ratios':width_ratios}})
        plt.subplots_adjust(wspace=0.2)
        
        MT = self.MT
        ISC = self.ISC
        if(average_posterior_anterior):
            STS = ['STS']
            language = ['temporal language']
        else:
            STS = self.STS
            language = self.language_ROI_names
        for fig,label in zip([g,f],['point','bar']):
            fig.set_titles("{col_name}")
            for ax_n in fig.axes:
                for ax in ax_n:
                    value = ax.title.get_text()
                    temp = value.split('(')[1]
                    region = temp.split(')')[0]
                    regions = MT if region=='MT' else ISC if region=='ISC' else language if region=='language' else STS
                    #set xlabels to just the ROI name, no hemisphere
                    xticklabels = ax.get_xticklabels()
                    new_xticktexts = [ticklabel.get_text().split(' ')[1] for ticklabel in xticklabels]
                    ax.set_xticks(ax.get_xticks())
                    ax.set_xticklabels(new_xticktexts)
                    # ax.set_ylim((-0.5,3.5))

                    #add left and right labels
                    for text_label,x_position in zip(['left','right'],[0.25,0.75]):
                        y_position = -0.1  # Negative value to place it below axis
                        ax.text(x_position, y_position, text_label, transform=ax.transAxes, ha='center', va='top')

                    temp_data = results.loc[results[column_group]==value,:]
                    independent_variable = 'Condition'
                    pairs = []
                    track = []
                    pvalues = []
                    parametric=True
                    for mask in [hemi+' ' + region for hemi in ['left','right'] for region in regions]:
                        ROI_pvalues = []
                        for model1 in ['interacting pointlights','non-interacting pointlights','intact speech','degraded speech']:
                            for model2 in ['interacting pointlights','non-interacting pointlights','intact speech','degraded speech']:
                                if((model1==model2)):#&(hemi1==hemi2)):
                                    pass 
                                elif( (model1 in ['interacting pointlights','non-interacting pointlights']) & (model2 in ['intact speech','degraded speech']) ):#&(hemi1!=hemi2)):
                                    pass
                                elif( (model2 in ['interacting pointlights','non-interacting pointlights']) & (model1 in ['intact speech','degraded speech']) ):#&(hemi1!=hemi2)):
                                    pass
                                else:
                                    temp = [(mask,model1),(mask,model2)]
                                    save = [mask,model1,model2]
                                    save.sort()
                                    save = '_'.join(save)
                                    if(save not in track):
                                        pairs.append(temp)
                                        track.append(save)
                                        temp = temp_data[ ((temp_data[independent_variable]==model1)|(temp_data[independent_variable]==model2)) & (temp_data['hemi_mask']==mask)]
                                        if(parametric):
                                            model = smf.mixedlm("glm_weight ~ "+independent_variable, data=temp,groups=temp["subject"],missing='drop')
                                            model_fit = model.fit()
                                            pvalue = model_fit.pvalues[1]
                                        else:
                                            x = temp[(temp_data[independent_variable]==model1)].encoding_response.tolist()
                                            y = temp[(temp_data[independent_variable]==model2)].encoding_response.tolist()
                                            result = wilcoxon(x,y,alternative='two-sided',nan_policy='omit')
                                            pvalue = result.pvalue
                                        ROI_pvalues.append(pvalue) #get the p-value of the intercept of the line between the two conditions
                        reject, ROI_pvalues_fdr = fdrcorrection(ROI_pvalues, alpha=0.05, method='n', is_sorted=False) #method is BY for dependence between variables
                        pvalues.extend(ROI_pvalues_fdr)
                    annot = Annotator(ax, pairs, data=temp_data,verbose=False,**params)
                    annot.configure(test=None)#, text_format='star',show_test_name=False,fontsize='x-small',hide_non_significant=False)#,comparisons_correction="Bonferroni")#,loc='outside'
                    annot.set_pvalues(pvalues)
                    annot.annotate()

                    # plot individual lines if point
                    dodge_width=0.2
                    if(label=='point'):
                        first_group = [hemi+' '+ROI for hemi in ['left','right'] for ROI in MT ]
                        second_group = [hemi+' '+ROI for hemi in ['left','right'] for ROI in STS ]
                        third_group = [hemi+' '+ROI for hemi in ['left','right'] for ROI in language]
                        
                        if(STS==language): #if they are the same ROIs, don't need the separate third group of ROIs
                            categories = first_group+second_group
                        else:
                            categories = first_group+second_group+third_group
                        # print(categories)
                        hues = params['hue_order']
                        # print(hues)
                        n_categories = len(categories)
                        n_hues = len(hues)
                        # Calculate the base positions of categories
                        cat_positions = np.arange(len(categories))

                        for subject_id, subject_data in temp_data.groupby('subject'):
                            # print(subject_data)
                            # Plot a line for each category-hue combination
                            for i, category in enumerate(categories):
                                cat_data = subject_data[subject_data['hemi_mask'] == category]
                                # print(category)
                                # print(cat_data)
                                for j, hue in enumerate(hues):
                                    hue_data = cat_data[cat_data[params['hue']] == hue]
                                    
                                    if not hue_data.empty:
                                        # print(hue)
                                        # print(hue_data)
                                        # Calculate the position of this point
                                        new_i = i
                                        if(category not in first_group): #update positions for pSTS and aSTS (need to move left)
                                            new_i = i - len(first_group)
                                            if(category not in second_group):#update positions for language ROIs (need to move even more left)
                                                new_i = i - len(first_group) - len(second_group)
                                        x = cat_positions[new_i] + (j - (n_hues - 1) / 2.0) * dodge_width
                                        y = hue_data[params['y']].values[0] # Replace with your y-value column
                                        # Plot a point (optional, for verification)
                                        # ax.scatter(x, y, color='black', s=10, zorder=3,)
                                        # ax.text(x, y, subject_id, color='black', ha='center', va='center',fontsize=8)
                                        
                                        # If not the first hue, of sets of 2, draw a line from the previous hue
                                        #basically if j is odd, draw line from previous hue
                                        if j % 2 !=0:
                                            prev_y = prev_hue_data[params['y']].values[0]
                                            prev_x = cat_positions[new_i] + (j - 1 - (n_hues - 1) / 2.0) * dodge_width
                                            ax.plot([prev_x, x], [prev_y, y], color='gray', zorder=2)

                                        prev_hue_data = hue_data

        # g.set_axis_labels("posterior               anterior",'average beta weight')
        # g.set(xticklabels=[hemi for mask in regions for hemi in hemis])
            fig.set_axis_labels("","beta weight")
            # fig.set_titles("{col_name} region")
            plt.savefig(os.path.join(self.figure_dir,self.sid+self.enc_file_label+ '_model-' + selection_model+'_'+file_label+'_perc_top_voxels-' + self.perc_top_voxels+'_glm_response_'+label+'.png'),bbox_inches='tight',dpi=300)
            plt.close()

    def plot_enc_response(self,results,average_posterior_anterior,column_group,col_order,width_ratios,response_label,label_dict,hue,loc_name,model_label,file_label,filepath_tag,plot_lines=True,plot_stacked=True,plot_noise_ceiling=False,stats_to_do=None):
        plt.rcParams.update({'font.size': 14,'font.family': 'Arial'})
        if (average_posterior_anterior):
            for prev_mask in self.STS:
                results['mask'].replace(prev_mask, 'STS', inplace=True) 
            for prev_mask in self.language:
                results['mask'].replace(prev_mask, 'temporal_language', inplace=True)
            #average across the masks 
            columns = [ind for ind in results.columns if ind not in ['glm_weight','encoding_reponse']]
            results = pd.pivot_table(data=results,values=['glm_weight','encoding_response'], index = columns, aggfunc="mean").reset_index()
        
        results.replace(self.labels_dict,inplace=True)
        localizer_label_dict = label_dict
        results['localizer_contrast_label'] = [localizer_contrast + ' (' + localizer_label_dict[localizer_contrast]+')' for localizer_contrast in results[loc_name]]
        results['hemi_localizer_contrast_mask'] = [hemi+ ' ' + localizer_contrast +' '+mask for (localizer_contrast,hemi,mask) in zip(results[loc_name],results['hemisphere'],results['mask'])]
        results['hemi_mask'] = [hemi + ' ' + mask for (hemi,mask) in zip(results['hemisphere'],results['mask'])]

        results['Feature Space'] =  [x for x in results[hue]]
        results['Model'] =  [x for x in results[model_label]]
        
        results = results.drop(columns=['Unnamed: 0'])
        # # Pivot the table to get right and left hemisphere responses for each group
        # group_columns = ['subject',loc_name,'mask']
        # df_pivot = results.pivot_table(index=group_columns, 
        #                                    columns='hemisphere', 
        #                                    values='proportion_voxels',
        #                                    aggfunc='mean').reset_index()

        # # Calculate the 'unilaterality'
        # df_pivot['unilaterality of glm localizer'] = ( (df_pivot['right'] - df_pivot['left']) ).abs()/((df_pivot['right'] + df_pivot['left']))

        # # Merging the 'unilaterality' back to the original dataframe
        # results = pd.merge(results, df_pivot[group_columns + ['unilaterality of glm localizer']], on=group_columns, how='left')
        ## REORDER the results dataframe so it is plotted correctly
        hemi_mask_ID_dict = {
                            'left lateral':-4,
                            'right lateral':-3,
                            'left ISC':-2,
                            'right ISC':-1,
                            'left MT':0,
                             'right MT':1,
                             'left STS':2,
                             'right STS':3,
                             'left pSTS':2,
                             'left aSTS':3,
                             'right pSTS':4,
                             'right aSTS':5,
                             'left temporal language':6,
                             'left frontal language':7,
                             'right temporal language':8,
                             'right frontal language':9,
                             'left language':10,
                             'right language':11,
                             'left post temporal language':6,
                             'left ant temporal language':7,
                             'right post temporal language':8,
                             'right ant temporal language':9,
                             'left pTemp':6,
                             'left aTemp':7,
                             'right pTemp':8,
                             'right aTemp':9
                                        }
        results['hemi_mask_ID'] = [hemi_mask_ID_dict[x] for x in results.hemi_mask]
        results = results.sort_values(by='hemi_mask_ID',ascending=True)

        if((response_label=='performance')|(response_label=='ISC')):
            subset = ['subject','hemi_mask','Model']
        else:
            subset = ['subject','hemi_mask','Feature Space','Model','localizer_contrast_label']
        temp_results = results.drop_duplicates(subset=['subject','hemi_mask','Feature Space','Model','localizer_contrast_label'])
    
    
        independent_variable = model_label
        hue = 'Model'
        if((response_label=='unique_variance')|(response_label=='ind_feature_performance')|(response_label=='ind_product_measure')|(response_label=='features_preferred_delay')):
            hue='Feature Space'
            independent_variable = hue
            
        params = {
                    'x':       'hemi_mask',#sagittal',
                    'y':       'encoding_response',
                    # 'order':[hemi+ ' '+ mask for hemi in ['left','right'] for mask in regions],
                    'color':'indianred'}#fcb001' }

        if((response_label=='performance')|(response_label=='ISC')):
            if(len(self.models)>1):
                params['hue']=hue
                params['hue_order']=[self.labels_dict[x] for x in self.models]
                params['palette']=self.colors_dict
            
            cat_for_stat = [self.labels_dict[x] for x in self.models]

        else:
            params['hue']=hue
            params['hue_order']=self.plot_features#self.plot_features#['social','valence','sbert','motion','alexnet']
            params['palette']=self.colors_dict
            # params['color'] = self.colors_dict['social']
            cat_for_stat = self.plot_features#self.plot_features#['social','valence','sbert','motion','alexnet']

        temp_results = temp_results.reset_index()
        # print(temp_results.Model)
        f = sns.catplot(kind="bar",data=temp_results,col=column_group, col_order=col_order, 
            alpha=1,
            # legend=False,
            edgecolor="black",linewidth=2,errorbar='se', errcolor="black",errwidth=2,
            height=4, aspect=1, sharex=False, sharey=True, **params,facet_kws={'gridspec_kws':{'width_ratios':width_ratios}})
        plt.subplots_adjust(wspace=0.2)
        # print(set(temp_results.Model))
        # temp_results = temp_results.drop_duplicates(inplace=True)
        g = sns.catplot(kind="strip",data=temp_results,col=column_group, col_order=col_order, 
            alpha=None,
            legend=False,
            edgecolor="black",linewidth=1.5,errorbar='se', dodge=True, jitter=False,
            height=4, aspect=1, sharex=False,sharey=True,**params,facet_kws={'gridspec_kws':{'width_ratios':width_ratios}})
        plt.subplots_adjust(wspace=0.2)
        
        MT = self.MT
        ISC = self.ISC
        if(average_posterior_anterior):
            STS = ['STS']
            language = ['temporal language']
        else:
            STS = self.STS
            language = self.language_ROI_names
        for fig,label in zip([g,f],['point','bar']):
        
            fig.set_titles("{col_name}")

            localizer_contrast_regions = []
            save_noise_ceiling_means = []
            save_noise_ceiling_sems = []
            for ax_n in fig.axes:
                for ax in ax_n:
                    value = ax.title.get_text()
                    # ax.set_title(value.split(' ()')[0])
                    temp = value.split('(')[1]
                    region = temp.split(')')[0]
                    temp_data = temp_results.loc[temp_results[column_group]==value,:]
                    # print(temp_data)
                    regions = MT if region=='MT' else ISC if region=='ISC' else language if region=='language' else STS#self.STS#['pSTS','aSTS']
                    #set xlabels to just the ROI name, no hemisphere
                    for region in regions:
                        localizer_contrast_regions.append(value.split('(')[0][:-1]+ ' ' + region)

                    xticklabels = ax.get_xticklabels()
                    new_xticktexts = [ticklabel.get_text().split(' ')[1] for ticklabel in xticklabels]
                    ax.set_xticks(ax.get_xticks())
                    ax.set_xticklabels(new_xticktexts)
                    ax.set_title(value.split(' (')[0] + ' regions')
                    if(plot_noise_ceiling):
                        noise_ceiling_means= []
                        noise_ceiling_stds= []
                        for hemi in ['left','right']:
                            for mask in regions:
                                mean_ISC=np.mean(temp_data[(temp_data['hemi_mask']==hemi+' '+mask)].ISC)
                                std_ISC = scipy.stats.sem(temp_data[(temp_data['hemi_mask']==hemi+' '+mask)].ISC,nan_policy='omit')
                                noise_ceiling_means.append(mean_ISC)
                                noise_ceiling_stds.append(std_ISC)

                        save_noise_ceiling_means.append(noise_ceiling_means) #save for stacked plot
                        save_noise_ceiling_sems.append(noise_ceiling_stds) #save for stacked plot
                        
                        bars = ax.patches
                        multiplier = 1.2
                        for bar, mean, std in zip(bars, noise_ceiling_means, noise_ceiling_stds):
                                # Get the center of the bar
                                bar_center = bar.get_x() + bar.get_width()/2

                                # Add the noise ceiling mean line
                                ax.plot([bar_center - bar.get_width()/2*multiplier, bar_center + bar.get_width()/2*multiplier], 
                                        [mean, mean], color='white',linestyle='dotted')

                                # Add the standard deviation shaded area
                                ax.fill_between([bar_center - bar.get_width()/2*multiplier, bar_center + bar.get_width()/2*multiplier], 
                                                mean - std, mean + std, color='thistle', alpha=0.6,linewidth=0)
                
                    # add left and right labels
                    for text_label,x_position in zip(['left','right'],[0.25,0.75]):
                        y_position = -0.1  # Negative value to place it below axis
                        ax.text(x_position, y_position, text_label, transform=ax.transAxes, ha='center', va='top')
                    
                    if(stats_to_do!=None):
                        parametric = False
                        if(stats_to_do=='compare_to_zero'):
                            pvalues = []
                            pairs = []
                            for mask in regions:
                                for hemi in ['left','right']:
                                    ROI_pvalues = []
                                    for feature_name in cat_for_stat:
                                        temp = temp_data[(temp_data[independent_variable]==feature_name)&(temp_data['hemi_mask']==hemi+' '+mask)]
                                        if(parametric):
                                            ## add in 0's to test against 0 <-- is this the correct approach?
                                            temp_copy = temp.copy()
                                            temp_copy['encoding_response'] = [0 for item in temp_copy['encoding_response']]
                                            temp_copy[independent_variable] = ['zero' for item in temp_copy[independent_variable]]
                                            temp = pd.concat([temp_copy,temp])
                                            model = smf.mixedlm("encoding_response ~ "+independent_variable, data=temp,groups=temp["subject"],missing='drop')
                                            model_fit = model.fit()
                                            pvalue = model_fit.pvalues[0]
                                        else:
                                            x = temp.encoding_response.tolist()
                                            result = wilcoxon(x,alternative='greater',nan_policy='omit')
                                            pvalue = result.pvalue
                                        ROI_pvalues.append(pvalue) #get the p-value of the intercept (different from)
                                        if(response_label=='performance'):
                                            pairs.append([(hemi+' '+mask),(hemi+' '+mask)])
                                        else:
                                            pairs.append([(hemi+' '+mask,feature_name),(hemi+' '+mask,feature_name)])
                                    reject, ROI_pvalues_fdr = fdrcorrection(ROI_pvalues, alpha=0.05, method='n', is_sorted=False) #method is BY for dependence between variables
                                    pvalues.extend(ROI_pvalues_fdr)
                        elif(stats_to_do=='compare_features'):
                            track = []
                            pairs = []
                            pvalues = []
                            for mask in regions:
                                for hemi in ['left','right']:
                                    ROI_pvalues = []
                                    for model1 in cat_for_stat:#[feature for feature in cat_for_stat if feature!=value.split(' (')[0]]:
                                        for model2 in cat_for_stat:#[feature for feature in cat_for_stat if feature!=value.split(' (')[0]]:
                                            temp_model1 = model1.split('_')
                                            temp_model2 = model2.split('_')
                                            final  = temp_model1 + temp_model2
                                            final.sort()
                                            final = '_'.join(final)
                                            if((model1==model2)):
                                                pass
                                            # elif( (model1=='SimCLR')&(model2!='SLIP') ):
                                            #     pass
                                            # elif( (model2=='SimCLR')&(model1!='SLIP') ):
                                            #     pass
                                            # elif( (model1=='SLIPtext')&(model2!='word2vec+GPT2') ):
                                            #     pass
                                            # elif( (model2=='SLIPtext')&(model1!='word2vec+GPT2') ):
                                            #     pass
                                            # elif( (model1=='SLIP')&(model2=='word2vec+GPT2') ):
                                            #     pass
                                            # elif( (model2=='SLIP')&(model1=='word2vec+GPT2') ):
                                            #     pass
                                            # elif ( (model1=='SimCLR_attention')&(model2!='SimCLR_embedding') ):
                                            #     pass
                                            # elif ( (model1=='SimCLR_embedding')&(model2!='SimCLR_attention') ):
                                            #     pass
                                            # elif ( (model1=='SLIP_attention')&(model2!='SLIP_embedding') ):
                                            #     pass
                                            # elif ( (model1=='SLIP_embedding')&(model2!='SLIP_attention') ):
                                            #     pass
                                            # elif( (model1=='SimCLR')&((model2=='+SLIPtext')| (model2=='+word2vec+GPT2') )):
                                            #     pass
                                            # elif( (model1=='+SLIP')&((model2=='+word2vec+GPT2') )):
                                            #     pass
                                            # elif( (model1=='+SLIPtext')&((model2=='+SimCLR'))):
                                            #     pass
                                            # elif( (model1=='+word2vec+GPT2')&((model2=='SimCLR')| (model2=='+SLIP') )):
                                            #     pass
                                            # elif( (model2=='SimCLR')&((model1=='+SLIPtext')| (model1=='+word2vec+GPT2') )):
                                            #     pass
                                            # elif( (model2=='+SLIP')&((model1=='+word2vec+GPT2') )):
                                            #     pass
                                            # elif( (model2=='+SLIPtext')&((model1=='+SimCLR'))):
                                            #     pass
                                            # elif( (model2=='+word2vec+GPT2')&((model1=='SimCLR')| (model1=='+SLIP') )):
                                            #     pass
                                            # elif( (model1=='ISC')|((model2=='ISC') )):
                                            #     pass
                                            else:
                                                if((response_label=='performance')&(len(self.models)==1)):
                                                    temp = [(hemi+' '+mask),(hemi+' '+mask)]
                                                else:
                                                    temp = [(hemi+' '+mask,model1),(hemi+' '+mask,model2)]
                                                save = [mask,model1,model2,hemi]
                                                save.sort()
                                                save = '_'.join(save)
                                                if(save not in track):
                                                    pairs.append(temp)
                                                    # print(pairs)
                                                    track.append(save)
                                                    temp = temp_data[ ((temp_data[independent_variable]==model1)|(temp_data[independent_variable]==model2)) & (temp_data['hemi_mask']==hemi+' '+mask)]
                                                    if(parametric):
                                                        model = smf.mixedlm("encoding_response ~ "+independent_variable, data=temp,groups=temp["subject"],missing='drop')
                                                        model_fit = model.fit()
                                                        pvalue = pvalues[1]
                                                    else:
                                                        x = temp[(temp[independent_variable]==model1)].encoding_response.tolist()
                                                        y = temp[(temp[independent_variable]==model2)].encoding_response.tolist()
                                                        result = wilcoxon(x,y,alternative='two-sided',nan_policy='omit')
                                                        pvalue = result.pvalue

                                                    ROI_pvalues.append(pvalue) #get the p-value
                                    reject, ROI_pvalues_fdr = fdrcorrection(ROI_pvalues, alpha=0.05, method='n', is_sorted=False) #method is BY for dependence between variables
                                    pvalues.extend(ROI_pvalues_fdr)
                        annot = Annotator(ax, pairs, data=temp_data,verbose=False,**params)
                        annot.configure(test=None, text_format='star',show_test_name=False,fontsize='x-small',hide_non_significant=False)#,comparisons_correction="Bonferroni")#,loc='outside'
                        annot.set_pvalues(pvalues)
                        annot.annotate()

                    #plot individual lines if point
                    if(plot_lines):
                        dodge_width=0.4
                        if(label=='point'):
                            first_group = [hemi+' '+ROI for hemi in ['left','right'] for ROI in MT ] #self.MT
                            second_group = [hemi+' '+ROI for hemi in ['left','right'] for ROI in STS ]
                            third_group = [hemi+' '+ROI for hemi in ['left','right'] for ROI in language]

                            if(STS==language): #if they are the same ROIs, don't need the separate third group of ROIs
                                categories = first_group+second_group
                            else:
                                categories = first_group+second_group+third_group
                            # print(categories)
                            hues = cat_for_stat
                            # print(hues)
                            n_categories = len(categories)
                            n_hues = len(hues)
                            # Calculate the base positions of categories
                            cat_positions = np.arange(len(categories))

                            for subject_id, subject_data in temp_data.groupby('subject'):
                                # print(subject_data)
                                # Plot a line for each category-hue combination
                                for i, category in enumerate(categories):
                                    cat_data = subject_data[subject_data['hemi_mask'] == category]
                                    for j, hue_ in enumerate(hues):
                                        hue_data = cat_data[cat_data[hue] == hue_]
                                        # print(hue_data)
                                        if not hue_data.empty:
                                            # Calculate the position of this point
                                            new_i = i
                                            if(category not in first_group): #update positions for pSTS and aSTS (need to move left)
                                                new_i = i - len(first_group)
                                                if(category not in second_group):#update positions for language ROIs (need to move even more left)
                                                    new_i = i - len(first_group) - len(second_group)
                                            x = cat_positions[new_i] + (j - (n_hues - 1) / 2.0) * dodge_width
                                            y = hue_data[params['y']].values[0] # Replace with your y-value column
                                            # Plot a point (optional, for verification)
                                            # ax.scatter(x, y, color='black', s=10, zorder=3,)
                                            # ax.text(x, y, subject_id, color='black', ha='center', va='center',fontsize=8)
                                            # If not the first hue, draw a line from the previous hue
                                            if j > 0:
                                                prev_y = prev_hue_data[params['y']].values[0]
                                                prev_x = cat_positions[new_i] + (j - 1 - (n_hues - 1) / 2.0) * dodge_width
                                                ax.plot([prev_x, x], [prev_y, y], color='gray', zorder=2)

                                            prev_hue_data = hue_data


            y_label = 'explained variance $R^2$'
            if(response_label=='features_preferred_delay'):
                y_label = 'preferred delay (TR)'
            elif(self.scale_by==''):
                y_label = 'explained variance $R^2$'
            elif(self.scale_by=='total_variance'):
                y_label = 'proportion of total $R^2$'
            elif(self.scale_by=='noise_ceiling'):
                y_label = 'proportion of noise ceiling'
            fig.set_axis_labels("",y_label)

            plt.savefig(os.path.join(self.figure_dir,self.sid+self.enc_file_label+ '_model-' + self.model+'_'+file_label+'_perc_top_voxels-' + self.perc_top_voxels+'_enc_response-'+response_label+'_'+label+'_'+filepath_tag+'.png'),bbox_inches='tight',dpi=300)
            plt.close()

        ## STACKED BAR PLOT ########
        #calculate the average for each feature before plotting
        if(plot_stacked):
            plt.rcParams.update({'font.size': 8,'font.family': 'Arial'})

            palette = {}
            for feature_name in self.feature_names:
                palette[feature_name] = self.colors_dict[feature_name]

            
            localizers = [col.split(' (')[0] for col in col_order]
            width = 6#7
            length = len(localizers)*2#4
            # localizers = [localizer.split(' (')[0] for col in col_order]
            from matplotlib import gridspec
            fig = plt.figure(figsize = [length,width])
            gs = gridspec.GridSpec(1, len(localizers), width_ratios=width_ratios) 
            axes = [plt.subplot(axis) for axis in gs]
            localizer_masks = self.localizer_masks
            orders = []
            for localizer in localizers:
                if(average_posterior_anterior):
                    if(localizer_masks[localizer]==self.STS):
                        localizer_masks[localizer] = ['STS']
                    elif(localizer_masks[localizer]==self.language_ROI_names):
                        localizer_masks[localizer] = ['temporal language']
                orders.append([hemi + ' '+ localizer+ ' '+region.replace('_',' ') for hemi in ['left','right'] for region in self.localizer_masks[localizer]])
            
                
            if(response_label=='ind_product_measure'):
                averaged_results = pd.pivot_table(data=temp_results,values=['encoding_response'], columns=[hue,'hemi_localizer_contrast_mask'],aggfunc='mean')
                averaged_results = pd.melt(averaged_results) #'value' is now the average
                for ind,(ax,order,localizer) in enumerate(zip(axes,orders,localizers)):
                    temp_averaged_results = averaged_results.copy()
                    
                    temp_averaged_results.hemi_localizer_contrast_mask = pd.Categorical(values=temp_averaged_results.hemi_localizer_contrast_mask, categories=order) ## Set the order for the column as you want
                    
                    features = [x for x in self.feature_names]# if x!=localizer]
                    features.reverse() #reverse the order for plotting

                    # print(temp_averaged_results_noise)
                    if(response_label !='performance'):
                        plot_pos_features = []
                        for mask in order:
                            # print(mask)
                            for feature_name in features:
                                result = stats.ttest_1samp(results[(results['hemi_localizer_contrast_mask']==mask)&(results[hue]==feature_name)].encoding_response,0)
                                mean = np.mean(results[(results['hemi_localizer_contrast_mask']==mask)&(results[hue]==feature_name)].encoding_response)
                                if(mean>0):
                                    plot_pos_features.append(feature_name)

                        features = [feature for feature in features if feature in plot_pos_features]
                        p = sns.histplot(data=temp_averaged_results, x='hemi_localizer_contrast_mask', hue=hue,hue_order=features,palette=palette,multiple='stack',weights='value',shrink=0.8,ax=ax,stat='count',linewidth=1.75,alpha=1)
                        if(plot_noise_ceiling):
                            bar_widths = []
                            bar_positions = []
                            for patch in p.patches:
                                bar_width = patch.get_width()
                                bar_position = patch.get_x()
                                bar_widths.append(bar_width)
                                bar_positions.append(bar_position)
                            unique_bar_widths = list(set(bar_widths))

                            # Iterate over each bin and add the noise ceiling lines and shaded areas
                            for ind_,(mean,sem,bar_x) in enumerate(zip(save_noise_ceiling_means[ind],save_noise_ceiling_sems[ind],bar_positions)):

                                # Get the center of the bin
                                # bin_centers = np.arange(len(save_noise_ceiling_means[ind])) * unique_bar_widths[0] + unique_bar_widths[0] / 2
                                bin_center = bar_x + unique_bar_widths[0] / 2
                                multiplier = 1.2

                                # Add the noise ceiling mean line
                                ax.plot([bin_center - (unique_bar_widths[0] / 2)*multiplier, bin_center + (unique_bar_widths[0] / 2)*multiplier], 
                                        [mean, mean], color='white', linestyle='dotted')

                                # Add the standard deviation shaded area
                                ax.fill_between([bin_center - (unique_bar_widths[0] / 2)*multiplier, bin_center + (unique_bar_widths[0] / 2)*multiplier], 
                                                mean - sem, mean + sem, color='black', alpha=0.3)
                            
                        ax.set_title(localizer)
                        if(ind==len(localizers)-1):
                            # ax.set_title('motion')
                            legend_elements = [Patch(facecolor=palette[item], edgecolor='k',
                                                     label=item) for item in palette.keys()]
                            ax.legend(handles=legend_elements, loc='center')
                            sns.move_legend(ax, loc='center left', bbox_to_anchor=(1, 0.5))
                            ax.yaxis.set_visible(False)

                        elif(ind!=0):
                            p.legend_.remove()
                            ax.yaxis.set_visible(False)
                        else:
                            # pass
                            p.legend_.remove()

                        ax.set_ylabel('cumulative explained variance $R^2$')
                        # ax.set_ylabel('variance decomposition ')
                        # ax.set_ylabel('proportion of total $R^2$')
                        ax.set_xlabel('')
                        # ax.set_ylim(0,0.3505)
                        ax.spines['top'].set_visible(False)
                        ax.spines['right'].set_visible(False)
                        
                        new_xticktexts = [ticklabel.replace(' language','-language').split(' ')[-1].split('-')[0] for ticklabel in order]
                        ax.set_xticks(ax.get_xticks())
                        ax.set_xticklabels(new_xticktexts)

                        for text_label,x_position in zip(['left','right'],[0.25,0.75]):
                            y_position = -0.06 # Negative value to place it below axis
                            ax.text(x_position, y_position, text_label, transform=ax.transAxes, ha='center', va='top')
                plt.savefig(os.path.join(self.figure_dir,self.sid+self.enc_file_label+ '_model-' + self.model+'_'+file_label+'_perc_top_voxels-' + self.perc_top_voxels+'_enc_response-'+response_label+'_stacked'+'-'+filepath_tag+'.png'),bbox_inches='tight',dpi=300)
                plt.close('all')

        ####### HEMISPHERIC PROPORTION ############
        params = {
                    'y':       'mask',
                    'x':       'proportion_voxels',
                    'hue':      loc_name,
                    'orient':   'h',
                    'palette':  self.colors_dict }
        temp_results = temp_results[temp_results[hue]==set(temp_results[hue]).pop()]
        temp_results = temp_results[temp_results['hemisphere']=='right']
        temp_results = temp_results.sort_values(by='hemi_mask_ID',ascending=False)
        row_order = col_order.copy()
        row_order.reverse()
        g = sns.catplot(kind="swarm",data=temp_results,row=column_group, row_order=row_order, size=6,
            edgecolor="black",linewidth=1.25,dodge=False,legend=False,
            height=1.5,aspect=3,**params,sharey=False,sharex=True)#facet_kws={'gridspec_kws':{'width_ratios':width_ratios}})
        plt.subplots_adjust(wspace=0.1)
        g.set_titles("{row_name}")
                #PLOT LINES FOR EACH SUBJECT

        plot_lines = False
        hue = params['hue']
        for ax_n in g.axes:
                for ax in ax_n:
                    value = ax.title.get_text()
                    # ax.set_title(value.split(' ()')[0])
                    temp = value.split('(')[1]
                    region = temp.split(')')[0]
                    ##draw a horizontal line at 0.5 to show what bilateral would look like
                    ax.axvspan(xmin=0.5,xmax=1,alpha=0.4,facecolor='gray',ec='black',lw=2)
                    ax.axvspan(xmin=0,xmax=0.5,alpha=0.1,facecolor='gray',ec='black',lw=2)
                    
                    ax.set_title(value.split(' (')[0])
                    temp_data = temp_results.loc[temp_results[column_group]==value,:]
                    sns.boxplot(data=temp_data,x='proportion_voxels',y='mask',ax=ax,saturation=0,color='white')

                    if (plot_lines):
                        temp_data = temp_results.loc[temp_results[column_group]==value,:]
                        dodge_width=0.4
                        first_group = self.MT
                        second_group = self.STS
                        third_group = self.language_ROI_names

                        if(self.STS==self.language): #if they are the same ROIs, don't need the separate third group of ROIs
                            categories = first_group+second_group
                        else:
                            categories = first_group+second_group+third_group
                        # print(categories)
                        hues = params['hue_order']
                        # print(hues)
                        n_categories = len(categories)
                        n_hues = len(hues)
                        # Calculate the base positions of categories
                        cat_positions = np.arange(len(categories))

                        for subject_id, subject_data in temp_data.groupby('subject'):
                            # print(subject_data)
                            # Plot a line for each category-hue combination
                            for i, category in enumerate(categories):
                                cat_data = subject_data[subject_data['mask'] == category]
                                for j, hue_ in enumerate(hues):
                                    hue_data = cat_data[cat_data[hue] == hue_]
                                    # print(hue_data)
                                    if not hue_data.empty:
                                        # Calculate the position of this point
                                        new_i = i
                                        if(category not in first_group): #update positions for pSTS and aSTS (need to move left)
                                            new_i = i - len(first_group)
                                            if(category not in second_group):#update positions for language ROIs (need to move even more left)
                                                new_i = i - len(first_group) - len(second_group)
                                        x = cat_positions[new_i] + (j - (n_hues - 1) / 2.0) * dodge_width
                                        y = hue_data[params['y']].values[0] # Replace with your y-value column
                                        # Plot a point (optional, for verification)
                                        # ax.scatter(x, y, color='black', s=10, zorder=3,)
                                        # ax.text(x, y, subject_id, color='black', ha='center', va='center',fontsize=8)
                                        # If not the first hue, draw a line from the previous hue
                                        if j > 0:
                                            prev_y = prev_hue_data[params['y']].values[0]
                                            prev_x = cat_positions[new_i] + (j - 1 - (n_hues - 1) / 2.0) * dodge_width
                                            ax.plot([prev_x, x], [prev_y, y], color='gray', zorder=2)

                                        prev_hue_data = hue_data

        g.set_axis_labels("hemispheric distribution of the most selective voxels\n (more in left $\longleftrightarrow$ more in right)","")
        g.set(xlim=(-0.02,1.02))
        plt.savefig(self.figure_dir + '/'+self.sid+self.enc_file_label+ '_model-' + self.model+'_'+file_label+'_perc_top_voxels-' + self.perc_top_voxels+'_hemi_proportion.png',bbox_inches='tight',dpi=300)
        plt.close('all')
        
        results = temp_results.copy()
        results = results[results['mask']!='MT']

        # Custom function to add correlation values
        def corrfunc(x, y, **kws):
            nas = np.logical_or(np.isnan(x), np.isnan(y))
            corr,p = scipy.stats.pearsonr(x[~nas],y[~nas])
            ax = plt.gca()
            ax.annotate(f'r = {corr:.2f}', xy=(0.1, 0.9), xycoords=ax.transAxes)
            ax.annotate(f'p = {p:.2f}', xy=(0.1, 0.84), xycoords=ax.transAxes)
        def remove_axes(x,y,**kws):
            ax = plt.gca()
            ax.set(xticks=[], yticks=[])
            ax.spines[['top', 'right', 'left', 'bottom']].set_visible(False) 
        results = pd.pivot_table(data=results,values='proportion_voxels',columns=['mask'],index=['subject'])
        # results = results[['pSTS','pTemp','aSTS','aTemp']]
        g = sns.PairGrid(results)
        g.map_lower(sns.scatterplot)
        g.map_lower(corrfunc)
        g.map_upper(remove_axes)
        # g.map_diag(remove_axes_diag)
        plt.savefig(self.figure_dir + '/'+self.sid+self.enc_file_label+ '_model-' + self.model+'_'+file_label+'_perc_top_voxels-' + self.perc_top_voxels+'_hemi_proportion_correlations.png',bbox_inches='tight',dpi=300)
        plt.close()
        # comparisons = [('pSTS','pTemp')]
        # for (region1,region2) in comparisons:
        #     ax = sns.scatterplot(data=results,x=region1,y=region2)
        #     x =np.array(results[region1])
        #     y =np.array(results[region2])
        #     nas = np.logical_or(np.isnan(x), np.isnan(y))
        #     r = scipy.stats.pearsonr(x[~nas],y[~nas])
        #     print(r)
        #     plt.show()



    def MDS(self,load=False,plot_ind=True,response_label='weights'):
        print('MDS')
        fsaverage = nilearn.datasets.fetch_surf_fsaverage(mesh='fsaverage')

        if(response_label=='weights'):
            response_folder = 'weights'
            resp_label = 'weights'
        elif(response_label=='ind_feature_performance'):
            response_folder = 'ind_feature_performance'
            resp_label = 'ind_perf'

        models = ['SimCLR','SLIP','CLIP']
        if(not load):
            params_list = []

            groups = ['glm-SIpointlights','glm-language']
            for group in groups:
                loc_model = group.split('-')[0] 
                if(loc_model=='enc'):
                    task='SIpointlights'
                    feature = group.split('-')[1]
                    localizer_contrasts = ''
                else:
                    task = group.split('-')[1]
                    feature = ''
                    localizer_contrasts = self.localizer_contrasts[task]
                
                for subject in self.subjects[task]:
                    for hemi in ['left','right']:
                        for model in models:
                            if(loc_model == 'enc'):
                                params_list.append((model,subject,'',feature,hemi))
                            elif(loc_model == 'glm'):
                                for localizer_contrast in localizer_contrasts:
                                    params_list.append((loc_model,subject,task,localizer_contrast,hemi,model))


            def process(loc_model,subject,task,specifics,hemi,model):
                enc_file_label = '_encoding_model-'+model + '_smoothingfwhm-'+str(self.smoothing_fwhm)+'_chunklen-'+str(self.chunklen)
                if(self.mask_name!=None):
                    enc_file_label = enc_file_label + '_mask-STS'
                response_filename = self.enc_dir + '/'+response_folder+'/'+subject+enc_file_label+'_measure-'+response_label+'_raw.nii.gz'
                response_nii = nibabel.load(response_filename)
                response_data = response_nii.get_fdata()
                if(loc_model=='glm'):
                    file_label = subject+self.glm_file_label#+'_encoding_model-'+self.model
                    localize_run = self.all_runs[task]
                    localizer_contrast = specifics
                    #load the binary localizer map for glm
                    localizer_filename = self.out_dir + '/localizer_masks/'+file_label+'_glm_loc-'+localizer_contrast+'_run-all_binary.nii.gz'
                elif(loc_model=='enc'):
                    file_label = subject+self.enc_file_label
                    feature = specifics
                    localizer_filename = self.out_dir + '/localizer_masks/'+file_label+'_measure-unique_var_enc_feature_loc-'+feature+'_binary.nii.gz'

                localizer_nii = nibabel.load(localizer_filename)
                localizer_data = localizer_nii.get_fdata()

                response_data[:,localizer_data==0] = np.nan #take out non-localizer values
                
                fullway = localizer_data.shape[0]
                halfway = int(fullway/2)

                if(hemi=='left'):
                    response_data=response_data[:,0:halfway]
                    localizer_data = localizer_data[0:halfway]
                elif(hemi=='right'):
                    response_data=response_data[:,halfway+1:fullway]
                    localizer_data = localizer_data[halfway+1:fullway]

                responses = response_data[:,localizer_data==1] #get selected voxels responses
                if(responses.shape[1]>0):
                    avg_responses = np.nanmean(responses,axis=1)
                else:
                    avg_responses = np.empty((responses.shape[0]))
                    avg_responses[:] = np.nan
                    subject='remove'
                subject_specifics = subject + '_' +specifics
                hemi_specifics = hemi+'_'+specifics
                subject_hemi_specifics = subject+'_'+hemi_specifics
                return (subject, loc_model, task, model, specifics, avg_responses,hemi,hemi_specifics,subject_specifics,subject_hemi_specifics)
            
            results = Parallel(n_jobs=-1)(delayed(process)(loc_model,subject,task,specifics,hemi,model) for (loc_model,subject,task,specifics,hemi,model) in params_list)
            results = np.array(results)
            results = pd.DataFrame(results,columns = ['subject', 'loc_model', 'task', 'model', 'specifics', 'avg_responses','hemi','hemi_specifics','subject_specifics','subject_hemi_specifics'])

            #remove nans
            results = results[results.subject!='remove']

            # results.to_csv(self.out_dir + '/'+'model-'+self.model+'_subject_level_glm_localizer_univariate_analysis.csv')
            results.to_csv(self.out_dir + '/'+self.sid+self.enc_file_label+'_perc_top_voxels-' + self.perc_top_voxels+'_subject_level_MDS_labels.csv')
            avgs_np = np.stack(np.array(results.avg_responses),axis=1).T
            avgs_df = pd.DataFrame(avgs_np)
            # avgs_np.tofile(self.out_dir + '/'+self.sid+self.enc_file_label+'_subject_level_MDS_input_measure-'+response_label+'.csv', sep = ',')
            avgs_df.to_csv(self.out_dir + '/'+self.sid+self.enc_file_label+'_subject_level_MDS_input_measure-'+response_label+'_perc_top_voxels-' + self.perc_top_voxels+'.csv')

        results = pd.read_csv(self.out_dir + '/'+self.sid+self.enc_file_label+'_perc_top_voxels-' + self.perc_top_voxels+'_subject_level_MDS_labels.csv')

        results.replace(self.labels_dict,inplace=True)


        X = pd.read_csv(self.out_dir + '/'+self.sid+self.enc_file_label+'_subject_level_MDS_input_measure-'+response_label+'_perc_top_voxels-' + self.perc_top_voxels+'.csv')
        mds = MDS(n_components=2, random_state=0,n_jobs=-1) 
        X_transformed = mds.fit_transform(X)
        print(X_transformed.shape)

        results['MDS_1'] = X_transformed[:,0]
        results['MDS_2'] = X_transformed[:,1]

        for hue in ['subject','specifics','hemi','hemi_specifics','subject_specifics','subject_hemi_specifics','model']:
            ax = sns.scatterplot(data=results,x='MDS_1',y='MDS_2',hue=hue,style='hemi')
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            # ax.set_ylim((-150,200))
            # ax.set_xlim((-150,260))
            plt.savefig(self.figure_dir + '/MDS/'+self.sid+self.enc_file_label+'_perc_top_voxels-' + self.perc_top_voxels+'_subject_level_MDS_analysis_hue-'+hue+'.pdf',bbox_inches='tight')
            plt.close()

        for model in['glm']:
            X_temp = X[results.loc_model==loc_model]
            mds = MDS(n_components=2, random_state=0,n_jobs=-1) 
            X_transformed = mds.fit_transform(X_temp)
            results_temp = results[results.loc_model==loc_model]
            results_temp['MDS_1'] = X_transformed[:,0]
            results_temp['MDS_2'] = X_transformed[:,1]

            ax=sns.scatterplot(data=results_temp,x='MDS_1',y='MDS_2',hue='specifics',style='hemi')
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            plt.savefig(self.figure_dir + '/MDS/'+self.sid+self.enc_file_label+'_perc_top_voxels-' + self.perc_top_voxels+'_subject_level_MDS_analysis_data-'+model+'_hue-specifics.pdf',bbox_inches='tight')
            plt.close()

        specifics = set(results.specifics)
        for group in specifics:
            X_temp = X[results.specifics==group]
            mds = MDS(n_components=2, random_state=0,n_jobs=-1) 
            X_transformed = mds.fit_transform(X_temp)
            results_temp = results[results.specifics==group]
            results_temp['MDS_1'] = X_transformed[:,0]
            results_temp['MDS_2'] = X_transformed[:,1]

            ax=sns.scatterplot(data=results_temp,x='MDS_1',y='MDS_2',hue='subject',style='hemi')
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            plt.savefig(self.figure_dir + '/MDS/'+self.sid+self.enc_file_label+'_perc_top_voxels-' + self.perc_top_voxels+'_subject_level_MDS_analysis_data-'+group+'_hue-subject.pdf',bbox_inches='tight')
            plt.close()

    def generate_binary_all_significant_maps_glm(self,plot=True,pvalue=0.05,mask=None,glm_task='SIpointlights'):

        glm_file_label = '_smoothingfwhm-'+str(self.smoothing_fwhm)

        if(mask!=None):
            glm_file_label = glm_file_label + '_mask-'+mask
            self.load_mask(mask)

        for localizer_contrast in self.localizer_contrasts[glm_task]:
            all_subjects = np.zeros(self.brain_shape) #collect all for group map
            for subject in self.subjects[glm_task]:
                subject_map = np.zeros(self.brain_shape)
                # for localize_run,response_run in self.run_groups: 
                for localize_run in self.all_runs[glm_task]:
                    #get filepath of pvalue nifti
                    file_label = subject+self.glm_file_label#+'_encoding_model-'+self.model
                    filepath = self.glm_dir + '/'+ subject+'/'+subject + "_task-"+ glm_task+'_space-'+self.space+ "_run-"+ localize_run + "_contrast-"+localizer_contrast+ "_measure-zscore.nii.gz"
                    zscore_map_img = nibabel.load(filepath)
                    zscore_map = zscore_map_img.get_fdata()
                    p_filepath = self.glm_dir + '/'+ subject+'/'+subject + "_task-"+ glm_task+'_space-'+self.space+ "_run-"+ localize_run + "_contrast-"+localizer_contrast+ "_measure-pvalue.nii.gz"
                    p_map_img = nibabel.load(p_filepath)
                    p_map = p_map_img.get_fdata()
                    mask = nilearn.image.resample_img(self.mask, target_affine=zscore_map_img.affine, target_shape=zscore_map.shape,interpolation='nearest').get_fdata()
                    zscore_map[mask==0] = 0
                    zscore_map[p_map>pvalue]= 0
                    subject_map = zscore_map
                subject_map_img = nibabel.Nifti1Image(subject_map,zscore_map_img.affine)
                if(plot):
                    map_filename = self.figure_dir + '/all_significant_voxels/'+subject+glm_file_label+'_glm_loc-'+localizer_contrast+'_run-all_sig-'+str(pvalue)+'.pdf'
                    helpers.plot_img_volume(subject_map_img,map_filename,threshold=0, cmap='YlOrBr')#,cmap='YlOrBr')
                subject_map = (subject_map>0)*1.0 #binarize to save to compare to enc localizer
                map_img_filename = self.out_dir + '/all_significant_voxels/'+subject+glm_file_label+'_glm_loc-'+localizer_contrast+'_run-all_sig-'+str(pvalue)+'.nii.gz'
                binary_map_img = nibabel.Nifti1Image(subject_map,zscore_map_img.affine)
                nibabel.save(binary_map_img,map_img_filename)

                all_subjects = all_subjects+subject_map
            file_label = self.sid+glm_file_label#'sub-all_encoding_model-'+self.model
            all_subjects_map_img = nibabel.Nifti1Image(all_subjects,zscore_map_img.affine)
            if(plot):
                map_filename = self.figure_dir + '/all_significant_voxels/'+file_label+'_glm_loc-'+localizer_contrast+'_run-all_sig-'+str(pvalue)+'.pdf'
                helpers.plot_img_volume(all_subjects_map_img,map_filename,threshold=0,vmax=len(self.subjects[glm_task]),cmap='YlGn')

    def generate_binary_localizer_maps_glm(self,plot=True,glm_task='SIpointlights'):
        print('generating binary glm voxel selection maps:')
        glm_file_label = '_smoothingfwhm-'+str(self.smoothing_fwhm)
        
        for localizer_contrast in self.localizer_contrasts[glm_task]:
            for subject in tqdm(self.subjects[glm_task],desc=localizer_contrast):
                combined_masks = np.zeros(self.brain_shape)
                for mask in self.localizer_masks[localizer_contrast]:
                    subject_map = np.zeros(self.brain_shape)
                    file_label = subject+glm_file_label+'_mask-'+mask#+'_encoding_model-'+self.model

                    filepath = self.out_dir + '/localizer_masks/'+file_label+'_glm_loc-'+localizer_contrast+'_run-'+self.all_runs[glm_task]+'.nii.gz'
                    localizer_map_img = nibabel.load(filepath)
                    localizer_map = localizer_map_img.get_fdata()
                    subject_map = subject_map+(localizer_map>0)*1.0
                    subject_map_img = nibabel.Nifti1Image(subject_map,localizer_map_img.affine)
                    # if(plot):
                    #   map_filename = self.figure_dir + '/localizer_masks/'+file_label+'_glm_loc-'+localizer_contrast+'_run-all_binary.pdf'
                    #   helpers.plot_map(subject_map_img,map_filename,threshold=0,vmax=len(self.run_groups[glm_task]),cmap='Greens')
                    subject_map = (subject_map>0)*1.0 #binarize to save to compare to enc localizer
                    map_img_filename = self.out_dir + '/localizer_masks/'+file_label+'_glm_loc-'+localizer_contrast+'_run-all_binary.nii.gz'
                    binary_map_img = nibabel.Nifti1Image(subject_map,localizer_map_img.affine)
                    nibabel.save(binary_map_img,map_img_filename)

                    combined_masks = combined_masks+subject_map
                
                file_label = subject+glm_file_label+'_mask-'+'_'.join(self.localizer_masks[localizer_contrast])#'sub-all_encoding_model-'+self.model
                combined_masks_map_img = nibabel.Nifti1Image(combined_masks,localizer_map_img.affine)
                map_img_filename = self.out_dir + '/localizer_masks/'+file_label+'_glm_loc-'+localizer_contrast+'_run-all_binary.nii.gz'
                nibabel.save(combined_masks_map_img,map_img_filename)
                if(plot):
                    color = self.colors_dict[localizer_contrast]
                    cmap = LinearSegmentedColormap.from_list('my_gradient', ((0.000, color),(1.000, color)))
                    map_filename = self.figure_dir + '/localizer_masks/'+file_label+'_glm_loc-'+localizer_contrast+'_run-all_binary'
                    helpers.plot_surface(nii=combined_masks_map_img,filename=map_filename,ROI_niis=[combined_masks_map_img],ROIs=[localizer_contrast],ROI_colors=['white'],views=['lateral'],threshold=0,vmax=None,cmap=cmap)

    def generate_binary_all_significant_maps_enc(self,plot=True,pvalue=0.05,mask=None,label='ind_feature_performance',glm_task='SIpointlights'):
        enc_file_label = '_encoding_model-'+self.model + '_smoothingfwhm-'+str(self.smoothing_fwhm)+'_chunklen-'+str(self.chunklen)

        if(mask!=None):
            enc_file_label = enc_file_label + '_mask-'+mask
            self.load_mask(mask)
        if(label=='performance'):
            features = [self.model]
            measure_label = 'perf'
        if(label=='ind_feature_performance'):
            features = self.features_loc
            measure_label = 'ind_perf'
        if(label=='ind_product_measure'):
            features = self.features_loc
            measure_label = 'ind_product_measure'

        elif(label=='unique_variance'):
            features = ['social','glove','DNN']
            measure_label = 'unique_var'

        elif(label=='shared_variance'):
            features = ['social','glove','DNN']
            measure_label = 'shared_var'
            

        for feature_name in features:
            all_subjects = np.zeros(self.brain_shape) #collect all for group map
            for subject in self.subjects[glm_task]:
                
                #get filepath
                file_label = subject+self.enc_file_label#'_encoding_model-'+self.model
                if((label=='unique_variance')|(label=='shared_variance')):
                    file_label = file_label+'_feature-'+feature_name
                filepath = self.enc_dir+'/'+label+'/'+file_label+'_measure-'+measure_label+'_raw.nii.gz'
                perf_map_img = nibabel.load(filepath)
                perf_map = perf_map_img.get_fdata()
                p_filepath = self.enc_dir+'/'+label+'/'+file_label+'_measure-'+measure_label+'_p_fdr.nii.gz'
                p_map_img = nibabel.load(p_filepath)
                p_map = p_map_img.get_fdata()
                if(label=='ind_feature_performance'):
                    feature_index = self.get_feature_index(subject, feature_name)
                    p_map = p_map[feature_index]
                    perf_map = perf_map[feature_index]


                mask = nilearn.image.resample_img(self.mask, target_affine=perf_map_img.affine, target_shape=perf_map.shape,interpolation='nearest').get_fdata()

                perf_map[mask==0] = 0
                perf_map[p_map>pvalue] = 0
                subject_map=perf_map
                subject_map_img = nibabel.Nifti1Image(subject_map,perf_map_img.affine)
                if(plot):
                    map_filename = self.figure_dir + '/all_significant_voxels/'+subject+enc_file_label+'_measure-'+measure_label+'_enc_feature_loc-'+feature_name+'_sig-'+str(pvalue)+'.pdf'
                    helpers.plot_img_volume(subject_map_img,map_filename,threshold=0,cmap='YlOrBr')
                subject_map = (perf_map>0)*1.0 
                map_img_filename = self.out_dir + '/all_significant_voxels/'+subject+enc_file_label+'_measure-'+measure_label+'_enc_feature_loc-'+feature_name+'_sig-'+str(pvalue)+'.nii.gz'
                binary_map_img = nibabel.Nifti1Image(subject_map,perf_map_img.affine)
                nibabel.save(binary_map_img,map_img_filename)

                all_subjects = all_subjects+subject_map
            file_label = self.sid+enc_file_label+'_measure-'+measure_label#'sub-all_encoding_model-'+self.model
            all_subjects_map_img = nibabel.Nifti1Image(all_subjects,perf_map_img.affine)
            if(plot):
                map_filename = self.figure_dir + '/all_significant_voxels/'+file_label+'_enc_feature_loc-'+feature_name+'_sig-'+str(pvalue)+'.pdf'
                helpers.plot_img_volume(all_subjects_map_img,map_filename,threshold=0,vmax=len(self.subjects['SIpointlights']),cmap='YlGn')

    def generate_binary_localizer_maps_enc(self,model=None,plot=True,label='ind_feature_performance'):
        print('generating binary maps of voxels with high '+label+' in the encoding model...')
        enc_file_label = '_encoding_model-'+model+'_smoothingfwhm-'+str(self.smoothing_fwhm)+'_chunklen-'+str(self.chunklen)
        print(label)
        if(label=='performance'):
            features_loc = [self.model]
            measure_label = 'perf_raw'
        if(label=='ind_feature_performance'):
            measure_label = 'ind_perf_raw'
            features_loc = self.plot_features_dict[model]#feature_names#['motion','alexnet','social','face','valence']
        elif(label=='ind_product_measure'):
            measure_label = 'ind_product_measure_raw'
            features_loc = self.plot_features_dict[model]
        elif(label=='unique_variance'):
            measure_label = 'unique_var'
            features_loc = self.plot_features_dict[model]#feature_names
        elif(label=='shared_variance'):
            measure_label = 'shared_var'
            features_loc = ['glove-social','DNN-social','DNN-glove']#'speaking','glove','DNN']

        for feature_name in features_loc:
            for subject in tqdm(self.subjects['SIpointlights'],desc=feature_name):
                combined_masks = np.zeros(self.brain_shape)
                for mask in self.localizer_masks[feature_name]:
                    #get filepath
                    file_label = subject+enc_file_label+'_mask-'+mask+'_measure-'+measure_label#'_encoding_model-'+self.model
                    filepath = self.out_dir + '/localizer_masks/'+file_label+'_enc_feature_loc-'+feature_name+'.nii.gz'
                    localizer_map_img = nibabel.load(filepath)
                    localizer_map = localizer_map_img.get_fdata()
                    subject_map= localizer_map
                    subject_map_img = nibabel.Nifti1Image(subject_map,localizer_map_img.affine)
                    # if(plot):
                    #   map_filename = self.figure_dir + '/localizer_masks/'+file_label+'_enc_feature_loc-'+feature_name+'.pdf'
                    #   helpers.plot_img_volume(subject_map_img,map_filename,threshold=0,vmax=1,cmap='Greens')
                    subject_map = (localizer_map>0)*1.0 
                    map_img_filename = self.out_dir + '/localizer_masks/'+file_label+'_enc_feature_loc-'+feature_name+'_binary.nii.gz'
                    binary_map_img = nibabel.Nifti1Image(subject_map,localizer_map_img.affine)
                    nibabel.save(binary_map_img,map_img_filename)
                    combined_masks = combined_masks+subject_map

                file_label = subject+enc_file_label+'_mask-'+'_'.join(self.localizer_masks[feature_name])+'_measure-'+measure_label#'sub-all_encoding_model-'+self.model
                combined_masks_map_img = nibabel.Nifti1Image(combined_masks,localizer_map_img.affine)
                map_img_filename = self.out_dir + '/localizer_masks/'+file_label+'_enc_feature_loc-'+feature_name+'_binary.nii.gz'
                nibabel.save(combined_masks_map_img,map_img_filename)
                if(plot):
                    color = self.colors_dict[feature_name]
                    cmap = LinearSegmentedColormap.from_list('my_gradient', ((0.000, color),(1.000, color)))
                    map_filename = self.figure_dir + '/localizer_masks/'+file_label+'_enc_feature_loc-'+feature_name+'_binary'
                    helpers.plot_surface(nii=combined_masks_map_img,filename=map_filename,ROI_niis=[combined_masks_map_img],ROIs=[feature_name],ROI_colors=['black'],views=['lateral'],threshold=0,vmax=None,cmap=cmap)

    def compute_response_similarity(self,load=False,plot=True,selection_type='top_percent',pvalue=None,response='weights',axes=[]):#or 'predicted_time_series' or 'weights_raw'):
        #TODO
        print('computing response similarity of voxel groups:')
        if(selection_type=='top_percent'):
            suffix = '_binary'
            folder = '/localizer_masks/'
            overlap_folder = '/localizer_overlap_maps/'
        elif(selection_type=='all_significant'):
            suffix = '_sig-'+str(pvalue)
            folder = '/all_significant_voxels/'
            overlap_folder = '/all_significant_overlap_maps/'

        localizer_contrasts_social = ['interact-no_interact']#,'interact&no_interact']
        localizer_contrasts_lang = ['intact-degraded']
        localizer_contrasts_motion = ['interact&no_interact']
        # ind_perf_features = ['social','sbert','alexnet','valence','motion','face']
        # unique_var_features = ['social','sbert','alexnet','valence','motion','face'] #'speaking'

        # encoding_ind_perf_names = [(name,'encoding-ind_product_measure') for name in ind_perf_features]
        # encoding_unique_var_names = [(name,'encoding-unique_var') for name in unique_var_features]
        localizer_names_social = [(name,'glm-SIpointlights-'+ROI) for name in localizer_contrasts_social for ROI in self.localizer_masks['SI pointlights']]
        localizer_names_lang = [(name,'glm-language-'+ROI) for name in localizer_contrasts_lang for ROI in self.localizer_masks['language']]
        localizer_names_motion = [(name,'glm-motionpointlights-'+ROI) for name in localizer_contrasts_motion for ROI in self.localizer_masks['motion pointlights']]
        # names = encoding_ind_perf_names + encoding_unique_var_names + localizer_names_social + localizer_names_lang
        # names = encoding_unique_var_names + localizer_names_social + localizer_names_lang
        # names = [localizer_names_social[0]] + [encoding_unique_var_names[0]]+[encoding_unique_var_names[1]]+[encoding_unique_var_names[2]] +localizer_names_lang + [encoding_unique_var_names[3]]+[localizer_names_social[1]] + [encoding_unique_var_names[4]]
        names = localizer_names_motion+localizer_names_social+localizer_names_lang

        selected_features = []
        for selections in [[(name1,name1_model_type),(name2,name2_model_type),(hemi1,hemi2)] for (name1,name1_model_type) in names for (name2,name2_model_type) in names for hemi1 in ['left','right'] for hemi2 in ['left','right']]:
            selections[0:1].sort()
            if(selections[2] != ('right','left')):
                selected_features.append((selections[0],selections[1],selections[2]))

        selected_features = list(set(selected_features))
        selected_features.sort()

        # axes = ['language-vision']#'social-language','auditory-visual','social-general','language-social','abstract-perceptual']#['auditory-visual','social-language','social-general']

        #load the feature space info (what slices each feature's weights are located at in the weights file)
        parser = argparse.ArgumentParser()
        parser.add_argument('--s_num', '-s', type=str, default='1')
        parser.add_argument('--task','-task',type=str,default='sherlock')
        parser.add_argument('--space','-space',type=str, default='MNI152NLin2009cAsym_res-2') #'native'
        parser.add_argument('--mask','-mask',type=str, default='ISC')
        parser.add_argument('--smoothing-fwhm','-smoothing-fwhm',type=float,default=3.0) #gallant lab doesn't do any smoothing for individuals
        parser.add_argument('--chunklen','-chunklen',type=int,default=30)
        parser.add_argument('--model','-model',type=str, default=self.model)
        parser.add_argument('--testing','-testing',type=str,default=False) 
        parser.add_argument('--dir', '-dir', type=str,
                            default='/Users/hsmall2/Documents/GitHub/deep_nat_lat')
        parser.add_argument('--data_dir', '-data_dir', type=str,
                            default='/Users/hsmall2/Documents/GitHub/Sherlock_ASD/data') #where the derivatives folder for the fMRI data is 
        parser.add_argument('--out_dir', '-output', type=str,
                            default='/Users/hsmall2/Documents/GitHub/deep_nat_lat/analysis')
        parser.add_argument('--figure_dir', '-figures', type=str,
                            default='/Users/hsmall2/Documents/GitHub/deep_nat_lat/figures')
        args = parser.parse_args()
        encoding_obj = encoding.EncodingModel(args)

        features_n_list = []
        for feature_space in encoding_obj.feature_names:
            # if(feature_space.split('_')[0] != 'run'):
            filepath = self.dir + '/features/'+encoding_obj.features_dict[feature_space].lower()+'.csv'
            data = np.array(pd.read_csv(filepath,header=None))
            features_n_list.append(data.shape[1])
        start_and_end = np.concatenate([[0], np.cumsum(features_n_list)])
        slices = [slice(start, end) for start, end in zip(start_and_end[:-1], start_and_end[1:])]
        feature_index_dict = dict(zip(encoding_obj.feature_names, slices))
        print(feature_index_dict)
        if(not load):
            def process(subject,mask1_name,mask1_name_type,mask2_name,mask2_name_type,hemi1,hemi2,axis):
                def get_info(subject,mask1_filepath,mask2_filepath,mask1_name,mask2_name,name1_measure_label,name2_measure_label,mask1_ROI,mask2_ROI,data_filepath,hemi1,hemi2,response,axis):
                    
                    # if(axis=='abstract-perceptual'):
                    #   features1 = ['word2vec','sbert','social','mentalization','valence','arousal']
                    #   features2 = ['alexnet','hue','pixel','amplitude','pitch','motion','music','indoor_outdoor','num_agents']
                    # if(axis=='social-language'):
                    #   features1 = ['social','valence','arousal','mentalization','num_agents','face']
                    #   features2 = ['sbert','word2vec','turn_taking','speaking']
                    # if(axis=='language-social'):
                    #   features1 = ['sbert','word2vec','turn_taking','speaking']
                    #   features2 = ['social','valence','arousal','mentalization','num_agents','face']
                    # if(axis=='social-general'):
                    #   features1 = ['social','face','num_agents','speaking','turn_taking','mentalization']
                    #   features2 = ['alexnet','hue','pixel','amplitude','pitch','music','indoor_outdoor','valence','arousal','word2vec','sbert','motion']
                    # if(axis=='auditory-visual'):
                    #   features1 = ['amplitude','pitch','speaking','turn_taking','music']
                    #   features2 = ['hue','pixel','indoor_outdoor','motion','num_agents','face','indoor_outdoor']
                    # if(axis=='language-vision'):
                    #   features1 = ['sbert','word2vec']
                    #   features2=['alexnet','motion']
                    features1=[axis]
                    features2=[axis]

                    try:
                        mask1_img = nibabel.load(mask1_filepath,mmap=True)
                        mask2_img = nibabel.load(mask2_filepath,mmap=True)

                        mask1 = mask1_img.get_fdata()>0
                        mask2 = mask2_img.get_fdata()>0

                        data_img = nibabel.load(data_filepath,mmap=True) #using memory mapping to save RAM
                        data = data_img.get_fdata()

                        fullway = data.shape[1]
                        halfway = int(fullway/2)

                        if(hemi1=='left'):
                            mask1[halfway+1:fullway] = 0 #zero out right side
                       
                        elif(hemi1=='right'):
                            mask1[0:halfway]=0 #zero out left side

                        if(hemi2=='left'):
                            mask2[halfway+1:fullway] = 0 #zero out right side

                        elif(hemi2=='right'):
                            mask2[0:halfway]=0 #zero out left side

                        data1 = np.nanmean(data[:, mask1],axis=1)
                        data2 = np.nanmean(data[:, mask2],axis=1)

                        features1_indices = []
                        for feature in features1:
                            if('weights' in data_filepath):
                                indices = range(data.shape[0])
                                feature_slice = feature_index_dict[feature]
                                features1_indices.extend(indices[feature_slice])
                            else:
                                feature_index = self.get_feature_index(subject,feature)
                                features1_indices.extend([feature_index])
                        features2_indices = []
                        for feature in features2:
                            if('weights' in data_filepath):
                                indices = range(data.shape[0])
                                feature_slice = feature_index_dict[feature]
                                features2_indices.extend(indices[feature_slice])
                            else:
                                feature_index = self.get_feature_index(subject,feature)
                                features2_indices.extend([feature_index])
                        
                        if len(data1) > 0:# and len(data2) > 0:
                            corr1_selected_data1 = data1[features1_indices]
                            corr1_selected_data2 = data2[features1_indices]
                            # corr2_selected_data1 = data1[features2_indices]
                            # corr2_selected_data2 = data2[features2_indices]

                            nas1 = np.logical_or(np.isnan(corr1_selected_data1), np.isnan(corr1_selected_data2))
                            # nas2 = np.logical_or(np.isnan(corr2_selected_data1), np.isnan(corr2_selected_data2))

                            corr1_results = scipy.stats.pearsonr(corr1_selected_data1[~nas1], corr1_selected_data2[~nas1])
                            # corr2_results = scipy.stats.pearsonr(corr2_selected_data1[~nas2], corr2_selected_data2[~nas2])

                            print(corr1_results)
                            corr1 = corr1_results[0]
                            pvalue1 = corr1_results[1]
                            # corr2 = corr2_results[0]
                            # pvalue2 = corr2_results[1]
                            corr = (corr1**2)*(corr1/np.abs(corr1)) #- corr2 #get the variance explained between these, maintaining the sign
                            corr2 = pvalue2 = np.nan
                        else:
                            corr = corr1 = corr2 = pvalue1 = pvalue2 = np.nan

                        hemi_label = hemi1+'_'+hemi2
                    except Exception as e:
                        print(f"Exception occurred: {e}")
                        return (np.nan,) * 14
                    
                    mask1_name = mask1_name+'-'+mask1_ROI
                    mask2_name = mask2_name+'-'+mask2_ROI
                    results = (subject,axis,mask1_name,mask2_name,name1_measure_label,name2_measure_label,hemi1,hemi2,hemi_label,corr,corr1,corr2,pvalue1,pvalue2)
                    print(results)
                    return results

                enc_file_label = subject+self.enc_file_label
                glm_file_label = subject+'_smoothingfwhm-'+str(self.smoothing_fwhm)

                mask1_name_type_split = mask1_name_type.split('-')
                mask1_name_model = mask1_name_type_split[0]
                mask1_name_measure_label = mask1_name_type_split[1]
                mask1_ROI = mask1_name_type_split[2]

                mask2_name_type_split = mask2_name_type.split('-')
                mask2_name_model = mask2_name_type_split[0]
                mask2_name_measure_label = mask2_name_type_split[1]
                mask2_ROI = mask2_name_type_split[2]

                if(mask1_name_model=='encoding'):
                    mask1_filepath = self.out_dir + folder+enc_file_label+'_measure-'+mask1_name_measure_label+'_enc_feature_loc-'+mask1_name+suffix+'_raw.nii.gz'
                elif(mask1_name_model=='glm'):
                    mask1_filepath = self.out_dir + folder+glm_file_label+'_mask-'+mask1_ROI+'_glm_loc-'+mask1_name+'_run-all'+suffix+'.nii.gz'

                if(mask2_name_model=='encoding'):
                    mask2_filepath = self.out_dir + folder+enc_file_label+'_measure-'+mask2_name_measure_label+'_enc_feature_loc-'+mask2_name+suffix+'_raw.nii.gz'
                elif(mask2_name_model=='glm'):
                    mask2_filepath = self.out_dir + folder+glm_file_label+'_mask-'+mask2_ROI+'_glm_loc-'+mask2_name+'_run-all'+suffix+'.nii.gz'

                if(response=='weights'):
                    data_filepath = self.enc_dir + '/'+response+'/'+subject+'_encoding_model-'+self.model+self.enc_file_label+'_measure-'+response+'_raw.nii.gz'
                elif(response=='ind_product_measure'):
                    data_filepath = self.enc_dir + '/'+response+'/'+subject+'_encoding_model-'+self.model+self.enc_file_label+'_measure-ind_product_measure_raw.nii.gz'
                else:
                    data_filepath = self.enc_dir + '/'+response+'/'+subject+self.enc_file_label+'_measure-'+response+'.nii.gz'

                results = get_info(subject,mask1_filepath,mask2_filepath, mask1_name,mask2_name,mask1_name_measure_label,mask2_name_measure_label,mask1_ROI,mask2_ROI,data_filepath,hemi1,hemi2,response,axis)
                # print(subject+'_mask1-'+mask1_name+'_measure-'+mask1_name_measure_label+'_mask2-'+mask2_name+'_measure-'+mask2_name_measure_label)
                print(results)
                return results

            # for (feature_name1,feature_name2,localizer_contrast1,localizer_contrast2) in selected_features
            # for subject in self.subjects:
            results = Parallel(n_jobs=3)(delayed(process)(subject,mask1_name,mask1_name_type,mask2_name,mask2_name_type,hemi1,hemi2,axis) 
                for subject in tqdm(self.subjects['SIpointlights']) 
                for ((mask1_name,mask1_name_type),(mask2_name,mask2_name_type),(hemi1,hemi2)) in selected_features 
                for axis in axes)
            # results = pd.DataFrame(zip(subjects,localizer_contrasts,num_voxels_glm,proportion_glm,overlap_types),columns=['subject','localizer_contrast','glm_num_voxels','proportion_glm','hemi'])
            results = np.array(results)
            results = pd.DataFrame(results,columns = ['subject', 'axis','mask1_name','mask2_name','mask1_measure_label','mask2_measure_label','hemi1','hemi2','hemi_label','corr','corr1','corr2','pvalue1','pvalue2'])
            file_label = self.sid+self.enc_file_label+'_'+response+'_response_similarity'#'sub-all_encoding_model-'+self.model
            #delete all repeat rows and save
            results.to_csv(self.out_dir+'/'+file_label+'_perc_top_voxels-' + self.perc_top_voxels+'_enc_feature_localizer_results.csv')
        #for each subject select voxels and average their time series data, then correlate that across hemispheres and across regions
        #average all individual matrices together
        file_label = self.sid+self.enc_file_label+'_'+response+'_response_similarity'#'sub-all_encoding_model-'+self.model

        results=pd.read_csv(self.out_dir+'/'+file_label+'_perc_top_voxels-' + self.perc_top_voxels+'_enc_feature_localizer_results.csv')
        if(self.remove_right_lateralized_subjects):
            results = self.remove_subjects(results, self.right_lateralized_subjects)

        
        # results.replace(self.labels_dict,inplace=True)

        def get_similarity_matrix(names,axis):
            similarity_matrix = np.zeros((len(names)*2,len(names)*2))
            # similarity_matrix = np.zeros((len(names),len(names)))
            print(similarity_matrix)
            print(names)
            # _names = [hemi+':'+self.labels_dict[name[0]]+'-'+name[1].split('-')[2] for hemi in ['left','right'] for name in names]
            _names = [hemi+':'+name[0]+'-'+name[1].split('-')[2] for hemi in ['left','right'] for name in names]
            names = [self.labels_dict[name[0]] for hemi in ['left','right'] for name in names]#,'right'] for name in names]
            print(names)

            for (ind1,label1) in enumerate(_names):
                for (ind2,label2) in enumerate(_names): 
                    name1 = label1.split(':')[1] 
                    hemi1 = label1.split(':')[0]
                    name2 = label2.split(':')[1] 
                    hemi2 = label2.split(':')[0]

                    if((hemi1=='right') & (hemi2=='left')):
                        hemi1 = 'left'
                        hemi2= 'right'

                    print(results)
                    print(name1)
                    temp_results = results[(results.axis==axis)&(results.hemi1==hemi1)&(results.mask1_name==name1)&(results.hemi2==hemi2)&(results.mask2_name==name2)]
                    value1 = np.nanmean(temp_results['corr'])
                    value2 = np.nanmean(temp_results['corr'])
                    print(label1+', '+label2+'--'+str(ind1)+', '+str(ind2)+': '+str(value1))
                    print(temp_results)
                    similarity_matrix[ind1][ind2]=value1
                    similarity_matrix[ind2][ind1]=value2
                    

            print(similarity_matrix)
            similarity_matrix[np.triu_indices(similarity_matrix.shape[0],1)] = np.nan
            fig = plt.figure(figsize=(15,15))
            ax = fig.add_subplot(111)
            # cax = ax.imshow(similarity_matrix,cmap='Greens',vmin=0)
            cax = ax.imshow(similarity_matrix,cmap='coolwarm',vmin=-1,vmax=1)
            sns.despine(left=True,bottom=True)
            fig.colorbar(cax)
            labels = _names
            print(labels)
            ax.set_xticks(range(0,len(_names)))
            ax.set_yticks(range(0,len(_names)))
            ax.set_xticklabels(labels)
            ax.set_yticklabels(labels)
            plt.xticks(rotation=90)
            plt.title(axis)

            # plt.rcParams.update({'font.size': 9})

            print(similarity_matrix.shape)

            for i in range(len(_names)):
                for j in range(len(_names)):
                    c = similarity_matrix[j,i]
                    if(~np.isnan(c)):
                        ax.text(i, j, str(np.round(c,2)), va='center', ha='center')

            plt.savefig(self.figure_dir + '/response_similarity/'+file_label+'_perc_top_voxels-' + self.perc_top_voxels+'-'+axis+'_similarity_matrix.pdf',bbox_inches='tight')
            plt.close()

            return similarity_matrix,_names

        # names = [localizer_names_social[1]]+[localizer_names_social[0]] + localizer_names_lang

        # similarity_matrix = get_similarity_matrix(names,'abstract-perceptual')
        # similarity_matrix = get_similarity_matrix(names,'social-language')
        # similarity_matrix = get_similarity_matrix(names,'language-social')
        # similarity_matrix = get_similarity_matrix(names,'auditory-visual')'
        sim_matrices = []
        for axis in axes:
            similarity_matrix,_names = get_similarity_matrix(names,axis)
            sim_matrices.append(similarity_matrix)

        #get the difference between them
        similarity_matrix = sim_matrices[1]-sim_matrices[0]
        fig = plt.figure(figsize=(15,15))
        ax = fig.add_subplot(111)
        # cax = ax.imshow(similarity_matrix,cmap='Greens',vmin=0)
        cax = ax.imshow(similarity_matrix,cmap=self.cmaps['teal_orange'],vmin=-0.5,vmax=0.5)
        sns.despine(left=True,bottom=True)
        fig.colorbar(cax)
        labels = _names
        print(labels)
        ax.set_xticks(range(0,len(_names)))
        ax.set_yticks(range(0,len(_names)))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        plt.xticks(rotation=90)
        plt.title(axes[1]+'-'+axes[0])

        # plt.rcParams.update({'font.size': 9})

        print(similarity_matrix.shape)

        for i in range(len(_names)):
            for j in range(len(_names)):
                c = similarity_matrix[j,i]
                if(~np.isnan(c)):
                    ax.text(i, j, str(np.round(c,2)), va='center', ha='center')

        plt.savefig(self.figure_dir + '/response_similarity/'+file_label+'_perc_top_voxels-' + self.perc_top_voxels+'_diff-'+axes[1]+'-'+axes[0]+'_similarity_matrix.pdf',bbox_inches='tight')
        plt.close()

        temp_results = results[results.hemi_label=='left_right']
        temp_results['name_label']=temp_results.mask1_name+'_'+temp_results.mask2_name
        temp_results = temp_results[temp_results.mask1_name==temp_results.mask2_name]

        order = ['interact&no_interact-MT','interact-no_interact-pSTS','interact-no_interact-aSTS','intact-degraded-temporal_language','intact-degraded-frontal_language']#['SI pointlights','language']
        barplot=True
        barplot_ylim = (-0.25,0.6)

        plt.rcParams.update({'font.size': 10})
        if(barplot):
            fig = plt.subplots(figsize=(13,5))
            ax =sns.barplot(data=temp_results,x='mask1_name',y='corr',hue='axis',order=order,errorbar='se',palette=self.colors_dict)
            ax.set_ylim(barplot_ylim)
            ax.set_ylabel('signed squared correlation')
            plt.savefig(self.figure_dir + '/response_similarity/'+file_label+'_perc_top_voxels-' + self.perc_top_voxels+'_across_hemi_barplot.pdf',bbox_inches='tight')

        # names = localizer_names_social + localizer_names_lang + [encoding_unique_var_names[0]] +[encoding_unique_var_names[2]]+[encoding_unique_var_names[3]]+[encoding_unique_var_names[4]]
        hue_order = [name[0]+'-'+name[1].split('-')[2] for name in names]#[self.labels_dict[name[0]] for name in names]
        cmap = None#[self.colors_dict[self.labels_dict[name[0]]] for name in names]
        
        for name in names:
            temp_results = results[(results.hemi_label=='left_left')]
            temp_results = temp_results[(temp_results.mask1_name==name[0]+'-'+name[1].split('-')[2])]#|(temp_results.mask2_name==self.labels_dict[name[0]])]
            temp_results['name_label']=temp_results.mask1_name+'_'+temp_results.mask2_name
            # temp_results['name_label']=temp_results.name1+'_'+temp_results.name2
            temp_results = temp_results[temp_results.mask1_name!=temp_results.mask2_name]
            print(temp_results.mask2_name)
            
            if(barplot):
                fig = plt.subplots(figsize=(13,5))
                temp_order = [item for item in order if item in temp_results.mask2_name.tolist()]
                ax =sns.barplot(data=temp_results,x='mask2_name',y='corr',hue='axis',order=temp_order,errorbar='se',palette=self.colors_dict)
                plt.title(name[0]+'-'+name[1].split('-')[2]+', left')
                ax.set_ylim(barplot_ylim)
                ax.set_ylabel('signed squared correlation')
                # plt.xticks(rotation=90)
                plt.savefig(self.figure_dir + '/response_similarity/'+file_label+'_perc_top_voxels-' + self.perc_top_voxels+'_'+name[0]+'-'+name[1].split('-')[2]+'_within_left_barplot.pdf',bbox_inches='tight')

            # data1 = temp_results[temp_results.axis=='abstract-perceptual']
            # data2 = temp_results[temp_results.axis=='social-general']['corr'].tolist()

            data1 = temp_results[temp_results.axis=='SimCLR']
            data2 = temp_results[temp_results.axis=='SLIP']['corr'].tolist()

            temp_data = data1
            axis1_name = 'SimCLR'#'perceptual--->abstract'
            axis2_name = 'SLIP'#general--->social'
            

            temp_data[axis1_name] = data1['corr']
            temp_data[axis2_name] = data2

            fig = plt.subplots(figsize=(10,10))
            ax = sns.lmplot(data=temp_data,x=axis1_name,y=axis2_name,hue='mask2_name',hue_order=hue_order,x_ci=None,ci=0,palette=cmap)
            plt.title('response similarity with '+name[0]+'-'+name[1].split('-')[2]+', left')
            plt.savefig(self.figure_dir + '/response_similarity/'+file_label+'_perc_top_voxels-' + self.perc_top_voxels+'_'+name[0]+'-'+name[1].split('-')[2]+'_within_left_scatterplot.pdf',bbox_inches='tight')
            plt.close()

            temp_results = results[results.hemi_label=='right_right']
            temp_results = temp_results[(temp_results.mask1_name==name[0]+'-'+name[1].split('-')[2])]#|(temp_results.mask2_name==self.labels_dict[name[0]])]
            temp_results['name_label']=temp_results.mask1_name+'_'+temp_results.mask2_name
            # temp_results['name_label']=temp_results.name1+'_'+temp_results.name2
            temp_results = temp_results[temp_results.mask1_name!=temp_results.mask2_name]   
            
            if(barplot):
                fig = plt.subplots(figsize=(13,5))
                temp_order = [item for item in order if item in temp_results.mask2_name.tolist()]
                ax =sns.barplot(data=temp_results,x='mask2_name',y='corr',hue='axis',order=temp_order,errorbar='se',palette=self.colors_dict)
                plt.title(name[0]+'-'+name[1].split('-')[2]+', right')
                ax.set_ylim(barplot_ylim)
                ax.set_ylabel('signed squared correlation')
                # plt.xticks(rotation=90)
                plt.savefig(self.figure_dir + '/response_similarity/'+file_label+'_perc_top_voxels-' + self.perc_top_voxels+'_'+name[0]+'-'+name[1].split('-')[2]+'_within_right_barplot.pdf',bbox_inches='tight')
                plt.close()

            # data1 = temp_results[temp_results.axis=='abstract-perceptual']
            # data2 = temp_results[temp_results.axis=='social-general']['corr'].tolist()
            data1 = temp_results[temp_results.axis=='SimCLR']
            data2 = temp_results[temp_results.axis=='SLIP']['corr'].tolist()


            temp_data = data1
            axis1_name = 'SimCLR'#'perceptual--->abstract'
            axis2_name = 'SLIP'#'general--->social'

            temp_data[axis1_name] = data1['corr']
            temp_data[axis2_name] = data2

            fig = plt.subplots(figsize=(10,10))
            ax = sns.lmplot(data=temp_data,x=axis1_name,y=axis2_name,hue='mask2_name',hue_order=hue_order,x_ci=None,ci=0,palette=cmap)
            plt.title('response similarity with '+name[0]+'-'+name[1].split('-')[2]+', right')
            plt.savefig(self.figure_dir + '/response_similarity/'+file_label+'_perc_top_voxels-' + self.perc_top_voxels+'_'+name[0]+'-'+name[1].split('-')[2]+'_within_right_scatterplot.pdf',bbox_inches='tight')
            plt.close()

        hue_order = [name[0]+'-'+name[1].split('-')[2] for name in names]#[self.labels_dict[name[0]] for name in names]

        for axis1 in axes:
            for axis2 in axes:
                if(axis1!=axis2):
                    temp_results = results[results.hemi_label=='left_right']
                    temp_results = temp_results[temp_results.mask1_name==temp_results.mask2_name]
                    
                    data1 = temp_results[temp_results.axis==axis1]
                    data2 = temp_results[temp_results.axis==axis2]['corr'].tolist()
                    print(data2)

                    temp_data = data1
                    # axis1_name = axis1.split('-')[1]+'--->'+axis1.split('-')[0]
                    # axis2_name = axis2.split('-')[1]+'--->'+axis2.split('-')[0]
                    axis1_name = axis1
                    axis2_name = axis2

                    temp_data[axis1_name] = data1['corr']
                    temp_data[axis2_name] = data2
                    print(temp_data)
                    print(hue_order)

                    fig = plt.subplots(figsize=(10,10))
                    g = sns.lmplot(data=temp_data,x=axis1_name,y=axis2_name,hue='mask2_name',hue_order=hue_order,x_ci=None,ci=0,palette=cmap)
                    plt.title('response similarity across hemispheres')
                    plt.savefig(self.figure_dir + '/response_similarity/'+file_label+'_perc_top_voxels-' + self.perc_top_voxels+'_'+axis1+'_'+axis2+'_across_hemi_scatterplot.pdf',bbox_inches='tight')
                    plt.close()

    def post_to_ant_line_analysis(self,load,features,measure,width=2,mask='lateral',file_tag=''):
        #right now this is basically a group analysis, to make an individual analysis:
        #for each subject, a linear regression between posterior to anterior axis and the encoding performance/level of abstractness (how to get a measure?), get the beta
        #then average the betas

        # self.load_mask()
        print('extracting posterior to anterior responses:')

        if(measure=='preference'):
            label = 'preference1_map'
            folder_label = 'preference_map'
            # features = self.feature_names
            ylabel = 'proportion of voxels'
        elif(measure=='top_unique_variance'):
            label = 'unique_var_raw'#_raw'
            # label = 'unique_var_p_fdr'
            folder_label = 'localizer_masks'
            # features = self.plot_features
            ylabel = 'proportion of voxels'
        elif(measure=='top_ind_feature_performance'):
            label = 'ind_perf_raw'
            folder_label = 'localizer_masks'
            # features = self.plot_features
            ylabel = 'proportion of voxels'
        elif(measure=='unique_variance'):
            label = 'unique_var_raw'
            # features = self.plot_features
            ylabel = 'average unique variance'
        elif(measure=='shared_variance'):
            label = 'shared_var_raw'
            features = ['glove-social','DNN-social','DNN-glove']
            ylabel = 'average shared variance'
        elif(measure=='ind_feature_performance'):
            label = 'ind_perf_raw'
            # features = self.plot_features
            ylabel = 'average ind feature performance'
        elif(measure=='ind_product_measure'):
            label = 'ind_product_measure_raw'
            # features = self.plot_features
            ylabel = 'average ind product measure'
        elif(measure=='performance'):
            label = 'perf_raw'
            # features = self.models
            ylabel = 'average performance'
        elif(measure=='glm_localizer'):
            label='glm_loc'
            folder_label = 'localizer_masks'#'localizer_masks'
            # features = ['interact-no_interact','interact&no_interact','intact-degraded']
            ylabel = 'proportion of voxels'
        elif(measure=='glm_zscore'):
            label='zscore'
            task = 'SIpointlights'
            # features = ['interact-no_interact']#,'interact&no_interact','intact-degraded']
            ylabel = 'average z-score'

        
        params = []
        for feature_ind, feature in enumerate(features):
            if('top' in measure.split('_')):
                masks = self.localizer_masks[feature]
            else: 
                # masks = [self.mask_name]
                masks = [mask]
            for mask in masks:
                for hemi in ['left','right']:
                    for subject in self.subjects['SIpointlights']:
                        params.append((subject, feature_ind, feature, hemi, mask))

        if(not load):
            def process(subject, feature_ind,feature,hemi,mask):
                enc_file_label = '_encoding_model-'+self.model+'_smoothingfwhm-'+str(self.smoothing_fwhm)+'_chunklen-'+str(self.chunklen)
                if(self.mask_name!=None):
                    enc_file_label = enc_file_label + '_mask-'+self.mask_name
                enc_file_label = subject+enc_file_label
                results_len = []
                if(measure=='preference'):
                    filepath = self.in_dir+'/'+folder_label+'/'+enc_file_label+'_measure-'+label+'.nii.gz'
                elif(measure=='top_unique_variance'):
                    filepath = self.out_dir+'/'+folder_label+'/'+enc_file_label+'_measure-'+label+'_enc_feature_loc-'+feature+'_binary.nii.gz' #add feature
                elif(measure=='top_ind_feature_performance'):
                    filepath = self.out_dir+'/'+folder_label+'/'+enc_file_label+'_measure-'+label+'_enc_feature_loc-'+feature+'_binary.nii.gz' #add feature
                elif(measure=='unique_variance'):
                    filepath = self.enc_dir+'/'+measure+'/'+enc_file_label+'_feature-'+feature+'_measure-'+label+'.nii.gz'
                elif(measure=='shared_variance'):
                    filepath = self.enc_dir+'/'+measure+'/'+enc_file_label+'_feature-'+feature+'_measure-'+label+'.nii.gz'
                elif(measure=='ind_feature_performance'):
                    filepath = self.enc_dir+'/'+measure+'/'+enc_file_label+'_measure-'+label+'.nii.gz'
                elif(measure=='ind_product_measure'):
                    filepath = self.enc_dir+'/'+measure+'/'+enc_file_label+'_measure-'+label+'.nii.gz'
                elif(measure=='performance'):
                    enc_file_label = '_encoding_model-'+feature+'_smoothingfwhm-'+str(self.smoothing_fwhm)+'_chunklen-'+str(self.chunklen)
                    if(self.mask_name!=None):
                        enc_file_label = enc_file_label + '_mask-'+self.mask_name
                    enc_file_label = subject+enc_file_label
                    filepath = self.enc_dir+'/'+measure+'/'+enc_file_label+'_measure-'+label+'.nii.gz'
                elif(measure=='glm_localizer'):
                    filepath = self.out_dir+'/'+folder_label+'/'+subject+self.glm_file_label+'_'+label+'-'+feature +'_run-all_binary.nii.gz' #add feature
                elif(measure=='glm_zscore'):
                    run = '123'
                    task='SIpointlights'
                    if(feature=='intact-degraded'):
                        run='12'
                        task='language'
                    filepath = self.glm_dir + '/'+ subject+'/'+subject + "_task-"+ task+'_space-'+self.space+ "_run-"+run+"_contrast-"+feature+ "_measure-"+label+".nii.gz"


                try:
                    nii = nibabel.load(filepath)
                    nii_data = nii.get_fdata()

                    if(self.scale_by=='total_variance'):
                        nii_data[nii_data<0] = 0 #clip response values to 0
                        nii_data = nii_data/nii_data.sum(axis=0,keepdims=1)
                    
                    if(measure=='preference'):
                        #set all features other than this one to zero and this feature to 1
                        nii_data[nii_data!=feature_ind+1] = 0
                        nii_data[nii_data==feature_ind+1] = 1
                    elif((measure=='ind_feature_performance')|(measure=='ind_product_measure')):
                        if(feature in self.combined_features):
                            for (ind,sub_feature_name) in enumerate(self.model_features_dict[feature]):
                                feature_ind = self.get_feature_index(subject,sub_feature_name)
                                sub_data = nii_data[feature_ind]
                                if(ind==0):
                                    overall = sub_data
                                else:
                                    overall = overall+sub_data
                            nii_data = overall
                        else:
                            feature_index = self.get_feature_index(subject, feature)
                            nii_data = nii_data[feature_index]


                    # if(hemi=='L'):
                    self.load_mask(hemi+'-'+mask)
                        # if((measure=='ind_feature_performance')|(measure=='unique_variance')|(measure=='shared_variance')):
                        #   self.load_mask('left_STS')
                    # if(hemi=='R'):
                    #   self.load_mask('right-'+mask)
                        # if((measure=='ind_feature_performance')|(measure=='unique_variance')|(measure=='shared_variance')):
                        #   self.load_mask('right_STS')
                    self.mask = nilearn.image.resample_img(self.mask, target_affine=nii.affine, target_shape=nii_data.shape,interpolation='nearest')
                    mask_data = self.mask.get_fdata()
                    nii_data[mask_data!=1] = np.nan

                    length = nii_data.shape[1]
                    summary = []
                    Y = []
                    curr_Y = 0
                    while curr_Y<length:
                        data = nii_data[:,curr_Y:curr_Y+width,:]
                        Y.append(curr_Y)
                        summary_ = np.nanmean(data)
                        if(np.isnan(summary_)):
                            summary_ = 0
                        summary.append(summary_)
                        curr_Y = curr_Y + 1
                    hemi = [hemi for x in range(length)]
                    mask = [mask for x in range(length)]
                    subject = [subject for x in range(length)]
                    feature = [feature for x in range(length)]
                except Exception as e:
                    print(subject)
                    print(e)
                    nan_list = [np.nan for x in range(self.brain_shape[1])]
                    return(nan_list,nan_list,nan_list,nan_list,nan_list,nan_list)
                
                return(subject,feature, hemi,mask,Y,summary)

            results = Parallel(n_jobs=-1)(delayed(process)(subject,feature_ind,feature,hemi,mask) for (subject, feature_ind, feature, hemi, mask) in tqdm(params))      
            subjects = []
            features_ = []
            hemis = []
            masks = []
            Ys = []
            summarys = []
            results = np.array(results).T
            # results = results[~np.isnan(results)]
            for ind,result in enumerate(results):
                # if(~(np.isnan(results[ind])).all()):
                    subjects.extend(results[ind][0])
                    features_.extend(results[ind][1])
                    hemis.extend(results[ind][2])
                    masks.extend(results[ind][3])
                    Ys.extend(results[ind][4])
                    summarys.extend(results[ind][5])
            results = pd.DataFrame([subjects,features_,hemis,masks,Ys,summarys])
            results = results.transpose()
            results.dropna(inplace=True)
            results.columns=['subject','feature','hemi','mask','Y','summary']
            results['hemi_mask'] = [hemi+', '+ mask for (hemi,mask) in zip(results['hemi'],results['mask']) ]
            results['hemi_mask_subject'] = [hemi+', '+ mask + ', ' + subject for (hemi,mask,subject) in zip(results['hemi'],results['mask'],results['subject']) ]
            file_label = self.sid+self.enc_file_label +'_'+ self.model+'_'+ measure + '_'+file_tag

            results.to_csv(self.out_dir+'/'+file_label+'_perc_top_voxels-' + self.perc_top_voxels+'_mask-'+mask+'_post_to_ant_line_results.csv')

        file_label = self.sid+self.enc_file_label +'_'+ self.model+'_'+ measure+ '_'+file_tag
        loaded_results = pd.read_csv(self.out_dir+'/'+file_label+'_perc_top_voxels-' + self.perc_top_voxels+'_mask-'+mask+'_post_to_ant_line_results.csv')
        loaded_results.replace(self.labels_dict,inplace=True)

        # list_colors = [self.colors_dict[self.labels_dict[feature]] for feature in features]
        # print(list_colors)
        # cmap = colors.ListedColormap(list_colors)
        # for hemi in ['L','R']:
            # temp_results = results[(results.hemi==hemi)]
        col_order = [hemi+ ', ' + mask for mask in masks for hemi in ['left','right'] ]

        g = sns.relplot(kind='line',estimator='mean',col='hemi_mask',col_order=col_order,col_wrap=2,data=loaded_results,x='Y',y='summary',hue='feature',alpha=0.2,palette=self.colors_dict,linewidth=3,errorbar='se',facet_kws={'sharey': True, 'sharex': True})
        # g = sns.relplot(kind='line',units='subject',estimator=None,col='hemi_mask',col_order=col_order,col_wrap=2,data=results,x='Y',y='summary',hue='feature',alpha=0.2,palette=self.colors_dict,errorbar='se',facet_kws={'sharey': True, 'sharex': True})
        g.set_titles("{col_name}")
        for ax, (name, group_df) in zip(g.axes.flat, loaded_results.groupby('hemi_mask')):
            # Check if the subplot should be skipped based on col_order
            # if name not in col_order:
            #   continue
            # Use seaborn lineplot to add mean lines
            sns.lineplot(data=group_df, x='Y', y='summary', hue='feature', estimator='mean',
                         errorbar=None, ax=ax,palette=self.colors_dict,linewidth=3,err_style='bars',legend=False)

            ax.set_title(name.split(',')[0])
        
        ylabel = ''
        if(measure=='unique_variance'):
            ylabel = 'average unique variance explained'
        if(measure=='performance'):
            ylabel = 'average explained variance $R^2$'
        if(measure=='ind_feature_performance'):
            ylabel = 'average individual feature explained variance $R^2$'
        if(measure=='ind_product_measure'):
            ylabel = 'proportion of total variance explained '
        if(measure=='glm_zscore'):
            ylabel = 'average zscore of interact-no_interact contrast'
        g.set_axis_labels('MNI Y coordinate',ylabel)
        if(mask=='STS'):
            xlim=(20,80)
        elif(mask=='lateral'):
            xlim=(17,68)
            xlim=(10,75)
        elif(mask=='MT'):
            xlim=(10,50)
        elif(mask=='ISC'):
            xlim=None
        g.set(xlim=xlim)
        # g.set(ylim=(0.15,0.4))

        leg = g._legend
        # g.set(ylim=(-0.002, 0.002))
        # ax.set_xlabel('posterior --> anterior')
        # ax.set_ylabel(ylabel +', width = '+ str(width))
        # ax.set_ylim((0,0.4))
        if(measure=='unique_variance'):
            # ax.set_ylim((-0.02, 0.04))
            leg.set_title('Feature Space')
        elif(measure=='performance'):
            leg.set_title('Model')
        elif(measure=='ind_feature_performance'):
            ax.set_ylim((-0.02, 0.04))
        # if(measure=='performance'):
            # ax.set_ylim((0,0.012))
        # ax.set_xlim((10,58))
        
        # ax.set_title(measure+', '+hemi+' hemisphere')
        # ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        # plt.savefig(self.figure_dir + '/'+file_label+'_'+hemi+'_post_to_ant_line.pdf',bbox_inches='tight')
        plt.savefig(self.figure_dir + '/'+file_label+'_mask-'+mask+'_post_to_ant_line.pdf',bbox_inches='tight')

        plt.clf()

        # col_order = [hemi+ ', ' + mask + ', ' +subject for subject in set(results.subject) for hemi in ['left','right'] for mask in [self.mask_name] ]
        # g = sns.relplot(kind='line',col='hemi_mask_subject',col_order = col_order,col_wrap=2,data=results,x='Y',y='summary',hue='feature',alpha=0.6,palette=self.colors_dict,estimator=None,units='subject')
        
        # g.set_titles("{col_name}")
        # ylabel = ''
        # if(measure=='unique_variance'):
        #   ylabel = 'average unique variance explained'
        # if(measure=='performance'):
        #   ylabel = 'average explained variance $R^2$'
        # if(measure=='ind_feature_performance'):
        #   ylabel = 'average individual feature explained variance $R^2$'
        # if(measure=='glm_zscore'):
        #   ylabel = 'average zscore of interact-no_interact contrast'
        # g.set_axis_labels('MNI Y coordinate',ylabel)
        # g.set(xlim=(0, 70))
        # leg = g._legend
        # if(measure=='unique_variance'):
        #   ax.set_ylim((-0.02, 0.04))
        #   leg.set_title('Feature Space')
        # elif(measure=='performance'):
        #   leg.set_title('Model')

        # plt.savefig(self.figure_dir + '/'+file_label+'_post_to_ant_line_ind_sub.pdf',bbox_inches='tight')

        # plt.close()

    def compute_all_overlap2(self,load=False,plot=True,selection_type='top_percent',pvalue=None,regions=[],label='',file_tag=''):
        from matplotlib import colors
        print('computing overlap between voxel groups:')

        if(selection_type=='top_percent'):
            suffix = '_binary'
            folder = '/localizer_masks/'
            overlap_folder = '/localizer_overlap_maps/'

        cmap = colors.ListedColormap(['white','mediumblue','gold','green','white','mediumblue','gold','green'])

        # masks_dict = {'glm STS pointlights':['interact-no_interact'],
        #             'glm STS language':['intact-degraded'],
        #             'glm MT pointlights':['interact&no_interact'],
        #             'glm MT language':'', #No contrast for this
        #             'enc STS': ['social'],#,'word2vec'],#['SimCLR','CLIP','sbert'],#'social','valence','face'],
        #             'enc MT':['motion'],#['motion','alexnet']
        #             'glm ISC pointlights':['interact-no_interact'],
        #             'glm ISC language':['intact-degraded'],
        #             'enc ISC': ['social','sbert'],
        #             'glm lateral pointlights':['interact-no_interact'],
        #             'glm lateral language':['intact-degraded'],
        #             'enc lateral': ['social','sbert','word2vec','motion','alexnet','valence']
        #               }
        # localizer_contrasts_pointlights = masks_dict['glm '+mask+' pointlights']
        # localizer_contrasts_lang = masks_dict['glm '+mask+' language']
        # encoding_features = masks_dict['enc '+mask]

        # masks = [sag+mask for sag in ['p','a']]
        # masks = [mask]

        # encoding_ind_perf_names = [(name,'encoding-ind_product_measure') for name in encoding_features]
        # encoding_unique_var_names = [(name,'encoding-unique_var') for name in encoding_features]
        # localizer_names_pointlights = [(name,'glm-SIpointlights') for name in localizer_contrasts_pointlights]
        # localizer_names_lang = [(name,'glm-language') for name in localizer_contrasts_lang]
        
        # names = encoding_unique_var_names + localizer_names_social + localizer_names_lang
        names = regions


        selected_features = []
        for selections in [[(name1,name1_model_type,model1,mask1),(name2,name2_model_type,model2,mask2)] for (name1,name1_model_type,model1,mask1) in names for (name2,name2_model_type,model2,mask2) in names]:
            selections.sort()
            selected_features.append((selections[0],selections[1]))

        selected_features = list(set(selected_features))
        selected_features.sort()

        if(not load):
            def process(subject,name1,name1_type,model1,mask1,name2,name2_type,model2,mask2,hemi,plot):
                def get_info(subject,data1_filepath,data2_filepath,name1,name2,name1_measure_label,name2_measure_label,model1,model2,name1_model,name2_model,mask1,mask2,hemi,plot_filepath):
                    try:
                        data1_img = nibabel.load(data1_filepath)
                        data2_img = nibabel.load(data2_filepath)

                        data1 = (data1_img.get_fdata()>0)*1.0
                        data2 = (data2_img.get_fdata()>0)*2.0
                        
                        overlap = data1+data2
                        overlap_img = nibabel.Nifti1Image(overlap.astype('int32'),data1_img.affine)
                        if(plot_filepath!='None'):
                            title = name1+': blue, '+name2+': yellow, overlap: green'
                            helpers.plot_img_volume(overlap_img,plot_filepath,threshold=0.99,cmap=cmap,title=title,vmax=3)
                        
                        #### save a map of the overlap only
                        overlap_only = (overlap==3)*1
                        overlap_only_img = nibabel.Nifti1Image(overlap_only.astype('int32'),data1_img.affine)
                        filepath = filepath1.split(subject)[0]+subject+'_overlap-'+name1+'_'+name2+'_binary.nii.gz'
                        nibabel.save(overlap_only_img, filepath)  
                        ####

                        total_voxels1_all = len(data1[data1==1])
                        total_voxels2_all = len(data2[data2==2])
                        total_voxels_all = total_voxels1_all+total_voxels2_all
                        voxels_overlap_all = len(overlap[overlap==3])

                        fullway = overlap.shape[0]
                        halfway = int(fullway/2)

                        if(hemi=='left'):
                            overlap=overlap[0:halfway]
                            data1= data1[0:halfway]
                            data2= data2[0:halfway]
                        elif(hemi=='right'):
                            overlap=overlap[halfway+1:fullway]
                            data1 = data1[halfway+1:fullway]
                            data2 = data2[halfway+1:fullway]

                        total_voxels1 = len(data1[data1==1])
                        total_voxels2 = len(data2[data2==2])
                        total_voxels = total_voxels1+total_voxels2

                        voxels1_only = len(overlap[overlap==1])
                        voxels2_only = len(overlap[overlap==2])
                        voxels_overlap = len(overlap[overlap==3])

                        # voxels1_0 = voxels1_only
                        # voxels2_0 = voxels2_only
                        
                        # if(voxels1_0==0):
                        #   voxels1_0 = np.nan
                        # if(voxels2_0==0):
                        #   voxels2_0 = np.nan

                        if(total_voxels==0):
                            total_voxels = np.nan
                        if(total_voxels_all==0):
                            total_voxels_all = np.nan
                        if(voxels_overlap_all==0):
                            voxels_overlap_all = np.nan

                        DICE_coef = total_voxels and (2*voxels_overlap)/total_voxels
                        proportion_of_voxels1_in_this_hemi = total_voxels1_all and total_voxels1/total_voxels1_all
                        proportion_of_voxels2_in_this_hemi = total_voxels2_all and total_voxels2/total_voxels2_all
                        proportion_of_all_voxels_in_this_hemi = total_voxels_all and total_voxels/total_voxels_all
                        proportion_of_overlap_in_this_hemi = voxels_overlap_all and voxels_overlap/voxels_overlap_all

                        proportion_of_1_that_is_also_2 = total_voxels1 and voxels_overlap/total_voxels1
                        proportion_of_2_that_is_also_1 = total_voxels2 and voxels_overlap/total_voxels2

                        proportion_of_voxels_that_is_voxels1 = total_voxels and voxels1_only/total_voxels
                        proportion_of_voxels_that_is_voxels2 =  total_voxels and voxels2_only/total_voxels
                        proportion_of_voxels_that_is_overlap =  total_voxels and voxels_overlap/total_voxels

                        names = [(name1,name1_measure_label),(name2,name2_measure_label)]
                        names.sort()
                        names_label = names[0][0]+'-'+names[0][1]+'_'+names[1][0]+'-'+names[1][1]
                    except Exception as e:
                        print(e)
                        results = (np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan)
                        return results

                    results = (subject, name1,name2,names_label,name1_measure_label,name2_measure_label,name1_model,name2_model,mask1,mask2,hemi,total_voxels1,total_voxels2,total_voxels,voxels1_only, voxels2_only, voxels_overlap,DICE_coef,
                        proportion_of_voxels1_in_this_hemi,proportion_of_voxels2_in_this_hemi,proportion_of_all_voxels_in_this_hemi,proportion_of_overlap_in_this_hemi, 
                        proportion_of_1_that_is_also_2, proportion_of_2_that_is_also_1,
                        proportion_of_voxels_that_is_voxels1, proportion_of_voxels_that_is_voxels2,proportion_of_voxels_that_is_overlap)
                    return results

                enc_file_label1 = subject+'_encoding_model-'+model1+'_smoothingfwhm-'+str(self.smoothing_fwhm)+'_chunklen-'+str(self.chunklen) #+ '_mask-'+mask_name
                enc_file_label2 = subject+'_encoding_model-'+model2+'_smoothingfwhm-'+str(self.smoothing_fwhm)+'_chunklen-'+str(self.chunklen)
                glm_file_label = subject+'_smoothingfwhm-'+str(self.smoothing_fwhm) #+ '_mask-'+mask_name


                name1_model = name1_type.split('-')[0]
                name1_measure_label = name1_type.split('-')[1]
                name2_model = name2_type.split('-')[0]
                name2_measure_label = name2_type.split('-')[1]

                if(name1_model=='encoding'):
                    filepath1 = self.out_dir + folder+enc_file_label1+ '_mask-'+mask1+'_measure-'+name1_measure_label+'_enc_feature_loc-'+name1+suffix+'.nii.gz'
                elif(name1_model=='glm'):
                    filepath1 = self.out_dir + folder+glm_file_label+ '_mask-'+mask1+'_glm_loc-'+name1+'_run-all'+suffix+'.nii.gz'
                elif(name1_model=='overlap'):
                    filepath1 = self.out_dir + folder+subject+ '_overlap-'+name1+suffix+'.nii.gz'

                if(name2_model=='encoding'):
                    filepath2 = self.out_dir + folder+enc_file_label2+ '_mask-'+mask2+'_measure-'+name2_measure_label+'_enc_feature_loc-'+name2+suffix+'.nii.gz'
                elif(name2_model=='glm'):
                    filepath2 = self.out_dir + folder+glm_file_label+ '_mask-'+mask2+'_glm_loc-'+name2+'_run-all'+suffix+'.nii.gz'
                elif(name2_model=='overlap'):
                    filepath2 = self.out_dir + folder+subject+ '_overlap-'+name2+suffix+'.nii.gz'
                
                if(plot):
                    plot_filepath = self.figure_dir + overlap_folder+self.enc_file_label+'_'+name1_type+'-'+name1+'_'+name2_type+'-'+name2+'_'+selection_type+'.pdf'

                results = get_info(subject,filepath1,filepath2, name1,name2,name1_measure_label,name2_measure_label,model1,model2,name1_model,name2_model,mask1,mask2,hemi,plot_filepath)
                # print(subject+'_name1-'+name1+'_measure-'+name1_measure_label+'_name2-'+name2+'_measure-'+name2_measure_label)

                return results


            results = Parallel(n_jobs=-1)(delayed(process)(subject,name1,name1_type,model1,mask1,name2,name2_type,model2,mask2,hemi,plot) for subject in tqdm(self.subjects['SIpointlights']) for ((name1,name1_type,model1,mask1),(name2,name2_type,model2,mask2)) in selected_features for hemi in ['all','left','right'])
            results = np.array(results)
            results = pd.DataFrame(results,columns = ['subject', 'name1','name2','names','name1_measure','name2_measure','name1_model','name2_model','mask1','mask2','hemi','total_voxels1','total_voxels2','total_voxels','voxels1_only','voxels2_only','voxels_overlap','DICE_coef',
                        'proportion_of_voxels1_in_this_hemi','proportion_of_voxels2_in_this_hemi','proportion_of_all_voxels_in_this_hemi','proportion_of_overlap_in_this_hemi', 
                        'proportion_of_1_that_is_also_2', 'proportion_of_2_that_is_also_1',
                        'proportion_of_voxels_that_is_voxels1', 'proportion_of_voxels_that_is_voxels2','proportion_of_voxels_that_is_overlap'])
            file_label = self.sid+self.enc_file_label+'_'+selection_type+'_overlap2'
            results.to_csv(self.out_dir+'/'+file_label+'_perc_top_voxels-' + self.perc_top_voxels+'_'+label+'_enc_feature_localizer_results_'+file_tag+'.csv')
        
        file_label = self.sid+self.enc_file_label+'_'+selection_type+'_overlap2'
        #reload the csv so that all dtypes are automatically correct
        results = pd.read_csv(self.out_dir+'/'+file_label+'_perc_top_voxels-' + self.perc_top_voxels+'_'+label+'_enc_feature_localizer_results_'+file_tag+'.csv')

        
        # names = [('interact&no_interact','glm-SIpointlights'),('interact-no_interact','glm-SIpointlights'),('intact-degraded','glm-language'),('SimCLR','encoding-unique_var'),('SLIP','encoding-unique_var'),('CLIP','encoding-unique_var')]#
        
        file_label = 'overlap/'+file_label

        ##TODO
        #across stimuli type
        #across social and language
        #

        overlap_matrix = np.zeros(())

        def get_overlap_matrix(names,hemi):
            overlap_matrix = np.zeros((len(names),len(names)))
            pvalue_matrix=overlap_matrix.copy()

            ind_tracker = []
            within_function_across_sel_method = []
            across_function_within_sel_method = []
            across_function_across_sel_method = []

            for (ind1,name1) in enumerate(names):
                for (ind2,name2) in enumerate(names):
                    if((ind1,ind2) not in ind_tracker):
                        temp_names = [name1,name2]
                        temp_names.sort()
                        # print(temp_names)
                        names_label = temp_names[0][0]+'-'+temp_names[0][1].split('-')[1]+'_'+temp_names[1][0]+'-'+temp_names[1][1].split('-')[1]
                        print(names_label)
                        temp_results = results[(results.hemi==hemi)&(results.names==names_label)]#&(results.mask_name==mask_name)]
                        stat_result = stats.ttest_1samp(temp_results.DICE_coef.dropna(),0)
                        # print(temp_results.mask_name)

                        value1 = np.nanmean(temp_results.DICE_coef)#proportion_of_1_that_is_also_2)
                        value2 = np.nanmean(temp_results.DICE_coef)#proportion_of_2_that_is_also_1)
                        overlap_matrix[ind1][ind2]=value1
                        overlap_matrix[ind2][ind1]=value2
                        pvalue_matrix[ind1][ind2]=stat_result.pvalue
                        ind_tracker.append((ind1,ind2))

                        sel_method1 = temp_names[0][1].split('-')[0]
                        sel_method2 = temp_names[1][1].split('-')[0]
                        name1_ = temp_names[0][0]
                        name2_ = temp_names[1][0]
                        # function_dict = {'interact-no_interact':'social',
                        #                'interact&no_interact':'visual',
                        #                'intact-degraded':'language',
                        #                'glove':'language' }
                        # function1 = function_dict[name1_]
                        # function2 = function_dict[name2_]
                        # print(sel_method1)
                        # print(function1)
                        # print(sel_method2)
                        # print(function2)

                        # if(sel_method1==sel_method2):
                        #   if(function1!=function2):
                        #       across_function_within_sel_method.append(value1)
                        # else: #different selection methods
                        #   if(function1==function2):
                        #       within_function_across_sel_method.append(value1)
                        #   else: #different functions
                        #       across_function_across_sel_method.append(value1)
            # print(overlap_matrix)
            # print('across_function_within_sel_method')
            # # print(across_function_within_sel_method)
            # print(np.mean(across_function_within_sel_method))
            # print('within_function_across_sel_method')
            # # print(within_function_across_sel_method)
            # print(np.mean(within_function_across_sel_method))
            # print('across_function_across_sel_method')
            # # print(across_function_across_sel_method)
            # print(np.mean(across_function_across_sel_method))
            overlap_matrix[np.triu_indices(overlap_matrix.shape[0],0)] = np.nan
            overlap_matrix[0,0] =np.nan
            overlap_matrix[overlap_matrix.shape[0]-1,overlap_matrix.shape[0]-1]=np.nan
            fig = plt.figure()
            ax = fig.add_subplot(111)
            cax = ax.imshow(overlap_matrix,cmap='Greens',vmin=0.0,vmax=0.3)

            sns.despine(left=True,bottom=True)
            fig.colorbar(cax)
            plt.title(hemi+' '+' and '.join(label.split('_')))
            ax.set_xticks(range(0,len(names)-1))
            ax.set_yticks(range(1,len(names)))

            for i in range(len(names)):
                for j in range((len(names))):
                    c = overlap_matrix[j,i]
                    pvalue = pvalue_matrix[j,i]
                    if(~np.isnan(c)):
                        if(pvalue<0.001):
                            ax.text(i, j, str(np.round(c,4)), va='center', ha='center',fontweight='heavy')
                        elif(pvalue<0.05):
                            ax.text(i, j, str(np.round(c,4)), va='center', ha='center')
                        else:
                            ax.text(i, j, str(np.round(c,4)), va='center', ha='center',color='gray')

            name_dict = {'interact-no_interact':' controlled \n  social   \ninteraction',
                         'intact-degraded':'controlled\nlanguage',
                         'social':'movie\nsocial',
                         'sbert+word2vec':'movie\nsbert+\nwordvec'}
            ax.set_xticklabels([name_dict[name[0]] for name in names[:-1]])
            ax.set_yticklabels([name_dict[name[0]] for name in names[1:]])
            # plt.xticks(rotation=90)
            plt.savefig(self.figure_dir + '/'+file_label+'_perc_top_voxels-' + self.perc_top_voxels+'_overlap_matrix_'+label+'_'+hemi+'_'+file_tag+'.pdf',bbox_inches='tight')
            plt.close()

            return overlap_matrix



        overlap_matrix = get_overlap_matrix(names,'left')
        overlap_matrix = get_overlap_matrix(names,'right')
        # overlap_matrix = get_overlap_matrix(names,'all')

        def plot_venn2(names):
            names.sort()
            name1 = names[0][0]
            name2 = names[1][0]
            names_label = names[0][0]+'-'+names[0][1].split('-')[1]+'_'+names[1][0]+'-'+names[1][1].split('-')[1]
            measure1 = 'voxels1_only'#'proportion_of_voxels_that_is_voxels1'#voxels1_only
            measure2 = 'voxels2_only'#'proportion_of_voxels_that_is_voxels2'#'voxels2_only' #p
            measure_overlap = 'voxels_overlap'#'proportion_of_voxels_that_is_overlap'#voxels_overlap
            temp_results = results[(results.hemi=='all')&(results.names==names_label)]
            one = np.round(np.nanmean(temp_results[measure1]),2)
            two = np.round(np.nanmean(temp_results[measure2]),2)
            overlap = np.round(np.nanmean(temp_results[measure_overlap]),2)
            v=venn2(subsets = (one, two,overlap), set_labels = (name1, name2))
            v.get_patch_by_id('10').set_color('mediumblue')
            v.get_patch_by_id('10').set_alpha(.8)
            v.get_patch_by_id('01').set_color('gold')
            v.get_patch_by_id('01').set_alpha(.8)
            if(v.get_patch_by_id('11') != None):
                v.get_patch_by_id('11').set_color('green')
                v.get_patch_by_id('11').set_alpha(.8)
            plt.title('bilateral '+label)
            plt.savefig(self.figure_dir + '/'+file_label+'_perc_top_voxels-' + self.perc_top_voxels+'_'+name1+'_'+name2+'_'+label+'_all_venn.pdf',bbox_inches='tight')
            plt.close()

            temp_results = results[(results.hemi=='left')&(results.names==names_label)]
            one = np.round(np.nanmean(temp_results[measure1]),2)
            two = np.round(np.nanmean(temp_results[measure2]),2)
            overlap = np.round(np.nanmean(temp_results[measure_overlap]),2)
            v=venn2(subsets = (one, two,overlap), set_labels = (name1, name2))
            v.get_patch_by_id('10').set_color('mediumblue')
            v.get_patch_by_id('10').set_alpha(.8)
            v.get_patch_by_id('01').set_color('gold')
            v.get_patch_by_id('01').set_alpha(.8)
            if(v.get_patch_by_id('11') != None):
                v.get_patch_by_id('11').set_color('green')
                v.get_patch_by_id('11').set_alpha(.8)
            plt.title('left '+label)
            plt.savefig(self.figure_dir + '/'+file_label+'_perc_top_voxels-' + self.perc_top_voxels+'_'+name1+'_'+name2+'_'+label+'_left_venn.pdf',bbox_inches='tight')
            plt.close()

            temp_results = results[(results.hemi=='right')&(results.names==names_label)&(results.name1_measure==names[0][1].split('-')[1])&(results.name2_measure==names[1][1].split('-')[1])]
            one = np.round(np.nanmean(temp_results[measure1]),2)
            two = np.round(np.nanmean(temp_results[measure2]),2)
            overlap = np.round(np.nanmean(temp_results[measure_overlap]),2)
            v = venn2(subsets = (one, two,overlap), set_labels = (name1, name2))
            v.get_patch_by_id('10').set_color('mediumblue')
            v.get_patch_by_id('10').set_alpha(.8)
            v.get_patch_by_id('01').set_color('gold')
            v.get_patch_by_id('01').set_alpha(.8)
            if(v.get_patch_by_id('11') != None):
                v.get_patch_by_id('11').set_color('green')
                v.get_patch_by_id('11').set_alpha(.8)
            plt.title('right '+label)
            plt.savefig(self.figure_dir + '/'+file_label+'_perc_top_voxels-' + self.perc_top_voxels+'_'+name1+'_'+name2+'_'+label+'_right_venn.pdf',bbox_inches='tight')
            plt.close()

        plot_venn_bool = False
        if(plot_venn_bool):
            for name1 in names:
                for name2 in names:
                    if(name1!=name2):
                        plot_venn2([name1]+[name2])
    def compute_all_overlap3(self,load=False,plot=True,selection_type='top_percent',pvalue=None,regions=[],label=''):
        from matplotlib import colors

        if(selection_type=='top_percent'):
            suffix = '_binary'
            folder = '/localizer_masks/'
            overlap_folder = '/localizer_overlap_maps/'
        elif(selection_type=='all_significant'):
            suffix = '_sig-'+str(pvalue)
            folder = '/all_significant_voxels/'
            overlap_folder = '/all_significant_overlap_maps/'

        cmap = colors.ListedColormap(['white','mediumblue','gold','green','red','red','red','purple','orange','grey','white','mediumblue','gold','green','red','red','red','purple','orange','grey'])


        # localizer_contrasts_social = ['interact-no_interact','interact&no_interact']
        # localizer_contrasts_lang = ['intact-degraded']
        # # ind_perf_features = ['social_nonsocial','speaking','mentalization','face','AverME']
        # unique_var_features = ['social','speaking','glove','DNN']

        # # encoding_ind_perf_names = [(name,'encoding-ind_perf') for name in ind_perf_features]
        # encoding_unique_var_names = [(name,'encoding-unique_var') for name in unique_var_features]
        # localizer_names_social = [(name,'glm-SIpointlights') for name in localizer_contrasts_social]
        # localizer_names_lang = [(name,'glm-language') for name in localizer_contrasts_lang]
        # # names = encoding_ind_perf_names + encoding_unique_var_names + localizer_names_social + localizer_names_lang
        # names = encoding_unique_var_names + localizer_names_social + localizer_names_lang

        names = regions

        # selected_features = []
        # for selections in [[(name1,name1_model_type),(name2,name2_model_type),(name3,name3_model_type)] for (name1,name1_model_type) in names for (name2,name2_model_type) in names for (name3,name3_model_type) in names]:
        #   selections.sort()
        #   if((selections[0]!=selections[1]) & (selections[0]!=selections[2]) & (selections[1]!=selections[2]) ):
        #       selected_features.append((selections[0],selections[1],selections[2]))
        selected_features = []
        for selections in [[(name1,name1_model_type,mask1),(name2,name2_model_type,mask2),(name3,name3_model_type,mask3)] for (name1,name1_model_type,mask1) in names for (name2,name2_model_type,mask2) in names for (name3,name3_model_type,mask3) in names]:
            selections.sort()
            if((selections[0]!=selections[1]) & (selections[0]!=selections[2]) & (selections[1]!=selections[2]) ):
                selected_features.append((selections[0],selections[1],selections[2]))
        print(len(selected_features))

        selected_features = list(set(selected_features))
        selected_features.sort()

        if(not load):
            def process(subject,name1,name1_type,mask1,name2,name2_type,mask2,name3,name3_type,mask3,hemi,plot):
                def get_info(subject,data1_filepath,data2_filepath,data3_filepath,name1,name2,name3,name1_measure_label,name2_measure_label,name3_measure_label,mask1,mask2,mask3,hemi,plot_filepath):
                    try:
                        data1_img = nibabel.load(data1_filepath)
                        data2_img = nibabel.load(data2_filepath)
                        data3_img = nibabel.load(data3_filepath)

                        data1 = (data1_img.get_fdata()>0)*1.0
                        data2 = (data2_img.get_fdata()>0)*2.0
                        data3 = (data3_img.get_fdata()>0)*6.0

                        overlap = data1+data2+data3
                        overlap_img = nibabel.Nifti1Image(overlap,data1_img.affine)
                        if(plot_filepath!='None'):
                            title = subject+ ' [' + name1+': blue, '+name2+': yellow, '+name3+': red]'
                            helpers.plot_img_volume(overlap_img,plot_filepath,threshold=0.99,cmap=cmap,title=title,vmax=9)

                        total_voxels1_all = len(data1[data1==1])
                        total_voxels2_all = len(data2[data2==2])
                        total_voxels3_all = len(data3[data3==6])
                        total_voxels_all = total_voxels1_all+total_voxels2_all+total_voxels3_all
                        voxels_overlap_all = len(overlap[overlap==9])

                        fullway = overlap.shape[0]
                        halfway = int(fullway/2)

                        if(hemi=='left'):
                            overlap=overlap[0:halfway]
                            data1= data1[0:halfway]
                            data2= data2[0:halfway]
                        elif(hemi=='right'):
                            overlap=overlap[halfway+1:fullway]
                            data1 = data1[halfway+1:fullway]
                            data2 = data2[halfway+1:fullway]

                        total_voxels1 = len(data1[data1==1])
                        total_voxels2 = len(data2[data2==2])
                        total_voxels3 = len(data3[data3==6])
                        total_voxels = total_voxels1+total_voxels2+total_voxels3

                        voxels1_only = len(overlap[overlap==1])
                        voxels2_only = len(overlap[overlap==2])
                        voxels3_only = len(overlap[overlap==6])
                        voxels12_only = len(overlap[overlap==3])
                        voxels13_only = len(overlap[overlap==7])
                        voxels23_only = len(overlap[overlap==8])
                        voxels_overlap = len(overlap[overlap==9])

                        voxels1_0 = voxels1_only
                        voxels2_0 = voxels2_only
                        voxels3_0 = voxels3_only

                        # if(voxels1_0==0):
                        #   voxels1_0 = np.nan
                        # if(voxels2_0==0):
                        #   voxels2_0 = np.nan
                        # if(voxels3_0==0):
                        #   voxels3_0 = np.nan

                        if(total_voxels==0):
                            total_voxels = np.nan
                        if(total_voxels_all==0):
                            total_voxels_all = np.nan
                        if(voxels_overlap_all==0):
                            voxels_overlap_all = np.nan

                        DICE_coef = total_voxels and (3*voxels_overlap)/total_voxels
                        proportion_of_voxels1_in_this_hemi = total_voxels1_all and total_voxels1/total_voxels1_all
                        proportion_of_voxels2_in_this_hemi = total_voxels2_all and total_voxels2/total_voxels2_all
                        proportion_of_voxels3_in_this_hemi = total_voxels3_all and total_voxels3/total_voxels3_all
                        proportion_of_all_voxels_in_this_hemi = total_voxels_all and total_voxels/total_voxels_all
                        proportion_of_overlap_in_this_hemi = voxels_overlap_all and voxels_overlap/voxels_overlap_all

                        proportion_of_1_that_is_also_23 = total_voxels1 and voxels_overlap/total_voxels1
                        proportion_of_2_that_is_also_13 = total_voxels2 and voxels_overlap/total_voxels2
                        proportion_of_3_that_is_also_12 = total_voxels3 and voxels_overlap/total_voxels3

                        proportion_of_voxels_that_is_voxels1 = total_voxels and voxels1_only/total_voxels
                        proportion_of_voxels_that_is_voxels2 = total_voxels and voxels2_only/total_voxels
                        proportion_of_voxels_that_is_voxels3 = total_voxels and voxels3_only/total_voxels
                        proportion_of_voxels_that_is_voxels12 = total_voxels and voxels12_only/total_voxels
                        proportion_of_voxels_that_is_voxels13 =  total_voxels and voxels13_only/total_voxels
                        proportion_of_voxels_that_is_voxels23 = total_voxels and voxels23_only/total_voxels
                        proportion_of_voxels_that_is_overlap =  total_voxels and voxels_overlap/total_voxels

                        names = [(name1,name1_measure_label),(name2,name2_measure_label),(name3,name3_measure_label)]
                        names.sort()
                        names_label = names[0][0]+'-'+names[0][1]+'_'+names[1][0]+'-'+names[1][1]+'_'+names[2][0]+'-'+names[2][1]
                    except Exception as e:
                        print(e)
                        results = [np.nan]*39
                        return results

                    results = (subject, name1,name2,name3,names_label,name1_measure_label,name2_measure_label,name3_measure_label,mask1,mask2,mask3,hemi,total_voxels1,total_voxels2,total_voxels3, total_voxels,
                        voxels1_only, voxels2_only, voxels3_only, voxels12_only,voxels13_only,voxels23_only, voxels_overlap,DICE_coef,
                        proportion_of_voxels1_in_this_hemi,proportion_of_voxels2_in_this_hemi,proportion_of_voxels3_in_this_hemi, proportion_of_all_voxels_in_this_hemi,proportion_of_overlap_in_this_hemi, 
                        proportion_of_1_that_is_also_23, proportion_of_2_that_is_also_13, proportion_of_1_that_is_also_23,
                        proportion_of_voxels_that_is_voxels1, proportion_of_voxels_that_is_voxels2, proportion_of_voxels_that_is_voxels3,
                        proportion_of_voxels_that_is_voxels12,proportion_of_voxels_that_is_voxels13,proportion_of_voxels_that_is_voxels23,proportion_of_voxels_that_is_overlap)
                    return results

                enc_file_label = subject+'_encoding_model-'+self.model+'_smoothingfwhm-'+str(self.smoothing_fwhm)+'_chunklen-'+str(self.chunklen) #+ '_mask-'+mask_name
                glm_file_label = subject+'_smoothingfwhm-'+str(self.smoothing_fwhm) #+ '_mask-'+mask_name

                name1_model = name1_type.split('-')[0]
                name1_measure_label = name1_type.split('-')[1]
                name2_model = name2_type.split('-')[0]
                name2_measure_label = name2_type.split('-')[1]
                name3_model = name3_type.split('-')[0]
                name3_measure_label = name3_type.split('-')[1]

                if(name1_model=='encoding'):
                    filepath1 = self.out_dir + folder+enc_file_label+ '_mask-'+mask1+'_measure-'+name1_measure_label+'_enc_feature_loc-'+name1+suffix+'.nii.gz'
                elif(name1_model=='glm'):
                    filepath1 = self.out_dir + folder+glm_file_label+ '_mask-'+mask1+'_glm_loc-'+name1+'_run-all'+suffix+'.nii.gz'
                elif(name1_model=='overlap'):
                    filepath1 = self.out_dir + folder+subject+ '_overlap-'+name1+suffix+'.nii.gz'

                if(name2_model=='encoding'):
                    filepath2 = self.out_dir + folder+enc_file_label+ '_mask-'+mask2+'_measure-'+name2_measure_label+'_enc_feature_loc-'+name2+suffix+'.nii.gz'
                elif(name2_model=='glm'):
                    filepath2 = self.out_dir + folder+glm_file_label+ '_mask-'+mask2+'_glm_loc-'+name2+'_run-all'+suffix+'.nii.gz'
                elif(name2_model=='overlap'):
                    filepath2 = self.out_dir + folder+subject+ '_overlap-'+name2+suffix+'.nii.gz'

                if(name3_model=='encoding'):
                    filepath3 = self.out_dir + folder+enc_file_label+ '_mask-'+mask3+'_measure-'+name3_measure_label+'_enc_feature_loc-'+name3+suffix+'.nii.gz'
                elif(name3_model=='glm'):
                    filepath3 = self.out_dir + folder+glm_file_label+ '_mask-'+mask3+'_glm_loc-'+name3+'_run-all'+suffix+'.nii.gz'
                elif(name3_model=='overlap'):
                    filepath3 = self.out_dir + folder+subject+ '_overlap-'+name3+suffix+'.nii.gz'
                
                if(plot):
                    plot_filepath = self.figure_dir + overlap_folder+enc_file_label+'_'+name1_type+'-'+name1+'_'+name2_type+'-'+name2+'_'+name3_type+'-'+name3+'_'+selection_type+'.png'

                results = get_info(subject,filepath1,filepath2,filepath3, name1,name2,name3,name1_measure_label,name2_measure_label,name3_measure_label,mask1,mask2,mask3,hemi,plot_filepath)
                print(subject+'_name1-'+name1+'_measure-'+name1_measure_label+'_name2-'+name2+'_measure-'+name2_measure_label+'_name3-'+name3+'_measure-'+name3_measure_label)

                return results


            # for (feature_name1,feature_name2,localizer_contrast1,localizer_contrast2) in selected_features
            # for subject in self.subjects:
            results = Parallel(n_jobs=-1)(delayed(process)(subject,name1,name1_type,mask1,name2,name2_type,mask2,name3,name3_type,mask3,hemi,plot) for subject in self.subjects['SIpointlights'] for ((name1,name1_type,mask1),(name2,name2_type,mask2),(name3,name3_type,mask3)) in selected_features for hemi in ['all','left','right'])
            # results = pd.DataFrame(zip(subjects,localizer_contrasts,num_voxels_glm,proportion_glm,overlap_types),columns=['subject','localizer_contrast','glm_num_voxels','proportion_glm','hemi'])
            results = np.array(results)
            results = pd.DataFrame(results,columns = ['subject', 'name1','name2','name3','names_label','name1_measure_label','name2_measure_label','name3_measure_label','mask1','mask2','mask3','hemi','total_voxels1','total_voxels2','total_voxels3', 'total_voxels',
                        'voxels1_only', 'voxels2_only', 'voxels3_only','voxels12_only','voxels13_only','voxels23_only', 'voxels_overlap','DICE_coef',
                        'proportion_of_voxels1_in_this_hemi','proportion_of_voxels2_in_this_hemi','proportion_of_voxels3_in_this_hemi', 'proportion_of_all_voxels_in_this_hemi','proportion_of_overlap_in_this_hemi', 
                        'proportion_of_1_that_is_also_23', 'proportion_of_2_that_is_also_13', 'proportion_of_1_that_is_also_23',
                        'proportion_of_voxels_that_is_voxels1', 'proportion_of_voxels_that_is_voxels2', 'proportion_of_voxels_that_is_voxels3',
                        'proportion_of_voxels_that_is_voxels12','proportion_of_voxels_that_is_voxels13','proportion_of_voxels_that_is_voxels23','proportion_of_voxels_that_is_overlap'])
            file_label = self.sid+self.enc_file_label+'_'+selection_type+'_overlap3_mask-'
            results.to_csv(self.out_dir+'/'+file_label+'_perc_top_voxels-' + self.perc_top_voxels+'_'+label+'_enc_feature_localizer_results.csv')
        
        file_label = self.sid+self.enc_file_label+'_'+selection_type+'_overlap3_mask-'
        #reload the csv so that all dtypes are automatically correct
        results = pd.read_csv(self.out_dir+'/'+file_label+'_perc_top_voxels-' + self.perc_top_voxels+'_'+label+'_enc_feature_localizer_results.csv')



        # results = results[(results.hemi=='all')&((results.name2=='interact-no_interact')|(results.name1=='interact-no_interact'))]
        # ax = sns.barplot(data=results,x='names',y='DICE_coef',alpha=0.6)
        # plt.xticks(rotation=45)
        # ax = sns.stripplot(data=results,x='names',y='DICE_coef',alpha=0.6)
        # ax.set_ylabel('Dice coef')
        # plt.savefig(self.figure_dir + '/'+file_label+'_STS_both_hemi_overlap.png',bbox_inches='tight')
        # plt.close()

        # results = results[(results.hemi!='all')&((results.name2=='interact-no_interact')|(results.name1=='interact-no_interact'))]
        # ax = sns.barplot(data=results,x='names',y='DICE_coef',hue='hemi',hue_order=['left','right'],alpha=0.6)
        # plt.xticks(rotation=45)
        # ax = sns.stripplot(data=results,x='names',y='DICE_coef',hue='hemi',hue_order=['left','right'],dodge=True)
        # ax.set_ylabel('Dice coef')
        # plt.savefig(self.figure_dir + '/'+file_label+'_STS_sep_hemi_overlap.png',bbox_inches='tight')
        # plt.close()

        #TODO -- plot venn diagram of the proportion of voxels
        from matplotlib_venn import venn3
        file_label = 'overlap/'+file_label
        def plot_venn3(names):
            names.sort()
            name1 = names[0][0]
            name2 = names[1][0]
            name3 = names[2][0]
            names_label = names[0][0]+'-'+names[0][1].split('-')[1]+'_'+names[1][0]+'-'+names[1][1].split('-')[1]+'_'+names[2][0]+'-'+names[2][1].split('-')[1]
            round_to = 2

            measure1 = 'voxels1_only'
            measure2 = 'voxels2_only'
            measure3 = 'voxels3_only'
            measure12 = 'voxels12_only'
            measure13 = 'voxels13_only'
            measure23 = 'voxels23_only'
            measure_overlap = 'voxels_overlap'

            temp_results = results[(results.hemi=='all')&(results.names_label==names_label)]
            print(temp_results)
            one = np.round(np.nanmean(temp_results[measure1]),round_to)
            two = np.round(np.nanmean(temp_results[measure2]),round_to)
            three = np.round(np.nanmean(temp_results[measure3]),round_to)
            overlap_12 = np.round(np.nanmean(temp_results[measure12]),round_to)
            overlap_13 = np.round(np.nanmean(temp_results[measure13]),round_to)
            overlap_23 = np.round(np.nanmean(temp_results[measure23]),round_to)
            overlap = np.round(np.nanmean(temp_results[measure_overlap]),round_to)
            try:
                v = venn3(subsets = (one, two,overlap_12,three,overlap_13,overlap_23,overlap), set_labels = (name1, name2,name3))
                if(v.get_patch_by_id('100') != None):
                    v.get_patch_by_id('100').set_color('mediumblue')
                    v.get_patch_by_id('100').set_alpha(.8)
                if(v.get_patch_by_id('010') != None):
                    v.get_patch_by_id('010').set_color('gold')
                    v.get_patch_by_id('010').set_alpha(.8)
                if(v.get_patch_by_id('001') != None):
                    v.get_patch_by_id('001').set_color('red')
                    v.get_patch_by_id('001').set_alpha(.8)
                if(v.get_patch_by_id('110') != None):
                    v.get_patch_by_id('110').set_color('green')
                    v.get_patch_by_id('110').set_alpha(.8)
                if(v.get_patch_by_id('101') != None):
                    v.get_patch_by_id('101').set_color('purple')
                    v.get_patch_by_id('101').set_alpha(.8)
                if(v.get_patch_by_id('011') != None):
                    v.get_patch_by_id('011').set_color('orange')
                    v.get_patch_by_id('011').set_alpha(.8)
                if(v.get_patch_by_id('111') != None):
                    v.get_patch_by_id('111').set_color('grey')
                    v.get_patch_by_id('111').set_alpha(.8)
            except Exception as e:
                print(e)
                print(names_label)
                print('all')
                print('one',one,'two',two,'three',three,'overlap12',overlap_12,'overlap13',overlap_13,'overlap23',overlap_23,'overlap',overlap)
                pass
            plt.title('bilateral '+label)
            plt.savefig(self.figure_dir + '/'+file_label+'_'+name1+'_'+name2+'_'+name3+'_'+label+'_all_venn.pdf',bbox_inches='tight')
            plt.close()

            temp_results = results[(results.hemi=='left')&(results.names_label==names_label)]
            one = np.round(np.nanmean(temp_results[measure1]),round_to)
            two = np.round(np.nanmean(temp_results[measure2]),round_to)
            three = np.round(np.nanmean(temp_results[measure3]),round_to)
            overlap_12 = np.round(np.nanmean(temp_results[measure12]),round_to)
            overlap_13 = np.round(np.nanmean(temp_results[measure13]),round_to)
            overlap_23 = np.round(np.nanmean(temp_results[measure23]),round_to)
            overlap = np.round(np.nanmean(temp_results[measure_overlap]),round_to)
            try:
                v = venn3(subsets = (one, two,overlap_12,three,overlap_13,overlap_23,overlap), set_labels = (name1, name2,name3))
                if(v.get_patch_by_id('100') != None):
                    v.get_patch_by_id('100').set_color('mediumblue')
                    v.get_patch_by_id('100').set_alpha(.8)
                if(v.get_patch_by_id('010') != None):
                    v.get_patch_by_id('010').set_color('gold')
                    v.get_patch_by_id('010').set_alpha(.8)
                if(v.get_patch_by_id('001') != None):
                    v.get_patch_by_id('001').set_color('red')
                    v.get_patch_by_id('001').set_alpha(.8)
                if(v.get_patch_by_id('110') != None):
                    v.get_patch_by_id('110').set_color('green')
                    v.get_patch_by_id('110').set_alpha(.8)
                if(v.get_patch_by_id('101') != None):
                    v.get_patch_by_id('101').set_color('purple')
                    v.get_patch_by_id('101').set_alpha(.8)
                if(v.get_patch_by_id('011') != None):
                    v.get_patch_by_id('011').set_color('orange')
                    v.get_patch_by_id('011').set_alpha(.8)
                if(v.get_patch_by_id('111') != None):
                    v.get_patch_by_id('111').set_color('grey')
                    v.get_patch_by_id('111').set_alpha(.8)
            except Exception as e:
                print(e)
                print(names_label)
                print('left')
                print('one',one,'two',two,'three',three,'overlap12',overlap_12,'overlap13',overlap_13,'overlap23',overlap_23,'overlap',overlap)
                pass
            
            plt.title('left '+label)
            plt.savefig(self.figure_dir + '/'+file_label+'_'+name1+'_'+name2+'_'+name3+'_'+label+'_left_venn.pdf',bbox_inches='tight')
            plt.close()

            temp_results = results[(results.hemi=='right')&(results.names_label==names_label)]
            one = np.round(np.nanmean(temp_results[measure1]),round_to)
            two = np.round(np.nanmean(temp_results[measure2]),round_to)
            three = np.round(np.nanmean(temp_results[measure3]),round_to)
            overlap_12 = np.round(np.nanmean(temp_results[measure12]),round_to)
            overlap_13 = np.round(np.nanmean(temp_results[measure13]),round_to)
            overlap_23 = np.round(np.nanmean(temp_results[measure23]),round_to)
            overlap = np.round(np.nanmean(temp_results[measure_overlap]),round_to)
            try:
                v = venn3(subsets = (one, two,overlap_12,three,overlap_13,overlap_23,overlap), set_labels = (name1, name2,name3))
                if(v.get_patch_by_id('100') != None):
                    v.get_patch_by_id('100').set_color('mediumblue')
                    v.get_patch_by_id('100').set_alpha(.8)
                if(v.get_patch_by_id('010') != None):
                    v.get_patch_by_id('010').set_color('gold')
                    v.get_patch_by_id('010').set_alpha(.8)
                if(v.get_patch_by_id('001') != None):
                    v.get_patch_by_id('001').set_color('red')
                    v.get_patch_by_id('001').set_alpha(.8)
                if(v.get_patch_by_id('110') != None):
                    v.get_patch_by_id('110').set_color('green')
                    v.get_patch_by_id('110').set_alpha(.8)
                if(v.get_patch_by_id('101') != None):
                    v.get_patch_by_id('101').set_color('purple')
                    v.get_patch_by_id('101').set_alpha(.8)
                if(v.get_patch_by_id('011') != None):
                    v.get_patch_by_id('011').set_color('orange')
                    v.get_patch_by_id('011').set_alpha(.8)
                if(v.get_patch_by_id('111') != None):
                    v.get_patch_by_id('111').set_color('grey')
                    v.get_patch_by_id('111').set_alpha(.8)
            except Exception as e:
                print(e)
                print(names_label)
                print('right')
                print('one',one,'two',two,'three',three,'overlap12',overlap_12,'overlap13',overlap_13,'overlap23',overlap_23,'overlap',overlap)
                pass
            plt.title('right '+label)
            plt.savefig(self.figure_dir + '/'+file_label+'_'+name1+'_'+name2+'_'+name3+'_'+label+'_right_venn.pdf',bbox_inches='tight')
            plt.close()

        for name1 in names:
            for name2 in names:
                for name3 in names:
                    if(name1 not in [name2,name3]):
                        if(name2!=name3):
                            plot_venn3([name1,name2,name3]) 

    def unique_variance_stats(self):
        #do the stats for unique variance explained in the localizer voxels

        results = pd.read_csv(self.out_dir + '/'+self.sid+self.glm_file_label_out+'_perc_top_voxels-' + self.perc_top_voxels+'_subject_level_glm_localizer_univariate_analysis.csv')

        print(results)
        for localizer_contrast in ['interact-no_interact','intact-degraded']:
            for encoding_feature in ['social','glove','DNN']:
                for hemi in['L','R']:
                    if('interact' in localizer_contrast):
                        localizer = 'SIpointlights'
                        data_selection_label_glm = 'interact'
                    else:
                        localizer = 'language'
                        data_selection_label_glm = 'intact'
                    temp_results = results[(results.glm_response_contrast==data_selection_label_glm)&(results.localizer_contrast==localizer_contrast)&(results.enc_feature_name==encoding_feature)&(results.hemi==hemi)]
                    values = temp_results.encoding_response

                    null_values = []
                    for subject in self.subjects[localizer]:
                        #load the null distribution
                        filename = self.enc_dir+'/unique_variance/'+subject+self.enc_file_label +'_feature-'+encoding_feature+'_measure-avg_null_unique_var.nii.gz'
                        nii = nibabel.load(filename)
                        nii_data = nii.get_fdata()
                        #mask with localizer mask
                        
                        filename = self.out_dir + '/localizer_masks/'+subject+self.glm_file_label_out+'_glm_loc-'+localizer_contrast+'_run-'+self.all_runs[localizer]+'.nii.gz'
                        mask_nii = nibabel.load(filename)
                        mask_nii_data = mask_nii.get_fdata()

                        if(hemi=='L'):
                            mask_name = 'left_'+self.mask_name
                        elif(hemi=='R'):
                            mask_name = 'right_'+self.mask_name
                        self.load_mask(mask_name)
                        # resampled_mask = nilearn.image.resample_img(self.mask, target_affine=z_scores_img.affine, target_shape=z_scores.shape,interpolation='nearest')
                        mask = self.mask.get_fdata()

                        masked_data = nii_data[(mask_nii_data==1)&(mask==1)]
                        if(len(masked_data)>1):
                            null_value = np.mean(masked_data)
                            null_values.append(null_value)
                        #get the average value and add to list of null_values

                    t_stat,p_value = scipy.stats.ttest_ind(values,null_values)
                    # null_n = len(null_values)
                    # print(null_n)
                    # null_values =np.array(null_values)
                    # value_mean = np.nanmean(values)
                    # if(np.isnan(value_mean)):
                    #   p_value=np.nan
                    # else:
                    #   null_n_over_sample = sum((null_values>value_mean).astype(int))
                    #   p_value = null_n_over_sample/null_n
                    print(hemi+' localizer: '+localizer_contrast+', unique variance explained by: '+encoding_feature)
                    # print(np.nanmean(values))
                    print(p_value)
                    if(p_value<0.001):
                        ast = '***'
                    elif(p_value<0.01):
                        ast='**'
                    elif(p_value<0.05):
                        ast = '*'
                    else:
                        ast = ''
                    print('pvalue: '+ast)
    def ind_feature_performance_stats(self):
        #do the stats for unique variance explained in the localizer voxels
        results = pd.read_csv(self.out_dir + '/'+self.sid+self.glm_file_label_out+'_perc_top_voxels-' + self.perc_top_voxels+'_subject_level_glm_localizer_univariate_analysis.csv')

        
        for localizer_contrast in ['interact-no_interact','intact-degraded']:
            for encoding_feature in ['social','glove','DNN']:
                for hemi in['L','R']:
                    if('interact' in localizer_contrast):
                        localizer = 'SIpointlights'
                        data_selection_label_glm = 'interact'
                    else:
                        localizer = 'language'
                        data_selection_label_glm = 'intact'
                    temp_results = results[(results.glm_response_contrast==data_selection_label_glm)&(results.localizer_contrast==localizer_contrast)&(results.enc_feature_name==encoding_feature)&(results.hemi==hemi)]
                    values = temp_results.encoding_response

                    null_values = []
                    for subject in self.subjects[localizer]:
                        #load the null distribution
                        filename = self.enc_dir+'/ind_feature_performance/'+subject+self.enc_file_label +'_feature-'+encoding_feature+'_measure-avg_null_unique_var.nii.gz'
                        nii = nibabel.load(filename)
                        nii_data = nii.get_fdata()
                        #mask with localizer mask
                        
                        filename = self.out_dir + '/localizer_masks/'+subject+self.glm_file_label_out+'_glm_loc-'+localizer_contrast+'_run-'+self.all_runs[localizer]+'.nii.gz'
                        mask_nii = nibabel.load(filename)
                        mask_nii_data = mask_nii.get_fdata()

                        if(hemi=='L'):
                            mask_name = 'left_'+self.mask_name
                        elif(hemi=='R'):
                            mask_name = 'right_'+self.mask_name
                        self.load_mask(mask_name)
                        # resampled_mask = nilearn.image.resample_img(self.mask, target_affine=z_scores_img.affine, target_shape=z_scores.shape,interpolation='nearest')
                        mask = self.mask.get_fdata()

                        masked_data = nii_data[(mask_nii_data==1)&(mask==1)]
                        if(len(masked_data)>1):
                            null_value = np.mean(masked_data)
                            null_values.append(null_value)
                        #get the average value and add to list of null_values

                    # t_stat,p_value = scipy.stats.ttest_ind(values,null_values)
                    null_n = len(null_values)

                    null_values =np.array(null_values)
                    value_mean = np.nanmean(values)
                    if(np.isnan(value_mean)):
                        p_value=np.nan
                    else:
                        null_n_over_sample = sum((null_values>value_mean).astype(int))
                        p_value = null_n_over_sample/null_n
                    print(hemi+' localizer: '+localizer_contrast+', unique variance explained by: '+encoding_feature)
                    print(np.nanmean(values))
                    print(p_value)
                    if(p_value<0.001):
                        ast = '***'
                    elif(p_value<0.01):
                        ast='**'
                    elif(p_value<0.05):
                        ast = '*'
                    else:
                        ast = ''
                    print('pvalue: '+ast)

    def plot_diff_maps(self,features=['CLIP','SimCLR'],measure='ind_feature_performance',threshold=None,limits=0.05,cmap_name='blue_neg_yellow_pos',ROI=None):
        fsaverage = nilearn.datasets.fetch_surf_fsaverage(mesh='fsaverage')

        enc_file_label = '_encoding_model-'+self.model+'_smoothingfwhm-'+str(self.smoothing_fwhm)+'_chunklen-'+str(self.chunklen)
        if(self.mask_name!=None):
            enc_file_label = enc_file_label + '_mask-'+self.mask_name

        if(measure=='performance'):
            folder = 'performance'
            label = 'perf'
        if(measure=='ind_feature_performance'):
            folder = 'ind_feature_performance'
            label = 'ind_perf'
        if(measure=='ind_product_measure'):
            folder = 'ind_product_measure'
            label = 'ind_product_measure'
        elif(measure=='unique_variance'):
            folder='unique_variance'
            label = 'unique_var'


        feature1 = features[0]
        feature2 = features[1]

        plot_label = measure + ': ' + feature1 + '-'+feature2

        
        for subject in tqdm(self.subjects['SIpointlights']):
            #load the unique variance of each feature
            if(measure=='unique_variance'):
                filepath = self.enc_dir+'/'+folder+'/'+subject+enc_file_label+'_feature-'+feature1+'_measure-'+label+'_raw.nii.gz'
                nii1 = nibabel.load(filepath)
                data1 = nii1.get_fdata()
                #scale by the total variance
                if(self.scale_by=='total_variance'):
                    data1[data1<0] = 0 #clip response values to 0
                    data1 = data1/data1.sum(axis=0,keepdims=1)
            else:
                filepath = self.enc_dir+'/'+folder+'/'+subject+enc_file_label+'_measure-'+label+'_raw.nii.gz'
                nii1 = nibabel.load(filepath)
                data1 = nii1.get_fdata()
                if(feature1 in self.combined_features):
                    for (ind,sub_feature_name) in enumerate(self.model_features_dict[feature1]):
                        feature_ind = self.get_feature_index(subject,sub_feature_name)
                        sub_data = data1[feature_ind]
                        if(ind==0):
                            overall = sub_data
                        else:
                            overall = overall+sub_data
                    data1 = overall
                else:
                    feature_index = self.get_feature_index(subject,feature1)
                    data1 = data1[feature_index]

            if(measure=='unique_variance'):
                filepath = self.enc_dir+'/'+folder+'/'+subject+enc_file_label+'_feature-'+feature2+'_measure-'+label+'_raw.nii.gz'
                nii2 = nibabel.load(filepath)
                data2 = nii2.get_fdata()
            else:
                filepath = self.enc_dir+'/'+folder+'/'+subject+enc_file_label+'_measure-'+label+'_raw.nii.gz'
                nii2 = nibabel.load(filepath)
                data2 = nii2.get_fdata()
                if(self.scale_by=='total_variance'):
                    data2[data2<0] = 0 #clip response values to 0
                    data2 = data2/data2.sum(axis=0,keepdims=1)
                if(feature2 in self.combined_features):
                    for (ind,sub_feature_name) in enumerate(self.model_features_dict[feature2]):
                        feature_ind = self.get_feature_index(subject,sub_feature_name)
                        sub_data = data2[feature_ind]
                        if(ind==0):
                            overall = sub_data
                        else:
                            overall = overall+sub_data
                    data2 = overall
                else:
                    feature_index = self.get_feature_index(subject,feature2)
                    data2 = data2[feature_index]
            #project to surface

            #mask the with statistically significant performance maps for both SLIP and SimCLR

            ## TODO mask with statistically significant unique variance maps!!!

            #load the fdr corrected p value map and threshold it at p<0.05
            # threshold = 0.05
            # # enc_file_label = '_encoding_model-'+feature1+'_smoothingfwhm-'+str(self.smoothing_fwhm)+'_chunklen-'+str(self.chunklen)
            # # if(self.mask_name!=None):
            # #     enc_file_label = enc_file_label + '_mask-'+self.mask_name
            # filepath = self.enc_dir+'/unique_variance/'+subject+enc_file_label+'_feature-'+feature1+'_measure-unique_var_p_fdr.nii.gz'
            # nii = nibabel.load(filepath)
            # p_fdr1 = nii.get_fdata()
            # p_fdr1_mask = p_fdr1*0
            # self.load_mask(self.mask_name)
            # self.mask = nilearn.image.resample_img(self.mask, target_affine=nii.affine, target_shape=p_fdr1.shape,interpolation='nearest')
            # p_fdr1_mask[p_fdr1>threshold]=0
            # p_fdr1_mask[p_fdr1<threshold]=1
            # p_fdr1_mask[(self.mask.get_fdata()==0)]=0

            # print(p_fdr1[(p_fdr1<1)&(p_fdr1>0)])

            # # enc_file_label = '_encoding_model-'+feature2+'_smoothingfwhm-'+str(self.smoothing_fwhm)+'_chunklen-'+str(self.chunklen)
            # # if(self.mask_name!=None):
            # #     enc_file_label = enc_file_label + '_mask-'+self.mask_name
            # filepath = self.enc_dir+'/unique_variance/'+subject+enc_file_label+'_feature-'+feature2+'_measure-unique_var_p_fdr.nii.gz'
            # nii = nibabel.load(filepath)
            # p_fdr2 = nii.get_fdata()
            # p_fdr2_mask = p_fdr2*0
            # self.load_mask(self.mask_name)
            # self.mask = nilearn.image.resample_img(self.mask, target_affine=nii.affine, target_shape=p_fdr2.shape,interpolation='nearest')
            # p_fdr2_mask[p_fdr2>threshold]=0
            # p_fdr2_mask[p_fdr2<threshold]=1
            # p_fdr2_mask[(self.mask.get_fdata()==0)]=0


            diff_map = data1-data2
            # diff_map[ ( (p_fdr1_mask==0) | (p_fdr2_mask==0) ) ]= 0 #zero out any voxel that isn't significantly predicted by either feature space
            # diff_map = np.log10(data1/data2)
            diff_map_nii = nibabel.Nifti1Image(diff_map, nii1.affine)

            #plot difference map
            enc_file_label = '_encoding_model-'+self.model+'_smoothingfwhm-'+str(self.smoothing_fwhm)+'_chunklen-'+str(self.chunklen)
            if(self.mask_name!=None):
                enc_file_label = enc_file_label + '_mask-'+self.mask_name
            filename = self.figure_dir + '/difference_maps/'+subject+enc_file_label+'_measure-'+measure+'_diff-'+feature1+'-'+feature2+'.pdf'
            from matplotlib import colors
            cmap_for_plotting = self.cmaps[cmap_name]
            helpers.plot_img_volume(diff_map_nii,filename,symmetric_cbar=True,threshold=threshold,cmap=cmap_for_plotting,vmin=None,vmax=None,title=subject+', '+feature1+'-'+feature2)
            

            glm_file_label_ = '_smoothingfwhm-'+str(self.smoothing_fwhm)
            # enc_file_label_ = '_encoding_model-full_smoothingfwhm-'+str(self.smoothing_fwhm)+'_chunklen-'+str(self.chunklen)
            # enc_file_label_ = '_encoding_model-SimCLR_SLIP_smoothingfwhm-'+str(self.smoothing_fwhm)+'_chunklen-'+str(self.chunklen)
            localizer_contrasts = [ROI] if ROI!=None else []#['interact-no_interact','intact-degraded']#,'intact-degraded']
            # localizer_contrasts = ['social']
            # localizer_contrasts = ['SimCLR']

            all_ROI_data = np.zeros(diff_map.shape)
            for localizer_contrast in localizer_contrasts:
                for ind,mask in enumerate(self.localizer_masks[localizer_contrast]):
                    file_label_ = subject+glm_file_label_+'_mask-'+mask
                    # file_label_ = subject+enc_file_label_+'_mask-'+mask+'_measure-unique_var'#+measure_label#'_encoding_model-'+self.model
                    ROI_file = self.out_dir + '/localizer_masks/'+file_label_+'_glm_loc-'+localizer_contrast+'_run-all_binary.nii.gz'
                    # ROI_file = self.out_dir + '/localizer_masks/'+file_label_+'_enc_feature_loc-'+localizer_contrast+'_binary.nii.gz'
                    ROI_data = nibabel.load(ROI_file).get_fdata()
                    all_ROI_data[ROI_data>0] = 1+ind

            all_ROI_nii = nibabel.Nifti1Image(all_ROI_data, nii1.affine)
            
            # cmap = 'Spectral'#'PiYG'#'PuOr'#Oranges
            cmap_for_plotting = self.cmaps[cmap_name]
            ROI_label = ROI if ROI!=None else ''#'_'.join(localizer_contrasts)
            filename = self.figure_dir + '/difference_maps/surface/'+subject+enc_file_label+'_measure-'+measure+'_'+'_diff-'+feature1+'-'+feature2+'_ROI-'+ROI_label+'_surf'
            # helpers.plot_lateral_surface(diff_map_nii,filename,symmetric_cbar=True,threshold=0.000000001,cmap='PuOr',vmax=None,title=subject+', '+feature1+'-'+feature2)
            if(self.scale_by==''):
                cbar_label = 'product measure $R^2$ difference'
            elif(self.scale_by=='total_variance'):
                cbar_label = 'difference in proportion of total $R^2$'
            helpers.plot_surface(diff_map_nii,filename,symmetric_cbar=True,threshold=threshold,cmap=cmap_for_plotting,vmin=-limits,vmax=limits,title=subject,colorbar_label=cbar_label)
    def plot_localizer(self,task,vmin=None,vmax=None,symmetric_cbar=True):
        label = 'zscore'
        for contrast in self.localizer_contrasts[task]:
            all_data= []
            subjects_included = []
            subject_tqdm = tqdm(self.subjects[task])
            for subject in subject_tqdm:
                subject_tqdm.set_description(contrast + ': ' +subject)
                try:
                    # print(subject)
                    filepath = self.glm_dir + '/'+ subject+ '/' +subject+ "_task-"+ task+'_space-'+self.space+ "_run-"+ self.all_runs[task] + "_contrast-"+contrast+ "_measure-"+label+".nii.gz"
                    nii = nibabel.load(filepath)
                    # plot individual subject too
                    ind_filepath = self.figure_dir + '/glm_zscores/'+subject+self.glm_file_label+'_contrast-'+contrast+'_measure-'+label+'_surf'
                    cmap = self.cmaps['yellow_hot']
                    helpers.plot_surface(nii,ind_filepath,threshold=2.5,title=subject,symmetric_cbar=symmetric_cbar,vmin=vmin,vmax=vmax,cmap=cmap,colorbar_label='z-score') #0.05
                except Exception as e:
                    print(e)
                    pass
    def plot_map(self,feature='social',measure='ind_feature_performance',localizers=[],threshold=0,vmin=None,vmax=None):
        fsaverage = nilearn.datasets.fetch_surf_fsaverage(mesh='fsaverage')

        enc_file_label = '_encoding_model-'+self.model+'_smoothingfwhm-'+str(self.smoothing_fwhm)+'_chunklen-'+str(self.chunklen)
        if(self.mask_name!=None):
            enc_file_label = enc_file_label + '_mask-'+self.mask_name

        if(measure=='performance'):
            folder = 'performance'
            label = 'perf'
        if(measure=='ind_feature_performance'):
            folder = 'ind_feature_performance'
            label = 'ind_perf'
        if(measure=='ind_product_measure'):
            folder = 'ind_product_measure'
            label = 'ind_product_measure'
        elif(measure=='unique_variance'):
            folder='unique_variance'
            label = 'unique_var'

        plot_label = measure + ': ' + feature

        for subject in tqdm(self.subjects['SIpointlights']):
            # try:
                #load the unique variance of each feature
                if(measure=='unique_variance'):
                    filepath = self.enc_dir+'/'+folder+'/'+subject+enc_file_label+'_feature-'+feature+'_measure-'+label+'_raw.nii.gz'
                    nii1 = nibabel.load(filepath)
                    data1 = nii1.get_fdata()
                if(measure=='performance'):
                    filepath = self.enc_dir+'/'+folder+'/'+subject+enc_file_label+'_measure-'+label+'_raw.nii.gz'
                    nii1 = nibabel.load(filepath)
                    data1 = nii1.get_fdata()
                else:
                    filepath = self.enc_dir+'/'+folder+'/'+subject+enc_file_label+'_measure-'+label+'_raw.nii.gz'
                    nii1 = nibabel.load(filepath)
                    data1 = nii1.get_fdata()
                    if(self.scale_by=='total_variance'):
                        data1[data1<0] = 0 #clip response values to 0
                        data1 = data1/data1.sum(axis=0,keepdims=1)
                    if(feature in self.combined_features):
                        for (ind,sub_feature_name) in enumerate(self.model_features_dict[feature]):
                            feature_ind = self.get_feature_index(subject,sub_feature_name)
                            sub_data = data1[feature_ind]
                            if(ind==0):
                                overall = sub_data
                            else:
                                overall = overall+sub_data
                        data1 = overall
                    else:
                        feature_index = self.get_feature_index(subject,feature)
                        data1 = data1[feature_index]

                #project to surface

                #mask the with statistically significant performance maps for both SLIP and SimCLR

                ## TODO mask with statistically significant unique variance maps!!!

                #load the fdr corrected p value map and threshold it at p<0.05
                # threshold = threshold
                # enc_file_label = '_encoding_model-'+feature1+'_smoothingfwhm-'+str(self.smoothing_fwhm)+'_chunklen-'+str(self.chunklen)
                # if(self.mask_name!=None):
                #   enc_file_label = enc_file_label + '_mask-'+self.mask_name
                # filepath = self.enc_dir+'/unique_variance/'+subject+enc_file_label+'_feature-'+feature+'_measure-unique_var_p_fdr.nii.gz'
                # nii = nibabel.load(filepath)
                # p_fdr1 = nii.get_fdata()
                # p_fdr1_mask = p_fdr1*0
                # self.load_mask(self.mask_name)
                # self.mask = nilearn.image.resample_img(self.mask, target_affine=nii.affine, target_shape=p_fdr1.shape,interpolation='nearest')
                # p_fdr1_mask[p_fdr1>threshold]=0
                # p_fdr1_mask[p_fdr1<threshold]=1
                # p_fdr1_mask[(self.mask.get_fdata()==0)]=0

                

                diff_map = data1
                # diff_map[(p_fdr1_mask==0)]= 0 #zero out any voxel that isn't significantly predicted by either feature space
                # diff_map = np.log10(data1/data2)
                diff_map_nii = nibabel.Nifti1Image(diff_map, nii1.affine)

                #plot difference map
                # enc_file_label = '_encoding_model-'+self.model+'_smoothingfwhm-'+str(self.smoothing_fwhm)+'_chunklen-'+str(self.chunklen)
                # if(self.mask_name!=None):
                #   enc_file_label = enc_file_label + '_mask-'+self.mask_name
                # filename = self.figure_dir + '/map_and_localizer/'+subject+enc_file_label+'_measure-'+measure+'_feature-'+feature+'.png'
                # from matplotlib import colors
                # cmap = 'Greens'
                # helpers.plot_img_volume(diff_map_nii,filename,symmetric_cbar=False,threshold=0.000001,cmap=cmap,vmin=0,vmax=0.1,title=subject+', '+feature)

                glm_file_label_ = '_smoothingfwhm-'+str(self.smoothing_fwhm)
                # enc_file_label_ = '_encoding_model-full_smoothingfwhm-'+str(self.smoothing_fwhm)+'_chunklen-'+str(self.chunklen)
                enc_file_label_ = '_encoding_model-SimCLR_SLIP_smoothingfwhm-'+str(self.smoothing_fwhm)+'_chunklen-'+str(self.chunklen)
                # if(localizer!=None):
                #     if(localizer=='intact-degraded'):
                #         if(subject in self.subjects['language']):
                #             localizer_contrasts = [localizer]
                #         else:
                #             localizer_contrasts = []
                #     else:
                #         localizer_contrasts = [localizer]
                # else:
                #     localizer_contrasts = []
                # else:
                #   localizer_contrasts = []
                # localizer_contrasts = []
                # localizer_contrasts = ['social']
                # localizer_contrasts = ['SimCLR']

                all_ROI_data = np.zeros(diff_map.shape)
                ROI_niis = []
                for ind,localizer_contrast in enumerate(localizers):
                    # for mask in self.localizer_masks[localizer_contrast]:
                    file_label_ = subject+glm_file_label_+'_mask-'+'_'.join(self.localizer_masks[localizer_contrast])#mask
                    # file_label_ = subject+enc_file_label_+'_mask-'+mask+'_measure-unique_var'#+measure_label#'_encoding_model-'+self.model
                    ROI_file = self.out_dir + '/localizer_masks/'+file_label_+'_glm_loc-'+localizer_contrast+'_run-all_binary.nii.gz'
                    # ROI_file = self.out_dir + '/localizer_masks/'+file_label_+'_enc_feature_loc-'+localizer_contrast+'_binary.nii.gz'
                    try:
                        ROI_data = nibabel.load(ROI_file).get_fdata()
                        # all_ROI_data[ROI_data>0] = ind + 1
                        ROI_niis.append(nibabel.load(ROI_file))
                    except Exception as e:
                        print(e)

                # all_ROI_nii = nibabel.Nifti1Image(all_ROI_data, nii1.affine)

                cmap = self.cmaps['yellow_hot']
                filename = self.figure_dir + '/map_and_localizer/surface/'+subject+enc_file_label+'_measure-'+measure+'_feature-'+feature+'_surf'#'_loc-'+localizer_contrasts[0]+'_surf'
                # helpers.plot_lateral_surface(diff_map_nii,filename,symmetric_cbar=True,threshold=0.000000001,cmap='PuOr',vmax=None,title=subject+', '+feature1+'-'+feature2)
                helpers.plot_surface(diff_map_nii,filename,ROI_niis=ROI_niis,symmetric_cbar=False,threshold=threshold,cmap=cmap,vmin=vmin,vmax=vmax,title=subject+', '+feature, colorbar_label='Explained Variance $R^2$',ROIs=localizers,ROI_colors=[self.colors_dict[localizer] for localizer in localizers])#self.colors_dict[localizer])

            # except Exception as e:
            #   print(e)
            #   pass
    def structure_function_diff_regression(self,load=False,features=['',''],measure='ind_feature_performance',mask_names=['pMT','aMT','pSTS','aSTS'],functionally_defined_rois=False,func_type='None',func_enc_model='None',func_enc_measure='None'):
        import sklearn
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        from sklearn.preprocessing import scale
        # from sklearn.linear_model import LinearRegression
        from sklearn.linear_model import RidgeCV
        from sklearn.model_selection import cross_val_score
        from sklearn.model_selection import KFold

        enc_file_label = '_encoding_model-'+self.model+'_smoothingfwhm-'+str(self.smoothing_fwhm)+'_chunklen-'+str(self.chunklen)
        
        if(self.mask_name!=None):
            enc_file_label = enc_file_label + '_mask-'+self.mask_name

        glm_file_label = '_smoothingfwhm-'+str(self.smoothing_fwhm)
        
    
        if(measure=='performance'):
            folder = 'performance'
            label = 'perf'
        elif(measure=='ind_feature_performance'):
            folder = 'ind_feature_performance'
            label = 'ind_perf'
        elif(measure=='unique_variance'):
            folder='unique_variance'
            label = 'unique_var'

        feature1 = features[0]
        feature2 = features[1]

        plot_label = measure + ': ' + feature1 + '-'+feature2

        masks = [hemi+'-'+mask for hemi in ['left','right'] for mask in mask_names] #+ ['left-'+self.mask_name, 'right-'+self.mask_name] #adding the whole mask to the end

        glm_mask_loc_dict = {'pMT':['interact&no_interact'],
                      'aMT':['interact&no_interact'],
                      'pSTS':['interact-no_interact','intact-degraded'],
                      'aSTS':['interact-no_interact','intact-degraded'],
                      }
        enc_mask_loc_dict = {'pMT':['motion'],
                             'aMT':['motion'],
                             'pSTS':['social','sbert'],
                             'aSTS':['social','sbert']}
        master_mask_loc_dict = {'glm': glm_mask_loc_dict,
                                'encoding':enc_mask_loc_dict}
        mask_loc_dict = master_mask_loc_dict[func_type]

        model_localizer_dict = {'glm':['interact&no_interact','interact-no_interact','intact-degraded'],
                                'encoding':['motion','social','sbert']}
        localizer_contrasts = model_localizer_dict[func_type]

        beta_data = []
        score_data = []
        intercept_data = []
        mask_data = []
        subject_data = []
        masked_1_list = []
        masked_2_list = []
        ind_data_point_dict = {}
        for mask in masks:
            hemi,mask_ = mask.split('-')
            for localizer_contrast in mask_loc_dict[mask_]:
                mask_name = mask+','+localizer_contrast
                ind_data_point_dict[mask_name+',masked_1'] = []
                ind_data_point_dict[mask_name+',masked_2'] = []
        print(ind_data_point_dict)
        if(not load):
            for mask in masks:
                for subject in tqdm(self.subjects['SIpointlights'],desc=mask):
                    #load the unique variance of each feature
                    enc_file_label = '_encoding_model-'+self.model+'_smoothingfwhm-'+str(self.smoothing_fwhm)+'_chunklen-'+str(self.chunklen)
        
                    if(self.mask_name!=None):
                        enc_file_label = enc_file_label + '_mask-'+self.mask_name

                    if(measure=='unique_variance'):
                        filepath = self.enc_dir+'/'+folder+'/'+subject+enc_file_label+'_feature-'+feature1+'_measure-'+label+'_raw.nii.gz'
                        nii1 = nibabel.load(filepath)
                        data1 = nii1.get_fdata()
                    else:
                        filepath = self.enc_dir+'/'+folder+'/'+subject+enc_file_label+'_measure-'+label+'_raw.nii.gz'
                        nii1 = nibabel.load(filepath)
                        data1 = nii1.get_fdata()
                        feature_index = self.get_feature_index(subject,feature1)
                        data1 = data1[feature_index]

                    
                    enc_file_label = '_encoding_model-'+self.model+'_smoothingfwhm-'+str(self.smoothing_fwhm)+'_chunklen-'+str(self.chunklen)
                    
                    if(self.mask_name!=None):
                        enc_file_label = enc_file_label + '_mask-'+self.mask_name

                    if(measure=='unique_variance'):
                        filepath = self.enc_dir+'/'+folder+'/'+subject+enc_file_label+'_feature-'+feature2+'_measure-'+label+'_raw.nii.gz'
                        nii2 = nibabel.load(filepath)
                        data2 = nii2.get_fdata()
                    else:
                        filepath = self.enc_dir+'/'+folder+'/'+subject+enc_file_label+'_measure-'+label+'_raw.nii.gz'
                        nii2 = nibabel.load(filepath)
                        data2 = nii2.get_fdata()
                        feature_index = self.get_feature_index(subject,feature2)
                        data2 = data2[feature_index]
                    # project to surface

                    # data1[data1<0.01]=0
                    # data2[data2<0.01]=0

                    #subtract surface plots and get nii
                    # diff_map = data1-data2 #making it so SimCLR is orange, SLIP is purple, similar to Wang et al. 2023
                    if(measure=='unique_variance'):
                        # diff_map = np.log10(data1/data2)
                        diff_map = data1-data2
                    elif(measure=='ind_feature_performance'):
                        diff_map = data1-data2
                        # diff_map = np.log10(data1/data2)
                    diff_map_nii = nibabel.Nifti1Image(diff_map, nii1.affine)

                    #do a regression on the difference map with y coordinates in STS
                    self.load_mask(mask)
                    self.mask = nilearn.image.resample_img(self.mask, target_affine=diff_map_nii.affine, target_shape=diff_map.shape,interpolation='nearest')

                    #get Y coords of every voxel in STS
                    Y_index_mask = np.zeros(diff_map.shape)
                    for i in range(diff_map.shape[0]):
                        for j in range(diff_map.shape[1]):
                            for k in range(diff_map.shape[2]):
                                Y_index_mask[i, j, k] = j#np.sqrt(i**2+(j+104)**2+k**2) #x,y,z
                    X_index_mask = np.zeros(diff_map.shape)
                    for i in range(diff_map.shape[0]):
                        for j in range(diff_map.shape[1]):
                            for k in range(diff_map.shape[2]):
                                Y_index_mask[i, j, k] = i #x,y,z
                    Z_index_mask = np.zeros(diff_map.shape)
                    for i in range(diff_map.shape[0]):
                        for j in range(diff_map.shape[1]):
                            for k in range(diff_map.shape[2]):
                                Y_index_mask[i, j, k] = k #x,y,z


                    ## TODO add the functionally defined maps here!!!
                    if(functionally_defined_rois):
                        #load this subject's specific localizer map
                        hemi,mask_name = mask.split('-')
                        
                        for localizer_contrast in mask_loc_dict[mask_name]:
                            try:
                                if(func_type=='glm'):
                                    file_label = subject+glm_file_label+'_mask-'+mask_name
                                    filename = self.out_dir + '/localizer_masks/'+file_label+'_glm_loc-'+localizer_contrast+'_run-all_binary.nii.gz'
                                elif(func_type=='encoding'):
                                    enc_file_label = '_encoding_model-'+func_enc_model+'_smoothingfwhm-'+str(self.smoothing_fwhm)+'_chunklen-'+str(self.chunklen)
                                    file_label = subject+enc_file_label+'_mask-'+mask_name
                                    filename = self.out_dir + '/localizer_masks/'+file_label+'_measure-'+func_enc_measure+'_enc_feature_loc-'+localizer_contrast+'_binary.nii.gz'
                                localizer_map = nibabel.load(filename).get_fdata()

                                masked_diff_map = diff_map[(self.mask.get_fdata()==1)&(localizer_map==1)] #use both mask (gets correct hemi) and localizer map
                                
                                masked_1 = data1[(self.mask.get_fdata()==1)&(localizer_map==1)]
                                masked_2 = data2[(self.mask.get_fdata()==1)&(localizer_map==1)]

                                ind_data_point_dict[mask+','+localizer_contrast+',masked_1'].extend(masked_1)
                                ind_data_point_dict[mask+','+localizer_contrast+',masked_2'].extend(masked_2)

                                max_1 = np.max(masked_1)
                                max_2 = np.max(masked_2)

                                max_n = np.max((max_1,max_2))


                                ax = sns.scatterplot(x=masked_1,y=masked_2)
                                ax.set_xlim((-0.005,max_n))
                                ax.set_ylim((-0.005,max_n))
                                ax.set_xlabel(feature1)
                                ax.set_ylabel(feature2)
                                ax.set_title(subject+' '+mask + ' ' + localizer_contrast)
                                filename = self.figure_dir + '/scatter/'+subject+self.enc_file_label+'_measure-'+label+'_'+feature1+'-'+feature2+'_perc_top_voxels-' + self.perc_top_voxels+func_type+'_'+func_enc_measure+'_'+mask+'_'+localizer_contrast+'_scatter.pdf'
                                plt.savefig(filename,bbox_inches='tight')
                                plt.close()


                                masked_indicesY = Y_index_mask[(self.mask.get_fdata()==1)&(localizer_map==1)]
                                masked_indicesX = X_index_mask[(self.mask.get_fdata()==1)&(localizer_map==1)]
                                masked_indicesZ = Z_index_mask[(self.mask.get_fdata()==1)&(localizer_map==1)]

                                data = pd.DataFrame(zip(masked_diff_map,masked_indicesY,masked_indicesX,masked_indicesZ),columns=['difference','Ycoordinate','Xcoordinate','Zcoordinate'])
                                # print(data)
                                data.dropna(inplace=True)
                                # print(data)
                                
                                data.reset_index(inplace=True)
                                
                                scaler = StandardScaler()
                                Y = 'difference'
                                X = 'Ycoordinate'

                                #cross validated linear regression
                                cv_outer = KFold(n_splits = 5, shuffle = True, random_state = 100)
                                splits = cv_outer.split(X=range(0,len(data)))
                                intercepts = []
                                coefs = []
                                scores = []
                                for (train_outer,test_outer) in splits:
                                    # rescale the features and split
                                    split_Xtrain = scaler.fit_transform(data[X].values.reshape(-1,1)[train_outer])
                                    split_Xtest = scaler.transform(data[X].values.reshape(-1,1)[test_outer])
                                    split_Ytrain = scaler.fit_transform(data[Y].values.reshape(-1,1)[train_outer])
                                    split_Ytest = scaler.transform(data[Y].values.reshape(-1,1)[test_outer])

                                    # sns.scatterplot(x=split_Xtrain.reshape(-1),y=split_Ytrain.reshape(-1))
                                    # plt.show()

                                    # lm = LinearRegression(fit_intercept=True)
                                    lm = RidgeCV(alphas=np.logspace(-10, 10, 20))#leave one out cv
                                    lm.fit(split_Xtrain,split_Ytrain)
                                    intercepts.append(lm.intercept_)
                                    coefs.append(lm.coef_)
                                    
                                    # scores = cross_val_score(lm, split_Xtrain, split_Ytrain, scoring='r2', cv=5)
                                    split_Ypred = lm.predict(split_Xtest)
                                    r2 = sklearn.metrics.r2_score(split_Ytest, split_Ypred)
                                    #testing that things are working correctyl
                                    # split_Ypred = lm.predict(split_Xtrain)
                                    # r2 = sklearn.metrics.r2_score(split_Ytrain, split_Ypred)
                                    scores.append(r2)
                                beta_data.append(np.mean(coefs))
                                mask_data.append(mask+','+localizer_contrast)
                                subject_data.append(subject)
                                score_data.append(np.mean(scores))
                                intercept_data.append(np.mean(intercepts))
                                masked_1_list.append(masked_1)
                                masked_2_list.append(masked_2)
                            except Exception as e:
                                print('Error')
                                print(e)
                                pass

                    else:
                        masked_diff_map = diff_map[(self.mask.get_fdata()==1)]
                        
                        masked_indices = Y_index_mask[(self.mask.get_fdata()==1)]

                        data = pd.DataFrame(zip(masked_diff_map,masked_indices),columns=['difference','Ycoordinate'])
                        # print(data)
                        data.dropna(inplace=True)
                        
                        data.reset_index(inplace=True)
                        
                        scaler = StandardScaler()
                        Y = 'difference'
                        X = 'Ycoordinate'

                        #cross validated linear regression
                        cv_outer = KFold(n_splits = 5, shuffle = True, random_state = 100)
                        splits = cv_outer.split(X=range(0,len(data)))
                        intercepts = []
                        coefs = []
                        scores = []
                        for (train_outer,test_outer) in splits:
                            # rescale the features and split
                            split_Xtrain = scaler.fit_transform(data[X].values.reshape(-1,1)[train_outer])
                            split_Xtest = scaler.fit_transform(data[X].values.reshape(-1,1)[test_outer])
                            split_Ytrain = scaler.fit_transform(data[Y].values.reshape(-1,1)[train_outer])
                            split_Ytest = scaler.fit_transform(data[Y].values.reshape(-1,1)[test_outer])

                            # sns.scatterplot(x=split_Xtrain.reshape(-1),y=split_Ytrain.reshape(-1))
                            # plt.show()

                            # lm = LinearRegression(fit_intercept=True)
                            lm = RidgeCV(alphas=np.logspace(-6, 6, 13))#leave one out cv
                            lm.fit(split_Xtrain,split_Ytrain)
                            intercepts.append(lm.intercept_)
                            coefs.append(lm.coef_)
                            
                            # scores = cross_val_score(lm, split_Xtrain, split_Ytrain, scoring='r2', cv=5)
                            split_Ypred = lm.predict(split_Xtest)
                            r2 = sklearn.metrics.r2_score(split_Ytest, split_Ypred)
                            #testing that things are working correctyl
                            # split_Ypred = lm.predict(split_Xtrain)
                            # r2 = sklearn.metrics.r2_score(split_Ytrain, split_Ypred)
                            scores.append(r2)
                        beta_data.append(np.mean(coefs))
                        mask_data.append(mask)
                        subject_data.append(subject)
                        score_data.append(np.mean(scores))
                        intercept_data.append(np.mean(intercepts))

            results = pd.DataFrame(zip(subject_data,mask_data,score_data,intercept_data,beta_data),columns=['subject','mask_name','score','intercept','beta'])
            filename = self.out_dir + '/'+self.sid+self.enc_file_label+'_measure-'+label+'_diff-'+feature1+'-'+feature2+'_perc_top_voxels-' + self.perc_top_voxels+func_type+'_'+func_enc_measure+'_struct_func_regression.csv'
            results.to_csv(filename)

        filename = self.out_dir + '/'+self.sid+self.enc_file_label+'_measure-'+label+'_diff-'+feature1+'-'+feature2+'_perc_top_voxels-' + self.perc_top_voxels+func_type+'_'+func_enc_measure+'_struct_func_regression.csv'
        results = pd.read_csv(filename)

        plt.rcParams.update({'font.size': 12,'font.family': 'Arial'})

        print(results.mask_name)
        print(ind_data_point_dict)
        try:
            for mask in masks:
                hemi,mask_ = mask.split('-')
                for localizer_contrast in mask_loc_dict[mask_]:
                    mask_name = mask+','+localizer_contrast
                    masked_1 = ind_data_point_dict[mask_name+',masked_1']
                    masked_2 = ind_data_point_dict[mask_name+',masked_2']
                    print(masked_1)
                    print(masked_2)

                    # masked_1_data = []
                    # for i in masked_1:
                    #   masked_1_data.extend(i)

                    # masked_2_data = []
                    # for i in masked_2:
                    #   masked_2_data.extend(i)

                    max_1 = np.max(masked_1)
                    max_2 = np.max(masked_2)

                    max_n = np.nanmax((max_1,max_2))

                    ax = sns.scatterplot(x=masked_1,y=masked_2)
                    # ax.set_xlim((-0.01,max_n))
                    # ax.set_ylim((-0.01,max_n))
                    ax.set_xlabel(feature1)
                    ax.set_ylabel(feature2)
                    ax.set_title(self.sid+' '+mask_name)
                    filename = self.figure_dir + '/'+self.sid+self.enc_file_label+'_measure-'+label+'_'+feature1+'-'+feature2+'_perc_top_voxels-' + self.perc_top_voxels+func_type+'_'+func_enc_measure+'_'+mask_name+'_'+localizer_contrast+'_scatter.pdf'
                    plt.savefig(filename,bbox_inches='tight')
                    plt.close()
        except:
            pass


        #only keep reg results with reg scores over 0
        # results = results[results.score>0]

        if(functionally_defined_rois):
            mask_order = []
            for hemi in ['left','right']:
                for localizer_contrast in localizer_contrasts:
                    for mask in self.localizer_masks[localizer_contrast]:
                        mask_order.append(hemi+'-'+mask+','+localizer_contrast)
        else:
            mask_order = masks
        
        print('coef')
        ax = sns.barplot(results,x='mask_name',y='beta',color='gray',errorbar='se',edgecolor='black',linewidth=1.5,order=mask_order)
        sns.stripplot(results,x='mask_name',y='beta',hue='subject',alpha=0.7,order=mask_order,jitter=False)
        # sns.pointplot(results,x='mask',y='beta',hue='subject',order=mask_order)
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        if(measure=='unique_variance'):
            plt.title('predicting '+feature1 + ' unique variance - '+feature2+' unique variance with Y coordinate')
        elif(measure=='performance'):
            plt.title('predicting '+self.model + ' encoding performance with Y coordinate')        
        elif(measure=='ind_feature_performance'):
            plt.title('predicting ind ' + feature1 + ' performance - ind '+feature2+' performance with Y coordinate')
        ax.set_ylabel('coef')
        ax.set_xticklabels([mask.split(',')[0] for mask in mask_order])
        
        ax.set_xlim((-0.5,-0.5+len(mask_order)))#set xlim to cut off the weird little blank
        plt.xticks(rotation=90)
        if(functionally_defined_rois):
            if(func_type=='glm'):
                ax.set_xlabel(' motion        social       language     motion      social     language')
            elif(func_type=='encoding'):
                ax.set_xlabel('motion       social        sbert            motion      social        sbert')
            filepath = self.figure_dir+'/'+self.sid+enc_file_label+ '_measure-'+measure+'_diff-'+feature1+'-'+feature2+'_struct_func_regression_beta_'+func_type+'_'+func_enc_measure+'_funcROI.pdf'
        else:
            filepath = self.figure_dir+'/'+self.sid+enc_file_label+'_measure-'+measure+'_diff-'+feature1+'-'+feature2+'_struct_func_regression_beta_anatROI.pdf'
        pvalues = []
        pairs = []
        for mask in mask_order:
            result = stats.ttest_1samp(results[results['mask_name']==mask].beta,0)
            print(mask)
            print(result.pvalue)
            pvalues.append(result.pvalue)
            pairs.append((mask,mask))
        annot = Annotator(ax, pairs,data=results, x='mask_name',y='beta',order=mask_order)
        annot.configure(test=None, loc='inside',comparisons_correction="Bonferroni")
        annot.set_pvalues(pvalues)
        annot.annotate()
        plt.savefig(filepath,bbox_inches='tight')
        plt.close()

        # print('score')
        ax = sns.boxplot(results,x='mask_name',y='score',color='gray',order=mask_order)
        sns.stripplot(results,x='mask_name',y='score',hue='subject',order=mask_order,jitter=False)
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ax.set_ylabel('score')
        ax.set_xticklabels([mask.split(',')[0] for mask in mask_order])
        # ax.set_ylim((-0.1,0.15))
        plt.xticks(rotation=90)
        if(functionally_defined_rois):
            ax.set_xlabel('    motion        social       language     motion      social     language')
            filepath = self.figure_dir+'/'+self.sid+enc_file_label+'_measure-'+measure+'_diff-'+feature1+'-'+feature2+'_struct_func_regression_score_'+func_type+'_'+func_enc_measure+'_funcROI.pdf'
        else:
            filepath = self.figure_dir+'/'+self.sid+enc_file_label+'_measure-'+measure+'_diff-'+feature1+'-'+feature2+'_struct_func_regression_score_anatROI.pdf'
        plt.savefig(filepath,bbox_inches='tight')
        # for mask in mask_order:
        #   result = stats.ttest_1samp(results[results['mask']==mask].score,0)
        #   print(mask)
        #   print(result.pvalue)
        plt.close()

        # ax = sns.boxplot(results,x='mask',y='intercept',color='gray',order=mask_order)
        # sns.stripplot(results,x='mask',y='intercept',hue='subject',order=mask_order)
        # ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        # if(measure=='unique_variance'):
        #   plt.title('Does language alignment explain more unique variance in neural respoonses in anterior regions compared to posterior?\nregression between Y coords and log10(SLIP/SimCLR) unique variance explained')
        # elif(measure=='performance'):
        #   plt.title('Does a self-supervised vision model better predict neural responses in anterior regions compared to posterior regions?\nregression between Y coords and SimCLR encoding performance')        
        # ax.set_ylabel('intercept')
        # # ax.set_ylim((-0.1,0.15))
        # plt.xticks(rotation=90)
        # plt.savefig(self.figure_dir+'/'+self.sid+enc_file_label+'_measure-'+measure+'_diff-'+feature1+'-'+feature2+'_ant_post_regression_intercept.pdf',bbox_inches='tight')
        # plt.close()
    def structure_function_regression(self,load=False,feature='',measure='ind_feature_performance',mask_names=['ISC'],functionally_defined_rois=False,func_type='None',func_enc_model='None',func_enc_measure='None'):
        import sklearn
        # from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        from sklearn.preprocessing import scale
        from sklearn.linear_model import LinearRegression
        from sklearn.linear_model import RidgeCV
        from sklearn.model_selection import cross_val_score
        from sklearn.model_selection import KFold

        enc_file_label = '_encoding_model-'+self.model+'_smoothingfwhm-'+str(self.smoothing_fwhm)+'_chunklen-'+str(self.chunklen)
        
        if(self.mask_name!=None):
            enc_file_label = enc_file_label + '_mask-'+self.mask_name

        glm_file_label = '_smoothingfwhm-'+str(self.smoothing_fwhm)
        
    
        if(measure=='performance'):
            folder = 'performance'
            label = 'perf'
        elif(measure=='ind_feature_performance'):
            folder = 'ind_feature_performance'
            label = 'ind_perf'
        elif(measure=='ind_product_measure'):
            folder = 'ind_product_measure'
            label = 'ind_product_measure'
        elif(measure=='unique_variance'):
            folder='unique_variance'
            label = 'unique_var'

        masks = [hemi+'-'+mask for hemi in ['left','right'] for mask in mask_names] #+ ['left-'+self.mask_name, 'right-'+self.mask_name] #adding the whole mask to the end
        glm_mask_loc_dict = {'pMT':['interact&no_interact'],
                      'aMT':['interact&no_interact'],
                      'pSTS':['interact-no_interact','intact-degraded'],
                      'aSTS':['interact-no_interact','intact-degraded'],
                      'lateral':['interact&no_interact','interact-no_interact','intact-degraded'],
                      }
        enc_mask_loc_dict = {'pMT':['motion'],
                             'aMT':['motion'],
                             'pSTS':['social','sbert'],
                             'aSTS':['social','sbert'],
                             'lateral':['motion','social','sbert']}
        master_mask_loc_dict = {'glm': glm_mask_loc_dict,
                                'encoding':enc_mask_loc_dict}
        mask_loc_dict = master_mask_loc_dict[func_type]

        beta_data = []
        score_data = []
        intercept_data = []
        mask_data = []
        subject_data = []
        if(not load):
            for mask in masks:
                for subject in tqdm(self.subjects['SIpointlights'],desc=mask):
                    #load the unique variance of each feature
                    enc_file_label = '_encoding_model-'+self.model+'_smoothingfwhm-'+str(self.smoothing_fwhm)+'_chunklen-'+str(self.chunklen)
                    
                    if(self.mask_name!=None):
                        enc_file_label = enc_file_label + '_mask-'+self.mask_name

                    if(measure=='unique_variance'):
                        filepath = self.enc_dir+'/'+folder+'/'+subject+enc_file_label+'_feature-'+feature+'_measure-'+label+'_raw.nii.gz'
                        nii = nibabel.load(filepath)
                        data1 = nii.get_fdata()
                    else:
                        filepath = self.enc_dir+'/'+folder+'/'+subject+enc_file_label+'_measure-'+label+'_raw.nii.gz'
                        nii = nibabel.load(filepath)
                        data1 = nii.get_fdata()
                        feature_index = self.get_feature_index(subject,feature)
                        data1 = data1[feature_index]

                    #subtract surface plots and get nii
                    # diff_map = data1-data2 #making it so SimCLR is orange, SLIP is purple, similar to Wang et al. 2023
                    nii = nibabel.Nifti1Image(data1, nii.affine)

                    #do a regression on the difference map with y coordinates in STS
                    self.load_mask(mask)
                    self.mask = nilearn.image.resample_img(self.mask, target_affine=nii.affine, target_shape=nii.shape,interpolation='nearest')

                    #get Y coords of every voxel in STS
                    Y_index_mask = np.zeros(nii.shape)
                    for i in range(nii.shape[0]):
                        for j in range(nii.shape[1]):
                            for k in range(nii.shape[2]):
                                Y_index_mask[i, j, k] = j #x,y,z

                    if(functionally_defined_rois):
                        #load this subject's specific localizer map
                        hemi,mask_name = mask.split('-')

                        for localizer_contrast in mask_loc_dict[mask_name]:
                            try:

                                if(func_type=='glm'):
                                    file_label = subject+glm_file_label+'_mask-'+mask_name
                                    filename = self.out_dir + '/localizer_masks/'+file_label+'_glm_loc-'+localizer_contrast+'_run-all_binary.nii.gz'
                                elif(func_type=='encoding'):
                                    enc_file_label = '_encoding_model-'+func_enc_model+'_smoothingfwhm-'+str(self.smoothing_fwhm)+'_chunklen-'+str(self.chunklen)
                                    file_label = subject+enc_file_label+'_mask-'+mask_name
                                    filename = self.out_dir + '/localizer_masks/'+file_label+'_measure-'+func_enc_measure+'_enc_feature_loc-'+localizer_contrast+'_binary.nii.gz'
                                
                                localizer_map = nibabel.load(filename).get_fdata()

                                masked_map = data1[(self.mask.get_fdata()==1)&(localizer_map==1)] #use both mask (gets correct hemi) and localizer map
                                masked_indices = Y_index_mask[(self.mask.get_fdata()==1)&(localizer_map==1)]

                                data = pd.DataFrame(zip(masked_map,masked_indices),columns=['difference','Ycoordinate'])
                                # print(data)
                                data.dropna(inplace=True)
                                
                                data.reset_index(inplace=True)
                                
                                scaler = StandardScaler()
                                Y = 'difference'
                                X = 'Ycoordinate'

                                #cross validated linear regression
                                cv_outer = KFold(n_splits = 5, shuffle = True, random_state = 100)
                                splits = cv_outer.split(X=range(0,len(data)))
                                intercepts = []
                                coefs = []
                                scores = []
                                for (train_outer,test_outer) in splits:
                                    
                                    # rescale the features and split
                                    split_Xtrain = scaler.fit_transform(data[X].values.reshape(-1,1)[train_outer])
                                    split_Xtest = scaler.fit_transform(data[X].values.reshape(-1,1)[test_outer])
                                    split_Ytrain = scaler.fit_transform(data[Y].values.reshape(-1,1)[train_outer])
                                    split_Ytest = scaler.fit_transform(data[Y].values.reshape(-1,1)[test_outer])

                                    # sns.scatterplot(x=split_Xtrain.reshape(-1),y=split_Ytrain.reshape(-1))
                                    # plt.show()

                                    lm = LinearRegression(fit_intercept=True)
                                    lm.fit(split_Xtrain,split_Ytrain)
                                    intercepts.append(lm.intercept_)
                                    coefs.append(lm.coef_)
                                    
                                    # scores = cross_val_score(lm, split_Xtrain, split_Ytrain, scoring='r2', cv=5)
                                    split_Ypred = lm.predict(split_Xtest)
                                    r2 = sklearn.metrics.r2_score(split_Ytest, split_Ypred)
                                    #testing that things are working correctyl
                                    # split_Ypred = lm.predict(split_Xtrain)
                                    # r2 = sklearn.metrics.r2_score(split_Ytrain, split_Ypred)
                                    scores.append(r2)
                                beta_data.append(np.mean(coefs))
                                mask_data.append(mask+','+localizer_contrast)
                                subject_data.append(subject)
                                score_data.append(np.mean(scores))
                                intercept_data.append(np.mean(intercepts))
                            except Exception as e:
                                print(e)
                                pass
                    else:
                        masked_map = data1[(self.mask.get_fdata()==1)]
                        
                        masked_indices = Y_index_mask[(self.mask.get_fdata()==1)]

                        data = pd.DataFrame(zip(masked_map,masked_indices),columns=['difference','Ycoordinate'])
                        # print(data)
                        data.dropna(inplace=True)
                        
                        data.reset_index(inplace=True)
                        
                        scaler = StandardScaler()
                        Y = 'difference'
                        X = 'Ycoordinate'

                        #cross validated linear regression
                        cv_outer = KFold(n_splits = 5, shuffle = True, random_state = 100)
                        splits = cv_outer.split(X=range(0,len(data)))
                        intercepts = []
                        coefs = []
                        scores = []
                        for (train_outer,test_outer) in splits:
                            # rescale the features and split
                            split_Xtrain = scaler.fit_transform(data[X].values.reshape(-1,1)[train_outer])
                            split_Xtest = scaler.fit_transform(data[X].values.reshape(-1,1)[test_outer])
                            split_Ytrain = scaler.fit_transform(data[Y].values.reshape(-1,1)[train_outer])
                            split_Ytest = scaler.fit_transform(data[Y].values.reshape(-1,1)[test_outer])

                            # sns.scatterplot(x=split_Xtrain.reshape(-1),y=split_Ytrain.reshape(-1))
                            # plt.show()

                            lm = LinearRegression(fit_intercept=True)
                            lm.fit(split_Xtrain,split_Ytrain)
                            intercepts.append(lm.intercept_)
                            coefs.append(lm.coef_)
                            
                            # scores = cross_val_score(lm, split_Xtrain, split_Ytrain, scoring='r2', cv=5)
                            split_Ypred = lm.predict(split_Xtest)
                            r2 = sklearn.metrics.r2_score(split_Ytest, split_Ypred)
                            #testing that things are working correctyl
                            # split_Ypred = lm.predict(split_Xtrain)
                            # r2 = sklearn.metrics.r2_score(split_Ytrain, split_Ypred)
                            scores.append(r2)
                        beta_data.append(np.mean(coefs))
                        mask_data.append(mask)
                        subject_data.append(subject)
                        score_data.append(np.mean(scores))
                        intercept_data.append(np.mean(intercepts))

            results = pd.DataFrame(zip(subject_data,mask_data,score_data,intercept_data,beta_data),columns=['subject','mask','score','intercept','beta'])
            filename = self.out_dir + '/'+self.sid+self.enc_file_label+'_measure-'+label+'_feature'+feature+'_perc_top_voxels-' + self.perc_top_voxels+func_type+'_'+func_enc_measure+'_struct_func_regression.csv'
            results.to_csv(filename)

        filename = self.out_dir + '/'+self.sid+self.enc_file_label+'_measure-'+label+'_feature'+feature+'_perc_top_voxels-' + self.perc_top_voxels+func_type+'_'+func_enc_measure+'_struct_func_regression.csv'
        results = pd.read_csv(filename)
        plt.rcParams.update({'font.size': 12,'font.family': 'Arial'})

        file_label = '_measure-'+measure+'_feature-'+feature

        if(functionally_defined_rois):
            mask_order = []
            for hemi in ['left','right']:
              for mask in mask_names:
                  for localizer_contrast in mask_loc_dict[mask]:
                      mask_order.append(hemi+'-'+mask+','+localizer_contrast)
        else:
            mask_order = masks
        
        print('coef')
        ax = sns.barplot(results,x='mask',y='beta',color='gray',errorbar='se',edgecolor='black',linewidth=1.5,order=mask_order)
        sns.stripplot(results,x='mask',y='beta',hue='subject',alpha=0.7,order=mask_order,jitter=False)
        # sns.pointplot(results,x='mask',y='beta',hue='subject',order=mask_order)
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        if(measure=='unique_variance'):
            plt.title('predicting '+feature +' unique variance with Y coordinate')
        elif(measure=='performance'):
            plt.title('predicting '+self.model + ' encoding performance with Y coordinate')        
        elif(measure=='ind_feature_performance'):
            plt.title('predicting ind ' + feature + ' with Y coordinate')
        ax.set_ylabel('coef')
        ax.set_xticklabels([mask.split(',')[0] for mask in mask_order])
        
        ax.set_xlim((-0.5,-0.5+len(mask_order)))#set xlim to cut off the weird little blank
        plt.xticks(rotation=90)
        if(functionally_defined_rois):
            if(func_type=='glm'):
                ax.set_xlabel('    motion        social       language     motion      social     language')
            elif(func_type=='encoding'):
                ax.set_xlabel('    motion        social       sbert     motion      social     sbert')
            filepath = self.figure_dir+'/'+self.sid+enc_file_label+file_label+'_struct_func_regression_beta_'+func_type+'_'+func_enc_measure+'_funcROI.pdf'
        else:
            filepath = self.figure_dir+'/'+self.sid+enc_file_label+file_label+'_struct_func_regression_beta_anatROI.pdf'
        pvalues = []
        pairs = []
        for mask in mask_order:
            result = stats.ttest_1samp(results[results['mask']==mask].beta,0)
            print(mask)
            print(result.pvalue)
            pvalues.append(result.pvalue)
            pairs.append((mask,mask))
        annot = Annotator(ax, pairs,data=results, x='mask',y='beta',order=mask_order)
        annot.configure(test=None, loc='inside',comparisons_correction="Bonferroni")
        annot.set_pvalues(pvalues)
        annot.annotate()
        plt.savefig(filepath,bbox_inches='tight')
        plt.close()

        # print('score')
        ax = sns.boxplot(results,x='mask',y='score',color='gray',order=mask_order)
        sns.stripplot(results,x='mask',y='score',hue='subject',order=mask_order,jitter=False)
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        if(measure=='unique_variance'):
            plt.title('Does language alignment explain more unique variance in neural respoonses in anterior regions compared to posterior?\nregression between Y coords and log10(SLIP/SimCLR) unique variance explained')
        elif(measure=='performance'):
            plt.title('Does a self-supervised vision model better predict neural responses in anterior regions compared to posterior regions?\nregression between Y coords and SimCLR encoding performance')        
        ax.set_ylabel('score')
        ax.set_xticklabels([mask.split(',')[0] for mask in mask_order])
        # ax.set_ylim((-0.1,0.15))
        plt.xticks(rotation=90)
        if(functionally_defined_rois):
            ax.set_xlabel('    motion        social       language     motion      social     language')
            filepath = self.figure_dir+'/'+self.sid+enc_file_label+file_label+'_struct_func_regression_score_'+func_type+'_'+func_enc_measure+'_funcROI.pdf'
        else:
            filepath = self.figure_dir+'/'+self.sid+enc_file_label+file_label+'_struct_func_regression_score_anatROI.pdf'
        plt.savefig(filepath,bbox_inches='tight')
        # for mask in mask_order:
        #   result = stats.ttest_1samp(results[results['mask']==mask].score,0)
        #   print(mask)
        #   print(result.pvalue)
        plt.close()

        # ax = sns.boxplot(results,x='mask',y='intercept',color='gray',order=mask_order)
        # sns.stripplot(results,x='mask',y='intercept',hue='subject',order=mask_order)
        # ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        # if(measure=='unique_variance'):
        #   plt.title('Does language alignment explain more unique variance in neural respoonses in anterior regions compared to posterior?\nregression between Y coords and log10(SLIP/SimCLR) unique variance explained')
        # elif(measure=='performance'):
        #   plt.title('Does a self-supervised vision model better predict neural responses in anterior regions compared to posterior regions?\nregression between Y coords and SimCLR encoding performance')        
        # ax.set_ylabel('intercept')
        # # ax.set_ylim((-0.1,0.15))
        # plt.xticks(rotation=90)
        # plt.savefig(self.figure_dir+'/'+self.sid+enc_file_label+'_measure-'+measure+'_diff-'+feature1+'-'+feature2+'_ant_post_regression_intercept.pdf',bbox_inches='tight')
        # plt.close()

    def run(self):
        self.subjects = helpers.get_subjects(self.population)

        #TODO tAKE out!!
        # self.feature_names.remove('turn_taking')
        print(self.subjects)
        main_pipeline=True

        self.remove_right_lateralized_subjects=False
        load = True
        plot=True
        pvalue=0.05
        plt.rcParams.update({'font.size': 16})

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model','-model',type=str,default='full')
    parser.add_argument('--task','-task',type=str,default='sherlock')
    parser.add_argument('--mask','-mask',type=str, default='ISC') #the mask that contains all masks of interest (overarching mask )
    parser.add_argument('--perc-top-voxels','-perc-top-voxels',type=int,default=None)
    parser.add_argument('--space','-space',type=str,default='MNI152NLin2009cAsym')
    parser.add_argument('--smoothing-fwhm','-smoothing-fwhm',type=float,default=0.0)
    parser.add_argument('--chunklen','-chunklen',type=int,default=17)
    parser.add_argument('--population','-population',type=str,default='NT')

    parser.add_argument('--dir', '-dir', type=str,
                        default='/Users/hsmall2/Documents/GitHub/deep_nat_lat')
    parser.add_argument('--out_dir', '-output', type=str,
                        default='/Users/hsmall2/Documents/GitHub/deep_nat_lat/analysis')
    parser.add_argument('--figure_dir', '-figures', type=str,
                        default='/Users/hsmall2/Documents/GitHub/deep_nat_lat/figures')
    args = parser.parse_args()
    SecondLevelIndividual(args).run()

if __name__ == '__main__':
    main()