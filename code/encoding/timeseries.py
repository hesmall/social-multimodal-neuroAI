import argparse
import glob
from pathlib import Path
import numpy as np
import nibabel
from nibabel import processing
import nilearn
import nilearn.datasets
from scipy import stats
import sys
sys.path.insert(1, './encoding')
import encoding
import seaborn as sns
import matplotlib.pyplot as plt
class TimeSeries(encoding.EncodingModel):

    def __init__(self, args):
        self.testing = args.testing
        self.process = 'TimeSeries'
        self.dir = args.dir
        self.data_dir = args.data_dir
        self.out_dir = args.out_dir + "/" + self.process
        self.sid = 'sub-'+str(int(args.s_num)).zfill(2)
        self.task = args.task
        self.mask_name = args.mask
        self.space = args.space
        self.mask = None
        self.mask_affine = None
        self.smoothing_fwhm = args.smoothing_fwhm #change?
        self.fMRI_data = []
        self.brain_shape = (97,115,97)
        self.affine = []
        self.wait_TR = 2
        self.stim_start = 26
        #previously 28, 948 and 988,1989? might have been getting higher performance -- is this where it should be trimmed?
        self.included_data_fMRI = [(self.stim_start+self.wait_TR,946+self.wait_TR),(975+self.wait_TR,1976+self.wait_TR)]
        Path(f'{self.out_dir}/').mkdir(exist_ok=True, parents=True)
        
        self.figure_dir = args.figure_dir + "/" + self.process
        Path(f'{self.figure_dir}/').mkdir(exist_ok=True, parents=True)

    def save_results(self):
        file_label = file_label = self.sid+'_smoothingfwhm-'+str(self.smoothing_fwhm)
        if(len(self.fMRI_data)>0):
            actual_time_series = self.unmask_reshape(self.fMRI_data)
            img = nibabel.Nifti1Image(actual_time_series,self.affine)
            nibabel.save(img, self.out_dir+'/'+file_label+'_mask-'+self.mask_name+'_measure-actual_time_series.nii.gz') 
            print('saved: actual_time_series')
            print(img.shape)
    def plot_timeseries(self):
        file_label = self.sid+'_smoothingfwhm-'+str(self.smoothing_fwhm)
        filepath = self.out_dir+'/'+file_label+'_mask-'+self.mask_name+'_measure-actual_time_series.nii.gz'
        data = nibabel.load(filepath).get_fdata()
        data = data[:,self.mask==1]
        print(data[data>0])
        average = np.nanmean(data,axis=1)
        print(average)

        fig = plt.figure(figsize=(30,10))
        sns.lineplot(x=np.arange(0,average.shape[0]),y=average)
        filepath = self.figure_dir+'/'+file_label+'_mask-'+self.mask_name+'_measure-averaged_time_series.png'
        plt.savefig(filepath)


    def run(self):
        self.explainable_variance_mask=False
        self.load_mask()
        self.load_preprocess_fMRI(smooth=True,denoise=True)
        self.trim_fMRI(norm=True)
        self.save_results()
        # self.plot_timeseries()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--s_num', '-s', type=str, default='1')
    parser.add_argument('--task','-task',type=str,default='sherlock')
    parser.add_argument('--mask','-mask',type=str,default='ISC')
    parser.add_argument('--space','-space',type=str, default='MNI152NLin2009cAsym_res-2')
    parser.add_argument('--smoothing-fwhm','-smoothing-fwhm',type=float,default=3.0)
    parser.add_argument('--testing','-testing',type=str,default=None) 
    #load -- loads precomputed weights and performances, no encoding or permutation testing
    #quickrun -- runs an abbreviated encoding model and permutation testing (fewer lambdas, folds, and iterations)


    parser.add_argument('--dir', '-dir', type=str,
                        default='/Users/hsmall2/Documents/GitHub/deep_nat_lat')
    parser.add_argument('--data_dir', '-data_dir', type=str,
                        default='/Users/hsmall2/Documents/GitHub/Sherlock_ASD/data') #where the derivatives folder for the fMRI data is 
    parser.add_argument('--out_dir', '-output', type=str,
                        default='/Users/hsmall2/Documents/GitHub/deep_nat_lat/data')
    parser.add_argument('--figure_dir', '-figures', type=str,
                        default='/Users/hsmall2/Documents/GitHub/deep_nat_lat/figures')
    args = parser.parse_args()
    TimeSeries(args).run()

if __name__ == '__main__':
    main()