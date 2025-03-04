import secondlevel_individual_himalaya
import secondlevel_group_himalaya
import secondlevel_group_glm
import intersubject_correlation
import argparse
from matplotlib import colors

def construct_args(args_dict):
	parser = argparse.ArgumentParser()
	parser.add_argument('--model','-model',type=str,default=args_dict['model'])
	parser.add_argument('--task','-task',type=str,default=args_dict['task'])
	parser.add_argument('--mask','-mask',type=str, default=args_dict['mask']) #the mask that contains all masks of interest (overarching mask )
	parser.add_argument('--perc-top-voxels','-perc-top-voxels',type=int,default=args_dict['perc_top_voxels'])
	parser.add_argument('--space','-space',type=str,default=args_dict['space'])
	parser.add_argument('--smoothing-fwhm','-smoothing-fwhm',type=float,default=args_dict['smoothing_fwhm'])
	parser.add_argument('--chunklen','-chunklen',type=int,default=args_dict['chunklen'])
	parser.add_argument('--population','-population',type=str,default=args_dict['population'])
	parser.add_argument('--feature-of-interest','-feature-of-interest',type=str,default=args_dict['feature_of_interest'])
	parser.add_argument('--ISC-type','-ISC-type',type=str,default=args_dict['ISC_type'])

	parser.add_argument('--dir', '-dir', type=str,
						default=args_dict['dir'])
	parser.add_argument('--out_dir', '-output', type=str,
						default=args_dict['out_dir'])
	parser.add_argument('--figure_dir', '-figures', type=str,
						default=args_dict['figure_dir'])
	args = parser.parse_args()
	return args
def get_default_args():
	args_dict = { 
				'model': None,
				'task':	'sherlock',
				'mask':	'ISC',
				'perc_top_voxels':	5,
				'space': 'MNI152NLin2009cAsym',
				'smoothing_fwhm': 3.0,
				'chunklen': 20,
				'population': 'NT',
				'feature_of_interest': None,
				'ISC_type':'leave_one_out',
				'dir': '/Users/hsmall2/Documents/GitHub/deep_nat_lat',
				'out_dir': '/Users/hsmall2/Documents/GitHub/deep_nat_lat/analysis',
				'figure_dir': '/Users/hsmall2/Documents/GitHub/deep_nat_lat/figures/second_project'
			}
	return args_dict
ISC=False
group = False
individual = True
def plot_secondlevel_group_glm(task):
	args_dict = get_default_args()
	args_dict['task']=task
	args_dict['mask']=None
	args = construct_args(args_dict)

	### GROUP ANALYSIS
	secondlevel_group = secondlevel_group_glm.SecondLevelGroup(args)
	secondlevel_group.do_stats = False

	secondlevel_group.compile_data()
	# secondlevel_group.plot_weights()
	secondlevel_group.plot_zscores(vmin=0,threshold=2.5,symmetric_cbar=False) #0.05 - 1.72, 0.01 -2.5, 0.001 - 3.6
def plot_secondlevel_group(model,plotting,feature_names,diff_cmap=''):
	args_dict = get_default_args()
	args_dict['model']=model
	args = construct_args(args_dict)

	### GROUP ANALYSIS
	secondlevel_group = secondlevel_group_himalaya.SecondLevelGroup(args)
	secondlevel_group.do_stats = False
	secondlevel_group.explainable_variance_mask = False
	secondlevel_group.feature_names = feature_names
	secondlevel_group.scale_by=''
	secondlevel_group.compile_data(smooth=0)
	if('preference_map' in plotting):
		secondlevel_group.compute_preference_maps(restricted=False)
	secondlevel_group.save_results()
	if('performance' in plotting):
		secondlevel_group.plot_performance('raw',threshold=0.001,vmin=0,vmax=None)
	if('ind_product_measure' in plotting):
		secondlevel_group.plot_ind_product_measure('raw',threshold=0.001,vmin=0,vmax=0.16)
	if('preference_map' in plotting):
		secondlevel_group.plot_preference_maps(label='both')
	if('difference' in plotting):
		secondlevel_group.plot_difference('raw',threshold=0.001,cmap=diff_cmap,vmax=0.15)
	
if(ISC):
	args_dict = get_default_args()
	args_dict['mask']='None'
	args_dict['smoothing_fwhm']=6.0
	args = construct_args(args_dict)
	ISC = intersubject_correlation.IntersubjectCorrelation(args)
	ISC.plot_intersubject_correlation(threshold=0.15,vmin=None,vmax=None)
if(group):

	plotting = ['performance']
	feature_names = ['SLIPtext']
	plot_secondlevel_group('SLIPtext',plotting,feature_names)

	plotting = ['ind_product_measure']#,'performance','ind_product_measure','difference','preference']
	feature_names = ['SimCLR_attention','SimCLR_embedding','SLIP_attention','SLIP_embedding']
	cmap = colors.LinearSegmentedColormap.from_list('my_gradient', (
				    # Edit this gradient at https://eltos.github.io/gradient/#7B5300-FFFFFF-005749
				    (0.000, (0.482, 0.325, 0.000)),
				    (0.500, (1.000, 1.000, 1.000)),
				    (1.000, (0.000, 0.341, 0.286))))
	plot_secondlevel_group('GPT2_SimCLR_SLIP_word2vec',plotting,feature_names,diff_cmap=cmap)

	plotting = ['preference_map']#,'performance','ind_product_measure','difference','preference']
	feature_names = ['SimCLR_attention','SimCLR_embedding','GPT2_1sent','word2vec']
	cmap = colors.LinearSegmentedColormap.from_list('my_gradient', (
				    # Edit this gradient at https://eltos.github.io/gradient/#7B5300-FFFFFF-005749
				    (0.000, (0.482, 0.325, 0.000)),
				    (0.500, (1.000, 1.000, 1.000)),
				    (1.000, (0.000, 0.341, 0.286))))
	plot_secondlevel_group('SimCLR_SLIP_SLIPtext',plotting,feature_names,diff_cmap=cmap)


### INDIVIDUAL ANALYSIS
if(individual):
	def get_secondlevel_individual(model):
		args_dict = get_default_args()
		args_dict['model']=model
		args = construct_args(args_dict)

		secondlevel_ind = secondlevel_individual_himalaya.SecondLevelIndividual(args)
		secondlevel_ind.localizer_contrasts = {'SIpointlights':{'interact&no_interact','interact-no_interact'},'social interaction':{'interact&no_interact','interact-no_interact'},'language':{'intact-degraded'}}
		secondlevel_ind.remove_right_lateralized_subjects=False
		MT = ['MT']#['MT']#['pMT','aMT']
		ISC = ['ISC']#['pISC','aISC']#['lateral']
		STS = ['pSTS','aSTS']#['pSTS','aSTS']#
		language =['pTemp','aTemp']#['temporal_language']#['pTemp','aTemp']#['temporal_language','frontal_language']
		language_ROI_names = ['pTemp','aTemp']#,'frontal language']#['pSTS','aSTS'] #label names!['temporal language']#
		secondlevel_ind.localizer_masks = {'interact&no_interact':MT,'interact-no_interact':STS,'intact-degraded':language,
								'motion pointlights':MT,'SI pointlights':STS, 'language':language,
								'social interaction':STS,
								'social':STS,'GPT2':STS,'sbert':language,'sbert_layers':language,'sbert+word2vec':language,
								
								'motion':MT,'num_agents':STS, 'alexnet':STS,'alexnet_layers':STS,
								'valence':STS,'face':STS,'mentalization':STS, 'arousal':STS,
								'SLIP':STS,'SimCLR':STS,'CLIP':language, 
								'SLIP_attention':STS,'SimCLR_attention':STS,'SLIP_embedding':STS,'SimCLR_embedding':STS,
								'glove':language,'word2vec':STS,
								'speaking':STS,'indoor_outdoor':STS,'pitch':STS,'amplitude':STS,
								'turn_taking':STS,'written_text':STS,'music':STS,'pixel':STS,'hue':STS,'none':STS}
		secondlevel_ind.MT = MT
		secondlevel_ind.ISC = ISC
		secondlevel_ind.STS = STS
		secondlevel_ind.language = language
		secondlevel_ind.language_ROI_names = language_ROI_names
		#['social','num_agents','face','valence','arousal','speaking','turn_taking','mentalization','word2vec','sbert','written_text','pitch','amplitude','music','pixel','hue','motion','alexnet']
		
		return secondlevel_ind
	
	
####### language aligned vs non-language aligned visual representations ########
	secondlevel_ind_SimCLR_SLIP = get_secondlevel_individual('SimCLR_SLIP')
	secondlevel_ind_SimCLR_SLIP.plot_features_dict['SimCLR_SLIP'] = ['SimCLR_attention_layer'+str(layer) for layer in [1,2,3,4,5,6,7,8,9,10,11,12]]+['SLIP_attention_layer'+str(layer) for layer in [1,2,3,4,5,6,7,8,9,10,11,12]]+['SimCLR_embedding_layer'+str(layer) for layer in [1,2,3,4,5,6,7,8,9,10,11,12]]+['SLIP_embedding_layer'+str(layer) for layer in [1,2,3,4,5,6,7,8,9,10,11,12]]#,'motion','alexnet']
	secondlevel_ind_SimCLR_SLIP.plot_features = ['SimCLR_attention_layer'+str(layer) for layer in [1,2,3,4,5,6,7,8,9,10,11,12]]+['SLIP_attention_layer'+str(layer) for layer in [1,2,3,4,5,6,7,8,9,10,11,12]]+['SimCLR_embedding_layer'+str(layer) for layer in [1,2,3,4,5,6,7,8,9,10,11,12]]+['SLIP_embedding_layer'+str(layer) for layer in [1,2,3,4,5,6,7,8,9,10,11,12]]
	secondlevel_ind_SimCLR_SLIP.plot_features_dict['full'] = ['motion','social','sbert+word2vec']

	enc_localizers_to_plot = secondlevel_ind_SimCLR_SLIP.plot_features_dict['SimCLR_SLIP']

	secondlevel_ind_SimCLR_SLIP.glm_voxel_selection(load=False,plot_ind=True,response_label='performance',pvalue=None,localizers_to_plot=glm_localizers_to_plot,localizer_label_dict=glm_localizer_label_dict,plot_noise_ceiling=True)
	secondlevel_ind_SimCLR_SLIP.preference_map_summary(load=False,ROIs = ['interact&no_interact','interact-no_interact','intact-degraded'],ROI_type='functional') 

	# extract everything first, then can plot
	all_features = ['SLIP','SimCLR','SLIP_attention','SLIP_embedding','SimCLR_attention','SimCLR_embedding']+['SimCLR_attention_layer'+str(layer) for layer in [1,2,3,4,5,6,7,8,9,10,11,12]]+['SLIP_attention_layer'+str(layer) for layer in [1,2,3,4,5,6,7,8,9,10,11,12]]+['SimCLR_embedding_layer'+str(layer) for layer in [1,2,3,4,5,6,7,8,9,10,11,12]]+['SLIP_embedding_layer'+str(layer) for layer in [1,2,3,4,5,6,7,8,9,10,11,12]]
	secondlevel_ind_SimCLR_SLIP.feature_names = all_features
	secondlevel_ind_SimCLR_SLIP.plot_features = all_features
	secondlevel_ind_SimCLR_SLIP.glm_voxel_selection(load=False,stats_to_do=None,plot_stacked=False,plot_ind=True,response_label='ind_product_measure',pvalue=None,localizers_to_plot=glm_localizers_to_plot,localizer_label_dict=glm_localizer_label_dict)

	glm_localizers_to_plot = ['motion','social interaction', 'language']
	filepath_tag = 'attention_vs_embedding'
	features = ['SimCLR_attention','SLIP_attention','SimCLR_embedding','SLIP_embedding']
	secondlevel_ind_SimCLR_SLIP.feature_names = features
	secondlevel_ind_SimCLR_SLIP.plot_features = features
	secondlevel_ind_SimCLR_SLIP.scale_by = 'total_variance'
	secondlevel_ind_SimCLR_SLIP.glm_voxel_selection(load=True,stats_to_do='compare_features',plot_stacked=True,plot_ind=True,response_label='ind_product_measure',pvalue=None,localizers_to_plot=glm_localizers_to_plot,localizer_label_dict=glm_localizer_label_dict,filepath_tag=filepath_tag)
	
