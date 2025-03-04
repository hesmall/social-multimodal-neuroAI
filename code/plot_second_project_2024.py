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
	pass
	# plot_secondlevel_group_glm('SIpointlights')
	# plot_secondlevel_group_glm('language')

	# plotting = ['performance']
	# feature_names = ['social']
	# plot_secondlevel_group('social',plotting,feature_names)

	# plotting = ['performance']
	# feature_names = ['motion']
	# plot_secondlevel_group('motion',plotting,feature_names)

	# plotting = ['performance']
	# feature_names = ['word2vec']
	# plot_secondlevel_group('word2vec',plotting,feature_names)

	# plotting = ['performance']
	# feature_names = ['sbert']
	# plot_secondlevel_group('sbert',plotting,feature_names) 

	# plotting = ['performance']
	# feature_names = ['glove']
	# plot_secondlevel_group('glove',plotting,feature_names)

	plotting = ['performance']
	feature_names = ['SLIPtext']
	plot_secondlevel_group('SLIPtext',plotting,feature_names)

	# plotting = ['performance','ind_product_measure','preference_map']#,'preference_map']
	# feature_names = ['sbert_layer'+str(layer) for layer in [1,2,3,4,5,6,7,8,9,10,11,12]]
	# plot_secondlevel_group('sbert_layers',plotting,feature_names) 

	# plotting = ['performance','ind_product_measure','preference_map']#['performance','ind_product_measure']#,'preference_map']
	# feature_names = ['alexnet_layer'+str(layer) for layer in [1,2,3,4,5,6,7]]
	# plot_secondlevel_group('alexnet_layers',plotting,feature_names) 

	# plotting = ['difference','ind_product_measure']#['ind_product_measure','performance','preference_map'] #'preference map','difference', 'performance'
	# feature_names = ['sbert+word2vec','alexnet','sbert','word2vec']#['sbert_layers','alexnet_layers','word2vec','social','motion']#',['sbert_layer'+str(layer) for layer in [1,2,3,4,5,6,7,8,9,10,11,12]]+['alexnet_layer'+str(layer) for layer in [1,2,3,4,5,6,7]]#
	# cmap='coolwarm' #for difference of first 2 features
	# plot_secondlevel_group('full',plotting,feature_names,diff_cmap=cmap)

	# plotting = ['ind_product_measure']#,'performance','ind_product_measure','difference','preference']
	# feature_names = ['SimCLR_attention','SimCLR_embedding','GPT2_1sent','word2vec']
	# feature_names = ['SimCLR_attention_layer'+str(layer) for layer in [1,2,3,4,5,6,7,8,9,10,11,12]] + ['SLIP_attention_layer'+str(layer) for layer in [1,2,3,4,5,6,7,8,9,10,11,12]]#+['SimCLR_attention_layer'+str(layer) for layer in [1,2,3,4,5,6,7,8,9,10,11,12]]+['SimCLR_attention_layer'+str(layer) for layer in [1,2,3,4,5,6,7,8,9,10,11,12]]
	# feature_names=['SimCLR+SLIP','GPT2+word2vec']
	# feature_names = ['SimCLR_attention','SimCLR_embedding','SLIP_attention','SLIP_embedding']
	# cmap = colors.LinearSegmentedColormap.from_list('my_gradient', (
	# 			    # Edit this gradient at https://eltos.github.io/gradient/#7B5300-FFFFFF-005749
	# 			    (0.000, (0.482, 0.325, 0.000)),
	# 			    (0.500, (1.000, 1.000, 1.000)),
	# 			    (1.000, (0.000, 0.341, 0.286))))
	# plot_secondlevel_group('GPT2_SimCLR_SLIP_word2vec',plotting,feature_names,diff_cmap=cmap)

	plotting = ['preference_map']#,'performance','ind_product_measure','difference','preference']
	feature_names = ['SimCLR_attention','SimCLR_embedding','GPT2_1sent','word2vec']
	feature_names = ['SimCLR_attention_layer'+str(layer) for layer in [1,2,3,4,5,6,7,8,9,10,11,12]] + ['SLIP_attention_layer'+str(layer) for layer in [1,2,3,4,5,6,7,8,9,10,11,12]]#+['SimCLR_attention_layer'+str(layer) for layer in [1,2,3,4,5,6,7,8,9,10,11,12]]+['SimCLR_attention_layer'+str(layer) for layer in [1,2,3,4,5,6,7,8,9,10,11,12]]
	feature_names=['SimCLR+SLIP','GPT2+word2vec']
	feature_names = ['SimCLR_attention','SimCLR_embedding','SLIP_attention','SLIP_embedding']
	feature_names = ['SimCLR','SLIP','SLIPtext']
	feature_names = ['SimCLR_attention_layer'+str(layer) for layer in [1,2,3,4,5,6,7,8,9,10,11,12]] + ['SimCLR_embedding_layer'+str(layer) for layer in [1,2,3,4,5,6,7,8,9,10,11,12]]
	feature_names = ['SLIP_attention_layer'+str(layer) for layer in [1,2,3,4,5,6,7,8,9,10,11,12]] + ['SLIP_embedding_layer'+str(layer) for layer in [1,2,3,4,5,6,7,8,9,10,11,12]]
	cmap = colors.LinearSegmentedColormap.from_list('my_gradient', (
				    # Edit this gradient at https://eltos.github.io/gradient/#7B5300-FFFFFF-005749
				    (0.000, (0.482, 0.325, 0.000)),
				    (0.500, (1.000, 1.000, 1.000)),
				    (1.000, (0.000, 0.341, 0.286))))
	plot_secondlevel_group('SimCLR_SLIP_SLIPtext',plotting,feature_names,diff_cmap=cmap)


def get_overlap(models,masks,glm_localizer_label_dict,enc_localizer_label_dict,load=False,file_tag='',n_top_voxels=100):
	
	models_enc_feature_to_plot_dict = { 'full':['social','sbert+word2vec'],#'num_agents','face','valence','arousal','speaking','turn_taking','mentalization','sbert_layers','written_text','pitch','amplitude','music','pixel','hue','motion','alexnet'],
										'social':['social'],
										'sbert+word2vec':['sbert+word2vec'] }

	file_tag = file_tag + '_' + str(n_top_voxels)
	regions = []
	for model in models:
		secondlevel = get_secondlevel_individual(model)
		secondlevel.n_top_voxels=str(n_top_voxels)
		#runs glm and encoding voxel selection again with the same mask across both domains
		glm_localizers_to_plot = ['social interaction', 'language']
		enc_localizers_to_plot = models_enc_feature_to_plot_dict[model]

		all_localizers = glm_localizers_to_plot + enc_localizers_to_plot
		for localizer in all_localizers:
			secondlevel.localizer_masks[localizer]=masks
			enc_localizer_label_dict[localizer]='STS'
		secondlevel.localizer_masks['interact-no_interact']=masks
		secondlevel.localizer_masks['intact-degraded']=masks

		secondlevel.localizer_contrasts = {'SIpointlights':{'interact-no_interact'},'social interaction':{'interact-no_interact'},'language':{'intact-degraded'}}
		secondlevel.language = masks
		secondlevel.language_ROI_names = masks
		secondlevel.glm_voxel_selection(load=load,plot_ind=True,response_label='performance',pvalue=None,plot_stacked=False,localizers_to_plot=glm_localizers_to_plot,localizer_label_dict=glm_localizer_label_dict,filepath_tag=file_tag)
	
		if(not load):
			secondlevel.generate_binary_localizer_maps_glm(glm_task='SIpointlights',plot=False)
			secondlevel.generate_binary_localizer_maps_glm(glm_task='language',plot=False)
		secondlevel.plot_features_dict['full']=enc_localizers_to_plot
		secondlevel.plot_features = enc_localizers_to_plot
		secondlevel.feature_names = enc_localizers_to_plot #for stacked barplot
		if(model=='full'):
			selection_label='ind_product_measure'
		else:
			selection_label='performance'
		secondlevel.encoding_voxel_selection(load=load,pvalue=None,selection_model=model,selection_label=selection_label,response_label='performance',localizers_to_plot=enc_localizers_to_plot,localizer_label_dict=enc_localizer_label_dict,filepath_tag=file_tag)
		if(not load):
			secondlevel.generate_binary_localizer_maps_enc(model=model,label=selection_label,plot=False)
		if(model=='full'):
			selection_label='ind_product_measure'
		else:
			selection_label='perf'
		for feature in enc_localizers_to_plot:
			regions.append((feature,'encoding-'+selection_label+'_raw',model,'_'.join(masks)))
			# regions_aSTS.append((feature,'encoding-'+selection_label+'_raw',model,'aSTS'))
	regions.extend([('interact-no_interact','glm-SIpointlights','NA','_'.join(masks)),('intact-degraded','glm-language','NA','_'.join(masks))])
	secondlevel.compute_all_overlap2(load=load,regions=regions,label='_'.join(masks),file_tag=file_tag)


	# secondlevel.compute_all_overlap2(load=load,regions=regions_aSTS,label='aSTS',file_tag=file_tag)

	# secondlevel.compute_all_overlap3(load=load,regions=regions_pSTS,label='pSTS')

	# secondlevel.compute_all_overlap3(load=load,regions=regions_aSTS,label='aSTS')


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
	
	# secondlevel_ind_word2vec = get_secondlevel_individual('word2vec')
	# secondlevel_ind_word2vec.plot_map(feature='word2vec',measure='performance',localizer='intact-degraded')
	
	# secondlevel_ind_sbert = get_secondlevel_individual('sbert')
	# secondlevel_ind_sbert.plot_map(feature='sbert',measure='performance',localizer='intact-degraded')

	# secondlevel_ind_GPT2 = get_secondlevel_individual('GPT2_1sent')
	# secondlevel_ind_GPT2.plot_map(feature='GPT2_1sent',measure='performance')

	# secondlevel_ind_GPT2 = get_secondlevel_individual('GPT2_2sent')
	# secondlevel_ind_GPT2.plot_map(feature='GPT2_2sent',measure='performance')

	# secondlevel_ind_social = get_secondlevel_individual('social')
	# secondlevel_ind_social.plot_map(feature='social',measure='performance',localizer='interact-no_interact',threshold=0.01)

	# secondlevel_ind_social = get_secondlevel_individual('motion')
	# secondlevel_ind_social.plot_map(feature='motion',measure='performance',localizer='interact&no_interact')

	# secondlevel_ind_alexnet = get_secondlevel_individual('alexnet')
	# secondlevel_ind_alexnet.plot_map(feature='alexnet',measure='performance',localizer=False)

	secondlevel_ind_full = get_secondlevel_individual('joint')
	# secondlevel_ind_full.plot_map(feature='full',measure='performance',vmin=0,vmax=0.3)

	## plotting info
	secondlevel_ind_full.plot_features_dict['joint'] = ['motion','social','sbert+word2vec']#['word2vec','sbert_layers','social','alexnet_layers','motion']#,'motion','alexnet']

	glm_localizers_to_plot = ['motion', 'social interaction', 'language']#'motion pointlights',
	glm_localizer_label_dict = {'motion':'MT','social interaction':'STS','language':'language'}
	enc_localizers_to_plot = secondlevel_ind_full.plot_features_dict['joint']
	enc_localizer_label_dict = {'motion':'MT',
									'num_agents':'STS',
									'social':'STS',
									'speaking':'STS',
									'turn_taking':'STS',
									'written_text':'ISC',
									'pitch':'ISC',
									'amplitude':'STS',
									'pixel':'ISC',
									'hue':'ISC',
									'music':'STS',
									'mentalization':'STS',
									'valence':'STS',
									'alexnet':'STS',
									'alexnet_layers':'STS',
									'sbert+word2vec':'language',
									'face':'STS',
									'SLIP':'STS',
									'SimCLR':'STS',
									'arousal':'STS',
									'glove':'STS',
									'sbert':'language',
									'sbert_layers':'language',
									'GPT2':'STS',
									'word2vec':'language',
									'CLIP':'STS',
									'CLIPtext':'STS',
									'SLIP_attention':'STS',
									'SimCLR_attention':'STS',
									'SLIP_embedding':'STS',
									'SimCLR_embedding':'STS'}
	secondlevel_ind_full.scale_by = None

	# ## extract responses and plot! 
	# secondlevel_ind_full.glm_voxel_selection(load=False,plot_ind=True,stats_to_do='compare_to_zero',response_label='performance',pvalue=None,localizers_to_plot=glm_localizers_to_plot,localizer_label_dict=glm_localizer_label_dict,plot_noise_ceiling=False)
	# secondlevel_ind_full.preference_map_summary(load=False,ROIs = ['interact&no_interact','interact-no_interact','intact-degraded'],ROI_type='functional') 

	# all_features = ['social','num_agents','face','valence','arousal','speaking','turn_taking','mentalization','word2vec','sbert','written_text','pitch','amplitude','music','pixel','hue','motion','alexnet']
	# secondlevel_ind_full.feature_names = all_features
	# secondlevel_ind_full.plot_features = all_features
	# secondlevel_ind_full.generate_preference_maps(restricted=True,threshold=0.01,features=all_features,file_tag='all_features')
	# features = ['social','num_agents','face','valence','arousal','speaking','turn_taking','mentalization','written_text','pitch','amplitude','music','pixel','hue']
	# secondlevel_ind_full.generate_preference_maps(restricted=True,threshold=0.01,features=features,file_tag='unidim_features')#threshold is proportion
	# features = ['sbert_layer'+str(layer) for layer in [12,11,10,9,8,7,6,5,4,3,2,1]]
	# secondlevel_ind_full.generate_preference_maps(restricted=True,threshold=0.01,features=features,file_tag='sbert')
	# features = ['alexnet_layer'+str(layer) for layer in [7,6,5,4,3,2,1]]
	# secondlevel_ind_full.generate_preference_maps(restricted=True,threshold=0.01,features=features,file_tag='alexnet')
	# secondlevel_ind_full.scale_by = 'total_variance'
	# secondlevel_ind_full.glm_voxel_selection(load=True,stats_to_do=None,plot_stacked=True,plot_ind=True,response_label='ind_product_measure',pvalue=None,localizers_to_plot=glm_localizers_to_plot,localizer_label_dict=glm_localizer_label_dict)
	

	# glm_localizers_to_plot = ['motion', 'social interaction', 'language']#'motion pointlights',
	# secondlevel_ind_full.glm_voxel_selection(load=False,plot_ind=True,stats_to_do='compare_to_zero',response_label='performance',pvalue=None,localizers_to_plot=glm_localizers_to_plot,localizer_label_dict=glm_localizer_label_dict,plot_noise_ceiling=True)
	
	# filepath_tag = 'all_features_layers_compiled'
	# all_features = ['social','num_agents','face','valence','arousal','speaking','turn_taking','mentalization','word2vec','sbert','written_text','pitch','amplitude','music','pixel','hue','motion','alexnet']
	# secondlevel_ind_full.feature_names = all_features
	# secondlevel_ind_full.plot_features = all_features
	# secondlevel_ind_full.scale_by = 'total_variance'
	# secondlevel_ind_full.glm_voxel_selection(load=False,stats_to_do=None,plot_stacked=True,plot_ind=True,response_label='ind_product_measure',pvalue=None,localizers_to_plot=glm_localizers_to_plot,localizer_label_dict=glm_localizer_label_dict)

	# filepath_tag = 'vision_vs_language'
	# secondlevel_ind_full.plot_features = ['alexnet','sbert+word2vec']
	# secondlevel_ind_full.scale_by = 'total_variance' #if scaling by anything, noise ceiling will not be plotted
	# # secondlevel_ind_full.generate_preference_maps(restricted=True,threshold=0.01,features=['motion','alexnet','sbert','word2vec'],file_tag='vis_lang')
	# secondlevel_ind_full.glm_voxel_selection(load=True,stats_to_do='compare_features',plot_stacked=False,plot_ind=True,response_label='ind_product_measure',pvalue=None,localizers_to_plot=glm_localizers_to_plot,localizer_label_dict=glm_localizer_label_dict,filepath_tag=filepath_tag)
	
	# filepath_tag = 'social_features'
	# secondlevel_ind_full.plot_features = ['num_agents','face','social','valence']
	# secondlevel_ind_full.scale_by = 'total_variance' #if scaling by anything, noise ceiling will not be plotted
	# secondlevel_ind_full.glm_voxel_selection(load=False,stats_to_do='compare_to_zero',plot_stacked=False,plot_ind=True,response_label='ind_product_measure',pvalue=None,localizers_to_plot=glm_localizers_to_plot,localizer_label_dict=glm_localizer_label_dict,filepath_tag=filepath_tag)

	# filepath_tag = 'hum_annot_features'
	# features =['social','num_agents','face','valence','arousal','speaking','turn_taking','mentalization','written_text','music']
	# secondlevel_ind_full.plot_features = features
	# secondlevel_ind_full.feature_names = features
	# secondlevel_ind_full.plot_features_dict['full'] = features
	# secondlevel_ind_full.scale_by = 'total_variance' #if scaling by anything, noise ceiling will not be plotted
	# # secondlevel_ind_full.generate_preference_maps(restricted=True,threshold=0.01,features=features,file_tag=filepath_tag)
	# secondlevel_ind_full.glm_voxel_selection(load=True,stats_to_do='compare_to_zero',plot_stacked=True,plot_ind=True,response_label='ind_product_measure',pvalue=None,localizers_to_plot=glm_localizers_to_plot,localizer_label_dict=glm_localizer_label_dict,filepath_tag=filepath_tag)

	# secondlevel_ind_full.generate_binary_localizer_maps_glm(glm_task='SIpointlights')
	# secondlevel_ind_full.generate_binary_localizer_maps_glm(glm_task='language')
	# # secondlevel_ind_full.plot_localizer(task='SIpointlights',symmetric_cbar=False,vmin=0)
	# # secondlevel_ind_full.plot_localizer(task='language',symmetric_cbar=False,vmin=0)
	# secondlevel_ind_full.plot_map(feature='motion',measure='ind_product_measure',localizer='interact&no_interact')
	# secondlevel_ind_full.scale_by=''
	# secondlevel_ind_full.plot_map(feature='social',measure='ind_product_measure',localizers=['interact-no_interact','intact-degraded'])#,vmin=0,vmax=0.02)
	# secondlevel_ind_full.plot_map(feature='face',measure='ind_product_measure',localizer='interact-no_interact')
	# secondlevel_ind_full.plot_map(feature='sbert',measure='ind_product_measure',localizers=['interact-no_interact','intact-degraded'])#,vmin=0,vmax=0.1)
	# secondlevel_ind_full.plot_map(feature='alexnet',measure='ind_product_measure',localizers=[],vmax=0.3)
	# secondlevel_ind_full.plot_map(feature='sbert',measure='ind_product_measure',localizers=[],vmax=0.3)

	# secondlevel_ind_full.scale_by=''
	# secondlevel_ind_full.plot_diff_maps(features=['sbert+word2vec','alexnet+motion'],measure='ind_product_measure',threshold=0.01,cmap_name='alexnet_sbert')#,ROI='interact-no_interact')
	# secondlevel_ind_full.plot_diff_maps(features=['sbert+word2vec','alexnet'],measure='ind_product_measure',threshold=0.01,cmap_name='alexnet_sbert',ROI='intact-degraded')

	# #these specify the "localizers"
	# enc_localizers_to_plot = ['social','sbert+word2vec']#'num_agents','face','valence','arousal','speaking','turn_taking','mentalization','word2vec','sbert_layers','written_text','pitch','amplitude','music','pixel','hue','motion','alexnet_layers']
	# secondlevel_ind_full.plot_features_dict['full'] = enc_localizers_to_plot
	# #these specify the responses
	# secondlevel_ind_full.plot_features = ['alexnet','motion']
	# secondlevel_ind_full.feature_names = ['alexnet','motion'] #for stacked barplot
	# secondlevel_ind_full.scale_by='total_variance'
	# secondlevel_ind_full.encoding_voxel_selection(load=True,pvalue=None,stats_to_do='compare_features',selection_model='full',selection_label='ind_product_measure',response_label='ind_product_measure',localizers_to_plot=enc_localizers_to_plot,localizer_label_dict=enc_localizer_label_dict)
	# secondlevel_ind_full.generate_binary_localizer_maps_enc(model='full',label='ind_product_measure',plot=False)

	# for n_top_voxels in [100]:#[550,400,250,100]:
	# 	masks = ['STS']#'pSTS','aSTS']
	# 	file_tag = 'only_one_feature_model_overlap_in_STS'
	# 	n_top_voxels = n_top_voxels
	# 	get_overlap(load=True,models=['social','sbert+word2vec'],masks=masks, n_top_voxels=n_top_voxels,
	# 		glm_localizer_label_dict=glm_localizer_label_dict,
	# 		enc_localizer_label_dict=enc_localizer_label_dict,
	# 		file_tag=file_tag)
	# 	file_tag= 'joint_model_overlap_in_STS'
	# 	get_overlap(load=True,models=['full'],masks=masks, n_top_voxels=n_top_voxels,
	# 		glm_localizer_label_dict=glm_localizer_label_dict,
	# 		enc_localizer_label_dict=enc_localizer_label_dict,
	# 		file_tag=file_tag)
	
	# secondlevel_ind_full.scale_by='total_variance'
	# file_tag = 'social_features'
	# secondlevel_ind_full.post_to_ant_line_analysis(load=True,features=['num_agents','social','face','valence','speaking'],measure='ind_product_measure',width=3,mask='STS',file_tag = file_tag)
	# secondlevel_ind_full.post_to_ant_line_analysis(load=True,features=['interact-no_interact'],measure='glm_zscore',width=3,mask='STS')
	# secondlevel_ind_full.scale_by='total_variance'
	# file_tag = 'vis_lang'
	# secondlevel_ind_full.post_to_ant_line_analysis(load=True,features=['sbert+word2vec','alexnet','motion'],measure='ind_product_measure',width=3,mask='STS',file_tag=file_tag)
	# secondlevel_ind_full.post_to_ant_line_analysis(load=True,features=['intact-degraded'],measure='glm_zscore',width=3,mask='STS')
	# file_tag = 'vis_lang_layers'
	# secondlevel_ind_full.post_to_ant_line_analysis(load=True,features=['sbert+word2vec','motion','alexnet']+['alexnet_layer'+str(layer) for layer in [1,2,3,4,5,6,7]],measure='ind_product_measure',width=3,mask='STS',file_tag=file_tag)

####### language aligned vs non-language aligned visual representations ########
	secondlevel_ind_SimCLR_SLIP = get_secondlevel_individual('SimCLR_SLIP')
	secondlevel_ind_SimCLR_SLIP.plot_features_dict['SimCLR_SLIP'] = ['SimCLR_attention_layer'+str(layer) for layer in [1,2,3,4,5,6,7,8,9,10,11,12]]+['SLIP_attention_layer'+str(layer) for layer in [1,2,3,4,5,6,7,8,9,10,11,12]]+['SimCLR_embedding_layer'+str(layer) for layer in [1,2,3,4,5,6,7,8,9,10,11,12]]+['SLIP_embedding_layer'+str(layer) for layer in [1,2,3,4,5,6,7,8,9,10,11,12]]#,'motion','alexnet']
	secondlevel_ind_SimCLR_SLIP.plot_features = ['SimCLR_attention_layer'+str(layer) for layer in [1,2,3,4,5,6,7,8,9,10,11,12]]+['SLIP_attention_layer'+str(layer) for layer in [1,2,3,4,5,6,7,8,9,10,11,12]]+['SimCLR_embedding_layer'+str(layer) for layer in [1,2,3,4,5,6,7,8,9,10,11,12]]+['SLIP_embedding_layer'+str(layer) for layer in [1,2,3,4,5,6,7,8,9,10,11,12]]
	secondlevel_ind_SimCLR_SLIP.plot_features_dict['full'] = ['motion','social','sbert+word2vec']

	enc_localizers_to_plot = secondlevel_ind_SimCLR_SLIP.plot_features_dict['SimCLR_SLIP']

	# secondlevel_ind_SimCLR_SLIP.glm_voxel_selection(load=False,plot_ind=True,response_label='performance',pvalue=None,localizers_to_plot=glm_localizers_to_plot,localizer_label_dict=glm_localizer_label_dict,plot_noise_ceiling=True)
	# secondlevel_ind_SimCLR_SLIP.preference_map_summary(load=False,ROIs = ['interact&no_interact','interact-no_interact','intact-degraded'],ROI_type='functional') 

	#extract everything first, then can plot
	# all_features = ['SLIP','SimCLR','SLIP_attention','SLIP_embedding','SimCLR_attention','SimCLR_embedding']+['SimCLR_attention_layer'+str(layer) for layer in [1,2,3,4,5,6,7,8,9,10,11,12]]+['SLIP_attention_layer'+str(layer) for layer in [1,2,3,4,5,6,7,8,9,10,11,12]]+['SimCLR_embedding_layer'+str(layer) for layer in [1,2,3,4,5,6,7,8,9,10,11,12]]+['SLIP_embedding_layer'+str(layer) for layer in [1,2,3,4,5,6,7,8,9,10,11,12]]
	# secondlevel_ind_SimCLR_SLIP.feature_names = all_features
	# secondlevel_ind_SimCLR_SLIP.plot_features = all_features
	# secondlevel_ind_SimCLR_SLIP.glm_voxel_selection(load=False,stats_to_do=None,plot_stacked=False,plot_ind=True,response_label='ind_product_measure',pvalue=None,localizers_to_plot=glm_localizers_to_plot,localizer_label_dict=glm_localizer_label_dict)

	# filepath_tag = 'total_separate_layers'
	# features = ['SimCLR_attention_layer'+str(layer) for layer in [1,2,3,4,5,6,7,8,9,10,11,12]]+['SLIP_attention_layer'+str(layer) for layer in [1,2,3,4,5,6,7,8,9,10,11,12]]+['SimCLR_embedding_layer'+str(layer) for layer in [1,2,3,4,5,6,7,8,9,10,11,12]]+['SLIP_embedding_layer'+str(layer) for layer in [1,2,3,4,5,6,7,8,9,10,11,12]]
	# secondlevel_ind_SimCLR_SLIP.feature_names = features
	# secondlevel_ind_SimCLR_SLIP.plot_features = features
	# secondlevel_ind_SimCLR_SLIP.glm_voxel_selection(load=True,stats_to_do='compare_to_zero',plot_stacked=True,plot_ind=True,response_label='ind_product_measure',pvalue=None,localizers_to_plot=glm_localizers_to_plot,localizer_label_dict=glm_localizer_label_dict,filepath_tag=filepath_tag)
	# glm_localizers_to_plot = ['motion','social interaction', 'language']
	# filepath_tag = 'attention_vs_embedding'
	# features = ['SimCLR_attention','SLIP_attention','SimCLR_embedding','SLIP_embedding']
	# secondlevel_ind_SimCLR_SLIP.feature_names = features
	# secondlevel_ind_SimCLR_SLIP.plot_features = features
	# secondlevel_ind_SimCLR_SLIP.scale_by = 'total_variance'
	# secondlevel_ind_SimCLR_SLIP.glm_voxel_selection(load=True,stats_to_do='compare_features',plot_stacked=True,plot_ind=True,response_label='ind_product_measure',pvalue=None,localizers_to_plot=glm_localizers_to_plot,localizer_label_dict=glm_localizer_label_dict,filepath_tag=filepath_tag)
	# filepath_tag = 'separate_layers_SimCLR_embedding'
	# features = ['SimCLR_embedding_layer'+str(layer) for layer in [1,2,3,4,5,6,7,8,9,10,11,12]]
	# secondlevel_ind_SimCLR_SLIP.feature_names = features
	# secondlevel_ind_SimCLR_SLIP.plot_features = features
	# secondlevel_ind_SimCLR_SLIP.scale_by = 'total_variance'
	# secondlevel_ind_SimCLR_SLIP.glm_voxel_selection(load=False,stats_to_do='compare_to_zero',plot_stacked=True,plot_ind=True,response_label='ind_product_measure',pvalue=None,localizers_to_plot=glm_localizers_to_plot,localizer_label_dict=glm_localizer_label_dict,filepath_tag=filepath_tag)

	# filepath_tag = 'separate_layers_SLIP_embedding'
	# features = ['SLIP_embedding_layer'+str(layer) for layer in [1,2,3,4,5,6,7,8,9,10,11,12]]
	# secondlevel_ind_SimCLR_SLIP.feature_names = features
	# secondlevel_ind_SimCLR_SLIP.plot_features = features
	# secondlevel_ind_SimCLR_SLIP.scale_by = 'total_variance'
	# secondlevel_ind_SimCLR_SLIP.glm_voxel_selection(load=False,stats_to_do='compare_to_zero',plot_stacked=True,plot_ind=True,response_label='ind_product_measure',pvalue=None,localizers_to_plot=glm_localizers_to_plot,localizer_label_dict=glm_localizer_label_dict,filepath_tag=filepath_tag)

	# filepath_tag = 'separate_layers_SimCLR_attention'
	# features = ['SimCLR_attention_layer'+str(layer) for layer in [1,2,3,4,5,6,7,8,9,10,11,12]]
	# secondlevel_ind_SimCLR_SLIP.feature_names = features
	# secondlevel_ind_SimCLR_SLIP.plot_features = features
	# secondlevel_ind_SimCLR_SLIP.scale_by = 'total_variance'
	# secondlevel_ind_SimCLR_SLIP.glm_voxel_selection(load=False,stats_to_do='compare_to_zero',plot_stacked=True,plot_ind=True,response_label='ind_product_measure',pvalue=None,localizers_to_plot=glm_localizers_to_plot,localizer_label_dict=glm_localizer_label_dict,filepath_tag=filepath_tag)

	# filepath_tag = 'separate_layers_SLIP_attention'
	# features = ['SLIP_attention_layer'+str(layer) for layer in [1,2,3,4,5,6,7,8,9,10,11,12]]
	# secondlevel_ind_SimCLR_SLIP.feature_names = features
	# secondlevel_ind_SimCLR_SLIP.plot_features = features
	# secondlevel_ind_SimCLR_SLIP.scale_by = 'total_variance'
	# secondlevel_ind_SimCLR_SLIP.glm_voxel_selection(load=False,stats_to_do='compare_to_zero',plot_stacked=True,plot_ind=True,response_label='ind_product_measure',pvalue=None,localizers_to_plot=glm_localizers_to_plot,localizer_label_dict=glm_localizer_label_dict,filepath_tag=filepath_tag)

	# filepath_tag = 'layers'
	# features = ['SimCLR_SLIP_layer'+str(layer) for layer in [1,2,3,4,5,6,7,8,9,10,11,12]]
	# secondlevel_ind_SimCLR_SLIP.feature_names = features
	# secondlevel_ind_SimCLR_SLIP.plot_features = features
	# secondlevel_ind_SimCLR_SLIP.scale_by = 'total_variance'
	# secondlevel_ind_SimCLR_SLIP.glm_voxel_selection(load=True,plot_stacked=True,stats_to_do='compare_features',plot_ind=False,response_label='ind_product_measure',pvalue=None,localizers_to_plot=glm_localizers_to_plot,localizer_label_dict=glm_localizer_label_dict,filepath_tag=filepath_tag)

	# filepath_tag = 'separate_layers_SimCLR_SLIP_attention'
	# features = [model+'_attention_layer'+str(layer) for layer in [10,11,12] for model in ['SimCLR','SLIP']]
	# secondlevel_ind_SimCLR_SLIP.feature_names = features
	# secondlevel_ind_SimCLR_SLIP.plot_features = features
	# secondlevel_ind_SimCLR_SLIP.scale_by = 'total_variance'
	# secondlevel_ind_SimCLR_SLIP.glm_voxel_selection(load=True,plot_stacked=True,stats_to_do='compare_features',plot_ind=False,response_label='ind_product_measure',pvalue=None,localizers_to_plot=glm_localizers_to_plot,localizer_label_dict=glm_localizer_label_dict,filepath_tag=filepath_tag)

	# filepath_tag = 'attention_only'
	# features = ['SimCLR_attention','SLIP_attention']
	# secondlevel_ind_SimCLR_SLIP.feature_names = features
	# secondlevel_ind_SimCLR_SLIP.plot_features = features
	# secondlevel_ind_SimCLR_SLIP.scale_by = 'total_variance' #if scaling by anything, noise ceiling will not be plotted
	# secondlevel_ind_SimCLR_SLIP.glm_voxel_selection(load=True,stats_to_do='compare_features',plot_stacked=True,plot_ind=True,response_label='ind_product_measure',pvalue=None,localizers_to_plot=glm_localizers_to_plot,localizer_label_dict=glm_localizer_label_dict,filepath_tag=filepath_tag)

	# filepath_tag = 'embedding_only'
	# features = ['SimCLR_embedding','SLIP_embedding']
	# secondlevel_ind_SimCLR_SLIP.feature_names = features
	# secondlevel_ind_SimCLR_SLIP.plot_features = features
	# secondlevel_ind_SimCLR_SLIP.scale_by = 'total_variance' #if scaling by anything, noise ceiling will not be plotted
	# secondlevel_ind_SimCLR_SLIP.glm_voxel_selection(load=True,stats_to_do='compare_features',plot_stacked=True,plot_ind=True,response_label='ind_product_measure',pvalue=None,localizers_to_plot=glm_localizers_to_plot,localizer_label_dict=glm_localizer_label_dict,filepath_tag=filepath_tag)
	
	# filepath_tag = 'SimCLR_vs_SLIP'
	# features = ['SimCLR','SLIP']
	# secondlevel_ind_SimCLR_SLIP.feature_names = features
	# secondlevel_ind_SimCLR_SLIP.plot_features = features
	# secondlevel_ind_SimCLR_SLIP.scale_by = 'total_variance' #if scaling by anything, noise ceiling will not be plotted
	# secondlevel_ind_SimCLR_SLIP.glm_voxel_selection(load=True,stats_to_do='compare_features',plot_stacked=True,plot_ind=True,response_label='ind_product_measure',pvalue=None,localizers_to_plot=glm_localizers_to_plot,localizer_label_dict=glm_localizer_label_dict,filepath_tag=filepath_tag)

	# # #these specify the "localizers"
	# enc_localizers_to_plot = ['motion','social','sbert+word2vec']#'num_agents','face','valence','arousal','speaking','turn_taking','mentalization','word2vec','sbert_layers','written_text','pitch','amplitude','music','pixel','hue','motion','alexnet_layers']
	# secondlevel_ind_SimCLR_SLIP.plot_features_dict['full'] = enc_localizers_to_plot
	# # # #these specify the responses
	# features = ['SimCLR_attention','SLIP_attention']
	# secondlevel_ind_SimCLR_SLIP.plot_features = features
	# secondlevel_ind_SimCLR_SLIP.feature_names = features #for stacked barplot
	# filepath_tag = 'attention_only'
	# secondlevel_ind_SimCLR_SLIP.encoding_voxel_selection(load=True,pvalue=None,stats_to_do='compare_features',selection_model='full',selection_label='ind_product_measure',response_label='ind_product_measure',localizers_to_plot=enc_localizers_to_plot,localizer_label_dict=enc_localizer_label_dict,filepath_tag=filepath_tag)



	# secondlevel_ind_SimCLR_SLIP.glm_voxel_selection(load=True,plot_ind=True,response_label='ind_product_measure',pvalue=None,localizers_to_plot=glm_localizers_to_plot,localizer_label_dict=glm_localizer_label_dict,plot_noise_ceiling=True)
	# secondlevel_ind_SimCLR_SLIP.encoding_voxel_selection(load=False,pvalue=None,selection_model='SimCLR_SLIP',selection_label='ind_product_measure',response_label='ind_product_measure',localizers_to_plot=enc_localizers_to_plot,localizer_label_dict=enc_localizer_label_dict)
	# secondlevel_ind_SimCLR_SLIP.encoding_voxel_selection(load=False,pvalue=None,selection_model='full',selection_label='ind_product_measure',response_label='ind_product_measure',localizers_to_plot=enc_localizers_to_plot,localizer_label_dict=enc_localizer_label_dict)
	# secondlevel_ind_SimCLR_SLIP.scale_by='total_variance'
	# file_tag = 'attn_emb'
	# secondlevel_ind_SimCLR_SLIP.post_to_ant_line_analysis(load=True,mask='STS',features=['SimCLR_attention','SLIP_attention','SimCLR_embedding','SLIP_embedding'],measure='ind_product_measure',width=3,file_tag=file_tag)
	# secondlevel_ind_SimCLR_SLIP.post_to_ant_line_analysis(load=True,mask='MT',features=['SimCLR_attention','SLIP_attention','SimCLR_embedding','SLIP_embedding'],measure='ind_product_measure',width=3)
	# secondlevel_ind_SimCLR_SLIP.scale_by='total_variance'
	# file_tag = 'SimCLR_attention_layers'
	# secondlevel_ind_SimCLR_SLIP.post_to_ant_line_analysis(load=True,mask='STS',features=['SimCLR_attention_layer'+str(layer) for layer in [1,2,3,4,5,6,7,8,9,10,11,12] ],measure='ind_product_measure',width=3,file_tag=file_tag)
	# file_tag = 'SLIP_attention_layers'
	# secondlevel_ind_SimCLR_SLIP.post_to_ant_line_analysis(load=True,mask='STS',features=['SLIP_attention_layer'+str(layer) for layer in [1,2,3,4,5,6,7,8,9,10,11,12] ],measure='ind_product_measure',width=3,file_tag=file_tag)
	
	# file_tag = 'SimCLR_SLIP_attention_final_layers'
	# secondlevel_ind_SimCLR_SLIP.post_to_ant_line_analysis(load=True,mask='STS',features=['SimCLR_attention_layer'+str(layer) for layer in [9,10,11,12] ]+['SLIP_attention_layer'+str(layer) for layer in [9,10,11,12] ],measure='ind_product_measure',width=3,file_tag=file_tag)
	# secondlevel_ind_SimCLR_SLIP.compute_response_similarity(load=True,axes=['SimCLR','SLIP']) #SLIP-SimCLR for difference
	# secondlevel_ind_SimCLR_SLIP.preference_map_summary(load=True, ROIs = ['pSTS','aSTS'])
	# secondlevel_ind_SimCLR_SLIP.scale_by=''
	# secondlevel_ind_SimCLR_SLIP.plot_diff_maps(features=['SLIP_attention','SimCLR_attention'],measure='ind_product_measure',threshold=None,limits=0.1,cmap_name='teal_orange')
	# secondlevel_ind_SimCLR_SLIP.plot_diff_maps(features=['SLIP','SimCLR'],measure='ind_product_measure',threshold=0.05,cmap_name='teal_orange')

	# # # secondlevel_ind.glm_voxel_selection(load=False,plot_ind=True,response_label='ind_feature_performance',pvalue=None,localizers_to_plot=glm_localizers_to_plot,localizer_label_dict=glm_localizer_label_dict)

	# secondlevel_ind.compute_response_similarity(load=True,response='weights')

	# # # # secondlevel_ind.encoding_voxel_selection(load=False,pvalue=None,selection_model='full',selection_label='ind_product_measure',response_label='performance',localizers_to_plot=enc_localizers_to_plot,localizer_label_dict=enc_localizer_label_dict)
	# secondlevel_ind.encoding_voxel_selection(load=False,pvalue=None,selection_model='full',selection_label='ind_product_measure',response_label='ind_product_measure',localizers_to_plot=enc_localizers_to_plot,localizer_label_dict=enc_localizer_label_dict)
	# secondlevel_ind.encoding_voxel_selection(load=False,pvalue=None,selection_model='full',selection_label='ind_feature_performance',response_label='ind_feature_performance',localizers_to_plot=enc_localizers_to_plot,localizer_label_dict=enc_localizer_label_dict)

	# secondlevel_ind.preference_map_summary(load=False)

	# # # ## generate binary localizer masks to prepare for overlap analysis
	# secondlevel_ind.generate_binary_localizer_maps_glm(glm_task='SIpointlights')
	# secondlevel_ind.generate_binary_localizer_maps_glm(glm_task='language')
	# secondlevel_ind.generate_binary_localizer_maps_enc(model='full',label='ind_product_measure')

	# # ## overlap analysis -- comparing right SI pSTS and language overlap
	# regions = [('social','encoding-ind_product_measure_raw','STS'),('interact-no_interact','glm-SIpointlights','STS'),('sbert','encoding-ind_product_measure_raw','STS'),('intact-degraded','glm-language','STS')]
	# region = 'STS'
	# regions = [('intact-degraded','glm-language',region),('sbert','encoding-ind_product_measure_raw',region),('word2vec','encoding-ind_product_measure_raw',region),('social','encoding-ind_product_measure_raw',region),('interact-no_interact','glm-SIpointlights',region)]#,('sbert','encoding-ind_product_measure_raw',enc_localizer_label_dict['sbert']),('intact-degraded','glm-language',glm_localizer_label_dict['language'])]
	# secondlevel_ind.compute_all_overlap2(load=True,regions=regions,label='STS')

	# region = 'language'
	# regions = [('sbert','encoding-ind_product_measure_raw',region),('word2vec','encoding-ind_product_measure_raw',region),('intact-degraded','glm-language',region)]
	# secondlevel_ind.compute_all_overlap2(load=False,regions=regions,label='language')

	# regions = [('interact-no_interact_social','overlap-','STS'),('intact-degraded_sbert','overlap-','language'),('intact-degraded_word2vec','overlap-','language')]
	# secondlevel_ind.compute_all_overlap2(load=True,regions=regions,label='overlap')

	######### language aligned vs. non-language aligned visual representation, controlling for spoken language in stimuli #######

	# secondlevel_ind_GPT2_SimCLR_SLIP_word2vec = get_secondlevel_individual('GPT2_SimCLR_SLIP_word2vec')
	# secondlevel_ind_GPT2_SimCLR_SLIP_word2vec.plot_features_dict['joint'] = ['motion','social','sbert+word2vec']

	# secondlevel_ind_GPT2_SimCLR_SLIP_word2vec.scale_by=''
	# secondlevel_ind_GPT2_SimCLR_SLIP_word2vec.plot_diff_maps(features=['SLIP_attention','SimCLR_attention'],measure='ind_product_measure',threshold=None,limits=0.1,cmap_name='teal_orange')
	
	# enc_localizers_to_plot = ['motion','social','sbert+word2vec','valence','arousal']#'num_agents','face','valence','arousal','speaking','turn_taking','mentalization','word2vec','sbert_layers','written_text','pitch','amplitude','music','pixel','hue','motion','alexnet_layers']
	# secondlevel_ind_GPT2_SimCLR_SLIP_word2vec.plot_features_dict['full'] = enc_localizers_to_plot
	# # # #these specify the responses
	# features = ['SimCLR','SLIP']
	# secondlevel_ind_GPT2_SimCLR_SLIP_word2vec.plot_features = features
	# secondlevel_ind_GPT2_SimCLR_SLIP_word2vec.feature_names = features #for stacked barplot
	# filepath_tag = ''
	# secondlevel_ind_GPT2_SimCLR_SLIP_word2vec.encoding_voxel_selection(load=False,pvalue=None,stats_to_do='compare_features',selection_model='full',selection_label='ind_product_measure',response_label='ind_product_measure',localizers_to_plot=enc_localizers_to_plot,localizer_label_dict=enc_localizer_label_dict,filepath_tag=filepath_tag)

	# enc_localizers_to_plot = ['SLIP','SimCLR','SLIP_attention','SimCLR_attention','SLIP_embedding','SimCLR_embedding']#'num_agents','face','valence','arousal','speaking','turn_taking','mentalization','word2vec','sbert_layers','written_text','pitch','amplitude','music','pixel','hue','motion','alexnet_layers']
	# secondlevel_ind_GPT2_SimCLR_SLIP_word2vec.plot_features_dict['GPT2_SimCLR_SLIP_word2vec'] = enc_localizers_to_plot
	# # # #these specify the responses
	# features = ['SLIP_attention','SimCLR_attention','SLIP_embedding','SimCLR_embedding']
	# secondlevel_ind_GPT2_SimCLR_SLIP_word2vec.plot_features = features
	# secondlevel_ind_GPT2_SimCLR_SLIP_word2vec.feature_names = features #for stacked barplot
	# filepath_tag = ''
	# secondlevel_ind_GPT2_SimCLR_SLIP_word2vec.encoding_voxel_selection(load=True,pvalue=None,stats_to_do='compare_features',selection_model='GPT2_SimCLR_SLIP_word2vec',selection_label='ind_product_measure',response_label='ind_product_measure',localizers_to_plot=enc_localizers_to_plot,localizer_label_dict=enc_localizer_label_dict,filepath_tag=filepath_tag)

	# secondlevel_ind_GPT2_SimCLR_SLIP_word2vec.glm_voxel_selection(load=False,plot_ind=True,response_label='performance',pvalue=None,localizers_to_plot=glm_localizers_to_plot,localizer_label_dict=glm_localizer_label_dict,plot_noise_ceiling=True)
	# filepath_tag = 'SimCLR_vs_SLIP'
	# features = ['SimCLR','SLIP']
	# secondlevel_ind_GPT2_SimCLR_SLIP_word2vec.feature_names = features
	# secondlevel_ind_GPT2_SimCLR_SLIP_word2vec.plot_features = features
	# secondlevel_ind_GPT2_SimCLR_SLIP_word2vec.scale_by = 'total_variance' #if scaling by anything, noise ceiling will not be plotted
	# secondlevel_ind_GPT2_SimCLR_SLIP_word2vec.glm_voxel_selection(load=False,stats_to_do='compare_features',plot_stacked=True,plot_ind=True,response_label='ind_product_measure',pvalue=None,localizers_to_plot=glm_localizers_to_plot,localizer_label_dict=glm_localizer_label_dict,filepath_tag=filepath_tag)
	
	# filepath_tag = 'all'
	# features = ['SimCLR','SLIP','GPT2_1sent','word2vec']
	# secondlevel_ind_GPT2_SimCLR_SLIP_word2vec.feature_names = features
	# secondlevel_ind_GPT2_SimCLR_SLIP_word2vec.plot_features = features
	# secondlevel_ind_GPT2_SimCLR_SLIP_word2vec.scale_by = 'total_variance' #if scaling by anything, noise ceiling will not be plotted
	# secondlevel_ind_GPT2_SimCLR_SLIP_word2vec.glm_voxel_selection(load=False,stats_to_do='compare_features',plot_stacked=True,plot_ind=True,response_label='ind_product_measure',pvalue=None,localizers_to_plot=glm_localizers_to_plot,localizer_label_dict=glm_localizer_label_dict,filepath_tag=filepath_tag)

	# filepath_tag = 'attn_vs_emb_SLIP'
	# features = ['SLIP_attention','SLIP_embedding']
	# secondlevel_ind_GPT2_SimCLR_SLIP_word2vec.feature_names = features
	# secondlevel_ind_GPT2_SimCLR_SLIP_word2vec.plot_features = features
	# secondlevel_ind_GPT2_SimCLR_SLIP_word2vec.scale_by = 'total_variance' #if scaling by anything, noise ceiling will not be plotted
	# secondlevel_ind_GPT2_SimCLR_SLIP_word2vec.glm_voxel_selection(load=True,stats_to_do='compare_features',plot_stacked=True,plot_ind=True,response_label='ind_product_measure',pvalue=None,localizers_to_plot=glm_localizers_to_plot,localizer_label_dict=glm_localizer_label_dict,filepath_tag=filepath_tag)

	# filepath_tag = 'attn_vs_emb_SimCLR'
	# features = ['SimCLR_attention','SimCLR_embedding']
	# secondlevel_ind_GPT2_SimCLR_SLIP_word2vec.feature_names = features
	# secondlevel_ind_GPT2_SimCLR_SLIP_word2vec.plot_features = features
	# secondlevel_ind_GPT2_SimCLR_SLIP_word2vec.scale_by = 'total_variance' #if scaling by anything, noise ceiling will not be plotted
	# secondlevel_ind_GPT2_SimCLR_SLIP_word2vec.glm_voxel_selection(load=True,stats_to_do='compare_features',plot_stacked=True,plot_ind=True,response_label='ind_product_measure',pvalue=None,localizers_to_plot=glm_localizers_to_plot,localizer_label_dict=glm_localizer_label_dict,filepath_tag=filepath_tag)

	# filepath_tag = 'separate_layers_SimCLR_embedding'
	# features = ['SimCLR_embedding_layer'+str(layer) for layer in [1,2,3,4,5,6,7,8,9,10,11,12]]
	# secondlevel_ind_GPT2_SimCLR_SLIP_word2vec.feature_names = features
	# secondlevel_ind_GPT2_SimCLR_SLIP_word2vec.plot_features = features
	# secondlevel_ind_GPT2_SimCLR_SLIP_word2vec.scale_by = 'total_variance'
	# secondlevel_ind_GPT2_SimCLR_SLIP_word2vec.glm_voxel_selection(load=False,stats_to_do='compare_to_zero',plot_stacked=True,plot_ind=True,response_label='ind_product_measure',pvalue=None,localizers_to_plot=glm_localizers_to_plot,localizer_label_dict=glm_localizer_label_dict,filepath_tag=filepath_tag)

	# filepath_tag = 'separate_layers_SLIP_embedding'
	# features = ['SLIP_embedding_layer'+str(layer) for layer in [1,2,3,4,5,6,7,8,9,10,11,12]]
	# secondlevel_ind_GPT2_SimCLR_SLIP_word2vec.feature_names = features
	# secondlevel_ind_GPT2_SimCLR_SLIP_word2vec.plot_features = features
	# secondlevel_ind_GPT2_SimCLR_SLIP_word2vec.scale_by = 'total_variance'
	# secondlevel_ind_GPT2_SimCLR_SLIP_word2vec.glm_voxel_selection(load=False,stats_to_do='compare_to_zero',plot_stacked=True,plot_ind=True,response_label='ind_product_measure',pvalue=None,localizers_to_plot=glm_localizers_to_plot,localizer_label_dict=glm_localizer_label_dict,filepath_tag=filepath_tag)

	# filepath_tag = 'separate_layers_SimCLR_attention'
	# features = ['SimCLR_attention_layer'+str(layer) for layer in [1,2,3,4,5,6,7,8,9,10,11,12]]
	# secondlevel_ind_GPT2_SimCLR_SLIP_word2vec.feature_names = features
	# secondlevel_ind_GPT2_SimCLR_SLIP_word2vec.plot_features = features
	# secondlevel_ind_GPT2_SimCLR_SLIP_word2vec.scale_by = 'total_variance'
	# secondlevel_ind_GPT2_SimCLR_SLIP_word2vec.glm_voxel_selection(load=True,stats_to_do='compare_to_zero',plot_stacked=True,plot_ind=True,response_label='ind_product_measure',pvalue=None,localizers_to_plot=glm_localizers_to_plot,localizer_label_dict=glm_localizer_label_dict,filepath_tag=filepath_tag)

	# filepath_tag = 'separate_layers_SLIP_attention'
	# features = ['SLIP_attention_layer'+str(layer) for layer in [1,2,3,4,5,6,7,8,9,10,11,12]]
	# secondlevel_ind_GPT2_SimCLR_SLIP_word2vec.feature_names = features
	# secondlevel_ind_GPT2_SimCLR_SLIP_word2vec.plot_features = features
	# secondlevel_ind_GPT2_SimCLR_SLIP_word2vec.scale_by = 'total_variance'
	# secondlevel_ind_GPT2_SimCLR_SLIP_word2vec.glm_voxel_selection(load=True,stats_to_do='compare_to_zero',plot_stacked=True,plot_ind=True,response_label='ind_product_measure',pvalue=None,localizers_to_plot=glm_localizers_to_plot,localizer_label_dict=glm_localizer_label_dict,filepath_tag=filepath_tag)
	
	# secondlevel_ind_GPT2_SimCLR_SLIP_word2vec.scale_by = 'total_variance'
	# file_tag = 'final_layers'
	# secondlevel_ind_GPT2_SimCLR_SLIP_word2vec.post_to_ant_line_analysis(load=True,mask='lateral',features=['SimCLR_attention_layer'+str(layer) for layer in [9,10,11,12] ]+['SLIP_attention_layer'+str(layer) for layer in [9,10,11,12] ],measure='ind_product_measure',width=3,file_tag=file_tag)

	# file_tag = 'SLIP_SimCLR_attention'
	# secondlevel_ind_GPT2_SimCLR_SLIP_word2vec.scale_by='total_variance'
	# secondlevel_ind_GPT2_SimCLR_SLIP_word2vec.post_to_ant_line_analysis(load=True,mask='lateral',features=['SimCLR_attention','SLIP_attention','SimCLR_embedding','SLIP_embedding'], measure='ind_product_measure',width=3,file_tag=file_tag)

	# file_tag = 'SLIP_SimCLR'
	# secondlevel_ind_GPT2_SimCLR_SLIP_word2vec.scale_by='total_variance'
	# secondlevel_ind_GPT2_SimCLR_SLIP_word2vec.post_to_ant_line_analysis(load=True,mask='lateral',features=['SimCLR','SLIP','GPT2_1sent','word2vec'], measure='ind_product_measure',width=3,file_tag=file_tag)


	### testing ###
	# secondlevel_ind = get_secondlevel_individual('hubert_lang_vis')
	# secondlevel_ind.plot_features_dict['full'] = ['motion','social','sbert+word2vec']
	# glm_localizers_to_plot = ['motion', 'social interaction','language']


	# # secondlevel_ind.glm_voxel_selection(load=False,plot_ind=True,response_label='performance',pvalue=None,localizers_to_plot=glm_localizers_to_plot,localizer_label_dict=glm_localizer_label_dict,plot_noise_ceiling=True)
	# filepath_tag = ''
	# features = ['SimCLR','sbert','word2vec','motion','hubert']
	# secondlevel_ind.feature_names = features
	# secondlevel_ind.plot_features = features
	# secondlevel_ind.scale_by = 'total_variance' #if scaling by anything, noise ceiling will not be plotted
	# secondlevel_ind.glm_voxel_selection(load=True,stats_to_do='compare_features',plot_stacked=True,plot_ind=True,response_label='ind_product_measure',pvalue=None,localizers_to_plot=glm_localizers_to_plot,localizer_label_dict=glm_localizer_label_dict,filepath_tag=filepath_tag)
	# features = ['hubert_layer'+str(layer) for layer in [1,2,3,4,5,6,7,8,9,10,11,12]]
	# secondlevel_ind.generate_preference_maps(restricted=True,threshold=0.01,features=features,file_tag='hubert')
	# secondlevel_ind.scale_by = ''
	# secondlevel_ind.plot_map(feature='sbert',measure='ind_product_measure',localizers=[],vmin=0)
	# secondlevel_ind.plot_map(feature='word2vec',measure='ind_product_measure',localizers=[],vmin=0)
	# secondlevel_ind.plot_map(feature='hubert',measure='ind_product_measure',localizers=[],vmin=0)
	# secondlevel_ind.plot_map(feature='SimCLR',measure='ind_product_measure',localizers=[],vmin=0)

	# secondlevel_ind = get_secondlevel_individual('GPT2_1sent_3sent_SimCLR_social')
	# secondlevel_ind.plot_features_dict['full'] = ['motion','social','sbert+word2vec']
	# glm_localizers_to_plot = ['motion', 'social interaction','language']

	# filepath_tag = ''
	# features = ['GPT2_1sent','GPT2_3sent','SimCLR','social']
	# secondlevel_ind.feature_names = features
	# secondlevel_ind.plot_features = features
	# secondlevel_ind.scale_by = 'total_variance' #if scaling by anything, noise ceiling will not be plotted
	# secondlevel_ind.glm_voxel_selection(load=True,stats_to_do='compare_features',plot_stacked=True,plot_ind=True,response_label='ind_product_measure',pvalue=None,localizers_to_plot=glm_localizers_to_plot,localizer_label_dict=glm_localizer_label_dict,filepath_tag=filepath_tag)
	
	# secondlevel_ind.scale_by = 'total_variance'
	# secondlevel_ind.plot_map(feature='GPT2_1sent',measure='ind_product_measure',localizers=[],vmin=0)
	# secondlevel_ind.plot_map(feature='GPT2_3sent',measure='ind_product_measure',localizers=[],vmin=0)
	# secondlevel_ind.plot_map(feature='social',measure='ind_product_measure',localizers=[],vmin=0)
	# secondlevel_ind.plot_map(feature='SimCLR',measure='ind_product_measure',localizers=[],vmin=0)


	# secondlevel_ind = get_secondlevel_individual('audio_lang')
	# # secondlevel_ind.plot_features_dict['full'] = ['motion','social','sbert+word2vec']
	# # glm_localizers_to_plot = ['motion', 'social interaction','language']

	# # filepath_tag = ''
	# # features = ['cochdnn','sbert','word2vec']
	# # secondlevel_ind.feature_names = features
	# # secondlevel_ind.plot_features = features
	# # secondlevel_ind.scale_by = 'total_variance' #if scaling by anything, noise ceiling will not be plotted
	# # secondlevel_ind.glm_voxel_selection(load=True,stats_to_do='compare_features',plot_stacked=True,plot_ind=True,response_label='ind_product_measure',pvalue=None,localizers_to_plot=glm_localizers_to_plot,localizer_label_dict=glm_localizer_label_dict,filepath_tag=filepath_tag)
	
	# # secondlevel_ind.scale_by = 'total_variance'
	# # secondlevel_ind.plot_map(feature='sbert',measure='ind_product_measure',localizers=[],vmin=0)
	# # secondlevel_ind.plot_map(feature='word2vec',measure='ind_product_measure',localizers=[],vmin=0)
	# # secondlevel_ind.plot_map(feature='cochdnn',measure='ind_product_measure',localizers=[],vmin=0)
	# features = ['cochdnn_layer'+str(layer) for layer in [0,1,2,3,4,5,6]]
	# secondlevel_ind_full.generate_preference_maps(restricted=True,threshold=0.01,features=features,file_tag='cochdnn')


	secondlevel_ind = get_secondlevel_individual('SLIPtext')
	glm_localizers_to_plot = ['motion', 'social interaction','language']
	filepath_tag = ''
	secondlevel_ind.glm_voxel_selection(load=False,stats_to_do='compare_to_zero',plot_stacked=False,plot_ind=True,response_label='performance',pvalue=None,localizers_to_plot=glm_localizers_to_plot,localizer_label_dict=glm_localizer_label_dict,filepath_tag=filepath_tag)


