import numpy as np
import pandas as pd
import seaborn as sns

from matplotlib.colors import LinearSegmentedColormap
cmap =  LinearSegmentedColormap.from_list('my_gradient', (
    # Edit this gradient at https://eltos.github.io/gradient/#FFFFFF-376A5F
    (0.000, (1.000, 1.000, 1.000)),
    (1.000, (0.216, 0.416, 0.373))))
features = np.array(pd.read_csv('full_features.csv',header=None)).tolist()[0]
print(features)
filepath = '/Users/hsmall2/Documents/GitHub/deep_nat_lat/analysis/FeatureSpaceCorrelation/'
features = ['alexnet_layer1', 'alexnet_layer2', 'alexnet_layer3', 'alexnet_layer4', 'alexnet_layer5', 'alexnet_layer6', 'alexnet_layer7','motion', 'face', 'num_agents','indoor_outdoor', 'written_text', 'pixel', 'hue', 'pitch', 'amplitude', 'music','speaking', 'turn_taking', 'mentalization','social', 'valence','arousal', 'word2vec','sbert_layer1', 'sbert_layer2', 'sbert_layer3', 'sbert_layer4', 'sbert_layer5', 'sbert_layer6', 'sbert_layer7', 'sbert_layer8', 'sbert_layer9', 'sbert_layer10', 'sbert_layer11', 'sbert_layer12']
features = ['motion', 'face', 'num_agents','indoor_outdoor', 'written_text', 'pixel', 'hue', 'pitch', 'amplitude', 'music','speaking', 'turn_taking', 'valence','arousal','mentalization','social','SLIPtext','word2vec']+['GPT2_1sent_layer'+str(layer) for layer in [1,2,3,4,5,6,7,8,9,10,11,12]]+['SLIP_attention_layer'+str(layer) for layer in [1,2,3,4,5,6,7,8,9,10,11,12]]  + ['SLIP_embedding_layer'+str(layer) for layer in [1,2,3,4,5,6,7,8,9,10,11,12]] + ['SimCLR_embedding_layer'+str(layer) for layer in [1,2,3,4,5,6,7,8,9,10,11,12]] + ['SimCLR_attention_layer'+str(layer) for layer in [1,2,3,4,5,6,7,8,9,10,11,12]]
# features = ['social','SLIPtext','word2vec',] +['SLIP_attention_layer'+str(layer) for layer in [1,2,3,4,5,6,7,8,9,10,11,12]]  + ['SLIP_embedding_layer'+str(layer) for layer in [1,2,3,4,5,6,7,8,9,10,11,12]] + ['SimCLR_embedding_layer'+str(layer) for layer in [1,2,3,4,5,6,7,8,9,10,11,12]] + ['SimCLR_attention_layer'+str(layer) for layer in [1,2,3,4,5,6,7,8,9,10,11,12]]
features =  ['word2vec']+['GPT2_1sent_layer'+str(layer) for layer in [1,2,3,4,5,6,7,8,9,10,11,12]]+['SLIPtext_layer'+str(layer) for layer in [1,2,3,4,5,6,7,8,9,10,11,12]]+['SLIP_attention_layer'+str(layer) for layer in [1,2,3,4,5,6,7,8,9,10,11,12]]  + ['SLIP_embedding_layer'+str(layer) for layer in [1,2,3,4,5,6,7,8,9,10,11,12]] + ['SimCLR_embedding_layer'+str(layer) for layer in [1,2,3,4,5,6,7,8,9,10,11,12]] + ['SimCLR_attention_layer'+str(layer) for layer in [1,2,3,4,5,6,7,8,9,10,11,12]]
# features =  ['SLIP_attention_layer'+str(layer) for layer in [1,2,3,4,5,6,7,8,9,10,11,12]]  + ['SLIP_embedding_layer'+str(layer) for layer in [1,2,3,4,5,6,7,8,9,10,11,12]] + ['SimCLR_embedding_layer'+str(layer) for layer in [1,2,3,4,5,6,7,8,9,10,11,12]] + ['SimCLR_attention_layer'+str(layer) for layer in [1,2,3,4,5,6,7,8,9,10,11,12]] +['SLIPtext','word2vec']

results_matrix = np.zeros((len(features),len(features)))*np.nan
track = []
for i,feature1 in enumerate(features):
	for j,feature2 in enumerate(features):
		alphabetical = [feature1,feature2]
		alphabetical.sort()
		if(feature1==feature2):
			continue
		# elif(alphabetical in track):
		# 	continue
		else:
			try:
				result = pd.read_csv(filepath+'-'.join([feature1,feature2])+'.csv',header=None)[0]
				print(feature1,'-',feature2,result)
			except Exception as e:
				result = np.nan
			if(np.isnan(results_matrix[i][j])):
				results_matrix[i][j] = result
			if(np.isnan(results_matrix[j][i])):
				results_matrix[j][i] = result
			track.append(alphabetical)

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

plt.rcParams.update({'font.family': 'Arial'})
results_matrix[np.triu_indices(results_matrix.shape[0],0)] = np.nan
fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot(111)

cax = ax.imshow(results_matrix,cmap=cmap,vmin=0,vmax=1)
sns.despine(left=True,bottom=True)
cbar = fig.colorbar(cax,aspect=40,label='Canonical correlation coefficient')
cbar.ax.yaxis.set_label_position('left')

plt.title('')
ax.set_xticks(range(0,len(features)))
ax.set_yticks(range(0,len(features)))

# ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False,labelright=True,labelleft=False)

group_labels = [name.split('_')[0] if name.split('_')[0] in ['alexnet','sbert'] else '' for name in features ]

group_labels = [name.split('_')[0]+' '+''.join([char for char in name.split('_')[1] if not char.isnumeric()]) if name.split('_')[0] in ['SimCLR','SLIP','SLIPtext','GPT2'] else '' for name in features ]

current_label = group_labels[0]
start_idx = 0
print(group_labels)
for i, label in enumerate(group_labels):
    if label != current_label or i == len(group_labels) - 1:
        current_label=current_label.replace('1sent','')
        end_idx = i if i == len(group_labels) - 1 else i - 1
        mid_point = (start_idx + end_idx) / 2
        shift_off_axis = 2.5
        ax.text(mid_point, len(features) + shift_off_axis + 1, current_label, ha='center', va='center', fontsize=10, fontweight='bold')
        ax.text(-shift_off_axis-2, mid_point, current_label, ha='center', va='center', fontsize=10, fontweight='bold', rotation=90)
        if(current_label in ['alexnet','sbert']):
	        rect_h = Rectangle((start_idx-0.5, len(features) + shift_off_axis), end_idx - start_idx + 1, 0,
	    		linewidth=2, edgecolor='black', facecolor='none',clip_on=False)
	        ax.add_patch(rect_h)
	        rect_v = Rectangle((-shift_off_axis-1, start_idx-0.5), 0, end_idx - start_idx + 1,
	    		linewidth=2, edgecolor='black', facecolor='none',clip_on=False)
	        ax.add_patch(rect_v)
        current_label = label
        start_idx = i

feature_labels = [name.replace('_',' ').replace('100ep layer','').replace('GPT2','').replace('1sent','').replace('alexnet','').replace('sbert','').replace('SimCLR ','').replace('SLIP ','').replace('SLIPtext ','').replace('attention','').replace('embedding','').replace('layer','') for name in features]
ax.set_xticks(np.arange(len(features)), minor=False)
ax.set_yticks(np.arange(len(features)), minor=False)
ax.set_xticklabels(feature_labels, minor=False, rotation=90)
ax.set_yticklabels(feature_labels, minor=False)
plt.xticks(rotation=90)
plt.savefig('/Users/hsmall2/Documents/GitHub/deep_nat_lat/figures/FeatureSpaceCorrelation/featurespace_CCA_SimCLR_SLIP.pdf',bbox_inches='tight')
plt.close()

# feature_spaces = ['motion', 'face', 'num_agents','indoor_outdoor', 'written_text', 'pixel', 'hue', 'pitch', 'amplitude', 'music','speaking', 'turn_taking', 'valence','arousal','mentalization','social','SLIPtext','word2vec']
# feature_spaces = ['SLIPtext','word2vec']
# ##plot the correlations over the layers of SLIP and SimCLR attention
print(features)
results_df = pd.DataFrame(results_matrix, columns=features)
results_df['feature'] = [feature+'_layer' for feature in features]
# for feature_space in feature_spaces:
# 	results_df = results_df[results_df.feature!=feature_space]
print(results_df)
results_df['model'] = [label.split('_')[0] for label in results_df['feature']]
results_df['layer'] = [label.split('layer')[1] for label in results_df['feature']]
results_df['output'] = [label.split('_')[1] for label in results_df['feature']]
results_df['model_output'] = [label.split('_')[0] for label in results_df['feature']]
print(results_df)


colors_dict = {'SimCLR':'#9D7625',
                'SLIP':'#1F6C5F'}


# for feature_space in feature_spaces:
# 	for output in ['attention','embedding']:
# 		temp_results_df =  results_df[[feature_space,'model','layer','output']]
# 		temp = temp_results_df[temp_results_df.output==output]
# 		print(temp)
# 		for model in ['SimCLR','SLIP']:
# 			print(model)
# 			print(np.mean(temp[temp.model==model]))
# 		ax = sns.lineplot(data=temp,y=feature_space,x='layer',hue='model',palette=colors_dict)
# 		ax.set_ylim(-0.1,0.7)
# 		plt.savefig('/Users/hsmall2/Documents/GitHub/deep_nat_lat/figures/FeatureSpaceCorrelation/'+feature_space+'_'+output+'_lineplot.pdf')
# 		plt.close()
# for i in range(len(features)):
#     for j in range((len(features))):
#         c = results_matrix[j,i]
#         ax.text(i, j, str(np.round(c,2)), va='center', ha='center',fontweight='heavy')
        # pvalue = results_p_array[j,i]
        # if(~np.isnan(c)):
        #     if(pvalue<0.001):
        #         ax.text(i, j, str(np.round(c,2)), va='center', ha='center',fontweight='heavy')
        #     elif(pvalue<0.05):
        #         ax.text(i, j, str(np.round(c,2)), va='center', ha='center')
        #     else:
        #         ax.text(i, j, str(np.round(c,2)), va='center', ha='center',color='gray')

results_df.columns = results_df['model_output'].tolist() + ['feature','model','layer','output','model_output']
results_df.to_csv('temp.csv')
#average over columns of same name
results_df=results_df.groupby(by='model_output', axis=0).mean()
results_df=results_df.groupby(by=results_df.columns, axis=1).mean()
print(results_df.columns)
features = ['SimCLR','SLIP','SLIPtext','word2vec','GPT2']
#populate a numpy matrix to display
results_matrix = np.ones((len(features),len(features)))
for i,feature1 in enumerate(features):
	for j,feature2 in enumerate(features):
		if(~np.isnan(results_df[feature1][feature2])):
			results_matrix[i][j] = results_df[feature1][feature2]
			results_matrix[j][i]  = results_df[feature1][feature2]

print(results_matrix)
results_matrix[np.triu_indices(results_matrix.shape[0],0)] = np.nan
fig = plt.figure(figsize=(4.7,4.7))
ax = fig.add_subplot(111)



cax = ax.imshow(results_matrix,cmap=cmap,vmin=0,vmax=1)
sns.despine(left=True,bottom=True)
cbar = fig.colorbar(cax,aspect=40,label='Canonical correlation coefficient')
cbar.ax.yaxis.set_label_position('left')

plt.title('')
ax.set_xticks(range(0,len(features)))
ax.set_yticks(range(0,len(features)))
feature_labels = [feature.replace('_',' ').replace(' layer','').replace('1sent','') for feature in features]
ax.set_xticklabels(feature_labels, minor=False, rotation=90)
ax.set_yticklabels(feature_labels, minor=False)

for i in range(len(features)):
    for j in range((len(features))):
        c = results_matrix[j,i]
        if(~np.isnan(c)):
        	ax.text(i, j, str(np.round(c,2)), va='center', ha='center',fontweight='heavy')
        # pvalue = results_p_array[j,i]
        # if(~np.isnan(c)):
        #     if(pvalue<0.001):
        #         ax.text(i, j, str(np.round(c,2)), va='center', ha='center',fontweight='heavy')
        #     elif(pvalue<0.05):
        #         ax.text(i, j, str(np.round(c,2)), va='center', ha='center')
        #     else:
        #         ax.text(i, j, str(np.round(c,2)), va='center', ha='center',color='gray')

plt.savefig('/Users/hsmall2/Documents/GitHub/deep_nat_lat/figures/FeatureSpaceCorrelation/featurespace_average_CCA_SimCLR_SLIP.pdf',bbox_inches='tight')



