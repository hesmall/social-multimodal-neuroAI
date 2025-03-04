import pandas as pd 
import numpy as np

features = np.array(pd.read_csv('SLIPtext_features.csv',header=None)).tolist()[0]
print(features)
filepath = 'featurespace_comparisons.tsv'
with open(filepath,'w') as file:
	file.write('comparisons\n')
	track = []
	for feature1 in features:
		for feature2 in features:
			alphabetical = [feature1,feature2]
			print(alphabetical)
			alphabetical.sort()
			if(feature1==feature2):
				continue
			elif(feature1.split('_')[0]!='SLIPtext'):
				continue
			# if((feature2.split('_')[0]=='SimCLR')|(feature2.split('_')[0]=='SLIP')):
			# 	continue
			# elif((feature1.split('_100ep')[0]=='SLIP')&(feature2.split('_100ep')[0]=='SLIP')):
			# 	continue
			# elif((feature1.split('_layer')[0]=='SLIP_attention')):
			# 	continue
			# elif((feature1.split('_layer')[0]=='SLIP_embedding')):
			# 	continue
			# if((feature1.split('_')[0]!='SimCLR')&(feature2.split('_')[0]!='SLIP')):
			# 	continue
			elif(alphabetical in track):
				continue
			else:
				file.write(feature1+'-'+feature2+'\n')
				track.append(alphabetical)

# filepath = 'featurespace_comparisons_both_ways.tsv'
# with open(filepath,'w') as file:
# 	file.write('comparisons\n')
# 	track = []
# 	for feature1 in features:
# 		for feature2 in features:
# 			label = [feature1,feature2]
# 			if(feature1==feature2):
# 				continue
# 			elif(alphabetical in track):
# 				continue
# 			else:
# 				file.write(feature1+'-'+feature2+'\n')
# 				track.append(label)