
def get_subjects(population):
    if(population=='NT'):
        subjects_social = ['sub-05','sub-06','sub-07','sub-08','sub-09','sub-10','sub-11','sub-12','sub-13','sub-14','sub-15','sub-16','sub-19','sub-21','sub-23','sub-25','sub-26','sub-28','sub-32','sub-33','sub-35','sub-36'] 
        subjects_language = ['sub-19','sub-21','sub-23','sub-25','sub-26','sub-28','sub-32','sub-33','sub-35','sub-36']
        bad_subjects = ['sub-09','sub-11','sub-12','sub-13','sub-19']

        subjects={'SIpointlights':[subject for subject in subjects_social if subject not in bad_subjects],
                  'language':[subject for subject in subjects_language if subject not in bad_subjects]}
        ## BAD SUBJECT EXPLANATIONS ##
        #sub 09 more than 15% of sherlock are motion outliers (FD>0.5mm)
        #sub 11 only had audio in the right ear
        #sub 12 only had audio in the right ear
        #sub-13 first run of sherlock is not correctly time locked <- check if we actually need to exclude this one
        #sub 19 more than 15% of sherlock are motion outliers (FD>0.5mm)

        ##TEMP
        # sub 16 doesnt have joint

    if(population=='ASD'):
        subjects_social = ['sub-04','sub-17','sub-18','sub-20','sub-22','sub-24','sub-27','sub-29','sub-34','sub-37'] #sub-04 <- check on this one
        subjects_language = ['sub-20','sub-27','sub-29','sub-34','sub-37']
        bad_subjects = ['sub-18','sub-24','sub-34']
        subjects={'SIpointlights':[subject for subject in subjects_social if subject not in bad_subjects],
                  'language':[subject for subject in subjects_language if subject not in bad_subjects]}
        ## BAD SUBJECT EXPLANATIONS ##
        # sub 18 more than 15% of sherlock are motion outliers (FD>0.5mm)
        # sub 24 more than 15% of sherlock are motion outliers (FD>0.5mm)
        # sub 34 more than 15% of sherlock are motion outliers (FD>0.5mm)

    subjects['sherlock'] = subjects['SIpointlights']
    return subjects

def get_cmaps():
    from matplotlib.colors import LinearSegmentedColormap
    cmaps = {
    'rainbow':LinearSegmentedColormap.from_list('my_gradient', (
    # Edit this gradient at https://eltos.github.io/gradient/#AD0D1D-F55464-FC9D13-FFD619-F4FF47-4DFF32-1EAF1B-28C448-25FFF9-20AAFD-2031FF-7F27FF-760DA1
    (0.000, (0.678, 0.051, 0.114)),
    (0.083, (0.961, 0.329, 0.392)),
    (0.167, (0.988, 0.616, 0.075)),
    (0.250, (1.000, 0.839, 0.098)),
    (0.333, (0.957, 1.000, 0.278)),
    (0.417, (0.302, 1.000, 0.196)),
    (0.500, (0.118, 0.686, 0.106)),
    (0.583, (0.157, 0.769, 0.282)),
    (0.667, (0.145, 1.000, 0.976)),
    (0.750, (0.125, 0.667, 0.992)),
    (0.833, (0.125, 0.192, 1.000)),
    (0.917, (0.498, 0.153, 1.000)),
    (1.000, (0.463, 0.051, 0.631)))),
    'rainbow_muted': LinearSegmentedColormap.from_list('my_gradient', (
    # Edit this gradient at https://eltos.github.io/gradient/#BF6C75-FF97A1-FFC571-FFEC94-F9FFA5-A2FF94-8ACC89-84D996-92FFFC-8AD3FF-A7AEFF-C59DFF-A678B9
    (0.000, (0.749, 0.424, 0.459)),
    (0.083, (1.000, 0.592, 0.631)),
    (0.167, (1.000, 0.773, 0.443)),
    (0.250, (1.000, 0.925, 0.580)),
    (0.333, (0.976, 1.000, 0.647)),
    (0.417, (0.635, 1.000, 0.580)),
    (0.500, (0.541, 0.800, 0.537)),
    (0.583, (0.518, 0.851, 0.588)),
    (0.667, (0.573, 1.000, 0.988)),
    (0.750, (0.541, 0.827, 1.000)),
    (0.833, (0.655, 0.682, 1.000)),
    (0.917, (0.773, 0.616, 1.000)),
    (1.000, (0.651, 0.471, 0.725)))),
    'SimCLR_embedding':LinearSegmentedColormap.from_list('SimCLR_embedding', (
    # Edit this gradient at https://eltos.github.io/gradient/#EDDFF7-3F2352
    (0.000, (0.929, 0.875, 0.969)),
    (1.000, (0.247, 0.137, 0.322)))),
    'SLIP_embedding': LinearSegmentedColormap.from_list('my_gradient', (
    # Edit this gradient at https://eltos.github.io/gradient/#F5D0DF-791932
    (0.000, (0.961, 0.816, 0.875)),
    (1.000, (0.475, 0.098, 0.196)))),
    'SimCLR_attention': LinearSegmentedColormap.from_list('my_gradient', (
    # Edit this gradient at https://eltos.github.io/gradient/#FFFDEA-79521A
    (0.000, (1.000, 0.992, 0.918)),
    (1.000, (0.475, 0.322, 0.102)))),
    'SLIP_attention': LinearSegmentedColormap.from_list('my_gradient', (
    # Edit this gradient at https://eltos.github.io/gradient/#EFF4F3-1A7975
    (0.000, (0.937, 0.957, 0.953)),
    (1.000, (0.102, 0.475, 0.459)))),
    'blue_neg_yellow_pos': LinearSegmentedColormap.from_list('yellow_hot', (
                # Edit this gradient at https://eltos.github.io/gradient/#4C71FF-0025B3-000000-C73A03-FCEB4A
                (0.000, (0.298, 0.443, 1.000)),
                (0.250, (0.000, 0.145, 0.702)),
                (0.500, (0.000, 0.000, 0.000)),
                (0.750, (0.780, 0.227, 0.012)),
                (1.000, (0.988, 0.922, 0.290)))),
    'yellow_hot': LinearSegmentedColormap.from_list('my_gradient', (
        # Edit this gradient at https://eltos.github.io/gradient/#0:000000-51:C73A03-100:FCEB4A
        (0.000, (0.000, 0.000, 0.000)),
        (0.510, (0.780, 0.227, 0.012)),
        (1.000, (0.988, 0.922, 0.290)))),
    'teal_orange': LinearSegmentedColormap.from_list('my_gradient', (
    # Edit this gradient at https://eltos.github.io/gradient/#0:FFAC00-19.9:9D7625-50:000000-82:1F6C5F-100:00FFD3
    (0.000, (1.000, 0.675, 0.000)),
    (0.199, (0.616, 0.463, 0.145)),
    (0.500, (0.000, 0.000, 0.000)),
    (0.820, (0.122, 0.424, 0.373)),
    (1.000, (0.000, 1.000, 0.827)))),
    'alexnet': LinearSegmentedColormap.from_list('my_gradient', (
    # Edit this gradient at https://eltos.github.io/gradient/#E1E9FE-173C92
    (0.000, (0.882, 0.914, 0.996)),
    (1.000, (0.090, 0.235, 0.573)))),
    'sbert': LinearSegmentedColormap.from_list('my_gradient', (
    # Edit this gradient at https://eltos.github.io/gradient/#FFEADA-E07F2E
    (0.000, (1.000, 0.918, 0.855)),
    (1.000, (0.878, 0.498, 0.180)))),
    'sbert_alexnet': LinearSegmentedColormap.from_list('my_gradient', (
    # Edit this gradient at https://eltos.github.io/gradient/#E07F2E-000000-173C92
    (0.000, (0.878, 0.498, 0.180)),
    (0.500, (0.000, 0.000, 0.000)),
    (1.000, (0.090, 0.235, 0.573)))),
    'alexnet_sbert': LinearSegmentedColormap.from_list('my_gradient', (
    # Edit this gradient at https://eltos.github.io/gradient/#173C92-000000-E07F2E
    (0.000, (0.090, 0.235, 0.573)),
    (0.500, (0.000, 0.000, 0.000)),
    (1.000, (0.878, 0.498, 0.180)))),
    'just_purple':LinearSegmentedColormap.from_list('my_gradient', (
    # Edit this gradient at https://eltos.github.io/gradient/#A28DC6
    (0.000, (0.635, 0.553, 0.776)),
    (1.000, (0.635, 0.553, 0.776)))),
    'audio':LinearSegmentedColormap.from_list('my_gradient', (
    # Edit this gradient at https://eltos.github.io/gradient/#FEFDFF-97002B
    (0.000, (0.996, 0.992, 1.000)),
    (1.000, (0.592, 0.000, 0.169)))),
    'red':LinearSegmentedColormap.from_list('my_gradient', (
    # Edit this gradient at https://eltos.github.io/gradient/#FFE0D6-902F31
    (0.000, (1.000, 0.878, 0.839)),
    (1.000, (0.565, 0.184, 0.192)))),
    'orange':LinearSegmentedColormap.from_list('my_gradient', (
    # Edit this gradient at https://eltos.github.io/gradient/#FFF1D6-A77000
    (0.000, (1.000, 0.945, 0.839)),
    (1.000, (0.655, 0.439, 0.000)))),
    'green':LinearSegmentedColormap.from_list('my_gradient', (
    # Edit this gradient at https://eltos.github.io/gradient/#D6FFD9-02770A
    (0.000, (0.839, 1.000, 0.851)),
    (1.000, (0.008, 0.467, 0.039)))),
    'blue':LinearSegmentedColormap.from_list('my_gradient', (
    # Edit this gradient at https://eltos.github.io/gradient/#D6D6FF-010677
    (0.000, (0.839, 0.839, 1.000)),
    (1.000, (0.004, 0.024, 0.467)))),
    'purple':LinearSegmentedColormap.from_list('my_gradient', (
    # Edit this gradient at https://eltos.github.io/gradient/#E5D6FF-522F90
    (0.000, (0.898, 0.839, 1.000)),
    (1.000, (0.322, 0.184, 0.565)))),
    'SLIP':LinearSegmentedColormap.from_list('my_gradient', (
    # Edit this gradient at https://eltos.github.io/gradient/#4A34B0-4A34B0
    (0.000, (0.290, 0.204, 0.690)),
    (1.000, (0.290, 0.204, 0.690)))),
    'SimCLR':LinearSegmentedColormap.from_list('my_gradient', (
    # Edit this gradient at https://eltos.github.io/gradient/#58B9C8-58B9C8
    (0.000, (0.345, 0.725, 0.784)),
    (1.000, (0.345, 0.725, 0.784)))),
    'GPT2':LinearSegmentedColormap.from_list('my_gradient', (
    # Edit this gradient at https://eltos.github.io/gradient/#810014-810014
    (0.000, (0.506, 0.000, 0.078)),
    (1.000, (0.506, 0.000, 0.078)))),
    'SLIPtext':LinearSegmentedColormap.from_list('my_gradient', (
        (0.000, (0.663, 0.624, 0.624)),
        (1.000, (0.584, 0.110, 0.929))))

    }

    cmaps['GPT2_1sent'] = cmaps['GPT2']
    cmaps['GPT2'] =cmaps['red']
    cmaps['GPT2_3sent'] = cmaps['sbert']
    cmaps['cochdnn']=cmaps['audio']
    cmaps['hubert']=cmaps['audio']
    cmaps['GPT2_1word'] = cmaps['red']
    cmaps['GPT2_1s'] = cmaps['orange']
    cmaps['GPT2_4s'] = cmaps['orange']
    cmaps['GPT2_5s'] = cmaps['orange']
    cmaps['GPT2_8s'] = cmaps['green']
    cmaps['GPT2_10s'] = cmaps['green']
    cmaps['GPT2_16s'] = cmaps['blue']
    cmaps['GPT2_20s'] = cmaps['blue']
    cmaps['GPT2_24s'] = cmaps['purple']
    cmaps['SimCLR_attention'] = cmaps['SimCLR']#cmaps['SimCLR']
    cmaps['SimCLR_embedding'] = cmaps['SimCLR']
    cmaps['SLIP_attention'] = cmaps['rainbow']#cmaps['SLIP']
    cmaps['SLIP_embedding'] = cmaps['SLIP']
    cmaps['SLIP_100ep_attention'] = cmaps['rainbow']
    cmaps['SLIP_100ep_embedding'] = cmaps['rainbow_muted']
    cmaps['SLIP_100ep'] = cmaps['rainbow']
    cmaps['SLIPtext_100ep'] = cmaps['rainbow']
    
    return cmaps

def get_models_dict():
    models_dict = {
                   'sbert':['sbert_layer'+str(layer) for layer in [1,2,3,4,5,6,7,8,9,10,11,12]],
                   'GPT2_1sent':['GPT2_1sent_layer'+str(layer) for layer in [1,2,3,4,5,6,7,8,9,10,11,12]],
                   'GPT2_3sent':['GPT2_3sent_layer'+str(layer) for layer in [1,2,3,4,5,6,7,8,9,10,11,12]],
                   'GPT2_1word':['GPT2_1word_layer'+str(layer) for layer in [1,2,3,4,5,6,7,8,9,10,11,12]],
                   'GPT2_4s':['GPT2_4s_layer'+str(layer) for layer in [1,2,3,4,5,6,7,8,9,10,11,12]],
                   'GPT2_8s':['GPT2_8s_layer'+str(layer) for layer in [1,2,3,4,5,6,7,8,9,10,11,12]],
                   'GPT2_16s':['GPT2_16s_layer'+str(layer) for layer in [1,2,3,4,5,6,7,8,9,10,11,12]],
                   'GPT2_24s':['GPT2_24s_layer'+str(layer) for layer in [1,2,3,4,5,6,7,8,9,10,11,12]],
                   'social':['social'],
                   'motion':['motion'],
                   'word2vec':['word2vec'],
                   'alexnet':['alexnet_layer'+str(layer) for layer in [1,2,3,4,5,6,7]],
                   'cochdnn':['cochdnn_layer'+str(layer) for layer in [0,1,2,3,4,5,6]],
                   'hubert':['hubert_layer'+str(layer) for layer in [1,2,3,4,5,6,7,8,9,10,11,12]],
                   'annotated':['social','num_agents','face','valence','arousal','speaking','turn_taking','mentalization','written_text','music'],
                   'SimCLR_attention':['SimCLR_attention_layer'+str(layer) for layer in [1,2,3,4,5,6,7,8,9,10,11,12]],
                   'SLIP_attention':['SLIP_attention_layer'+str(layer) for layer in [1,2,3,4,5,6,7,8,9,10,11,12]],
                   'SimCLR_embedding':['SimCLR_embedding_layer'+str(layer) for layer in [1,2,3,4,5,6,7,8,9,10,11,12]],
                   'SLIP_embedding':['SLIP_embedding_layer'+str(layer) for layer in [1,2,3,4,5,6,7,8,9,10,11,12]],
                   'SLIP_100ep_attention':['SLIP_100ep_attention_layer'+str(layer) for layer in [1,2,3,4,5,6,7,8,9,10,11,12]],
                   'SLIP_100ep_embedding':['SLIP_100ep_embedding_layer'+str(layer) for layer in [1,2,3,4,5,6,7,8,9,10,11,12]],
                   'SLIPtext':['SLIPtext_layer'+str(layer) for layer in [1,2,3,4,5,6,7,8,9,10,11,12]],
                   'SLIPtext_100ep':['SLIPtext_100ep_layer'+str(layer) for layer in [1,2,3,4,5,6,7,8,9,10,11,12]]          }
        # self.model_features_dict['full']=self.model_features_dict['alexnet_layers']+self.model_features_dict['sbert_layers']+['social','num_agents','speaking','turn_taking','mentalization','word2vec','valence','arousal','motion','face','indoor_outdoor','written_text','pixel','hue','pitch','amplitude','music']
    models_dict['GPT2'] = models_dict['GPT2_1sent']
    models_dict['SLIP']=models_dict['SLIP_attention']#+models_dict['SLIP_embedding']
    models_dict['SimCLR']=models_dict['SimCLR_attention']#+models_dict['SimCLR_embedding']
    models_dict['SLIP_100ep']=models_dict['SLIP_100ep_attention']#+models_dict['SLIP_100ep_embedding']
    
    models_dict['SimCLR_SLIP']=models_dict['SimCLR']+models_dict['SLIP']
    models_dict['SimCLR_SLIP_SLIPtext']=models_dict['SimCLR_attention']+models_dict['SLIP_attention']+models_dict['SLIPtext']
    models_dict['SimCLR_SLIP_SLIPtext_word2vec']=models_dict['SimCLR']+models_dict['SLIP']+models_dict['SLIPtext']+['word2vec']
    models_dict['SLIP_SLIPtext']=models_dict['SLIP']+models_dict['SLIPtext']

    models_dict['SLIP_SLIPtext_word2vec']=models_dict['SLIP']+models_dict['SLIPtext']+models_dict['word2vec']
    models_dict['SimCLR_SLIPtext_word2vec']=models_dict['SimCLR']+models_dict['SLIPtext']+models_dict['word2vec']

    models_dict['SimCLR_SLIPtext']=models_dict['SimCLR']+models_dict['SLIPtext']
    models_dict['SLIPtext_word2vec']=models_dict['SLIPtext'] +models_dict['word2vec']
    models_dict['SLIPtext_word2vec_GPT2']=models_dict['SLIPtext'] +models_dict['word2vec']+models_dict['GPT2_1sent']
    models_dict['SimCLR_SLIP_SLIPtext_GPT2_word2vec']=models_dict['SimCLR']+models_dict['SLIP']+models_dict['SLIPtext']+models_dict['GPT2_1sent']+models_dict['word2vec']
    models_dict['SimCLR_SLIPtext_GPT2_word2vec']=models_dict['SimCLR']+models_dict['SLIPtext']+models_dict['GPT2_1sent']+models_dict['word2vec']
    models_dict['SimCLR_SLIP_GPT2_word2vec']=models_dict['SimCLR']+models_dict['SLIP']+models_dict['GPT2_1sent']+models_dict['word2vec']
    models_dict['SLIP_SLIPtext_GPT2_word2vec']=models_dict['SLIP']+models_dict['SLIPtext']+models_dict['GPT2_1sent']+models_dict['word2vec']
    models_dict['SimCLR_SLIP_SLIPtext_GPT2']=models_dict['SimCLR']+models_dict['SLIP']+models_dict['SLIPtext']+models_dict['GPT2_1sent']

    models_dict['SimCLR_SLIP_SLIPtext_GPT2_word2vec']=models_dict['SimCLR_attention']+models_dict['SLIP_attention']+models_dict['SLIPtext']+models_dict['GPT2_1sent']+models_dict['word2vec']


    models_dict['SimCLR_SLIP_SLIPtext_100ep'] = models_dict['SimCLR_attention'] + models_dict['SimCLR_embedding']+ models_dict['SLIP_100ep_attention'] + models_dict['SLIP_100ep_embedding'] + models_dict['SLIPtext_100ep']
    models_dict['SimCLR_SLIP_100ep'] = models_dict['SimCLR_attention'] + models_dict['SimCLR_embedding']+models_dict['SLIP_100ep_attention'] + models_dict['SLIP_100ep_embedding'] 
    
    # models_dict['SLIP_SLIPtext_100ep_word2vec_GPT2']=models_dict['SLIP_100ep']+models_dict['SLIPtext_100ep']+models_dict['GPT2_1sent']+models_dict['word2vec']
    
    models_dict['SimCLR_SLIP_SLIPtext_100ep_word2vec_GPT2']=models_dict['SimCLR']+models_dict['SLIP_100ep']+models_dict['SLIPtext_100ep']+models_dict['GPT2_1sent']+models_dict['word2vec']

    models_dict['attention'] = models_dict['SimCLR_attention'] + models_dict['SLIP_attention']
    models_dict['embedding'] = models_dict['SimCLR_embedding'] + models_dict['SLIP_embedding']

    models_dict['SimCLR+SLIP']=models_dict['SimCLR']+models_dict['SLIP']
    models_dict['alexnet+motion'] = models_dict['alexnet']+models_dict['motion']
    models_dict['sbert+word2vec'] = models_dict['sbert']+models_dict['word2vec']
    models_dict['hubert+sbert+word2vec'] = models_dict['hubert']+models_dict['sbert']+models_dict['word2vec']
    models_dict['GPT2+word2vec'] = models_dict['GPT2_1sent']+models_dict['word2vec']
    models_dict['word2vec+GPT2'] = models_dict['GPT2+word2vec']
    
    models_dict['audio_lang'] = models_dict['sbert']+models_dict['cochdnn']+models_dict['word2vec']+models_dict['alexnet']+models_dict['motion']
    models_dict['hubert_lang_vis'] = models_dict['sbert']+models_dict['hubert']+models_dict['word2vec']+models_dict['SimCLR']+models_dict['motion']
    
    models_dict['joint'] = models_dict['annotated'] + models_dict['motion'] + models_dict['alexnet'] + models_dict['hubert'] + models_dict['word2vec'] + models_dict['sbert']
    models_dict['full'] = models_dict['annotated'] + models_dict['motion'] + models_dict['alexnet'] + models_dict['word2vec'] + models_dict['sbert']
    models_dict['joint_transformers'] = models_dict['annotated'] + models_dict['motion'] + models_dict['SimCLR_attention'] + models_dict['hubert'] + models_dict['word2vec'] + models_dict['GPT2_1sent']
    models_dict['GPT2_diff_contexts'] = models_dict['SimCLR_attention'] + models_dict['GPT2_1word'] + models_dict['GPT2_4s'] + models_dict['GPT2_8s'] + models_dict['GPT2_16s'] + models_dict['GPT2_24s']
    models_dict['GPT2_SimCLR_SLIP_word2vec'] = models_dict['GPT2_1sent']+ models_dict['SimCLR']+models_dict['SLIP'] + models_dict['word2vec']
    for layer in [1,2,3,4,5,6,7,8,9,10,11,12]:
        models_dict['SimCLR_SLIP_layer'+str(layer)] = [model+'_'+layer_type+'_layer'+str(layer) for layer_type in ['attention','embedding'] for model in ['SimCLR','SLIP']]
    
    return models_dict

def get_combined_features():
    return ['SLIPtext','cochdnn','hubert','GPT2','GPT2_1sent','GPT2_3sent','sbert_layers','alexnet_layers','alexnet','sbert+word2vec','sbert','alexnet+motion','SimCLR_attention','SLIP_attention','SLIP_100ep_attention','SimCLR_embedding','SLIP_embedding','SLIP_100ep_embedding','SLIP','SLIP_100ep','SLIPtext_100ep','SimCLR','SimCLR+SLIP','GPT2+word2vec','word2vec+GPT2'] + ['SimCLR_SLIP_layer'+str(layer) for layer in [1,2,3,4,5,6,7,8,9,10,11,12]]

def get_colors_dict():
    models_dict = get_models_dict()
    cmaps = get_cmaps()
    colors_dict = { 
                'ISC':'lightgray',
                None:'white',
                'GPT2_layer8':'darkorange',
                'GPT2':'#810014',
                'GPT2_1sent':'#810014',
                'GPT2_3sent':'darkorange',
                'SimCLR':'#58B9C8',
                'SLIP':'#5E3DC6',
                'SimCLR only':'#58B9C8',
                'SLIP only':'#5E3DC6',
                'SimCLR+SLIP':'gray',#'darkseagreen',#'plum',
                '+SLIP':'#5E3DC6',
                'SimCLR+SLIP+100ep':'yellow',
                'SimCLR+SLIP+SLIPtext':'#A324BF',#'darkviolet',
                '+SLIPtext':'#A324BF',
                'SimCLR+SLIP+SLIPtext+100ep':'orange',
                'SimCLR+SLIP+SLIPtext+word2vec':'#D61532',
                '+word2vec':'#D61532',
                'SimCLR+SLIP+SLIPtext+word2vec+GPT2':'#D61532',
                '+GPT2':'#D61532',
                '+word2vec+GPT2':'#D61532',
                'word2vec+GPT2':'#D61532',
                'SimCLR+SLIP+SLIPtext+GPT2+word2vec':'#810014',
                'SimCLR+SLIP+SLIPtext+100ep+word2vec+GPT2':'red',
                'SLIP_100ep':'#4A34B0',
                'SLIPtext':'purple',
                'attention':'olive',
                'embedding': 'darkkhaki',
                'blank':'white',
                'social':'limegreen',
                'num_agents':'olive',
                'face': 'green',
                'valence': 'plum',
                'arousal': 'palevioletred',
                'turn_taking':'brown',
                'mentalization': 'darkred',
                'word2vec':'indigo',
                'sbert': 'darkorange',
                # 'sbert_layers': 'darkorange',
                # 'glove': 'burlywood',
                'written_text': 'darkkhaki',
                'speaking': 'lightcoral',
                'amplitude': 'peru', 
                'pitch': 'sandybrown',
                'music': 'saddlebrown',
                'indoor_outdoor': 'aquamarine',
                'hue':'#856798', #xkcd dark lavender
                'pixel': '#d6b4fc',#xkcd light violet
                'motion':'#952e8f', #xkcd warm purple
                # 'alexnet': 'steelblue',
                # 'alexnet_layers': 'steelblue'
                'full model with SLIP trained 25 epochs': '#CE8CDD',
                'full model with SLIP trained 100 epochs':'#A324BF'
                 }
    for feature in ['hubert','cochdnn','alexnet','sbert','SimCLR_attention','SLIP_attention','SimCLR_embedding','SLIP_embedding','SLIP_100ep','SLIPtext_100ep','SLIP_100ep_attention','SLIP_100ep_embedding','GPT2_1sent','GPT2_3sent','GPT2']:
        colors_dict[feature]=cmaps[feature](0.9) #get the color for the combined layers, then assign more specific colors to each layer
        for i,layer in enumerate(models_dict[feature]):
            colors_dict[layer] = cmaps[feature](i/len(models_dict[feature])) #normalized by total number of layers

    colors_dict['SimCLR_attention'] = '#58B9C8'
    colors_dict['SLIP_attention'] = '#4A34B0'
    colors_dict['SLIP_100ep_attention'] =  '#4A34B0'
    colors_dict['SLIP_100ep'] =  '#4A34B0'
    colors_dict['SimCLR_embedding'] = '#A2C3C8'
    colors_dict['SLIP_embedding'] = '#7E73B0'
    colors_dict['SLIP_100p_embedding'] = '#7E73B0'
    colors_dict['interact-no_interact'] = colors_dict['social']
    colors_dict['interact&no_interact'] = colors_dict['motion']
    colors_dict['intact-degraded'] = colors_dict['sbert']
    colors_dict['social interaction'] = colors_dict['social']
    colors_dict['language'] = colors_dict['sbert']
    colors_dict['alexnet+motion'] = colors_dict['alexnet']
    colors_dict['alexnet'] = colors_dict['alexnet']
    colors_dict['sbert+word2vec'] = colors_dict['sbert']
    colors_dict['hubert'] = colors_dict['hubert']
    colors_dict['cochdnn'] = colors_dict['cochdnn']

    colors_dict['SimCLR only'] = '#58B9C8'
    colors_dict['SLIP only'] = '#5E3DC6'
    colors_dict['word2vec'] = '#D61532'
    colors_dict['SLIPtext'] = '#A324BF'
    colors_dict['SLIPtext_100ep'] = '#951CED'
    colors_dict['GPT2_1sent'] = '#810014'
    colors_dict['GPT2'] = '#810014'
    colors_dict['+GPT2'] = '#810014'
    colors_dict['+word2vec+GPT2'] = '#D61532'


    return colors_dict

def get_top_percent(data, percent):
    import numpy as np
    
    # data = np.abs(data) ???
    percentile = 100-percent

    threshold = np.nanpercentile(data, percentile)

    num_top_voxels = len(data[data>threshold])

    return (threshold, num_top_voxels)

def get_top_n(data, n):
    import numpy as np
    
    # data = np.abs(data)
    if(data.shape[0]>100):
        x = n/data.shape[0]*100
        percentile = 100-x
        if(percentile<0):
            percentile=0
    else:
        percentile = 0 #take all of the voxels if there are less than 100
    
    # print('percentile')
    # print(percentile)
    # print('data')
    # print(data)
    threshold = np.nanpercentile(data, percentile)

    num_top_voxels = len(data[data>threshold])

    return (threshold, num_top_voxels)
def get_bottom_n(data, n):
    import numpy as np
    
    # data = np.abs(data)
    if(data.shape[0]>100):
        x = n/data.shape[0]*100
        percentile = x
    else:
        percentile = 0 #take all of the voxels if there are less than 100
    
    print('percentile')
    print(percentile)
    threshold = np.nanpercentile(data, percentile)

    num_bottom_voxels = len(data[data<threshold])

    return (threshold, num_bottom_voxels)

def get_bottom_percent(data, percent):
    import numpy as np
    
    data = np.abs(data)
    percentile = percent

    threshold = np.nanpercentile(data, percentile)

    num_bottom_voxels = len(data[data<threshold])

    return (threshold, num_bottom_voxels)

def plot_img_volume(img,filepath,threshold=None,vmin=None,vmax=None,cmap='cold_hot',title=None,symmetric_cbar='auto'):
    import nibabel
    import numpy as np
    from nilearn import plotting

    #NOTE: view_surf simply plots a surface mesh on a brain, no tri averaging like plot_surf_stat_map
    #NOTE: you can't have any NaN values when using view_surf -- it doesn't handle the threshold correctly
    # so, converting all NaN's negative infinity and then turning all negatives to 0 for plotting
    
    display = plotting.plot_glass_brain(
            stat_map_img = img,
            output_file=filepath,
            colorbar=True,
            cmap=cmap,
            threshold=threshold,
            display_mode='lr',#'lyrz',lzr
            vmin=vmin,
            vmax=vmax,
            title=title,
            symmetric_cbar=symmetric_cbar,
            plot_abs=False
            # norm=norm
            ) 
    # view = plotting.view_img(
    #     img, title=title, cut_coords=[36, -27, 66],vmin=0,vmax=0.5, symmetric_cmap=False,opacity=0.5,
    # )
    # view.open_in_browser()

def plot_surface(nii, filename, ROI_niis=[], threshold=None, vmin=None, vmax=None, title=None, cmap='cold_hot', symmetric_cbar='auto',colorbar_label='',views=['lateral','ventral'],ROIs=[],ROI_colors=[]):
    import nilearn.datasets
    from nilearn import surface
    from nilearn import plotting
    import matplotlib.pyplot as plt
    import numpy as np
    import os

    temp_filename = filename#'/'.join(filename.split('/')[:-1])

    fsaverage = nilearn.datasets.fetch_surf_fsaverage(mesh='fsaverage') 

    #get the max value if not specified
    if(vmax is None):
        temp =[]
        for hemi in ('left','right'):
            transform_mesh = fsaverage['pial_' + hemi]
            texture = surface.vol_to_surf(nii, transform_mesh, interpolation='nearest')
            temp.append(round(np.nanmax(texture),2))
        vmax = np.max(temp)
    if(vmin is None):
        temp=[]
        for hemi in ('left','right'):
            transform_mesh = fsaverage['pial_' + hemi]
            texture = surface.vol_to_surf(nii, transform_mesh, interpolation='nearest')
            temp.append(round(np.nanmin(texture),2))
        vmin = np.min(temp)

    vmax = np.max([vmax,-vmin])
    if(vmin!=0):
        vmin = -vmax
    #### SAVE each brain separately ######
    
    plt.rcParams.update({'font.size': 10, 'font.family': 'Arial'})
    
    for (hemi, view) in [(hemi,view) for view in views for hemi in ['left','right']]:
        fig, ax = plt.subplots(subplot_kw={'projection': '3d'}, figsize=(5, 5))
        
        transform_mesh = fsaverage['pial_' + hemi]
        plot_mesh = fsaverage['infl_' + hemi]
        bg_map = fsaverage['sulc_' + hemi]
        inner_mesh = fsaverage['white_' + hemi]

        n_points_to_sample = 50
        texture = surface.vol_to_surf(nii, transform_mesh, inner_mesh=inner_mesh, depth = np.linspace(0, 1, n_points_to_sample), interpolation='nearest')
        texture = np.nan_to_num(texture, nan=0)

        if vmin is not None:
            texture[texture < vmin] = vmin
        ## TODO, when plotting, should I threshold the volume and not threshold the surface?
        if threshold is not None:
            texture[np.abs(texture) < threshold] = 0

        plotting.plot_surf_stat_map(plot_mesh, texture, hemi=hemi,
                                    view=view, colorbar=False,
                                    threshold=0.000000000000000000001,
                                    bg_map=bg_map,
                                    cmap=cmap,
                                    symmetric_cbar=symmetric_cbar,
                                    # vmin=vmin,
                                    vmax=vmax,
                                    axes=ax,#axes_dict[(hemi, view)],
                                    engine='matplotlib')
        for ind,ROI_nii in enumerate(ROI_niis):
            roi_texture = surface.vol_to_surf(ROI_nii,transform_mesh,inner_mesh=inner_mesh, depth = np.linspace(0, 1, n_points_to_sample),interpolation='linear')
            roi_texture = (roi_texture>0.03)*1 #binarize the surface ROI map, anything that was part of the ROI in volume should be in surface
            try:
                # plotting.plot_surf_contours(plot_mesh,roi_texture,avg_method='mean',axes=ax,levels=[ind for ind in np.arange(1,len(ROIs)+1)],labels=ROIs,colors=ROI_colors)
                plotting.plot_surf_contours(plot_mesh,roi_texture,avg_method='mean',axes=ax,levels=[1],labels=[ROIs[ind]],colors=[ROI_colors[ind]])
            except Exception as e:
                print(filename)
                print(e)
                pass

        if((hemi=='left')&(view=='ventral')):
            # axes_dict[(hemi, view)].view_init(elev=270,azim=180)# Rotate the bottom left subplot (left, ventral) by 180 degrees
            ax.view_init(elev=270,azim=180)# Rotate the (left, ventral) view by 180 degrees

        plt.savefig(temp_filename+'_'+hemi+'_'+view+'.png', bbox_inches='tight', transparent=True, dpi=300)
        plt.close()

    #get an image for the colorbar
    if(colorbar_label!=''): #if there is a colorbar label, generate a colorbar
        save_colorbar(cmap, vmin, vmax, temp_filename+'_colorbar.png',colorbar_label)
        colorbar_filepath = temp_filename+'_colorbar.png'
    else:
        colorbar_filepath = ''
    ##crop each separate brain plot
    for (hemi,view) in [(hemi,view) for view in views for hemi in ['left','right']]:
        if(view=='lateral'):
            left=150
            top=240
            width=1100
            height=950
        elif(view=='ventral'):
            left=150
            top=400
            width=1100
            height=775
        elif(view=='medial'):
            left=150
            top=215
            width=1100
            height=950
        crop_image(temp_filename+'_'+hemi+'_'+view+'.png',temp_filename+'_'+hemi+'_'+view+'.png',left,top,width,height)
    
    ### put together all of the cropped images into one plot
    list_images = [temp_filename+'_'+hemi+'_'+view+'.png' for view in views for hemi in ['left','right']]
    compose_final_figure(filename+'.png', list_images, colorbar_filepath, title=title)
    for (hemi,view) in [(hemi,view) for view in views for hemi in ['left','right']]:
            delete_file = temp_filename+'_'+hemi+'_'+view+'.png'
            os.remove(delete_file)
    if(colorbar_label!=''):
        os.remove(colorbar_filepath)

def save_colorbar(cmap, vmin, vmax, filename,colorbar_label,make_cmap=True):
    import matplotlib.pyplot as plt
    import numpy as np
    
    colorbar_height = 3.5
    fig, ax = plt.subplots(figsize=(0.5, colorbar_height))
    fig.subplots_adjust(right=0.5)
    
    if(make_cmap):
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
    else:
        sm=plt.cm.ScalarMappable(cmap=cmap)

    
    cbar = plt.colorbar(sm, cax=ax,aspect=50)
    cbar.set_label(colorbar_label)
    cbar.ax.yaxis.set_label_position('left')
    cbar.ax.yaxis.set_ticks_position('right')
    cbar.ax.set_ylim(vmin, vmax)
    
    plt.savefig(filename, bbox_inches='tight', transparent=True, dpi=300)
    plt.close()

def crop_image(input_path, output_path, left, top, right, bottom):
    from PIL import Image
    with Image.open(input_path) as img:
        cropped_img = img.crop((left, top, right, bottom))
        cropped_img.save(output_path)

def compose_final_figure(output_filename, cropped_images, colorbar_image, title=None):
    from PIL import Image, ImageDraw, ImageFont
    import math

    images = [Image.open(img_path) for img_path in cropped_images]
    widths, heights = zip(*(img.size for img in images))

    max_width = max(widths)
    max_height = max(heights)

    # Calculate the grid size
    num_images = len(images)
    num_columns = 2
    num_rows = math.ceil(num_images / num_columns)

    if colorbar_image != '':
        colorbar = Image.open(colorbar_image)
        total_width = num_columns * max_width + colorbar.size[0]
    else:
        total_width = num_columns * max_width
    
      # Calculate cumulative heights
    cumulative_heights = []
    for row in range(num_rows):
        row_heights = heights[row * num_columns:(row + 1) * num_columns]
        cumulative_heights.append(max(row_heights))

    total_height = sum(cumulative_heights)

    final_image = Image.new('RGBA', (total_width, total_height), (255, 255, 255, 0))

    # Place images in the grid
    y_offset = 0
    for row in range(num_rows):
        for col in range(num_columns):
            index = row * num_columns + col
            if index < num_images:
                img = images[index]
                position = (col * max_width, y_offset)
                final_image.paste(img, position)
        y_offset += cumulative_heights[row]
    
    if(colorbar_image!=''):
        final_image.paste(colorbar, (2 * max_width,int(total_height/10) ))

    if title:
        draw = ImageDraw.Draw(final_image)
        try:
            font = ImageFont.truetype("Arial Bold.ttf", 30)
        except IOError:
            font = ImageFont.load_default()
        draw.text((max_width-85, 10), title, fill="black", font=font)

    final_image.save(output_filename)

def plot_preference_surf(textures,filepath,color_dict,labels,views=['lateral','ventral'],threshold=None,vmin=None,vmax=None,cmap='cold_hot',make_cmap=True,title=None,symmetric_cbar='auto',colorbar_label=''):
    import nilearn.datasets
    from nilearn import surface
    from nilearn import plotting
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    fsaverage = nilearn.datasets.fetch_surf_fsaverage(mesh='fsaverage') 
    temp_filename = filepath

    plt.rcParams.update({'font.size': 10,'font.family': 'Arial'})
    # views = [('left','lateral'),('right','lateral'),('left','ventral'),('right','ventral')]
    for (hemi, view, texture) in [(hemi,view,texture) for view in views for (hemi,texture) in zip(['left','right'],textures)]:
    # for (hemi,view),texture in zip(views,[textures[0],textures[1],textures[0],textures[1]]):
        fig, ax = plt.subplots(subplot_kw={'projection': '3d'}, figsize=(5, 5))

        transform_mesh = fsaverage['pial_'+hemi]
        plot_mesh = fsaverage['infl_'+hemi]
        bg_map = fsaverage['sulc_'+hemi]

        # print(texture.shape)
        # print(plot_mesh.shape)

        # texture = np.nan_to_num(texture, nan=0)
        plotting.plot_surf_roi(plot_mesh, texture, hemi=hemi,
                          view=view, colorbar=False,
                          threshold=threshold,
                          label=title,
                          bg_map=bg_map,
                          cmap=cmap,
                          vmax=vmax,
                          axes=ax,
                          engine='matplotlib')
        
        if((hemi=='left')&(view=='ventral')):
            # axes_dict[(hemi, view)].view_init(elev=270,azim=180)# Rotate the bottom left subplot (left, ventral) by 180 degrees
            ax.view_init(elev=270,azim=180)# Rotate the (left, ventral) view by 180 degrees

        plt.savefig(filepath+'_'+hemi+'_'+view+'.png', bbox_inches='tight', transparent=True, dpi=300)
        plt.close()

    #get an image for the colorbar
    if(colorbar_label!=''): #if there is a colorbar label, generate a colorbar
        save_colorbar(cmap, vmin, vmax, temp_filename+'_colorbar.png',colorbar_label,make_cmap=make_cmap)
        colorbar_filepath = temp_filename+'_colorbar.png'
    else:
        colorbar_filepath = ''
    ##crop each separate brain plot
    for (hemi,view) in [(hemi,view) for view in views for hemi in ['left','right']]:
        if(view=='lateral'):
            left=150
            top=240
            width=1100
            height=950
        elif(view=='ventral'):
            left=150
            top=400
            width=1100
            height=775
        elif(view=='medial'):
            left=150
            top=215
            width=1100
            height=950
        crop_image(temp_filename+'_'+hemi+'_'+view+'.png',temp_filename+'_'+hemi+'_'+view+'.png',left,top,width,height)
    
    ### put together all of the cropped images into one plot
    list_images = [temp_filename+'_'+hemi+'_'+view+'.png' for view in views for hemi in ['left','right']]
    compose_final_figure(temp_filename+'.png', list_images, colorbar_filepath, title=title)
    for (hemi,view) in [(hemi,view) for view in ['lateral','ventral'] for hemi in ['left','right']]:
            delete_file = temp_filename+'_'+hemi+'_'+view+'.png'
            os.remove(delete_file)
    if(colorbar_label!=''):
        os.remove(colorbar_filepath)
def plot_preference_img_volume(img,filepath,color_dict,labels,threshold=None,vmin=None,vmax=None,cmap='cold_hot',title=None,):
    import nibabel
    import numpy as np
    from nilearn import plotting
    import matplotlib.pyplot as plt

    #NOTE: view_surf simply plots a surface mesh on a brain, no tri averaging like plot_surf_stat_map
    #NOTE: you can't have any NaN values when using view_surf -- it doesn't handle the threshold correctly
    # so, converting all NaN's negative infinity and then turning all negatives to 0 for plotting

    fig = plt.figure(1,figsize=(6.4,3.7))
    display = plotting.plot_glass_brain(
            stat_map_img = img,
            # output_file=filepath,
            colorbar=False,
            cmap=cmap,
            threshold=threshold,
            display_mode='lyrz',#'lyrz',lzr
            vmin=vmin,
            vmax=vmax,
            title=title,
            figure=fig
            # norm=norm
            )


    scale_factor = 7.5
    ax = display.axes['l'].ax
    ylim = ax.get_ylim()
    xlim = ax.get_xlim()
    x_scale = abs(xlim[1]-xlim[0])/scale_factor
    y_scale = abs(ylim[1]-ylim[0])/scale_factor
    xlim_start = xlim[0]-x_scale
    ylim_start = ylim[0]-y_scale
    for ind,label in enumerate(labels[1:6]):
        ax.text(xlim_start, ylim_start-(y_scale*ind), label, fontsize='small',
                bbox={'fc':color_dict[label], 'pad':2})

    ax = display.axes['y'].ax
    ylim = ax.get_ylim()
    xlim = ax.get_xlim()
    x_scale = abs(xlim[1]-xlim[0])/scale_factor
    y_scale = abs(ylim[1]-ylim[0])/scale_factor
    xlim_start = xlim[0] + x_scale
    ylim_start = ylim[0]-y_scale
    for ind,label in enumerate(labels[6:11]):
        ax.text(xlim_start, ylim_start-(y_scale*ind), label,fontsize='small',
                bbox={'fc':color_dict[label], 'pad':2})

    ax = display.axes['r'].ax
    ylim = ax.get_ylim()
    xlim = ax.get_xlim()
    x_scale = abs(xlim[1]-xlim[0])/scale_factor
    y_scale = abs(ylim[1]-ylim[0])/scale_factor
    xlim_start = xlim[0] + x_scale
    ylim_start = ylim[0]-y_scale
    for ind,label in enumerate(labels[11:16]):
        ax.text(xlim_start, ylim_start-(y_scale*ind), label,fontsize='small',
                bbox={'fc':color_dict[label], 'pad':2})

    ax = display.axes['z'].ax
    ylim = ax.get_ylim()
    xlim = ax.get_xlim()
    x_scale = abs(xlim[1]-xlim[0])/scale_factor
    y_scale = abs(ylim[1]-ylim[0])/scale_factor
    xlim_start = xlim[0] + x_scale
    ylim_start = ylim[0]-y_scale
    for ind,label in enumerate(labels[16:20]):
        ax.text(xlim_start, ylim_start-(y_scale*ind), label,fontsize='small',
                bbox={'fc':color_dict[label], 'pad':2})

    display.savefig(filepath)
    plt.close()


