import numpy as np
from sklearn.model_selection import KFold,check_cv,GroupKFold
from sklearn.preprocessing import StandardScaler
import pandas as pd 
from pathlib import Path
import sys
sys.path.insert(1, './encoding')
import encoding
import argparse
from deepjuice import reduction
import torch
class FeatureSpaceCorrelation(encoding.EncodingModel):
    def __init__(self, args):
        self.process = 'FeatureSpaceCorrelation'
        self.chunklen = args.chunklen
        self.features = args.features
        self.feature1 = args.features.split('-')[0]
        self.feature2 = args.features.split('-')[1]
        self.method = args.method
        self.srp_matrices = {} #for feature space dim reduction
        self.dir = args.dir
        self.out_dir = args.out_dir + "/" + self.process
        self.figure_dir = args.figure_dir + "/" + self.process
        Path(f'{self.out_dir}/').mkdir(exist_ok=True, parents=True)
        self.features_dict = {
                       'sbert':['sbert_layer'+str(layer) for layer in [1,2,3,4,5,6,7,8,9,10,11,12]],
                       'GPT2_1sent':['GPT2_1sent_layer'+str(layer) for layer in [1,2,3,4,5,6,7,8,9,10,11,12]],
                       'GPT2_3sent':['GPT2_3sent_layer'+str(layer) for layer in [1,2,3,4,5,6,7,8,9,10,11,12]],
                       'GPT2_1word':['GPT2_1word_layer'+str(layer) for layer in [1,2,3,4,5,6,7,8,9,10,11,12]],
                       'GPT2_4s':['GPT2_4s_layer'+str(layer) for layer in [1,2,3,4,5,6,7,8,9,10,11,12]],
                       'GPT2_8s':['GPT2_8s_layer'+str(layer) for layer in [1,2,3,4,5,6,7,8,9,10,11,12]],
                       'GPT2_16s':['GPT2_16s_layer'+str(layer) for layer in [1,2,3,4,5,6,7,8,9,10,11,12]],
                       'GPT2_24s':['GPT2_24s_layer'+str(layer) for layer in [1,2,3,4,5,6,7,8,9,10,11,12]],
                       'SLIP':'slip_vit_b_yfcc15m',
                       'SimCLR':'slip_vit_b_simclr_yfcc15m',
                       'CLIP':'slip_vit_b_clip_yfcc15m',
                       'CLIPtext':'clip_base_25ep_text_embeddings',
                       'SLIPtext':'slip_base_25ep_text_embeddings',
                       'SLIPtext_100ep':'downsampled_slip_base_100ep_embeddings',
                       'CLIP_ViT':'clip_vitb32',
                       'CLIP_RN':'clip_rn50',
                       'alexnet_layer1':'torchvision_alexnet_imagenet1k_v1_MaxPool2d-2-3',
                       'alexnet_layer2':'torchvision_alexnet_imagenet1k_v1_MaxPool2d-2-6',
                       'alexnet_layer3':'torchvision_alexnet_imagenet1k_v1_ReLU-2-8',
                       'alexnet_layer4':'torchvision_alexnet_imagenet1k_v1_ReLU-2-10',
                       'alexnet_layer5':'torchvision_alexnet_imagenet1k_v1_MaxPool2d-2-13',
                       'alexnet_layer6':'torchvision_alexnet_imagenet1k_v1_ReLU-2-16',
                       'alexnet_layer7':'torchvision_alexnet_imagenet1k_v1_ReLU-2-19',
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
                       'social':'social',
                       'num_agents':'num_agents',
                       'turn_taking':'turn_taking',
                       'speaking':'speaking',
                       'mentalization': 'mentalization',
                       'valence':'valence',
                       'arousal':'arousal',
                       'motion':'pymoten',
                       'face': 'face',
                       'indoor_outdoor':'indoor_outdoor',
                       'written_text':'written_text',
                       'pixel':'pixel',
                       'hue':'hue',
                       'amplitude':'amplitude',
                       'pitch':'pitch',
                       'music':'music',
                       'glove':'glove',
                       'word2vec':'word2vec',
                       'alexnet':'alexnet_layer5_pca',
                       'speaking_turn_taking':'speaking_turn_taking',
                       'pitch_amplitude':'pitch_amplitude'}
        # for layer in [1,2,3,4,5,6,7,8,9,10,11,12]:
        #     self.features_dict['sbert_layer'+str(layer)]='downsampled_all-mpnet-base-v2_layer'+str(layer)
        for layer in self.features_dict['GPT2_1sent']:
            self.features_dict[layer]='downsampled_GPT2_'+layer.split('_')[2] + '_context-sent-1_embeddings'
        for layer in self.features_dict['GPT2_3sent']:
            self.features_dict[layer]='downsampled_GPT2_'+layer.split('_')[2] + '_context-sent-3_embeddings'
        for layer in self.features_dict['sbert']:
            self.features_dict[layer]='downsampled_all-mpnet-base-v2_'+layer.split('_')[1]
        for layer in self.features_dict['GPT2_1sent']:
            self.features_dict[layer]='downsampled_GPT2_'+layer.split('_')[2] + '_context-sent-1_embeddings'
        for layer in self.features_dict['GPT2_3sent']:
            self.features_dict[layer]='downsampled_GPT2_'+layer.split('_')[2] + '_context-sent-3_embeddings'
        for layer in self.features_dict['GPT2_1word']:
            self.features_dict[layer]='GPT2_'+layer.split('_')[2] + '_word'
        for layer in self.features_dict['hubert']:
            self.features_dict[layer]='hubert-base-ls960-ft_'+layer.split('_')[1]
        for layer in self.features_dict['SLIPtext']:
            self.features_dict[layer]='downsampled_sliptext_base_25ep_'+layer.split('_')[1]+'_embeddings'

        for time_chunk in [4,8,16,24]:
            for layer in self.features_dict['GPT2_'+str(time_chunk)+'s']:
                self.features_dict[layer]='GPT2_'+layer.split('_')[2] + '_time_chunk-'+str(time_chunk)
        
        tracker=2
        for layer in self.features_dict['SimCLR_attention']:
            self.features_dict[layer]='slip_vit_b_simclr_yfcc15m_attention-3-'+str(tracker) +'_srp' #get the SRP reduced version
            tracker=tracker+8

        tracker=6
        for layer in self.features_dict['SimCLR_embedding']:
            self.features_dict[layer]='slip_vit_b_simclr_yfcc15m_mlp-3-'+str(tracker) +'_srp' #get the SRP reduced version
            tracker=tracker+8
        
        tracker=2
        for layer in self.features_dict['SLIP_attention']:
            self.features_dict[layer]='slip_vit_b_yfcc15m_attention-3-'+str(tracker) +'_srp' #get the SRP reduced version
            tracker=tracker+8
        
        tracker=6
        for layer in self.features_dict['SLIP_embedding']:
            self.features_dict[layer]='slip_vit_b_yfcc15m_mlp-3-'+str(tracker) +'_srp' #get the SRP reduced version
            tracker=tracker+8

        tracker=2
        for layer in self.features_dict['SLIP_100ep_attention']:
            self.features_dict[layer]='slip_vit_b_max_yfcc15m_attention-3-'+str(tracker) +'_srp' #get the SRP reduced version
            tracker=tracker+8
        
        tracker=6
        for layer in self.features_dict['SLIP_100ep_embedding']:
            self.features_dict[layer]='slip_vit_b_max_yfcc15m_mlp-3-'+str(tracker) +'_srp' #get the SRP reduced version
            tracker=tracker+8


    def canonical_correlation_analysis(self,feature_names=[],latent_dimensions='auto',regularized=False,outer_folds=10,inner_folds=5):
        from cca_zoo.model_selection import GridSearchCV
        from cca_zoo.nonparametric import KCCA
        from cca_zoo.linear import CCA,rCCA

        #### load feature spaces
        ######## and do dimensionality reduction for any multidimensional feature spaces

        loaded_features = {}
        dimensions = [] 
        for feature_space in feature_names:
            # if(feature_space.split('_')[0] != 'run'):
            filepath = self.dir + '/features/'+self.features_dict[feature_space].lower()+'.csv'
            data = np.array(pd.read_csv(filepath,header=None))
            n_samples, n_features = data.shape
            if n_features > 7000: #Sparse random project the multi-dimensional feautures larger than 6000 dimensions
                self.initialize_sparse_random_projection(feature_space,n_samples,n_features,device='cpu') #automatically gets components
                data = self.sparse_random_projection(data,self.srp_matrices[feature_space],device='cpu') 
                # data = data
                n_samples, n_features = data.shape
                try:
                    print('checking if matrix is positive definite')
                    np.linalg.cholesky(data)
                    print('positive definite')
                except:
                    print('not positive definite')
            loaded_features[feature_space] = data.astype(dtype="float32")
            dimensions.append(n_features)
        reg_params = np.logspace(-5,0,5) #np.logspace(-10,20,50)
        print(reg_params)
        # param_grid = {"kernel": ["linear"], "c": [reg_params, reg_params]}
        param_grid = {"c": [reg_params, reg_params]}
        n_splits_outer = outer_folds
        n_splits_inner = inner_folds
        
        
        correlations_train = []
        correlations_test = []
        cv_outer = GroupKFold(n_splits=n_splits_outer)
        n_chunks = int(n_samples/self.chunklen)
        #set groups so that it chunks the data according to chunk len, and then the chunks are split into training and test
        groups = [str(ind) for ind in range(0,n_chunks) for x in range(0,self.chunklen)]
        if(len(groups)!=n_samples): #add final group that didn't divide evenly into chunklen if necessary
            print('adding outer stragglers')
            diff = n_samples-len(groups)
            groups.extend([str(n_chunks) for x in range(0,diff)]) 
        splits = cv_outer.split(X=range(0,n_samples),groups=groups)
        for i, (train_outer, test_outer) in enumerate(splits):
            data1 = loaded_features[feature_names[0]]
            data2 = loaded_features[feature_names[1]]

            scaler_X = StandardScaler()
            scaler_Y = StandardScaler()

            train1 = data1[train_outer].astype(dtype="float32")
            train2 = data2[train_outer].astype(dtype="float32")
            train1 = scaler_X.fit_transform(train1)
            train2 = scaler_Y.fit_transform(train2)
            train1 = np.nan_to_num(train1)
            train2 = np.nan_to_num(train2)

            test1 = data1[test_outer].astype(dtype="float32")
            test2 = data2[test_outer].astype(dtype="float32")
            test1 = scaler_X.transform(test1)
            test2 = scaler_Y.transform(test2)
            test1 = np.nan_to_num(test1)
            test2 = np.nan_to_num(test2)


            if(regularized):
                #do temporal chunking for the inner loop as well
                cv_inner = GroupKFold(n_splits=n_splits_inner)
                n_chunks = int(n_samples/self.chunklen)
                n_samples = train1.shape[0]
                n_chunks = int(n_samples/self.chunklen)
                groups = [str(ind) for ind in range(0,n_chunks) for x in range(0,self.chunklen)]
                if(len(groups)!=n_samples): #add final group that didn't divide evenly into chunklen if necessary
                    diff = n_samples-len(groups)
                    groups.extend([str(n_chunks) for x in range(0,diff)])
                inner_splits = cv_inner.split(X=range(0,n_samples),groups=groups)
                # for split in inner_splits:
                #     dimensions.append(len(split[0]))
                # print(dimensions)
                # if(latent_dimensions=='auto'):
                #     latent_dimensions = np.min(dimensions) #take the maximum number of latent dimensions, which is the dimensions of the smallest feature space

                rCCA_model = rCCA(latent_dimensions=1)
                # Tuning hyperparameters using GridSearchCV for the linear kernel.
                model = GridSearchCV(rCCA_model,param_grid=param_grid,cv=cv_inner,verbose=4, error_score='raise').fit((train1,train2),groups=groups)
            else:
                dimensions.append(train1.shape[0])
                dimensions.append(train2.shape[0])
                # print(dimensions)
                # if(latent_dimensions=='auto'):
                #     latent_dimensions = np.min(dimensions) #take the maximum number of latent dimensions, which is the dimensions of the smallest feature space

                model = CCA(latent_dimensions=1).fit((train1,train2))
            # model.fit_transform((train1, train2))
            # print(rCCA_model.explained_variance((test1,test2)))
            # correlations_train.append(model.score((train1, train2)))
            correlations_test.append(model.score((test1, test2)))
            # correlations_test.append(np.sum(rCCA_model.explained_variance((test1,test2))))
        # correlation_train = np.mean(correlations_train)
        correlation_test = np.mean(correlations_test) #get the mean over the folds

        return correlation_test

    def ridge_regression(self,Xfeature,Yfeature,outer_folds=10,inner_folds=5,num_alphas=10,backend='torch_cuda',permutations=None):
        """
            This function performs the fitting and testing of the encoding model. Saves results in object variables.

            Parameters
            ----------
            outer_folds : int specifying how many cross validation folds for the outer loop of nested cross validation
            inner_folds: int specifying how many cross validation folds for the inner loop of nested cross validation
            num_alphas: int specifing how many alphas to try in the inner loop
            backend : str, backend for himalaya. could be 'torch_cuda' for use on GPU, or 'numpy' for CPU
            permutations: int, how many permutations to generate for the null distributions
        """
        
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from voxelwise_tutorials.delayer import Delayer
        from himalaya.backend import set_backend
        from himalaya.kernel_ridge import MultipleKernelRidgeCV,KernelRidgeCV
        from himalaya.ridge import RidgeCV
        from himalaya.kernel_ridge import Kernelizer
        from himalaya.kernel_ridge import ColumnKernelizer
        from himalaya.scoring import r2_score_split
        from himalaya.scoring import correlation_score_split


        backend_name = backend
        backend = set_backend(backend, on_error="raise")
        print(backend)

        solver = "svd"#"random_search"
        solver_function = RidgeCV.ALL_SOLVERS[solver]

        n_iter = 1000#self.random_search_n 

        alphas = np.logspace(-10,20,50)#10**-5, 10**20, 30) # 30 logspaced values ranging from 10^-5 to 10^20, from https://gallantlab.org/papers/Deniz.F.etal.2023.pdf
        #TODO? try logspace -5, 20
        #or no.logspace -8, 10, 100 -> this is what Wang et al 2023 did
        # print(alphas)
        # alphas = np.logspace(-10,10,21)
        # alphas = np.logspace(1,20,20)
        # n_targets_batch = 200
        # n_alphas_batch = 5
        n_targets_batch = 10000 #the number of targets to compute at one time
        n_alphas_batch = 30 
        n_targets_batch_refit = 200

        solver_params = dict(#n_iter=n_iter, #alphas=alphas,
                             n_targets_batch=n_targets_batch,
                             n_alphas_batch=n_alphas_batch,
                             n_targets_batch_refit=n_targets_batch_refit)
                             # local_alpha=False)#share alphas for all targets (here: dimensions of a feature space), because in the model to brain model, one alpha is used for the entire feature space when fitting one voxel) -- makes this just ridge regression?

        # n_delays = 6 #the number of time delays (in TRs) to use in the encoding model (widens the model)
        # delayer = Delayer(delays=[x for x in np.arange(n_delays)])#delays of 1.5-9 seconds
        preprocess_pipeline = make_pipeline(
            #delayer,  #no delayer bc not predicting fMRI data
            Kernelizer(kernel="linear"),
        )
        loaded_features = {}
        dimensions = [] 
        for feature_space in [Yfeature,Xfeature]:
            # if(feature_space.split('_')[0] != 'run'):
            filepath = self.dir + '/features/'+self.features_dict[feature_space].lower()+'.csv'
            data = np.array(pd.read_csv(filepath,header=None))
            n_samples, n_features = data.shape
            if n_features > 7000: #Sparse random project the multi-dimensional features larger than 6000 dimensions
                self.initialize_sparse_random_projection(feature_space,n_samples,n_features,device='cuda') #automatically gets components
                data = self.sparse_random_projection(data,self.srp_matrices[feature_space],device='cuda') 
                # data = data
                
            loaded_features[feature_space] = data.astype(dtype="float32")

        #data split
        n_splits_outer = outer_folds
        n_splits_inner = inner_folds
        n_samples = 1921 #should be 1921
        weight_estimates_list = []
        performance_list = []
        individual_feature_performance_list = []
        individual_product_measure_list = []

        permuted_scores_list = []
        permuted_ind_perf_scores_list = []
        permuted_ind_product_scores_list = []
        best_alphas = []
        cv_scores = []
        
        #outer loop - 10 fold, 9 folds to get weight estimates and hyperparameters, 1 for evaluating the performance of the model, averaged across 10
        cv_outer = GroupKFold(n_splits=n_splits_outer)
        n_chunks = int(n_samples/self.chunklen)
        #set groups so that it chunks the data according to chunk len, and then the chunks are split into training and test
        groups = [str(ind) for ind in range(0,n_chunks) for x in range(0,self.chunklen)]
        if(len(groups)!=n_samples): #add final group that didn't divide evenly into chunklen if necessary
            print('adding outer stragglers')
            diff = n_samples-len(groups)
            groups.extend([str(n_chunks) for x in range(0,diff)]) 
        splits = cv_outer.split(X=range(0,n_samples),groups=groups)
        for i, (train_outer, test_outer) in enumerate(splits):
            print('starting cross-validation fold '+str(i+1) +'/'+str(n_splits_outer))
            # print('train_indices')
            # print(train_outer)
            # print('test_indices')
            # print(test_outer)
            data = loaded_features[Yfeature]
            scaler = StandardScaler(with_mean=True, with_std=True)
            Y_train = scaler.fit_transform(data[train_outer].astype(dtype="float32"))
            # Y_train = np.nan_to_num(Y_train)
            Y_test = scaler.transform(data[test_outer].astype(dtype="float32"))
            # Y_test = np.nan_to_num(Y_test)
            
            data = loaded_features[Xfeature]
            scaler=StandardScaler(with_mean=True, with_std=True)
            X_train = scaler.fit_transform(data[train_outer].astype(dtype="float32"))
            # X_train = np.nan_to_num(X_train)
            X_test = scaler.transform(data[test_outer].astype(dtype="float32"))
            # X_test = np.nan_to_num(X_test)
            
            feature_names = [Xfeature]
            features_n_list = [data[train_outer].shape[1]]

            n_samples = X_train.shape[0]

            print("(n_samples_train, n_features_total) =", X_train.shape)
            print("(n_samples_test, n_features_total) =", X_test.shape)
            print("[features_n,...] =", features_n_list)

            # start_and_end = np.concatenate([[0], np.cumsum(features_n_list)])
            # slices = [
            #     slice(start, end)
            #     for start, end in zip(start_and_end[:-1], start_and_end[1:])
            # ]
            # print(slices)

            # kernelizers_tuples = [(name, preprocess_pipeline, slice_)
            #                       for name, slice_ in zip(feature_names, slices)]
            # column_kernelizer = ColumnKernelizer(kernelizers_tuples)

            #do temporal chunking for the inner loop as well
            cv_inner = GroupKFold(n_splits=n_splits_inner)
            n_chunks = int(n_samples/self.chunklen)
            
            groups = [str(ind) for ind in range(0,n_chunks) for x in range(0,self.chunklen)]
            if(len(groups)!=n_samples): #add final group that didn't divide evenly into chunklen if necessary
                diff = n_samples-len(groups)
                groups.extend([str(n_chunks) for x in range(0,diff)])
            inner_splits = cv_inner.split(X=range(0,n_samples),groups=groups)

            mkr_model = RidgeCV(alphas=alphas, solver='svd', solver_params=solver_params, cv=inner_splits)

            pipeline = make_pipeline(
                # column_kernelizer,
                mkr_model,
            )

            pipeline.fit(X_train, Y_train)
            print(pipeline)
            scores_mask = pipeline.score(X_test, Y_test) #
            scores_mask = backend.to_numpy(scores_mask)
            print("(n_voxels_mask,) =", scores_mask.shape)
            print(scores_mask[scores_mask>0])
            print(len(scores_mask[scores_mask>0]))
            print('scores less than -1')
            print(len(scores_mask[scores_mask<-1]))
            print('avg performance:' +str(np.nanmean(scores_mask)))
            performance_list.append(scores_mask)
            best_alphas.append(mkr_model.best_alphas_.cpu().numpy())
            cv_scores.append(mkr_model.cv_scores_.cpu().numpy())

            # disentangle the contribution of the two feature spaces -- individual feature space performances

            permuted_scores = []
            permuted_ind_product_scores = []
            permuted_ind_perf_scores = []
            if(permutations!=None):
                for iteration in np.arange(0,permutations):
                    # shuffle BOLD time series in blocks of 10 and then correlate with the predicted time-series
                    # Block permutation preserves autocorrelation statistics of the time series (Kunsch, 1989) 
                    # and thus provides a sensible null hypothesis for these significance tests

                    #shuffle the BOLD time series in chunks to account for temporal autocorrelation
                    #similar to how they did it here: https://gallantlab.org/papers/Deniz.F.etal.2023.pdf
                    np.random.shuffle(Y_test.reshape(-1,chunklen,Y_test.shape[1]))
                    scores_mask = pipeline.score(X_test, Y_test)
                    scores_mask = backend.to_numpy(scores_mask)
                    permuted_scores.append(scores_mask)

                permuted_scores_list.append(permuted_scores)

            # primal_coefs = mkr_model.get_primal_coef(column_kernelizer.get_X_fit())

            # average_coefs =[]
            # for curr_coefs in primal_coefs:
            #     primal_coef_per_delay = torch.stack(np.array_split(curr_coefs, n_delays, axis=0))
            #     # print(primal_coef_per_delay.size())
            #     average_coef = np.mean(np.array(primal_coef_per_delay), axis=0)
            #     # print(average_coef.shape)
            #     average_coefs.append(average_coef)
            # average_coefs = np.concatenate(average_coefs) #concatenate all weights, now averaged over the time delays
            
            # weight_estimates_list.append(average_coefs)
            
            debugging=True
            if(debugging):
                import matplotlib.pyplot as plt
                from himalaya.viz import plot_alphas_diagnostic
                
                for best_alphas_ in best_alphas:
                    plot_alphas_diagnostic(best_alphas_, alphas)
                plt.title("Best alphas selected by cross-validation")
                plt.savefig('quality_check/'+self.features+'_best_alphas.png')
                plt.close()
                
                # for cv_scores_ in cv_scores:
                #     current_max = np.maximum.accumulate(cv_scores_, axis=0)
                #     mean_current_max = np.mean(current_max, axis=1)
                #     x_array = np.arange(1, len(mean_current_max) + 1)
                #     plt.plot(x_array, mean_current_max, '-o')
                # plt.grid("on")
                # plt.xlabel("Number of kernel weights sampled")
                # plt.ylabel("L2 negative loss (higher is better)")
                # plt.title("Convergence curve, averaged over targets")
                # plt.tight_layout()
                # plt.savefig('quality_check/'+self.features+'_cv_scores.png')
                # plt.close()
                

        print('finished with outer cross-validation')
        # print('weights')
        # weight_estimates = np.array(weight_estimates_list)  
        # print(weight_estimates.shape) 
        # average_weights = np.mean(weight_estimates,axis=0)
        # print(average_weights.shape)

        print('all features performance')
        performance = np.array(performance_list)    
        # self.performance_outer_folds = performance
        print(performance.shape) #should be 10, # n voxels
        average_performance = np.mean(performance)
        print(average_performance.shape)

        if(permutations!=None):
            print('all features performance null')
            performance_null = np.array(permuted_scores_list)    
            # self.performance_outer_folds = performance
            print(performance_null.shape) #should be 10, # n voxels
            average_performance_null = np.mean(performance_null,axis=(0,2))
            print(average_performance_null.shape)

            print(average_performance)
            print(average_performance_null)

            #do permutation statistics
            
            null_n = permutations
            null_n_over_sample = sum((average_performance_null>average_performance).astype(int))
            p = null_n_over_sample/null_n
        else:
            p = np.nan
        
        return average_performance, p

    def back_to_back_ridge_regression(self,Xfeature,Yfeature,outer_folds=10,inner_folds=5,num_alphas=10,backend='torch_cuda',permutations=None):
        """
            This function performs the fitting and testing of the encoding model. Saves results in object variables.

            Parameters
            ----------
            outer_folds : int specifying how many cross validation folds for the outer loop of nested cross validation
            inner_folds: int specifying how many cross validation folds for the inner loop of nested cross validation
            num_alphas: int specifing how many alphas to try in the inner loop
            backend : str, backend for himalaya. could be 'torch_cuda' for use on GPU, or 'numpy' for CPU
            permutations: int, how many permutations to generate for the null distributions
        """
        
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from voxelwise_tutorials.delayer import Delayer
        from himalaya.backend import set_backend
        from himalaya.kernel_ridge import MultipleKernelRidgeCV,KernelRidgeCV
        from himalaya.ridge import RidgeCV
        from himalaya.kernel_ridge import Kernelizer
        from himalaya.kernel_ridge import ColumnKernelizer
        from himalaya.scoring import r2_score_split
        from himalaya.scoring import correlation_score_split


        backend_name = backend
        backend = set_backend(backend, on_error="raise")
        print(backend)

        solver = "svd"#"random_search"
        solver_function = RidgeCV.ALL_SOLVERS[solver]

        n_iter = 1000#self.random_search_n 

        alphas = np.logspace(-10,30,50)#10**-5, 10**20, 30) # 30 logspaced values ranging from 10^-5 to 10^20, from https://gallantlab.org/papers/Deniz.F.etal.2023.pdf
        #TODO? try logspace -5, 20
        #or no.logspace -8, 10, 100 -> this is what Wang et al 2023 did
        # print(alphas)
        # alphas = np.logspace(-10,10,21)
        # alphas = np.logspace(1,20,20)
        # n_targets_batch = 200
        # n_alphas_batch = 5
        n_targets_batch = 10000 #the number of targets to compute at one time
        n_alphas_batch = 30 
        n_targets_batch_refit = 200

        solver_params = dict(#n_iter=n_iter, alphas=alphas,
                             n_targets_batch=n_targets_batch,
                             n_alphas_batch=n_alphas_batch,
                             n_targets_batch_refit=n_targets_batch_refit)
                             # local_alpha=False)#share alphas for all targets (here: dimensions of a feature space), because in the model to brain model, one alpha is used for the entire feature space when fitting one voxel) -- makes this just ridge regression?

        # n_delays = 6 #the number of time delays (in TRs) to use in the encoding model (widens the model)
        # delayer = Delayer(delays=[x for x in np.arange(n_delays)])#delays of 1.5-9 seconds
        preprocess_pipeline = make_pipeline(
            #delayer,  #no delayer bc not predicting fMRI data
            Kernelizer(kernel="linear"),
        )
        loaded_features = {}
        dimensions = [] 
        for feature_space in [Yfeature,Xfeature]:
            # if(feature_space.split('_')[0] != 'run'):
            filepath = self.dir + '/features/'+self.features_dict[feature_space].lower()+'.csv'
            data = np.array(pd.read_csv(filepath,header=None))
            n_samples, n_features = data.shape
            if n_features > 7000: #Sparse random project the multi-dimensional feautures larger than 6000 dimensions
                self.initialize_sparse_random_projection(feature_space,n_samples,n_features,device='cpu') #automatically gets components
                data = self.sparse_random_projection(data,self.srp_matrices[feature_space],device='cpu') 
                # data = data
                
            loaded_features[feature_space] = data.astype(dtype="float32")

        #data split
        n_splits_outer = outer_folds
        n_splits_inner = inner_folds
        n_samples = 1921 #should be 1921
        weight_estimates_list = []
        performance_list = []
        individual_feature_performance_list = []
        individual_product_measure_list = []

        permuted_scores_list = []
        permuted_ind_perf_scores_list = []
        permuted_ind_product_scores_list = []
        best_alphas = []
        cv_scores = []
        
        #outer loop - 10 fold, 9 folds to get weight estimates and hyperparameters, 1 for evaluating the performance of the model, averaged across 10
        cv_outer = GroupKFold(n_splits=n_splits_outer)
        n_chunks = int(n_samples/self.chunklen)
        #set groups so that it chunks the data according to chunk len, and then the chunks are split into training and test
        groups = [str(ind) for ind in range(0,n_chunks) for x in range(0,self.chunklen)]
        if(len(groups)!=n_samples): #add final group that didn't divide evenly into chunklen if necessary
            print('adding outer stragglers')
            diff = n_samples-len(groups)
            groups.extend([str(n_chunks) for x in range(0,diff)]) 
        splits = cv_outer.split(X=range(0,n_samples),groups=groups)
        for i, (train_outer, test_outer) in enumerate(splits):
            print('starting cross-validation fold '+str(i+1) +'/'+str(n_splits_outer))
            # print('train_indices')
            # print(train_outer)
            # print('test_indices')
            # print(test_outer)
            data = loaded_features[Yfeature]
            scaler = StandardScaler(with_mean=True, with_std=True)
            Y_train = scaler.fit_transform(data[train_outer].astype(dtype="float32"))
            Y_train = np.nan_to_num(Y_train)
            Y_test = scaler.transform(data[test_outer].astype(dtype="float32"))
            Y_test = np.nan_to_num(Y_test)
            
            data = loaded_features[Xfeature]
            scaler=StandardScaler(with_mean=True, with_std=True)
            X_train = scaler.fit_transform(data[train_outer].astype(dtype="float32"))
            X_train = np.nan_to_num(X_train)
            X_test = scaler.transform(data[test_outer].astype(dtype="float32"))
            X_test = np.nan_to_num(X_test)

            #split training data in half for each of the regressions, in chunks still!
            n_samples = X_train.shape[0]

            cv_half = GroupKFold(n_splits=2)
            n_chunks = int(n_samples/self.chunklen)

            groups = [str(ind) for ind in range(0,n_chunks) for x in range(0,self.chunklen)]
            if(len(groups)!=n_samples): #add final group that didn't divide evenly into chunklen if necessary
                diff = n_samples-len(groups)
                groups.extend([str(n_chunks) for x in range(0,diff)])
            half_splits = cv_half.split(X=range(0,n_samples),groups=groups)

            half_split = next(half_splits) #get only one of the splits
            feat1_train = Y_train
            feat2_train = X_train

            feat1_train_half1 = Y_train[half_split[0]] #first half
            feat1_train_half2 = Y_train[half_split[1]] #second half
            
            feat2_train_half1 = X_train[half_split[0]]
            feat2_train_half2 = X_train[half_split[1]]

            feature_names = [Xfeature]
            features_n_list = [data[train_outer].shape[1]]

            n_samples = feat1_train_half1.shape[0]

            #do temporal chunking for the inner loop as well
            cv_inner = GroupKFold(n_splits=n_splits_inner)
            n_chunks = int(n_samples/self.chunklen)
            
            groups = [str(ind) for ind in range(0,n_chunks) for x in range(0,self.chunklen)]
            if(len(groups)!=n_samples): #add final group that didn't divide evenly into chunklen if necessary
                diff = n_samples-len(groups)
                groups.extend([str(n_chunks) for x in range(0,diff)])
            inner_splits = cv_inner.split(X=range(0,n_samples),groups=groups)

            first_model = RidgeCV(alphas=alphas, solver='svd', solver_params=solver_params, cv=inner_splits)

            pipeline1 = make_pipeline(
                # column_kernelizer,
                first_model,
            )

            # pipeline.fit(feat1_train_half1, feat2_train_half1) 
            pipeline1.fit(feat1_train, feat2_train) 
            print('pipeline1 score')
            scores_mask = pipeline1.score(X_test, Y_test) #get the score on the test sets
            scores_mask = backend.to_numpy(scores_mask)
            print("(n_voxels_mask,) =", scores_mask.shape)
            print(scores_mask)
            print('avg performance:' +str(np.nanmean(scores_mask)))
            #get the predicted Y_train2
            # predicted_feat2_train_half2 = first_model.predict(feat1_train_half2)
            predicted_feat2_train = first_model.predict(feat1_train)
            # print(predicted_feat2_train_half2)

            #predict with the other half of Y_train
            second_model = RidgeCV(alphas=alphas, solver='svd', solver_params=solver_params, cv=inner_splits)

            pipeline2 = make_pipeline(
                # column_kernelizer,
                second_model,
            )
            # pipeline.fit(feat2_train_half2, predicted_feat2_train_half2) #predict the predicted time series
            pipeline2.fit(feat2_train, predicted_feat2_train) #predict the predicted time series
            print(pipeline2)
            scores_mask = pipeline2.score(X_test, Y_test) #get the score on the test sets
            scores_mask = backend.to_numpy(scores_mask)
            print("(n_voxels_mask,) =", scores_mask.shape)
            print(scores_mask)
            print('avg performance:' +str(np.nanmean(scores_mask)))
            performance_list.append(scores_mask)
            best_alphas.append(mkr_model.best_alphas_.cpu().numpy())
            cv_scores.append(mkr_model.cv_scores_.cpu().numpy())

            # disentangle the contribution of the two feature spaces -- individual feature space performances

            permuted_scores = []
            permuted_ind_product_scores = []
            permuted_ind_perf_scores = []
            if(permutations!=None):
                for iteration in np.arange(0,permutations):
                    # shuffle BOLD time series in blocks of 10 and then correlate with the predicted time-series
                    # Block permutation preserves autocorrelation statistics of the time series (Kunsch, 1989) 
                    # and thus provides a sensible null hypothesis for these significance tests

                    #shuffle the BOLD time series in chunks to account for temporal autocorrelation
                    #similar to how they did it here: https://gallantlab.org/papers/Deniz.F.etal.2023.pdf
                    np.random.shuffle(Y_test.reshape(-1,chunklen,Y_test.shape[1]))
                    scores_mask = pipeline.score(X_test, Y_test)
                    scores_mask = backend.to_numpy(scores_mask)
                    permuted_scores.append(scores_mask)

                permuted_scores_list.append(permuted_scores)

            # primal_coefs = mkr_model.get_primal_coef(column_kernelizer.get_X_fit())

            # average_coefs =[]
            # for curr_coefs in primal_coefs:
            #     primal_coef_per_delay = torch.stack(np.array_split(curr_coefs, n_delays, axis=0))
            #     # print(primal_coef_per_delay.size())
            #     average_coef = np.mean(np.array(primal_coef_per_delay), axis=0)
            #     # print(average_coef.shape)
            #     average_coefs.append(average_coef)
            # average_coefs = np.concatenate(average_coefs) #concatenate all weights, now averaged over the time delays
            
            # weight_estimates_list.append(average_coefs)
            
            debugging=True
            if(debugging):
                import matplotlib.pyplot as plt
                from himalaya.viz import plot_alphas_diagnostic
                
                for best_alphas_ in best_alphas:
                    plot_alphas_diagnostic(best_alphas_, alphas)
                plt.title("Best alphas selected by cross-validation")
                plt.savefig('quality_check/'+self.features+'_best_alphas.png')
                plt.close()
                
                # for cv_scores_ in cv_scores:
                #     current_max = np.maximum.accumulate(cv_scores_, axis=0)
                #     mean_current_max = np.mean(current_max, axis=1)
                #     x_array = np.arange(1, len(mean_current_max) + 1)
                #     plt.plot(x_array, mean_current_max, '-o')
                # plt.grid("on")
                # plt.xlabel("Number of kernel weights sampled")
                # plt.ylabel("L2 negative loss (higher is better)")
                # plt.title("Convergence curve, averaged over targets")
                # plt.tight_layout()
                # plt.savefig('quality_check/'+self.features+'_cv_scores.png')
                # plt.close()
                

        print('finished with outer cross-validation')
        # print('weights')
        # weight_estimates = np.array(weight_estimates_list)  
        # print(weight_estimates.shape) 
        # average_weights = np.mean(weight_estimates,axis=0)
        # print(average_weights.shape)

        print('all features performance')
        performance = np.array(performance_list)    
        # self.performance_outer_folds = performance
        print(performance.shape) #should be 10, # n voxels
        average_performance = np.mean(performance)
        print(average_performance.shape)

        if(permutations!=None):
            print('all features performance null')
            performance_null = np.array(permuted_scores_list)    
            # self.performance_outer_folds = performance
            print(performance_null.shape) #should be 10, # n voxels
            average_performance_null = np.mean(performance_null,axis=(0,2))
            print(average_performance_null.shape)

            print(average_performance)
            print(average_performance_null)

            #do permutation statistics
            
            null_n = permutations
            null_n_over_sample = sum((average_performance_null>average_performance).astype(int))
            p = null_n_over_sample/null_n
        else:
            p = np.nan
        
        return average_performance, p
    def run(self):

        # results = {}

        # featuresY = ['social','speaking']#'sbert_layer12','alexnet_layer7']#,'num_agents','speaking','turn_taking','mentalization','word2vec','sbert','valence','arousal','motion','face','indoor_outdoor','written_text','pixel','hue','pitch','amplitude','music','alexnet']
        # featuresX = featuresY#['social','num_agents','speaking','turn_taking','mentalization','word2vec','sbert','valence','arousal','motion','face','indoor_outdoor','written_text','pixel','hue','pitch','amplitude','music','alexnet']
        # results_array = np.zeros((len(featuresY),len(featuresX)))
        # results_p_array = np.zeros((len(featuresY),len(featuresX)))
        # for (ind1,featureY) in enumerate(featuresY):
        #     for (ind2,featureX) in enumerate(featuresX):
        #         if(featureY!=featureX):
        #             # result = self.banded_ridge_regression(featureY,featureX,backend='torch_cuda',permutations=None,outer_folds=10,chunklen=20)
        #             result = self.canonical_correlation_analysis(feature_names = [featureY,featureX],outer_folds=10)
        #             results[featureX+'-'+featureY] = result
        #             results_array[ind1][ind2] = result[0]
        #             results_p_array[ind1][ind2]=result[1]

        # df = pd.DataFrame(results)
        # df.to_csv(self.out_dir+'/feature_space_regression_results.csv')

        # np.savetxt(self.out_dir+'/feature_space_regression_results_array.csv', results_array, delimiter=",")

        # import matplotlib.pyplot as plt
        # import seaborn as sns

        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # cax = ax.imshow(results_array,cmap='Greens',vmin=0.0,vmax=1)
        # sns.despine(left=True,bottom=True)
        # fig.colorbar(cax)

        # sns.despine(left=True,bottom=True)
        # fig.colorbar(cax)
        # plt.title('')
        # ax.set_xticks(range(0,len(featuresY)))
        # ax.set_yticks(range(0,len(featuresX)))

        # for i in range(len(featuresY)):
        #     for j in range((len(featuresX))):
        #         c = results_array[j,i]
        #         pvalue = results_p_array[j,i]
        #         if(~np.isnan(c)):
        #             if(pvalue<0.001):
        #                 ax.text(i, j, str(np.round(c,2)), va='center', ha='center',fontweight='heavy')
        #             elif(pvalue<0.05):
        #                 ax.text(i, j, str(np.round(c,2)), va='center', ha='center')
        #             else:
        #                 ax.text(i, j, str(np.round(c,2)), va='center', ha='center',color='gray')

        # ax.set_xticklabels([name for name in featuresY])
        # ax.set_yticklabels([name for name in featuresX])
        # plt.xticks(rotation=90)
        # plt.savefig(self.out_dir+'/feature_space_regression.png',bbox_inches='tight')
        # plt.close()
        print(self.features)
        if(self.method=='CCA'):
            result = self.canonical_correlation_analysis(feature_names = [self.feature1,self.feature2],regularized=True,outer_folds=10)
        elif(self.method=='regression'):
            result = self.ridge_regression(self.feature1,self.feature2,backend='torch_cuda',permutations=None,outer_folds=10)
        elif(self.method=='b2b_regression'):
            result = self.back_to_back_ridge_regression(self.feature1,self.feature2,backend='torch_cuda',permutations=None,outer_folds=10)
        #save the result in a csv
        filepath = self.out_dir + '/'+self.features+'.csv'
        with open(filepath, 'w') as file:
            file.write(str(result))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--chunklen','-chunklen',type=int,default=20)
    parser.add_argument('--features','-features',type=str,default='-') #formatted with features separated by '-' ex) 'social-speaking'
    parser.add_argument('--method','-method',type=str,default='regression') #CCA

    parser.add_argument('--dir', '-dir', type=str,
                        default='/Users/hsmall2/Documents/GitHub/deep_nat_lat')
    parser.add_argument('--out_dir', '-output', type=str,
                        default='/Users/hsmall2/Documents/GitHub/deep_nat_lat/analysis')
    parser.add_argument('--figure_dir', '-figures', type=str,
                        default='/Users/hsmall2/Documents/GitHub/deep_nat_lat/figures')
    args = parser.parse_args()
    FeatureSpaceCorrelation(args).run()

if __name__ == '__main__':
    main()

