import ast
import pickle
import random
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
from collections import Counter
from matplotlib import pyplot as plt

from sklearn.svm import SVC
from scipy.stats import entropy
from sklearn.cluster import KMeans
from collections import defaultdict
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import pairwise_distances

from sklearn.metrics import accuracy_score, f1_score

from Preprocess.data_augmentation import dropout_augmentation

import warnings
warnings.filterwarnings("ignore")


class ActiveLearningPipeline:
    """
    A class to implement an Active Learning Pipeline for iterative model training and sample selection.

    Attributes:
    -----------
    df : The DataFrame containing the dataset.
    feature_vectors : A NumPy array of feature vectors derived from the specified feature column.
    augmentation_features_name : List of feature names to be used for augmentation in training.
    labels : A NumPy array of labels corresponding to the samples in the DataFrame.
    train_indices : Indices of the samples that are initially part of the training set
    available_pool_indices : Indices of the samples that are available for selection in the active learning pool.
    test_features : Features of the test set.
    test_labels : Labels of the test set.
    iterations :Number of iterations to perform in the active learning loop.
    budget_per_iter : The maximum number of samples to select during each iteration.
    model : The machine learning model to be trained.
    selection_criterion : The criterion used to select samples from the available pool.
    method : The sampling method corresponding to the selection criterion.
    seed : Random seed for reproducibility.
    """
    def __init__(self, model,
                 available_pool_indices,
                 train_indices,
                 test_features,
                 test_labels,
                 selection_criterion,
                 iterations,
                 budget_per_iter,
                 df,
                 feature,
                 augmentation_features_name,
                 seed):
        self.df = df
        self.feature_vectors = np.array(df[feature].tolist())
        self.augmentation_features_name = augmentation_features_name
        self.labels = np.array(df['y'].tolist())

        self.train_indices = np.array(train_indices)
        self.available_pool_indices = np.array(available_pool_indices)
        
        self.test_features = np.array(test_features.to_list())
        self.test_labels = np.array(test_labels.to_list())
        
        self.iterations = iterations
        self.budget_per_iter = budget_per_iter

        self.model = model
        self.selection_criterion = selection_criterion  
        self.method = self._define_method()
        
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)
        
        
    def _define_method(self):
        """
        defines the sampling method based on the provided selection criterion
        """
        if self.selection_criterion == 'random':
            method = self._random_sampling
        elif self.selection_criterion == 'uncertainty':
            method = self._uncertainty_sampling
        elif self.selection_criterion == 'diversity':
            method = self._diversity_sampling
        elif self.selection_criterion == 'sentiment':
            method = self._sentiment_sampling
        elif self.selection_criterion == 'longest':
            method = self._largest_sampling
        elif self.selection_criterion == 'shortest':
            method = self._shortest_sampling
        elif self.selection_criterion == 'cluster_uncertainty':
            method = self._clustering_uncertainty
        elif self.selection_criterion == 'cluster_disagreement':
            method = self._clustering_disagreement
        elif self.selection_criterion == 'unbalanced':
            method = self._unbalanced_sampling
        elif self.selection_criterion == 'mistake':
            method = self._mistake_sampling
        elif self.selection_criterion == 'query_by_commitee':
            method = self.query_by_committee
        else:
            raise ValueError("Invalid selection cretarion")
        return method

    def init_model(self):
        """
        initializes the machine learning model based on the provided model name
        """
        if self.model == 'RandomForestClassifier':
            self.model = RandomForestClassifier(random_state=self.seed)
        elif self.model == 'SVC':
            self.model = SVC(probability=True)
        elif self.model == 'LogisticRegression':
            self.model = LogisticRegression()
    
    def train_model(self):
        """
        trains the model using the selected features and labels, with optional augmentation.
        """
        self.init_model()
        train_features = self.feature_vectors[self.train_indices]
        train_labels = self.labels[self.train_indices]
        for augmentation_feature_name in self.augmentation_features_name:
            additional_features = np.array(self.df[augmentation_feature_name][self.train_indices].to_list())
            additional_labels = np.array(self.df['y'][self.train_indices].to_list())
            
            train_features = np.concatenate([train_features, additional_features])
            train_labels = np.concatenate([train_labels, additional_labels])
        self.model.fit(train_features, train_labels)
    
    def _random_sampling(self):
        """
        selects a random subset of indices from the available pool for sampling
        """
        n_select = min(self.budget_per_iter, len(self.available_pool_indices))
        sampled_indices = np.random.choice(self.available_pool_indices, size=n_select, replace=False)
        self.available_pool_indices = np.setdiff1d(self.available_pool_indices, sampled_indices)
        return np.array(sampled_indices)
    
    def _uncertainty_sampling(self):
        """
        selects a subset of indices from the available pool using uncertainty sampling
        """
        n_select = min(self.budget_per_iter, len(self.available_pool_indices))
        available_pool_features = self.feature_vectors[self.available_pool_indices]
        probabilities = self.model.predict_proba(available_pool_features)
        # calculate uncertainty (entropy)
        uncertainties = entropy(probabilities.T)
        # select samples with highest uncertainty
        sampled_locations = np.argsort(uncertainties)[-n_select:]
        selected_indices = self.available_pool_indices[sampled_locations]
        self.available_pool_indices = np.setdiff1d(self.available_pool_indices, selected_indices)
        return selected_indices
    
    def _diversity_sampling(self):
        """
        selects a subset of indices from the available pool using diversity sampling
        """
        def top_k(arr, k):
            return np.argpartition(arr, -k)[-k:]
        n_select = min(self.budget_per_iter, len(self.available_pool_indices))
        distances = pairwise_distances(self.feature_vectors[self.available_pool_indices], self.feature_vectors[self.train_indices])
        min_distances = distances.min(axis=1)
        top_indices = top_k(min_distances, min(n_select, len(self.available_pool_indices)))     
        selected_indices = self.available_pool_indices[top_indices]
        self.available_pool_indices = np.setdiff1d(self.available_pool_indices, selected_indices)
        return selected_indices
    
    def _sentiment_sampling(self):
        """
        selects a subset of indices from the available pool based on sentiment certainty, 
        choosing the samples with the lowest certainty scores.
        """
        n_select = min(self.budget_per_iter, len(self.available_pool_indices))
        df_pool = self.df.loc[self.available_pool_indices]
        lowest_sentiment_indices = df_pool.nsmallest(n_select, 'sentiment_certainty').index
        selected_indices = np.intersect1d(self.available_pool_indices, lowest_sentiment_indices)
        self.available_pool_indices = np.setdiff1d(self.available_pool_indices, selected_indices)
        return selected_indices
    
    def _largest_sampling(self):
        """
        selects a subset of indices from the available pool by choosing samples with the largest number of words
        """
        n_select = min(self.budget_per_iter, len(self.available_pool_indices))
        df_pool = self.df.loc[self.available_pool_indices]
        max_length_indices = df_pool.nlargest(n_select, 'num_of_words').index
        selected_indices = np.intersect1d(self.available_pool_indices, max_length_indices)
        self.available_pool_indices = np.setdiff1d(self.available_pool_indices, selected_indices)
        return selected_indices
    
    def _shortest_sampling(self):
        """
        selects a subset of indices from the available pool by choosing samples with the smallest number of words
        """
        n_select = min(self.budget_per_iter, len(self.available_pool_indices))
        df_pool = self.df.loc[self.available_pool_indices]
        min_length_indices = df_pool.nsmallest(n_select, 'num_of_words').index
        selected_indices = np.intersect1d(self.available_pool_indices, min_length_indices)
        self.available_pool_indices = np.setdiff1d(self.available_pool_indices, selected_indices)
        return selected_indices
    
    def _clustering_uncertainty(self, k_means_n_clusters=5):
        """
        selects a subset of indices from the available pool based on uncertainty sampling within clusters, using k-means clustering. 
        it computes the average uncertainty for each cluster and then samples indices from clusters proportionally to their uncertainty.
        """
        n_select = min(self.budget_per_iter, len(self.available_pool_indices))
        available_pool_features = self.feature_vectors[self.available_pool_indices]
        kmeans = KMeans(n_clusters=k_means_n_clusters, random_state=self.seed)
        kmeans.fit(available_pool_features)
        
        # calculate uncertainty (entropy) for each cluster
        clusters_uncertainty_list = []
        for i in range(kmeans.n_clusters):
            cluster_indices = np.where(kmeans.labels_ == i)[0]
            cluster_points = available_pool_features[cluster_indices]
            
            probabilities = self.model.predict_proba(cluster_points)
            uncertainties = entropy(probabilities.T)
            clusters_uncertainty_list.append(np.mean(uncertainties))
            
        clusters_uncertainty_prob = np.array(clusters_uncertainty_list)
        clusters_uncertainty_prob /= sum(clusters_uncertainty_prob)
        
        # randomly sample points from clusters, with probability proportional to cluster uncertainty
        selected_indices_list = []
        n_selected = 0
        while n_selected < n_select:
            index = np.random.choice(len(clusters_uncertainty_prob), p=clusters_uncertainty_prob)
            cluster_indices = np.where(kmeans.labels_ == index)[0]
            cluster_indices = self.available_pool_indices[cluster_indices]
            select_from = set(cluster_indices) - set(selected_indices_list)
            if len(select_from) == 0:
                continue
            selected_index = random.choice(list(select_from))
            selected_indices_list.append(selected_index)
            n_selected += 1
            
        selected_indices = np.intersect1d(self.available_pool_indices, selected_indices_list)
        self.available_pool_indices = np.setdiff1d(self.available_pool_indices, selected_indices)
        
        return selected_indices
    
    
    def query_by_committee(self):
        """
        employs multiple models (Logistic Regression, SVM, and Random Forest) to predict labels for samples in the available pool. 
        the selected indices are those with the highest disagreement among the models, indicating uncertainty about their labels.
        """
        n_select = min(self.budget_per_iter, len(self.available_pool_indices))
        available_pool_features = self.feature_vectors[self.available_pool_indices]
        train_features = self.feature_vectors[self.train_indices]
        train_labels = self.labels[self.train_indices]
        
        # fit each model to the training data and get predictions on the available pool features
        lr_model = LogisticRegression()
        svm_model = SVC(probability=True)
        rf_model = RandomForestClassifier(random_state=self.seed)

        lr_model.fit(train_features, train_labels)
        svm_model.fit(train_features, train_labels)
        rf_model.fit(train_features, train_labels)

        lr_predict = lr_model.predict(available_pool_features)
        svm_predict = svm_model.predict(available_pool_features)
        rf_predict = rf_model.predict(available_pool_features)
        
        # select the top indices with the highest disagreement
        committee_predictions = np.stack([lr_predict, svm_predict, rf_predict], axis=1)
        disagreement_scores = np.apply_along_axis(
            lambda x: len(np.unique(x)), 1, committee_predictions
        )
        sorted_indices = np.argsort(-disagreement_scores)
        selected_indices = sorted_indices[:n_select]
        selected_pool_indices = self.available_pool_indices[selected_indices]

        selected_pool_indices = np.intersect1d(self.available_pool_indices, selected_pool_indices)
        self.available_pool_indices = np.setdiff1d(self.available_pool_indices, selected_pool_indices)
        
        return selected_pool_indices


    def _evaluate_model(self, trained_model):
        """
        evaluate the performance of the trained model on the test set
        """
        preds = trained_model.predict(self.test_features)
        true_label = self.test_labels
        accuracy = round(accuracy_score(true_label, preds), 3)
        f1 = round(f1_score(true_label, preds), 3)
        return accuracy, f1


    def run_pipeline(self):
        """
        execute the active learning pipeline over a specified number of iterations
        in each iteration, the model is trained on an expanding set of training indices, 
        and performance metrics (accuracy and F1 score) are evaluated
        """
        accuracy_scores = []
        f1_scores = []
        self.train_model()
        accuracy, f1 = self._evaluate_model(self.model)
        accuracy_scores.append(accuracy)
        f1_scores.append(f1)
        for iteration in range(self.iterations-1):
            self.train_indices = np.concatenate([self.train_indices, self.method()]) 
            self.train_model()
            
            accuracy, f1 = self._evaluate_model(self.model)
            accuracy_scores.append(accuracy)
            f1_scores.append(f1)
        return accuracy_scores, f1_scores
    

def choose_initial_train_indeces(df, feature_name, seed, init_train_method='random', random_train_size=20, k_means_n_clusters=20):
    """
    select initial training indices using the specified method- options are 'random' or 'kmeans'.
    """
    np.random.seed(seed)
    random.seed(seed)
    X = np.array(df[feature_name].to_list())
    train_indices = []
    if init_train_method == 'random':
        # randomly select training indices
        train_indices = np.random.choice(df.index, size=random_train_size, replace=False)
    elif init_train_method == 'kmeans':
        # apply KMeans clustering to select training indices based on cluster centroids
        kmeans = KMeans(n_clusters=k_means_n_clusters, random_state=seed)
        kmeans.fit(X)
        for i in range(kmeans.n_clusters):
            cluster_indices = np.where(kmeans.labels_ == i)[0]
            cluster_points = X[cluster_indices]
            distances = np.linalg.norm(cluster_points - kmeans.cluster_centers_[i], axis=1)
            closest_index_within_cluster = np.argmin(distances)
            closest_index = cluster_indices[closest_index_within_cluster]
            train_indices.append(int(closest_index))
    return train_indices


def plot_models_results(scores_dict, init_train_method, feature, model, save_path=True):
    """
    plot the F1 scores of various models over iterations
    """
    plt.figure(figsize=(10, 6))

    for criterion, scores in scores_dict.items():
        plt.plot(range(1, len(scores) + 1), scores, marker='o', label=criterion)

    plt.xlabel('Iteration')
    plt.ylabel('F1')
    plt.title(f'F1 Score Over 11 Iterations with Random Sampling (15 Samples/Iteration), Kmeans Initialization, and DeBERTa Embeddings', fontweight='bold')
    plt.legend()
    plt.xticks(range(1, len(next(iter(scores_dict.values()))) + 1))
    if save_path:
        path = f'Plots/model_F1_plot_{init_train_method}_{feature}_{model}.png'
        plt.savefig(path, format='png', dpi=300, bbox_inches='tight')  # Save the figure as a PNG file with high resolution
    plt.show()
    

def init_pipeline(path, feature, seed, init_train_method, use_dropout_aug=False, use_shorter_aug=False, use_rephrase_aug=False):
    """
    initialize the active learning pipeline by loading the dataset, applying augmentations, and selecting initial training indices
    """
    df = pd.read_csv(path)
    augmentation_features_name = []
    df[feature] = df[feature].apply(ast.literal_eval)
    # apply augmentations if specified
    if use_dropout_aug:
        df[f'{feature}_with_dropout'] = df[feature].apply(lambda x: dropout_augmentation(x, threshold=0.2))
        augmentation_features_name.append(f'{feature}_with_dropout')
    if use_shorter_aug:
        df['GPT3.5_shorter_text_embedding'] = df['GPT3.5_shorter_text_embedding'].apply(ast.literal_eval)
        augmentation_features_name.append('GPT3.5_shorter_text_embedding')
    if use_rephrase_aug:
        df['GPT3.5_rephrase_text_embedding'] = df['GPT3.5_rephrase_text_embedding'].apply(ast.literal_eval)
        augmentation_features_name.append('GPT3.5_rephrase_text_embedding')
    
    test_df = df[df['group'] == 'test']
    test_features = test_df[feature]
    test_labels = test_df['y']
    df = df[df['group'] != 'test'].reset_index(drop=True)
    # choose initial training indices based on the specified method
    train_indices = choose_initial_train_indeces(df, feature_name=feature, seed=seed, init_train_method=init_train_method)
    df.loc[train_indices, 'group'] = 'train'
    train_indices = list(df[df['group'] == 'train'].index)
    available_pool_indices = list(df[df['group'] == 'pool'].index)
    
    return train_indices, available_pool_indices, df, test_features, test_labels, augmentation_features_name
       

def run_pipeline(path, features, criteria, models, seeds, init_train_method='random', iterations=11, budget_per_iter=15,
                 use_dropout_aug=False, use_shorter_aug=False, use_rephrase_aug=False): 
    """
    run the active learning pipeline for multiple features, models, and criteria, collecting accuracy and F1 scores
    """
    scores_dict = defaultdict(list)
    for feature in features:
        scores_dict[feature] = {}
        for model in models: 
            scores_dict[feature][model] = {}
            for criterion in criteria:
                accuracy_list = []
                F1_list = []
                for seed in seeds:
                    train_indices, available_pool_indices, df, test_features, test_labels, augmentation_features_name = init_pipeline(path, feature, seed, init_train_method, use_dropout_aug, use_shorter_aug, use_rephrase_aug)
                    AL_class = ActiveLearningPipeline(model=model,
                                test_features = test_features,
                                test_labels = test_labels,
                                available_pool_indices=available_pool_indices,
                                train_indices=train_indices,
                                selection_criterion=criterion,
                                iterations=iterations,
                                budget_per_iter=budget_per_iter,
                                df=df,
                                feature=feature,
                                augmentation_features_name=augmentation_features_name,
                                seed=seed)
                    # run the active learning pipeline and collect accuracy and F1 scores
                    accuracy, f1 = AL_class.run_pipeline()
                    accuracy_list.append(accuracy)
                    F1_list.append(f1)
                # calculate mean F1 score for the current feature, model, and criterion
                scores_dict[feature][model][f'{criterion}'] = np.mean(F1_list, axis=0)
        plot_models_results(scores_dict[feature][model], init_train_method, feature, model)
    return scores_dict


if __name__ == "__main__":
    path = '/home/student/Project/Datasets/processed_df.csv'
    features = ['tfidf', 'bert', 'roberta', 'deberta']
    criteria = ['random', 'uncertainty', 'diversity', 'query_by_commitee', 'longest', 'shortest', 'sentiment', 'unbalanced', 'cluster_uncertainty', 'cluster_disagreement', 'mistake']       
    models = ['RandomForestClassifier', 'LogisticRegression', 'SVC']
    seeds = [1,2,3,4,5]

    accuracy_scores_dict = run_pipeline(path, features, criteria, models, seeds, init_train_method='kmeans')

    # Plot Different Choices of Train Indeces
    plot_init_train_indeces = False
    if plot_init_train_indeces:
        random_accuracy_scores_dict = run_pipeline(path, features, criteria, models, seeds=seeds, init_train_method='random') 
        kmeans_accuracy_scores_dict = run_pipeline(path, features, criteria, models, seeds=seeds, init_train_method='kmeans') 

        inital_accuracy_dict = {}
        inital_accuracy_dict['random'] = random_accuracy_scores_dict[features[0]][models[0]]
        inital_accuracy_dict['kmeans'] = kmeans_accuracy_scores_dict[features[0]][models[0]]
        
        plot_models_results(inital_accuracy_dict, init_train_method='', feature=features[0], model=models[0], is_comb=True)

    # Plot Different Augmentations Methods
    plot_augmentation_indeces = False
    if plot_augmentation_indeces:
        without_augmentation_accuracy_scores_dict = run_pipeline(path, features, criteria, models, seeds=seeds, init_train_method='kmeans') 
        dropout_augmentation_accuracy_scores_dict = run_pipeline(path, features, criteria, models, seeds=seeds, init_train_method='kmeans', use_dropout_aug=True) 
        shorter_augmentation_accuracy_scores_dict = run_pipeline(path, features, criteria, models, seeds=seeds, init_train_method='kmeans', use_shorter_aug=True) 
        rephrase_augmentation_accuracy_scores_dict = run_pipeline(path, features, criteria, models, seeds=seeds, init_train_method='kmeans', use_rephrase_aug=True) 
        all_augmentation_accuracy_scores_dict = run_pipeline(path, features, criteria, models, seeds=seeds, init_train_method='kmeans', use_dropout_aug=True, use_shorter_aug=True, use_rephrase_aug=True) 

        augmentation_accuracy_dict = {}
        augmentation_accuracy_dict['without'] = without_augmentation_accuracy_scores_dict[features[0]][models[0]][criteria[0]]
        augmentation_accuracy_dict['dropout'] = dropout_augmentation_accuracy_scores_dict[features[0]][models[0]][criteria[0]]
        augmentation_accuracy_dict['shorter'] = shorter_augmentation_accuracy_scores_dict[features[0]][models[0]][criteria[0]]
        augmentation_accuracy_dict['rephrase'] = rephrase_augmentation_accuracy_scores_dict[features[0]][models[0]][criteria[0]]
        augmentation_accuracy_dict['all'] = all_augmentation_accuracy_scores_dict[features[0]][models[0]][criteria[0]]
        
        plot_models_results(augmentation_accuracy_dict, init_train_method='', feature=features[0], model=models[0])
        
    # Plot all Random VS the best
    plot_comparison = False
    if plot_comparison:
        random_dict = run_pipeline(path, features, ['random'], models, seeds=seeds, init_train_method='random') 
        best_dict = run_pipeline(path, features, ['largest'], models, seeds=seeds, init_train_method='kmeans', use_shorter_aug=True) 
        
        augmentation_accuracy_dict = {}
        augmentation_accuracy_dict['random'] = random_dict[features[0]][models[0]]['random']
        augmentation_accuracy_dict['largest'] = best_dict[features[0]][models[0]]['largest']
        
        plot_models_results(augmentation_accuracy_dict, init_train_method='', feature=features[0], model=models[0])
