import logging
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.utils import shuffle
from sklearn.preprocessing import PolynomialFeatures

import cell_cycle_classifier.features as features


def train_model(feature_data, feature_names, random_state=None):
    """ Train s phase state classifier
    
    Args:
        feature_data (pandas.DataFrame): feature data
        feature_names (list of str): List of feature names.
        random_state (int, optional): Random state to initialize during training.
    
    Returns:
        model: Cell state classification model
    """
    X = feature_data[feature_names].values
    y = feature_data['cell_cycle_state'].values == 'S'

    classifier = RandomForestClassifier(
        n_estimators=100, max_depth=2,
        random_state=random_state)

    classifier.fit(X, y)

    logging.info(
        'Accuracy of classifier on training set: {:.2f}'
        .format(classifier.score(X, y)))

    return classifier


def predict(classifier, feature_data, feature_names=None):
    """ Predict s phase state
    
    Args:
        classifier: Cell state classifier
        feature_data (pandas.DataFrame): feature data
        feature_names (list of str, optional): List of feature names. Defaults to None.
    
    Returns:
        pandas.DataFrame: Predictions
    """
    if feature_names is None:
        feature_names = features.all_feature_names

    X = feature_data[feature_names].values

    y_pred = classifier.predict(X)
    y_pred_proba = classifier.predict_proba(X)[::,1]

    predictions = pd.DataFrame({
        'cell_id': feature_data['cell_id'],
        'is_s_phase': y_pred,
        'is_s_phase_prob': y_pred_proba,
    })

    return predictions


def train_test_model(
        feature_data,
        figures_prefix=None,
        feature_names=None,
        random_seed=None,
        use_rt_features=True,
        use_pca_features=True
    ):
    """ Train and test the model given annotated input copy number data.
    
    Args:
        features_data (pandas.DataFrame): precalculated feature data
        figures_prefix (str, optional): Prefix for figure filenames. Defaults to None.
        feature_names (list of str, optional): Subset of features. Defaults to None, all features.
        random_seed (int, optional): Random seed for selecting test set. Defaults to None.
        use_rt_features (bool, optional): Mark false when replication timing features should be ignored.
        use_pca_features (bool, optional): Mark false when PCA features should be ignored.    
    Returns:
        [type]: [description]
    """

    if feature_names is None:
        feature_names = features.all_feature_names

    # remove rt or pca feature names if necessary
    rt_features = ['r_ratio', 'r_G1b', 'r_S4', 'num_unique_bk']
    pca_features = ['PC1', 'PC2', 'PC3']
    if use_rt_features is False and set(rt_features).issubset(set(feature_names)):
        feature_names = [x for x in feature_names if x not in rt_features]
    if use_pca_features is False and set(pca_features).issubset(set(feature_names)):
        feature_names = [x for x in feature_names if x not in pca_features]

    print('feature_names', feature_names)

    training_data = feature_data.query('training_context == "training"')
    testing_data = feature_data.query('training_context == "holdout"')
    test_cell_ids = testing_data['cell_id'].values
    testing_data['library_id'] = testing_data['cell_id'].apply(lambda x: x.split('-')[1])
    testing_data['sample_id'] = testing_data['cell_id'].apply(lambda x: x.split('-')[0])
    # sc_legend['cell_id'] = sc_legend['Sample'].apply(lambda x: x.split('.')[0])

    logging.info('training model')
    classifier = train_model(training_data, feature_names, random_state=random_seed)

    for sample_id, chunk in testing_data.groupby('sample_id'):
        X = chunk[feature_names].values
        y = chunk['cell_cycle_state'].values == 'S'

        logging.info(
            'Accuracy of classifier on test set: {:.2f}'
            .format(classifier.score(X, y)))

        y_pred = classifier.predict(X)

        logging.info("Accuracy: {}".format(metrics.accuracy_score(y, y_pred)))
        logging.info("Precision: {}".format(metrics.precision_score(y, y_pred)))

        y_pred_proba = classifier.predict_proba(X)[::,1]
        fpr, tpr, _ = metrics.roc_curve(y, y_pred_proba)

        if figures_prefix:
            fig = plt.figure()
            auc = metrics.roc_auc_score(y, y_pred_proba)
            plt.plot(fpr,tpr,label="AUC={:.2f}, n={}".format(auc, y.shape[0]))
            plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
            plt.legend(loc=4)
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(sample_id)
            sns.despine(offset=True, trim=True)
            fig.savefig(figures_prefix + sample_id + 'roc.pdf', bbox_inches='tight')
        
            fig = plt.figure()
            feature_importance = pd.Series(dict(zip(feature_names, classifier.feature_importances_)))
            feature_importance.plot.bar()
            fig.savefig(figures_prefix + 'features.pdf', bbox_inches='tight')

            if use_rt_features:
                fig = plt.figure()
                sns.scatterplot(x=chunk['r_G1b'].values, y=chunk['r_S4'].values, hue=y)
                plt.title(sample_id)
                plt.xlabel('G1b (early) correlation')
                plt.ylabel('S4 (late) correlation')
                plt.legend(title='S-phase status (flow)')
                fig.savefig(figures_prefix + sample_id + '_rt_specificity.pdf', bbox_inches='tight')

        stats = dict(
            accuracy=metrics.accuracy_score(y, y_pred),
            precision=metrics.precision_score(y, y_pred),
            recall=metrics.recall_score(y, y_pred),
            f1=metrics.f1_score(y, y_pred),
            auc=metrics.roc_auc_score(y, y_pred_proba),
            random_seed=random_seed,
        )

    return classifier, stats, y, y_pred, y_pred_proba, test_cell_ids


