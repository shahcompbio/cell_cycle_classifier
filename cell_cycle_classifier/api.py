import logging
import pkg_resources
import pandas as pd

import cell_cycle_classifier.model as model
import cell_cycle_classifier.features as features


def train_classify(cn_data, metrics_data, align_metrics_data, figures_prefix=None,
                   use_rt_features=True, use_pca_features=False, use_curated_labels=True):
    logging.info('training a classifier')


    if use_rt_features and not use_pca_features:
        if use_curated_labels:
            training_data_filename = pkg_resources.resource_filename('cell_cycle_classifier', 'data/training/curated_feature_data_v2.csv')
        else:
            training_data_filename = pkg_resources.resource_filename('cell_cycle_classifier', 'data/training/feature_data_rt_v2.csv')
    elif use_rt_features and use_pca_features:
        if use_curated_labels:
            training_data_filename = pkg_resources.resource_filename('cell_cycle_classifier', 'data/training/curated_feature_data_rt_pca.csv')
        else:
            training_data_filename = pkg_resources.resource_filename('cell_cycle_classifier', 'data/training/feature_data_rt_pca.csv')
    else:
        if use_curated_labels:
            training_data_filename = pkg_resources.resource_filename('cell_cycle_classifier', 'data/training/curated_feature_data_v2.csv')
        else:
            training_data_filename = pkg_resources.resource_filename('cell_cycle_classifier', 'data/training/feature_data.csv.gz')

    training_data = pd.read_csv(training_data_filename)

    classifier, stats, __, __, __, __ = model.train_test_model(
        training_data,
        figures_prefix=figures_prefix,
        random_seed=42,
        use_rt_features=use_rt_features,
        use_pca_features=use_pca_features
    )

    logging.info(stats)

    for data in (cn_data, metrics_data, align_metrics_data):
        data['sample_id'] = data['cell_id'].apply(lambda a: a.split('-')[0])
        data['library_id'] = data['cell_id'].apply(lambda a: a.split('-')[1])

    logging.info('calculating features')

    feature_data = features.calculate_features(
        cn_data,
        metrics_data,
        align_metrics_data,
        figures_prefix=figures_prefix,
        use_rt_features=use_rt_features,
        use_pca_features=use_pca_features
    )

    logging.info('predicting cell cycle')

    predictions = model.predict(
        classifier,
        feature_data,
        use_rt_features=use_rt_features,
        use_pca_features=use_pca_features
    )

    predictions = predictions.merge(metrics_data[['cell_id']].drop_duplicates(), how='right')
    predictions['is_s_phase'] = predictions['is_s_phase'].fillna(False)

    return predictions
