import logging
import pkg_resources
import pandas as pd

import cell_cycle_classifier.model as model
import cell_cycle_classifier.features as features


def train_classify(cn_data, metrics_data, figures_prefix=None):
    logging.info('training a classifier')

    training_data_filename = pkg_resources.resource_filename('cell_cycle_classifier', 'data/training/feature_data.csv.gz')

    training_data = pd.read_csv(training_data_filename)

    classifier, stats = model.train_test_model(
        training_data,
        figures_prefix=figures_prefix,
        random_seed=42,
    )

    logging.info(stats)

    cn_data = cn_data.merge(metrics_data[['cell_id', 'library_id']].drop_duplicates(), how='left')
    assert not cn_data['library_id'].isnull().any()

    logging.info('calculating features')

    feature_data = features.calculate_features(
        cn_data,
        metrics_data,
        figures_prefix=figures_prefix,
    )

    logging.info('predicting cell cycle')

    predictions = model.predict(
        classifier,
        feature_data,
    )

    predictions = predictions.merge(metrics_data[['cell_id']].drop_duplicates(), how='right')
    predictions['is_s_phase'] = predictions['is_s_phase'].fillna(False)

    return predictions
