import sys
import click
import logging
import pickle
import pkg_resources
import pandas as pd

import cell_cycle_classifier.model as model
import cell_cycle_classifier.features as features

LOGGING_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"

logging.basicConfig(format=LOGGING_FORMAT, stream=sys.stderr, level=logging.INFO)


@click.group()
def cli():
    pass


@cli.command()
@click.argument('features_filename')
@click.argument('shared_access_signature')
@click.option('--figures_prefix')
def get_features(features_filename, shared_access_signature, figures_prefix=None):
    logging.info('obtaining training data')

    training_data = features.get_features(
        shared_access_signature,
        figures_prefix=figures_prefix,
        proportion_s_train=0.3,
        proportion_s_test=0.3,
        random_seed=42,
    )

    logging.info('training a classifier to test performance')

    _, stats = model.train_test_model(
        training_data,
        figures_prefix=figures_prefix,
        random_seed=42,
    )

    logging.info(stats)

    training_data.to_csv(features_filename, index=False)


@cli.command()
@click.argument('cn_filename')
@click.argument('metrics_filename')
@click.argument('align_metrics_filename')
@click.argument('predictions_filename')
@click.option('--figures_prefix')
def train_classify(cn_filename, metrics_filename, align_metrics_filename, predictions_filename, figures_prefix=None):
    logging.info('training a classifier')

    training_data_filename = pkg_resources.resource_filename('cell_cycle_classifier', 'data/training/feature_data.csv.gz')

    training_data = pd.read_csv(training_data_filename)

    classifier, stats = model.train_test_model(
        training_data,
        figures_prefix=figures_prefix,
        random_seed=42,
    )

    logging.info(stats)

    cn_data = pd.read_csv(cn_filename)
    metrics_data = pd.read_csv(metrics_filename)
    align_metrics_data = pd.read_csv(align_metrics_filename)

    for data in (cn_data, metrics_data, align_metrics_data):
        data['sample_id'] = data['cell_id'].apply(lambda a: a.split('-')[0])
        data['library_id'] = data['cell_id'].apply(lambda a: a.split('-')[1])

    logging.info('calculating features')

    feature_data = features.calculate_features(
        cn_data,
        metrics_data,
        align_metrics_data,
        figures_prefix=figures_prefix,
    )

    logging.info('predicting cell cycle')

    predictions = model.predict(
        classifier,
        feature_data,
    )

    predictions.to_csv(predictions_filename, index=False)


if __name__ == '__main__':
    cli()
