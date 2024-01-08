import sys
import click
import logging
import pickle
import pkg_resources
import pandas as pd

import cell_cycle_classifier.model as model
import cell_cycle_classifier.features as features
import cell_cycle_classifier.api as api

LOGGING_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"

logging.basicConfig(format=LOGGING_FORMAT, stream=sys.stderr, level=logging.INFO)


@click.group()
def cli():
    pass


@cli.command()
@click.argument('training_url_prefix')
@click.argument('features_filename')
@click.option('--shared_access_signature', default='')
@click.option('--figures_prefix')
def get_features(training_url_prefix, features_filename, shared_access_signature='', figures_prefix=None):
    logging.info('obtaining training data')

    training_data = features.get_features(
        training_url_prefix,
        shared_access_signature=shared_access_signature,
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
def train_classify(cn_filename, metrics_filename, predictions_filename, figures_prefix=None):
    cn_data = pd.read_csv(cn_filename)
    metrics_data = pd.read_csv(metrics_filename)

    predictions = api.train_classify(cn_data, metrics_data, figures_prefix=figures_prefix)

    predictions.to_csv(predictions_filename, index=False)


if __name__ == '__main__':
    cli()
