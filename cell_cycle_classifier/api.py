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
            training_data_filename = pkg_resources.resource_filename('cell_cycle_classifier', 'data/training/curated_feature_data_rt_v2.csv')
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

    print(training_data_filename)

    training_data = pd.read_csv(training_data_filename)

    classifier, stats, __, __, __, __ = model.train_test_model(
        training_data,
        figures_prefix=figures_prefix,
        random_seed=42,
        use_rt_features=use_rt_features,
        use_pca_features=use_pca_features
    )

    logging.info(stats)

    cn_data = cn_data.merge(metrics_data[['cell_id', 'library_id']].drop_duplicates(), how='left')
    assert not cn_data['library_id'].isnull().any()

    # split data into human and non-human
    metrics_data_nonhuman = metrics_data.query("species != 'grch37'")
    metrics_data_human = metrics_data.query("species == 'grch37'")
    nonhuman_cells = metrics_data_nonhuman.cell_id.unique()
    align_metrics_data_nonhuman = align_metrics_data.loc[align_metrics_data['cell_id'].isin(nonhuman_cells)]
    align_metrics_data_human = align_metrics_data.loc[~align_metrics_data['cell_id'].isin(nonhuman_cells)]
    cn_data_nonhuman = cn_data.loc[cn_data['cell_id'].isin(nonhuman_cells)]
    cn_data_human = cn_data.loc[~cn_data['cell_id'].isin(nonhuman_cells)]

    # calculate features & make predictions for human cells
    logging.info('calculating features for human cells')
    feature_data = features.calculate_features(
        cn_data_human,
        metrics_data_human,
        align_metrics_data_human,
        figures_prefix=figures_prefix,
        use_rt_features=use_rt_features,
        use_pca_features=use_pca_features
    )

    logging.info('predicting cell cycle for human cells')
    predictions = model.predict(
        classifier,
        feature_data,
        use_rt_features=use_rt_features,
        use_pca_features=use_pca_features
    )
    
    # calculate features, train classifier, and predict state without RT or PCA features
    # for the non-human cells
    if len(nonhuman_cells) > 0:
        logging.info('calculating features for non-human cells')
        figures_prefix_nonhuman = figures_prefix + 'nonhuman_'

        feature_data_nonhuman = features.calculate_features(
            cn_data_nonhuman,
            metrics_data_nonhuman,
            align_metrics_data_nonhuman,
            figures_prefix=figures_prefix_nonhuman,
            use_rt_features=False,
            use_pca_features=False
        )

        # if all cells get filtered entirely when calculating features
        # then we can jump out of non-human analysis
        if not feature_data_nonhuman.empty:
            logging.info('training a classifier for non-human cells')
            if use_curated_labels:
                training_data_filename = pkg_resources.resource_filename('cell_cycle_classifier', 'data/training/curated_feature_data_v2.csv')
            else:
                training_data_filename = pkg_resources.resource_filename('cell_cycle_classifier', 'data/training/feature_data.csv.gz')
            training_data_vanilla = pd.read_csv(training_data_filename)
            classifier_vanilla, stats, __, __, __, __ = model.train_test_model(
                training_data_vanilla,
                figures_prefix=figures_prefix_nonhuman,
                random_seed=42,
                use_rt_features=False,
                use_pca_features=False
            )

            logging.info('predicting cell cycle for non-human cells')
            predictions_nonhuman = model.predict(
                classifier_vanilla,
                feature_data_nonhuman,
                use_rt_features=False,
                use_pca_features=False
            )

            predictions = pd.concat([predictions, predictions_nonhuman])

    predictions = predictions.merge(metrics_data[['cell_id']].drop_duplicates(), how='right')
    predictions['is_s_phase'] = predictions['is_s_phase'].fillna(False)

    return predictions
