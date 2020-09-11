import sys
import click
import logging
import pickle
import pkg_resources
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

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

    figures2_prefix = str(figures_prefix) + '2_'

    training_data = features.get_features(
        training_url_prefix,
        shared_access_signature=shared_access_signature,
        figures_prefix=figures_prefix,
        proportion_s_train=0.3,
        proportion_s_test=0.3,
        random_seed=42,
        use_rt_features=False,
        use_pca_features=True
    )

    training_data2 = features.get_features(
        training_url_prefix,
        shared_access_signature=shared_access_signature,
        figures_prefix=figures2_prefix,
        proportion_s_train=0.3,
        proportion_s_test=0.3,
        random_seed=42,
        use_rt_features=False,
        use_pca_features=False
    )

    cn_data, metrics_data, align_metrics_data = features.get_data(training_url_prefix, shared_access_signature)

    logging.info('training a classifier to test performance')

    classifier1, stats1, yg1, yp1, ypp1, cell_ids1 = model.train_test_model(
        training_data,
        figures_prefix=figures_prefix,
        random_seed=42,
        use_rt_features=False,
        use_pca_features=True
    )

    classifier2, stats2, yg2, yp2, ypp2, cell_ids2 = model.train_test_model(
        training_data2,
        figures_prefix=figures2_prefix,
        random_seed=42,
        use_rt_features=False,
        use_pca_features=False
    )

    yg1 = yg1.astype(int)
    yg2 = yg2.astype(int)
    yp1 = yp1.astype(int)
    yp2 = yp2.astype(int)

    y_g_p1 = np.subtract(yg1, yp1)
    y_g_p2 = np.subtract(yg2, yp2)
    y_p1_p2 = np.subtract(yp1, yp2)

    save_y_arrays(yg1, yg2, yp1, yp2, ypp1, ypp2, cell_ids1, cell_ids2)
    cell_heatmap(yg1, y_g_p1, y_g_p2, y_p1_p2, figures_prefix)
    confusion_mats(yg1, yg2, yp1, yp2, figures_prefix)
    all_misclassified = misclassified_cells(y_g_p1, y_g_p2, y_p1_p2, cell_ids1, figures_prefix)

    cn_data.set_index('cell_id', inplace=True)
    cn_missed = cn_data.loc[list(all_misclassified), :]
    cn_missed.to_csv('test_cn_missed.tsv', sep='\t')

    logging.info(stats1)
    logging.info(stats2)

    training_data.to_csv(features_filename, index=False)


def misclassified_cells(y_g_p1, y_g_p2, y_p1_p2, cell_ids, figures_prefix):
    false_pos_1 = cell_ids[np.where(y_g_p1 == 1)]
    false_pos_2 = cell_ids[np.where(y_g_p2 == 1)]

    false_neg_1 = cell_ids[np.where(y_g_p1 == -1)]
    false_neg_2 = cell_ids[np.where(y_g_p2 == -1)]

    false_pos_both = np.intersect1d(false_pos_1, false_pos_2)
    false_pos_only1 = np.setdiff1d(false_pos_1, false_pos_2)
    false_pos_only2 = np.setdiff1d(false_pos_2, false_pos_1)

    false_neg_both = np.intersect1d(false_neg_1, false_neg_2)
    false_neg_only1 = np.setdiff1d(false_neg_1, false_neg_2)
    false_neg_only2 = np.setdiff1d(false_neg_2, false_neg_1)

    all_misclassified = np.unique(np.hstack([false_neg_1, false_neg_2, false_pos_1, false_pos_2]))

    colnames = ['cell_id', 'sample_id', 'library_id', 
                'false_pos_1', 'false_pos_2', 'false_neg_1', 'false_neg_2',
                'false_pos_both', 'false_pos_only1', 'false_pos_only2',
                'false_neg_both', 'false_neg_only1', 'false_neg_only2']
    bool_colnames = ['false_pos_1', 'false_pos_2', 'false_neg_1', 'false_neg_2',
                    'false_pos_both', 'false_pos_only1', 'false_pos_only2',
                    'false_neg_both', 'false_neg_only1', 'false_neg_only2']
    legend = pd.DataFrame(columns=colnames)
    legend.set_index('cell_id', inplace=True)

    # create large legend with bool marker for each cell's misclassification type 
    for cell in list(all_misclassified):
        temp_sample_id = cell.split('-')[0]
        temp_library_id = cell.split('-')[1]
        temp_list = [temp_sample_id, temp_library_id]
        for item in bool_colnames:
            temp_list.append(cell in eval(item))
        legend.loc[cell, :] = temp_list

    # shorten legend into sums that can be used for a sample-split bar plot
    short_df = pd.DataFrame(columns=colnames)
    short_df.drop(columns=['cell_id', 'library_id'], inplace=True)
    short_df.set_index('sample_id', inplace=True)
    for sample in list(legend['sample_id'].unique()):
        temp_legend = legend[legend['sample_id']==sample]
        for item in short_df.columns:
            short_df.loc[sample, item] = temp_legend[item].sum()

    # create bar plot colored by sample_id
    fig, ax = plt.subplots(1, 1, figsize=(8, 8), tight_layout=True)
    short_df.T.plot.bar(rot=30, ax=ax)
    ax.set_title('Summary of Misclassifications')
    ax.set_ylabel('number of cells')
    ax.set_xlabel('misclassification condition')
    ax.set_xticklabels(['FP w/ rt & pca features', 'FP w/o rt & pca features', 'FN w/ rt & pca features', 'FN w/o rt & pca features',
                        'FP both methods', 'FP only w/ rt & pca features', 'FP only w/o rt & pca features',
                        'FN both methods', 'FN only w/ rt & pca features', 'FN only w/o rt & pca features'])
    fig.savefig('{prefix}misclassified_hist.png'.format(prefix=figures_prefix))

    # save legend as tsv
    legend.to_csv('test_set_misclassifications.tsv', sep='\t', index=True)

    return all_misclassified



def cell_heatmap(yg1, y_g_p1, y_g_p2, y_p1_p2, figures_prefix):
    mat = np.vstack([yg1, y_g_p1, y_g_p2, y_p1_p2]).T
    fig, ax = plt.subplots(1, 1, figsize=(8, 11), tight_layout=True)
    sns.heatmap(mat, ax=ax)
    ax.set_ylabel('cell')
    ax.set_xticklabels(['truth', 'truth\n- Y1 pred', 'truth\n- Y2 pred', 'Y1 pred\n- Y2 pred'])
    fig.savefig('{prefix}cell_heatmap.png'.format(prefix=figures_prefix))


def confusion_mats(yg1, yg2, yp1, yp2, figures_prefix):
    # plot confusion matrix for both classifiers
    fig, ax = plt.subplots(1, 2, figsize=(10, 5), tight_layout=True)
    ax = ax.flatten()
    classes = ['non-s-phase', 's-phase']

    # subplot for classifier1
    confusion1 = confusion_matrix(yg1, yp1)
    sns.heatmap(confusion1, annot=True, fmt=".0f", cmap=plt.cm.Blues, ax=ax[0])
    ax[0].set_xlabel('Predicted label')
    ax[0].set_ylabel('True label')
    ax[0].set_xticklabels(classes)
    ax[0].set_yticklabels(classes)
    ax[0].set_title('Confusion Matrix\nwith RT & PCA features')

    # subplot for classifier1
    confusion2 = confusion_matrix(yg2, yp2)
    sns.heatmap(confusion2, annot=True, fmt=".0f", cmap=plt.cm.Blues, ax=ax[1])
    ax[1].set_xlabel('Predicted label')
    ax[1].set_ylabel('True label')
    ax[1].set_xticklabels(classes)
    ax[1].set_yticklabels(classes)
    ax[1].set_title('Confusion Matrix\nwithout RT & PCA features')

    fig.savefig('{prefix}confusion_mats.png'.format(prefix=figures_prefix))


def save_y_arrays(yg1, yg2, yp1, yp2, ypp1, ypp2, cell_ids1, cell_ids2):
    if not np.equal(yg1, yg2).all():
        print("ground truth arrays aren't equal")
        print(yg1[:10])
        print(yg2[:10])

    if not np.equal(cell_ids1, cell_ids2).all():
        print("cell_id arrays aren't equal")
        print(cell_ids1[:10])
        print(cell_ids2[:10])

    np.savetxt('Y_ground_truth_1.txt', yg1)
    np.savetxt('Y_ground_truth_2.txt', yg2)
    np.savetxt('Y_predicted_1.txt', yp1)
    np.savetxt('Y_predicted_2.txt', yp2)
    np.savetxt('Y_predicted_prob_1.txt', ypp1)
    np.savetxt('Y_predicted_prob_2.txt', ypp2)
    np.savetxt('test_set_cell_ids_1.txt', cell_ids1, fmt="%s")
    np.savetxt('test_set_cell_ids_2.txt', cell_ids2, fmt="%s")



@cli.command()
@click.argument('cn_filename')
@click.argument('metrics_filename')
@click.argument('align_metrics_filename')
@click.argument('predictions_filename')
@click.option('--figures_prefix')
def train_classify(cn_filename, metrics_filename, align_metrics_filename, predictions_filename, figures_prefix=None):
    cn_data = pd.read_csv(cn_filename)
    metrics_data = pd.read_csv(metrics_filename)
    align_metrics_data = pd.read_csv(align_metrics_filename)

    predictions = api.train_classify(cn_data, metrics_data, align_metrics_data, figures_prefix=figures_prefix)

    predictions.to_csv(predictions_filename, index=False)


if __name__ == '__main__':
    cli()
