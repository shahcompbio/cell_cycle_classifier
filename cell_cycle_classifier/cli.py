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
from sklearn import metrics

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
    figures3_prefix = str(figures_prefix) + '3_'
    figures4_prefix = str(figures_prefix) + '4_'

    training_data = features.get_features(
        training_url_prefix,
        shared_access_signature=shared_access_signature,
        figures_prefix=figures_prefix,
        proportion_s_train=0.3,
        proportion_s_test=0.3,
        random_seed=42,
        use_rt_features=False,
        use_pca_features=False
    )

    training_data2 = features.get_features(
        training_url_prefix,
        shared_access_signature=shared_access_signature,
        figures_prefix=figures2_prefix,
        proportion_s_train=0.3,
        proportion_s_test=0.3,
        random_seed=42,
        use_rt_features=True,
        use_pca_features=False
    )

    training_data2.to_csv('cell_cycle_classifier/data/training/feature_data_rt_v2.csv', index=False)

    training_data3 = features.get_features(
        training_url_prefix,
        shared_access_signature=shared_access_signature,
        figures_prefix=figures3_prefix,
        proportion_s_train=0.3,
        proportion_s_test=0.3,
        random_seed=42,
        use_rt_features=False,
        use_pca_features=False,
        curated_labels='cell_cycle_classifier/data/training/curated_cell_cycle_state.csv'
    )

    training_data3.to_csv('cell_cycle_classifier/data/training/curated_feature_data_v2.csv', index=False)

    training_data4 = features.get_features(
        training_url_prefix,
        shared_access_signature=shared_access_signature,
        figures_prefix=figures4_prefix,
        proportion_s_train=0.3,
        proportion_s_test=0.3,
        random_seed=42,
        use_rt_features=True,
        use_pca_features=False,
        curated_labels='cell_cycle_classifier/data/training/curated_cell_cycle_state.csv'
    )

    training_data4.to_csv('cell_cycle_classifier/data/training/curated_feature_data_rt_v2.csv', index=False)


    # cn_data, metrics_data, align_metrics_data = features.get_data(training_url_prefix, shared_access_signature)

    logging.info('training a classifier to test performance')

    classifier1, stats1, yg1, yp1, ypp1, testing_data1 = model.train_test_model(
        training_data,
        figures_prefix=figures_prefix,
        random_seed=42,
        use_rt_features=False,
        use_pca_features=False
    )

    classifier2, stats2, yg2, yp2, ypp2, testing_data2 = model.train_test_model(
        training_data2,
        figures_prefix=figures2_prefix,
        random_seed=42,
        use_rt_features=True,
        use_pca_features=False
    )

    classifier3, stats3, yg3, yp3, ypp3, testing_data3 = model.train_test_model(
        training_data3,
        figures_prefix=figures3_prefix,
        random_seed=42,
        use_rt_features=False,
        use_pca_features=False
    )

    classifier4, stats4, yg4, yp4, ypp4, testing_data4 = model.train_test_model(
        training_data4,
        figures_prefix=figures4_prefix,
        random_seed=42,
        use_rt_features=True,
        use_pca_features=False
    )

    testing_data1.to_csv('testing_data1.tsv', sep='\t')
    testing_data2.to_csv('testing_data2.tsv', sep='\t')
    testing_data3.to_csv('testing_data3.tsv', sep='\t')
    testing_data4.to_csv('testing_data4.tsv', sep='\t')

    # plot ROC curves for all conditions on one plot
    fpr1, tpr1, _ = metrics.roc_curve(yg1, ypp1)
    fpr2, tpr2, _ = metrics.roc_curve(yg2, ypp2)
    fpr3, tpr3, _ = metrics.roc_curve(yg3, ypp3)
    fpr4, tpr4, _ = metrics.roc_curve(yg4, ypp4)

    if figures_prefix:
        fig = plt.figure()
        plt.plot(fpr1, tpr1, color='g',
            label="old features, old data,\nAUC={:.2f}, n={}".format(stats1['auc'], yg1.shape[0]))
        plt.plot(fpr2, tpr2, color='r',
            label="new features, old data,\nAUC={:.2f}, n={}".format(stats2['auc'], yg2.shape[0]))
        plt.plot(fpr3, tpr3, color='k',
            label="old features, new data,\nAUC={:.2f}, n={}".format(stats3['auc'], yg3.shape[0]))
        plt.plot(fpr4, tpr4, color='b',
            label="new features, new data,\nAUC={:.2f}, n={}".format(stats4['auc'], yg4.shape[0]))
        plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
        plt.legend(loc=4)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        sns.despine(offset=True, trim=True)
        fig.savefig(figures_prefix + 'combined_roc.pdf', bbox_inches='tight')


    # plot ROC curves for all conditions, split by sample_id
    for sample_id in testing_data1.sample_id.unique():
        chunk1 = testing_data1.query('sample_id == "{}"'.format(sample_id))
        chunk2 = testing_data2.query('sample_id == "{}"'.format(sample_id))
        chunk3 = testing_data3.query('sample_id == "{}"'.format(sample_id))
        chunk4 = testing_data4.query('sample_id == "{}"'.format(sample_id))

        # get old and new feature names used in classifier
        new_feature_names = features.all_feature_names
        old_feature_names = new_feature_names
        rt_features = ['r_ratio', 'r_G1b', 'r_S4', 'slope_ratio',
                    'slope_G1b', 'slope_S4', 'num_unique_bk', 'norm_bk']
        pca_features = ['PC1', 'PC2', 'PC3']
        if set(rt_features).issubset(set(old_feature_names)):
            old_feature_names = [x for x in old_feature_names if x not in rt_features]
        if set(pca_features).issubset(set(old_feature_names)):
            old_feature_names = [x for x in old_feature_names if x not in pca_features]

        # get X and y arrays
        X1 = chunk1[old_feature_names].values
        X2 = chunk2[new_feature_names].values
        X3 = chunk3[old_feature_names].values
        X4 = chunk4[new_feature_names].values
        y1 = chunk1['cell_cycle_state'].values == 'S'
        y2 = chunk2['cell_cycle_state'].values == 'S'
        y3 = chunk3['cell_cycle_state'].values == 'S'
        y4 = chunk4['cell_cycle_state'].values == 'S'

        # calculate y predicted probabilities
        y_pred_proba1 = classifier1.predict_proba(X1)[::,1]
        y_pred_proba2 = classifier2.predict_proba(X2)[::,1]
        y_pred_proba3 = classifier3.predict_proba(X3)[::,1]
        y_pred_proba4 = classifier4.predict_proba(X4)[::,1]

        # calculate FPRs and TPRs
        fpr1, tpr1, _ = metrics.roc_curve(y1, y_pred_proba1)
        fpr2, tpr2, _ = metrics.roc_curve(y2, y_pred_proba2)
        fpr3, tpr3, _ = metrics.roc_curve(y3, y_pred_proba3)
        fpr4, tpr4, _ = metrics.roc_curve(y4, y_pred_proba4)

        # calculate AUC scores
        auc1 = metrics.roc_auc_score(y1, y_pred_proba1)
        auc2 = metrics.roc_auc_score(y2, y_pred_proba2)
        auc3 = metrics.roc_auc_score(y3, y_pred_proba3)
        auc4 = metrics.roc_auc_score(y4, y_pred_proba4)

        # plot the fprs and tprs into an ROC plot for this sample_id
        if figures_prefix:
            fig = plt.figure()
            plt.plot(fpr1, tpr1, color='g',
                label="old features, old data,\nAUC={:.2f}, n={}".format(auc1, y1.shape[0]))
            plt.plot(fpr2, tpr2, color='r',
                label="new features, old data,\nAUC={:.2f}, n={}".format(auc2, y2.shape[0]))
            plt.plot(fpr3, tpr3, color='k',
                label="old features, new data,\nAUC={:.2f}, n={}".format(auc3, y3.shape[0]))
            plt.plot(fpr4, tpr4, color='b',
                label="new features, new data,\nAUC={:.2f}, n={}".format(auc4, y4.shape[0]))
            plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
            plt.legend(loc=4)
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(sample_id)
            sns.despine(offset=True, trim=True)
            fig.savefig(figures_prefix + sample_id + '_combined_roc.pdf', bbox_inches='tight')


    # cell_ids1 = testing_data1['cell_id'].values
    # cell_ids2 = testing_data2['cell_id'].values

    # # save cn profiles so we can query for probably flow errors
    # testing_data1 = testing_data1.merge(cn_data, on=['cell_id', 'library_id', 'sample_id', 'cell_cycle_state'])
    # testing_data1.to_csv('testing_data1.tsv', sep='\t')

    # yg1 = yg1.astype(int)
    # yg2 = yg2.astype(int)
    # yp1 = yp1.astype(int)
    # yp2 = yp2.astype(int)

    # y_g_p1 = np.subtract(yg1, yp1)
    # y_g_p2 = np.subtract(yg2, yp2)
    # y_p1_p2 = np.subtract(yp1, yp2)

    # save_y_arrays(yg1, yg2, yp1, yp2, ypp1, ypp2, cell_ids1, cell_ids2)
    # cell_heatmap(yg1, y_g_p1, y_g_p2, y_p1_p2, figures_prefix)
    # confusion_mats(yg1, yg2, yp1, yp2, figures_prefix)
    # all_misclassified = misclassified_cells(y_g_p1, y_g_p2, y_p1_p2, cell_ids1, figures_prefix)

    # cn_data.set_index('cell_id', inplace=True)
    # cn_missed = cn_data.loc[list(all_misclassified), :]
    # cn_missed.to_csv('test_cn_missed.tsv', sep='\t')

    # logging.info(stats1)
    # logging.info(stats2)

    # training_data.to_csv(features_filename, index=False)


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
    ax.set_xticklabels(['FP w/ PCA features', 'FP w/o PCA features', 'FN w/ PCA features', 'FN w/o PCA features',
                        'FP both methods', 'FP only w/ PCA features', 'FP only w/o PCA features',
                        'FN both methods', 'FN only w/ PCA features', 'FN only w/o PCA features'])
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
    ax[0].set_title('Confusion Matrix\nwith PCA features')

    # subplot for classifier1
    confusion2 = confusion_matrix(yg2, yp2)
    sns.heatmap(confusion2, annot=True, fmt=".0f", cmap=plt.cm.Blues, ax=ax[1])
    ax[1].set_xlabel('Predicted label')
    ax[1].set_ylabel('True label')
    ax[1].set_xticklabels(classes)
    ax[1].set_yticklabels(classes)
    ax[1].set_title('Confusion Matrix\nwithout PCA features')

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

    predictions = api.train_classify(cn_data, metrics_data, align_metrics_data, figures_prefix=figures_prefix,
                        use_rt_features=True, use_pca_features=False)

    predictions.to_csv(predictions_filename, index=False)


if __name__ == '__main__':
    cli()
