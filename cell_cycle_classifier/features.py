import logging
import joblib
import seaborn
import scipy.stats
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.utils import shuffle
from cell_cycle_classifier.rt_profile import add_rt_features
from cell_cycle_classifier.bk_profile import add_uniqe_bk
from cell_cycle_classifier.pca import add_pca_features


all_feature_names = [
    'slope',
    'slope0',
    'slope1',
    'slope2',
    'slope3',
    'correlation',
    'correlation0',
    'correlation1',
    'correlation2',
    'correlation3',
    'percent_duplicate_reads',
    'mean_insert_size',
    'unpaired_mapped_reads',
    'standard_deviation_insert_size',
    'ploidy',
    'breakpoints',
    'r_ratio',
    'r_G1b',
    'r_S4',
    'slope_ratio',
    'slope_G1b',
    'slope_S4',
    'num_unique_bk',
    'norm_bk',
    'PC1',
    'PC2',
    'PC3'
]

core_feature_names = [
    'correlation',
    'correlation0',
    'correlation1',
    'correlation2',
    'correlation3',
    'library_id',
    'pvalue',
    'slope',
    'slope0',
    'slope1',
    'slope2',
    'slope3'
]

align_metrics_columns = [
    'cell_id',
    'unpaired_mapped_reads', 'paired_mapped_reads',
    'unpaired_duplicate_reads', 'paired_duplicate_reads',
    'unmapped_reads', 'percent_duplicate_reads',
    'total_reads', 'total_mapped_reads',
    'total_duplicate_reads', 'total_properly_paired', 'coverage_breadth',
    'coverage_depth', 'median_insert_size', 'mean_insert_size',
    'standard_deviation_insert_size',
]


def subset_by_cell_cycle(cn_data, proportion_s):
    cell_states = cn_data[['cell_id', 'cell_cycle_state']].drop_duplicates()
    state_cell_ids = {'G1': np.array([]), 'S': np.array([])}

    for cell_cycle_state, df in cell_states.groupby('cell_cycle_state'):
        state_cell_ids[cell_cycle_state] = shuffle(df['cell_id'].values.astype(str))

    logging.info('{} S phase cells'.format(len(state_cell_ids['S'])))
    logging.info('{} G1 phase cells'.format(len(state_cell_ids['G1'])))

    num_cells = min(len(state_cell_ids['S']), len(state_cell_ids['G1']))

    cell_ids = (
        list(state_cell_ids['S'][:int(proportion_s * num_cells)]) +
        list(state_cell_ids['G1'][:int((1. - proportion_s) * num_cells)])
    )
    
    return cell_ids


def calculate_features(cn_data, metrics_data, align_metrics_data, agg_proportion_s=None, figures_prefix=None, use_rt_features=True, use_pca_features=True):
    """ Calculate features based on copy number data
    
    Args:
        cn_data (pandas.DataFrame): HMMCopy reads data
        metrics_data (pandas.DataFrame): HMMCopy metrics data
        align_metrics_data (pandas.DataFrame): Alignment metrics data
        agg_proportion_s (float, optional): Proportion of s to use in aggregate correction. Defaults to None, all available.
        figures_prefix (str, optional): Prefix for figures. Defaults to None, no figures.
        use_rt_features (bool, optional): Mark false when replication timing features should be ignored.
        use_pca_features (bool, optional): Mark false when PCA features should be ignored.
    
    Returns:
        pandas.DataFrame: Feature data
    """

    cn_data = cn_data.merge(align_metrics_data[['cell_id', 'median_insert_size']])

    corr_data = []

    for library_id, library_cn_data in cn_data.groupby('library_id'):
        logging.info(f'calculating features for {library_id}')

        library_cn_data = library_cn_data[library_cn_data['gc'] < 1.]
        library_cn_data = library_cn_data[library_cn_data['gc'] > 0.]
        library_cn_data = library_cn_data[library_cn_data['state'] < 9]

        library_cn_data = library_cn_data.merge(
            library_cn_data.groupby('cell_id')['reads'].sum().rename('total_reads').reset_index())
        library_cn_data['norm_reads'] = 1e6 * library_cn_data['reads'] / library_cn_data['total_reads']
        library_cn_data = library_cn_data.query('state > 0').copy()
        library_cn_data['norm_reads'] = library_cn_data['norm_reads'] / library_cn_data['state']

        if len(library_cn_data.index) == 0:
            logging.warning(f'library {library_id} filtered entirely')
            continue

        #
        # Correct GC with aggregate data
        #
        logging.info(f'calculating aggregate features')
        for use_norm_reads in (True, False):
            if use_norm_reads:
                reads_col = 'norm_reads'
            else:
                reads_col = 'reads'

            if agg_proportion_s is not None:
                cell_ids = subset_by_cell_cycle(library_cn_data, agg_proportion_s)
                agg_data = library_cn_data[library_cn_data['cell_id'].isin(cell_ids)]
            else:
                agg_data = library_cn_data
            agg_data = agg_data.groupby(['chr', 'start'])[reads_col].sum().reset_index()
            agg_data = agg_data.merge(cn_data[['chr', 'start', 'gc']].drop_duplicates())

            z = np.polyfit(agg_data['gc'].values, agg_data[reads_col].astype(float).values, 3)
            p = np.poly1d(z)

            if figures_prefix is not None:
                fig = plt.figure(figsize=(3, 3))
                ax = plt.gca()
                seaborn.scatterplot(
                    'gc', reads_col,
                    data=agg_data,
                    alpha=0.01,
                    ax=ax)
                x = np.linspace(agg_data['gc'].min(), agg_data['gc'].max(), 100)
                plt.plot(x, p(x))
                plt.title('agg fit on ' + library_id)
                fig.savefig(figures_prefix + f'{library_id}_norm{use_norm_reads}_agg_fit.pdf')

            library_cn_data['copy2_{}'.format(use_norm_reads * 1)] = library_cn_data[reads_col] / p(library_cn_data['gc'].values)

        #
        # Correct GC with per cell data
        #
        logging.info(f'calculating independent features')
        for use_insert_size in (True, False):
            if agg_proportion_s is not None:
                cell_ids = subset_by_cell_cycle(library_cn_data, agg_proportion_s)
                agg_data = library_cn_data[library_cn_data['cell_id'].isin(cell_ids)]
            else:
                agg_data = library_cn_data

            if use_insert_size:
                X = agg_data[['gc', 'median_insert_size']].values
            else:
                X = agg_data[['gc']].values

            y = agg_data['norm_reads']

            poly = PolynomialFeatures(3)
            X_poly = poly.fit_transform(X)

            reg = LinearRegression().fit(X_poly, y)

            logging.info(
                'Library {}, accuracy of Logistic regression classifier on training set: {:.4f}'
                .format(library_id, reg.score(X_poly, y)))

            if use_insert_size:
                X = library_cn_data[['gc', 'median_insert_size']].values
            else:
                X = library_cn_data[['gc']].values

            X_poly = poly.fit_transform(X)

            corrected_column = 'copy3_{}'.format(use_insert_size * 1)
            library_cn_data[corrected_column] = library_cn_data['norm_reads'] / reg.predict(X_poly)

            cell_id = library_cn_data.sort_values('total_reads')['cell_id'].iloc[0]
            plot_data = library_cn_data.query('cell_id == "{}"'.format(cell_id))

            median_insert_size = plot_data['median_insert_size'].values[0]
            if 'cell_cycle_state' in plot_data:
                cell_cycle_state = plot_data['cell_cycle_state'].values[0]
            else:
                cell_cycle_state = 'unknown'

            x = np.linspace(plot_data['gc'].min(), plot_data['gc'].max(), 100)

            if use_insert_size:
                x = np.array([x, median_insert_size * np.ones(x.shape)]).T
            else:
                x = np.array([x]).T

            if figures_prefix is not None:
                fig = plt.figure(figsize=(6, 6))
                ax = plt.gca()
                seaborn.scatterplot(
                    'gc', 'norm_reads',
                    data=plot_data,
                    alpha=0.1,
                    ax=ax)
                plt.plot(x[:, 0], reg.predict(poly.fit_transform(x)))
                plt.title('gc norm reads ' + corrected_column + ' ' + cell_id + ' ' + cell_cycle_state)
                fig.savefig(figures_prefix + f'{library_id}_useinsert{use_insert_size}_gc_norm_reads.pdf')

            if figures_prefix is not None:
                fig = plt.figure(figsize=(6, 6))
                ax = plt.gca()
                seaborn.scatterplot(
                    'gc', corrected_column,
                    data=plot_data,
                    alpha=0.1,
                    ax=ax)
                plt.title('gc corrected ' + corrected_column + ' ' + cell_id + ' ' + cell_cycle_state)
                fig.savefig(figures_prefix + f'{library_id}_useinsert{use_insert_size}_gc_corrected.pdf')

        if use_rt_features:
            library_cn_data = add_uniqe_bk(library_cn_data)
            rt, library_cn_data, filtered_mat = add_rt_features(library_cn_data)

        if use_pca_features and use_rt_features:
            library_cn_data = add_pca_features(library_cn_data, rt=rt)

        if use_pca_features and not use_rt_features:
            library_cn_data = add_pca_features(library_cn_data)

        logging.info(f'statistical tests and tabulation')
        library_corr_data = []
        for cell_id, cell_data in library_cn_data.groupby('cell_id'):
            if cell_data.empty:
                continue
            correlation0, pvalue = scipy.stats.spearmanr(cell_data['gc'], cell_data['copy2_0'])
            correlation1, pvalue = scipy.stats.spearmanr(cell_data['gc'], cell_data['copy2_1'])
            correlation2, pvalue = scipy.stats.spearmanr(cell_data['gc'], cell_data['copy3_0'])
            correlation3, pvalue = scipy.stats.spearmanr(cell_data['gc'], cell_data['copy3_1'])
            correlation, pvalue = scipy.stats.spearmanr(cell_data['gc'], cell_data['norm_reads'])
            slope0 = np.polyfit(cell_data['gc'].values, cell_data['copy2_0'].values, 1)[1]
            slope1 = np.polyfit(cell_data['gc'].values, cell_data['copy2_1'].values, 1)[1]
            slope3 = np.polyfit(cell_data['gc'].values, cell_data['copy3_0'].values, 1)[1]
            slope2 = np.polyfit(cell_data['gc'].values, cell_data['copy3_1'].values, 1)[1]
            slope = np.polyfit(cell_data['gc'].values, cell_data['norm_reads'].values, 1)[1]
            temp_dict = dict(
                    correlation=correlation,
                    correlation0=correlation0,
                    correlation1=correlation1,
                    correlation2=correlation2,
                    correlation3=correlation3,
                    pvalue=pvalue,
                    cell_id=cell_id,
                    slope0=slope0,
                    slope1=slope1,
                    slope2=slope2,
                    slope3=slope3,
                    slope=slope)
            if use_rt_features:
                temp_dict['r_ratio'] = cell_data['r_ratio'].values[0]
                temp_dict['r_G1b'] = cell_data['r_G1b'].values[0]  # all values should be same for the cell
                temp_dict['r_S4'] = cell_data['r_S4'].values[0]  # all values should be same for the cell
                temp_dict['slope_ratio'] = cell_data['slope_ratio'].values[0]
                temp_dict['slope_G1b'] = cell_data['slope_G1b'].values[0]
                temp_dict['slope_S4'] = cell_data['slope_S4'].values[0]
                temp_dict['num_unique_bk'] = cell_data['num_unique_bk'].values[0]
            if use_pca_features:
                temp_dict['PC1'] = cell_data['PC1'].values[0]
                temp_dict['PC2'] = cell_data['PC2'].values[0]
                temp_dict['PC3'] = cell_data['PC3'].values[0]
            library_corr_data.append(temp_dict)

        library_corr_data = pd.DataFrame(library_corr_data)
        library_corr_data['library_id'] = library_id

        corr_data.append(library_corr_data)

        if figures_prefix is not None:
            fig = plt.figure(figsize=(6, 6))
            library_corr_data['correlation'].hist(bins=100)
            plt.title('correlation hist ' + library_id)
            fig.savefig(figures_prefix + f'{library_id}_correlation_hist.pdf')

        plt.close('all')

    ploidy = cn_data.groupby('cell_id')['state'].mean().rename('ploidy').reset_index()

    if not corr_data:
        corr_data = pd.DataFrame(index=align_metrics_data.index, columns=core_feature_names)
        corr_data = corr_data.fillna(0)
        corr_data['cell_id'] = align_metrics_data.cell_id
    else:
        corr_data = pd.concat(corr_data, sort=True, ignore_index=True)
        corr_data = corr_data.dropna()

    corr_data = corr_data.merge(align_metrics_data[align_metrics_columns].drop_duplicates())

    corr_data = corr_data.merge(metrics_data[['cell_id', 'breakpoints']].drop_duplicates())
    if 'cell_cycle_state' in metrics_data:
        corr_data = corr_data.merge(metrics_data[['cell_id', 'cell_cycle_state']].drop_duplicates())
    corr_data = corr_data.merge(ploidy)

    # normalize the breakpoint count per library
    if use_rt_features:
        corr_data['norm_bk'] = None
        for library_id, library_cn_data in corr_data.groupby('library_id'):
            mean_bk = library_cn_data.breakpoints.mean() + np.finfo(float).eps
            corr_data.loc[library_cn_data.index, 'norm_bk'] = library_cn_data.breakpoints / mean_bk
    
    return corr_data


cn_data_urls = [
    'SC-1563/results/results/hmmcopy_autoploidy/A90553C_multiplier0_reads.csv.gz',
    'SC-1561/results/results/hmmcopy_autoploidy/A73044A_multiplier0_reads.csv.gz',
    'SC-1583/results/results/hmmcopy_autoploidy/A96139A_multiplier0_reads.csv.gz',
    'SC-1585/results/results/hmmcopy_autoploidy/A96147A_multiplier0_reads.csv.gz',
]

metrics_data_urls = [
    'SC-1563/results/results/hmmcopy_autoploidy/A90553C_multiplier0_metrics.csv.gz',
    'SC-1561/results/results/hmmcopy_autoploidy/A73044A_multiplier0_metrics.csv.gz',
    'SC-1583/results/results/hmmcopy_autoploidy/A96139A_multiplier0_metrics.csv.gz',
    'SC-1585/results/results/hmmcopy_autoploidy/A96147A_multiplier0_metrics.csv.gz',
]

align_metrics_data_urls = [
    'SC-1563/results/results/alignment/A90553C_alignment_metrics.csv.gz',
    'SC-1561/results/results/alignment/A73044A_alignment_metrics.csv.gz',
    'SC-1583/results/results/alignment/A96139A_alignment_metrics.csv.gz',
    'SC-1585/results/results/alignment/A96147A_alignment_metrics.csv.gz',
]

cache_dir = './cachedir'

memory = joblib.Memory(cache_dir, verbose=10)

@memory.cache
def get_data(prefix, sas, curated_labels=False):
    cn_data = []
    for cn_data_url in cn_data_urls:
        url = prefix + cn_data_url + sas
        logging.info(f'reading from {url}')
        cn_data.append(pd.read_csv(url, compression='gzip'))
    cn_data = pd.concat(cn_data, sort=True, ignore_index=True)

    metrics_data = []
    for metrics_data_url in metrics_data_urls:
        url = prefix + metrics_data_url + sas
        logging.info(f'reading from {url}')
        metrics_data.append(pd.read_csv(url, compression='gzip'))
    metrics_data = pd.concat(metrics_data, sort=True, ignore_index=True)

    align_metrics_data = []
    for align_metrics_data_url in align_metrics_data_urls:
        url = prefix + align_metrics_data_url + sas
        logging.info(f'reading from {url}')
        align_metrics_data.append(pd.read_csv(url, compression='gzip'))
    align_metrics_data = pd.concat(align_metrics_data, sort=True, ignore_index=True)

    for data in (cn_data, metrics_data, align_metrics_data):
        data['sample_id'] = [a.split('-')[0] for a in data['cell_id']]
        data['library_id'] = [a.split('-')[1] for a in data['cell_id']]

    # Fix total_mapped_reads_hmmcopy column
    fix_read_count = metrics_data['total_mapped_reads_hmmcopy'].isnull()
    metrics_data.loc[fix_read_count, 'total_mapped_reads_hmmcopy'] = (
        metrics_data.loc[fix_read_count, 'total_mapped_reads'])

    metrics_data = metrics_data.query('total_mapped_reads_hmmcopy > 500000')

    logging.info('library sizes:\n{}'.format(metrics_data.groupby('library_id').size()))

    cn_data = cn_data.merge(metrics_data[['cell_id']].drop_duplicates())
    cn_data = cn_data.merge(metrics_data[['cell_id', 'experimental_condition', 'log_likelihood', 'MBRSM_dispersion',
                                        'MBRSI_dispersion_non_integerness', 'mean_state_mads', 'mad_hmmcopy']])

    # Remap experimental conditions and filter
    conditions = {
        'A': 'G1',
        'A-BSA': 'G1',
        'A-NCC': 'G1',
        'B': 'S',
        'B-NCC': 'S',
        'C': 'G2',
        'C-NCC': 'G2',
        'G1': 'G1',
        'G2': 'G2',
        'S': 'S',
        'D': 'D',
    }

    conditions = pd.Series(conditions)
    conditions.index.name = 'experimental_condition'
    conditions.name = 'cell_cycle_state'
    conditions = conditions.reset_index()

    metrics_data = metrics_data.merge(conditions)
    cn_data = cn_data.merge(conditions)

    metrics_data = metrics_data.query('cell_cycle_state != "D"')
    cn_data = cn_data.query('cell_cycle_state != "D"')

    # have hand-curated labels replace the flow-determined cell cycle state
    if curated_labels is not None:
        curated_df = pd.read_csv(curated_labels)

        # merge new_cell_cycle_state into metrics data
        metrics_data = pd.merge(metrics_data, curated_df, how='left', on=['cell_id', 'cell_cycle_state'])

        # copy flow-determined state for cells without new_cell_cycle_state
        for key, row in metrics_data.iterrows():
            if pd.isnull(row['new_cell_cycle_state']):
                metrics_data.loc[key, 'new_cell_cycle_state'] = metrics_data.loc[key, 'cell_cycle_state']
                metrics_data.loc[key, 'checked'] = False
                metrics_data.loc[key, 'corrected'] = False

        # merge assignments into cn_data
        full_curated_df = metrics_data[['cell_id', 'new_cell_cycle_state', 'checked', 'corrected']]
        cn_data = pd.merge(cn_data, full_curated_df, how='left', on='cell_id')

        # assign new_cell_cycle_state to cell_cycle_state
        cn_data.cell_cycle_state = cn_data.new_cell_cycle_state
        metrics_data.cell_cycle_state = metrics_data.new_cell_cycle_state
        cn_data.drop(columns=['new_cell_cycle_state'], inplace=True)
        metrics_data.drop(columns=['new_cell_cycle_state'], inplace=True)

    return cn_data, metrics_data, align_metrics_data


def get_features(
        training_url_prefix,
        shared_access_signature='',
        figures_prefix=None,
        feature_names=None,
        proportion_s_train=None,
        proportion_s_test=None,
        random_seed=None,
        use_rt_features=True,
        use_pca_features=True,
        curated_labels=None
    ):
    """ Train and test the model given annotated input copy number data.
    
    Args:
        shared_access_signature (str): Shared access signature for accessing training data.
        figures_prefix (str, optional): Prefix for figure filenames. Defaults to None.
        feature_names (list of str, optional): Subset of features. Defaults to None, all features.
        proportion_s_train (float, optional): Proportion of s-phase used in aggregate correction. Defaults to None.
        proportion_s_test (float, optional): Proportion of s-phase used in aggregate correction for testing. Defaults to None.
        random_seed (int, optional): Random seed for selecting test set. Defaults to None.
        use_rt_features (bool, optional): Mark false when replication timing features should be ignored.
        use_pca_features (bool, optional): Mark false when PCA features should be ignored.

    Returns:
        [type]: [description]
    """

    cn_data, metrics_data, align_metrics_data = get_data(training_url_prefix, shared_access_signature, curated_labels)

    np.random.seed(random_seed)

    cell_ids = cn_data['cell_id'].unique().astype(str)
    cell_ids = shuffle(cell_ids)

    split_idx = int(0.75 * len(cell_ids))
    training_cell_ids = cell_ids[:split_idx]
    test_cell_ids = cell_ids[split_idx:]

    if feature_names is None:
        feature_names = all_feature_names

    # remove rt or pca feature names if necessary
    rt_features = ['r_ratio', 'r_G1b', 'r_S4', 'slope_ratio',
                'slope_G1b', 'slope_S4', 'num_unique_bk', 'norm_bk']
    pca_features = ['PC1', 'PC2', 'PC3']
    if use_rt_features is False and set(rt_features).issubset(set(feature_names)):
        feature_names = [x for x in feature_names if x not in rt_features]
    if use_pca_features is False and set(pca_features).issubset(set(feature_names)):
        feature_names = [x for x in feature_names if x not in pca_features]

    # Training features
    #
    training_figures_prefix = None
    if figures_prefix is not None:
        training_figures_prefix = figures_prefix + 'training_'
    logging.info('calculating features')
    training_data = calculate_features(
        cn_data[cn_data['cell_id'].isin(training_cell_ids)],
        metrics_data,
        align_metrics_data,
        agg_proportion_s=proportion_s_train,
        figures_prefix=training_figures_prefix,
        use_rt_features=use_rt_features,
        use_pca_features=use_pca_features
    )

    training_data = training_data.query('cell_cycle_state != "D"')
    training_data = training_data.query('ploidy < 6')

    # Testing features
    # 
    logging.info('calculating test features')
    testing_data = calculate_features(
        cn_data[cn_data['cell_id'].isin(test_cell_ids)],
        metrics_data,
        align_metrics_data,
        agg_proportion_s=proportion_s_test,
        use_rt_features=use_rt_features,
        use_pca_features=use_pca_features
    )

    testing_data = testing_data.query('cell_cycle_state != "D"')
    logging.info(testing_data.groupby('cell_cycle_state').size())

    # Plot
    # 
    if figures_prefix is not None:
        logging.info('plotting')
        g = seaborn.catplot(
            y='breakpoints',
            x='cell_cycle_state',
            data=testing_data,
            kind='violin',
            height=4,
        )
        g.fig.dpi = 150
        g.axes[0][0].set_xlabel('Cell cycle state')
        g.fig.savefig(figures_prefix + 'breakpoints.pdf', bbox_inches='tight')

        g = seaborn.catplot(
            y='correlation0',
            x='cell_cycle_state',
            data=testing_data,
            kind='violin',
            height=4,
        )
        g.fig.dpi = 150
        g.axes[0][0].set_xlabel('Cell cycle state')
        g.fig.savefig(figures_prefix + 'correlation0.pdf', bbox_inches='tight')

    training_data['training_context'] = 'training'
    testing_data['training_context'] = 'holdout'

    feature_data = pd.concat([training_data, testing_data], sort=True, ignore_index=True)

    return feature_data
