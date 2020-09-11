import pyBigWig
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from sklearn.decomposition import PCA

def generate_bin_edge_id(chrom, start, end):
	bin_edge_id = str(chrom) + '_' + str(start) + '_' + str(end)
	return bin_edge_id


def add_chr_bin_from_loci(loci):
	[chrom, start, end] = loci.split('_')
	if chrom == 'X':
		chrom = '30'
	if chrom == 'Y':
		chrom = '40'
	return int(chrom), int(start), int(end)


def get_norm_reads_mat(cn):
	""" Turn cn data into matrix where rows are loci and columns are cells """
	cn['loci'] = cn.apply(lambda row : generate_bin_edge_id(row['chr'], row['start'], row['end']), axis = 1)

	mat = cn.pivot(index='loci', columns='cell_id', values='copy2_1')

	mat.reset_index(inplace=True)

	mat['chr'], mat['start'], mat['end'] = zip(*mat['loci'].apply(add_chr_bin_from_loci))

	mat.sort_values(by=['chr', 'start', 'end'], inplace=True)

	mat.drop(columns=['chr', 'start', 'end'], inplace=True)

	# remove Y chromosome
	mat = mat[~mat.loci.str.contains('Y')]

	mat.set_index('loci', inplace=True)
	mat = mat.T

	return mat


def run_pca(cn, n_components):
	pca = PCA(n_components=n_components)
	transformed = pca.fit_transform(cn.values)
	return transformed, pca.components_


def rt_correlation(rt, components):
	""" Find correlation between rep_ratio and each PC's loadings vector. """
	corrs = []
	for i, loadings_vector in components.iteritems():
		print('loadings_vector', loadings_vector.shape)
		print(loadings_vector.head())
		print('loadings num null', loadings_vector.isnull().sum())
		print('rt num null', rt['rep_ratio'].isnull().sum())

		# remove loci that have NA or inf in loadings
		loadings_vector.replace([np.inf, -np.inf], np.nan, inplace=True)
		mask = loadings_vector.isnull()
		temp_loadings = loadings_vector.loc[~mask]
		temp_rt = rt.loc[~mask, 'rep_ratio']

		# remove loci that have NA or inf in rt
		temp_rt.replace([np.inf, -np.inf], np.nan, inplace=True)
		mask = temp_rt.isnull()
		temp_rt = temp_rt.loc[~mask]
		temp_loadings = temp_loadings.loc[~mask]

		print('temp_loadings shape', temp_loadings.shape)
		print('temp_rt', temp_rt.shape)
		print(temp_rt.head())

		temp_loadings.to_csv('temp_loadings.tsv', sep='\t')
		temp_rt.to_csv('temp_rt.tsv', sep='\t')

		r_ratio, pval_ratio = pearsonr(temp_loadings, temp_rt)
		corrs.append(r_ratio)

	return corrs


def sort_PCs_by_rt_correlation(rt, components, transformed, num_pcs, mat):
	corrs = rt_correlation(rt, components)
	print('corrs\n', corrs)
	abs_corrs = [abs(x) for x in corrs]
	print('abs_corrs\n', abs_corrs)
	flip_scores = [True if x<0 else False for x in corrs]
	print('flip_scores\n', flip_scores)
	res = sorted(range(len(abs_corrs)), key = lambda x: abs_corrs[x], reverse=True)[:num_pcs]
	print('res\n', res)

	trans_df = pd.DataFrame(transformed, index=mat.index)
	trans_df = trans_df.reset_index().rename(columns={'index': 'cell_id'})
	print('trans_df\n', trans_df.shape, '\n', trans_df.head())
	print(trans_df['cell_id'].head())
	print(trans_df[0].head())

	pca_data = pd.DataFrame()
	pca_data['cell_id'] = trans_df['cell_id']
	for n, idx in enumerate(res):
		if flip_scores[idx]:
			pca_data[f'PC{n+1}'] = -1*trans_df[idx]
		else:
			pca_data[f'PC{n+1}'] = trans_df[idx]

	return pca_data



def add_pca_features(library_cn_data, num_pcs=3, rt=None):
	""" Takes in library_cn_data and adds pca features for each cell. """
	mat = get_norm_reads_mat(library_cn_data)
	print('mat\n', mat.shape, '\n', mat.head())

	# filter out null
	num_null = mat.isnull().sum(axis=1)
	mat = mat[num_null <= 800]
	mat = mat.dropna(axis='columns')
	print('after filtering mat', mat.shape)


	# run pca to get scores and loadings
	transformed, components = run_pca(mat, 10)
	print('transformed shape', transformed.shape)
	print('components shape', components.shape)

	# store components as df with genomic position as the index
	components = pd.DataFrame(components.T, index=mat.columns)
	print('components df\n', components.shape, '\n', components.head())

	if rt is not None:
		# any genomic loci dropped should also be removed from rt
		print('rt before filtering', rt.shape)
		rt = rt.loc[mat.columns, :]
		print('rt after filtering', rt.shape)
		# re-order PCs so that they are ranked by the replication timing correlation
		pca_data = sort_PCs_by_rt_correlation(rt, components, transformed, num_pcs, mat)
	# keep PCs in their original order
	else:
		pca_data = pd.DataFrame(transformed, index=mat.index)
		pca_data = pca_data.iloc[:, :num_pcs]
		pca_data.columns = [f'PC{n+1}' for n in range(len(pca_data.columns))]
		pca_data = pca_data.reset_index().rename(columns={'index': 'cell_id'})

	print('pca_data\n', pca_data.shape, '\n', pca_data.head())

	library_cn_data = pd.merge(library_cn_data, pca_data, on='cell_id')

	print('library_cn_data\n', library_cn_data.shape, '\n', library_cn_data.columns, '\n', library_cn_data.head())	

	return library_cn_data

