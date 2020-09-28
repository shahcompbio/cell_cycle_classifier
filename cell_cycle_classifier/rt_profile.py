import pyBigWig
import pandas as pd
import numpy as np
from scipy.stats import pearsonr

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

	return mat


def argmax_score(G1b_int, S1_int, S2_int, S3_int, S4_int, G2_int):
	return np.argmax([G2_int, S4_int, S3_int, S2_int, S1_int, G1b_int])


def ratio_score(G1b_int, S1_int, S2_int, S3_int, S4_int, G2_int):
	early = G1b_int + S1_int
	late = S4_int + G2_int
	return np.log2(early/late)


def level_score(G1b_int, S1_int, S2_int, S3_int, S4_int, G2_int):
	G1b_out = G1b_int
	S1_out = G1b_out + S1_int
	S2_out = S1_out + S2_int
	S3_out = S2_out + S3_int
	S4_out = S3_out + S4_int
	G2_out = S4_out + G2_int

	# normalize by G2_out so values sum to 100
	G1b_out = (G1b_out / G2_out) * 100.
	S1_out = (S1_out / G2_out) * 100.
	S2_out = (S2_out / G2_out) * 100.
	S3_out = (S3_out / G2_out) * 100.
	S4_out = (S4_out / G2_out) * 100.

	return G1b_out, S1_out, S2_out, S3_out, S4_out


def get_rt_annotation(mat):
	""" Pull replication timing data for the loci in mat and filter mat accordingly. """
	mat['rep_argmax'] = None
	mat['rep_G1b'] = None
	mat['rep_ratio'] = None

	# TODO: change way that bigwig files get loaded in
	G1b = pyBigWig.open('cell_cycle_classifier/data/replication_timing/wgEncodeUwRepliSeqMcf7G1bPctSignalRep1.bigWig')
	S1 = pyBigWig.open('cell_cycle_classifier/data/replication_timing/wgEncodeUwRepliSeqMcf7S1PctSignalRep1.bigWig')
	S2 = pyBigWig.open('cell_cycle_classifier/data/replication_timing/wgEncodeUwRepliSeqMcf7S2PctSignalRep1.bigWig')
	S3 = pyBigWig.open('cell_cycle_classifier/data/replication_timing/wgEncodeUwRepliSeqMcf7S3PctSignalRep1.bigWig')
	S4 = pyBigWig.open('cell_cycle_classifier/data/replication_timing/wgEncodeUwRepliSeqMcf7S4PctSignalRep1.bigWig')
	G2 = pyBigWig.open('cell_cycle_classifier/data/replication_timing/wgEncodeUwRepliSeqMcf7G2PctSignalRep1.bigWig')

	for key, row in mat.iterrows():
		[chr_num, start, stop] = key.split('_')
		chr_name = 'chr' + str(chr_num)

		try:
			G1b_intensity = G1b.stats(chr_name, int(start), int(stop), exact=True)
			G1b_intensity = G1b_intensity[0]
			S1_intensity = S1.stats(chr_name, int(start), int(stop), exact=True)
			S1_intensity = S1_intensity[0]
			S2_intensity = S2.stats(chr_name, int(start), int(stop), exact=True)
			S2_intensity = S2_intensity[0]
			S3_intensity = S3.stats(chr_name, int(start), int(stop), exact=True)
			S3_intensity = S3_intensity[0]
			S4_intensity = S4.stats(chr_name, int(start), int(stop), exact=True)
			S4_intensity = S4_intensity[0]
			G2_intensity = G2.stats(chr_name, int(start), int(stop), exact=True)
			G2_intensity = G2_intensity[0]
		except:
			print("couldn't find intensity at:")
			print(chr_num, start, stop)
			mat.loc[key, 'rep_argmax'] = np.nan
			mat.loc[key, 'rep_ratio'] = np.nan
			mat.loc[key, 'rep_G1b'] = np.nan
			mat.loc[key, 'rep_S1'] = np.nan
			mat.loc[key, 'rep_S2'] = np.nan
			mat.loc[key, 'rep_S3'] = np.nan
			mat.loc[key, 'rep_S4'] = np.nan

		if (G1b_intensity is None) or (S1_intensity is None) or (S2_intensity is None) or (S3_intensity is None) or (S4_intensity is None) or (G2_intensity is None):
			print("bad key encoutered at:")
			print(chr_num, start, stop)
			mat.loc[key, 'rep_argmax'] = np.nan
			mat.loc[key, 'rep_ratio'] = np.nan
			mat.loc[key, 'rep_G1b'] = np.nan
			mat.loc[key, 'rep_S1'] = np.nan
			mat.loc[key, 'rep_S2'] = np.nan
			mat.loc[key, 'rep_S3'] = np.nan
			mat.loc[key, 'rep_S4'] = np.nan

		else:
			score1 = argmax_score(G1b_intensity, S1_intensity, S2_intensity, S3_intensity, S4_intensity, G2_intensity)
			score2 = ratio_score(G1b_intensity, S1_intensity, S2_intensity, S3_intensity, S4_intensity, G2_intensity)

			G1b_out, S1_out, S2_out, S3_out, S4_out = level_score(G1b_intensity, S1_intensity, S2_intensity, S3_intensity, S4_intensity, G2_intensity)

			mat.loc[key, 'rep_argmax'] = score1
			mat.loc[key, 'rep_ratio'] = score2
			mat.loc[key, 'rep_G1b'] = G1b_out
			mat.loc[key, 'rep_S1'] = S1_out
			mat.loc[key, 'rep_S2'] = S2_out
			mat.loc[key, 'rep_S3'] = S3_out
			mat.loc[key, 'rep_S4'] = S4_out

	# drop all genomic bins that contain NA
	mat.dropna(axis=0, inplace=True)

	# only return replication timing annotations in rt
	rt = mat[['rep_argmax', 'rep_ratio', 'rep_G1b', 'rep_S1', 'rep_S2', 'rep_S3', 'rep_S4']]

	# remove replication timing annotations for filtered version of mat that gets returned
	filtered_mat = mat.drop(columns=['rep_argmax', 'rep_ratio', 'rep_G1b', 'rep_S1', 'rep_S2', 'rep_S3', 'rep_S4'])

	return rt, filtered_mat


def rt_correlation(rt, filtered_mat):
	""" Find correlation between rt_profiles and norm_reads for each cell. """
	df = pd.DataFrame(columns = ['cell_id', 'r_argmax', 'pval_argmax', 'r_ratio', 'pval_ratio',
						'r_G1b', 'pval_G1b', 'r_S1', 'pval_S1', 'r_S2', 'pval_S2',
						'r_S3', 'pval_S3', 'r_S4', 'pval_S4'])
	df.set_index('cell_id', inplace=True)
	for cell_name, cell_data in filtered_mat.iteritems():
		try:
			r_argmax, pval_argmax = pearsonr(cell_data, rt['rep_argmax'])
		except:
			r_argmax, pval_argmax = "NA", "NA"
		try:
			r_ratio, pval_ratio = pearsonr(cell_data, rt['rep_ratio'])
		except:
			r_ratio, pval_ratio = "NA", "NA"
		try:
			r_G1b, pval_G1b = pearsonr(cell_data, rt['rep_G1b'])
		except:
			r_G1b, pval_G1b = "NA", "NA"
		try:
			r_S1, pval_S1 = pearsonr(cell_data, rt['rep_S1'])
		except:
			r_S1, pval_S1 = "NA", "NA"
		try:
			r_S2, pval_S2 = pearsonr(cell_data, rt['rep_S2'])
		except:
			r_S2, pval_S2 = "NA", "NA"
		try:
			r_S3, pval_S3 = pearsonr(cell_data, rt['rep_S3'])
		except:
			r_S3, pval_S3 = "NA", "NA"
		try:
			r_S4, pval_S4 = pearsonr(cell_data, rt['rep_S4'])
		except:
			r_S4, pval_S4 = "NA", "NA"
		temp = [r_argmax, pval_argmax, r_ratio, pval_ratio,
				r_G1b, pval_G1b, r_S1, pval_S1, r_S2, pval_S2,
				r_S3, pval_S3, r_S4, pval_S4]
		df.loc[cell_name] = temp

	return df


def add_rt_features(library_cn_data):
	""" Takes in library_cn_data and adds replication timing correlations for each cell. """
	mat = get_norm_reads_mat(library_cn_data)
	rt, filtered_mat = get_rt_annotation(mat)
	df = rt_correlation(rt, filtered_mat)
	library_cn_data = pd.merge(library_cn_data, df, on='cell_id')
	return rt, library_cn_data, filtered_mat

