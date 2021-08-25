import logging
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
			logging.info('failed to extract intensities from bigWig')
			mat.loc[key, 'rep_argmax'] = np.nan
			mat.loc[key, 'rep_ratio'] = np.nan
			mat.loc[key, 'rep_G1b'] = np.nan
			mat.loc[key, 'rep_S1'] = np.nan
			mat.loc[key, 'rep_S2'] = np.nan
			mat.loc[key, 'rep_S3'] = np.nan
			mat.loc[key, 'rep_S4'] = np.nan

		if (G1b_intensity is None) or (S1_intensity is None) or (S2_intensity is None) or (S3_intensity is None) or (S4_intensity is None) or (G2_intensity is None):
			logging.info('one of the rt intensities was None')
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

	# only return replication timing annotations in rt
	rt = mat[['rep_argmax', 'rep_ratio', 'rep_G1b', 'rep_S1', 'rep_S2', 'rep_S3', 'rep_S4']]
	rt.dropna(inplace=True)

	return rt


def calc_rt_features(rt, mat):
	""" Find correlation and slope between rt_profiles and norm_reads for each cell. """
	df = pd.DataFrame(columns = ['cell_id', 'r_ratio', 'pval_ratio',
						'r_G1b', 'pval_G1b', 'r_S4', 'pval_S4',
						'slope_ratio', 'slope_G1b', 'slope_S4'])
	df.set_index('cell_id', inplace=True)
	for cell_name, cell_data in mat.iteritems():
		# removing missing values for this cell
		cell_data = cell_data.copy().dropna()
		temp_rt = rt.loc[cell_data.index]

		try:
			# find correlation between cell copy and rt features
			r_ratio, pval_ratio = pearsonr(cell_data, temp_rt['rep_ratio'])
			r_G1b, pval_G1b = pearsonr(cell_data, temp_rt['rep_G1b'])
			r_S4, pval_S4 = pearsonr(cell_data, temp_rt['rep_S4'])

			# find slope between cell copy and rt features
			slope_ratio = np.polyfit(cell_data.astype('float').values, temp_rt['rep_ratio'].astype('float').values, 1)[1]
			slope_G1b = np.polyfit(cell_data.astype('float').values, temp_rt['rep_G1b'].astype('float').values, 1)[1]
			slope_S4 = np.polyfit(cell_data.astype('float').values, temp_rt['rep_S4'].astype('float').values, 1)[1]
		
			temp = [r_ratio, pval_ratio, r_G1b, pval_G1b,
					r_S4, pval_S4, slope_ratio, slope_G1b, slope_S4]
		except:
			temp = [np.nan] * 9
		
		df.loc[cell_name] = temp

	return df


def add_rt_features(library_cn_data):
	""" Takes in library_cn_data and adds replication timing correlations for each cell. """
	logging.info('in add_rt_features()...')
	mat = get_norm_reads_mat(library_cn_data)
	rt = get_rt_annotation(mat)
	df = calc_rt_features(rt, mat)
	library_cn_data = pd.merge(library_cn_data, df, on='cell_id')
	return rt, library_cn_data, mat

