import pandas as pd
import numpy as np
import os
import matplotlib
if os.environ.get('DISPLAY','') == '':
	print('no display found. Using non-interactive Agg backend')
	matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scgenome.cnplot


def plot_cn_pdf(df, output_path, max_cells=np.inf):
	df.loc[:, 'chr'] = df['chr'].astype('category')
	df.loc[:, 'state'] = df['state'].astype('category')
	df.loc[:, 'r_S4'] = df['r_S4'].astype('float')
	df.loc[:, 'r_G1b'] = df['r_G1b'].astype('float')
	num_cells = min(len(df.cell_id.unique()), max_cells)
	fig, ax = plt.subplots(num_cells, 1, figsize=(16, 4*num_cells))
	i = 0
	for cell_id, plot_data in df.groupby('cell_id'):
		print(i)
		ax[i].set_title(str(cell_id) + '\nflow: ' + str(plot_data['cell_cycle_state'].values[0]) + \
			'model s_prob: ' + str(round(plot_data['is_s_phase_prob'].values[0], 3)) + \
			'\nG1b corr: ' + str(round(plot_data['r_G1b'].values[0], 3)) + \
			', S4 corr: ' + str(round(plot_data['r_S4'].values[0], 3)))
		_ = scgenome.cnplot.plot_cell_cn_profile(
			ax[i],
			plot_data,
			'copy',
			'state',
		)
		i += 1
		if i >= max_cells:
			break

	fig.savefig(output_path)


def probable_flowS_FP(df):
	df = df.query("cell_cycle_state == 'S' & r_G1b < 0 & r_S4 < 0")
	plot_cn_pdf(df, 'probable_flowS_FP.pdf')


def probable_flowS_FN(df):
	df = df.query("cell_cycle_state != 'S' & is_s_phase == True")
	print(df.shape)
	plot_cn_pdf(df, 'probable_flowS_FN.pdf', max_cells=50)


def confident_s_phase(df):
	df = df.query("cell_cycle_state == 'S' & r_G1b > 0.2 & r_S4 > 0.2")
	plot_cn_pdf(df, 'confident_s_phase.pdf', max_cells=50)


def confident_G1_phase(df):
	df = df.query("cell_cycle_state == 'G1' & r_G1b < 0. & r_S4 < 0.")
	plot_cn_pdf(df, 'confident_G1_phase.pdf', max_cells=50)


def confident_G2_phase(df):
	df = df.query("cell_cycle_state == 'G2' & r_G1b < 0. & r_S4 < 0.")
	plot_cn_pdf(df, 'confident_G2_phase.pdf', max_cells=50)


if __name__ == '__main__':
	df = pd.read_csv('testing_data1.tsv', sep='\t', index_col=False)
	probable_flowS_FP(df)
	probable_flowS_FN(df)
	confident_s_phase(df)
	confident_G1_phase(df)
	confident_G2_phase(df)
