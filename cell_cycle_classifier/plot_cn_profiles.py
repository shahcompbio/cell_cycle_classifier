import pandas as pd
import os
import matplotlib
if os.environ.get('DISPLAY','') == '':
	print('no display found. Using non-interactive Agg backend')
	matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scgenome.cnplot


def probable_flow_errors():
	df = pd.read_csv('probable_flow_errs.tsv', sep='\t', index_col=False)

	df['chr'] = df['chr'].astype('category')
	df['state'] = df['state'].astype('category')

	num_cells = len(df.cell_id.unique())
	print('num_cells', num_cells)
	fig, ax = plt.subplots(num_cells, 1, figsize=(16, 4*num_cells))
	i = 0
	for cell_id, plot_data in df.groupby('cell_id'):
		print(cell_id)
		ax[i].set_title(str(cell_id) +
			'\nG1b corr: ' + str(round(plot_data['r_G1b'].values[0], 3)) +
			', S4 corr: ' + str(round(plot_data['r_S4'].values[0], 3)))
		_ = scgenome.cnplot.plot_cell_cn_profile(
			ax[i],
			plot_data,
			'copy',
			'state',
		)
		i += 1

	fig.savefig('probable_flow_errs.pdf', bbox_inches='tight')


def confident_s_phase():
	df = pd.read_csv('testing_data1.tsv', sep='\t', index_col=False)

	df = df.query("cell_cycle_state == 'S' & r_G1b > 0.2 & r_S4 > 0.2")

	df['chr'] = df['chr'].astype('category')
	df['state'] = df['state'].astype('category')

	num_cells = len(df.cell_id.unique())
	print('num_cells', num_cells)
	fig, ax = plt.subplots(num_cells, 1, figsize=(16, 4*num_cells))
	i = 0
	for cell_id, plot_data in df.groupby('cell_id'):
		print(cell_id)
		ax[i].set_title(str(cell_id) + 
			'\nG1b corr: ' + str(round(plot_data['r_G1b'].values[0], 3)) + 
			', S4 corr: ' + str(round(plot_data['r_S4'].values[0], 3)))
		_ = scgenome.cnplot.plot_cell_cn_profile(
			ax[i],
			plot_data,
			'copy',
			'state',
		)
		i += 1

	fig.savefig('confident_s_phase.pdf', bbox_inches='tight')

if __name__ == '__main__':
	probable_flow_errors()
	confident_s_phase()
