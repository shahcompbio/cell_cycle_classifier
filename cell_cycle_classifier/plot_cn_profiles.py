import pandas as pd
import os
import matplotlib
if os.environ.get('DISPLAY','') == '':
	print('no display found. Using non-interactive Agg backend')
	matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scgenome.cnplot

if __name__ == '__main__':
	df = pd.read_csv('probable_flow_errs.tsv', sep='\t', index_col=False)

	df['chr'] = df['chr'].astype('category')
	df['state'] = df['state'].astype('category')

	num_cells = len(df.cell_id.unique())
	print('num_cells', num_cells)
	fig, ax = plt.subplots(num_cells, 1, figsize=(16, 4*num_cells))
	i = 0
	for cell_id, plot_data in df.groupby('cell_id'):
		print(cell_id)
		ax[i].set_title(str(cell_id) + '\nG1b corr: ' + str(plot_data['r_G1b'].values[0]) + '\nS4 corr: ' + str(plot_data['r_S4'].values[0]))
		_ = scgenome.cnplot.plot_cell_cn_profile(
			ax[i],
			plot_data,
			'copy',
			'state',
		)
		i += 1

	fig.savefig('probable_flow_errs.pdf', bbox_inches='tight')