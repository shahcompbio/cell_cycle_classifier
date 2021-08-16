import pandas as pd


def generate_bin_edge_ids(chrom, chrom_cns):
    starts = chrom_cns['start'].iloc[:-1].astype(str).values
    ends = chrom_cns['end'].iloc[:-1].astype(str).values
    bin_edge_ids = chrom + '_' + starts + '_' + ends

    return bin_edge_ids


def convert_cn_to_breakpoints(cn_data):
	# print('in convert_cn_to_breakpoints...')
	# print('original number of cells', cn_data.cell_id.drop_duplicates().shape)
	cn = cn_data[['chr', 'start', 'end', 'cell_id', 'state']]

	cn.sort_values(by=['cell_id', 'chr', 'start', 'end'], inplace=True)

	bk = pd.DataFrame()
	# i = 0
	for cell, group1 in cn.groupby('cell_id'):
		pieces = []
		for chrom, group in group1.groupby('chr'):
			bin_edge_ids = generate_bin_edge_ids(chrom, group)
			bin_gap = (
				group['start'].iloc[1:].values - group['end'].iloc[:-1].values
			)

			group.set_index(['cell_id', 'chr', 'start', 'end'], inplace=True)

			chrom_brkpt = (group.diff().iloc[1:] != 0).astype(int)
			chrom_brkpt['loci'] = bin_edge_ids
			chrom_brkpt.set_index('loci', inplace=True)
			#  chrom_brkpt = chrom_brkpt[bin_gap == 1]
			pieces.append(chrom_brkpt)

		cell_profile = pd.concat(pieces)
		cell_profile.rename(columns={'state': cell}, inplace=True)
		# print(cell_profile)
		if bk.empty:
			# print('bk is empty')
			bk = cell_profile
		else:
			bk = pd.merge(bk, cell_profile, on='loci', how='outer').fillna(0)
		# print('i:', i, 'cell:', cell, 'bk.shape:', bk.shape)
		# i += 1
		# if i>5:
		# 	break

	return bk


def get_unique_breakpoints(bk):
	bk = bk.T  # transpose so that cells are rows instead of columns

	num_cells = len(bk.index)

	# add row to represent frequency of each breakpoint across all cells in the dataset
	bk.loc['freq'] = bk.sum() / num_cells

	# choose threshold of 1% breakpoint frequency or that the breakpoint only occurs in one cell (if <100 cells present)
	thresh = max(0.01, 1/num_cells)

	# iterate through bins and remove all breakpoints if frequency at that bin is above the threshold
	for label, content in bk.items():
		if bk.loc['freq', label] > thresh:
			bk[label] = 0

	# find number of unique breakpoints per cell
	bk['num_unique_bk'] = bk.sum(axis=1)
	bk = bk.reset_index().rename(columns={'index': 'cell_id'})
	bk = bk[['cell_id', 'num_unique_bk']]

	return bk


def add_uniqe_bk(library_cn_data):
	# print('\nin add_unique_bk()...')
	# print('library_cn_data.shape', library_cn_data.shape)
	bk = convert_cn_to_breakpoints(library_cn_data)
	# print('library_cn_data.shape', library_cn_data.shape)
	# print('bk.shape', bk.shape)
	bk = get_unique_breakpoints(bk)
	# print('bk.shape', bk.shape)
	# print(bk.head())

	library_cn_data = pd.merge(library_cn_data, bk, on='cell_id')
	# print('library_cn_data.shape', library_cn_data.shape)
	# print('leaving add_unique_bk()...\n')
	return library_cn_data