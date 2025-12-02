import scanpy as sc
import pandas as pd
import anndata as ad
import numpy as np
from tools import _score_pixels,select_layer_data,gen_new_layer_key
import os


def get_select_b4(b4_coors, region_x_start, region_x_end, region_y_start, region_y_end):
    select_b4 = []
    for idx in range(b4_coors.shape[0]):
        ele = b4_coors[idx]
        if ele[0]>region_x_start and ele[0]<region_x_end and ele[1]>region_y_start and ele[1]<region_y_end:
            select_b4.append(idx)
    return select_b4


def filter_low_qual_b40(b40_slide, b40_result, select_ct_lst):
    salus_bar_names = []
    salus_bar_name_dict = {}
    max_cols = b40_result.idxmax(axis=1)
    max_values = b40_result.max(axis=1)

    # 创建条件掩码：最大值列在目标列表中且值 >= 0.15
    mask = (max_cols.isin(select_ct_lst)) & (max_values >= 0.15)

    # 筛选符合条件的索引并去重
    salus_bar_names = b40_result.index[mask].unique().tolist()

    # 构建字典（如果仍需要）
    salus_bar_name_dict = {idx: 1 for idx in salus_bar_names}

    salus_b40 = b40_slide[salus_bar_names]
    return salus_b40, salus_bar_names


def generate_count_mask_matrix(select_unsplice_adata, occupied_b4_dict,select_genes, region_x_start, region_x_end, region_y_start, region_y_end):

    select_count_matrix = np.zeros((int((region_x_end-region_x_start)/4)+1, int((region_y_end-region_y_start)/4)+1))
    select_mask_matrix = np.zeros((int((region_x_end-region_x_start)/4)+1, int((region_y_end-region_y_start)/4)+1))
    unsplice_count_ls = np.sum(select_unsplice_adata[:,select_genes].X,axis=1)
    coor = select_unsplice_adata.obsm['spatial']
    b4_names = list(select_unsplice_adata.obs_names)
    for idx in range(coor.shape[0]):
        tmp_x = coor[idx][0]
        tmp_y = coor[idx][1]
        idx_x = int((tmp_x-region_x_start)/4)
        idx_y = int((tmp_y-region_y_start)/4)
        select_count_matrix[idx_x,idx_y] = unsplice_count_ls[idx,0]
    for idx in range(coor.shape[0]):
        if b4_names[idx] in occupied_b4_dict:
            tmp_x = coor[idx][0]
            tmp_y = coor[idx][1]
            idx_x = int((tmp_x-region_x_start)/4)
            idx_y = int((tmp_y-region_y_start)/4)
            select_mask_matrix[idx_x,idx_y] = 1
    return select_count_matrix, select_mask_matrix




def generate_bar_matrix(select_b4_adata, x_min, x_max, y_min, y_max):
    bar_matrix_idx_dict = {}
    tmp_lst = []
    for ele in list(select_b4_adata.var_names):
        res = ele.upper()
        tmp_lst.append(res)
    select_b4_adata.var_names = tmp_lst

    coor = select_b4_adata.obsm['spatial']
    x_min = x_min
    x_max = x_max

    y_min = y_min
    y_max = y_max
    default_string = ""
    bar_matrix = []
    for idx in range(int((x_max - x_min) / 4) + 1):
        tmp_lst = []
        for jdx in range(int((y_max - y_min) / 4) + 1):
            tmp_lst.append(default_string)
        bar_matrix.append(tmp_lst)
    # bar_matrix = [[default_string for _ in range(int((x_max-x_min)/4)+1)] for _ in range(int((y_max-y_min)/4)+1)]

    obs_names = list(select_b4_adata.obs_names)
    for idx in range(coor.shape[0]):
        tmp_x = coor[idx][0]
        tmp_y = coor[idx][1]
        idx_x = int((tmp_x - x_min) / 4)
        idx_y = int((tmp_y - y_min) / 4)
        bar_matrix[idx_x][idx_y] = obs_names[idx]
        bar_matrix_idx_dict.update({obs_names[idx]: [idx_x, idx_y]})
    bar_matrix = np.array(bar_matrix)
    return bar_matrix, bar_matrix_idx_dict





def generate_score_matrix(work_path, b4_adata_path, unsplice_b4_adata_path, b40_adata_path, threshold = 0.2, split_num = 2):
    output_path = work_path + 'score_matrix_files/'
    unsplice_adata = sc.read_h5ad(unsplice_b4_adata_path)
    b4_adata = sc.read_h5ad(b4_adata_path)
    
    slide = sc.read_h5ad(b40_adata_path)
    
    gene_info_path = work_path+'mu_gene_expression.csv'

    gene_info = pd.read_csv(gene_info_path, delimiter = ',', header = 0, index_col = 0)
    
    select_genes = list(gene_info.columns)
    
    ct_list = list(gene_info.index)


    if len(set(select_genes)&set(list(unsplice_adata.var_names)))==0:
        tmp_lst = []
        for ele in list(unsplice_adata.var_names):
            res = ele.upper()
            tmp_lst.append(res)
        unsplice_adata.var_names = tmp_lst
        
        tmp_lst = []
        for ele in list(b4_adata.var_names):
            res = ele.upper()
            tmp_lst.append(res)
        b4_adata.var_names = tmp_lst

    cell_type_proportion_path = work_path+'result.csv'

    b40_result = pd.read_csv(cell_type_proportion_path, delimiter = ',', header = 0, index_col = 0)
    b40_slide = slide[list(b40_result.index)]
    b40_slide.obs[list(b40_result.columns)] = b40_result
    b40_slide = ad.AnnData(b40_slide.X, obs=b40_slide.obs, var=b40_slide.var,obsm=b40_slide.obsm)


    '''salus_bar_names = []
    salus_bar_name_dict = {}


    max_cols = b40_result.idxmax(axis=1)
    max_values = b40_result.max(axis=1)

    # 创建条件掩码：最大值列在目标列表中且值 >= 0.15
    mask = (max_cols.isin(ct_list)) & (max_values >= 0.15)

    # 筛选符合条件的索引并去重
    salus_bar_names = b40_result.index[mask].unique().tolist()

    # 构建字典（如果仍需要）
    salus_bar_name_dict = {idx: 1 for idx in salus_bar_names}

    salus_b40 = b40_slide[salus_bar_names]'''
    
    
    
    salus_b40,salus_bar_names = filter_low_qual_b40(b40_slide, b40_result, ct_list)
    

    points = salus_b40.obsm['spatial']
    obs_names = salus_b40.obs_names
    x_min = min(points[:,0])
    x_max = max(points[:,0])
    y_min = min(points[:,1])
    y_max = max(points[:,1])
    region_x_start = x_min
    region_x_end = x_max
    region_y_start = y_min
    region_y_end = y_max



    occupied_b4_dict = {}
    occupied_b4_coor_lst = []
    for bin40 in salus_bar_names:
        x = int(bin40.split('_')[1])
        y = int(bin40.split('_')[2])
        for tmp_x in range(x,x+40,1):
            for tmp_y in range(y,y+40,1):
                occupied_b4_dict.update({'b4_'+str(tmp_x)+'_'+str(tmp_y):1})
                occupied_b4_coor_lst.append([tmp_x,tmp_y])
    b4_coors = unsplice_adata.obsm['spatial']

    
    select_b4 = get_select_b4(b4_coors, region_x_start, region_x_end, region_y_start, region_y_end)

            

    select_unsplice_adata = unsplice_adata[select_b4]


    '''select_count_matrix = np.zeros((int((region_x_end-region_x_start)/4)+1, int((region_y_end-region_y_start)/4)+1))
    select_mask_matrix = np.zeros((int((region_x_end-region_x_start)/4)+1, int((region_y_end-region_y_start)/4)+1))
    unsplice_count_ls = np.sum(select_unsplice_adata[:,select_genes].X,axis=1)
    coor = select_unsplice_adata.obsm['spatial']
    b4_names = list(select_unsplice_adata.obs_names)
    for idx in range(coor.shape[0]):
        tmp_x = coor[idx][0]
        tmp_y = coor[idx][1]
        idx_x = int((tmp_x-x_min)/4)
        idx_y = int((tmp_y-y_min)/4)
        select_count_matrix[idx_x,idx_y] = unsplice_count_ls[idx,0]
    for idx in range(coor.shape[0]):
        if b4_names[idx] in occupied_b4_dict:
            tmp_x = coor[idx][0]
            tmp_y = coor[idx][1]
            idx_x = int((tmp_x-x_min)/4)
            idx_y = int((tmp_y-y_min)/4)
            select_mask_matrix[idx_x,idx_y] = 1'''


    select_count_matrix, select_mask_matrix = generate_count_mask_matrix(select_unsplice_adata, occupied_b4_dict, select_genes, region_x_start, region_x_end, region_y_start, region_y_end)

            
    scores = get_score(select_count_matrix, select_mask_matrix)
            
    b4_coors = b4_adata.obsm['spatial']
    select_b4 = get_select_b4(b4_coors, region_x_start, region_x_end, region_y_start, region_y_end)



    #bar_matrix_idx_dict = {}



    select_b4_adata = b4_adata[select_b4,select_genes]
    sc.pp.filter_cells(select_b4_adata, min_genes=1)

    '''coor = select_b4_adata.obsm['spatial']

    default_string = ""
    bar_matrix = []
    for idx in range(int((region_x_end-region_x_start)/4)+1):
        tmp_lst = []
        for jdx in range(int((region_y_end-region_y_start)/4)+1):
            tmp_lst.append(default_string)
        bar_matrix.append(tmp_lst)
    #bar_matrix = [[default_string for _ in range(int((x_max-x_min)/4)+1)] for _ in range(int((y_max-y_min)/4)+1)]

    b4_bars = list(select_b4_adata.obs_names)
    for idx in range(coor.shape[0]):
        tmp_x = coor[idx][0]
        tmp_y = coor[idx][1]
        idx_x = int((tmp_x-region_x_start)/4)
        idx_y = int((tmp_y-region_y_start)/4)
        bar_matrix[idx_x][idx_y] = b4_bars[idx]
        bar_matrix_idx_dict.update({b4_bars[idx]:[idx_x,idx_y]})
    bar_matrix = np.array(bar_matrix)'''


    bar_matrix, bar_matrix_idx_dict = generate_bar_matrix(select_b4_adata, region_x_start, region_x_end, region_y_start, region_y_end)

    b4_bars = list(select_b4_adata.obs_names)
    score_matrix = show_score_region(b4_bars, bar_matrix_idx_dict, scores)


    if not os.path.exists(output_path):
        os.makedirs(output_path)


    x_max = score_matrix.shape[0]
    x_mid = int(x_max/split_num)
    y_max = score_matrix.shape[1]
    y_mid = int(y_max/split_num)
    for x_idx in range(split_num):
        for y_idx in range(split_num):
            points = []
            for idx in range(x_mid*x_idx,x_mid*(x_idx+1)):
                for jdx in range(y_mid*y_idx,y_mid*(y_idx+1)):
                    if score_matrix[idx,jdx]>threshold:
                        points.append([idx,jdx,score_matrix[idx,jdx]])
            if len(points)==0:
                continue
            points = np.array(points)
            coordinates = points[:, :2]
            tmp_scores = points[:, 2]
            np.save(output_path+str(x_idx)+'_'+str(y_idx)+'_coor.npy', coordinates)
            np.save(output_path+str(x_idx)+'_'+str(y_idx)+'_score.npy', tmp_scores)
            print(len(coordinates))

    return scores

        
def get_score(select_count_matrix, select_mask_matrix):
    unsplice_adata_tmp = ad.AnnData(select_count_matrix)
    unsplice_adata_tmp.layers['unspliced_bins'] = select_mask_matrix
    unsplice_adata_tmp.layers['unspliced'] = select_count_matrix
    adata = unsplice_adata_tmp
    
    layer = 'unspliced'
    adata.layers[layer] = adata.X
    
    method='EM+BP'
    moran_kwargs = None,
    em_kwargs = None,
    vi_kwargs=dict(downsample=0.1, seed=0),
    bp_kwargs = None,
    threshold = None,
    use_knee = False,
    mk = None,
    bins_layer = None,
    certain_layer = None
    scores_layer = None
    mask_layer= None
    
    
    
    X = select_layer_data(adata, layer, make_dense=True)
    certain_mask = None
    if certain_layer:
        certain_mask = select_layer_data(adata, certain_layer).astype(bool)
    bins = None
    if bins_layer is not False:
        bins_layer = gen_new_layer_key(layer, "bins")
        if bins_layer in adata.layers:
            bins = select_layer_data(adata, bins_layer)
    method = method.lower()
    print(f"Scoring pixels with {method} method.")
    k=5 
    scores = _score_pixels(X, k, method, moran_kwargs, em_kwargs, vi_kwargs, bp_kwargs, certain_mask, bins)
    return scores


def show_score_region(select_bars, bar_matrix_idx_dict, scores):
    select_scores_coor = []
    for bar in select_bars:
        tmp_loc = bar_matrix_idx_dict[bar]
        select_scores_coor.append(tmp_loc)
    select_scores_coor = np.array(select_scores_coor)
    score_region_x_min = min(select_scores_coor[:,0])
    score_region_x_max = max(select_scores_coor[:,0])
    score_region_y_min = min(select_scores_coor[:,1])
    score_region_y_max = max(select_scores_coor[:,1])
    tmp_score_matrix = scores[score_region_x_min:score_region_x_max,score_region_y_min:score_region_y_max]
    return tmp_score_matrix
