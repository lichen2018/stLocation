import scanpy as sc
import pandas as pd
import anndata as ad
import numpy as np
import torch
import warnings
warnings.filterwarnings('ignore')
import glob
from tqdm import tqdm

#b4_adata = sc.read_h5ad('/share/result/spatial/test_lic/tuoyexian/B2_3/bin4_in_region.h5ad')
def top_10_indices_per_row(arr):
    sorted_indices = np.argsort(arr, axis=1)
    top_10_indices = sorted_indices[:, -10:]
    return np.fliplr(top_10_indices)





def get_output_adata_nearest(nearest_anchor2anchor, a_logits_dis_prop, c_logits_dis_prop, ct_list, all_st_adata_bin4, select_bars, threshold, batch_id, ct_threshold, anchor_threshold):
    row_sums = a_logits_dis_prop.sum(dim=1, keepdim=True)
    normalized_anchor_prop = a_logits_dis_prop / row_sums
    
    row_sums = c_logits_dis_prop.sum(dim=1, keepdim=True)
    normalized_prop = c_logits_dis_prop / row_sums
    selected_anchors = torch.argmax(a_logits_dis_prop, dim=1)
    selected_ct = torch.argmax(normalized_prop, dim=1)
    
    tmp_selected_anchors = nearest_anchor2anchor[torch.arange(selected_anchors.size()[0]), selected_anchors.cpu()]
    cell_loc_ls = []
    ct_labels = []
    cell_expr_matrix = []
    tmp_obs_names = []
    assigned_clu_dict = {}
    X_sparse = all_st_adata_bin4.X.tocsr()
    score_lst = []
    for idx in range(normalized_prop.shape[0]):
        if max(normalized_prop[idx]).cpu().numpy() < ct_threshold:
            continue
        bar_lst = []
        index = torch.where(tmp_selected_anchors == idx)
        index=index[0].cpu().numpy()
        ct_index = selected_ct[idx].cpu().numpy()
        for ele in index:
            if normalized_anchor_prop[ele][nearest_anchor2anchor[ele].tolist().index(idx)] < anchor_threshold:
                continue
            bar = select_bars[ele]
            bar_lst.append(bar)
        if len(bar_lst) == 0:
            continue
        spatial_coords = np.array([all_st_adata_bin4[bar].obsm['spatial'][0] for bar in bar_lst])
        x_coords = spatial_coords[:, 0]
        y_coords = spatial_coords[:, 1]
        if (max(y_coords) - min(y_coords))/4 > threshold:
            continue
        if (max(x_coords) - min(x_coords))/4 > threshold:
            continue
        for bar in bar_lst:
            assigned_clu_dict.update({bar:idx})
        indices = [all_st_adata_bin4.obs_names.get_loc(bar) for bar in bar_lst]
        cell_expr = np.array(X_sparse[indices].sum(axis=0))[0]
        ct_labels.append(ct_list[ct_index])
        cell_expr_matrix.append(cell_expr)
        cell_loc_ls.append([np.mean(x_coords),np.mean(y_coords)])
        tmp_obs_names.append(str(batch_id)+'_'+str(idx))
        score_lst.append(max(normalized_prop[idx]).cpu().numpy())
    cell_expr_matrix = np.array(cell_expr_matrix)
    cell_loc_array = np.array(cell_loc_ls)
    
    tmp_adata = ad.AnnData(cell_expr_matrix)
    tmp_adata.obs_names = tmp_obs_names
    tmp_adata.var_names = all_st_adata_bin4.var_names
    tmp_adata.obs['cell_type'] = ct_labels
    tmp_adata.obsm['spatial'] = cell_loc_array
    return tmp_adata, assigned_clu_dict, score_lst



def get_adata(work_path, b4_adata_path):
    data_path = work_path+'anchor_files/'
    result_path = work_path+'result/'
    b4_adata = sc.read_h5ad(b4_adata_path)
    csv_path = work_path+'/mu_gene_expression.csv'
    gene_info = pd.read_csv(csv_path, delimiter = ',', header = 0, index_col = 0)
    ct_list = list(gene_info.index)

    select_ct_lst = ct_list

    threshold = 20
    datafiles = []
    for file in glob.glob(data_path + "filtered_select_bars*"):
        datafiles.append(file)
    
    
    adata_lst = []
    score_lst = []
    assigned_clu_dict_lst = []

    for idx in tqdm(range(len(datafiles))):
        batch_id = str(idx)
        anchor_weights = np.load(data_path+'anchor_weights_'+str(idx)+'.npy')
        result_indices = top_10_indices_per_row(anchor_weights)
        rows = np.arange(result_indices.shape[0])[:, np.newaxis]
        #nearest_anchor_weights = anchor_weights[rows, result_indices]
        nearest_anchor2anchor = torch.tensor(result_indices.copy())
        filtered_select_bars = np.load(data_path+'filtered_select_bars_'+str(idx)+'.npy')
        a_logits_dis_prop = torch.load(result_path+'a_logits_dis_prop_'+str(idx)+'.pt')
        c_logits_dis_prop = torch.load(result_path+'c_logits_dis_prop_'+str(idx)+'.pt')
        tmp_adata,assigned_clu_dict, tmp_score_lst = get_output_adata_nearest(nearest_anchor2anchor, a_logits_dis_prop, c_logits_dis_prop, ct_list, b4_adata, filtered_select_bars, threshold, batch_id, 0,0.5)
        adata_lst.append(tmp_adata)
        assigned_clu_dict_lst.append(assigned_clu_dict)
        score_lst.append(tmp_score_lst)
    cell_score_lst = []
    for ele in score_lst:
        cell_score_lst.extend(ele)
    adata = ad.concat(adata_lst)
    return adata, cell_score_lst

