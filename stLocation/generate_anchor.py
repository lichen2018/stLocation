import pandas as pd
import numpy as np
import scanpy as sc
import os
import sys
import anndata as ad
import torch
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
np.random.seed(0)
torch.manual_seed(0)
import warnings
from torch import nn
from sklearn.metrics import accuracy_score
import torch.nn.functional as F
from torch.distributions.utils import (
    broadcast_all,
    lazy_property,
    logits_to_probs,
    probs_to_logits,
)
import torch.optim as optim
import torch.nn.functional as f
from torch.autograd import Variable
from torch.distributions import NegativeBinomial, Normal
from collections import Counter
import glob

from collections import Counter
class ST_Vae1(nn.Module):
    def __init__(self, n_input, n_class, n_layers, n_latent, n_hidden = 1024, dropout_rate = 0.1, use_batch_norm = True, use_layer_norm= True, use_activation= True):
        super(ST_Vae1, self).__init__()
        layers_dim = [n_input] + (n_layers - 1) * [n_hidden]
        modules = []
        for i, (n_in, n_out) in enumerate(zip(layers_dim[:-1], layers_dim[1:])):
            modules.append(
                nn.Sequential(
                    nn.Linear(n_in, n_out),
                    nn.BatchNorm1d(n_out, momentum=0.01, eps=0.001) if use_batch_norm else None,
                    nn.LayerNorm(n_out, elementwise_affine=False) if use_layer_norm else None,
                    nn.LeakyReLU() if use_activation else None,
                    nn.Dropout(p=dropout_rate) if dropout_rate > 0 else None,)
            )
        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(n_hidden, n_latent)
        self.fc_var = nn.Linear(n_hidden, n_latent)
        layers_dim = [n_latent] + (n_layers - 1) * [n_hidden]
        modules = []
        for i, (n_in, n_out) in enumerate(zip(layers_dim[:-1], layers_dim[1:])):
            modules.append(
                nn.Sequential(
                    nn.Linear(n_in, n_out),
                    nn.BatchNorm1d(n_out, momentum=0.01, eps=0.001) if use_batch_norm else None,
                    nn.LayerNorm(n_out, elementwise_affine=False) if use_layer_norm else None,
                    nn.LeakyReLU() if use_activation else None,
                    nn.Dropout(p=dropout_rate) if dropout_rate > 0 else None,)
            )
        self.decoder = nn.Sequential(*modules)
        self.final_layer = nn.Sequential(
                    nn.Linear(layers_dim[-1], n_class),
                    nn.BatchNorm1d(n_class, momentum=0.01, eps=0.001) if use_batch_norm else None,
                    nn.LayerNorm(n_class, elementwise_affine=False) if use_layer_norm else None,
                    nn.Tanh())
        self.px_r =nn.Linear(layers_dim[-1], n_input)
        self.scale =nn.Parameter(torch.randn(n_input))
        self.additive =nn.Parameter(torch.randn(n_input))
    def encode(self, input_x):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x F]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input_x)
        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        return mu, log_var
    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu
    def decode(self, z):
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x F]
        """
        result = self.decoder(z)
        y_proportion = self.final_layer(result)
        p_r = torch.exp(self.px_r(result))
        return y_proportion, p_r
    def forward(self, xs):
        mu, log_var = self.encode(xs)
        z = self.reparameterize(mu, log_var)
        y_proportion, p_r = self.decode(z)
        scale = nn.Sigmoid()(self.scale)
        additive = torch.exp(self.additive)
        return  y_proportion, scale, additive, mu, log_var
    


def get_coor_bar_dict(st_adata):
    coor_bar_dict = {}
    obs_names= list(st_adata.obs_names)
    coor_arr = st_adata.obsm['spatial']
    for idx in range(len(st_adata)):
        x = str(int(coor_arr[idx][0]))
        y = str(int(coor_arr[idx][1]))
        bar = obs_names[idx]
        coor_bar_dict.update({x+'_'+y:bar})
    return coor_bar_dict


def get_bar_coor_dict(st_adata):
    bar_coor_dict = {}
    obs_names= list(st_adata.obs_names)
    coor_arr = st_adata.obsm['spatial']
    for idx in range(len(st_adata)):
        bar = obs_names[idx]
        bar_coor_dict.update({bar:coor_arr[idx]})
    return bar_coor_dict
    
    
def get_process_region_barcodes(select_bars, bar_matrix_idx_dict, bar_matrix):
    select_coor = []
    for bar in select_bars:
        tmp_loc = bar_matrix_idx_dict[bar]
        select_coor.append(tmp_loc)
    select_coor = np.array(select_coor)
    process_region_x_min = min(select_coor[:,0])
    process_region_x_max = max(select_coor[:,0])
    process_region_y_min = min(select_coor[:,1])
    process_region_y_max = max(select_coor[:,1])
    process_barcode_matrix = bar_matrix[process_region_x_min:process_region_x_max,process_region_y_min:process_region_y_max]
    return process_barcode_matrix

from sparsemax import Sparsemax
torch.manual_seed(0)
def get_stvae_pred(model, feat):
    model.eval()
    feat = torch.tensor(feat.astype(np.float32))
    feat = feat.to(device)
    with torch.no_grad():
        pred_ys, scale, additive, mu, log_var = model(feat)
    pred_ys_norm = Sparsemax()(pred_ys)
    if feat.shape[0] ==1:
        tmp_pred_ys = np.array(pred_ys_norm.cpu())[0,:]
    tmp_pred_ys = np.array(pred_ys_norm.cpu())
    return tmp_pred_ys


def get_anchor_prop_lst(cluster_centers, bar_matrix, stvae_b40, adata, width, height):
    w = 4
    h = 4
    anchor_prop_lst = []
    updated_cluster_centers = []
    tmp_feat_lst = []
    tmp_pred_ys = []
    neighbor_num_lst = []
    X_sparse = adata.X.tocsr()
    for clu_coor in cluster_centers:
        x = int(clu_coor[0])
        y = int(clu_coor[1])
        min_x = max(x-w,0)
        max_x = min(x+w+1,width)
        min_y = max(y-h,0)
        max_y = min(y+h+1,height)
        sub_matrix = bar_matrix[min_x:max_x, min_y:max_y]
        non_empty_mask = (sub_matrix != '')
        neighbor_bars = sub_matrix[non_empty_mask].tolist()
        if len(neighbor_bars) < 5:
            continue
        neighbor_num_lst.append(len(neighbor_bars))
        updated_cluster_centers.append(clu_coor)

        indices = [adata.obs_names.get_loc(bar) for bar in neighbor_bars]
        feat = X_sparse[indices].sum(axis=0)
        tmp_feat_lst.append(np.array(feat)[0])
        
    if len(tmp_feat_lst) > 0:
        feat = np.array(tmp_feat_lst)
        tmp_pred_ys = get_stvae_pred(stvae_b40, feat)
    return tmp_pred_ys, updated_cluster_centers, neighbor_num_lst


import math

def accelerated_filter(region_b4_bars, region_clu_lst, bar_matrix_idx_dict):
    # 1. 预处理：将坐标转换为NumPy数组（避免循环中查字典）
    # 提取所有b4的坐标，形状为 (n_b4, 2)
    b4_coors = np.array([bar_matrix_idx_dict[b4] for b4 in region_b4_bars])
    # 聚类坐标转换为数组，形状为 (n_clu, 2)
    clu_coors = np.array(region_clu_lst)
    
    # 2. 向量化计算距离并筛选
    filtered_select_bars = []
    filter_index_lst = []
    
    # 提前计算距离阈值的平方（避免开方运算，加速比较）
    dist_threshold_sq = 7.5 ** 2  # 56.25
    
    # 遍历每个b4（仅保留外层循环，内层用向量化替代）
    for b4_idx in range(len(region_b4_bars)):
        # 当前b4的坐标（形状：(2,)）
        b4_coor = b4_coors[b4_idx]
        
        # 向量化计算与所有聚类中心的距离平方（形状：(n_clu,)）
        # (x1-x2)^2 + (y1-y2)^2，避免开方，提高效率
        dists_sq = np.sum((clu_coors - b4_coor) ** 2, axis=1)
        
        # 判断是否存在距离小于7.5的聚类中心
        if np.any(dists_sq < dist_threshold_sq):
            filtered_select_bars.append(region_b4_bars[b4_idx])
            filter_index_lst.append(b4_idx)
    
    return filtered_select_bars, filter_index_lst

    

def filter_low_qual_b40(b40_slide, b40_result, select_ct_lst):
    salus_bar_names = []
    salus_bar_name_dict = {}
    max_cols = b40_result.idxmax(axis=1)
    max_values = b40_result.max(axis=1)

    mask = (max_cols.isin(select_ct_lst)) & (max_values >= 0.15)

    salus_bar_names = b40_result.index[mask].unique().tolist()

    salus_bar_name_dict = {idx: 1 for idx in salus_bar_names}

    salus_b40 = b40_slide[salus_bar_names]
    return salus_b40


def generate_bar_matrix(select_b4_adata, x_min, x_max, y_min, y_max):
    bar_matrix_idx_dict = {}
    tmp_lst = []
    for ele in list(select_b4_adata.var_names):
        res = ele.upper()
        tmp_lst.append(res)
    select_b4_adata.var_names = tmp_lst

    coor = select_b4_adata.obsm['spatial']

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


def get_select_b4(b4_coors, region_x_start, region_x_end, region_y_start, region_y_end):
    select_b4 = []
    for idx in range(b4_coors.shape[0]):
        ele = b4_coors[idx]
        if ele[0]>region_x_start and ele[0]<region_x_end and ele[1]>region_y_start and ele[1]<region_y_end:
            select_b4.append(idx)
    return select_b4

'''coor = select_b4_adata.obsm['spatial']
x_min = region_x_start
x_max = region_x_end

y_min = region_y_start
y_max = region_y_end
default_string = ""
bar_matrix = []
for idx in range(int((x_max-x_min)/4)+1):
    tmp_lst = []
    for jdx in range(int((y_max-y_min)/4)+1):
        tmp_lst.append(default_string)
    bar_matrix.append(tmp_lst)
#bar_matrix = [[default_string for _ in range(int((x_max-x_min)/4)+1)] for _ in range(int((y_max-y_min)/4)+1)]

obs_names = list(select_b4_adata.obs_names)
for idx in range(coor.shape[0]):
    tmp_x = coor[idx][0]
    tmp_y = coor[idx][1]
    idx_x = int((tmp_x-x_min)/4)
    idx_y = int((tmp_y-y_min)/4)
    bar_matrix[idx_x][idx_y] = obs_names[idx]
    bar_matrix_idx_dict.update({obs_names[idx]:[idx_x,idx_y]})
bar_matrix = np.array(bar_matrix)'''


def generate_anchor(work_path, b40_adata_path, b4_adata_path, split_num = 7):
    
    score_matrix_path = work_path + 'score_matrix_files/'
    cluster_center_path = work_path + 'cluster_center_files/'
    output_path = work_path + 'anchor_files/'
    b4_adata = sc.read_h5ad(b4_adata_path)


    gene_info_path = work_path+'mu_gene_expression.csv' 
    gene_info = pd.read_csv(gene_info_path, delimiter = ',', header = 0, index_col = 0)
    ct_list = list(gene_info.index)
    select_ct_lst = ct_list
    select_genes = list(gene_info.columns)
    stvae_b40  = ST_Vae1(len(select_genes), len(ct_list), n_layers = 3, n_latent = 128)
    stvae_weights_file = work_path+'model_weight.pkl'
    stvae_b40.load_state_dict(torch.load(stvae_weights_file))
    use_cuda = True
    if use_cuda:
        stvae_b40.cuda()


    params = stvae_b40.state_dict()

    scale_prior = params['scale']

    torch.save(scale_prior, work_path+"scale_prior.pt")

    additive_prior = params['additive']

    torch.save(additive_prior, work_path+"additive_prior.pt")

    slide = sc.read_h5ad(b40_adata_path)




    cell_type_proportion_path = work_path+'result.csv'
    #b40_result = pd.read_csv(csv_path, delimiter = ',', header = 0, index_col = 0)
    b40_result = pd.read_csv(cell_type_proportion_path, delimiter = ',', header = 0, index_col = 0)

    b40_slide = slide[list(b40_result.index)]
    b40_slide.obs[list(b40_result.columns)] = b40_result
    b40_slide = ad.AnnData(b40_slide.X, obs=b40_slide.obs, var=b40_slide.var,obsm=b40_slide.obsm)


    '''salus_bar_names = []
    salus_bar_name_dict = {}


    max_cols = b40_result.idxmax(axis=1)
    max_values = b40_result.max(axis=1)

    # 创建条件掩码：最大值列在目标列表中且值 >= 0.15
    mask = (max_cols.isin(select_ct_lst)) & (max_values >= 0.15)

    # 筛选符合条件的索引并去重
    salus_bar_names = b40_result.index[mask].unique().tolist()

    # 构建字典（如果仍需要）
    salus_bar_name_dict = {idx: 1 for idx in salus_bar_names}

    salus_b40 = b40_slide[salus_bar_names]'''

    salus_b40 = filter_low_qual_b40(b40_slide, b40_result, select_ct_lst)


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


    b4_coors = b4_adata.obsm['spatial']

            
    select_b4 = get_select_b4(b4_coors, region_x_start, region_x_end, region_y_start, region_y_end)

    #bar_matrix_idx_dict = {}
    if len(set(select_genes)&set(list(b4_adata.var_names)))==0:        
        tmp_lst = []
        for ele in list(b4_adata.var_names):
            res = ele.upper()
            tmp_lst.append(res)
        b4_adata.var_names = tmp_lst


    select_b4_adata = b4_adata[select_b4,select_genes]
    sc.pp.filter_cells(select_b4_adata, min_genes=1)

    '''coor = select_b4_adata.obsm['spatial']
    x_min = region_x_start
    x_max = region_x_end

    y_min = region_y_start
    y_max = region_y_end
    default_string = ""
    bar_matrix = []
    for idx in range(int((x_max-x_min)/4)+1):
        tmp_lst = []
        for jdx in range(int((y_max-y_min)/4)+1):
            tmp_lst.append(default_string)
        bar_matrix.append(tmp_lst)
    #bar_matrix = [[default_string for _ in range(int((x_max-x_min)/4)+1)] for _ in range(int((y_max-y_min)/4)+1)]

    obs_names = list(select_b4_adata.obs_names)
    for idx in range(coor.shape[0]):
        tmp_x = coor[idx][0]
        tmp_y = coor[idx][1]
        idx_x = int((tmp_x-x_min)/4)
        idx_y = int((tmp_y-y_min)/4)
        bar_matrix[idx_x][idx_y] = obs_names[idx]
        bar_matrix_idx_dict.update({obs_names[idx]:[idx_x,idx_y]})
    bar_matrix = np.array(bar_matrix)'''



    bar_matrix, bar_matrix_idx_dict = generate_bar_matrix(select_b4_adata, region_x_start, region_x_end, region_y_start, region_y_end)




    coor_bar_dict_bin4 = get_coor_bar_dict(select_b4_adata)
    bar_coor_dict_bin4 = get_bar_coor_dict(select_b4_adata)


    cluster_center_lst = []
    for file in glob.glob(cluster_center_path + "*cluster_center.npy"):
        cluster_center_lst.append(np.load(file))


    cluster_centers = np.concatenate(cluster_center_lst, axis=0)


    b4_bars = list(select_b4_adata.obs_names)
    process_barcode_matrix = get_process_region_barcodes(b4_bars, bar_matrix_idx_dict, bar_matrix)



    updated_cluster_centers = []
    count = 0
    for ele in cluster_centers:
        x = int(ele[0])
        y = int(ele[1])
        search_radius = 0
        flag = True
        while flag:
            for candidate_x in range(x-search_radius, x+search_radius+1):
                if candidate_x < 0 or candidate_x>=process_barcode_matrix.shape[0]:
                    continue
                for candidate_y in range(y-search_radius, y+search_radius+1):
                    if candidate_y < 0 or candidate_y>=process_barcode_matrix.shape[1]:
                        continue
                    barcode = process_barcode_matrix[candidate_x, candidate_y]
                    if barcode in bar_matrix_idx_dict:
                        clu_coor = bar_matrix_idx_dict[barcode]
                        updated_cluster_centers.append(clu_coor)
                        flag = False
                    if flag == False:
                        break
                if flag == False:
                    break
            search_radius += 1
        
    anchor_prop_lst, updated_cluster_centers, neighbor_num_lst = get_anchor_prop_lst(updated_cluster_centers, bar_matrix, stvae_b40, select_b4_adata, 100000, 100000)
    neighbor_bars = []
    cluster_neighbor_bars_dict = {}
    w = 20
    h = 20
    width = 100000000
    height = 100000000
    for idx in range(len(updated_cluster_centers)):
        clu_coor = updated_cluster_centers[idx]
        x = int(clu_coor[0])
        y = int(clu_coor[1])
        min_x = max(x-w,0)
        max_x = min(x+w+1,width)
        min_y = max(y-h,0)
        max_y = min(y+h+1,height)

        sub_matrix = bar_matrix[min_x:max_x, min_y:max_y]
        non_empty_mask = (sub_matrix != '')
        neighbor_bars.extend(sub_matrix[non_empty_mask].tolist())
        cluster_neighbor_bars_dict.update({str(idx):sub_matrix[non_empty_mask].tolist()})
        

    counter = Counter(neighbor_bars)
    neighbor_bars = list(counter.keys())



    ct_list = list(gene_info.index)
    select_genes = list(gene_info.columns)
    select_gene_index_list = []
    filter_select_genes = []
    for gene in select_genes:
        if gene in list(gene_info.columns):
            filter_select_genes.append(gene)
    for gene in filter_select_genes:
        select_gene_index_list.append(list(gene_info.columns).index(gene))

    select_ct_index_list = []
    for ct in select_ct_lst:
        select_ct_index_list.append(ct_list.index(ct))


    dist_hyper = 5

    x_coor = select_b4_adata.obsm['spatial'][:,0]
    y_coor = select_b4_adata.obsm['spatial'][:,1]
    obs_names = select_b4_adata.obs_names
        
    x_min = 0
    x_max = bar_matrix.shape[0]
    y_min = 0
    y_max = bar_matrix.shape[1]



    width = bar_matrix.shape[0]
    height = bar_matrix.shape[1]

    '''cluster_neighbor_bars_dict
    updated_cluster_centers
    counter = Counter(neighbor_bars)
    neighbor_bars = list(counter.keys())'''

    x_split= split_num
    y_split= split_num
    x_delta = int((x_max - x_min)/x_split)
    y_delta = int((y_max - y_min)/y_split)
    X_sparse = select_b4_adata.X.tocsr()
    count = 0
    for x_idx in range(x_split):
        for y_idx in range(y_split):
            x_coor_start = x_min + x_idx*x_delta
            x_coor_end = x_min + (x_idx+1)*x_delta
            y_coor_start = y_min + y_idx*y_delta
            y_coor_end = y_min + (y_idx+1)*y_delta
            if y_coor_start >= y_max or x_coor_start>= x_max:
                continue
            region_clu_idx_lst = []
            for clu_idx in cluster_neighbor_bars_dict:
                clu_coor = updated_cluster_centers[int(clu_idx)]
                if clu_coor[0]>=x_coor_start and clu_coor[0]<x_coor_end and clu_coor[1]>=y_coor_start and clu_coor[1]<y_coor_end:
                    region_clu_idx_lst.append(int(clu_idx))
            region_clu_lst = [updated_cluster_centers[clu_idx] for clu_idx in region_clu_idx_lst]
            region_anchor_prop_lst = [anchor_prop_lst[clu_idx] for clu_idx in region_clu_idx_lst]
            
            region_b4_bars = []
            for clu_idx in region_clu_idx_lst:
                region_b4_bars.extend(cluster_neighbor_bars_dict[str(clu_idx)])
            counter = Counter(region_b4_bars)
            region_b4_bars = list(counter.keys())
            if len(region_b4_bars)==0:
                continue

            bar_coors = np.array([bar_matrix_idx_dict[bar] for bar in region_b4_bars])
            dist_matrix = np.linalg.norm(bar_coors[:, np.newaxis] - region_clu_lst, axis=-1)
            anchor_weights = np.exp(1 - (dist_matrix / dist_hyper) ** 3)
            tmp_anchor_weights = anchor_weights.tolist()

            filtered_select_bars, filter_index_lst = accelerated_filter(region_b4_bars, region_clu_lst, bar_matrix_idx_dict)
                        
            anchor_weights = [tmp_anchor_weights[i] for i in filter_index_lst]
            indices = [select_b4_adata.obs_names.get_loc(bar) for bar in filtered_select_bars]
            counts = X_sparse[indices].toarray()
            a = np.sum(counts,axis =1)
            select_b4_lst = []
            for ele in range(a.shape[0]):
                if a[ele] != 0:
                    select_b4_lst.append(ele)
            counts = counts[select_b4_lst]
            tmp_anchor_weights = [anchor_weights[i] for i in select_b4_lst]

            tmp_filtered_select_bars = [filtered_select_bars[i] for i in select_b4_lst]

                
            counts_arr = np.array(counts)
            anchor_prop_arr = np.array(region_anchor_prop_lst)[:,select_ct_index_list]
            anchor_weights_arr = np.array(tmp_anchor_weights)
            filtered_select_bars_arr = np.array(tmp_filtered_select_bars)
            cluster_center_arr = np.array(region_clu_lst)
            save_path = output_path
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            np.save(save_path+'counts_'+str(count)+'.npy', counts_arr)
            np.save(save_path+'anchor_prop_'+str(count)+'.npy', anchor_prop_arr)
            np.save(save_path+'anchor_weights_'+str(count)+'.npy', anchor_weights_arr)
            np.save(save_path+'filtered_select_bars_'+str(count)+'.npy', filtered_select_bars_arr)
            np.save(save_path+'cluster_center_'+str(count)+'.npy', cluster_center_arr)
            print(str(count)+'_'+str(len(region_clu_lst))+'_'+str(counts_arr.shape[0]))
            import gc
            gc.collect()
            count += 1
