import os
import pyro
import pyro.distributions as dist
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm

from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
import glob

torch.manual_seed(0)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def top_10_indices_per_row(arr):
    sorted_indices = np.argsort(arr, axis=1)
    top_10_indices = sorted_indices[:, -10:]
    return np.fliplr(top_10_indices)




SMALL_CONSTANT_1 = 0.001
SMALL_CONSTANT_2 = 0.000000001

      


def model(counts, anchor_proportions, anchor_weights, nearest_anchor2anchor, sc_rate, disper, scale_prior, additive_prior):

    std = torch.tensor(1).to(device)
    gamma = pyro.sample("gamma", dist.Normal(scale_prior, std).to_event(1))
    epsilon = pyro.sample("epsilon", dist.Normal(additive_prior, std).to_event(1))
    

    gamma = torch.sigmoid(gamma).to(device)
    epsilon = torch.exp(epsilon).to(device)
    
    num_anchors = anchor_proportions.shape[0]
    with pyro.plate("anchor_plate", num_anchors):
        c_logits = pyro.sample(f'c_logits', dist.Dirichlet(anchor_proportions + 0.001))

    n = len(counts)
    with pyro.plate("data", n):
        a_logits = pyro.sample(f'a_logits', dist.Dirichlet(anchor_weights+0.000000001))
        a = pyro.sample('a', dist.Categorical(a_logits))
        row_indices = torch.arange(nearest_anchor2anchor.size(0), device=nearest_anchor2anchor.device)
        a = nearest_anchor2anchor[row_indices, a]
        c = pyro.sample('c', dist.Categorical(c_logits[a]))

        total_counts = counts.sum(dim=1)
        rates = torch.index_select(sc_rate, 0, c)
        rates = total_counts.unsqueeze(1) * rates

        rates = torch.add(torch.mul(rates, gamma), epsilon)
        pyro.sample('counts', dist.NegativeBinomial(rates, logits=disper).to_event(1), obs=counts)


def guide(counts, anchor_proportions, anchor_weights, nearest_anchor2anchor, sc_rate, disper, scale_prior, additive_prior):
    #anchor_proportions: num_anchor*num_celltype
    #anchor_weights: num_spot*num_anchor
    gamma_loc = pyro.param('gamma_loc', scale_prior)
    gamma_scale = pyro.param('gamma_scale', torch.tensor(1.).to(device),
                         constraint=dist.constraints.positive)
    epsilon_loc = pyro.param('epsilon_loc', additive_prior)
    epsilon_scale = pyro.param('epsilon_scale', torch.tensor(1.).to(device),
                         constraint=dist.constraints.positive)
    gamma = pyro.sample("gamma", dist.Normal(gamma_loc, gamma_scale).to_event(1))
    epsilon = pyro.sample("epsilon", dist.Normal(epsilon_loc, epsilon_scale).to_event(1))

    gamma = torch.sigmoid(gamma)
    epsilon = torch.exp(epsilon)
    
    num_anchors = anchor_proportions.shape[0]
    n = len(counts)
    # Variational parameters for a

    a_concentration = pyro.param("a_concentration", (anchor_weights+0.000000001).clone(), constraint=dist.constraints.positive)
    a_concentration = a_concentration.to(torch.float64)
    c_concentration_prior = anchor_proportions
    c_concentration = pyro.param("c_concentration", (c_concentration_prior+0.001).clone(), constraint=dist.constraints.positive)
    c_concentration = c_concentration.to(torch.float64)
    with pyro.plate("anchor_plate", num_anchors):
        c_logits = pyro.sample(f'c_logits', dist.Dirichlet(c_concentration))

    with pyro.plate("data", n):
        a_logits = pyro.sample(f'a_logits', dist.Dirichlet(a_concentration))
        a = pyro.sample('a', dist.Categorical(a_logits))
        row_indices = torch.arange(nearest_anchor2anchor.size(0), device=nearest_anchor2anchor.device)
        a = nearest_anchor2anchor[row_indices, a]
        c = pyro.sample('c', dist.Categorical(c_logits[a]))



def train_model(work_path,start=0, num_epochs = 30000):
    data_path = work_path+'anchor_files/'
    result_path = work_path+'result/'

    scale_prior = torch.load(work_path+"scale_prior.pt")
    additive_prior = torch.load(work_path+"additive_prior.pt")
    
    
    csv_path = work_path+'mu_gene_expression.csv'
    gene_info = pd.read_csv(csv_path, delimiter = ',', header = 0, index_col = 0)
    ct_list = list(gene_info.index)

    select_ct_lst = ct_list


    select_genes = list(gene_info.columns)

    select_gene_index_list = []
    for gene in select_genes:
        select_gene_index_list.append(select_genes.index(gene))



    select_ct_index_list = []
    for ct in select_ct_lst:
        select_ct_index_list.append(ct_list.index(ct))



    dist_hyper = 5

    mu_expr_file = work_path+'mu_gene_expression.csv'

    disper_file = work_path+'disp_gene_expression.csv'
    disper = torch.tensor(pd.read_csv(disper_file, delimiter = ',', header = None).values.astype(np.float32))[0]
    
    
    
    
    pyro.clear_param_store()
    # Generate some sample data for demonstration


    datafiles = []
    for file in glob.glob(data_path + "filtered_select_bars*"):
        datafiles.append(file)
        
    
    threshold = 20

    pyro.util.set_rng_seed(0)
    for idx in tqdm(range(start,len(datafiles))):
        pyro.clear_param_store()
        counts = np.load(data_path+'counts_'+str(idx)+'.npy')
        anchor_prop_lst = np.load(data_path+'anchor_prop_'+str(idx)+'.npy')
        anchor_weights = np.load(data_path+'anchor_weights_'+str(idx)+'.npy')
        filtered_select_bars = np.load(data_path+'filtered_select_bars_'+str(idx)+'.npy')

        result_indices = top_10_indices_per_row(anchor_weights)

        rows = np.arange(result_indices.shape[0])[:, np.newaxis]
        nearest_anchor_weights = anchor_weights[rows, result_indices]

        
        counts = torch.tensor(counts)
        anchor_proportions = torch.tensor(anchor_prop_lst)
        #anchor_weights = torch.tensor(anchor_weights)



        nearest_anchor2anchor = torch.tensor(result_indices.copy()).to(device)

        nearest_anchor_weights = torch.tensor(nearest_anchor_weights).to(device)



        sc_rate = torch.tensor(pd.read_csv(mu_expr_file, delimiter = ',', header = 0, index_col = 0).values.astype(np.float32))
        sc_rate = sc_rate[select_ct_index_list][:,select_gene_index_list]
        sc_rate = torch.nn.functional.softplus(sc_rate)    
        sc_rate = sc_rate.to(device)

        anchor_proportions = anchor_proportions.to(device)
        counts = counts.to(device)

        # Set up the optimizer
        optimizer = Adam({"lr": 0.001})
        
        # Set up the SVI algorithm
        svi = SVI(model, guide, optimizer, loss=Trace_ELBO())
        
        print('Process data file: '+str(idx))
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        for epoch in range(num_epochs):
            loss = svi.step(counts, anchor_proportions, nearest_anchor_weights, nearest_anchor2anchor, sc_rate, disper.to(device), scale_prior, additive_prior)
            c_logits_dis_prop  = pyro.param("c_concentration").detach()
            a_logits_dis_prop = pyro.param("a_concentration").detach()

            if epoch % 10000 == 0:
                print(f"Epoch {epoch}, Loss: {loss}")
                
        # Get the optimized variational parameters
        a_logits_dis_prop = pyro.param("a_concentration").detach()
        c_logits_dis_prop  = pyro.param("c_concentration").detach()
        
        # Compute the posterior probabilities for a and c

        torch.save(a_logits_dis_prop, result_path+'a_logits_dis_prop_'+str(idx)+'.pt')
        torch.save(c_logits_dis_prop, result_path+'c_logits_dis_prop_'+str(idx)+'.pt')
