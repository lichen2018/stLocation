import os
import pyro
import pyro.distributions as dist
import torch
import pandas as pd
import numpy as np
import anndata as ad
import scanpy as sc
pyro.clear_param_store()
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

scale_prior = torch.tensor(np.load('/share/result/spatial/test_lic/subcellular_detection/bin4/region_0/b40/dataset/scale.npy')).to(device)
additive_prior = torch.tensor(np.load('/share/result/spatial/test_lic/subcellular_detection/bin4/region_0/b40/dataset/additive.npy')).to(device)

mu_expr_file = '/share/result/spatial/test_lic/subcellular_detection/bin4/region_0/mu_gene_expression.csv'
mu_expr = pd.read_csv(mu_expr_file, delimiter = ',', header = 0, index_col = 0)
ct_list = list(mu_expr.index)

disper_file = '/share/result/spatial/test_lic/subcellular_detection/bin4/region_0/disp_gene_expression.csv'
disper = torch.tensor(pd.read_csv(disper_file, delimiter = ',', header = None).values.astype(np.float32))[0].to(device)


def merge_bins_by_anchor(counts, anchors):
    unique_anchors = torch.unique(anchors)
    result = []
    for anchor in unique_anchors:
        mask = anchors == anchor
        selected_bins = counts[mask]
        summed_bin = torch.sum(selected_bins, dim=0)
        result.append(summed_bin)
    return torch.stack(result)


def calculate_proportion(choosed_ct_tensor, anchors, ct_num):
    unique_anchors = torch.unique(anchors)
    num_anchors = len(unique_anchors)
    result_matrix = torch.zeros(num_anchors, ct_num)
    for anchor in unique_anchors:
        anchor_mask = anchors == anchor
        selected_values = choosed_ct_tensor[anchor_mask]
        for j in range(ct_num):
            value_count = (selected_values == j).sum()
            total_count = len(selected_values)
            if total_count > 0:
                result_matrix[anchor, j] = value_count / total_count
    return result_matrix



def model(counts, anchor_proportions, anchor_weights, sc_rate, disper):
    #gamma = pyro.sample("gamma", dist.Normal(torch.zeros(sc_rate.shape[1]).to(device), torch.ones(sc_rate.shape[1]).to(device)).to_event(1))
    #gamma = torch.exp(gamma).to(device)
    #epsilon = torch.exp(epsilon).to(device)
    
    std = torch.tensor(1).to(device)
    gamma = pyro.sample("gamma", dist.Normal(scale_prior, std).to_event(1))
    epsilon = pyro.sample("epsilon", dist.Normal(additive_prior, std).to_event(1))
    
    gamma = torch.sigmoid(gamma).to(device)
    epsilon = torch.exp(epsilon).to(device)
    
    num_anchors = anchor_proportions.shape[0]
    with pyro.plate("anchor_plate", num_anchors):
        c_logits = pyro.sample(f'c_logits', dist.Dirichlet(anchor_proportions + 0.001))
    #c_logits = pyro.sample(f'c_logits', dist.Dirichlet(anchor_proportions+0.001).to_event(1))
    #c_logits = normalize_2Dtensor(c_logits)
    #a_logits = normalized_anchor_weights
    #c_logits = anchor_proportions
    n = len(counts)
    with pyro.plate("data", n):
        a_logits = pyro.sample(f'a_logits', dist.Dirichlet(anchor_weights+0.000000001))
        #a_logits = normalize_2Dtensor(a_logits)

        a = pyro.sample('a', dist.Categorical(a_logits))
        c = pyro.sample('c', dist.Categorical(c_logits[a]))

        total_counts = counts.sum(dim=1)
        rates = torch.index_select(sc_rate, 0, c)
        #print(rates)

        rates = total_counts.unsqueeze(1) * rates

        rates = torch.add(torch.mul(rates, gamma), epsilon)
        #print(rates)
        pyro.sample('counts', dist.NegativeBinomial(rates, logits=disper).to_event(1), obs=counts)

def guide(counts, anchor_proportions, anchor_weights, sc_rate, disper):
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
    #c_concentration_prior = torch.softmax(torch.randn(anchor_proportions.shape[0], anchor_proportions.shape[1]), dim=1).to(device)
    c_concentration_prior = anchor_proportions
    c_concentration = pyro.param("c_concentration", (c_concentration_prior+0.001).clone(), constraint=dist.constraints.positive)
    c_concentration = c_concentration.to(torch.float64)
    with pyro.plate("anchor_plate", num_anchors):
        c_logits = pyro.sample(f'c_logits', dist.Dirichlet(c_concentration))
    #c_logits = pyro.sample(f'c_logits', dist.Dirichlet(c_concentration).to_event(1))
    #c_logits = normalize_2Dtensor(c_logits)
    with pyro.plate("data", n):
        a_logits = pyro.sample(f'a_logits', dist.Dirichlet(a_concentration))
        #a_logits = normalize_2Dtensor(a_logits)
        a = pyro.sample('a', dist.Categorical(a_logits))
        c = pyro.sample('c', dist.Categorical(c_logits[a]))


from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam

# Generate some sample data for demonstration


threshold = 20
adata_no_gamma_lst = []
no_gamma_lst = []
pyro.util.set_rng_seed(0)
for idx in range(28):
    pyro.clear_param_store()
    counts = np.load('./dataset/counts_'+str(idx)+'.npy')
    anchor_prop_lst = np.load('./dataset/anchor_prop_'+str(idx)+'.npy')
    anchor_weights = np.load('./dataset/anchor_weights_'+str(idx)+'.npy')
    filtered_select_bars = np.load('./dataset/filtered_select_bars_'+str(idx)+'.npy')

    
    counts = torch.tensor(counts)
    anchor_proportions = torch.tensor(anchor_prop_lst)
    anchor_weights = torch.tensor(anchor_weights)
    sc_rate = torch.tensor(mu_expr.values.astype(np.float32))
    sc_rate = torch.nn.functional.softplus(sc_rate)
    
    sc_rate = sc_rate.to(device)
    anchor_weights = anchor_weights.to(device)
    anchor_proportions = anchor_proportions.to(device)
    counts = counts.to(device)

    # Set up the optimizer
    optimizer = Adam({"lr": 0.001})
    
    # Set up the SVI algorithm
    svi = SVI(model, guide, optimizer, loss=Trace_ELBO())
    
    num_epochs = 35000
    for epoch in range(num_epochs):
        loss = svi.step(counts, anchor_proportions, anchor_weights, sc_rate, disper)
        c_logits_dis_prop  = pyro.param("c_concentration").detach()
        c_probs_dis_prop = torch.softmax(c_logits_dis_prop, dim=1)
        if epoch % 5000 == 0:
            print(f"Epoch {epoch}, Loss: {loss}")
        if epoch % 10000 == 0:
            print(f"Epoch {epoch}, Loss: {c_probs_dis_prop}")
            
            
    # Get the optimized variational parameters
    a_logits_dis_prop = pyro.param("a_concentration").detach()
    c_logits_dis_prop  = pyro.param("c_concentration").detach()
    
    # Compute the posterior probabilities for a and c

    torch.save(a_logits_dis_prop, './result/a_logits_dis_prop_'+str(idx)+'.pt')
    torch.save(c_logits_dis_prop, './result/c_logits_dis_prop_'+str(idx)+'.pt')
