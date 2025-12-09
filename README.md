# stLocation
stLocation is a method to locate and segment cells in spatial transcriptomic dataset. Especially, through integrating single-cell RNA-seq data and high-resolusion spatial transcriptomic, stLocation
could aggregate enough 1um bins to anchors determined by clusters of unspliced RNAs to reconstruct gene profiles of cells and recover their spatial distribution in the tissue.

## Table of Contents
1. [Installation](#installation)
2. [API](#api)
3. [Data](#data)
4. [Example workflow](#example-workflow)
## Installation

### Install stLocation
Download and install pytorch
```
pip3 install --no-cache-dir --find-links https://download.pytorch.org/whl/cu124/torch/ torch==2.6.0
```
Download and install torch-scatter
```
pip3 install https://data.pyg.org/whl/torch-2.6.0%2Bcu124/torch_scatter-2.1.2%2Bpt26cu124-cp310-cp310-linux_x86_64.whl
```
Download and install torch-sparse
```
pip3 install https://data.pyg.org/whl/torch-2.6.0%2Bcu124/torch_sparse-0.6.18%2Bpt26cu124-cp310-cp310-linux_x86_64.whl
```
Download and install torchaudio
```
pip3 install https://download.pytorch.org/whl/cu124/torchaudio-2.6.0%2Bcu124-cp310-cp310-linux_x86_64.whl#sha256=6b54f97fff96b4ba3da44b6b3f50727c25122d1479107b119d1275944ec83ea1
```
Download and install torchvision
```
pip3 install https://download.pytorch.org/whl/cu124/torchvision-0.21.0%2Bcu124-cp310-cp310-linux_x86_64.whl#sha256=3d3e74018eaa7837c73e3764dad3b7792b7544401c25a42977e9744303731bd3
```
Download and install stLocation
```
conda create -n stLocation python=3.10.18
conda activate stLocation
git clone --recursive https://github.com/lichen2018/stLocation.git
cd stLocation
pip3 install -r requirements.txt
python3 setup.py build
python3 setup.py install
```
## API
### Calculate scores for each spot
```python
generate_score_matrix(work_path, b4_adata_path, unsplice_b4_adata_path, b40_adata_path, threshold = 0.2, split_num = 2)
```
#### Description
  ```
  Calculate scores for each spot that reflect the probability of cellular occupancy according to unsplice RNA density.
  ```
#### Parameters  
  ```
  work_path                 path to store intermediate result
  b4_adata_path             path to the AnnData object storing spatial RNA data at 1 μm resolution.
  unsplice_b4_adata_path    path to the AnnData object storing spatial unsplice RNA data at 1 μm resolution.
  b40_adata_path            path to the AnnData object storing spatial RNA data at 10 μm resolution.
  threshold                 threshold for filtering out spots with low scores.
  split_num                 the tissue is split to split_num*split_num regions.
  ```
#### Return 
  ```
  score_matrix              score matrix of the whole tissue.
  ```

### Get cluster centers of score matrix

```python
generate_cluster_centers(work_path, split_num = 2, max_iter=100)
```
#### Description
  ```
  Get cluster centers of score matrix using mean-shift algorithm
  ```
#### Parameters  
  ```
  work_path                 path to store intermediate result.
  split_num                 the tissue has been split to split_num*split_num regions.
  max_iter                  iteration number.  
  ```


### Generate anchors to indicate cellular positions
```python
generate_anchor(work_path, b40_adata_path, b4_adata_path, split_num = 7)
```
#### Description
  ```
  Generate anchors to indicate cellular positions
  ```
#### Parameters  
  ```
  work_path                 path to store intermediate result.
  b40_adata_path            path to the AnnData object storing spatial RNA data at 10 μm resolution.
  b4_adata_path             path to the AnnData object storing spatial RNA data at 1 μm resolution.
  split_num                 the tissue is split to split_num*split_num regions.
  ```


### Train stLocation
```python
train_model(work_path,start=0, num_epochs = 30000)
```
#### Description
  ```
  Train stLocation model.
  ```
#### Parameters  
  ```
  work_path                 path to store intermediate result.
  start                     start index of region to process.  
  num_epochs                number of epochs.
  ```


### Get result
```python
get_adata(work_path, b4_adata_path)
```
#### Description
  ```
  Get output of stLocation model.
  ```
#### Parameters  
  ``` 
  work_path                 path to store intermediate result.
  b4_adata_path             path to the AnnData object storing spatial RNA data at 1 μm resolution.
  ```
#### Return 
  ```
  adata                     result of stLocation
  score_lst                 list of score for each cell
  ```



## Data
All propcessed data could be downloaded from the shared link: https://drive.google.com/drive/folders/11djR7vxr6Y1VTpz2EVJKH3MvJNGm9VoR?usp=share_link  

## Example workflow
### Utilize stVAE to deconvolve the cell-type composition of spots at 10um resolution
See the stVAE tutorial at: https://github.com/lichen2018/stVAE/tree/main. The output of stVAE should be stored in the work path, which would also store the intermediate result of stLocation.
### Calculate score matrix
```python
from stLocation.get_score_matrix import generate_score_matrix
from stLocation.generate_cluster_center import generate_cluster_centers
from stLocation.generate_anchor import generate_anchor
from stLocation.get_cell import train_model
from stLocation.process_result import get_adata
spatial_data_path = './'
work_path = spatial_data_path+'690/'
b4_adata_path = spatial_data_path+'b4_in_tissue.h5ad'
unsplice_b4_adata_path = spatial_data_path+'unsplice_in_tissue.h5ad'
b40_adata_path = spatial_data_path+'b40_in_tissue.h5ad'
scores = generate_score_matrix(work_path, b4_adata_path, unsplice_b4_adata_path, b40_adata_path)
# Get cluster centers of score matrix
generate_cluster_centers(work_path, split_num = 4, max_iter=80)
# Generate anchors to indicate cellular positions
generate_anchor(work_path, b40_adata_path, b4_adata_path, split_num = 7)
# Train stLocation
train_model(work_path)
# Get output of stLocation
adata, score_lst = get_adata(work_path, b4_adata_path)
```
