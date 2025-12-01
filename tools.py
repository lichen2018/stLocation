"""Functions to segment regions of a slice by UMI density.
"""

from scipy import sparse
from collections import Counter
from typing import List, Dict, Optional, Tuple, Union
import math
import cv2
import numpy as np
from anndata import AnnData
from kneed import KneeLocator
from scipy.sparse import csr_matrix, issparse, lil_matrix, spmatrix
from sklearn import cluster
from typing_extensions import Literal
from skimage import filters
from functools import partial


"""Utility functions for cell segmentation.
"""
from typing import Optional, Union

from scipy import signal, sparse

import plotly.graph_objects as go
from skimage.color.colorlabel import DEFAULT_COLORS
import matplotlib as mpl

import sys
import em, vi
from em import run_em
from bp import run_bp

def contours(adata: AnnData, layer: str, colors: Optional[List] = None, scale: float = 0.05) -> go.Figure:
    """Interactively display UMI density bins.

    Args:
        adata: Anndata containing aggregated UMI counts.
        layer: Layer to display
        colors: List of colors.
        scale: Scale width and height by this amount.

    Returns:
        A Plotly figure
    """

    bins = select_layer_data(adata, layer)
    if colors is None:
        colors = DEFAULT_COLORS

    figure = go.Figure()
    color_i = 0
    for bin in np.unique(bins):
        if bin > 0:
            mask = bins == bin
            mtx = mask.astype(np.uint8)
            mtx[mtx > 0] = 255
            contours = cv2.findContours(mtx, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
            for contour in contours:
                contour = contour.squeeze(1)
                figure.add_trace(
                    go.Scatter(
                        x=contour[:, 0],
                        y=-contour[:, 1],
                        text=str(bin),
                        line_width=0,
                        fill="toself",
                        mode="lines",
                        showlegend=False,
                        hoverinfo="text",
                        hoveron="fills",
                        fillcolor=mpl.colors.to_hex(colors[color_i % len(colors)]),
                    )
                )
            color_i += 1

    figure.update_layout(
        width=bins.shape[1] * scale,
        height=bins.shape[0] * scale,
        xaxis=dict(showgrid=False, visible=False),
        yaxis=dict(showgrid=False, visible=False),
        margin=dict(l=0, r=0, t=0, b=0),
    )

    return figure

def circle(k: int) -> np.ndarray:
    """Draw a circle of diameter k.

    Args:
        k: Diameter

    Returns:
        8-bit unsigned integer Numpy array with 1s and 0s

    Raises:
        ValueError: if `k` is even or less than 1
    """
    if k < 1 or k % 2 == 0:
        raise ValueError(f"`k` must be odd and greater than 0.")

    r = (k - 1) // 2
    return cv2.circle(np.zeros((k, k), dtype=np.uint8), (r, r), r, 1, -1)


def knee_threshold(X: np.ndarray, n_bins: int = 256, clip: int = 5) -> float:
    """Find the knee thresholding point of an arbitrary array.

    Note:
        This function does not find the actual knee of X. It computes a
        value to be used to threshold the elements of X by finding the knee of
        the cumulative counts.

    Args:
        X: Numpy array of values
        n_bins: Number of bins to use if `X` is a float array.

    Returns:
        Knee
    """
    # Check if array only contains integers.
    _X = X.astype(int)
    if np.array_equal(X, _X):
        x = np.sort(np.unique(_X))
    else:
        x = np.linspace(X.min(), X.max(), n_bins)
    y = np.array([(X <= val).sum() for val in x]) / X.size

    x = x[clip:]
    y = y[clip:]

    kl = KneeLocator(x, y, curve="concave")
    return kl.knee


def gaussian_blur(X: np.ndarray, k: int) -> np.ndarray:
    """Gaussian blur

    This function is not designed to be called directly. Use :func:`conv2d`
    with `mode="gauss"` instead.

    Args:
        X: UMI counts per pixel.
        k: Radius of gaussian blur.

    Returns:
        Blurred array
    """
    return cv2.GaussianBlur(src=X.astype(float), ksize=(k, k), sigmaX=0, sigmaY=0)


def median_blur(X: np.ndarray, k: int) -> np.ndarray:
    """Median blur

    This function is not designed to be called directly. Use :func:`conv2d`
    with `mode="median"` instead.

    Args:
        X: UMI counts per pixel.
        k: Radius of median blur.

    Returns:
        Blurred array
    """
    return cv2.medianBlur(src=X.astype(np.uint8), ksize=k)


def conv2d(
    X: np.ndarray, k: int, mode: Literal["gauss", "median", "circle", "square"], bins: Optional[np.ndarray] = None
) -> np.ndarray:
    """Convolve an array with the specified kernel size and mode.

    Args:
        X: The array to convolve.
        k: Kernel size. Must be odd.
        mode: Convolution mode. Supported modes are:
            gauss:
            circle:
            square:
        bins: Convolve per bin. Zeros are ignored.

    Returns:
        The convolved array

    Raises:
        ValueError: if `k` is even or less than 1, or if `mode` is not a
            valid mode, or if `bins` does not have the same shape as `X`
    """
    if k < 1 or k % 2 == 0:
        raise ValueError(f"`k` must be odd and greater than 0.")
    if mode not in ("median", "gauss", "circle", "square"):
        raise ValueError(f'`mode` must be one of "median", "gauss", "circle", "square"')
    if bins is not None and X.shape != bins.shape:
        raise ValueError("`bins` must have the same shape as `X`")
    if k == 1:
        return X

    def _conv(_X):
        if mode == "gauss":
            return gaussian_blur(_X, k)
        if mode == "median":
            return median_blur(_X, k)
        kernel = np.ones((k, k), dtype=np.uint8) if mode == "square" else circle(k)
        return signal.convolve2d(_X, kernel, boundary="symm", mode="same")

    if bins is not None:
        conv = np.zeros(X.shape)
        for label in np.unique(bins):
            if label > 0:
                mask = bins == label
                conv[mask] = _conv(X * mask)[mask]
        return conv
    return _conv(X)


def scale_to_01(X: np.ndarray) -> np.ndarray:
    """Scale an array to [0, 1].

    Args:
        X: Array to scale

    Returns:
        Scaled array
    """
    return (X - X.min()) / (X.max() - X.min())


def scale_to_255(X: np.ndarray) -> np.ndarray:
    """Scale an array to [0, 255].

    Args:
        X: Array to scale

    Returns:
        Scaled array
    """
    return scale_to_01(X) * 255


def mclose_mopen(mask: np.ndarray, k: int, square: bool = False) -> np.ndarray:
    """Perform morphological close and open operations on a boolean mask.

    Args:
        X: Boolean mask
        k: Kernel size
        square: Whether or not the kernel should be square

    Returns:
        New boolean mask with morphological close and open operations performed.

    Raises:
        ValueError: if `k` is even or less than 1
    """
    if k < 1 or k % 2 == 0:
        raise ValueError(f"`k` must be odd and greater than 0.")

    kernel = np.ones((k, k), dtype=np.uint8) if square else circle(k)
    mclose = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    mopen = cv2.morphologyEx(mclose, cv2.MORPH_OPEN, kernel)

    return mopen.astype(bool)


def apply_threshold(X: np.ndarray, k: int, threshold: Optional[Union[float, np.ndarray]] = None) -> np.ndarray:
    """Apply a threshold value to the given array and perform morphological close
    and open operations.

    Args:
        X: The array to threshold
        k: Kernel size of the morphological close and open operations.
        threshold: Threshold to apply. By default, the knee is used.

    Returns:
        A boolean mask.
    """
    # Apply threshold and mclose,mopen
    threshold = threshold if threshold is not None else knee_threshold(X)
    print(f"threshold: {threshold}")
    mask = mclose_mopen(X >= threshold, k)
    return mask


def gen_new_layer_key(layer_name: str, key: str, sep: str = "_") -> str:
        if layer_name == "":
            return key
        if layer_name[-1] == sep:
            return layer_name + key
        return sep.join([layer_name, key])

def select_layer_data(
    adata: AnnData, layer: str, copy: bool = False, make_dense: bool = False
) -> Union[np.ndarray, sparse.spmatrix]:
    if layer is None:
        layer = "X"
    res_data = None
    if layer == "X":
        res_data = adata.X
    else:
        res_data = adata.layers[layer]
    if make_dense and sparse.issparse(res_data):
        return res_data.toarray()
    if copy:
        return res_data.copy()
    return res_data

def set_layer_data(
    adata: AnnData, layer: str, vals: np.ndarray, var_indices: Optional[np.ndarray] = None, replace: bool = False
):
    # Mostly for testing
    if replace:
        adata.layers[layer] = vals
        return

    if var_indices is None:
        var_indices = slice(None)
    if layer == "X":
        adata.X[:, var_indices] = vals
    elif layer in adata.layers:
        adata.layers[layer][:, var_indices] = vals
    else:
        # layer does not exist in adata
        # ignore var_indices and set values as a new layer
        adata.layers[layer] = vals

def bin_indices(coords: np.ndarray, coord_min: float, binsize: int = 50) -> int:
    """Take a DNB coordinate, the mimimum coordinate and the binsize, calculate the index of bins for the current
    coordinate.

    Parameters
    ----------
        coord: `float`
            Current x or y coordinate.
        coord_min: `float`
            Minimal value for the current x or y coordinate on the entire tissue slide measured by the spatial
            transcriptomics.
        binsize: `float`
            Size of the bins to aggregate data.

    Returns
    -------
        num: `int`
            The bin index for the current coordinate.
    """
    num = np.floor((coords - coord_min) / binsize)
    return num.astype(np.uint32)

def bin_matrix(X: Union[np.ndarray, spmatrix], binsize: int) -> Union[np.ndarray, csr_matrix]:
    """Bin a matrix.

    Args:
        X: Dense or sparse matrix.
        binsize: Bin size

    Returns:
        Dense or spares matrix, depending on what the input was.
    """
    shape = (math.ceil(X.shape[0] / binsize), math.ceil(X.shape[1] / binsize))

    def _bin_sparse(X):
        nz = X.nonzero()
        x, y = nz
        data = X[nz].A.flatten()
        x_bin = bin_indices(x, 0, binsize)
        y_bin = bin_indices(y, 0, binsize)
        return csr_matrix((data, (x_bin, y_bin)), shape=shape, dtype=X.dtype)

    def _bin_dense(X):
        binned = np.zeros(shape, dtype=X.dtype)
        for x in range(X.shape[0]):
            x_bin = bin_indices(x, 0, binsize)
            for y in range(X.shape[1]):
                y_bin = bin_indices(y, 0, binsize)
                binned[x_bin, y_bin] += X[x, y]
        return binned

    if issparse(X):
        return _bin_sparse(X)
    return _bin_dense(X)


def _replace_labels(labels: np.ndarray, mapping: Dict[int, int]) -> np.ndarray:
    """Replace labels according to mapping.

    Args:
        labels: Numpy array containing integer labels.
        mapping: Dictionary mapping from labels to labels.

    Returns:
        Replaced labels
    """
    replacement = np.full(labels.max() + 1, -1, dtype=int)
    for from_label, to_label in mapping.items():
        replacement[from_label] = to_label

    new_labels = labels.copy()
    for i in range(labels.shape[0]):
        for j in range(labels.shape[1]):
            new_label = replacement[labels[i, j]]
            if new_label >= 0:
                new_labels[i, j] = new_label
    return new_labels



def _create_spatial_adjacency(shape: Tuple[int, int]) -> csr_matrix:
    """Create a sparse adjacency matrix for a 2D grid graph of specified shape.
    https://stackoverflow.com/a/16342639

    Args:
        shape: Shape of grid

    Returns:
        A sparse adjacency matrix
    """
    n_rows, n_cols = shape
    n_nodes = n_rows * n_cols
    adjacency = lil_matrix((n_nodes, n_nodes))
    for r in range(n_rows):
        for c in range(n_cols):
            i = r * n_cols + c
            # Two inner diagonals
            if c > 0:
                adjacency[i - 1, i] = adjacency[i, i - 1] = 1
            # Two outer diagonals
            if r > 0:
                adjacency[i - n_cols, i] = adjacency[i, i - n_cols] = 1
    return adjacency.tocsr()


def _schc(X: np.ndarray, distance_threshold: Optional[float] = None) -> np.ndarray:
    """Spatially-constrained hierarchical clustering.

    Perform hierarchical clustering with Ward linkage on an array
    containing UMI counts per pixel. Spatial constraints are
    imposed by limiting the neighbors of each node to immediate 4
    pixel neighbors.

    This function runs in two steps. First, it computes a Ward linkage tree
    by calling :func:`sklearn.cluster.ward_tree`, with `return_distance=True`,
    which yields distances between clusters. then, if `distance_threshold` is not
    provided, a dynamic threshold is calculated by finding the inflection (knee)
    of the distance (x) vs number of clusters (y) line using the top 1000
    distances, making the assumption that for the vast majority of cases, there
    will be less than 1000 density clusters.

    Args:
        X: UMI counts per pixel
        distance_threshold: Distance threshold for the Ward linkage
            such that clusters will not be merged if they have
            greater than this distance.

    Returns:
        Clustering result as a Numpy array of same shape, where clusters are
        indicated by integers.
    """
    print("Constructing spatial adjacency matrix.")
    adjacency = _create_spatial_adjacency(X.shape)
    X_flattened = X.flatten()
    print("Computing Ward tree.")
    children, _, n_leaves, _, distances = cluster.ward_tree(X_flattened, connectivity=adjacency, return_distance=True)

    # Find distance threshold if not provided
    if not distance_threshold:
        print("Finding dynamic distance threshold by using knee of the top 1000 distances.")
        x = np.sort(np.unique(distances))[-1000:]
        y = np.array([(distances >= val).sum() + 1 for val in x])
        kl = KneeLocator(
            x, y, curve="convex", direction="decreasing", online=True, interp_method="polynomial", polynomial_degree=10
        )
        distance_threshold = kl.knee

    n_clusters = (distances >= distance_threshold).sum() + 1
    print("Finding {n_clusters} assignments.")
    assignments = cluster._agglomerative._hc_cut(n_clusters, children, n_leaves)

    return assignments.reshape(X.shape)


def _segment_densities(
    X: Union[spmatrix, np.ndarray], k: int, dk: int, distance_threshold: Optional[float] = None
) -> np.ndarray:
    """Segment a matrix containing UMI counts into regions by UMI density.

    Args:
        X: UMI counts per pixel
        k: Kernel size for Gaussian blur
        dk: Kernel size for final dilation
        distance_threshold: Distance threshold for the Ward linkage
            such that clusters will not be merged if they have
            greater than this distance.

    Returns:
        Clustering result as a Numpy array of same shape, where clusters are
        indicated by positive integers.
    """
    # Warn on too large array
    if X.size > 5e5:
        print(
            f"Array has {X.size} elements. This may take a while and a lot of memory. "
            "Please consider condensing the array by increasing the binsize."
        )

    # Make dense and normalize.
    if issparse(X):
        print("Converting to dense matrix.")
        X = X.toarray()
    print("Normalizing matrix.")
    X = X / X.max()

    print(f"Applying Gaussian blur with k={k}.")
    X = conv2d(X, k, mode="gauss")

    # Add 1 because 0 should indicate background!
    bins = _schc(X, distance_threshold=distance_threshold) + 1

    print("Dilating labels in ascending mean density order.")
    dilated = np.zeros_like(bins)
    labels = np.unique(bins)
    for label in sorted(labels, key=lambda label: X[bins == label].mean()):
        mask = bins == label
        dilate = cv2.dilate(mask.astype(np.uint8), circle(dk))
        dilated[mclose_mopen(dilate, dk) > 0] = label
    return dilated






def merge_densities(
    adata: AnnData,
    layer: str,
    mapping: Optional[Dict[int, int]] = None,
    out_layer: Optional[str] = None,
):
    """Merge density bins either using an explicit mapping or in a semi-supervised
    way.

    Args:
        adata: Input Anndata
        layer: Layer that was used to generate density bins. Defaults to
            using `{layer}_bins`. If not present, will be taken as a literal.
        mapping: Mapping to use to transform bins
        out_layer: Layer to store results. Defaults to same layer as input.
    """
    # TODO: implement semi-supervised way of merging density bins
    _layer = gen_new_layer_key(layer, "bins")
    if _layer not in adata.layers:
        _layer = layer
    bins = select_layer_data(adata, _layer)
    print(f"Merging densities with mapping {mapping}.")
    replaced = _replace_labels(bins, mapping)
    set_layer_data(adata, out_layer or _layer, replaced)
    
    


def _initial_nb_params(
    X: np.ndarray, bins: Optional[np.ndarray] = None
) -> Union[Dict[str, Tuple[float, float]], Dict[int, Dict[str, Tuple[float, float]]]]:
    """Calculate initial estimates for the negative binomial mixture.

    Args:
        X: UMI counts
        bins: Density bins

    Returns:
        Dictionary containing initial `w`, `mu`, `var` parameters. If `bins` is also
        provided, the dictionary is nested with the outer dictionary containing each
        bin label as the keys. If `zero_inflated=True`, then the dictionary also contains
        a `z` key.
    """
    samples = {}
    if bins is not None:
        for label in np.unique(bins):
            if label > 0:
                samples[label] = X[bins == label]
    else:
        samples[0] = X.flatten()

    params = {}
    for label, _samples in samples.items():
        # Background must have at least 2 different values to prevent the mean from being zero.
        threshold = max(filters.threshold_otsu(_samples), 1)
        mask = _samples > threshold
        background_values = _samples[~mask]
        foreground_values = _samples[mask]
        w = np.array([_samples.size - mask.sum(), mask.sum()]) / _samples.size
        mu = np.array([background_values.mean(), foreground_values.mean()])
        var = np.array([background_values.var(), foreground_values.var()])

        # Negative binomial distribution requires variance > mean
        if var[0] <= mu[0]:
            print(
                f"Bin {label} estimated variance of background ({var[0]:.2e}) is less than the mean ({mu[0]:.2e}). "
                "Initial variance will be arbitrarily set to 1.1x of the mean. "
                "This is usually due to extreme sparsity. Please consider increasing `k` or using "
                "the zero-inflated distribution."
            )
            var[0] = mu[0] * 1.1
        if var[1] <= mu[1]:
            print(
                f"Bin {label} estimated variance of foreground ({var[1]:.2e}) is less than the mean ({mu[1]:.2e}). "
                "Initial variance will be arbitrarily set to 1.1x of the mean. "
                "This is usually due to extreme sparsity. Please consider increasing `k` or using "
                "the zero-inflated distribution."
            )
            var[1] = mu[1] * 1.1
        params[label] = dict(w=tuple(w), mu=tuple(mu), var=tuple(var))
    return params[0] if bins is None else params






def _score_pixels(
    X: Union[spmatrix, np.ndarray],
    k: int,
    method: Literal["gauss", "moran", "EM", "EM+gauss", "EM+BP", "VI+gauss", "VI+BP"],
    moran_kwargs: Optional[dict] = None,
    em_kwargs: Optional[dict] = None,
    vi_kwargs: Optional[dict] = None,
    bp_kwargs: Optional[dict] = None,
    certain_mask: Optional[np.ndarray] = None,
    bins: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Score each pixel by how likely it is a cell. Values returned are in
    [0, 1].

    Args:
        X: UMI counts per pixel as either a sparse or dense array.
        k: Kernel size for convolution.
        method: Method to use. Valid methods are:
            gauss: Gaussian blur
            moran: Moran's I based method
            EM: EM algorithm to estimate cell and background expression
                parameters.
            EM+gauss: Negative binomial EM algorithm followed by Gaussian blur.
            EM+BP: EM algorithm followed by belief propagation to estimate the
                marginal probabilities of cell and background.
            VI+gauss: Negative binomial VI algorithm followed by Gaussian blur.
                Note that VI also supports the zero-inflated negative binomial (ZINB)
                by providing `zero_inflated=True`.
            VI+BP: VI algorithm followed by belief propagation. Note that VI also
                supports the zero-inflated negative binomial (ZINB) by providing
                `zero_inflated=True`.
        moran_kwargs: Keyword arguments to the :func:`moran.run_moran` function.
        em_kwargs: Keyword arguments to the :func:`em.run_em` function.
        bp_kwargs: Keyword arguments to the :func:`bp.run_bp` function.
        certain_mask: A boolean Numpy array indicating which pixels are certain
            to be occupied, a-priori. For example, if nuclei staining is available,
            this would be the nuclei segmentation mask.
        bins: Pixel bins to segment separately. Only takes effect when the EM
            algorithm is run.

    Returns:
        [0, 1] score of each pixel being a cell.

    Raises:
        SegmentationError: If `bins` and/or `certain_mask` was provided but
            their sizes do not match `X`
    """
    if method.lower() not in ("gauss", "moran", "em", "em+gauss", "em+bp", "vi+gauss", "vi+bp"):
        print(f"Unknown method `{method}`")
    if certain_mask is not None and X.shape != certain_mask.shape:
        print("`certain_mask` does not have the same shape as `X`")
    if bins is not None and X.shape != bins.shape:
        print("`bins` does not have the same shape as `X`")

    method = method.lower()
    moran_kwargs = moran_kwargs or {}

    if em_kwargs[0] == None:
        em_kwargs = {}
    else:
        em_kwargs = em_kwargs
    if vi_kwargs[0] == None:
        vi_kwargs = {}
    else:
        vi_kwargs = vi_kwargs
        
    if bp_kwargs[0] == None:
        bp_kwargs = {}
    else:
        bp_kwargs = bp_kwargs

    if moran_kwargs and "moran" not in method:
        print(f"`moran_kwargs` will be ignored.")
    if em_kwargs and "em" not in method:
        print(f"`em_kwargs` will be ignored.")
    if vi_kwargs and "vi" not in method:
        print(f"`vi_kwargs` will be ignored.")
    if bp_kwargs and "bp" not in method:
        print(f"`bp_kwargs` will be ignored.")

    # Convert X to dense array
    if issparse(X):
        print("Converting X to dense array.")
        X = X.toarray()

    # All methods require some kind of 2D convolution to start off
    print(f"Computing 2D convolution with k={k}.")
    res = conv2d(X, k, mode="gauss" if method in ("gauss", "moran") else "circle", bins=bins)

    # All methods other than gauss requires EM
    if method == "gauss":
        # For just "gauss" method, we should rescale to [0, 1] because all the
        # other methods eventually produce an array of [0, 1] values.
        res = scale_to_01(res)
    else:
        # Obtain initial parameter estimates with Otsu thresholding.
        # These may be overridden by providing the appropriate kwargs.
        nb_kwargs = dict(params=_initial_nb_params(res, bins=bins))
        if "em" in method:
            nb_kwargs.update(em_kwargs)
            print(f"Running EM with kwargs {nb_kwargs}.")
            em_results = run_em(res, bins=bins, **nb_kwargs)
            conditional_func = partial(em.conditionals, em_results=em_results, bins=bins)
        else:
            nb_kwargs.update(vi_kwargs)
            print(f"Running VI with kwargs {nb_kwargs}.")
            vi_results = run_vi(res, bins=bins, **nb_kwargs)
            conditional_func = partial(vi.conditionals, vi_results=vi_results, bins=bins)

        if "bp" in method:
            print("Computing conditionals.")
            background_cond, cell_cond = conditional_func(res)
            if certain_mask is not None:
                background_cond[certain_mask] = 1e-2
                cell_cond[certain_mask] = 1 - (1e-2)
            print(f"Running BP with kwargs {bp_kwargs}.")
            res = run_bp(background_cond, cell_cond, **bp_kwargs)
        else:
            print("Computing confidences.")
            res = em.confidence(res, em_results=em_results, bins=bins)
            if certain_mask is not None:
                res = np.clip(res + certain_mask, 0, 1)

        if "gauss" in method:
            print("Computing Gaussian blur.")
            res = conv2d(res, k, mode="gauss", bins=bins)
    return res



    
    
def score_and_mask_pixels(
    adata: AnnData,
    layer: str,
    k: int,
    method: Literal["gauss", "moran", "EM", "EM+gauss", "EM+BP", "VI+gauss", "VI+BP"],
    moran_kwargs: Optional[dict] = None,
    em_kwargs: Optional[dict] = None,
    vi_kwargs: Optional[dict] = None,
    bp_kwargs: Optional[dict] = None,
    threshold: Optional[float] = None,
    use_knee: Optional[bool] = False,
    mk: Optional[int] = None,
    bins_layer: Optional[Union[Literal[False], str]] = None,
    certain_layer: Optional[str] = None,
    scores_layer: Optional[str] = None,
    mask_layer: Optional[str] = None,
):
    """Score and mask pixels by how likely it is occupied.

    Args:
        adata: Input Anndata
        layer: Layer that contains UMI counts to use
        k: Kernel size for convolution
        method: Method to use to obtain per-pixel scores. Valid methods are:
            gauss: Gaussian blur
            moran: Moran's I based method
            EM: EM algorithm to estimate cell and background expression
                parameters.
            EM+gauss: Negative binomial EM algorithm followed by Gaussian blur.
            EM+BP: EM algorithm followed by belief propagation to estimate the
                marginal probabilities of cell and background.
            VI+gauss: Negative binomial VI algorithm followed by Gaussian blur.
                Note that VI also supports the zero-inflated negative binomial (ZINB)
                by providing `zero_inflated=True`.
            VI+BP: VI algorithm followed by belief propagation. Note that VI also
                supports the zero-inflated negative binomial (ZINB) by providing
                `zero_inflated=True`.
        moran_kwargs: Keyword arguments to the :func:`moran.run_moran` function.
        em_kwargs: Keyword arguments to the :func:`em.run_em` function.
        bp_kwargs: Keyword arguments to the :func:`bp.run_bp` function.
        threshold: Score cutoff, above which pixels are considered occupied.
            By default, a threshold is automatically determined by using
            Otsu thresholding.
        use_knee: Whether to use knee point as threshold. By default is False. If
            True, threshold would be ignored.
        mk: Kernel size of morphological open and close operations to reduce
            noise in the mask. Defaults to `k`+2 if EM or VI is run. Otherwise,
            defaults to `k`-2.
        bins_layer: Layer containing assignment of pixels into bins. Each bin
            is considered separately. Defaults to `{layer}_bins`. This can be
            set to `False` to disable binning, even if the layer exists.
        certain_layer: Layer containing a boolean mask indicating which pixels are
            certain to be occupied. If the array is not a boolean array, it is
            casted to boolean.
        scores_layer: Layer to save pixel scores before thresholding. Defaults
            to `{layer}_scores`.
        mask_layer: Layer to save the final mask. Defaults to `{layer}_mask`.
    """
    X = select_layer_data(adata, layer, make_dense=True)
    certain_mask = None
    if certain_layer:
        certain_mask = select_layer_data(adata, certain_layer).astype(bool)
    bins = None
    if bins_layer is not False:
        bins_layer = bins_layer or gen_new_layer_key(layer, "bins")
        if bins_layer in adata.layers:
            bins = select_layer_data(adata, bins_layer)
    method = method.lower()
    print(f"Scoring pixels with {method} method.")
    scores = _score_pixels(X, k, method, moran_kwargs, em_kwargs, vi_kwargs, bp_kwargs, certain_mask, bins)
    scores_layer = scores_layer or gen_new_layer_key(layer, "scores")
    set_layer_data(adata, scores_layer, scores)

    if not threshold and not use_knee:
        print("Finding Otsu threshold.")
        threshold = filters.threshold_otsu(scores)
        print(f"Applying threshold {threshold}.")

    mk = mk or (k + 2 if any(m in method for m in ("em", "vi")) else max(k - 2, 3))
    if use_knee:
        threshold = None
    mask = apply_threshold(scores, mk, threshold)
    if certain_layer:
        mask += certain_mask
    mask_layer = mask_layer or gen_new_layer_key(layer, "mask")
    set_layer_data(adata, mask_layer, mask)