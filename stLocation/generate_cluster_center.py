import numpy as np
import os
import time
import multiprocessing
from numba import njit, prange, set_num_threads
from numba import cuda, float32
import cupy as cp
import math
from pathlib import Path
# Set CPU threads (for non-GPU parts, if any)
set_num_threads(max(1, multiprocessing.cpu_count() - 2))
np.set_printoptions(suppress=True)
@cuda.jit(fastmath=True, device=True)
def gpu_exp(x):
    return cuda.libdevice.exp(x)
# --------------------------
# 1. Kernel Optimization (Shared Memory + Float32 + Merged Access)
# --------------------------
@cuda.jit(fastmath=True)  # Enable fast math (reduces precision checks, 10-20% speedup)
def weighted_kernel_density_gpu_ultra(
    shifted_points,  # Shape: (n_points, 2) - float32, C-order (contiguous)
    X,              # Shape: (n_points, 2) - float32, C-order (contiguous)
    scores,         # Shape: (n_points,) - float32, C-order (contiguous)
    neg_inv_bandwidth_sq_2,  # Precomputed: -1/(2*bandwidth2) (float32)
    new_shifted_points,       # Shape: (n_points, 2) - float32, C-order (contiguous)
    n_points        # Total points (int32)
):
    # Thread index (1D grid: each thread processes 1 point)
    i = cuda.grid(1)
    if i >= n_points:
        return

    thread_id_in_block = cuda.threadIdx.x
    threads_per_block = cuda.blockDim.x

    SHARED_X_CHUNK_SIZE = 128
    shared_X = cuda.shared.array((SHARED_X_CHUNK_SIZE, 3), dtype=float32)  # Pad to 3 columns
    shared_scores = cuda.shared.array(SHARED_X_CHUNK_SIZE, dtype=float32)

    px = shifted_points[i, 0]
    py = shifted_points[i, 1]

    sum_wx = float32(0.0)
    sum_wy = float32(0.0)
    sum_weights = float32(0.0)

    num_chunks = (n_points + SHARED_X_CHUNK_SIZE - 1) // SHARED_X_CHUNK_SIZE
    for chunk in range(num_chunks):
        chunk_start = chunk * SHARED_X_CHUNK_SIZE
        chunk_end = min(chunk_start + SHARED_X_CHUNK_SIZE, n_points)
        num_in_chunk = chunk_end - chunk_start

        # Each thread loads 1 row of X (2 features) + 1 score (coalesced access)
        if thread_id_in_block < num_in_chunk:
            j = chunk_start + thread_id_in_block
            # Load X[j,0] and X[j,1] in one contiguous read (no d loop!)
            shared_X[thread_id_in_block, 0] = X[j, 0]
            shared_X[thread_id_in_block, 1] = X[j, 1]
            shared_scores[thread_id_in_block] = scores[j]

        # Wait for all threads to load (critical for shared memory consistency)
        cuda.syncthreads()

        for j_in_chunk in range(num_in_chunk):
            # Load from shared memory (no bank conflicts due to padding)
            xj = shared_X[j_in_chunk, 0]
            yj = shared_X[j_in_chunk, 1]
            sj = shared_scores[j_in_chunk]

            # Compute squared distance (FULLY UNROLLED for n_features=2)
            dx = xj - px
            dy = yj - py
            dist_sq = dx * dx + dy * dy
            weight = gpu_exp(dist_sq * neg_inv_bandwidth_sq_2) * sj

            sum_weights += weight
            sum_wx += xj * weight
            sum_wy += yj * weight

        cuda.syncthreads()
    if sum_weights < float32(1e-10):
        new_shifted_points[i, 0] = px
        new_shifted_points[i, 1] = py
    else:
        inv_sum_w = float32(1.0) / sum_weights
        new_shifted_points[i, 0] = sum_wx * inv_sum_w
        new_shifted_points[i, 1] = sum_wy * inv_sum_w



def gpu_ultra_fast_weighted_mean_shift(X, scores, bandwidth, max_iter=100, tol=1e-3, use_float32=True):
    X_cpu = np.ascontiguousarray(X, dtype=np.float32)
    scores = np.ascontiguousarray(scores, dtype=np.float16)
    n_points, n_features = X_cpu.shape
    bandwidth_sq = bandwidth ** 2
    neg_inv_bandwidth_sq_2 = -1.0 / (2 * bandwidth_sq)  # Precompute negative sign
    neg_inv_bandwidth_sq_2 = np.float16(neg_inv_bandwidth_sq_2)

    # 3. GPU memory allocation (avoid repeated allocations in loop)
    X_gpu = cuda.to_device(X_cpu)
    scores_gpu = cuda.to_device(scores)
    shifted_points_gpu = cuda.to_device(X)
    new_shifted_points_gpu = cuda.device_array_like(shifted_points_gpu)

    # --------------------------
    # Thread Configuration (Match Your GPU Architecture!)
    # --------------------------
    # Get GPU specs (auto-tune for your hardware)
    dev = cuda.get_current_device()
    max_threads_per_block = dev.MAX_THREADS_PER_BLOCK  # e.g., 1024 for A100, 512 for RTX 3090
    threads_per_block = min(128, max_threads_per_block)  # Match SHARED_X_CHUNK_SIZE=128
    blocks_per_grid = (n_points + threads_per_block - 1) // threads_per_block

    # Optimize blocks_per_grid for GPU SM count (maximize occupancy)
    num_sms = dev.MULTIPROCESSOR_COUNT
    blocks_per_grid = max(blocks_per_grid, num_sms * 2)    
    iter_count = 0
    converged = False
    stream = cp.cuda.Stream()

    shifted_points_cpu = X
    prev_shifted_points_gpu = cuda.device_array_like(shifted_points_gpu)
    while iter_count < max_iter:
        with stream:
            weighted_kernel_density_gpu_ultra[blocks_per_grid, threads_per_block](
                shifted_points_gpu, X_gpu, scores_gpu,
                neg_inv_bandwidth_sq_2, new_shifted_points_gpu,
                np.int32(n_points)
            )      
        stream.synchronize()
        prev_shifted_points_gpu, shifted_points_gpu = shifted_points_gpu, new_shifted_points_gpu
        iter_count += 1
        print(iter_count)

    stream.synchronize()

    T1 = time.time()

    shifted_points_cpu = shifted_points_gpu.copy_to_host()
    T2 =time.time()
    print('gpu time:%s ms' % ((T2 - T1)*1000))
    from sklearnex import patch_sklearn
    patch_sklearn()
    from sklearn.cluster import DBSCAN
    db = DBSCAN(
        eps=bandwidth / 2, 
        min_samples=3, 
        n_jobs=-1
    ).fit(shifted_points_cpu.astype(np.float16))
    labels = db.labels_

    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels != -1]
    cluster_centers = np.empty((len(unique_labels), n_features), dtype=np.float16)

    for i, k in enumerate(unique_labels):
        mask = labels == k
        cluster_centers[i] = np.average(X_cpu[mask], axis=0, weights=scores[mask])

    return labels, cluster_centers



def generate_cluster_centers(work_path, split_num = 2, max_iter=100):
    score_matrix_path = work_path + 'score_matrix_files/'
    output_path = work_path + 'cluster_center_files/'
    if not os.path.exists(output_path):
        os.makedirs(output_path)


    bandwidth =3
    for x_idx in range(split_num):
        for y_idx in range(split_num):
            file = Path(score_matrix_path+str(x_idx)+'_'+str(y_idx)+'_coor.npy')
            if file.exists():
                coordinates = np.load(score_matrix_path+str(x_idx)+'_'+str(y_idx)+'_coor.npy')

                filter_scores = np.load(score_matrix_path+str(x_idx)+'_'+str(y_idx)+'_score.npy')

                labels, cluster_centers = gpu_ultra_fast_weighted_mean_shift(coordinates, filter_scores, bandwidth, max_iter)

                np.save(output_path+str(x_idx)+'_'+str(y_idx)+'_cluster_center.npy', cluster_centers)
