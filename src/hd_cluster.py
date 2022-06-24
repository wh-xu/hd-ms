from distutils.log import error
import time, logging, copy
import math
from typing import Callable, Optional, Tuple

from tqdm import tqdm

import numpy as np
np.random.seed(0)

import numba as nb
from numba import cuda
from numba.typed import List

import pandas as pd

import scipy.sparse as ss
import config

from joblib import Parallel, delayed

from sklearn.cluster import DBSCAN
import cupy as cp
import cuml

import rmm


rmm.reinitialize(pool_allocator=False, managed_memory=True)

# cp.cuda.set_allocator(rmm.rmm_cupy_allocator)
# cuda.set_memory_manager(rmm.RMMNumbaManager)
# rmm.mr.set_current_device_resource(rmm.mr.ManagedMemoryResource())
# cp.cuda.set_allocator(cp.cuda.MemoryPool(cp.cuda.malloc_managed).malloc)


def gen_lvs(D: int, Q: int):
    base = np.ones(D)
    base[:D//2] = -1.0
    l0 = np.random.permutation(base)
    levels = list()
    for i in range(Q+1):
        flip = int(int(i/float(Q) * D) / 2)
        li = np.copy(l0)
        li[:flip] = l0[:flip] * -1
        levels.append(list(li))
    return cp.array(levels, dtype=cp.float32).ravel()


def gen_idhvs(D: int, totalFeatures: int, flip_factor):
    nFlip = int(D//flip_factor)

    mu = 0
    sigma = 1
    bases = np.random.normal(mu, sigma, D)

    import copy
    generated_hvs = [copy.copy(bases)]

    for _ in range(totalFeatures-1):        
        idx_to_flip = np.random.randint(0, D, size=nFlip)
        bases[idx_to_flip] *= (-1)
        generated_hvs.append(copy.copy(bases))

    return cp.array(generated_hvs, dtype=cp.float32).ravel()


def gen_lv_id_hvs(
    D: int,
    Q: int,
    bin_len: int
):
    flip_factor = 2
    lv_hvs = gen_lvs(D, Q)
    lv_hvs = cuda_bit_packing(lv_hvs, Q+1, D)
    id_hvs = gen_idhvs(D, bin_len, flip_factor)
    id_hvs = cuda_bit_packing(id_hvs, bin_len, D)
    return lv_hvs, id_hvs


def cuda_bit_packing(orig_vecs, N, D):
    pack_len = (D+32-1)//32
    packed_vecs = cp.zeros(N * pack_len, dtype=cp.uint32)
    packing_cuda_kernel = cp.RawKernel(r'''
                    extern "C" __global__
                    void packing(unsigned int* output, float* arr, int origLength, int packLength, int numVec) {
                        int i = blockDim.x * blockIdx.x + threadIdx.x;
                        if (i >= origLength)
                            return;
                        for (int sample_idx = blockIdx.y; sample_idx < numVec; sample_idx += blockDim.y * gridDim.y) 
                        {
                            int tid = threadIdx.x;
                            int lane = tid % warpSize;
                            int bitPattern=0;
                            if (i < origLength)
                                bitPattern = __brev(__ballot_sync(0xFFFFFFFF, arr[sample_idx*origLength+i] > 0));
                            if (lane == 0) {
                                output[sample_idx*packLength+ (i / warpSize)] = bitPattern;
                            }
                        }
                    }
                    ''', 'packing')
    threads = 1024
    packing_cuda_kernel(((D + threads - 1) // threads, N), (threads,), (packed_vecs, orig_vecs, D, pack_len, N))

    return packed_vecs.reshape(N, pack_len)


def hd_encode_spectra_packed(csr_spectra, id_hvs_packed, lv_hvs_packed, N, D, Q, output_type):
    packed_dim = (D + 32 - 1) // 32
    encoded_spectra = cp.zeros(N * packed_dim, dtype=cp.uint32)
    
    spectra_data = cp.array(csr_spectra.data, dtype=cp.float32).ravel()
    spectra_indices = cp.array(csr_spectra.indices, dtype=cp.int32).ravel()
    spectra_indptr = cp.array(csr_spectra.indptr, dtype=cp.int32).ravel()
    
    hd_enc_lvid_packed_cuda_kernel = cp.RawKernel(r'''
                __device__ float* get2df(float* p, const int x, int y, const int stride) {
                    return (float*)((char*)p + x*stride) + y;
                }
                __device__ char get2d_bin(unsigned int* p, const int i, const int DIM, const int d) {
                    unsigned int v = ((*(p + i * ((DIM + 32-1)/32) + d/32)) >> ((32-1) - d % 32)) & 0x01;
                    if (v == 0) {
                        return -1;
                    } else {
                        return 1;
                    }
                }
                extern "C" __global__
                void hd_enc_lvid_packed_cuda(unsigned int* __restrict__ id_hvs_packed, unsigned int* __restrict__ level_hvs_packed, 
                                            int* __restrict__ feature_indices, float* __restrict__ feature_values, 
                                            int* __restrict__ csr_info, unsigned int* hv_matrix,
                                            int N, int Q, int D, int packLength) {
                    const int d = threadIdx.x + blockIdx.x * blockDim.x;
                    if (d >= D)
                        return;
                    for (int sample_idx = blockIdx.y; sample_idx < N; sample_idx += blockDim.y * gridDim.y) 
                    {
                        // we traverse [start, end-1]
                        float encoded_hv_e = 0.0;
                        unsigned int start_range = csr_info[sample_idx];
                        unsigned int end_range = csr_info[sample_idx + 1];
                        #pragma unroll 1
                        for (int f = start_range; f < end_range; ++f) {
                            // encoded_hv_e += level_hvs[((int)(feature_values[f] * Q))*D+d] * id_hvs[feature_indices[f]*D+d];
                            // encoded_hv_e += level_hvs[(int)(info.Intensity * Q) * D + d] * id_hvs[info.Idx * D + d];
                            encoded_hv_e += get2d_bin(level_hvs_packed, (int)(feature_values[f] * Q), D, d) * \
                                            get2d_bin(id_hvs_packed, feature_indices[f], D, d);
                        }
                        
                        // hv_matrix[sample_idx*D+d] = (encoded_hv_e > 0)? 1 : -1;
                        int tid = threadIdx.x;
                        int lane = tid % warpSize;
                        int bitPattern=0;
                        if (d < D)
                            bitPattern = __ballot_sync(0xFFFFFFFF, encoded_hv_e > 0);
                        if (lane == 0) {
                            hv_matrix[sample_idx * packLength + (d / warpSize)] = bitPattern;
                        }
                    }
                }
                ''', 'hd_enc_lvid_packed_cuda')
                
    threads = 1024
    max_block = cp.cuda.runtime.getDeviceProperties(0)['maxGridSize'][1]
    hd_enc_lvid_packed_cuda_kernel(((D + threads - 1) // threads, min(N, max_block)), (threads,), (id_hvs_packed, lv_hvs_packed, spectra_indices, spectra_data, spectra_indptr, encoded_spectra, N, Q, D, packed_dim))

    if output_type=='numpy':
        return encoded_spectra.reshape(N, packed_dim).get()
    elif output_type=='cupy':
        return encoded_spectra.reshape(N, packed_dim)


@cuda.jit('float32(uint32, uint32)', device=True, inline=True)
def fast_hamming_op(a, b):
    return nb.float32(cuda.libdevice.popc(a^b))

TPB = 32
TPB1 = 33

@cuda.jit('void(uint32[:,:], float32[:,:], float32[:], float32, int32, int32)')
def fast_pw_dist_cosine_mask_packed(A, D, prec_mz, prec_tol, N, pack_len):
    """
        Pair-wise cosine distance
    """
    sA = cuda.shared.array((TPB, TPB1), dtype=nb.uint32)
    sB = cuda.shared.array((TPB, TPB1), dtype=nb.uint32)

    x, y = cuda.grid(2)
    tx, ty = cuda.threadIdx.x, cuda.threadIdx.y
    bx = cuda.blockIdx.x

    tmp = nb.float32(.0)
    for i in range((pack_len+TPB-1) // TPB):
        if y < N and (i*TPB+tx) < pack_len:
            sA[ty, tx] = A[y, i*TPB+tx]
        else:
            sA[ty, tx] = .0

        if (TPB*bx+ty) < N and (i*TPB+tx) < pack_len:
            sB[ty, tx] = A[TPB*bx+ty, i*TPB+tx]
        else:
            sB[ty, tx] = .0  
        cuda.syncthreads()

        for j in range(TPB):
            tmp += fast_hamming_op(sA[ty, j], sB[tx, j])

        cuda.syncthreads()

    if x<N and y<N and y>x:
        if cuda.libdevice.fabsf((prec_mz[x]-prec_mz[y])/prec_mz[y])>=prec_tol:
            D[x,y] = 1.0
            D[y,x] = 1.0
        else:
            tmp/=(16*pack_len)
            D[x,y] = tmp
            D[y,x] = tmp


def fast_nb_cosine_dist_mask(hvs, prec_mz, prec_tol, output_type, stream=None):
    N, pack_len = hvs.shape

    # start = time.time()

    hvs_d = cp.array(hvs)
    prec_mz_d = cp.array(prec_mz.ravel())
    prec_tol_d = nb.float32(prec_tol/1e6)
    dist_d = cp.empty((N,N), dtype=cp.float32)
    # print("Data loading time: ", time.time()-start)

    TPB = 32
    threadsperblock = (TPB, TPB)
    blockspergrid_x = math.ceil(N / threadsperblock[0])
    blockspergrid_y = math.ceil(N / threadsperblock[1])
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    # start = time.time()
    fast_pw_dist_cosine_mask_packed[blockspergrid, threadsperblock]\
        (hvs_d, dist_d, prec_mz_d, prec_tol_d, N, pack_len)
    cuda.synchronize()
    # print("CUDA computing time: ", time.time()-start)

    # start = time.time()
    if output_type=='cupy':
        dist = dist_d
    else:
        dist = dist_d.get()
    del dist_d
    # print("Data fetching time: ", time.time()-start)

    return dist


def get_dim(min_mz: float, max_mz: float, bin_size: float) \
        -> Tuple[int, float, float]:
    """
    Compute the number of bins over the given mass range for the given bin
    size.

    Parameters
    ----------
    min_mz : float
        The minimum mass in the mass range (inclusive).
    max_mz : float
        The maximum mass in the mass range (inclusive).
    bin_size : float
        The bin size (in Da).

    Returns
    -------
        A tuple containing (i) the number of bins over the given mass range for
        the given bin size, (ii) the highest multiple of bin size lower than
        the minimum mass, (iii) the lowest multiple of the bin size greater
        than the maximum mass. These two final values are the true boundaries
        of the mass range (inclusive min, exclusive max).
    """
    start_dim = min_mz - min_mz % bin_size
    end_dim = max_mz + bin_size - max_mz % bin_size
    # print(start_dim, end_dim, min_mz, max_mz, bin_size, math.ceil((end_dim - start_dim) / bin_size))
    return math.ceil((end_dim - start_dim) / bin_size), start_dim, end_dim


# @nb.njit(cache=True)
def _to_csr_vector(
    spectra: pd.DataFrame, 
    min_mz: float, 
    bin_size: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mz = spectra['mz'].to_numpy()
    intensity = spectra['intensity'].to_numpy()

    indptr = np.zeros(len(mz)+1, np.int32)
    indptr[1:] = np.array([len(spec) for spec in mz], np.int32)
    indptr = np.cumsum(indptr).ravel()

    indices = np.floor((np.hstack(mz).ravel()-min_mz)/bin_size)
    data = np.hstack(intensity).ravel()
    return data, indices, indptr


from multiprocessing import shared_memory
class SharedMem:
    """A simple example class"""

    def __init__(self, name: str, data: np.ndarray=None):

        if data is None:
            self.shm_data = shared_memory.SharedMemory(name=name)
        else:
            self.name = name
            self.nbytes = data.nbytes
            self.dtype = data.dtype
            self.shape = data.shape

            try:
                self.shm_data = shared_memory.SharedMemory(name=self.name, create=True, size=self.nbytes)
                self.put(data)
            except FileExistsError:
                self.shm_data = shared_memory.SharedMemory(name=self.name, create=False, size=self.nbytes)
                self.put(data)
            except Exception as shm_e:
                raise error(shm_e)

    def get_meta(self):
        return {'name': self.name, 'nbytes': self.nbytes, 'shape': self.shape, 'dtype': self.dtype}
        
    def put(self, data: np.ndarray):
        data_shm = np.ndarray(self.shape, dtype=self.dtype, buffer=self.shm_data.buf)
        data_shm[:] = data[:]

    def gather(self, idx: list = None):
        if idx is None:
            arr = np.ndarray(self.shape, dtype=self.dtype, buffer=self.shm_data.buf)
        else:
            col_size = 1 if len(self.shape)==1 else self.shape[1]
            shape = list(self.shape)
            shape[0] = idx[1]-idx[0]

            arr = np.ndarray(
                shape, dtype=self.dtype, 
                buffer=self.shm_data.buf[idx[0]*self.dtype.itemsize*col_size: idx[1]*self.dtype.itemsize*col_size])
        return arr

    def close(self):
        self.shm_data.close()
        self.shm_data.unlink()


def encode_func_shm(
    slice_idx: tuple,
    shm_dict: dict,
    D: int,
    Q: int,
    dim: int,
    output_type: str
) -> np.ndarray:
    lv_hvs = cp.array(shm_dict['lv_hvs'].gather())
    id_hvs = cp.array(shm_dict['id_hvs'].gather())

    indptr = shm_dict['indptr'].gather([slice_idx[0], slice_idx[1]+1])

    shm_idx = [indptr[0], indptr[-1]]
    data = shm_dict['intensity'].gather(shm_idx)
    indices = shm_dict['indices'].gather(shm_idx)    

    batch_size = len(indptr)-1
    csr_vec = ss.csr_matrix(
            (data, indices, indptr-indptr[0]), (batch_size, dim), np.float32, False)

    return hd_encode_spectra_packed(csr_vec, id_hvs, lv_hvs, batch_size, D, Q, output_type)


def encode_preprocessed_spectra_shm(
    spectra_df: pd.DataFrame, 
    config: config,
    dim: int,
    lv_hvs_packed: cp.array,
    id_hvs_packed: cp.array,
    logger: logging,
    batch_size: int = 5000,
    output_type: str='numpy'
)-> List:
    # Create shared memory
    start = time.time()

    num_batch = len(spectra_df)//batch_size+1

    lv_hvs_np = cp.asnumpy(lv_hvs_packed).ravel()
    shm_lv_hvs = SharedMem(name='shm_lv_hvs', data=lv_hvs_np)

    id_hvs_np = cp.asnumpy(id_hvs_packed).ravel()
    shm_id_hvs = SharedMem(name='shm_id_hvs', data=id_hvs_np)

    intensity, indices, indptr = _to_csr_vector(spectra_df, config.min_mz, config.fragment_tol)
    spectra_df.drop(columns=['mz', 'intensity'], inplace=True)

    shm_intensity = SharedMem(name='shm_intensity', data=intensity)
    shm_indices = SharedMem(name='shm_indices', data=indices)
    shm_indptr = SharedMem(name='shm_indptr', data=indptr)
    
    shm_dict = {'lv_hvs': shm_lv_hvs, 'id_hvs': shm_id_hvs,\
        'intensity': shm_intensity, 'indices': shm_indices, 'indptr': shm_indptr}

    logger.info("Create shm in {:.4f}s".format(time.time()-start))

    start = time.time()
    with Parallel(n_jobs=config.cpu_core) as parallel_pool:
        encoded_spectra = \
            parallel_pool\
                (delayed(encode_func_shm)\
                    ([i*batch_size, min((i+1)*batch_size, len(spectra_df))], shm_dict, config.hd_dim, config.hd_Q, dim, output_type) 
                    for i in tqdm(range(num_batch)))
    logger.info("Calc. {} vectorized_spectra in {:.4f}s".format(len(encoded_spectra), time.time()-start))

    encoded_spectra = np.concatenate(encoded_spectra, dtype=np.uint32)\
        if output_type=='numpy' else encoded_spectra

    # Close shm safely
    for i in shm_dict.values():
        i.close()

    return encoded_spectra


def encode_func(
    slice_idx: tuple,
    data_dict: dict,
    D: int,
    Q: int,
    dim: int,
    output_type: str
) -> np.ndarray:
    indptr = data_dict['indptr'][slice_idx[0]: slice_idx[1]+1]
    
    data, indices = data_dict['intensity'][indptr[0]: indptr[-1]], data_dict['indices'][indptr[0]: indptr[-1]]

    batch_size = len(indptr)-1
    csr_vec = ss.csr_matrix(
            (data, indices, indptr-indptr[0]), (batch_size, dim), np.float32, False)

    lv_hvs = cp.array(data_dict['lv_hvs'])
    id_hvs = cp.array(data_dict['id_hvs'])

    return hd_encode_spectra_packed(csr_vec, id_hvs, lv_hvs, batch_size, D, Q, output_type)



def encode_preprocessed_spectra(
    spectra_df: pd.DataFrame, 
    config: config,
    dim: int,
    lv_hvs_packed: cp.array,
    id_hvs_packed: cp.array,
    logger: logging,
    batch_size: int = 5000,
    output_type: str='numpy'
)-> List:
    # Create shared memory
    start = time.time()

    num_batch = len(spectra_df)//batch_size+1

    lv_hvs = cp.asnumpy(lv_hvs_packed).ravel()
    id_hvs = cp.asnumpy(id_hvs_packed).ravel()

    intensity, indices, indptr = _to_csr_vector(spectra_df, config.min_mz, config.fragment_tol)
    spectra_df.drop(columns=['mz', 'intensity'], inplace=True)

    data_dict = {'lv_hvs': lv_hvs, 'id_hvs': id_hvs, 'intensity': intensity, 'indices': indices, 'indptr': indptr}

    encoded_spectra = [ encode_func(
        [i*batch_size, min((i+1)*batch_size, len(spectra_df))], 
        data_dict, config.hd_dim, config.hd_Q, dim, output_type) for i in tqdm(range(num_batch)) ] 
                    
    encoded_spectra = np.concatenate(encoded_spectra, dtype=np.uint32)\
        if output_type=='numpy' else encoded_spectra

    logger.info("Encode {} batches of spectra in {:.4f}s".format(len(encoded_spectra), time.time()-start))

    return encoded_spectra


def _get_bucket_idx_list(
    spectra_by_charge_df: pd.DataFrame,
    logger: logging
):
    # Get bucket list
    buckets = spectra_by_charge_df.bucket.unique()
    num_bucket = len(buckets)

    bucket_idx_arr = np.zeros((num_bucket ,2), dtype=np.int32)
    bucket_size_arr = np.zeros(num_bucket, dtype=np.int32)
    for i, b_i in enumerate(buckets):
        bucket_idx_i = (spectra_by_charge_df.bucket==b_i).to_numpy()
        bucket_idx_i = np.argwhere(bucket_idx_i==True).flatten()
        bucket_idx_arr[i, :] = [bucket_idx_i[0], bucket_idx_i[-1]]
        bucket_size_arr[i] = bucket_idx_i[-1]-bucket_idx_i[0]+1
    
    hist, bins = np.histogram(
        bucket_size_arr, bins=[0, 300, 1000, 5000, 10000, 20000, 30000], density=False)

    logger.info("There are {} buckets. Maximum bucket size = {}".format(num_bucket, max(bucket_size_arr)))
    logger.info("Bucket size distribution:")
    for i in range(len(bins)-1):
        logger.info("{:.2f}% of bucket size between {} and {}".format(hist[i]/num_bucket*100, bins[i], bins[i+1]))

    return bucket_idx_arr, bucket_size_arr


def schedule_bucket(
    spectra_by_charge_df: pd.DataFrame,
    logger: logging
):
    bucket_idx_arr, bucket_size_arr = _get_bucket_idx_list(spectra_by_charge_df, logger)

    # Sort the buckets based on their sizes
    sort_idx = np.argsort(-bucket_size_arr)
    sorted_bucket_idx_arr = bucket_idx_arr[sort_idx]

    reorder_idx = np.argsort(sort_idx)

    return {
        'sort_bucket_idx_arr': sorted_bucket_idx_arr, 
        'reorder_idx': reorder_idx}


def cluster_bucket_shm(
    bucket_slice: tuple, 
    shm_dict: dict, 
    prec_tol: float, 
    cluster_func: Callable,
    output_type: str='numpy'
):
    if bucket_slice[1]-bucket_slice[0]==0:
        return np.array([-1])
    else:
        bucket_slice[1] += 1
        bucket_hv = shm_dict['shm_hv'].gather(bucket_slice)
        bucket_prec_mz = shm_dict['shm_prec_mz'].gather(bucket_slice)
        
        # start = time.time()
        pw_dist = fast_nb_cosine_dist_mask(bucket_hv, bucket_prec_mz, prec_tol, output_type)
        # print("pw dist + mask  time ", time.time()-start)

        # start = time.time()
        cluster_func.fit(pw_dist) #
        # print("dbscan  time ", time.time()-start)
        L = cluster_func.labels_
        del pw_dist

        return L


def cluster_encoded_spectra_shm(
    spectra_by_charge_df: pd.DataFrame,
    encoded_spectra_hv: np.array,
    config: config,
    logger: logging
):
    # Save data to shared memory
    start = time.time()
    logger.info("Copying {} enc. hv to shared_mem...".format(encoded_spectra_hv.shape))
    shm_hv = SharedMem(name='shm_hv', data=encoded_spectra_hv)
    del encoded_spectra_hv

    prec_mz = np.vstack(spectra_by_charge_df.precursor_mz).astype(np.float32)
    shm_prec_mz = SharedMem(name='shm_prec_mz', data=prec_mz)
    del prec_mz

    shm_dict = {'shm_hv': shm_hv, 'shm_prec_mz': shm_prec_mz}
    logger.info("Create shm in {:.4f}s".format(time.time()-start))

    ## Start clustering in GPU or CPU ##
    bucket_idx_dict = schedule_bucket(spectra_by_charge_df, logger)
    if config.use_gpu_cluster:
        # DBSCAN clustering on GPU
        dbscan_cluster_func = cuml.DBSCAN(
            eps=config.eps, min_samples=2, metric='precomputed',
            calc_core_sample_indices=False, output_type='numpy')
        cluster_device = 'GPU'
    else:
        # DBSCAN clustering on CPU
        dbscan_cluster_func = DBSCAN(eps=config.eps, min_samples=2, metric='precomputed', n_jobs=config.cpu_core)
        cluster_device = 'CPU'
   
    start = time.time()
    with Parallel(n_jobs=config.cpu_core_cluster, batch_size='auto') as parallel_pool:
        cluster_labels = parallel_pool(
            delayed(cluster_bucket_shm)(
                bucket_slice=b_slice_i, 
                shm_dict=shm_dict,
                prec_tol=config.precursor_tol[0], 
                cluster_func=dbscan_cluster_func,
                output_type='cupy' if config.use_gpu_cluster else 'numpy')
                for b_slice_i in tqdm(bucket_idx_dict['sort_bucket_idx_arr']))
    logger.info("{} clustering in {:.4f} s".format(cluster_device, time.time()-start))

    for i in shm_dict.values():
        i.close()

    cluster_labels = [cluster_labels[i] for i in bucket_idx_dict['reorder_idx']]
    return cluster_labels


def cluster_bucket(
    bucket_slice: tuple, 
    data_dict: dict, 
    prec_tol: float, 
    cluster_func: Callable,
    output_type: str='numpy'
):
    if bucket_slice[1]-bucket_slice[0]==0:
        return np.array([-1])
    else:
        bucket_slice[1] += 1
        bucket_hv = data_dict['hv'][bucket_slice[0]: bucket_slice[1]]
        bucket_prec_mz = data_dict['prec_mz'][bucket_slice[0]: bucket_slice[1]]

        pw_dist = fast_nb_cosine_dist_mask(bucket_hv, bucket_prec_mz, prec_tol, output_type)
        cluster_func.fit(pw_dist) #
        L = cluster_func.labels_
        del pw_dist

        return L


def cluster_encoded_spectra(
    spectra_by_charge_df: pd.DataFrame,
    encoded_spectra_hv: np.array,
    config: config,
    logger: logging
):
    # Save data to shared memory
    start = time.time()
    prec_mz = np.vstack(spectra_by_charge_df.precursor_mz).astype(np.float32)
    data_dict = {'hv': encoded_spectra_hv, 'prec_mz': prec_mz}

    ## Start clustering in GPU or CPU ##
    bucket_idx_dict = schedule_bucket(spectra_by_charge_df, logger)
    if config.use_gpu_cluster:
        # DBSCAN clustering on GPU
        dbscan_cluster_func = cuml.DBSCAN(
            eps=config.eps, min_samples=2, metric='precomputed',
            calc_core_sample_indices=False, output_type='numpy')
        cluster_device = 'GPU'
    else:
        # DBSCAN clustering on CPU
        dbscan_cluster_func = DBSCAN(eps=config.eps, min_samples=2, metric='precomputed', n_jobs=config.cpu_core)
        cluster_device = 'CPU'
   

    cluster_labels = [cluster_bucket(
        bucket_slice=b_slice_i, 
        data_dict=data_dict,
        prec_tol=config.precursor_tol[0], 
        cluster_func=dbscan_cluster_func,
        output_type='cupy' if config.use_gpu_cluster else 'numpy') 
        for b_slice_i in tqdm(bucket_idx_dict['sort_bucket_idx_arr'])]

    cluster_labels = [cluster_labels[i] for i in bucket_idx_dict['reorder_idx']]

    logger.info("{} clustering in {:.4f} s".format(cluster_device, time.time()-start))

    return cluster_labels


def cluster_spectra_stage_shm(
    spectra_by_charge_df: pd.DataFrame,
    config: config,
    logger: logging,
    bin_len: int,
    lv_hvs: cp.array,
    id_hvs: cp.array
):
    # Encode spectra
    logger.info("Start encoding")
    # encoded_spectra_hv_gt = encode_preprocessed_spectra_shm(
    #         spectra_df=spectra_by_charge_df, 
    #         config=config, dim=bin_len, logger=logger,
    #         lv_hvs_packed=lv_hvs, id_hvs_packed=id_hvs,
    #         output_type='numpy')

    encoded_spectra_hv = encode_preprocessed_spectra(
            spectra_df=spectra_by_charge_df, 
            config=config, dim=bin_len, logger=logger,
            lv_hvs_packed=lv_hvs, id_hvs_packed=id_hvs,
            output_type='numpy')

    # print(encoded_spectra_hv.shape, encoded_spectra_hv_test.shape)
    # print("Encoding error: ", np.mean(np.abs(encoded_spectra_hv-encoded_spectra_hv_gt)))
    # raise

    # Cluster encoded spectra
    logger.info("Start clustering")
    # cluster_labels_gt = cluster_encoded_spectra_shm(
    #     spectra_by_charge_df=spectra_by_charge_df,
    #     encoded_spectra_hv=encoded_spectra_hv_gt,
    #     config=config, logger=logger)
    
    cluster_labels = cluster_encoded_spectra(
        spectra_by_charge_df=spectra_by_charge_df,
        encoded_spectra_hv=encoded_spectra_hv,
        config=config, logger=logger)

    return cluster_labels


def encode_cluster_bucket_shm(
    bucket_slice: tuple, 
    shm_dict: dict,
    D: int,
    Q: int,
    prec_tol: float,
    bin_len: int,
    cluster_func: Callable,
    output_type: str='numpy'
):
    if bucket_slice[1]-bucket_slice[0]==0:
        return np.array([-1])
    else:
        bucket_slice[1] += 1
        bucket_hv = encode_func_shm(
            slice_idx=bucket_slice,
            shm_dict=shm_dict,
            D=D, Q=Q,
            dim=bin_len,
            output_type=output_type)


        bucket_prec_mz = shm_dict['shm_prec_mz'].gather(bucket_slice)

        # start = time.time()
        pw_dist = fast_nb_cosine_dist_mask(
            bucket_hv, bucket_prec_mz, prec_tol, output_type)
        # print("pw dist + mask  time ", time.time()-start)

        cluster_func.fit(pw_dist) #
        L = cluster_func.labels_
        del pw_dist, bucket_hv
        
        return L


# def cluster_spectra_pipeline_shm(
#     spectra_by_charge_df: pd.DataFrame,
#     config: config,
#     logger: logging,
#     bin_len: int,
#     lv_hvs_packed: cp.array,
#     id_hvs_packed: cp.array  
# ):
#     # Get bucket list
#     bucket_idx_dict = schedule_bucket(spectra_by_charge_df, logger)

#     # Define DBSCAN clustering function
#     dbscan_cluster_func = DBSCAN(eps=config.eps, min_samples=2, metric='precomputed', n_jobs=config.cpu_core_dbscan) \
#         if not config.use_gpu_cluster else cuml.DBSCAN(eps=config.eps, min_samples=2, metric='precomputed', output_type='numpy')
    
#     # Create shm
#     start = time.time()
#     lv_hvs_np = cp.asnumpy(lv_hvs_packed).ravel()
#     shm_lv_hvs = SharedMem(name='shm_lv_hvs', data=lv_hvs_np)

#     id_hvs_np = cp.asnumpy(id_hvs_packed).ravel()
#     shm_id_hvs = SharedMem(name='shm_id_hvs', data=id_hvs_np)

#     intensity, indices, indptr = _to_csr_vector(spectra_by_charge_df, config.min_mz, config.fragment_tol)
#     spectra_by_charge_df.drop(columns=['mz', 'intensity'], inplace=True)

#     shm_intensity = SharedMem(name='shm_intensity', data=intensity)
#     shm_indices = SharedMem(name='shm_indices', data=indices)
#     shm_indptr = SharedMem(name='shm_indptr', data=indptr)
    
#     prec_mz = np.vstack(spectra_by_charge_df.precursor_mz).astype(np.float32)

#     shm_prec_mz = SharedMem(name='shm_prec_mz', data=prec_mz)

#     shm_dict = {'lv_hvs': shm_lv_hvs, 'id_hvs': shm_id_hvs,\
#         'intensity': shm_intensity, 'indices': shm_indices, 'indptr': shm_indptr,\
#         'shm_prec_mz': shm_prec_mz}

#     logger.info("Create shm in {:.4f}s".format(time.time()-start))

#     # Encode and cluster for each bucket
#     with Parallel(n_jobs=config.cpu_core_cluster, batch_size='auto') as parallel_pool:
#         cluster_labels = parallel_pool(
#             delayed(encode_cluster_bucket_shm)(
#                 bucket_slice = b_slice_i, 
#                 shm_dict=shm_dict,
#                 D=config.hd_dim, Q=config.hd_Q,
#                 prec_tol=config.precursor_tol[0],
#                 bin_len=bin_len,
#                 cluster_func=dbscan_cluster_func,
#                 output_type='cupy' if config.use_gpu_cluster else 'numpy')
#                 for b_slice_i in tqdm(bucket_idx_dict['sort_bucket_idx_arr']))

#     logger.info("\nEncode and cluster in {:.4f} s".format(time.time()-start))

#     for i in shm_dict.values():
#         i.close()

#     cluster_labels = [cluster_labels[i] for i in bucket_idx_dict['reorder_idx']]
#     return cluster_labels


def encode_cluster_bucket(
    bucket_slice: tuple, 
    data_dict: dict,
    D: int,
    Q: int,
    prec_tol: float,
    bin_len: int,
    cluster_func: Callable,
    output_type: str='numpy'
):
    if bucket_slice[1]-bucket_slice[0]==0:
        return np.array([-1])
    else:
        bucket_slice[1] += 1
        bucket_hv = encode_func(
            slice_idx=bucket_slice,
            data_dict=data_dict,
            D=D, Q=Q,
            dim=bin_len,
            output_type=output_type)

        bucket_prec_mz = data_dict['prec_mz'][bucket_slice[0]: bucket_slice[1]]

        # start = time.time()
        pw_dist = fast_nb_cosine_dist_mask(
            bucket_hv, bucket_prec_mz, prec_tol, output_type)

        cluster_func.fit(pw_dist) #
        L = cluster_func.labels_
        del pw_dist, bucket_hv
        
        return L


def cluster_spectra_pipeline_shm(
    spectra_by_charge_df: pd.DataFrame,
    config: config,
    logger: logging,
    bin_len: int,
    lv_hvs_packed: cp.array,
    id_hvs_packed: cp.array  
):
    # Get bucket list
    bucket_idx_dict = schedule_bucket(spectra_by_charge_df, logger)

    # Define DBSCAN clustering function
    dbscan_cluster_func = DBSCAN(eps=config.eps, min_samples=2, metric='precomputed', n_jobs=config.cpu_core_dbscan) \
        if not config.use_gpu_cluster else cuml.DBSCAN(eps=config.eps, min_samples=2, metric='precomputed', output_type='numpy')
    
    # Create shm
    start = time.time()
    lv_hvs = cp.asnumpy(lv_hvs_packed).ravel()
    id_hvs = cp.asnumpy(id_hvs_packed).ravel()

    intensity, indices, indptr = _to_csr_vector(spectra_by_charge_df, config.min_mz, config.fragment_tol)
    spectra_by_charge_df.drop(columns=['mz', 'intensity'], inplace=True)
    prec_mz = np.vstack(spectra_by_charge_df.precursor_mz).astype(np.float32)

    data_dict = {'lv_hvs': lv_hvs, 'id_hvs': id_hvs,\
        'intensity': intensity, 'indices': indices, 'indptr': indptr,\
        'prec_mz': prec_mz}

    # Encode and cluster for each bucket
    with Parallel(n_jobs=config.cpu_core_cluster, batch_size='auto') as parallel_pool:
        cluster_labels = [
            encode_cluster_bucket(
                bucket_slice = b_slice_i, 
                data_dict=data_dict,
                D=config.hd_dim, Q=config.hd_Q,
                prec_tol=config.precursor_tol[0],
                bin_len=bin_len,
                cluster_func=dbscan_cluster_func,
                output_type='cupy' if config.use_gpu_cluster else 'numpy')
                for b_slice_i in tqdm(bucket_idx_dict['sort_bucket_idx_arr'])]

    cluster_labels = [cluster_labels[i] for i in bucket_idx_dict['reorder_idx']]

    logger.info("\nEncode and cluster in {:.4f} s".format(time.time()-start))

    return cluster_labels


def post_process_cluster(bucket_cluster_labels):
    reorder_labels = []
    label_base = 0
    for cluster_i in bucket_cluster_labels:
        cluster_i = np.array(cluster_i)
        noise_idx = cluster_i == -1

        num_clusters = np.amax(cluster_i) + 1
        num_noises = np.sum(noise_idx)

        cluster_i[noise_idx] = np.arange(num_clusters, num_clusters + num_noises)
        cluster_i += label_base
        
        label_base += (num_clusters+num_noises)
        reorder_labels.append(cluster_i)
    
    return np.hstack(reorder_labels)
