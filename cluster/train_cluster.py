import argparse
import logging
import os
import time
from pathlib import Path

import numpy as np
import torch
import tqdm
from kmeans import KMeansGPU
from sklearn.cluster import KMeans, MiniBatchKMeans

from .. import configs

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_cluster(in_dir, n_clusters, use_minibatch=True, verbose=False,use_gpu=False):#gpu_minibatch真拉，虽然库支持但是也不考虑
    if str(in_dir).endswith(".ipynb_checkpoints"):
        logger.info(f"Ignore {in_dir}")

    logger.info(f"Loading features from {in_dir}")
    features = []
    nums = 0
    tmp = []
    for i in os.listdir(in_dir):
        if i[-8:] == ".soft.pt":
            tmp.append(os.path.join(args.source_dir, args.speaker, "44k", i))
    for path in tqdm.tqdm(tmp):
    # for name in os.listdir(in_dir):
    #     path="%s/%s"%(in_dir,name)
        features.append(torch.load(path,map_location="cpu").squeeze(0).numpy().T)
        # print(features[-1].shape)
    features = np.concatenate(features, axis=0)
    print(nums, features.nbytes/ 1024**2, "MB , shape:",features.shape, features.dtype)
    features = features.astype(np.float32)
    logger.info(f"Clustering features of shape: {features.shape}")
    t = time.time()
    if(use_gpu is False):
        if use_minibatch:
            kmeans = MiniBatchKMeans(n_clusters=n_clusters,verbose=verbose, batch_size=4096, max_iter=80).fit(features)
        else:
            kmeans = KMeans(n_clusters=n_clusters,verbose=verbose).fit(features)
    else:
            kmeans = KMeansGPU(n_clusters=n_clusters, mode='euclidean', verbose=2 if verbose else 0,max_iter=500,tol=1e-2)#
            features=torch.from_numpy(features)#.to(device)
            kmeans.fit_predict(features)#

    print(time.time()-t, "s")

    x = {
            "n_features_in_": kmeans.n_features_in_ if use_gpu is False else features.shape[1],
            "_n_threads": kmeans._n_threads if use_gpu is False else 4,
            "cluster_centers_": kmeans.cluster_centers_ if use_gpu is False else kmeans.centroids.cpu().numpy(),
    }
    print("end")

    return x

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu',action='store_true', default=False ,
                        help='to use GPU')
    parser.add_argument("--speaker", type=str, default="", help="speaker name")
    parser.add_argument("--source_dir", type=str, default=configs.data_dir, help="path to source dir")
    parser.add_argument("--config_dir", type=str, default=configs.data_dir, help="path to config dir")
    parser.add_argument("--model_dir", type=str, default=configs.model_dir, help="path to val list")

    args = parser.parse_args()

    if args.speaker == "":
        raise Exception("type speaker")

    checkpoint_dir = os.path.join(args.model_dir, args.speaker)
    dataset = os.path.join(args.source_dir, args.speaker, "44k")
    use_gpu = args.gpu
    n_clusters = 10000
    
    ckpt = {}
    if os.path.isdir(dataset):
        print(f"train kmeans for {args.speaker}...")
        in_dir = dataset
        x = train_cluster(in_dir, n_clusters,use_minibatch=False,verbose=False,use_gpu=use_gpu)
        ckpt[args.speaker] = x

    checkpoint_path = os.path.join(checkpoint_dir, f"kmeans_{n_clusters}.pt")
    # checkpoint_path.parent.mkdir(exist_ok=True, parents=True)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    torch.save(
        ckpt,
        checkpoint_path,
    )
    
