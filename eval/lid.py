import argparse
import os
import torch
import faiss
import numpy as np
from sklearn import metrics
from sklearn.neighbors import NearestNeighbors


def roc(labels, scores):
    auroc = metrics.roc_auc_score(labels, scores)
    return auroc


# def compute_lid(y, sampled_feats, k_list=None):
#     """
#     Code borrowed and modified from this repo:
#     https://github.com/TideDancer/iclr21_isotropy_contxt
#     """
#
#     if k_list is None:
#         k_list = [200]
#     cpu_index = faiss.IndexFlatL2(sampled_feats.shape[1])
#     cpu_index.add(np.ascontiguousarray(sampled_feats))
#
#     avg_lids = []
#
#     for k in k_list:
#         i = 0
#         D = []
#         b, nid = cpu_index.search(y, k)
#         b = np.sqrt(b)
#         D.append(b)
#
#         D = np.vstack(D)
#         rk = np.max(D, axis=1)
#         rk[rk == 0] = 1e-8
#         lids = D / rk[:, None]
#         lids = -1 / np.mean(np.log(lids), axis=1)
#         lids[np.isinf(lids)] = y.shape[1]  # if inf, set as space dimension
#         lids = lids[~np.isnan(lids)]  # filter nan
#         avg_lids.append(lids.tolist())
#     avg_lids = np.array(avg_lids).mean(axis=0)
#     return avg_lids

def compute_lid(activations, k=10):
    num_samples = len(activations)

    if num_samples < k:
        print(f"Not enough samples for k={k}. Reducing k to {num_samples - 1}.")
        k = num_samples - 1

    # Use Nearest Neighbors to compute LID
    nbrs = NearestNeighbors(n_neighbors=k).fit(activations.numpy())
    distances, indices = nbrs.kneighbors(activations.numpy())
    lids = []

    for i in range(len(activations)):
        d = distances[i][1:]  # Exclude the distance to itself
        rk = np.max(d) if np.max(d) > 0 else 1e-8  # Prevent division by zero
        lid = -1 / np.mean(np.log(d / rk))  # LID formula
        lids.append(lid)

    return np.array(lids)

def main(parser):

    script_args = parser.parse_args()
    model_name_base = os.path.basename(script_args.model_name)

    layers = [i for i in range(1, 2)]
    p_value_for_layers = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for i in layers:

        attack = torch.load(f"../output_tensors/{model_name_base}_all_layer_{i}_perturbed_text.pt", map_location=device)
        original = torch.load(f"../output_tensors/{model_name_base}_all_layer_{i}_original_text.pt", map_location=device)
        labels = torch.load(f"../output_tensors/{model_name_base}_all_layer_{i}_original_output.pt", map_location=device)

        # choose the first 66 examples as test, set for testing
        test_idxs = [i for i in range(original.shape[0]) if i < 5]
        train_idxs = [i for i in range(original.shape[0]) if i not in test_idxs]

        train_pd = original[train_idxs, :].cpu().numpy().astype('float32')
        train_labels = labels[train_idxs]

        test_pd = np.array(attack[test_idxs, :]).astype('float32')
        test_gt = original[test_idxs, :]
        test_labels = labels[test_idxs]

        correct_batch = []
        for p, l in zip(train_pd, train_labels):
            if l.item() == 1:
                correct_batch.append(p.tolist())
        correct_batch = np.array(correct_batch).astype('float32')

        numbers = correct_batch.shape[0]
        k_list = [numbers - 1]
        # for k in k_list:
        #     lids = compute_lid(test_pd, correct_batch, [k])
        #     auroc = roc(test_labels, -lids)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("--model_name", type=str, help="The name of the model")

    main(parser)

