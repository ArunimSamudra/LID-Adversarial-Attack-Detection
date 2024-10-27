import os
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors

def compute_lid(sample, neighbors, k_list):
    lids = []
    for k in k_list:
        nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(neighbors)
        distances, _ = nbrs.kneighbors(sample)
        # Compute LID using MLE
        for d in distances:
            max_dist = np.max(d)
            d[d == 0] = 1e-8  # Avoid division by zero
            lid = -1 / np.mean(np.log(d / max_dist))
            lids.append(lid)
    return np.mean(lids)  # Return average LID

def main(parser):
    script_args = parser.parse_args()
    model_name_base = os.path.basename(script_args.model_name)

    layers = [i for i in range(32)]
    p_value_for_layers = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for layer in layers:
        # Load data
        attack = torch.load(f"../output_tensors/{model_name_base}_all_layer_{layer}_perturbed_text.pt", map_location=device, weights_only=True)
        original = torch.load(f"../output_tensors/{model_name_base}_all_layer_{layer}_original_text.pt", map_location=device, weights_only=True)
        labels = torch.load(f"../output_tensors/{model_name_base}_all_layer_{layer}_original_output.pt", map_location=device, weights_only=True)
        labels = torch.load(f"../output_tensors_local/{model_name_base}_all_layer_{layer}_result_type.pt",
                            map_location=device, weights_only=True)

        # Split indices for train and test
        total_samples = original.shape[0]
        test_idxs = np.random.choice(total_samples, size=int(0.2 * total_samples), replace=False)
        train_idxs = [i for i in range(total_samples) if i not in test_idxs]

        # Train data with only clean samples
        train_pd = original[train_idxs, :].cpu().numpy().astype('float32')
        train_labels = labels[train_idxs]

        # Test data with unclean samples
        test_pd = np.array(attack[test_idxs, :]).astype('float32')
        # test_gt = original[test_idxs, :].cpu().numpy().astype('float32')
        test_labels = labels[test_idxs]

        # Prepare correct examples batch
        correct_batch = []
        for p, l in zip(train_pd, train_labels):
            if l.item() == 1:
                correct_batch.append(p.tolist())
        correct_batch = np.array(correct_batch).astype('float32')

        # Calculate average LID for train and test samples
        k_list = [min(correct_batch.shape[0] - 1, 10)]  # Adjust k as needed
        #k_list = [correct_batch.shape[0] - 1]  # Adjust k as needed

        # Compute LID for test samples
        lids_test = [compute_lid(np.array([x]), correct_batch, k_list) for x in test_pd]
        lids_train = [compute_lid(np.array([x]), correct_batch, k_list) for x in train_pd]

        # Compute ROC AUC score using LIDs and test labels
        roc_auc = roc_auc_score(test_labels.cpu().numpy(), -np.array(lids_test))  # Use -LID as scores
        print(f"Layer {layer} - ROC AUC Score: {roc_auc}")

        # Average LID values for train and test sets for each layer
        avg_lid_train = np.mean(lids_train)
        avg_lid_test = np.mean(lids_test)

        print(f"Layer {layer} - Avg Train LID: {str(avg_lid_train)}, Avg Test LID: {str(avg_lid_test)}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, help="Base name of the model")
    main(parser)
