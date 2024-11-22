import os
import random

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors

def plot_auc_roc(auc_roc, layers):
    """
    Plot average LIDs for adversarial and clean samples over different layers.
    :param avg_test_adv_lid: List of average LIDs for adversarial samples.
    :param avg_test_clean_lid: List of average LIDs for clean samples.
    :param layers: List of layer indices.
    """
    plt.clf()
    plt.plot(layers, auc_roc)

    # Adding labels and title
    plt.xlabel('Layers')
    plt.ylabel('AUROC')
    plt.title('Detection Performance Layers')
    plt.grid()
    plt.legend()
    plt.savefig('../plots/auc_roc.png')

def plot_avg_lid(avg_adv_lid, avg_clean_lid, layers):
    """
    Plot average LIDs for adversarial and clean samples over different layers.
    :param avg_adv_lid: List of average LIDs for adversarial samples.
    :param avg_clean_lid: List of average LIDs for clean samples.
    :param layers: List of layer indices.
    """
    #plt.figure(figsize=(12, 6))

    # Plotting average LIDs for adversarial samples
    plt.clf()
    plt.plot(layers, avg_adv_lid, marker='o', label='Average LID (Adversarial)')

    # Plotting average LIDs for clean samples
    plt.plot(layers, avg_clean_lid, marker='o', label='Average LID (Clean)')

    # Adding labels and title
    plt.xlabel('Layers')
    plt.ylabel('Average LID')
    plt.title('Average LID for Adversarial vs. Clean Samples Across Layers')
    plt.legend()

    plt.grid()
    plt.savefig('../plots/lid_vs_layers.png')

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

def train_lid_detector(X_train, y_train):
    """
    Train a logistic regression detector based on LID features.
    Args:
        X_train: Training features (LID values).
        y_train: Training labels (adversarial=1, normal/noisy=0).
    Returns:
        Trained logistic regression model.
    """
    detector = LogisticRegression()
    detector.fit(X_train, y_train)
    return detector

def main(parser):
    script_args = parser.parse_args()
    model_name_base = os.path.basename(script_args.model_name)

    layers = [i for i in range(32)]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    auc_roc_score = []
    avg_train_lid, avg_test_adv_lid, avg_test_clean_lid = [], [], []

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
        train_clean = original[train_idxs, :].cpu().numpy().astype('float32')
        train_labels = labels[train_idxs]

        # Test data with unclean samples
        test_adv = attack[test_idxs, :].cpu().numpy().astype('float32')
        test_clean = original[test_idxs, :].cpu().numpy().astype('float32')
        test_labels = labels[test_idxs]

        # Prepare correct examples batch
        correct_batch = []
        for p, l in zip(train_clean, train_labels):
            if l.item() == 1:
                correct_batch.append(p.tolist())
        correct_batch = np.array(correct_batch).astype('float32')

        # Calculate average LID for train and test samples
        k_list = [10]  # Adjust k as needed
        #k_list = [correct_batch.shape[0] - 1]  # Adjust k as needed

        # Compute LID for test samples
        lids_adv_test = [compute_lid(np.array([x]), correct_batch, k_list) for x in test_adv]
        lids_adv_clean = [compute_lid(np.array([x]), correct_batch, k_list) for x in test_clean]
        lids_train = [compute_lid(np.array([x]), correct_batch, k_list) for x in train_clean]

        # Average LID values for train and test sets for each layer
        avg_lid_train = np.mean(lids_train)
        avg_lid_adv_test = np.mean(lids_adv_test)
        avg_lid_clean_test = np.mean(lids_adv_clean)
        avg_train_lid.append(avg_lid_train)
        avg_test_adv_lid.append(avg_lid_adv_test)
        avg_test_clean_lid.append(avg_lid_clean_test)

        print(f"Layer {layer} - Avg Train LID: {str(avg_lid_train)}, Avg Test Adv LID: {str(avg_lid_adv_test)}, Avg Test Clean LID: {str(avg_lid_clean_test)}")

        # Compute ROC AUC score using LIDs and test labels
        roc_auc = roc_auc_score(test_labels.cpu().numpy(), -np.array(lids_adv_test))  # Use -LID as scores
        print(f"Layer {layer} - ROC AUC Score: {roc_auc}")
        auc_roc_score.append(roc_auc)

    plot_avg_lid(avg_test_adv_lid, avg_train_lid, layers)
    plot_auc_roc(auc_roc_score, layers)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, help="Base name of the model")
    main(parser)
