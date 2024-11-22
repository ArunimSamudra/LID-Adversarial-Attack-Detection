import os

import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from sklearn.neighbors import NearestNeighbors, KernelDensity
from sklearn.linear_model import LogisticRegression

import numpy as np


def silverman_bandwidth(data):
    """
    Calculate Silverman's bandwidth for KDE.
    Args:
        data (numpy array): Data array of shape (n_samples, n_features).
    Returns:
        float: Bandwidth value.
    """
    n, d = data.shape
    std_dev = np.std(data, axis=0).mean()  # Average standard deviation across dimensions
    return (4 * (std_dev ** 5) / (3 * n)) ** (1 / 5)


def scott_bandwidth(data):
    """
    Calculate Scott's bandwidth for KDE.
    Args:
        data (numpy array): Data array of shape (n_samples, n_features).
    Returns:
        float: Bandwidth value.
    """
    n, d = data.shape
    std_dev = np.std(data, axis=0).mean()  # Average standard deviation across dimensions
    return n ** (-1 / (d + 4)) * std_dev


def softmax(logits):
    """
    Compute softmax probabilities from logits.
    Args:
        logits (ndarray): Activations from the model, shape (num_samples, num_classes).
    Returns:
        ndarray: Softmax probabilities, shape (num_samples, num_classes).
    """
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))  # Stabilize with max subtraction
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)


def calculate_log_probs(probabilities):
    """
    Compute log-probabilities for each sample.
    Args:
        probabilities (ndarray): Probabilities for each sample, shape (num_samples, num_classes).
    Returns:
        list of float: Log-probabilities for each sample's predicted class.
    """
    return np.log(probabilities[np.arange(len(probabilities)), np.argmax(probabilities, axis=1)])


def predictive_entropy(log_probs):
    return -np.mean(log_probs)


def length_normalized_entropy(log_probs, lengths):
    ln_log_probs = [log_prob / length for log_prob, length in zip(log_probs, lengths)]
    return -np.mean(ln_log_probs)


def semantic_entropy(probabilities, groups):
    group_probs = []
    for group in groups:
        group_prob = np.sum([np.exp(probabilities[i]) for i in group])
        group_probs.append(np.log(group_prob))
    return -np.mean(group_probs)


def compute_lid(sample, neighbors, k):
    """
    Compute the Local Intrinsic Dimensionality (LID) for a sample.
    Args:
        sample: The data sample(s) for which LID is computed.
        neighbors: The dataset to compute neighbors from.
        k: Number of nearest neighbors.
    Returns:
        LID value for the sample.
    """
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(neighbors)
    distances, _ = nbrs.kneighbors(sample)
    lids = []
    for d in distances:
        max_dist = np.max(d)
        d[d == 0] = 1e-8  # Avoid division by zero
        lid = -1 / np.mean(np.log(d / max_dist))
        lids.append(lid)
    return np.array(lids)


def train_detector(X_train, y_train):
    """
    Train a logistic regression detector based on LID features.
    Args:
        X_train: Training features (LID values).
        y_train: Training labels (adversarial=1, normal/noisy=0).
    Returns:
        Trained logistic regression model.
    """
    detector = LogisticRegression(max_iter=1000)
    detector.fit(X_train, y_train)
    return detector


def evaluate_detector(detector, X_test, y_test):
    """
    Evaluate the LID-based detector on the test set.
    Args:
        detector: Trained logistic regression classifier.
        X_test: Test features (normal examples).
        y_test: Test labels (adversarial=1, normal=0).
    Returns:
        Accuracy, AUC, and confusion matrix for the test set evaluation.
    """
    # Predict with the trained detector
    y_pred = detector.predict(X_test)

    # Evaluate performance
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    return accuracy, auc, cm


def main(parser):
    # script_args = parser.parse_args()
    # model_name_base = os.path.basename(script_args.model_name)
    #
    # #for k in range(2, 25):
    # layers = [i for i in range(32)]
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # lid_neg, lid_pos = [], []
    # entropy_clean, entropy_adv = [], []
    # kde_scores_clean, kde_scores_adv = [], []
    # bu_uncertainty_clean, bu_uncertainty_adv = [], []
    #
    # for layer in layers:
    #     # Load data
    #     attack = torch.load(f"../model/output_tensors_tf/{model_name_base}_all_layer_{layer}_perturbed_text.pt",
    #                         map_location=device, weights_only=True)
    #     original = torch.load(f"../model/output_tensors_tf/{model_name_base}_all_layer_{layer}_original_text.pt",
    #                           map_location=device, weights_only=True)
    #
    #     clean = original.cpu().numpy().astype('float32')
    #     adv = attack.cpu().numpy().astype('float32')
    #
    #     k = 10
    #     lids_clean = [compute_lid(np.array([x]), clean, k) for x in clean]
    #     lids_adv = [compute_lid(np.array([x]), clean, k) for x in adv]
    #
    #     print(f"Layer {layer} - Avg Clean LID: {str(np.mean(lids_clean))}, Avg Adv LID: {str(np.mean(lids_adv))}")
    #
    #     lid_pos.append(lids_clean)
    #     lid_neg.append(lids_adv)
    #
    #     # Compute softmax probabilities for clean and adversarial examples
    #     probs_clean = softmax(clean)
    #     probs_adv = softmax(adv)
    #
    #     # Calculate entropy for each sample
    #     entropy_clean_layer = -np.sum(probs_clean * np.log(probs_clean + 1e-8), axis=1)  # Shape: (num_samples,)
    #     entropy_adv_layer = -np.sum(probs_adv * np.log(probs_adv + 1e-8), axis=1)  # Shape: (num_samples,)
    #
    #     # Store per-sample entropy
    #     entropy_clean.append(entropy_clean_layer)
    #     entropy_adv.append(entropy_adv_layer)
    #
    #     # Fit KDE on clean data activations
    #     kde = KernelDensity(kernel='gaussian', bandwidth=0.3)
    #     kde.fit(clean)
    #
    #     # Compute KDE scores for clean and adversarial examples
    #     log_density_clean = kde.score_samples(clean)  # Log-density for clean examples
    #     log_density_adv = kde.score_samples(adv)  # Log-density for adversarial examples
    #
    #     kde_scores_clean.append(log_density_clean)
    #     kde_scores_adv.append(log_density_adv)
    #
    # total_samples = 666
    # test_idxs = np.random.choice(total_samples, size=int(0.2 * total_samples), replace=False)
    # train_idxs = [i for i in range(total_samples) if i not in test_idxs]
    #
    # X = np.vstack([np.hstack(lid_neg), np.hstack(lid_pos)])
    # y = np.array([0] * 333 + [1] * 333)
    # print("LID")
    # evaluate(X, y, test_idxs, train_idxs)
    #
    # print("Predictive Entropy")
    # X = np.hstack([np.vstack(entropy_adv), np.vstack(entropy_clean)]).T
    # evaluate(X, y, test_idxs, train_idxs)
    #
    # print("KDE")
    # X = np.hstack([np.vstack(kde_scores_adv), np.vstack(kde_scores_clean)]).T
    # evaluate(X, y, test_idxs, train_idxs)
    script_args = parser.parse_args()
    model_name_base = os.path.basename(script_args.model_name)

    # for k in range(2, 25):
    layers = [i for i in range(32)]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lid_neg_tf, lid_pos_tf, lid_neg_bae, lid_pos_bae, lid_neg_dw, lid_pos_dw = [], [], [], [], [], []
    entropy_clean_tf, entropy_adv_tf, entropy_clean_bae, entropy_adv_bae, entropy_clean_dw, entropy_adv_dw = [], [], [], [], [], []
    kde_scores_clean_tf, kde_scores_adv_tf, kde_scores_clean_bae, kde_scores_adv_bae, kde_scores_clean_dw, kde_scores_adv_dw = [], [], [], [], [], []

    for layer in layers:
        # Load data
        attack_tf = torch.load(f"../model/output_tensors_tf/{model_name_base}_all_layer_{layer}_perturbed_text.pt",
                               map_location=device, weights_only=True)
        original_tf = torch.load(f"../model/output_tensors_tf/{model_name_base}_all_layer_{layer}_original_text.pt",
                                 map_location=device, weights_only=True)
        attack_bae = torch.load(f"../model/output_tensors_bae/{model_name_base}_all_layer_{layer}_perturbed_text.pt",
                                map_location=device, weights_only=True)
        original_bae = torch.load(f"../model/output_tensors_bae/{model_name_base}_all_layer_{layer}_original_text.pt",
                                  map_location=device, weights_only=True)
        attack_dw = torch.load(f"../output_tensors/{model_name_base}_all_layer_{layer}_perturbed_text.pt",
                               map_location=device, weights_only=True)
        original_dw = torch.load(f"../output_tensors/{model_name_base}_all_layer_{layer}_original_text.pt",
                                 map_location=device, weights_only=True)

        #### TF
        clean = original_tf.cpu().numpy().astype('float32')
        adv = attack_tf.cpu().numpy().astype('float32')

        k = 10
        lids_clean = [compute_lid(np.array([x]), clean, k) for x in clean]
        lids_adv = [compute_lid(np.array([x]), clean, k) for x in adv]

        print(f"Layer {layer} - Avg Clean LID: {str(np.mean(lids_clean))}, Avg Adv LID: {str(np.mean(lids_adv))}")

        lid_pos_tf.append(lids_clean)
        lid_neg_tf.append(lids_adv)

        # Compute softmax probabilities for clean and adversarial examples
        probs_clean = softmax(clean)
        probs_adv = softmax(adv)

        # Calculate entropy for each sample
        entropy_clean_layer = -np.sum(probs_clean * np.log(probs_clean + 1e-8), axis=1)  # Shape: (num_samples,)
        entropy_adv_layer = -np.sum(probs_adv * np.log(probs_adv + 1e-8), axis=1)  # Shape: (num_samples,)

        # Store per-sample entropy
        entropy_clean_tf.append(entropy_clean_layer)
        entropy_adv_tf.append(entropy_adv_layer)

        # Fit KDE on clean data activations
        kde = KernelDensity(kernel='gaussian', bandwidth=0.3)
        kde.fit(clean)

        # Compute KDE scores for clean and adversarial examples
        log_density_clean = kde.score_samples(clean)  # Log-density for clean examples
        log_density_adv = kde.score_samples(adv)  # Log-density for adversarial examples

        kde_scores_clean_tf.append(log_density_clean)
        kde_scores_adv_tf.append(log_density_adv)

        #### BAE
        clean = original_bae.cpu().numpy().astype('float32')
        adv = attack_bae.cpu().numpy().astype('float32')

        k = 10
        lids_clean = [compute_lid(np.array([x]), clean, k) for x in clean]
        lids_adv = [compute_lid(np.array([x]), clean, k) for x in adv]

        print(f"Layer {layer} - Avg Clean LID: {str(np.mean(lids_clean))}, Avg Adv LID: {str(np.mean(lids_adv))}")

        lid_pos_bae.append(lids_clean)
        lid_neg_bae.append(lids_adv)

        # Compute softmax probabilities for clean and adversarial examples
        probs_clean = softmax(clean)
        probs_adv = softmax(adv)

        # Calculate entropy for each sample
        entropy_clean_layer = -np.sum(probs_clean * np.log(probs_clean + 1e-8), axis=1)  # Shape: (num_samples,)
        entropy_adv_layer = -np.sum(probs_adv * np.log(probs_adv + 1e-8), axis=1)  # Shape: (num_samples,)

        # Store per-sample entropy
        entropy_clean_bae.append(entropy_clean_layer)
        entropy_adv_bae.append(entropy_adv_layer)

        # Fit KDE on clean data activations
        kde = KernelDensity(kernel='gaussian', bandwidth=0.3)
        kde.fit(clean)

        # Compute KDE scores for clean and adversarial examples
        log_density_clean = kde.score_samples(clean)  # Log-density for clean examples
        log_density_adv = kde.score_samples(adv)  # Log-density for adversarial examples

        kde_scores_clean_bae.append(log_density_clean)
        kde_scores_adv_bae.append(log_density_adv)

        #### DW
        clean = original_dw.cpu().numpy().astype('float32')
        adv = attack_dw.cpu().numpy().astype('float32')

        k = 10
        lids_clean = [compute_lid(np.array([x]), clean, k) for x in clean]
        lids_adv = [compute_lid(np.array([x]), clean, k) for x in adv]

        print(f"Layer {layer} - Avg Clean LID: {str(np.mean(lids_clean))}, Avg Adv LID: {str(np.mean(lids_adv))}")

        lid_pos_dw.append(lids_clean)
        lid_neg_dw.append(lids_adv)

        # Compute softmax probabilities for clean and adversarial examples
        probs_clean = softmax(clean)
        probs_adv = softmax(adv)

        # Calculate entropy for each sample
        entropy_clean_layer = -np.sum(probs_clean * np.log(probs_clean + 1e-8), axis=1)  # Shape: (num_samples,)
        entropy_adv_layer = -np.sum(probs_adv * np.log(probs_adv + 1e-8), axis=1)  # Shape: (num_samples,)

        # Store per-sample entropy
        entropy_clean_dw.append(entropy_clean_layer)
        entropy_adv_dw.append(entropy_adv_layer)

        # Fit KDE on clean data activations
        kde = KernelDensity(kernel='gaussian', bandwidth=0.3)
        kde.fit(clean)

        # Compute KDE scores for clean and adversarial examples
        log_density_clean = kde.score_samples(clean)  # Log-density for clean examples
        log_density_adv = kde.score_samples(adv)  # Log-density for adversarial examples

        kde_scores_clean_dw.append(log_density_clean)
        kde_scores_adv_dw.append(log_density_adv)

    total_samples = 666
    test_idxs = np.random.choice(total_samples, size=int(0.2 * total_samples), replace=False)
    train_idxs = [i for i in range(total_samples) if i not in test_idxs]

    X = np.vstack([np.hstack(lid_neg_dw), np.hstack(lid_pos_dw)])
    y = np.array([0] * 333 + [1] * 333)
    print("--------LID----------")
    lid_detector = train(X, y, train_idxs)
    print("DW vs DW")
    evaluate_out_of_sample(X, y, test_idxs, lid_detector)
    print("DW vs BAE")
    X = np.vstack([np.hstack(lid_neg_bae), np.hstack(lid_pos_bae)])
    evaluate_out_of_sample(X, y, test_idxs, lid_detector)
    print("DW vs TF")
    X = np.vstack([np.hstack(lid_neg_tf), np.hstack(lid_pos_tf)])
    evaluate_out_of_sample(X, y, test_idxs, lid_detector)

    X = np.vstack([np.hstack(lid_neg_bae), np.hstack(lid_pos_bae)])
    lid_detector = train(X, y, train_idxs)
    print("BAE vs BAE")
    evaluate_out_of_sample(X, y, test_idxs, lid_detector)
    print("BAE vs DW")
    X = np.vstack([np.hstack(lid_neg_dw), np.hstack(lid_pos_dw)])
    evaluate_out_of_sample(X, y, test_idxs, lid_detector)
    print("BAE vs TF")
    X = np.vstack([np.hstack(lid_neg_tf), np.hstack(lid_pos_tf)])
    evaluate_out_of_sample(X, y, test_idxs, lid_detector)

    X = np.vstack([np.hstack(lid_neg_tf), np.hstack(lid_pos_tf)])
    lid_detector = train(X, y, train_idxs)
    print("TF vs TF")
    evaluate_out_of_sample(X, y, test_idxs, lid_detector)
    print("TF vs BAE")
    X = np.vstack([np.hstack(lid_neg_bae), np.hstack(lid_pos_bae)])
    evaluate_out_of_sample(X, y, test_idxs, lid_detector)
    print("TF vs DW")
    X = np.vstack([np.hstack(lid_neg_dw), np.hstack(lid_pos_dw)])
    evaluate_out_of_sample(X, y, test_idxs, lid_detector)

    print("----------Predictive Entropy-------------")
    X = np.hstack([np.vstack(entropy_adv_bae), np.vstack(entropy_clean_bae)]).T
    entropy_detector = train(X, y, train_idxs)
    print("BAE vs BAE")
    evaluate_out_of_sample(X, y, test_idxs, entropy_detector)
    print("BAE vs DW")
    X = np.hstack([np.vstack(entropy_adv_dw), np.vstack(entropy_clean_dw)]).T
    evaluate_out_of_sample(X, y, test_idxs, entropy_detector)
    print("BAE vs TF")
    X = np.hstack([np.vstack(entropy_adv_tf), np.vstack(entropy_clean_tf)]).T
    evaluate_out_of_sample(X, y, test_idxs, entropy_detector)

    X = np.hstack([np.vstack(entropy_adv_tf), np.vstack(entropy_clean_tf)]).T
    entropy_detector = train(X, y, train_idxs)
    print("TF vs TF")
    evaluate_out_of_sample(X, y, test_idxs, entropy_detector)
    print("TF vs BAE")
    X = np.hstack([np.vstack(entropy_adv_bae), np.vstack(entropy_clean_bae)]).T
    evaluate_out_of_sample(X, y, test_idxs, entropy_detector)
    print("TF vs DW")
    X = np.hstack([np.vstack(entropy_adv_dw), np.vstack(entropy_clean_dw)]).T
    evaluate_out_of_sample(X, y, test_idxs, entropy_detector)

    X = np.hstack([np.vstack(entropy_adv_dw), np.vstack(entropy_clean_dw)]).T
    entropy_detector = train(X, y, train_idxs)
    print("DW vs DW")
    evaluate_out_of_sample(X, y, test_idxs, entropy_detector)
    print("DW vs BAE")
    X = np.hstack([np.vstack(entropy_adv_bae), np.vstack(entropy_clean_bae)]).T
    evaluate_out_of_sample(X, y, test_idxs, entropy_detector)
    print("DW vs TF")
    X = np.hstack([np.vstack(entropy_adv_tf), np.vstack(entropy_clean_tf)]).T
    evaluate_out_of_sample(X, y, test_idxs, entropy_detector)

    print("------------KDE----------------")
    X = np.hstack([np.vstack(kde_scores_adv_dw), np.vstack(kde_scores_clean_dw)]).T
    kde_detector = train(X, y, train_idxs)
    print("DW vs DW")
    evaluate_out_of_sample(X, y, test_idxs, kde_detector)
    print("DW vs BAE")
    X = np.hstack([np.vstack(kde_scores_adv_bae), np.vstack(kde_scores_clean_bae)]).T
    evaluate_out_of_sample(X, y, test_idxs, kde_detector)
    print("DW vs TF")
    X = np.hstack([np.vstack(kde_scores_adv_tf), np.vstack(kde_scores_clean_tf)]).T
    evaluate_out_of_sample(X, y, test_idxs, kde_detector)

    X = np.hstack([np.vstack(kde_scores_adv_tf), np.vstack(kde_scores_clean_tf)]).T
    kde_detector = train(X, y, train_idxs)
    print("TF vs TF")
    evaluate_out_of_sample(X, y, test_idxs, kde_detector)
    print("TF vs BAE")
    X = np.hstack([np.vstack(kde_scores_adv_bae), np.vstack(kde_scores_clean_bae)]).T
    evaluate_out_of_sample(X, y, test_idxs, kde_detector)
    print("TF vs DW")
    X = np.hstack([np.vstack(kde_scores_adv_dw), np.vstack(kde_scores_clean_dw)]).T
    evaluate_out_of_sample(X, y, test_idxs, kde_detector)

    X = np.hstack([np.vstack(kde_scores_adv_bae), np.vstack(kde_scores_clean_bae)]).T
    kde_detector = train(X, y, train_idxs)
    print("BAE vs BAE")
    evaluate_out_of_sample(X, y, test_idxs, kde_detector)
    print("BAE vs DW")
    X = np.hstack([np.vstack(kde_scores_adv_dw), np.vstack(kde_scores_clean_dw)]).T
    evaluate_out_of_sample(X, y, test_idxs, kde_detector)
    print("BAE vs TF")
    X = np.hstack([np.vstack(kde_scores_adv_tf), np.vstack(kde_scores_clean_tf)]).T
    evaluate_out_of_sample(X, y, test_idxs, kde_detector)



def train(X, y, train_idxs):
    X_train = X[train_idxs, :]
    y_train = y[train_idxs]
    # Train detector
    detector = train_detector(X_train, y_train)
    return detector


def evaluate_out_of_sample(X, y, test_idxs, detector):
    X_test = X[test_idxs, :]
    y_test = y[test_idxs]
    accuracy, auc, cm = evaluate_detector(detector, X_test, y_test)
    # Print the evaluation results
    print("Accuracy: ", accuracy)
    print("AUC: ", auc)


def evaluate(X, y, test_idxs, train_idxs):
    X_train = X[train_idxs, :]
    y_train = y[train_idxs]
    X_test = X[test_idxs, :]
    y_test = y[test_idxs]
    # Train detector
    detector = train_detector(X_train, y_train)
    accuracy, auc, cm = evaluate_detector(detector, X_test, y_test)
    # Print the evaluation results
    print("Accuracy: ", accuracy)
    print("AUC: ", auc)
    # print("Confusion Matrix:\n", cm)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, help="Base name of the model")
    main(parser)
