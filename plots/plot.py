import re

from matplotlib import pyplot as plt


def plot_avg_lid(avg_adv_lid, avg_clean_lid):
    """
    Plot average LIDs for adversarial and clean samples over different layers.
    :param avg_adv_lid: List of average LIDs for adversarial samples.
    :param avg_clean_lid: List of average LIDs for clean samples.
    :param layers: List of layer indices.
    """
    #plt.figure(figsize=(12, 6))
    layers = list(range(len(avg_adv_lid)))

    # Plotting average LIDs for adversarial samples
    plt.clf()
    plt.plot(layers, avg_adv_lid, marker='o', label='Average LID (Adversarial)')

    # Plotting average LIDs for clean samples
    plt.plot(layers, avg_clean_lid, marker='o', label='Average LID (Clean)')

    # Adding labels and title
    plt.xlabel('Layers')
    plt.ylabel('Average LID')
    plt.title('Average LID for Adversarial vs. Clean Samples Across Layers (TextFooler)')
    plt.legend()

    plt.grid()
    plt.savefig('../plots/lid_vs_layers_tf.png')


def plot_auc(auc_values):
    # Layers corresponding to AUC values
    layers = list(range(len(auc_values)))

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(layers, auc_values, marker='o', linestyle='-', label="AUC Values")

    # Add labels and title
    plt.xlabel("Layer Number", fontsize=12)
    plt.ylabel("AUC", fontsize=12)
    plt.title("AUC Values Across Layers (DeepWordBug)", fontsize=14)
    #plt.grid(True)
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_edgecolor('black')  # Set border color
        spine.set_linewidth(2)
    plt.legend()
    plt.xticks(layers)  # Ensure all layers are shown on the x-axis
    plt.tight_layout()

    # Show plot
    plt.show()

if __name__ == '__main__':
    text = """
Layer 0 - Avg Clean LID: 0.54745978408178, Avg Adv LID: 3.255686368013295
Layer 1 - Avg Clean LID: 0.532870479883218, Avg Adv LID: 3.548929203704988
Layer 2 - Avg Clean LID: 0.5268426604787873, Avg Adv LID: 4.0419044063456955
Layer 3 - Avg Clean LID: 0.5135205368691337, Avg Adv LID: 5.047342846519113
Layer 4 - Avg Clean LID: 0.5042489091178121, Avg Adv LID: 5.669627997617349
Layer 5 - Avg Clean LID: 0.49903561308453365, Avg Adv LID: 6.160275137253706
Layer 6 - Avg Clean LID: 0.4956124480224359, Avg Adv LID: 6.572838602753748
Layer 7 - Avg Clean LID: 0.4919542687805705, Avg Adv LID: 6.973075304266118
Layer 8 - Avg Clean LID: 0.4900594053752307, Avg Adv LID: 7.485921580642554
Layer 9 - Avg Clean LID: 0.4877280982612573, Avg Adv LID: 7.715587780793514
Layer 10 - Avg Clean LID: 0.48577560840982126, Avg Adv LID: 7.992408153693762
Layer 11 - Avg Clean LID: 0.4859795960588915, Avg Adv LID: 8.400392678593695
Layer 12 - Avg Clean LID: 0.48373050830112535, Avg Adv LID: 8.354017410462612
Layer 13 - Avg Clean LID: 0.48064930757279817, Avg Adv LID: 7.852590479839008
Layer 14 - Avg Clean LID: 0.478920139429387, Avg Adv LID: 7.745463665774924
Layer 15 - Avg Clean LID: 0.4775707426924106, Avg Adv LID: 7.552191639336198
Layer 16 - Avg Clean LID: 0.47435849329256247, Avg Adv LID: 7.2198676948450204
Layer 17 - Avg Clean LID: 0.471611371446011, Avg Adv LID: 7.159070422647484
Layer 18 - Avg Clean LID: 0.46980424426212897, Avg Adv LID: 7.111416594988575
Layer 19 - Avg Clean LID: 0.46855807904639113, Avg Adv LID: 7.200644720783791
Layer 20 - Avg Clean LID: 0.4666865456018348, Avg Adv LID: 7.288548036610347
Layer 21 - Avg Clean LID: 0.46446741467468733, Avg Adv LID: 7.069089892887743
Layer 22 - Avg Clean LID: 0.4629067202474455, Avg Adv LID: 7.0022778170201425
Layer 23 - Avg Clean LID: 0.4610577750052702, Avg Adv LID: 6.899040974004831
Layer 24 - Avg Clean LID: 0.4595586346976782, Avg Adv LID: 6.907444919133996
Layer 25 - Avg Clean LID: 0.4576874604542672, Avg Adv LID: 6.773146660813152
Layer 26 - Avg Clean LID: 0.4558479283632552, Avg Adv LID: 6.697618026841062
Layer 27 - Avg Clean LID: 0.4540833595749571, Avg Adv LID: 6.660181682933429
Layer 28 - Avg Clean LID: 0.45243167038439513, Avg Adv LID: 6.653542387626826
Layer 29 - Avg Clean LID: 0.4498976742898065, Avg Adv LID: 6.609613132831838
Layer 30 - Avg Clean LID: 0.4468428384944082, Avg Adv LID: 6.5640482642196725
Layer 31 - Avg Clean LID: 0.43873081557158006, Avg Adv LID: 6.639155106145816

    """
    # Extract values using regex
    clean_lid_values = re.findall(r'Avg Clean LID: ([0-9.]+)', text)
    adv_lid_values = re.findall(r'Avg Adv LID: ([0-9.]+)', text)

    # Convert to floats for further computation, if needed
    clean_lid_values = [float(value) for value in clean_lid_values]
    adv_lid_values = [float(value) for value in adv_lid_values]

    # Output
    print("Clean LID values:", clean_lid_values)
    print("Adv LID values:", adv_lid_values)

    auc = """
    Accuracy:  0.9849624060150376
    AUC:  0.984375

    Accuracy:  0.9548872180451128
    AUC:  0.9538461538461538

    Accuracy:  0.9849624060150376
    AUC:  0.9841269841269842

    Accuracy:  0.9699248120300752
    AUC:  0.9666666666666667

    Accuracy:  0.9699248120300752
    AUC:  0.9649122807017544

    Accuracy:  0.9699248120300752
    AUC:  0.9710144927536232

    Accuracy:  0.9699248120300752
    AUC:  0.9661016949152542

    Accuracy:  0.9849624060150376
    AUC:  0.9857142857142857

    Accuracy:  0.9548872180451128
    AUC:  0.9545454545454545

    Accuracy:  0.9924812030075187
    AUC:  0.9923076923076923

    Accuracy:  0.9849624060150376
    AUC:  0.9846153846153847

    Accuracy:  0.9699248120300752
    AUC:  0.9692307692307692

    Accuracy:  0.9624060150375939
    AUC:  0.9615384615384616

    Accuracy:  0.9624060150375939
    AUC:  0.962121212121212
    Confusion Matrix:

    Accuracy:  0.9774436090225563
    AUC:  0.9788732394366197

    Accuracy:  0.9548872180451128
    AUC:  0.9571428571428571

    Accuracy:  0.9624060150375939
    AUC:  0.9632352941176471

    Accuracy:  0.9624060150375939
    AUC:  0.963768115942029

    Accuracy:  0.9774436090225563
    AUC:  0.9765625

    Accuracy:  0.9774436090225563
    AUC:  0.9807692307692308

    Accuracy:  0.9774436090225563
    AUC:  0.9791666666666667

    Accuracy:  0.9699248120300752
    AUC:  0.9672131147540984

    Accuracy:  0.9849624060150376
    AUC:  0.9857142857142857
    """

    # Extract AUC values using regex
    auc_values = re.findall(r'AUC:  ([0-9.]+)', auc)

    # Convert extracted AUC values to floats for further use
    auc_values = [float(value) for value in auc_values]

    # Output
    print("Extracted AUC values:", auc_values)

    plot_avg_lid(adv_lid_values, clean_lid_values)
    #plot_auc(auc_values)