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
Layer 0 - Avg Clean LID: 0.4453848198000855, Avg Adv LID: 2.892812385845454
Layer 1 - Avg Clean LID: 0.431184889906056, Avg Adv LID: 3.1881705569078576
Layer 2 - Avg Clean LID: 0.42512722973281436, Avg Adv LID: 3.501975048162607
Layer 3 - Avg Clean LID: 0.413314083588573, Avg Adv LID: 4.363028785758059
Layer 4 - Avg Clean LID: 0.4057923976074109, Avg Adv LID: 4.899370219251968
Layer 5 - Avg Clean LID: 0.4016384756660419, Avg Adv LID: 5.306437714935328
Layer 6 - Avg Clean LID: 0.398725019032407, Avg Adv LID: 5.686477310871297
Layer 7 - Avg Clean LID: 0.39583365959332256, Avg Adv LID: 6.113223099887914
Layer 8 - Avg Clean LID: 0.3941287904400846, Avg Adv LID: 6.576119957643385
Layer 9 - Avg Clean LID: 0.39210269063394665, Avg Adv LID: 6.792994790508006
Layer 10 - Avg Clean LID: 0.3904660931408915, Avg Adv LID: 7.023626886308759
Layer 11 - Avg Clean LID: 0.39069563110532035, Avg Adv LID: 7.358451829385083
Layer 12 - Avg Clean LID: 0.38876052004193423, Avg Adv LID: 7.26261198288376
Layer 13 - Avg Clean LID: 0.38663962342576746, Avg Adv LID: 7.061046745464907
Layer 14 - Avg Clean LID: 0.38536911032757637, Avg Adv LID: 6.758968665570137
Layer 15 - Avg Clean LID: 0.3840606037909023, Avg Adv LID: 6.602544478648321
Layer 16 - Avg Clean LID: 0.38140214716065024, Avg Adv LID: 6.253892148274041
Layer 17 - Avg Clean LID: 0.3792635839332507, Avg Adv LID: 6.331364304688897
Layer 18 - Avg Clean LID: 0.37784364345382343, Avg Adv LID: 6.275828130514031
Layer 19 - Avg Clean LID: 0.3767498371346542, Avg Adv LID: 6.588368487456846
Layer 20 - Avg Clean LID: 0.3754376449551506, Avg Adv LID: 6.590558672310904
Layer 21 - Avg Clean LID: 0.37375689270618895, Avg Adv LID: 6.263431214420765
Layer 22 - Avg Clean LID: 0.37243774493313403, Avg Adv LID: 6.234459905126326
Layer 23 - Avg Clean LID: 0.3709656189012834, Avg Adv LID: 6.028164791541359
Layer 24 - Avg Clean LID: 0.3696837039773177, Avg Adv LID: 6.099187798246817
Layer 25 - Avg Clean LID: 0.36822924402454277, Avg Adv LID: 6.010531425167604
Layer 26 - Avg Clean LID: 0.366626414507838, Avg Adv LID: 5.896135587268322
Layer 27 - Avg Clean LID: 0.3651339769159383, Avg Adv LID: 5.815077841175585
Layer 28 - Avg Clean LID: 0.36369856618397217, Avg Adv LID: 5.801815968745398
Layer 29 - Avg Clean LID: 0.3617475726952286, Avg Adv LID: 5.697344580573415
Layer 30 - Avg Clean LID: 0.3591260393672384, Avg Adv LID: 5.748839959549467
Layer 31 - Avg Clean LID: 0.3529432230999547, Avg Adv LID: 5.912206113225795

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