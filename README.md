# LID-Adversarial-Attack-Detection

This repository implements adversarial attack detection using Local Intrinsic Dimensionality (LID). The approach is inspired by the paper "[Characterizing Adversarial Subspaces Using Local Intrinsic Dimensionality](https://arxiv.org/abs/1801.02613)" by Ma et al.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
  - [Data Preparation](#data-preparation)
  - [Training the Model](#training-the-model)
  - [Generating Adversarial Examples](#generating-adversarial-examples)
  - [Detecting Adversarial Examples](#detecting-adversarial-examples)
- [Results](#results)
- [References](#references)

## Introduction

Adversarial attacks pose significant challenges to the robustness of machine learning models. This project utilizes the concept of Local Intrinsic Dimensionality (LID) to detect adversarial samples by analyzing the dimensional properties of data manifolds. The primary goal is to provide a robust method for identifying adversarial examples across different datasets and attack methods.

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/ArunimSamudra/LID-Adversarial-Attack-Detection.git
   cd LID-Adversarial-Attack-Detection
   ```

2. **Install the required dependencies:**

   Create a virtual env and install the dependencies

   ```bash
   pip install -r requirements.txt
   ```
   
## Usage

### Data Preparation

Download and preprocess the dataset (here it is IMDB) using the provided script:

```bash
bash data.sh
```

This new dataset contains both the clean and adversarial texts.

### Capturing Activations

To calculate LIDs for the generated dataset, run `run.sh`. The script `(script.py)` contains the code to iterate over all layers of the model and store the activations as tensors.

### Detecting Adversarial Examples

Use the stored activations to calculate the LIDs using `eval.sh`. This will also train a classifier and display the results.


