# DeepLearning Project AS 24: Benchmarking Neural Latent Representations on EEG data for Sleep Stage Classification

This file explains how our project implementation integrates in this repo and how to reproduce the training and benchmarks.

Please find the report also in this repo: ""

## Abstract
 Sleep Stage Classification (SSC) is critical in understanding
sleep physiology and diagnosing disorders.
This study addresses a key gap in understanding the
role of Self-Supervised-Learning (SSL) in EEG-based
SSC by benchmarking three SSL paradigms in combination
with three representative encoder architectures.
We focus on the learned representation quality, offering
insights into optimizing SSL frameworks for robust
and generalizable SSC systems. Our Latent-space
benchmarks and linear evaluation suggest clear
evidence of how to best combine the SSL paradigm and
backbone architectures. 

## Project Integration

We added 3 different training scripts to train our backbone models with three different Self-Supervised-Learning paradigms:

- train_mp.py
- train_crl_dlproj.py
- train_hybrid.py

We also implemented the pretrain backbones in:

- models/cnn
- models/cnn_attention
- models/transformer
- models/main_model_dlproj.py

To run the training and benchmarks you must use the configurations that start with "DLPROJ_pretrain" in the configs/ folder.
To calculate latent space benchmarks and create visualizations we implemented:

- latent_space_evaluation/
- latent_space_evaluator.py

Finally we implemented some utils and loss functions and adapted the requirements in 
- utils.py
- loss.py
- sleepnet_environment.yaml (for conda environments)
- requirements.txt 
- 
## Run training

To run the training you need to choose which pretraining paradigm to use and choose the respectivetraining script.
If you want to run training for Masked Prediction the command would be:

```bash
python train_mp.py --config configs/DLPROJ_pretrain_MP_CNN_Sleep-EDF-2018.json
```

This will run the backbone pre training, generate the latent space vectors on the test-set for later latent-space evaluation, train the classifier linear evaluation and finally perform it.

Keep in mind, to only use configurations that are meant for the respective SSL paradigm. The pre-train mode that is supported by a config is indicated by:

- "MP" for Masked Prediction and use with "train_mp.py"
- "CRL" for Contrastive Learning and use with "train_crl_dlproj.py"
- "HYBRID" for a mix of CRL and MP and use with "train_hybrid.py"

After this you will find the model checkpoint as well as the generated embeddings in the "checkpoints/" folder and tensorboard logs in the "logs/" folder.
Linear evaluation results are printed to terminal and written to a text file in "results/".

## Run Latentspace Benchmarks

To run benchmarks and visualizations for the latent space you need to have the embeddings already generated (they will be in the respective checkpoint folder after running the training script).
Also you will need a lot of RAM (around 32 Gb at minimum).

To finally run the benchmarks execute:

```bash
python latent_space_evaluator.py --config configs/DLPROJ_pretrain_MP_CNN_Sleep-EDF-2018.json
```

The config here is used to find the stored weights and determine where to store the results.
The latent space visualizations and metrics will be found in a subfolder of "results/" which is named as the configuration.
