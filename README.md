# Investigating the Impact of Model Instability on Explanations and Uncertainty
This is a repository for the paper [Investigating the Impact of Model Instability on Explanations and Uncertainty](https://arxiv.org/abs/2402.13006).

Explainable AI methods facilitate the understanding of model behaviour, yet, small, imperceptible perturbations to inputs can vastly distort explanations. As these explanations are typically evaluated holistically, before model deployment, it is difficult to assess when a particular explanation is trustworthy. Some studies have tried to create confidence estimators for explanations, but none have investigated an existing link between uncertainty and explanation quality. We artificially simulate epistemic uncertainty in text input by introducing noise at inference time. In this large-scale empirical study, we insert different levels of noise perturbations and measure the effect on the output of pre-trained language models and different uncertainty metrics. Realistic perturbations have minimal effect on performance and explanations, yet masking has a drastic effect. We find that high uncertainty doesn't necessarily imply low explanation plausibility; the correlation between the two metrics can be moderately positive when noise is exposed during the training process. This suggests that noise-augmented models may be better at identifying salient tokens when uncertain. Furthermore, when predictive and epistemic uncertainty measures are over-confident, the robustness of a saliency map to perturbation can indicate model stability issues. Integrated Gradients shows the overall greatest robustness to perturbation, while still showing model-specific patterns in performance; however, this phenomenon is limited to smaller Transformer-based language models

## Structure

### utils
The folder contains utility functions shared across all other codes

### training
This folder contains scripts for hyperparameter tuning and training of the models investigated; given the size of the models used and our compute restrictions, we have seperate files for _small' and '_large' models; Large models (i.e. greater than 300 million parameters) are run with lower-precision and parallel processing across two GPUs. The supported models are BERT, ELECTRA, RoBERTa, GPT2, and OPT. The supported datasets are SST-2, SemEval, eSNLI, and HateXplain. The models and datasets are not included in this repository.

### processing
This folder contains scripts used to process and noise-augment the datasets.

### experiments
This folder contains scripts to run experiments on clean and noise-augmented data.

### evaluation
This folder contains scripts to aggregate and process the results of experiments for evaluation, as well as to evaluate the final results.
