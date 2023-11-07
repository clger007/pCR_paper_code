# Project Readme: Pathology Report Analysis with Language Models

## Overview

This project aims to evaluate the capabilities of Large Language Models (LLMs) in interpreting natural text within pathology reports, particularly in identifying pathologic complete response (pCR) within these reports. We conducted three experiments to assess the performance of LLMs, utilizing various evaluation metrics and techniques.

## Methods

### Evaluation Metrics

To assess the capabilities of LLMs, we used several evaluation metrics, including:
- Positive Predictive Value (PPV)
- Sensitivity
- Specificity
- Negative Predictive Value (NPV)
- F1-score
- Accuracy
- Area Under the Curve (AUC)
- Precision-Recall AUC (PRAUC)

We employed a bootstrapping resampling method to determine the 95% confidence interval for each of these metrics to enhance the reliability of our results.

### Experiment One: Identifying pCR from De-identified Reports with GPT APIs

In this experiment, we focused on de-identified pathology reports. We utilized OpenAI's Generative Pretrained Transformer (GPT) versions 3.5 and 4 via APIs. The experiment followed a two-step strategy:

**Step 1**: We used GPT-3.5 to condense the de-identified pathology reports, retaining only relevant information. The guiding prompt instructed the model to provide a summary while excluding patient-specific and de-identified data.

**Step 2**: GPT-4 API was used to determine if a patient achieved pCR based on the summarized pathology report. The model adhered strictly to predefined guidelines to identify pCR.

### Experiment Two: Identify pCR with Custom Machine Learning Pipelines

In this experiment, we processed the pathology reports without de-identification. We aimed to evaluate LLMs' performance with complete information. We utilized locally deployable models from the Bidirectional Encoder Representations from Transformers (BERT), BART, T5, and GPT families. As these models have input length limitations, we divided longer texts into manageable chunks with an overlapping strategy to preserve context.

**Logistic Regression Classifier**: Extracted embeddings were fed into a logistic regression classifier. We balanced the dataset using SMOTE and optimized hyperparameters using Bayesian optimization.

### Experiment Three: Fine-Tuning Local Hosted LLMs

In this experiment, we fine-tuned a smaller variant of the GPT-2 architecture (GPT-2 small) to enhance its performance for pCR identification.

**Model Architecture**: The neural architecture involved mean pooling of GPT-2 output embeddings, followed by a fully connected layer and a sigmoid activation function for pCR classification.

**Fine-Tuning Procedure**: We employed Low-Rank Adaptation of Large Language Models (LoRA) to reduce the number of trainable parameters efficiently. The Adam optimizer and a learning rate scheduler were used to fine-tune the model.

## Conclusion

This project explores the use of Large Language Models in interpreting pathology reports for pCR identification. Through three experiments, we evaluated their performance using various techniques and metrics, aiming to provide valuable insights into the capabilities of LLMs in the medical domain.

For further details, refer to our research paper and code repository.
