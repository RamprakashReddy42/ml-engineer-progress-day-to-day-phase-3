# Phase 3: Feature Engineering — Machine Learning Learning Journey

## Overview

This phase focuses on understanding how to transform existing features, validate the transformations, analyze distributions, and prepare features for the next stage of a Machine Learning workflow.

I am following a step-by-step Machine Learning learning path using Python and a Credit Card Fraud Detection dataset to develop practical skills relevant to a Machine Learning Engineer role.

**Focus:** Learning, understanding, validating, and documenting ML workflows — not building a separate project.

## Learning Objectives

- Understand feature engineering and why it matters in Machine Learning.
- Apply logarithmic transformation to numerical features.
- Compare original and transformed distributions.
- Analyze skewness before and after transformation.
- Validate engineered features programmatically.
- Understand feature correlation and descriptive statistics.
- Check missing values and infinite values.
- Understand target-class distribution and class imbalance.
- Review feature readiness before model training.

## Tools and Libraries

| Tool / Library | Purpose |
|---|---|
| Python | Core programming language |
| Pandas | Data manipulation and statistical summaries |
| NumPy | Numerical calculations and transformation validation |
| Matplotlib | Visualizing feature distributions |
| Jupyter Notebook | Running code and analyzing results |

## Dataset Understanding

The learning workflow uses a Credit Card Fraud Detection dataset containing features such as:

- `Time` — transaction time information.
- `V1` to `V28` — anonymized numerical features.
- `Amount` — transaction amount.
- `Class` — target label indicating legitimate or fraudulent transactions.

In the commonly used version of this dataset, `Class = 0` represents a legitimate transaction and `Class = 1` represents a fraudulent transaction.

## Step-by-Step Learning Progress

### Step 14: Feature Transformation

Learned how to create a logarithmically transformed feature from `Amount`.

```python
import numpy as np

df["Log_Amount"] = np.log1p(df["Amount"])
```

**What I learned:**
- `np.log1p(x)` calculates the natural logarithm of `1 + x`.
- Log transformation compresses large numerical values.
- Transformation can help make some statistical patterns easier to analyze.
- A transformation does not guarantee improved model performance.

### Step 15: Feature Validation

Learned to validate the engineered feature against its expected calculation.

```python
expected = np.log1p(df["Amount"])

valid = np.allclose(
    df["Log_Amount"],
    expected,
    equal_nan=True
)

print("Transformation valid:", valid)
```

This checks whether the transformed feature matches the expected values, allowing for small floating-point differences.

### Step 16: Feature Correlation

Learned to calculate correlations between the original amount, transformed amount, and target label.

```python
correlation_matrix = df[
    ["Amount", "Log_Amount", "Class"]
].corr()

print(correlation_matrix)
```

**What I learned:**
- Correlation measures the strength and direction of a linear relationship.
- The original and transformed features may be strongly correlated.
- Correlation does not establish causation or prove predictive usefulness.
- Correlation with the target alone is not enough to select a feature.

### Steps 17–18: Distribution Analysis and Visualization

Learned to compare the original and transformed features using histograms.

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].hist(df["Amount"].dropna(), bins=50)
axes[0].set_title("Original Amount Distribution")
axes[0].set_xlabel("Amount")
axes[0].set_ylabel("Frequency")

axes[1].hist(df["Log_Amount"].dropna(), bins=50)
axes[1].set_title("Log_Amount Distribution")
axes[1].set_xlabel("Log_Amount")
axes[1].set_ylabel("Frequency")

plt.tight_layout()
plt.show()
```

**What I learned:**
- Histograms show the frequency of values across intervals called bins.
- Log transformation changes the scale of the feature.
- Histograms should be interpreted with their different scales in mind.
- Similar-looking charts do not automatically mean a transformation failed.

### Step 19: Skewness Comparison

Learned to compare the skewness of the original and transformed features.

```python
amount_skew = df["Amount"].skew()
log_amount_skew = df["Log_Amount"].skew()

print("Amount skewness:", amount_skew)
print("Log_Amount skewness:", log_amount_skew)
```

**What I learned:**
- Positive skewness indicates a longer right tail.
- Negative skewness indicates a longer left tail.
- Comparing absolute skewness helps assess whether asymmetry decreased.
- Lower skewness does not automatically mean better model performance.

### Step 20: Feature Transformation Review

Reviewed the transformation using:
- Original and transformed values.
- Transformation formula validation.
- Skewness comparison.
- Descriptive statistics.
- Quantiles and distribution changes.

The objective is to evaluate the effect of a transformation using evidence rather than assumptions.

### Step 21: Final Feature Readiness Review

Learned to perform basic checks before moving to data preparation.

```python
required_columns = ["Amount", "Log_Amount", "Class"]

print(df[required_columns].isnull().sum())
```

Other checks covered in this step include:
- Verifying required columns.
- Checking missing values.
- Checking infinite values.
- Validating the transformation formula.
- Reviewing target-class distribution.

The target column `Class` must remain separate from model input features to prevent target leakage.

## Key Takeaways

1. Feature engineering involves more than creating new columns.
2. Every transformation should be validated.
3. Statistical summaries and visualizations help explain feature behavior.
4. Correlation and skewness provide useful information, but neither guarantees predictive value.
5. Class imbalance matters when preparing fraud-detection data.
6. Features should be evaluated carefully before model training.

## Phase 3 Status

**Status:** Learning and review of feature engineering concepts covered in Steps 14–21.

**Next phase:** Phase 4 — Data Preparation.

The next stage will focus on preparing model inputs and targets, splitting the data appropriately, and preventing data leakage before model training.

## Learning Goal

My goal is to develop practical Machine Learning Engineer skills through consistent coding practice, statistical analysis, feature validation, and an understanding of industry-style workflows.

This repository documents my learning progress step by step as I continue strengthening my Python and Machine Learning fundamentals.
