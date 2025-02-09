
## Custom Loss Function to Improve TFT Performance on Seasonal Data 🎯

## Overview 📊

This project focuses on enhancing the performance of the Temporal Fusion Transformer (TFT) model, specifically for forecasting on seasonal product categories with extreme sales spikes. The goal was to develop a custom loss function that can help the TFT model better handle these seasonal sales patterns while maintaining stable performance during low sales periods. The custom loss function increases the weight given to weeks with high sales and products with high sales to improve model accuracy on seasonal peaks.

## Background 📚

The current TFT model, which achieves a weighted SMAPE of 37%, faces challenges with seasonal categories due to the imbalance in sales data. Most weeks have low or zero sales, followed by extreme spikes during certain seasons. Standard loss functions tend to overfit the model to low sales weeks, causing poor performance during seasonal peaks. To address this, we introduced a custom loss function to emphasize high sales weeks and Joint-IDs.

## Project Goals 🎯

1. **Understand the TFT model and our in-house trained model and code.**
2. **Improve TFT performance on seasonal categories.**
3. **Implement these improvements with minimal computational complexity.**

## Temporal Fusion Transformer (TFT) 📈

The TFT is an attention-based model designed for multi-horizon forecasting tasks. It can incorporate both time-series and static input variables and utilizes recurrent layers, attention mechanisms, and exogenous data to improve prediction accuracy. Given its design, the TFT is well-suited for forecasting tasks where multiple variables affect the target variable, like retail sales.

### Key Features of TFT ✨
- Handles multiple input variables.
- Integrates known future inputs for improved forecasting.
- Utilizes attention mechanisms to learn temporal relationships.

## Tools, Libraries, and Specifications 🛠️

- **Libraries**: PyTorch Forecasting, PyTorch Lightning
- **Hardware**: Trained on 1 NVIDIA Tesla V100 16GB GPU
- **Monitoring**: TensorBoard for tracking model performance during training

## Dataset 🗃️

Category 30 Adversion A was used as the benchmark dataset for this project.

## Hyperparameters ⚙️

Hyperparameters were kept the same as the final model used by the team, with no changes made.

## Loss Functions Tested 🧪

1. MSE
2. RMSE
3. SMAPE
4. Log Transform on Data + SMAPE
5. Custom Weighted Loss
6. Log Transform on Data + Custom Weighted Loss
7. Quadratically-Rooted and Squared Custom Weighted Loss

(Note: Only the best-performing models are reported here.)

## Loss Function Requirements 📝

Given the imbalanced nature of seasonal data, the custom loss function had to meet the following requirements:
1. Stable performance for targets and predictions near 0.
2. Increased weight for weeks and Joint-IDs with high sales.
3. Easy integration into the existing training process.

## Loss Function Discussion 🔍

### MAPE and SMAPE 📏

MAPE and SMAPE are popular loss functions but are unstable when targets or predictions are near 0. Since seasonal data often includes zero or near-zero sales, both MAPE and SMAPE were unsuitable for this task due to their tendency to penalize small errors heavily.

### MSE, RMSE, and MAE 📉

While these functions effectively highlight outliers, MSE and RMSE did not train well on the seasonal data, as the model tended to predict zero for most weeks, neglecting sales peaks. 

### Log Transform of Inputs 🔄

A log transformation was applied to the input data (before scaling) to reduce skew and improve training. This preprocessing step improved the model’s performance across both SMAPE and custom loss functions.

## Custom Weighted Loss ⚖️

### Goals 🎯
The custom weighted loss was designed to:
- Increase weight on high sales weeks and Joint-IDs.
- Handle edge cases like 0 or near-zero sales.
- Be easily integrated into the existing training pipeline.

### Loss Calculation 🧮

The weighted custom loss is calculated by:
1. **SMAPE** as the baseline.
2. **Local Weight**: The loss is multiplied by SCAN_QTY of that time-step to give higher weight to high sales weeks.
3. **Global Weight**: The result is scaled by the sum of SCAN_QTY for the particular Joint-ID over the forecast horizon.

### Edge Case Handling ⚠️

The custom loss ensures small loss values for edge cases where the target or predicted values are zero or near zero.

## Training Results 📈

The performance of various models was compared based on their overall SMAPE and SMAPE for high sales weeks (SCAN_QTY > 300). The custom loss function showed a significant improvement in high-sales week performance.

| Model | Overall SMAPE | High SCAN-QTY SMAPE | Improvement (Overall) | Improvement (High SCAN-QTY) |
|-------|----------------|---------------------|-----------------------|----------------------------|
| SMAPE Baseline | 39% | 91% | - | - |
| Log Transform + SMAPE | 35% | 71% | +4% | +20% |
| Custom Loss | 44% | 65% | -5% | +26% |
| Log Transform + Custom Loss | 43% | 64% | -4% | +27% |
| Balanced Ensemble | 34% | 64% | +5% | +27% |
| High SCAN-QTY Ensemble | 41% | 61% | +2% | +30% |

The **best standalone model** was Log Transform + Custom Loss, with a 27% improvement in high sales weeks.

## Model Attention Plots 👁️

Analyzing the attention weights revealed that the custom loss model focused more on high sales periods, correlating well with the weekly sales trends. The custom model also distributed attention more evenly across Joint-IDs with higher sales, improving the model's focus on important periods.

### Attention Weights Across Training Weeks ⏳

- **Custom loss model**: Attention was more correlated with weekly sales, showing stability and gradually increasing attention during seasonal peaks.
- **Baseline SMAPE model**: No clear relationship between attention and weekly sales.

### Attention Weights Across Joint-IDs 📊

- **Custom loss model**: Showed a correlation between Joint-IDs' total sales and attention weights.
- **Baseline SMAPE model**: Showed no correlation between sales and attention weights.

## Conclusion 🎯

The custom loss function successfully improved the TFT model's ability to forecast high sales weeks in seasonal categories. While performance on low-sales weeks decreased slightly, the model's focus on high-sales weeks aligns better with the business needs, making it a more suitable solution for seasonal forecasting.
**
