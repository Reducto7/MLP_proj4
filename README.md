# Team Project 4: Hull Tactical Market Prediction with BTC Extension

**Teammates:**  
• XU LINRUI, 50251600  
• HUANG FANRU, 20214788  
• FANG JINGYI, 20223178  
• FEI XIZE, 20212288  

---

## Overview

This project focuses on the Hull Tactical Market Prediction task, which aims to predict daily excess returns of the S&P 500 using machine learning under a volatility-constrained trading framework. The original dataset is provided through a Kaggle competition and contains multi-source economic, market, sentiment, and price-related features (P, M, E, I, V, S, D series).

We first establish baseline regression and momentum-based models, followed by advanced machine learning and deep learning approaches including LightGBM and LSTM with time-series feature engineering and walk-forward validation. Model performance is evaluated using the modified Sharpe ratio, cumulative return, volatility ratio, and drawdown analysis.

As an optional bonus extension, we further adapt the full modeling pipeline to the cryptocurrency market (BTC-USD). We construct a custom dataset using the Yahoo Finance API, map BTC OHLCV data into a compatible Hull Tactical feature format, and apply an optimized LSTM forecasting system with uncertainty estimation.

This repository contains all code, processed datasets, evaluation results, and final report for full reproducibility. The project was developed as part of the CS 53744 Machine Learning Project (Fall 2025).

---

## Reproduction Instructions

### 1. Environment Setup
- Platform: Kaggle Notebook  
- Python Version: Python 3.10  
- Core Libraries:
  - numpy, pandas  
  - scikit-learn  
  - lightgbm  
  - tensorflow / keras  
  - matplotlib, seaborn  
- Random seed is fixed for reproducibility.

---

### 2. Main S&P 500 Model Execution
1. Open the Kaggle notebook for the Hull Tactical competition.
2. Load the official dataset:
   - `/kaggle/input/hull-tactical-market-prediction/train.csv`
   - `/kaggle/input/hull-tactical-market-prediction/test.csv`
3. Run the modeling pipeline in order:
   - EDA & Data Quality Analysis  
   - Baseline Model (Regression / Momentum Strategy)  
   - Feature Engineering (Momentum, Volatility, Technical Indicators)  
   - Advanced Models (LightGBM / LSTM)  
   - Time-Series Cross-Validation  
   - Backtesting & Sharpe-Ratio Evaluation  
4. Generate and submit the final prediction CSV to Kaggle.

---

### 3. BTC Bonus Extension Execution
1. Run the BTC data adaptation script to generate:
   - `btc_train_adapted.csv`
   - `btc_test_adapted.csv`
2. Run the optimized BTC LSTM forecasting notebook including:
   - Advanced feature engineering  
   - Outlier control  
   - Time-series window construction  
   - Regularized LSTM training  
   - Uncertainty quantification (confidence intervals)  
3. Output files:
   - `btc_optimized_results.csv`  
   - `btc_enhanced_prediction_results.png`

---

### 4. Runtime
- Main Kaggle model training: ~1–2 hours  
- BTC LSTM extension: ~30–60 minutes  
- Recommended hardware: GPU-enabled Kaggle environment  

---

## Model Architecture

### S&P 500 Main Model
- Baseline: Linear regression, momentum-based allocation  
- Advanced Model: LightGBM with engineered market features  
- Validation: Walk-forward time-series split  
- Risk Control: Volatility ≤ 120% of benchmark  
- Evaluation:
  - Modified Sharpe ratio  
  - Cumulative return  
  - Volatility ratio  
  - Maximum drawdown  

### BTC Extension Model
- Input: OHLCV → Mapped into P1–P4, V1  
- Feature Engineering:
  - Price range, RSI, EMA, momentum, volatility  
- Model:
  - Regularized LSTM (32 units)  
  - Dropout + L2 weight decay  
- Output:
  - Price-change prediction  
  - Reconstructed price forecast  
  - 95% confidence interval  

---

## Key Results

### S&P 500 Task
- Baseline model successfully established the Kaggle submission pipeline.  
- LightGBM and LSTM significantly improved:
  - Return stability  
  - Risk-adjusted Sharpe score  
- Volatility constraint (≤ 1.2× benchmark) successfully satisfied.

### BTC Bonus Task
- Optimized LSTM achieved:
  - Improved R² and MAPE over baseline  
  - Stable confidence intervals  
  - Strong trend-direction accuracy  
- Full uncertainty-aware BTC forecasting successfully implemented.

---

## Limitations and Notes

- Financial time-series data is inherently noisy and partially non-stationary.  
- Extreme crypto volatility introduces unavoidable prediction uncertainty.  
- Hyperparameter tuning was limited by computational budget.  
- This project is for educational and research purposes only and must not be used for real-world trading or investment decisions.

---

## Contact

For questions or technical issues, please contact any Team 8 member via GitHub Issues:

- XU LINRUI  
- HUANG FANRU  
- FANG JINGYI  
- FEI XIZE  
