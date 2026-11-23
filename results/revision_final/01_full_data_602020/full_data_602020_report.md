# Definitive full-data results: chronological 60/20/20 development protocol

Full `train.csv` = 530,706 records, split chronologically without shuffling. The earlier 40,000-record uniform thinning is not used.

| block | n | role |
|---|---|---|
| training (first 60 %) | 318,423 | physics calibration, learner fitting |
| -- parameter fitting | 286,580 | first 90 % of the training block |
| -- early stopping | 31,843 | last 10 % of the training block, boosted libraries only |
| validation (next 20 %) | 106,141 | backend selection, lambda selection |
| internal test (final 20 %) | 106,142 | frozen check |
| dev_in | 18,368 | benchmark holdout |
| dev_out | 18,368 | benchmark holdout |

Physics scale k = 0.393586, fitted on the training block alone. Neither `dev_in` nor `dev_out` entered any fit, selection, tuning or early-stopping decision.

## Excluded models - measured, not assumed

| model | stated cost | fit at 1k | fit at 2k | exponent | projected at full n | decision |
|---|---|---|---|---|---|---|
| GaussianProcess | O(n^3) kernel factorisation | 10.46 s | 40.28 s | 1.95 | 215.0 h | EXCLUDED |
| SVR (RBF) | O(n^2)-O(n^3) dual solve | 0.13 s | 0.53 s | 2.00 | 3.7 h | EXCLUDED |
| PNN (GRNN) | O(n_train x n_query) kernel-weighted prediction | 0.17 s | 0.26 s | 0.65 | 0.0 h | retained |
| KNN | O(n_train x n_query) neighbour search | 0.05 s | 0.03 s | -0.83 | 0.0 h | retained |
| PNN (GRNN) | O(n_train x n_query) kernel-weighted prediction; 286,580 x 106,141 = 3.04e10 kernel evaluations | nan s | nan s | nan | nan h | EXCLUDED |

Excluded models are omitted from the ranking entirely rather than subsampled, so every retained model is compared on identical training data.

## Section 5.1 - model selection on the validation block

| model | MAPE | sMAPE | MAE | nMAE | RMSE | NRMSE | R2 | mean rank | ES |
|---|---|---|---|---|---|---|---|---|---|
| Huber | 8.0010 | 7.2963 | 1251.3 | 0.0590 | 1649.3 | 0.0777 | 0.9444 | 1.00 |  |
| CatBoost | 8.8039 | 8.2737 | 1612.9 | 0.0760 | 1956.8 | 0.0922 | 0.9218 | 6.29 | yes |
| OLS | 8.8854 | 7.8145 | 1278.9 | 0.0603 | 1659.6 | 0.0782 | 0.9437 | 2.14 |  |
| Ridge | 8.8854 | 7.8145 | 1278.9 | 0.0603 | 1659.6 | 0.0782 | 0.9437 | 3.14 |  |
| Lasso | 8.8854 | 7.8145 | 1278.9 | 0.0603 | 1659.6 | 0.0782 | 0.9437 | 4.14 |  |
| BayesianRidge | 8.8855 | 7.8145 | 1278.9 | 0.0603 | 1659.7 | 0.0782 | 0.9437 | 5.14 |  |
| ElasticNet | 8.8859 | 7.8148 | 1279.3 | 0.0603 | 1660.3 | 0.0782 | 0.9437 | 6.14 |  |
| XGBoost | 10.0657 | 9.3121 | 1868.2 | 0.0880 | 2217.1 | 0.1045 | 0.8996 | 8.43 | yes |
| LightGBM | 10.3984 | 9.6440 | 1991.0 | 0.0938 | 2397.4 | 0.1130 | 0.8826 | 9.71 | yes |
| MLP | 12.3911 | 11.5063 | 2084.3 | 0.0982 | 2990.2 | 0.1409 | 0.8174 | 13.43 |  |
| DecisionTree | 12.9065 | 11.5429 | 2209.8 | 0.1041 | 2644.0 | 0.1246 | 0.8572 | 12.86 |  |
| ExtraTrees | 12.9401 | 11.0926 | 1938.2 | 0.0913 | 2197.1 | 0.1035 | 0.9014 | 9.14 |  |
| RandomForest | 14.1528 | 12.2002 | 2193.3 | 0.1034 | 2556.7 | 0.1205 | 0.8665 | 11.86 |  |
| Bagging | 14.2567 | 12.2548 | 2203.9 | 0.1039 | 2567.9 | 0.1210 | 0.8653 | 12.86 |  |
| AdaBoost | 16.3314 | 12.9647 | 2358.7 | 0.1112 | 2746.8 | 0.1294 | 0.8459 | 15.00 |  |
| KNN | 17.0537 | 13.6988 | 2530.0 | 0.1192 | 3126.0 | 0.1473 | 0.8004 | 17.14 |  |
| HistGB | 18.4020 | 14.4963 | 2430.2 | 0.1145 | 2705.7 | 0.1275 | 0.8505 | 15.43 |  |
| GBT | 19.5050 | 15.5664 | 2657.6 | 0.1252 | 2874.4 | 0.1355 | 0.8313 | 17.14 |  |

Selected on the primary metric (MAPE): **Huber**.

## Section 5.2 - lambda selection on the same validation block

| metric | optimal lambda | at optimum | at 0 (WBM) | at 1 (S-GBM) |
|---|---|---|---|---|
| MAPE **(primary)** | **0.7** | 7.7206 | 9.0593 | 8.0010 |
| sMAPE | **0.76** | 7.1280 | 8.5885 | 7.2963 |
| MAE | **1** | 1251.2815 | 1783.6487 | 1251.2815 |
| nMAE | **1** | 0.0590 | 0.0841 | 0.0590 |
| RMSE | **1** | 1649.2641 | 2276.8786 | 1649.2641 |
| NRMSE | **1** | 0.0777 | 0.1073 | 0.0777 |
| R2 | **1** | 0.9444 | 0.8941 | 0.9444 |

lambda* = **0.7** on MAPE. Across the seven metrics the optimum ranges 0.7-1. Endpoint identities verified: lambda = 0 reproduces the WBM and lambda = 1 the S-GBM exactly on all seven metrics.

## Internal test (frozen)

| architecture | MAPE | sMAPE | MAE | nMAE | RMSE | NRMSE | R2 |
|---|---|---|---|---|---|---|---|
| WBM | 8.4602 | 7.7739 | 1405.5 | 0.0703 | 1975.2 | 0.0988 | 0.9434 |
| BBM | 19.2203 | 16.1431 | 1960.5 | 0.0980 | 2528.5 | 0.1264 | 0.9072 |
| S-GBM | 10.7241 | 9.0669 | 1524.2 | 0.0762 | 1975.4 | 0.0988 | 0.9434 |
| C-GBM (lambda*=0.7) | 9.6547 | 8.3536 | 1455.1 | 0.0728 | 1938.7 | 0.0969 | 0.9454 |

## Final benchmark evaluation (frozen)

| partition | architecture | MAPE | sMAPE | MAE | nMAE | RMSE | NRMSE | R2 |
|---|---|---|---|---|---|---|---|---|
| dev_in | WBM | 8.8477 | 8.5124 | 1534.1 | 0.0754 | 2110.0 | 0.1037 | 0.9392 |
| dev_in | BBM | 18.8200 | 14.5595 | 1724.4 | 0.0848 | 2485.5 | 0.1222 | 0.9156 |
| dev_in | S-GBM | 8.7883 | 7.6725 | 1277.5 | 0.0628 | 1762.7 | 0.0866 | 0.9575 |
| dev_in | C-GBM (lambda*=0.7) | 8.2376 | 7.4107 | 1290.3 | 0.0634 | 1808.1 | 0.0889 | 0.9553 |
| dev_out | WBM | 9.2137 | 9.0818 | 1358.3 | 0.0817 | 1931.5 | 0.1162 | 0.9110 |
| dev_out | BBM | 21.9304 | 17.8871 | 2216.2 | 0.1333 | 2862.9 | 0.1722 | 0.8045 |
| dev_out | S-GBM | 9.5449 | 8.2396 | 1169.9 | 0.0704 | 1608.1 | 0.0967 | 0.9383 |
| dev_out | C-GBM (lambda*=0.7) | 8.7324 | 7.8212 | 1149.9 | 0.0691 | 1630.8 | 0.0981 | 0.9366 |

Shift comparison (MAPE, dev_out - dev_in):

| architecture | dev_in | dev_out | dMAPE |
|---|---|---|---|
| BBM | 18.8200 | 21.9304 | +3.1103 |
| C-GBM (lambda*=0.7) | 8.2376 | 8.7324 | +0.4948 |
| S-GBM | 8.7883 | 9.5449 | +0.7566 |
| WBM | 8.8477 | 9.2137 | +0.3660 |

## Comparison with the previous 40,000-record experiment

- training data: 318,423 vs 34,000 records (9.4x)
- selection data: 106,141 vs 6,000 records
- selected backend: **Huber** vs CatBoost
- lambda*: **0.7** vs 0.25
- an internal test block now sits between selection and the benchmark sets, which the earlier protocol lacked entirely.

