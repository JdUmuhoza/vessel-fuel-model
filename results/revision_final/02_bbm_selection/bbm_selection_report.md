# Standalone BBM selection on the full-data 60/20/20 protocol

Folder 01 selected a backend in the **residual** role and reused it as the standalone BBM. This stage selects the BBM independently, training every candidate directly on measured shaft power. The residual learner (Huber) and lambda* = 0.7 are carried over unchanged.

## Split identity - no drift, no leakage

| partition | n here | n folder 01 | time_id range matches | used in BBM selection |
|---|---|---|---|---|
| train | 318,423 | 318,423 | yes | yes |
| validation | 106,141 | 106,141 | yes | yes |
| internal_test | 106,142 | 106,142 | yes | no |
| dev_in | 18,368 | 18,368 | yes | no |
| dev_out | 18,368 | 18,368 | yes | no |

Physics scale k = 0.393586, identical to folder 01. `dev_in` and `dev_out` were used only after the BBM was selected.

## BBM validation ranking (target = measured shaft power)

| model | MAPE | sMAPE | MAE | nMAE | RMSE | NRMSE | R2 | mean rank |
|---|---|---|---|---|---|---|---|---|
| ExtraTrees | 13.1393 | 10.4965 | 1661.9 | 0.0783 | 2031.6 | 0.0957 | 0.9157 | 2.43 |
| CatBoost | 13.5398 | 11.1411 | 1945.5 | 0.0917 | 2256.5 | 0.1063 | 0.8960 | 4.43 |
| XGBoost | 14.2246 | 11.6809 | 1943.4 | 0.0916 | 2235.6 | 0.1054 | 0.8979 | 4.00 |
| LightGBM | 15.7228 | 12.2798 | 1967.8 | 0.0927 | 2263.3 | 0.1067 | 0.8954 | 5.71 |
| MLP | 16.7180 | 13.3091 | 1979.8 | 0.0933 | 2554.0 | 0.1204 | 0.8668 | 9.29 |
| HistGB | 17.1608 | 14.1910 | 2632.9 | 0.1241 | 2926.8 | 0.1379 | 0.8250 | 10.57 |
| ElasticNet | 18.5407 | 14.7923 | 1639.3 | 0.0773 | 2284.3 | 0.1076 | 0.8934 | 4.43 |
| BayesianRidge | 18.5682 | 14.8069 | 1641.0 | 0.0773 | 2286.6 | 0.1078 | 0.8932 | 5.43 |
| Lasso | 18.5683 | 14.8070 | 1641.0 | 0.0773 | 2286.6 | 0.1078 | 0.8932 | 6.43 |
| Ridge | 18.5683 | 14.8070 | 1641.0 | 0.0773 | 2286.6 | 0.1078 | 0.8932 | 7.43 |
| OLS | 18.5683 | 14.8070 | 1641.0 | 0.0773 | 2286.6 | 0.1078 | 0.8932 | 8.43 |
| Bagging | 18.7341 | 15.2698 | 2769.7 | 0.1305 | 3203.5 | 0.1510 | 0.7904 | 14.29 |
| RandomForest | 18.7647 | 15.2646 | 2756.3 | 0.1299 | 3190.8 | 0.1504 | 0.7921 | 13.57 |
| DecisionTree | 20.0142 | 17.9481 | 3474.0 | 0.1637 | 4171.4 | 0.1966 | 0.6446 | 16.57 |
| Huber | 20.9070 | 15.7871 | 1687.6 | 0.0795 | 2444.0 | 0.1152 | 0.8780 | 10.57 |
| AdaBoost | 22.0973 | 15.5184 | 2614.1 | 0.1232 | 3068.2 | 0.1446 | 0.8077 | 13.29 |
| GBT | 23.2098 | 17.6079 | 3032.1 | 0.1429 | 3282.5 | 0.1547 | 0.7799 | 16.14 |
| KNN | 26.9475 | 22.1253 | 4460.0 | 0.2102 | 5494.8 | 0.2589 | 0.3834 | 18.00 |

**Selected standalone BBM: ExtraTrees**

## Frozen evaluation - internal_test

| architecture | MAPE | sMAPE | MAE | nMAE | RMSE | NRMSE | R2 |
|---|---|---|---|---|---|---|---|
| BBM (ExtraTrees, selected on y) | 14.7301 | 11.8489 | 1811.7 | 0.0906 | 2192.2 | 0.1096 | 0.9302 |
| BBM (Huber, folder-01 reuse of the residual learner) | 19.2203 | 16.1431 | 1960.5 | 0.0980 | 2528.5 | 0.1264 | 0.9072 |
| WBM | 8.4602 | 7.7739 | 1405.5 | 0.0703 | 1975.2 | 0.0988 | 0.9434 |
| S-GBM (Huber residual) | 10.7241 | 9.0669 | 1524.2 | 0.0762 | 1975.4 | 0.0988 | 0.9434 |
| C-GBM (Huber residual, lambda*=0.7) | 9.6547 | 8.3536 | 1455.1 | 0.0728 | 1938.7 | 0.0969 | 0.9454 |

## Frozen evaluation - dev_in

| architecture | MAPE | sMAPE | MAE | nMAE | RMSE | NRMSE | R2 |
|---|---|---|---|---|---|---|---|
| BBM (ExtraTrees, selected on y) | 8.4474 | 6.7069 | 1013.6 | 0.0498 | 1528.1 | 0.0751 | 0.9681 |
| BBM (Huber, folder-01 reuse of the residual learner) | 18.8200 | 14.5595 | 1724.4 | 0.0848 | 2485.5 | 0.1222 | 0.9156 |
| WBM | 8.8477 | 8.5124 | 1534.1 | 0.0754 | 2110.0 | 0.1037 | 0.9392 |
| S-GBM (Huber residual) | 8.7883 | 7.6725 | 1277.5 | 0.0628 | 1762.7 | 0.0866 | 0.9575 |
| C-GBM (Huber residual, lambda*=0.7) | 8.2376 | 7.4107 | 1290.3 | 0.0634 | 1808.1 | 0.0889 | 0.9553 |

## Frozen evaluation - dev_out

| architecture | MAPE | sMAPE | MAE | nMAE | RMSE | NRMSE | R2 |
|---|---|---|---|---|---|---|---|
| BBM (ExtraTrees, selected on y) | 19.3482 | 14.4923 | 2053.0 | 0.1235 | 2509.1 | 0.1509 | 0.8498 |
| BBM (Huber, folder-01 reuse of the residual learner) | 21.9304 | 17.8871 | 2216.2 | 0.1333 | 2862.9 | 0.1722 | 0.8045 |
| WBM | 9.2137 | 9.0818 | 1358.3 | 0.0817 | 1931.5 | 0.1162 | 0.9110 |
| S-GBM (Huber residual) | 9.5449 | 8.2396 | 1169.9 | 0.0704 | 1608.1 | 0.0967 | 0.9383 |
| C-GBM (Huber residual, lambda*=0.7) | 8.7324 | 7.8212 | 1149.9 | 0.0691 | 1630.8 | 0.0981 | 0.9366 |

