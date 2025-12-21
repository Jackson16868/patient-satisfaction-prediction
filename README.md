# Patient Satisfaction Prediction

Hospital Operational Data Machine Learning Study

## Overview

This repository builds a machine learning pipeline that predicts patient satisfaction using operational hospital variables such as bed availability, service demand, admission volume, refusal counts, and staff morale.

Three regression models were compared to evaluate prediction performance and identify the most suitable method:

* Linear Regression (baseline)
* Random Forest Regressor
* XGBoost

The goal is to generate actionable insights for hospital resource allocation and management decisions.

## Key Results

Random Forest performs significantly better than linear and boosting methods, demonstrating that patient satisfaction is driven by nonlinear interactions.

Performance comparison:

| Model             |    MAE |    RMSE |      R2 |
| ----------------- | -----: | ------: | ------: |
| Linear Regression | 8.9753 | 10.6345 |  0.0043 |
| XGBoost           | 9.1564 | 11.3030 | -0.1248 |
| Random Forest     | 0.1340 |  0.1704 |  0.9563 |

## Feature Importance

Top drivers based on Random Forest:

1. Staff Morale
2. Available Beds
3. Admitted Patients
4. Patient Requests
5. Refused Patients


