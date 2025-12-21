🚑 Patient Satisfaction Prediction
Predicting Hospital Patient Satisfaction Using Operational Data (Beds, Demand, Staff Morale)
Overview

This repository demonstrates a data-driven framework to predict patient satisfaction based on operational hospital indicators, including bed supply, admission pressure, refusal counts, and staff morale.

The project evaluates three machine learning models:

Linear Regression (Baseline)

Random Forest Regressor

XGBoost

The objective is to determine whether patient satisfaction can be accurately predicted and to identify key operational drivers for hospital management decisions.

📊 Key Results
Model	MAE	RMSE	R²
Linear Regression	8.9753	10.6345	0.0043
XGBoost	9.1564	11.3030	-0.1248
⭐ Random Forest	⭐ 0.1340	⭐ 0.1704	⭐ 0.9563

Conclusion:
➡️ Patient satisfaction is non-linear
➡️ Random Forest overwhelmingly outperforms other models
➡️ Staff morale is the most critical feature

🧠 Feature Importance (Random Forest)

Staff Morale

Available Beds

Admitted Patients

Patient Requests

Refusals

Human factors matter more than physical bed supply — improving morale yields the highest ROI.
