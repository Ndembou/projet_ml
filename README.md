 Frequency Modeling of Automobile Insurance Claims
 Project Overview

This project focuses on modeling the frequency of automobile insurance claims using classical statistical methods and machine learning models.
The objective is to predict the number of claims (ClaimNb) for each insurance policy based on driver, vehicle, and geographical characteristics.

This type of modeling is a core task in actuarial science and insurance pricing, as it directly impacts risk segmentation, tariff construction, and portfolio management.

The project is implemented in Python using scikit-learn, pandas, and matplotlib, and is based on the well-known French MTPL insurance dataset (freMTPL2freq).

 Dataset

Name: freMTPL2freq

Source: French Motor Third Party Liability (MTPL) dataset

Size: 678,013 insurance contracts

Target variable:

ClaimNb — Number of claims observed during the exposure period

Key features:

Exposure — Exposure duration (in years)

DrivAge — Driver age

VehAge — Vehicle age

VehPower — Vehicle power

BonusMalus — Bonus-malus coefficient

Density — Population density

Area, Region, VehBrand, VehGas — Categorical variables

Each row corresponds to one insurance contract.

 Problem Definition

The goal is to predict:

ClaimNb∼f(Driver features, Vehicle features, Geography, Exposure)

This is a count data problem with the following characteristics:

Strongly zero-inflated target variable (≈ 95% of policies have zero claims)

Highly imbalanced data

Weak linear correlations between features and the target

Because of these properties, classical linear regression is theoretically inappropriate, and Poisson-based models are natural candidates.

Models Implemented

The following models were trained and evaluated:

Linear Regression (baseline)

Decision Tree Regressor

Random Forest Regressor

Gradient Boosting Regressor

Linear Support Vector Regression (SVR)

Poisson Regression (GLM) with exposure as sample weight

Poisson Regression with hyperparameter tuning

Poisson Regression with resampling

Tree-based models aim to capture non-linear relationships, while Poisson regression provides a statistically coherent framework for count data.

Methodology

The project follows these main steps:

Data loading and library imports

Exploratory data analysis and visualization

Outlier filtering and data cleaning

One-hot encoding of categorical variables

Train / test split (80% / 20%)

Model training

Hyperparameter optimization using GridSearchCV

Model evaluation and comparison

Resampling experiments to address class imbalance

Evaluation Metrics

Models are evaluated using:

Mean Squared Error (MSE)

Easy to interpret but not optimal for count data

Poisson Deviance (primary metric)

Theoretically appropriate for Poisson-distributed targets

R² Score

Used only for completeness (not reliable for sparse count data)

Results Summary
Model	MSE	Poisson Deviance
Random Forest	0.0560	0.2963
Gradient Boosting	0.0561	0.2985
Decision Tree	0.0571	0.3052
Linear Regression	0.0572	0.3159
Poisson Regression	0.0578	0.3198
Linear SVR	Worst	Worst

Key observations:

Random Forest and Gradient Boosting achieve the best predictive performance.

Poisson Regression is theoretically sound and interpretable but less flexible.

Linear SVR performs poorly due to the extreme sparsity of the target variable.

Resampling Experiments

To reduce class imbalance, we oversampled policies with claims.

Poisson Regression showed an improvement in MSE

Poisson deviance increased (expected side effect)

Random Forest performance degraded due to overfitting rare events

This highlights the trade-off between bias and variance in imbalanced settings.

Limitations

Extremely imbalanced target variable

Low explanatory power of available features

Lack of behavioral or telematics data

Poisson regression limited by strict log-linear assumptions

Tree-based models less interpretable for actuarial purposes

Conclusion

This project demonstrates that insurance claim frequency modeling is inherently difficult due to rare events and weak signal.

Machine learning models provide better predictive accuracy

Poisson regression remains the reference model for interpretability and actuarial coherence

Model choice depends on business objectives: prediction vs explainability

Future Improvements

Possible extensions include:

Zero-inflated Poisson or Negative Binomial models

Poisson Gradient Boosting (e.g. XGBoost with Poisson loss)

Joint modeling of frequency and severity

Use of generalized additive models (GAMs)

Technologies Used

Python

pandas, numpy

scikit-learn

matplotlib, seaborn

Authors

This project was developed as part of an academic group assignment in data science and actuarial modeling.
