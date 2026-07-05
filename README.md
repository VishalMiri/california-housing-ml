# California Housing Price Predictor (End-to-End ML Pipeline)

An end-to-end regression machine learning pipeline built using Scikit-Learn to predict the median housing price in California districts. This project follows the structured workflow outlined in *Hands-On Machine Learning*.

## 🚀 Project Overview
* **Objective:** Predict `median_house_value` using geographical, demographic, and real estate attributes.
* **Type:** Supervised Learning, Batch Learning, Univariate Regression.
* **Key Performance Metric:** Root Mean Squared Error (RMSE).

## 🛠️ Tech Stack & Workflow
* **Environment Management:** Miniconda & VS Code (`Python 3.12`)
* **Libraries:** Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn
* **Data Engineering:** Automated fetching scripts, stratified sampling based on income tiers, custom transformers for ratio feature additions (`rooms_per_household`), and parallelized `ColumnTransformer` preprocessing pipelines.

## 📊 Evaluation Results
| Model | Base Training RMSE | 10-Fold CV Mean RMSE | Status |
| :--- | :--- | :--- | :--- |
| **Linear Regression** | $68,628 | $69,050 | Underfitting |
| **Decision Tree** | $0 | $71,400 | Extreme Overfitting |
| **Random Forest (Base)** | $18,600 | $50,100 | Promising (Slight Overfitting) |
| **Random Forest (Fine-Tuned)**| — | **$49,680** | **Best Model** |

* **Final Performance on Test Set:** **$47,730** (95% Confidence Interval: $45,600 to $49,800).

## 💡 Key Learnings
* Developed robust, reusable data pipelines preventing data leakage.
* Implemented **Stratified Shuffle Split** to eliminate sampling bias.
* Utilized **GridSearchCV** and **RandomizedSearchCV** to automate hyperparameter tuning.
* Discovered that `median_income` is the single highest predictor of housing prices via feature importance analysis.
