# Customer Churn Prediction – Final Project Report

## 1. Problem Statement
The objective of this project is to predict customer churn using machine learning. The model identifies customers who are likely to discontinue their services, allowing the business to take proactive retention measures.

## 2. Dataset
The project uses the Telco Customer Churn dataset. It contains customer demographics, account information, service subscriptions, contract details, tenure and billing information. Additional features were engineered, including TotalServices, TenureGroup, IsMonthToMonth, AverageMonthlySpend and HasFamily.

## 3. Methodology
Three classification algorithms were evaluated:
- Logistic Regression
- Random Forest
- XGBoost

Class imbalance was addressed using appropriate class-weighting or scale_pos_weight techniques. RandomizedSearchCV with five-fold stratified cross-validation and ROC-AUC scoring was used to tune Logistic Regression and Random Forest.

Random Forest was selected based on model comparison. A business-oriented classification threshold of 0.51 was selected to target approximately 75% churn recall.

## 4. Results
Final evaluation on the held-out test dataset produced:

| Metric | Result |
|---|---:|
| ROC-AUC | 0.8421 |
| F1 Score | 0.6312 |
| Accuracy | 76.37% |
| Churn Recall | 76.20% |
| Churn Precision | 53.88% |

Confusion matrix:
- True Negatives: 791
- False Positives: 244
- False Negatives: 89
- True Positives: 285

The model successfully identified approximately 76% of actual churners.

## 5. Feature Importance
The five most important features were:

1. Contract – Month-to-month: Customers without long-term contracts may be more likely to leave.
2. IsMonthToMonth: Month-to-month contract status is an important indicator of customer retention risk.
3. Tenure: Customers with shorter service histories may be less loyal.
4. TotalCharges: Customer billing history can help indicate engagement and account lifecycle.
5. OnlineSecurity – No: Customers without online security services may have lower service engagement or different retention needs.

## 6. Limitations
The model may not capture all reasons behind customer churn. Feature importance indicates predictive contribution, not causation. False positives may lead to unnecessary retention expenses. The dataset may also not represent future customer behaviour perfectly.

## 7. Recommendations
- Identify high-risk customers early.
- Offer suitable retention incentives to customers on month-to-month contracts.
- Improve customer engagement and online security adoption.
- Monitor model performance regularly using new customer data.
- Evaluate retention campaign effectiveness before expanding interventions.

## Conclusion
The Random Forest model achieved an ROC-AUC of 0.8421 and churn recall of 76.20%. It provides a useful starting point for identifying customers at risk of leaving and supporting proactive customer retention strategies.