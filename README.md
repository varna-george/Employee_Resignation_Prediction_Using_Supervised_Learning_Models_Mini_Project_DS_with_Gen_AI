Data Science with Gen AI - Employee Resignation Prediction Using Supervised Learning Models - Mini Project

Project Title:
Employee Resignation Prediction Using Supervised Learning Models

Objective:
To develop the most accurate employee resignation prediction model using multiple supervised machine learning algorithms and evaluate their performance using classification metrics.

Dataset Used:
Source: Employee Performance and Productivity Data (Kaggle)
Link: https://www.kaggle.com/datasets/mexwell/employee-performance-and-productivity-data

Models Used:
In this project, I applied the following supervised machine learning classification models to predict employee resignation:

1. Logistic Regression
Logistic Regression was used as a simple and interpretable baseline model. It helps to understand how each feature linearly affects the probability of an employee resigning or staying, and provides easy-to-read coefficients.

2. Decision Tree Classifier
The Decision Tree model was chosen because it can capture non-linear relationships and interactions between features. It is also easy to visualize and interpret, which is useful for explaining the decision process to HR stakeholders.

3. Random Forest Classifier
Random Forest, an ensemble of multiple decision trees, was used as a more powerful model to improve predictive performance and reduce overfitting. It also provides feature importance scores, helping to identify which factors contribute most to employee resignation. This model gave one of the best overall performances.

4. K-Nearest Neighbors (KNN)
KNN was included as a distance-based model that classifies an employee based on the behavior of similar employees in the feature space. It is simple to understand and serves as a good contrast to tree-based and linear models.

5. Gradient Boosting Classifier
Gradient Boosting was used as another ensemble method that builds trees sequentially, where each new tree tries to correct the errors of the previous ones. It is effective for capturing complex patterns and often performs well on structured tabular data like this HR dataset.

Additionally, SMOTE (Synthetic Minority Over-sampling Technique) was applied on the training data to handle class imbalance between “Resigned” and “Not Resigned”, ensuring that the models learn better patterns for the minority class.


Interpretation of Results (Conclusion)
Logistic Regression - Accuracy: 0.68205
Decision Tree - Accuracy: 0.77345
Random Forest - Accuracy: 0.87775
KNN - Accuracy: 0.68525
Gradient Boosting - Accuracy: 0.81655
Tuned Random Forest - Accuracy: 0.8787
Random Forest was selected as one of the main models for employee resignation prediction. After applying RandomizedSearchCV for hyperparameter tuning, the overall accuracy improved slightly from 87.78% to 87.87%. The confusion matrix shows a small reduction in false positives (from 500 to 477) and a slight improvement in the F1-score for the majority “Not Resigned” class.
However, the performance for the minority “Resigned” class did not significantly improve. The recall for this class remained very low (around 3%), meaning the model is still unable to correctly identify most employees who actually resign. This highlights the challenge of working with an imbalanced dataset: even after tuning, the model tends to favor the majority class to maintain high overall accuracy.
Therefore, while hyperparameter tuning marginally improved overall metrics, it did not solve the main business problem of reliably detecting employees at risk of resignation. In a real-world scenario, additional techniques (such as different resampling strategies, class weights, or alternative algorithms) would be needed to better handle the minority class.
