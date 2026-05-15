# Diabetes Prediction Project

## Project Overview
This project focuses on building and evaluating machine learning models to predict the onset of diabetes based on diagnostic measurements. The dataset used contains several medical predictor variables and one target variable, `Outcome`, indicating whether the patient has diabetes (1) or not (0).

## Dataset
The dataset `diabetes.csv` includes the following features:
- `Pregnancies`: Number of times pregnant
- `Glucose`: Plasma glucose concentration a 2 hours in an oral glucose tolerance test
- `BloodPressure`: Diastolic blood pressure (mm Hg)
- `SkinThickness`: Triceps skin fold thickness (mm)
- `Insulin`: 2-Hour serum insulin (mu U/ml)
- `BMI`: Body mass index (weight in kg/(height in m)^2)
- `DiabetesPedigreeFunction`: Diabetes pedigree function
- `Age`: Age in years
- `Outcome`: Class variable (0 or 1) indicating non-diabetic or diabetic

## Data Preprocessing
The following steps were performed to prepare the data for model training:

### 1. Data Imputation: Handling Zero Values
Several features like 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', and 'BMI' contained zero values that are biologically implausible and represent missing data. These zeros were imputed using appropriate strategies (mean or median) based on the distribution of each feature. Specifically:
- `Pregnancies`: Imputed with the median.
- `Glucose`, `BloodPressure`, `BMI`: Imputed with the mean.
- `SkinThickness`, `Insulin`: Imputed with the median due to skewed distributions.

### 2. Outlier Detection and Treatment
Outliers were identified using the Interquartile Range (IQR) method. For 'Insulin', even after initial imputation and scaling, persistent outliers were handled by applying a quantile-based capping method, where values above the 95th percentile were removed to mitigate their impact.

### 3. Feature Scaling
`StandardScaler` was used to transform features to have a mean of 0 and a standard deviation of 1. This step is crucial for many machine learning algorithms that are sensitive to the scale of input features.

### 4. Handling Data Imbalance with SMOTE
The target variable (`Outcome`) exhibited class imbalance. The Synthetic Minority Over-sampling Technique (SMOTE) was applied to the training data to generate synthetic samples for the minority class (diabetic), thereby balancing the class distribution and preventing model bias.

## Model Training and Evaluation
The processed data was split into training (67%) and testing (33%) sets. The following classification models were trained and evaluated:

### 1. Logistic Regression
- **Description**: A linear model used for binary classification, estimating the probability of an instance belonging to a particular class.
- **Performance**: Achieved an accuracy of approximately 73.03% on the test set. The `classification_report` indicated a recall of 69% for the 'Diabetic' class.

### 2. K-Nearest Neighbors (KNN)
- **Description**: A non-parametric, instance-based learning algorithm that classifies data points based on the majority class of its 'k' nearest neighbors.
- **Performance**: Achieved an accuracy of approximately 71.0% on the test set. The `classification_report` indicated a recall of 71% for the 'Diabetic' class.

### 3. Gaussian Naive Bayes
- **Description**: A probabilistic classifier based on Bayes' theorem, assuming features are conditionally independent and follow a Gaussian distribution.
- **Performance**: Achieved an accuracy of approximately 72.20% on the test set. The `confusion_matrix` and `classification_report` provide detailed metrics.

## Key Takeaways
- Data imputation and outlier treatment are critical steps to handle biologically implausible zero values and extreme observations.
- Feature scaling ensures that no single feature dominates the learning process due to its scale.
- Addressing class imbalance with techniques like SMOTE is essential for developing robust models, especially when the cost of misclassifying the minority class (e.g., diabetics) is high.
- For healthcare applications, recall is often a crucial metric, as minimizing false negatives (missing actual diabetic cases) is of high importance.

## How to Run
1. Ensure you have the required libraries installed (`pandas`, `numpy`, `seaborn`, `matplotlib`, `scikit-learn`, `imblearn`).
2. Download the `diabetes.csv` dataset and place it in the project directory (or adjust the path in the code).
3. Run the Jupyter Notebook cells sequentially to reproduce the data preprocessing, model training, and evaluation steps.
