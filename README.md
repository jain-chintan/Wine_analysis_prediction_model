Wine Quality Prediction

This project aims to predict the quality of red wine based on various physicochemical tests using machine learning models. The wine dataset includes several features such as acidity, residual sugar, alcohol content, and more, which are used to determine the quality of the wine. Multiple classification models are built, compared, and evaluated based on their performance.

Dataset

The dataset used in this project is the Wine Quality Dataset, specifically for red wine, which can be found in file("winequality-red.csv"). It consists of the following attributes:

1. fixed acidity
2. volatile acidity
3. citric acid
4. residual sugar
5. chlorides
6. free sulfur dioxide
7. total sulfur dioxide
8. density
9. pH
10, sulphates
11. alcohol
12. quality (target variable, ranges from 0 to 10)

A new binary column goodquality is added, where:

- 1 indicates wines with a quality rating of 7 or more (good quality).
- 0 indicates wines with a quality rating less than 7 (not good quality).

Requirements

To run this project, you will need the following Python libraries:

    pip install numpy pandas seaborn matplotlib scikit-learn xgboost
Code Overview
1. Importing Libraries

We import necessary libraries like numpy, pandas, seaborn, and machine learning libraries from scikit-learn.

2. Loading and Preprocessing the Dataset

The dataset is loaded using pandas. Basic exploratory analysis is done including:

- Displaying random samples of the dataset.
- Checking for null values.
- Calculating statistical summaries using describe().
- Adding the goodquality column to make it a binary classification problem.

      wine = pd.read_csv("winequality-red.csv")
      wine['goodquality'] = [1 if x >= 7 else 0 for x in wine['quality']]

3. Exploratory Data Analysis (EDA)

We visualize the dataset with various plots:

- Count Plot: To show the distribution of the quality column.
- Box Plots: For each attribute to check for outliers.
- Histograms: To observe the distribution of different features.

      sns.countplot(wine['quality'])
      wine.hist(figsize=(10,10), bins=50)

4. Feature Selection

An ExtraTreesClassifier is used to determine the importance of each feature in predicting the wine quality.

    from sklearn.ensemble import ExtraTreesClassifier
    classifiern = ExtraTreesClassifier()
    classifiern.fit(X, Y)
    print(classifiern.feature_importances_)

5. Model Training and Evaluation

Several machine learning models are trained and evaluated:

- Logistic Regression
- K-Nearest Neighbors (KNN)
- Support Vector Classifier (SVC)
- Decision Tree Classifier
- Gaussian Naive Bayes
- Random Forest Classifier
- XGBoost Classifier

Each model is trained using the train_test_split() method with 30% of the data as the test set. The accuracy of each model is calculated and stored in a dataframe for comparison.

    from sklearn.metrics import accuracy_score
    model_res = pd.DataFrame(columns=['Model', 'Score'])
    model_res.loc[len(model_res)] = ['LogisticRegression', accuracy_score(Y_test, y_pred)]

6. Results Comparison

After training, all models' performances are compared based on accuracy, and the results are sorted to identify the best-performing model.

    model_res = model_res.sort_values(by='Score', ascending=False)
    print(model_res)

Models and Their Performance

Here’s a brief summary of the models used and their accuracy:

| Model	| Accuracy |
|-------|----------|
Logistic Regression	| XX%
K-Nearest Neighbors	| XX%
Support Vector Classifier	| XX%
Decision Tree Classifier	| XX%
Gaussian Naive Bayes	| XX%
Random Forest Classifier	| XX%
XGBoost Classifier	| XX%

The model with the highest accuracy is chosen as the best-performing model.

How to Use

Clone the repository:

    git clone https://github.com/yourusername/wine-quality-prediction.git

Navigate to the project directory:

    cd wine-quality-prediction

Install dependencies:

    pip install -r requirements.txt

Run the Python script:

    python wine_quality_prediction.py

License

This project is licensed under the MIT License.
