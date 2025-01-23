## Dataset

The dataset used for this project can be found on Kaggle: [Taiwan Default Credit Card Clients](https://www.kaggle.com/datasets/jishnukoliyadan/taiwan-default-credit-card-clients).

# Credit-Card-Fraud-Detection

This project applies various data mining algorithms to a credit card dataset to predict defaulters. The goal is to identify patterns and relationships within the data that can help in predicting whether a client will default on their credit card payment. The project includes the implementation of classification algorithms (Decision Tree, Naive Bayes, Logistic Regression), clustering techniques (K-Means, DBSCAN, hierarchical clustering), and association rule mining (Apriori algorithm). Each technique is demonstrated in separate Jupyter notebooks, with detailed steps for data preprocessing, model training, evaluation, and visualization of results.

## K-Nearest Neighbour Notebook

The `KNearest neighbour.ipynb` notebook demonstrates the implementation of the K-Nearest Neighbour (KNN) algorithm for classification tasks. It includes data preprocessing, splitting the dataset into training and testing sets, and applying the KNN algorithm to predict the class labels. The notebook also evaluates the model's performance using accuracy metrics and provides insights into the classification results.

## Clustering Notebook

The `clustering.ipynb` notebook demonstrates various clustering techniques using popular machine learning libraries. It includes data preprocessing, visualization, and the application of clustering algorithms such as K-Means, DBSCAN, and hierarchical clustering. The notebook also evaluates the performance of these algorithms using metrics like silhouette score and provides visual insights into the clustering results.

## Association Rule Mining Notebook

The `Association rule mining.ipynb` notebook demonstrates the process of discovering interesting relationships between variables in large datasets using association rule mining techniques. It includes data preprocessing, applying the Apriori algorithm to find frequent itemsets, and generating association rules. The notebook also evaluates the rules based on metrics like support, confidence, and lift, and provides insights into the discovered associations.

## Classification Notebook

The `Classification.ipynb` notebook demonstrates the implementation of various classification algorithms to predict credit card defaulters. It includes data preprocessing, feature selection, and the application of algorithms such as Decision Tree, Naive Bayes, and Logistic Regression. The notebook also evaluates the performance of these models using metrics like accuracy, precision, recall, and F1-score, and provides insights into the classification results.