The main goal of this project is to build a Linear Regression model to predict apartment rental prices in 9 European cities as part of the author's learning process and journey to becoming a data scientist.

The dataset used in this project was downloaded from Kaggle (https://www.kaggle.com/datasets/dipeshkhemani/airbnb-cleaned-europe-dataset). Unfortunately, the source lacks crucial information about the dataset, such as the exact time of data collection (we only know it was last updated two years ago) and the method used to obtain it. It appears to be a modified version of another Kaggle project originally created via web scraping. Additionally, some metrics in the dataset are poorly described, requiring interpretation based on their names and common sense.

The author began the project with basic Exploratory Data Analysis (EDA) to familiarize themselves with the downloaded dataset, understand the included metrics, and identify (and, if necessary, address) any issues that might hinder the use of the dataset for Linear Regression. The analysis revealed that the dataset was very clean and contained no null values, providing a green light for proceeding with Linear Regression.

The author thoroughly examined all dataset metrics and delved into the relationships between the dependent feature, Price, and the remaining independent features. Additionally, a detailed analysis of the dependent feature's distribution led to the decision to exclude certain outlier observations with extremely high prices from further investigation.

During the project, the author implemented several feature engineering techniques and created functions designed to streamline the process. These functions are not only useful for this project but are also likely to benefit future projects. The functions include:

- Predicting the most promising individual features from the set of independent features.
- Automatically building a Linear Regression model by iteratively selecting the next best independent feature to add, resulting in an optimized model.
- Enhancing pd.cut() with a function that automatically generates bin labels based on the provided numeric bins.
