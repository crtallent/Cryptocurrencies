# Cryptocurrencies

## Overview

For this project, we are working with a company that would like to offer a new crytocurrency investment portfolio for its clients. However, due to the popularity of some of the cryptocurrencies currently available, the company wants to ensure that this portfolio would be an option for its clients. Some of the more popular cryptocurrencies are quite expensive now, but there are so many additional ones available that may be profitable for investors. Due the vast majority of these newer cryptocurrencies, it will be necessary to cluster them into groups based on trends to determine the best options to offer to clients. Since many of these cryptocurrencies are newer, it will be necessary to perform unsupervised machine learning techniques to complete the classifation. Once the clusters have been created, the visualizations will be shared with the board to make the decision on whether to offer this new investment portfolio option to clients.

## Resources

* Software: Python 3.7.6, Jupyter Notebook 7.29.0
* Data Source: [cryptocurrency_data.csv](https://min-api.cryptocompare.com/data/all/coinlist) from CryptoCompare. 
* All code can be found [here](https://github.com/crtallent/Cryptocurrencies/blob/main/crypto_clustering.ipynb).

## Processing the Data

Our first step in processing the cryptocurrency data was to read the csv file into a Pandas DataFrame. To prepare the data for preprocessing, we filtered the dataset to show only the cryptocurrencies currently being traded, and removed all rows with Null values. We then filtered our results to show only the cryptocurrencies where coins had been mined:

<img src="https://github.com/crtallent/Cryptocurrencies/blob/main/Resources/Images/new_crypto_df.png" alt="Crypto_df" title="Crypto_df" />

We then created variables for the text features in our dataset, Algorithm and ProofType, and stored this data in a DataFrame, named "X". We used StandardScaler's fit_transform function to standardize the features from X DataFrame so that Principal Component Analysis (PCA) could be performed on the dataset.  

~~~
# Use get_dummies() to create variables for text features.
X = pd.get_dummies(new_crypto_df, columns=['Algorithm', 'ProofType'])
X
~~~
