# Cryptocurrencies

## Overview

For this project, we are working with a company that would like to offer a new crytocurrency investment portfolio for its clients. However, due to the popularity of some of the cryptocurrencies currently available, the company wants to ensure that this portfolio would be an option for its clients. Some of the more popular cryptocurrencies are quite expensive now, but there are so many additional ones available that may be profitable for investors. Due the vast majority of these newer cryptocurrencies, it will be necessary to cluster them into groups based on trends to determine the best options to offer to clients. Since many of these cryptocurrencies are newer, it will be necessary to perform unsupervised machine learning techniques to complete the classifation. Once the clusters have been created, the visualizations will be shared with the board to make the decision on whether to offer this new investment portfolio option to clients.

## Resources

* Software: Python 3.7.6, Jupyter Notebook 7.29.0, hvPlot 0.7.2, Plotly 5.6.0
* Data Source: [cryptocurrency_data.csv](https://min-api.cryptocompare.com/data/all/coinlist) from CryptoCompare. 
* All code can be found [here](https://github.com/crtallent/Cryptocurrencies/blob/main/crypto_clustering.ipynb).

## Processing the Data

Our first step in processing the cryptocurrency data was to read the csv file into a Pandas DataFrame. To prepare the data for preprocessing, we filtered the dataset to show only the cryptocurrencies currently being traded, and removed all rows with Null values. We then filtered our results to show only the cryptocurrencies where coins had been mined:

<img src="https://github.com/crtallent/Cryptocurrencies/blob/main/Resources/Images/new_crypto_df.png" alt="Crypto_df" title="Crypto_df" />

We then created variables for the text features in our dataset, Algorithm and ProofType, and stored this data in a DataFrame, named "X". We used StandardScaler's fit_transform function to standardize the features from X DataFrame so that Principal Component Analysis (PCA) could be performed on the dataset.  

~~~
# Use get_dummies() to create variables for text features.
X = pd.get_dummies(new_crypto_df, columns=['Algorithm', 'ProofType'])

# Standardize the data with StandardScaler().
X_scaled = StandardScaler().fit_transform(X)
~~~

PCA was then performed on the X dataset to prepare the data for clustering with machine learning algorithms. Using PCA, the dimension of the data was reduced to three principal components, to ensure more efficient processing, and a new DataFrame was constructing with these components:

<img src="https://github.com/crtallent/Cryptocurrencies/blob/main/Resources/Images/PCA.png" alt="pcs_df" title="pcs_df" />

With the new pcs_df DataFrame, we created an elbow curve using hvPlot to find the best value for K (the number of our clusters in our dataset):

<img src="https://github.com/crtallent/Cryptocurrencies/blob/main/Resources/Images/Elbow.png" alt="elbow curve plot" title="Elbow Curve" />

Using the results of our elbow curve plot, we then initialized and fit our K-Means model for 4 clusters, and used a random_state of 42 to fit our data and make our predictions. With this information, we then created a new DataFrame, clustered_df with the predicted values: 

<img src="https://github.com/crtallent/Cryptocurrencies/blob/main/Resources/Images/Clustered.png" alt="clustered_df" title="clustered_df" />

The clustered_df was then used to plot our clusters on a 3D Scatter Plot with Plotly:

<img src="https://github.com/crtallent/Cryptocurrencies/blob/main/Resources/Images/3d_scatter.png" alt="3D Scatter Plot" title="3D Scatter Plot" />

We then created a table with hvPlot, and found that the total number of tradable cryptocurrencies was 534:

<img src="https://github.com/crtallent/Cryptocurrencies/blob/main/Resources/Images/hvplot_table.png" alt="hvPlot table" title="hvPlot table" />

We then scaled our TotalCoinSupply and TotalCoinsMined using MinMaxScaler to prepare our data for creating a 2D scatter plot and created our final DataFrame with the scaled data, plot_df:

~~~
X_copy = X.copy()
scaler = MinMaxScaler()
X_copy[["TotalCoinSupply", "TotalCoinsMined"]] = scaler.fit_transform(X_copy[["TotalCoinSupply", "TotalCoinsMined"]])
~~~

<img src="https://github.com/crtallent/Cryptocurrencies/blob/main/Resources/Images/plot_df.png" alt="plot_df" title="plot_df" />

We then created 2D scatter plots of our clusters using both Plotly and hvPlot:

<p float="left">
  <img src="https://github.com/crtallent/Cryptocurrencies/blob/main/Resources/Images/hvplot1.png" title="hvPlot" width="400" height="175"/>
  <img src="https://github.com/crtallent/Cryptocurrencies/blob/main/Resources/Images/hvplot.png" title="hvPlot with hover" width="400" height="175" /> 
  <img src="https://github.com/crtallent/Cryptocurrencies/blob/main/Resources/Images/plotly%20plot_df.png" title="plotly plot" width="400" height="175" />
  <img src="https://github.com/crtallent/Cryptocurrencies/blob/main/Resources/Images/plotly1.png" title="plotly plot with hover" width="400" height="175" />
</p>

## Summary

With the clusters created, we now have four classifications for the company. Hppefully, these visualizations will show the board that it is possible to predict which cryptocurrencies should be added to their clients' portfolio options. The clusters created can then be analyzed to determine the best options based on their clients' investment preferences. 
