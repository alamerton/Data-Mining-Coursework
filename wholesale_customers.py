import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Part 2: Cluster Analysis

# Return a pandas dataframe containing the data set that needs to be extracted from the data_file.
# data_file will be populated with the string 'wholesale_customers.csv'.


def read_csv_2(data_file):
    return pd.read_csv(data_file).drop(columns=['Channel', 'Region'])

# Return a pandas dataframe with summary statistics of the data.
# Namely, 'mean', 'std' (standard deviation), 'min', and 'max' for each attribute.
# These strings index the new dataframe columns.
# Each row should correspond to an attribute in the original data and be indexed with the attribute name.


def summary_statistics(df):
    return df.describe().drop(['count', '25%', '50%', '75%']).T

# Given a dataframe df with numeric values, return a dataframe (new copy)
# where each attribute value is subtracted by the mean and then divided by the
# standard deviation for that attribute.


def standardize(df):
    standardised_df = (df - df.mean()) / df.std()
    return standardised_df

# Given a dataframe df and a number of clusters k, return a pandas series y
# specifying an assignment of instances to clusters, using kmeans.
# y should contain values in the set {0,1,...,k-1}.
# To see the impact of the random initialization,
# using only one set of initial centroids in the kmeans run.


def kmeans(df, k):
    kmeans = KMeans(n_clusters=k, random_state=0).fit(df)
    y = pd.Series(kmeans.labels_)
    return y

# Given a dataframe df and a number of clusters k, return a pandas series y
# specifying an assignment of instances to clusters, using kmeans++.
# y should contain values from the set {0,1,...,k-1}.


def kmeans_plus(df, k):
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=0).fit(df)
    y = pd.Series(kmeans.labels_)
    y = pd.Series(kmeans.labels_)
    # TODO: could be predict here, not labels_
    return y

# Given a dataframe df and a number of clusters k, return a pandas series y
# specifying an assignment of instances to clusters, using agglomerative hierarchical clustering.
# y should contain values from the set {0,1,...,k-1}.


def agglomerative(df, k):
    ac = AgglomerativeClustering(n_clusters=k)
    ac.fit(df)
    y = pd.Series(ac.labels_)
    return y

# Given a data set X and an assignment to clusters y
# return the Silhouette score of this set of clusters.


def clustering_score(X, y):
    return silhouette_score(X, y)

# Perform the cluster evaluation described in the coursework description.
# Given the dataframe df with the data to be clustered,
# return a pandas dataframe with an entry for each clustering algorithm execution.
# Each entry should contain the:
# 'Algorithm' name: either 'Kmeans' or 'Agglomerative',
# 'data' type: either 'Original' or 'Standardized',
# 'k': the number of clusters produced,
# 'Silhouette Score': for evaluating the resulting set of clusters.


def cluster_evaluation(df):
    k_values = [3, 5, 10]
    executions_df = pd.DataFrame(
        columns=['Algorithm', 'data', 'k', 'Silhouette Score'])

    for k in range(0, len(k_values)):
        for _ in range(0, 10):
            kmeans_data = kmeans(df, k_values[k])
            silhouette_score = clustering_score(df, kmeans_data)
            instance = pd.DataFrame({
                'Algorithm': 'Kmeans',
                'data': kmeans_data,
                'k': k_values[k],
                'Silhouette Score': silhouette_score
            })
            executions_df = pd.concat(
                [executions_df, instance], ignore_index=True)

        for _ in range(0, 10):
            agglomerative_data = agglomerative(df, k_values[k])
            silhouette_score = clustering_score(df, agglomerative_data)
            instance = pd.DataFrame({
                'Algorithm': 'Agglomerative',
                'data': agglomerative_data,
                'k': k_values[k],
                'Silhouette Score': silhouette_score
            })
            executions_df = pd.concat(
                [executions_df, instance], ignore_index=True)

    return executions_df

# Given the performance evaluation dataframe produced by the cluster_evaluation function,
# return the best computed Silhouette score.


def best_clustering_score(rdf):
    if isinstance(rdf, pd.DataFrame):
        return rdf['Silhouette Score'].max()
    else:
        raise Exception(
            "Error: input is not a dataframe, please input a DataFrame.")

# Run the Kmeans algorithm with k=3 by using the standardized data set.
# Generate a scatter plot for each pair of attributes.
# Data points in different clusters should appear with different colors.


def scatter_plots(df):
    k = 3
    standardised_input = standardize(df)
    if not standardised_input.equals(df):
        df = standardised_input

    df_array = df.to_numpy()
    kmeans = KMeans(n_clusters=k).fit(df_array)
    labels = kmeans.fit_predict(df_array)

    features = df_array.shape[1]
    for i in range(features):
        for j in range(i+1, features):
            plt.figure(figsize=(10, 10))
            plt.scatter(df_array[:, i], df_array[:, j],
                        c=labels, cmap='viridis')
            plt.xlabel(f'Feature {i+1}')
            plt.ylabel(f'Feature {j+1}')
            plt.title(f'Scatter plot of Feature {i+1} vs Feature {j+1}')
            plt.colorbar()
            plt.savefig(f"Figure_{j}.pdf")
            plt.show()
