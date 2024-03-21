import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Part 2: Cluster Analysis

# Return a pandas dataframe containing the data set that needs to be extracted from the data_file.
# data_file will be populated with the string 'wholesale_customers.csv'.


def read_csv_2(data_file):
    df = pd.read_csv(data_file)
    df = df.drop(columns=['Channel', 'Region'])
    return df

# Return a pandas dataframe with summary statistics of the data.
# Namely, 'mean', 'std' (standard deviation), 'min', and 'max' for each attribute.
# These strings index the new dataframe columns.
# Each row should correspond to an attribute in the original data and be indexed with the attribute name.


def summary_statistics(df):
    summary = df.describe().drop(['count', '25%', '50%', '75%']).T
    return summary

# Given a dataframe df with numeric values, return a dataframe (new copy)
# where each attribute value is subtracted by the mean and then divided by the
# standard deviation for that attribute.


def standardize(df):
    # scaler = StandardScaler()
    # standardised_df = scaler.fit_transform(df)
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
    # TODO: consider affinity and linkage
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
    # set k
    k = 3

    # check dataframe is standardised. If not, run standardise function on it
    standardised_input = standardize(df)
    if not standardised_input.equals(df):
        df = standardised_input

    df_array = df.to_numpy()

    # Run K-means clustering
    k = 3
    kmeans = KMeans(n_clusters=k).fit(df_array)
    labels = kmeans.fit_predict(df_array)

    # Generate scatter plots for each pair of attributes
    n_features = df_array.shape[1]
    for i in range(n_features):
        for j in range(i+1, n_features):
            plt.figure(figsize=(10, 10))
            plt.scatter(df_array[:, i], df_array[:, j],
                        c=labels, cmap='viridis')
            plt.xlabel(f'Feature {i+1}')
            plt.ylabel(f'Feature {j+1}')
            plt.title(f'Scatter plot of Feature {i+1} vs Feature {j+1}')
            plt.colorbar()
            plt.savefig(f"Figure_{j}")
            plt.show()

    # fig, axs = plt.subplots(5, 5, figsize=(15, 15)) # Adjusted to 5x5 for 25 plots
    # axs = axs.ravel() # Flatten the array of axes

    # n_features = df_array.shape[1]
    # plot_index = 0
    # for i in range(n_features):
    #     for j in range(i+1, n_features):
    #         if plot_index < len(axs):
    #             axs[plot_index].scatter(df_array[:, i], df_array[:, j], c=labels, cmap='viridis')
    #             axs[plot_index].set_xlabel(f'Feature {i+1}')
    #             axs[plot_index].set_ylabel(f'Feature {j+1}')
    #             axs[plot_index].set_title(f'Scatter plot of Feature {i+1} vs Feature {j+1}')
    #             plot_index += 1

    # for i in range(plot_index, len(axs)):
    #     fig.delaxes(axs[i])

    # plt.tight_layout()
    # plt.show()

    # Output 15 scatter plots. Generate one pdf file for each of the 15 plots, according to dletsios (https://keats.kcl.ac.uk/mod/forum/discuss.php?d=656563)
    pass

# Print statements to check outputs. TODO: remove before submitting


path = "data/wholesale_customers.csv"
df = read_csv_2(path)

# print(f"0. Data pre-processing: \n{df.shape}")

# print(f"1. Compute the mean, standard dev, minimum and maximum value for each attribute: \n{summary_statistics(df)}")

standardised_df = standardize(df)

# print(f"2.1. Return standardised dataframe: {standardised_df.shape, standardised_df}")

# k = 5

# kmeans_assignment = kmeans(df, k)

# print(f"2.2. Kmeans: \n{kmeans_assignment}")

# kmeans_pp_assignment = kmeans_plus(df, k)

# print(f"2.3. Kmeans++: \n{kmeans_pp_assignment}")

# agglomerative_assignment = agglomerative(df, k)

# print(f"2.4. Agglomerative hierarchical clustering: \n{agglomerative_assignment}")

# print(f"2.5. Silhouette score: {clustering_score(standardised_df, kmeans_assignment)}")

# eval = cluster_evaluation(standardised_df)

# # print(f"2.6. Cluster evaluation: \n{eval}")

# # print(f"2.7. Best silhouette score: {best_clustering_score(eval)}")

# not_a_dataframe = "house"

# # print(f"2.7.1 Best silhouette score: {best_clustering_score(not_a_dataframe)}")

print(f"2.8 Scatter plot: {scatter_plots(standardised_df)}")
