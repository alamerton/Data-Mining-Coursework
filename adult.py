import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


# Part 1: Decision Trees with Categorical Attributes

# Return a pandas dataframe with data set to be mined.
# data_file will be populated with a string 
# corresponding to a path to the adult.csv file.

def read_csv_1(data_file):
	df = pd.read_csv(data_file)
	df = df.drop(columns=['fnlwgt'])
	return df

# Return the number of rows in the pandas dataframe df.
def num_rows(df):
	return len(df)

# Return a list with the column names in the pandas dataframe df.
def column_names(df):
	return list(df.columns)

# Return the number of missing values in the pandas dataframe df.
def missing_values(df):
	return df.isnull().sum().sum()

# Return a list with the columns names containing at least one missing value in the pandas dataframe df.
def columns_with_missing_values(df):
	missing_values = df.isnull().any()
	column_list = list(df.columns[missing_values])
	return column_list	

# Return the percentage of instances corresponding to persons whose education level is 
# Bachelors or Masters (by rounding to the first decimal digit)
# in the pandas dataframe df containing the data set in the adult.csv file.
# For example, if the percentage is 21.547%, then the function should return 21.6.
def bachelors_masters_percentage(df):
	total_bachelors = df['education'].value_counts()['Bachelors']
	total_masters = df['education'].value_counts()['Masters']
	percentage = round(((total_bachelors + total_masters)/len(df) * 100), 1)
	return percentage

# Return a pandas dataframe (new copy) obtained from the pandas dataframe df 
# by removing all instances with at least one missing value.
def data_frame_without_missing_values(df):
	df.dropna() # TODO: check this is doing what I mean
	return df

# Return a pandas dataframe (new copy) from the pandas dataframe df 
# by converting the df categorical attributes to numeric using one-hot encoding.
# The function's output should not contain the target attribute.
def one_hot_encoding(df):
	# identify categorical columns (all but education num), used a fun way might not be best @TODO
	categorical_columns = list(df.loc[:, df.columns != 'education-num'].columns)
	encoded_columns_df = pd.get_dummies(df, columns=categorical_columns)
	return encoded_columns_df

# Return a pandas series (new copy), from the pandas dataframe df, 
# containing only one column with the labels of the df instances
# converted to numeric using label encoding. 
def label_encoding(df):
	class_as_series = pd.Series(df['class'])
	label_encoder = LabelEncoder()
	encoded_series = label_encoder.fit_transform(class_as_series)
	return encoded_series

# Given a training set X_train containing the input attribute values 
# and labels y_train for the training instances,
# build a decision tree and use it to predict labels for X_train. 
# Return a pandas series with the predicted values. 
def dt_predict(X_train,y_train):
	decision_tree = DecisionTreeClassifier()
	decision_tree.fit(X_train, y_train)
	prediction = decision_tree.predict(X_train)
	prediction_as_series = pd.Series(prediction)
	return prediction_as_series

# Given a pandas series y_pred with the predicted labels and a pandas series y_true with the true labels,
# compute the error rate of the classifier that produced y_pred.  
def dt_error_rate(y_pred, y_true):
	accuracy = accuracy_score(y_true, y_pred)
	error_rate = 1 - accuracy
	return error_rate

# Print statements to check outputs. TODO: remove before submitting

path = "data/adult.csv"

df = read_csv_1(path)

print(f"1. Load the data set: {df.shape}")

print(f"a) Compute number of instances: {num_rows(df)}")

print(f"b) Compute list with attribute names: \n{column_names(df)}")

print(f"c) Compute the number of missing attribute values: {missing_values(df)}")

print(f"d) Return a list with the columns names containing at least one missing value in the pandas dataframe df: {columns_with_missing_values(df)}")

print(f"e) Return the percentage of instances corresponding to persons whose education level is Bachelors or Masters: {bachelors_masters_percentage(df)}%")

no_missing_values_df = data_frame_without_missing_values(df)

print(f"2.1 Drop all instances with missing values: {df.shape}")

one_hot_df = one_hot_encoding(no_missing_values_df)

print(f"2.2 Convert all input attributes to numeric using one-hot encoding: {one_hot_df.shape}")

label_encoded_classes = label_encoding(df)

print(f"2.3 Convert the class values to numeric with label encoding: {label_encoded_classes.shape, label_encoded_classes}")

y_pred = dt_predict(one_hot_df, label_encoded_classes)

print(f"3.1 Build a decision tree: {y_pred}")

error = dt_error_rate(y_pred, label_encoded_classes)

print(f"3.2 Compute the training error rate of the resulting tree: {error}")

# TODO: I think there's something I'm missing here where I've plugged the wrong values in or something.
# Of course it always has an error rate of zero, it's predicting the same labels it was trained on.
# What am I not seeing?
