from collections import Counter
import numpy as np
import pandas as pd
import requests
import nltk
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score



# Part 3: Text mining.

# Return a pandas dataframe containing the data set.
# Specify a 'latin-1' encoding when reading the data.
# data_file will be populated with a string 
# corresponding to a path containing the wholesale_customers.csv file.
def read_csv_3(data_file):
	return pd.read_csv(data_file, encoding='latin-1')

# Return a list with the possible sentiments that a tweet might have.
def get_sentiments(df):
	return df['Sentiment'].unique().tolist()

# Return a string containing the second most popular sentiment among the tweets.
def second_most_popular_sentiment(df):
	sentiment_counts = df['Sentiment'].value_counts()
	return sentiment_counts[sentiment_counts == sentiment_counts.values[1]].index.tolist()[0]
	#TODO: what if there is a tie in the dataset?

# Return the date (string as it appears in the data) with the greatest number of extremely positive tweets.
def date_most_popular_tweets(df):
	extremely_positive_df = df[df['Sentiment'] == 'Extremely Positive']
	date_counts = extremely_positive_df['TweetAt'].value_counts()
	date_with_max = date_counts.idxmax()
	return date_with_max

# Modify the dataframe df by converting all tweets to lower case. 
def lower_case(df):
	df['OriginalTweet'] = df['OriginalTweet'].str.lower()
	return df

# Modify the dataframe df by replacing each characters which is not alphabetic or whitespace with a whitespace.
def remove_non_alphabetic_chars(df):
	for column in df.columns:
		if df[column].dtype == 'object':
			df[column] = df[column].str.replace(r'[^a-zA-Z\s]', ' ', regex=True)
	return df

# Modify the dataframe df with tweets after removing characters which are not alphabetic or whitespaces.
def remove_multiple_consecutive_whitespaces(df):
	df['OriginalTweet'] = df['OriginalTweet'].str.replace(r'\s+', ' ', regex=True)
	return df

# Given a dataframe where each tweet is one string with words separated by single whitespaces,
# tokenize every tweet by converting it into a list of words (strings).
def tokenize(df):
	df['OriginalTweet'] = df['OriginalTweet'].str.split()
	return df

# Given dataframe tdf with the tweets tokenized, return the number of words in all tweets including repetitions.
def count_words_with_repetitions(tdf):
	count = 0
	for row in tdf['OriginalTweet']:
		count += len(row)
	return count

# Given dataframe tdf with the tweets tokenized, return the number of distinct words in all tweets.
def count_words_without_repetitions(tdf):
	return len(tdf['OriginalTweet'].explode().unique())

# Given dataframe tdf with the tweets tokenized, return a list with the k distinct words that are most frequent in the tweets.
def frequent_words(tdf,k):
	words_df = tdf['OriginalTweet'].explode()
	return words_df.value_counts()[:k].index.tolist()

# Given dataframe tdf with the tweets tokenized, remove stop words and words with <=2 characters from each tweet.
# The function should download the list of stop words via:
# https://raw.githubusercontent.com/fozziethebeat/S-Space/master/data/english-stop-words-large.txt
def remove_stop_words(tdf):
	response = requests.get("https://raw.githubusercontent.com/fozziethebeat/S-Space/master/data/english-stop-words-large.txt")
	if response.status_code == 200:
		stop_words = response.text
	else:
		raise Exception("GET request failed")
	tdf['OriginalTweet'] = tdf['OriginalTweet'].apply(lambda tweet_text: ' '.join([word for word in tweet_text if word not in stop_words and len(word) > 2]))
	return tdf


# Given dataframe tdf with the tweets tokenized, reduce each word in every tweet to its stem.
def stemming(tdf):
	nltk.download('punkt')
	ps = PorterStemmer()
	tdf['OriginalTweet'] = tdf['OriginalTweet'].apply(lambda tweet_text: ' '.join([ps.stem(word) for word in nltk.word_tokenize(tweet_text)]))
	return tdf

# Given a pandas dataframe df with the original coronavirus_tweets.csv data set,
# build a Multinomial Naive Bayes classifier. 
# Return predicted sentiments (e.g. 'Neutral', 'Positive') for the training set
# as a 1d array (numpy.ndarray). 
def mnb_predict(df):
	# Generate train test data

	# Build and fit classifier
	documents_train, documents_test, labels_train, labels_test = train_test_split(df['OriginalTweet'], df['Sentiment'], test_size=0.2, random_state=42)

	# Convert text data into a term-document matrix
	vectorizer = CountVectorizer()
	X_train = vectorizer.fit_transform(documents_train)
	X_test = vectorizer.transform(documents_test)

	# Train the Multinomial Naive Bayes model
	nb = MultinomialNB()
	nb.fit(X_train, labels_train)

	# Make predictions on the test set
	y_pred = nb.predict(X_test)

	# Evaluate the model
	return "Accuracy:", accuracy_score(labels_test, y_pred)

# Given a 1d array (numpy.ndarray) y_pred with predicted labels (e.g. 'Neutral', 'Positive') 
# by a classifier and another 1d array y_true with the true labels, 
# return the classification accuracy rounded in the 3rd decimal digit.
def mnb_accuracy(y_pred,y_true):
	pass

# Print statements

path = 'data/coronavirus_tweets.csv'
df = read_csv_3(path)

# print(df)
# print(get_sentiments(df))
# print(second_most_popular_sentiment(df))
# print(date_most_popular_tweets(df))
# print(lower_case(df))
newDf = remove_non_alphabetic_chars(df)
# print(remove_multiple_consecutive_whitespaces(newDf))
tokenised_tweets_df = tokenize(newDf)
# print(count_words_with_repetitions(tokenised_tweets_df))
# print(count_words_without_repetitions(tokenised_tweets_df))
# print(frequent_words(tokenised_tweets_df, 10))
df_no_stop_words = remove_stop_words(tokenised_tweets_df)
# print(remove_stop_words(tokenised_tweets_df))

# print(stemming(df_no_stop_words))
print(mnb_predict(df))