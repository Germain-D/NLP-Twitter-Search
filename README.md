# Twitter Search with Word Embeddings

## Description

This Python project demonstrates how to perform a search on Twitter tweets using word embeddings. Given a search query, the program retrieves the top 5 most relevant tweets based on two distance algorithms: **cosine similarity** and **Euclidean distance**. This results in a total of 10 tweets being returned.

The project involves the following steps:

1. **Pre-processing of tweets**: Cleaning and tokenizing the tweets to handle typos, non-conventional characters, and other noise.
2. **Word Embedding**: Using the GloVe model to convert tweets into their vector representations by averaging the word embeddings of all words in a tweet.
3. **Query Embedding**: Converting the search query into its vector representation using the same GloVe model.
4. **Distance Calculation**: Calculating the distance between the query vector and all tweet vectors using cosine similarity and Euclidean distance.
5. **Result Retrieval**: Sorting the tweets based on the calculated distances and returning the top 5 tweets for each distance algorithm.

## Requirements

- Python 3.x
- Libraries: `tabulate`, `gensim`, `sklearn`, `contractions`, `pandas`, `nltk`

Install the required libraries using the following commands:

```bash
pip install tabulate gensim scikit-learn contractions pandas nltk
```

## Dataset

The dataset used in this project is provided in a `tweets.csv` file. The file contains the tweets to be searched, with each tweet stored in the `text` column.

## GloVe Model

The program uses the GloVe model for word embeddings. Specifically, the `glove.twitter.27B.200d.txt` pre-trained model is used. You can download the model from [here](https://nlp.stanford.edu/projects/glove/).

## How to Use

1. **Pre-process the Tweets**: The `clean` function pre-processes the tweets by removing URLs, handles, non-letter characters, and more. It also converts contractions to their longer forms and tokenizes the tweets.

2. **Vectorize Tweets and Query**: The `vectorize` function converts tokens into their vector representations using the GloVe model.

3. **Calculate Distances**: The program calculates the cosine similarity and Euclidean distance between the query vector and the tweet vectors.

4. **Retrieve Top Tweets**: The top 5 most relevant tweets for each distance algorithm are retrieved and displayed using the `tabulate` library.

## Usage Example

```python
# Enter the search query
sentence = "i want to go to the moon"

# Preprocess the query
phrase = pd.DataFrame(data={'text': [sentence]})
phrase['tokens'] = phrase['text'].apply(tknzr.tokenize)
phrase['vectorized'] = phrase['tokens'].apply(vectorize)

# Calculate cosine similarity and retrieve top 5 tweets
cosi = []
for i in data_clean.index:
    cosi.append(float(cosine_similarity(phrase['vectorized'][0].reshape(1, -1), data_clean['vectorized'][i].reshape(1, -1))))
data_clean['cosine'] = cosi
print(tabulate(data_clean[['text', 'cosine']].nlargest(5, ['cosine']), headers='keys', tablefmt='psql'))

# Calculate Euclidean distance and retrieve top 5 tweets
eucl = []
for i in data_clean.index:
    eucl.append(float(euclidean_distances(phrase['vectorized'][0].reshape(1, -1), data_clean['vectorized'][i].reshape(1, -1))))
data_clean['euclidian'] = eucl
print(tabulate(data_clean[['text', 'euclidian']].nsmallest(5, ['euclidian']), headers='keys', tablefmt='psql'))
```

## Results

The program will output the top 5 most relevant tweets based on both cosine similarity and Euclidean distance. The results are displayed in a tabular format for easy comparison.

---

This project provides a basic framework for searching tweets using word embeddings. It can be extended by using different pre-trained models, incorporating more advanced pre-processing techniques, or applying additional distance algorithms.