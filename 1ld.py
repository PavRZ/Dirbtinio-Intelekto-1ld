import pandas as pd
import nltk
import string
import streamlit as st

#For text Pre-proccesing
from nltk.tokenize import word_tokenize #Word tokenization
from nltk.corpus import stopwords # Stopwords removal
from nltk.stem import WordNetLemmatizer #Lemmetization
from nltk.stem import PorterStemmer #Stemmer

#For grouping
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import linear_kernel

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')  # Download the WordNet corpus for lemmatization

stop_words = set(stopwords.words('english'))

df = pd.read_csv("test.csv")
df = df.drop(columns = ["ID","ABSTRACT"])


#Text Pre-Processing-------------------------------------------------------------------------------------
#Punctuation and symbol removal
df['NEW'] = df['TITLE'].str.replace('[{}]'.format(string.punctuation), '')

#Tokenizing words
df['NEW'] = df['NEW'].apply(lambda x: word_tokenize(x))

#Stop words removal and lowercasing
df['NEW'] = df['NEW'].apply(lambda x: [word.lower() for word in x if word.lower() not in stop_words])

#Lemmetization
lemmatizer = WordNetLemmatizer()
df["NEW"] = df["NEW"].apply(lambda x: [lemmatizer.lemmatize(word) for word in x])

#Stemming
stemmer = PorterStemmer()
df['NEW'] = df['NEW'].apply(lambda x: [stemmer.stem(word) for word in x])


#Grouping articles by 5 keywords--------------------------------------------------------------------
#Returning data to string values
df["NEW"] = df['NEW'].apply(' '.join)

# Create a CountVectorizer
vectorizer = CountVectorizer()

# Fit and transform the processed text
X = vectorizer.fit_transform(df['NEW'])

# Get the sum of each term
term_frequencies = X.sum(axis=0)

# Get the feature names
feature_names = vectorizer.get_feature_names_out()

# Create a DataFrame with term frequencies
term_df = pd.DataFrame({'term': feature_names, 'frequency': term_frequencies.tolist()[0]})

# Sort by frequency in descending order and get the top 5 terms
top_keywords = term_df.sort_values(by='frequency', ascending=False).head(5)['term'].tolist()

# Group 5 random articles for each top keyword
grouped_articles = pd.DataFrame(columns=['keyword', 'TITLE']) #Creates column names

for keyword in top_keywords:
    keyword_articles = df[df['NEW'].apply(lambda x: keyword in x)].sample(5)
    keyword_articles['keyword'] = keyword
    grouped_articles = pd.concat([grouped_articles, keyword_articles[['keyword', 'TITLE']]])

#Streamlit app with search engine--------------------------------------------------------------------
def search(query, X, vectorizer, df):
    query_vector = vectorizer.transform([query])
    cosine_similarities = linear_kernel(query_vector, X).flatten()
    related_articles_indices = cosine_similarities.argsort()[:-6:-1]

    results = df.loc[related_articles_indices, ['TITLE']]
    return results

def main():
    st.title("Grouping and search of articles using AI algorithms")

    # Print previously seen grouped articles
    st.subheader("Top 5 categories:")
    for keyword in top_keywords:
        keyword_articles = grouped_articles[grouped_articles['keyword'] == keyword]
        st.write(f"{keyword.capitalize()}:")
        for _, article in keyword_articles.iterrows():
            st.write(f"  - {article['TITLE']}")

#    st.table(grouped_articles)

    # Search input
    search_query = st.text_input("Enter a search query:")

    if search_query:
        results = search(search_query.lower(), X, vectorizer, df)

        if not results.empty:
            st.subheader("Search Results:")
            st.table(results)
        else:
            st.info("No articles found for the query.")

if __name__ == "__main__":
    main()st