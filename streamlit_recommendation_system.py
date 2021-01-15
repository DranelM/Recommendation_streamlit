import numpy as np
import pandas as pd
import streamlit as st
from scipy import sparse
from scipy.special import softmax
from sklearn.metrics.pairwise import cosine_similarity


def normalize(row):
    return (row - row.mean()) / (row.max() - row.min())


def get_similar_movie(movie_name, user_rating):
    predictions = df_cosine[movie_name] * (user_rating - 2.5)
    predictions = predictions.sort_values(ascending=False)
    return predictions


def get_similar_movies(ratings):
    if not ratings:
        return []
    predictions = pd.DataFrame()
    for name, rating in ratings:
        predictions = predictions.append(get_similar_movie(name, rating))
    predictions = predictions.sum(axis=0).sort_values(ascending=False)
    predictions = softmax(predictions)

    return predictions


df = pd.read_csv("./data/movies.csv", index_col=0)
df = df.fillna(0)

st.write("The Ratings Table")
st.dataframe(df)

# Normalize table
df = df.apply(normalize)

after_cosine = cosine_similarity(df.T)
df_cosine = pd.DataFrame(after_cosine, index=df.columns, columns=df.columns)


st.sidebar.write("Select the movies you have already seen")
left_column, right_column = st.sidebar.beta_columns(2)
action1 = ["action1", left_column.checkbox("Action 1")]
action2 = ["action2", right_column.checkbox("Action 2")]
action3 = ["action3", left_column.checkbox("Action 3")]
romantic1 = ["romantic1", right_column.checkbox("Romantic 1")]
romantic2 = ["romantic2", left_column.checkbox("Romantic 2")]
romantic3 = ["romantic3", right_column.checkbox("Romantic 3")]

choices = [action1, action2, action3, romantic1, romantic2, romantic3]

st.sidebar.beta_columns(1)
st.sidebar.write("")
st.sidebar.write("How do you rate these movies")

left_column, right_column = st.sidebar.beta_columns(2)
ratings = []
sides = [left_column, right_column]
side = -1
for movie_name, selected in choices:
    if selected:
        side = (side + 1) % 2
        ratings.append([movie_name, sides[side].slider(movie_name, 0, 5)])

predictions = pd.DataFrame(get_similar_movies(ratings), columns=["Predictions"])
st.write("Predicted Suggestions")
st.dataframe(predictions)
