import os
import sys
sys.path.append(os.path.dirname("/mount/src/mysolutions_mlzoomcamp2024_datatalks.club/NLP_with_Disaster_Tweets_Capstone_2/scripts/utils.py"))

import os
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pandas as pd
from utils import init_configure, preprocessing_text, preprocessing_keyword, get_tokens
import cloudpickle
import spacy

path = "/mount/src/MySolutions_MLZoomcamp2024_DataTalks.Club/NLP_with_Disaster_Tweets_Capstone_2"
#path="."

nlp = spacy.load("en_core_web_lg")
matcher, _, max_length_tweet, max_length_keyword = init_configure(nlp)

with open(path+'/models/tokenizer.bin', 'rb') as f_in:
    tokenizer, dict_words = cloudpickle.load(f_in)

def transform_text(tweet):
    tweet = tweet.fillna(" ")
    tweet["clean_text"], tweet["n_words"], tweet["n_characters"], tweet["n_hashtags"], tweet["n_mentions"], tweet["n_urls"], tweet["n_punctuations"] = zip(*tweet["text"].apply(lambda text: preprocessing_text(text, nlp, matcher, dict_words)))
    tweet["clean_keyword"] = tweet.keyword.apply(lambda text: preprocessing_keyword(text, nlp))

    tweet["tokenized_text"] = list(get_tokens(tweet["clean_text"].tolist(), tokenizer=tokenizer, max_length=max_length_tweet, fit=False, padding=True))
    tweet["tokenized_keyword"] = list(get_tokens(tweet["clean_keyword"].tolist(), tokenizer=tokenizer, max_length=max_length_keyword, fit=False, padding=True))
    
    tokenized_text = tweet["tokenized_text"].to_list()
    tokenized_keyword = tweet["tokenized_keyword"].to_list()
    context = tweet[["n_words", "n_characters", "n_hashtags", "n_mentions", "n_urls", "n_punctuations"]].to_numpy()
    
    tokenized_text = np.expand_dims(tokenized_text, axis=-1)
    tokenized_keyword = np.expand_dims(tokenized_keyword, axis=-1)
    context = np.expand_dims(context, axis=-1)
    
    return tokenized_text, tokenized_keyword, context

def process_prediction(prediction):
    pred = round(prediction)
    if pred > 0.5:
        return "Disaster tweet"
    else: 
        return "Non-disaster tweet"

st.set_page_config(layout="wide")

st.write("#### [DataTalks.Club](https://datatalks.club)'s Capstone 2 project by [Alexander D. Rios](https://linktr.ee/aletbm)")
st.write("# 🌋 Natural Language Processing with Disaster Tweets")
st.image("https://earthdaily.com/wp-content/uploads/2023/12/EarthDaily-Disaster-Banner-scaled.jpg", caption="Source: EarthDailyAnalytics")

tab1, tab2, tab3 = st.tabs(["Competition description", "Dataset analysis", "Get a prediction"])

with tab1:
    st.write("## Competition Description")
    st.image("https://i.postimg.cc/tCqNMRBp/imagen-2025-02-01-134751767.png")
    st.write("""Twitter has become an important communication channel in times of emergency.
             The ubiquitousness of smartphones enables people to announce an emergency they’re observing in real-time. Because of this, more agencies are interested in programatically monitoring Twitter (i.e. disaster relief organizations and news agencies).
             But, it’s not always clear whether a person’s words are actually announcing a disaster. Take this example:""")

    col1, col2 = st.columns(spec=[0.35, 0.65], gap="small")
    col1.image("https://storage.googleapis.com/kaggle-media/competitions/tweet_screenshot.png", width=550)
    col2.write("""
    The author explicitly uses the word “ABLAZE” but means it metaphorically. This is clear to a human right away, especially with the visual aid. But it’s less clear to a machine.

    In this competition, we’re challenged to build a machine learning model that **predicts which Tweets are about real disasters and which one’s aren’t**. We’ll have access to a dataset of **10,000 tweets** that were hand classified.

    > Disclaimer: The dataset for this competition contains text that may be considered profane, vulgar, or offensive.

    ## Acknowledgments
    This dataset was created by the company figure-eight and originally shared on their [‘Data For Everyone’ website here](https://www.figure-eight.com/data-for-everyone/).

    Tweet source: [https://twitter.com/AnyOtherAnnaK/status/629195955506708480](https://twitter.com/AnyOtherAnnaK/status/629195955506708480)

    ### What am I predicting?
    You are predicting whether a given tweet is about a real disaster or not. If so, predict a 1. If not, predict a 0.
    
    ### Kaggle competition
    
    You can find the competition in the following link: [https://www.kaggle.com/competitions/nlp-getting-started](https://www.kaggle.com/competitions/nlp-getting-started)
    
    This particular challenge is perfect for data scientists looking to get started with Natural Language Processing.
    
    ### My resource about this competition
    I would like to share you my resource about this topic:
    + My blog on Notion: [Natural Language Processing using spaCy, TensorFlow and BERT model architecture](https://volcano-camp-325.notion.site/Natural-Language-Processing-using-spaCy-TensorFlow-and-BERT-model-architecture-1895067176b380d09484d4b0338b0c5e?pvs=4) (Comming soon!)
    + My GitHub repository: [NLP with Disaster Tweets](https://github.com/aletbm/MySolutions_MLZoomcamp2024_DataTalks.Club/tree/main/NLP_with_Disaster_Tweets_Capstone_2)
    + My code on Kaggle: [🌋 NLP with Disaster Tweets](https://www.kaggle.com/code/aletbm/nlp-with-disaster-tweets/notebook)
    """)

with tab2:
    df = pd.read_csv(path+"/dataset/nlp-getting-started/train.csv", index_col="id")
    df_clean = pd.read_csv(path+"/streamlit_app/train_clean.csv", index_col="id")

    st.write(f"""## Dataset Description
The dataset for training contains {len(df)} records about tweets.

Each sample in the train and test set has the following information:

+ The `text` of a tweet
+ A `keyword` from that tweet (although this may be blank!)
+ The `location` the tweet was sent from (may also be blank)""")
    
    st.write("### The training set")
    st.dataframe(df)
    df = df.drop(df[df["keyword"].isna()].index)
    df = df.drop(["location"], axis=1)
    df = df.drop_duplicates(subset=["text"])
    labels = ["Non-Disaster Tweets", "Disaster Tweets",]
    values = [df[df['target'] == 0].shape[0], df[df['target'] == 1].shape[0]]

    st.write("### Checking the balance of the target")
    st.write(f"After cleaning the null values and removing duplicate records, the total records is {len(df)}, and we can see the target balance.")
    st.image("https://i.postimg.cc/rF8HFHr8/5f12609a226e8f50f8c895623dbd7f8083a2e545a2556aab03a752a3.png")
    st.write(f"This dataset is a imbalace dataset that contains {values[1]} disaster tweets and {values[0]} non-disaster tweets.")
    
    st.write("### Word clouds")
    st.write("#### Most frequent words in Disasters Tweets")
    st.image("https://i.postimg.cc/G2FMbP38/f2284cbda1a4b7759ed2e2f40be651a1bd0730d4f32f4c5420966f19.png")
    st.write("We can see that the most common words in disaster tweets are related to natural disasters, accidents, and fatalities, such as fire, storm, Hiroshima, and suicide bomber.")
    
    st.write("#### Most frequent words in Non-Disasters Tweets")
    st.image("https://i.postimg.cc/VNbGVQ3F/62bcc0c9b2c6676791eadf1accd2ba2d6c65a61656c1f54f9abb443c.png")
    st.write("We can see that the most common words in non-disaster tweets are related to cotidian words.")
    
    st.write("""### Working on the text of the tweets
             
#### spaCy for NLP

spaCy is a free, open-source library for advanced Natural Language Processing (NLP) in Python. It’s designed specifically for production use and helps you build applications that process and “understand” large volumes of text. It can be used to build information extraction or natural language understanding systems.

The processing pipeline consists of one or more pipeline components that are called on the Doc in order. The tokenizer runs before the components.
""")
    st.image("https://spacy.io/images/pipeline.svg", width=600)
    st.write("""#### The preprocessing pipeline that we used in this project for cleaning the texts

1. Digits to words
2. Dealing with the Twitter mentions
3. Dealing with the hashtags
4. Dealing with the emoticons
5. Extracting, replacing and removing special and numerical characters, puntuactions, mentions, URLs and hashtags
6. Lemmatizing
7. Named-Entity Recognition
8. Lowercasing
9. Stopword removal
10. Tokenization
11. Padding sequence""")
    
    st.write("### The training set after cleaning")
    st.dataframe(df_clean)
    
    st.write("### N-Grams")
    st.write("After cleaning the tweet texts, we can analyze the n-grams in the sentences.")
    st.write("#### Unigrams")
    st.image("https://i.postimg.cc/5NnRHmB8/9a33e1ba17e45e6bc93f7795559a00f142575e67269aa2983417a037.png")
    st.write("We can see the difference between the unigrams of disaster tweets and non-disaster tweets. The disaster tweets contain unigrams with negative connotations.")
    
    st.write("#### Bigrams")
    st.image("https://i.postimg.cc/MpHr1j5N/99fd1a5435d28f94a709be7130018e6feaebbabb971d7ff58fdff70e.png")
    st.write("We can see the difference between the bigrams of disaster tweets and non-disaster tweets. The disaster tweets contain bigrams with negative connotations.")
    
    st.write("#### Trigrams")
    st.image("https://i.postimg.cc/QdzYmxnt/9b962c0b53e9aa487d1647b3683e41670cb31803fc03db2088ed7eee.png")
    st.write("We can see the difference between the trigrams of disaster tweets and non-disaster tweets. The disaster tweets contain trigrams with negative connotations.")
    
    st.write("### Distribution of the context feature by tweet type")
    
    st.image("https://i.postimg.cc/tTQS0vbG/675dad3f64a5c38314f15a488d23727c85202448e662266c2510458f.png")
    
    col1, col2 = st.columns(2)
    col1.write("#### Statistical analysis of context features in non-disaster tweets.")
    col1.dataframe(df_clean.loc[df_clean["target"] == 0, ["n_words", "n_characters", "n_hashtags", "n_mentions", "n_urls", "n_punctuations"]].describe().T)
    
    col2.write("#### Statistical analysis of context features in disaster tweets.")
    col2.dataframe(df_clean.loc[df_clean["target"] == 1, ["n_words", "n_characters", "n_hashtags", "n_mentions", "n_urls", "n_punctuations"]].describe().T)
    
    st.write("In general terms, the context features have very similar statistical characteristics between the different tweet types.")
    
with tab3:
    st.warning("Attention: The model may not always make accurate predictions. I am currently working to improve it. This application is for educational purposes only.", icon="📢")
    #model = tf.keras.models.load_model("./models/model_base.h5")
    model = tf.keras.models.load_model(path+"/models/model_base.h5")
    st.write("## Load your data")
    st.write("### Complete the following information")
    
    tweet_text = st.text_area(
        "Enter your tweet here 👇",
        label_visibility="visible",
        disabled=False,
        placeholder="What is happening?!",
        max_chars=280
        )
    tweet_keyword = st.text_input(
        "Enter your keyword here 👇",
        label_visibility="visible",
        disabled=False,
        placeholder="What is your keyword?",
        max_chars=50
        )
    if st.button("Classify my tweet"):
        if tweet_text.replace(" ", "") == "" or tweet_keyword.replace(" ", "") == "":
            st.error('The fields must not be empty.', icon="🚨")
        else:
            tweet = pd.DataFrame(data={"id":[0], "keyword":[tweet_keyword], "text":[tweet_text]}).set_index("id")
            tokenized_text, tokenized_keyword, context = transform_text(tweet)
            prediction = model.predict((tokenized_text, tokenized_keyword, context))[0][0]
            if round(prediction) > 0.5:
                st.warning(f"Your tweet is a: {process_prediction(prediction)}.", icon="😬")
            else:
                st.success(f"Your tweet is a: {process_prediction(prediction)}.", icon="😀")
    
    st.write("### Or, you can upload your CSV file")
    st.write("#### Example")
    df_test = pd.read_csv(path+"/dataset/nlp-getting-started/test.csv", index_col="id")
    df_test = df_test.drop(df_test[df_test["keyword"].isna()].index).head()
    st.dataframe(df_test[["keyword", "text"]])
    st.write("""Your file must contain the following fields: `id`, `keyword` and `text`.""")
    
    st.write("#### Load your file")
    df_client = st.file_uploader("Choose your CSV file", accept_multiple_files=False, type=['csv'])
    if df_client is not None:
        col1, col2 = st.columns(2)
        df_client = pd.read_csv(df_client, index_col="id")
        col1.write("Let's take a look at your CSV file:")
        col1.dataframe(df_client)
    if st.button("Classify my tweets"):
        if df_client is not None:
            col2.write("These are your predictions:")
            tokenized_text, tokenized_keyword, context = transform_text(df_client)
            prediction = model.predict((tokenized_text, tokenized_keyword, context))
            df_pred = pd.DataFrame(data={"id":df_client.index.values, "prediction":np.squeeze(prediction)}).set_index("id")
            df_pred.prediction = df_pred.prediction.apply(process_prediction)
            col2.dataframe(df_pred)
        else:
            st.error('Please, load your file.', icon="🚨")
            
        
#streamlit run ./streamlit_app/my_app.py