import numpy as np
import pandas as pd
import os
import random
import warnings
pd.options.mode.copy_on_write = True
warnings.simplefilter(action='ignore', category=FutureWarning)

import tensorflow as tf
from tensorflow.data import Dataset
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Embedding, Dropout, Dense, Bidirectional, LSTM, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, CSVLogger, ModelCheckpoint
from tensorflow.keras.metrics import BinaryAccuracy

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import spacy
import pandas as pd
import string
import cloudpickle

from utils import solve, init_configure, preprocessing_text, preprocessing_keyword, get_tokens
from load_data import download_dataset

def get_baseline_model():
    tf.keras.backend.clear_session()
    
    inp1 = Input(shape=(max_length_tweet,), name="text")
    x = Embedding(input_dim=vocab_size, output_dim=16, input_length=max_length_tweet)(inp1)
    x = Bidirectional(LSTM(4, dropout=0.4, recurrent_dropout=0.0))(x)
    out1 = Dense(1, activation="sigmoid")(x)
    
    inp2 = Input(shape=(max_length_keyword,), name="keyword")
    t = Embedding(input_dim=vocab_size, output_dim=16, input_length=max_length_tweet)(inp2)
    t = Bidirectional(LSTM(4, dropout=0.4, recurrent_dropout=0.0))(t)
    out2 = Dense(1, activation="sigmoid")(t)
    
    inp3 = Input(shape=(6,), name="context")
    z = Dense(32, activation="relu")(inp3)
    z = Dropout(0.4)(z)
    z = Dense(16, activation="relu")(z)
    out3 = Dense(1, activation="sigmoid")(z)
    
    k = Concatenate()([out1, out2, out3])
    out = Dense(1, activation="sigmoid")(k)
    
    model = Model(inputs=[inp1, inp2, inp3], outputs=out)
    return model

def train_model(model, train_data, val_data, epochs=100, version="base"):
    checkpoint_filepath = f'./kaggle/working/models/model_{version}.h5'
    checkpoint = ModelCheckpoint(checkpoint_filepath, monitor='val_loss', save_best_only=True)
    reduce = ReduceLROnPlateau(monitor='val_loss', patience=6, verbose=1)
    early = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)
    csvlogger = CSVLogger(f"./kaggle/working/histories/history_model_{version}.csv", separator=',')
    
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=[BinaryAccuracy()])
    history = model.fit(train_data,
                        validation_data=val_data,
                        epochs=epochs,
                        callbacks=[reduce, early, checkpoint, csvlogger],
                        class_weight=class_weight
                       )
    return history

download_dataset()

seed = 42
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
np.random.seed(seed)

nlp = spacy.load("en_core_web_lg")
matcher, tokenizer, max_length_tweet, max_length_keyword = init_configure(nlp)

with open('./kaggle/input/479k-english-words/english_words_479k.txt', 'r') as txt:
    words = txt.read().split('\n')
    words = [x.lower() for x in words]
    
words2 = pd.read_csv('./kaggle/input/english-word-frequency/unigram_freq.csv')['word'].to_list()

all_words = words + [str(x).lower() for x in words2]
all_words = sorted(list(set(words)))

all_words = [x for x in all_words if (len(x) > 2
            and x.isalpha()
            and len(x.replace("a", "").replace("e", "").replace("i", "").replace("o", "").replace("u", "")) > 0
            and len(solve(x)) > 2
            and x not in nlp.Defaults.stop_words)]

dict_words = {}
for letter1 in string.ascii_lowercase:
    dict_words[letter1] = {}
    for letter2 in string.ascii_lowercase:
        dict_words[letter1][letter2] = [x for x in all_words if x[0:2] == letter1+letter2]

if __name__ == "__main__":
    df = pd.read_csv("./kaggle/input/nlp-getting-started/train.csv", index_col="id")
    df_cl_nl = df.drop(df[df["keyword"].isna()].index)
    df_cl_nl = df_cl_nl.drop(["location"], axis=1)
    df_cl = df_cl_nl.drop_duplicates(subset=["text"])

    df_cl["clean_text"], df_cl["n_words"], df_cl["n_characters"], df_cl["n_hashtags"], df_cl["n_mentions"], df_cl["n_urls"], df_cl["n_punctuations"] = zip(*df_cl["text"].apply(lambda text: preprocessing_text(text, nlp, matcher, dict_words)))
    df_cl["clean_keyword"] = df_cl.keyword.apply(lambda text: preprocessing_keyword(text, nlp))

    df_cl["tokenized_text"] = list(get_tokens(text=df_cl["clean_text"].tolist(), tokenizer=tokenizer, max_length=max_length_tweet, fit=True, padding=True))
    df_cl["tokenized_keyword"] = list(get_tokens(text=df_cl["clean_keyword"].tolist(), tokenizer=tokenizer, max_length=max_length_keyword, fit=True, padding=True))

    with open('./models/tokenizer.bin', 'wb') as f_out:
        cloudpickle.dump((tokenizer, dict_words), f_out)
    
    X = df_cl.drop(["target"], axis=1)
    y = df_cl["target"]

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, stratify=y, shuffle=True, random_state=seed)
    X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=0.50, stratify=y_val, shuffle=True, random_state=seed)

    class_weight = compute_class_weight(class_weight="balanced", classes=np.unique(y_train), y=y_train)
    class_weight = dict(zip(np.unique(y_train), class_weight))

    vocab_size = len(tokenizer.word_index) + 1

    X_train_tokenized = X_train.loc[:, ["tokenized_text", "tokenized_keyword", "n_words", "n_characters", "n_hashtags", "n_mentions", "n_urls", "n_punctuations"]]
    X_val_tokenized = X_val.loc[:, ["tokenized_text", "tokenized_keyword", "n_words", "n_characters", "n_hashtags", "n_mentions", "n_urls", "n_punctuations"]]
    X_test_tokenized = X_test.loc[:, ["tokenized_text", "tokenized_keyword", "n_words", "n_characters", "n_hashtags", "n_mentions", "n_urls", "n_punctuations"]]

    train_text, train_keyword, train_context = X_train_tokenized["tokenized_text"].to_list(), X_train_tokenized["tokenized_keyword"].to_list(), X_train_tokenized.drop(["tokenized_text", "tokenized_keyword"], axis=1)
    train_data_text = Dataset.from_tensor_slices(train_text)
    train_data_keyword = Dataset.from_tensor_slices(train_keyword)
    train_data_context = Dataset.from_tensor_slices(train_context)
    train_labels = Dataset.from_tensor_slices(np.expand_dims(y_train.values, axis=-1))

    val_text, val_keyword, val_context = X_val_tokenized["tokenized_text"].to_list(), X_val_tokenized["tokenized_keyword"].to_list(), X_val_tokenized.drop(["tokenized_text", "tokenized_keyword"], axis=1)
    val_data_text = Dataset.from_tensor_slices(val_text)
    val_data_keyword = Dataset.from_tensor_slices(val_keyword)
    val_data_context = Dataset.from_tensor_slices(val_context)
    val_labels = Dataset.from_tensor_slices(np.expand_dims(y_val.values, axis=-1))

    test_text, test_keyword, test_context = X_test_tokenized["tokenized_text"].to_list(), X_test_tokenized["tokenized_keyword"].to_list(), X_test_tokenized.drop(["tokenized_text", "tokenized_keyword"], axis=1)
    test_data_text = Dataset.from_tensor_slices(test_text)
    test_data_keyword = Dataset.from_tensor_slices(test_keyword)
    test_data_context = Dataset.from_tensor_slices(test_context)
    test_labels = Dataset.from_tensor_slices(np.expand_dims(y_test.values, axis=-1))

    train_data_tokenized = Dataset.zip(((train_data_text, train_data_keyword, train_data_context), train_labels))
    val_data_tokenized = Dataset.zip(((val_data_text, val_data_keyword, val_data_context), val_labels))
    test_data_tokenized = Dataset.zip(((test_data_text, test_data_keyword, test_data_context), test_labels))

    AUTOTUNE = tf.data.AUTOTUNE
    batch_size = 32

    train_data_tokenized = train_data_tokenized.batch(batch_size).prefetch(buffer_size=AUTOTUNE)
    val_data_tokenized = val_data_tokenized.batch(batch_size).prefetch(buffer_size=AUTOTUNE)
    test_data_tokenized = test_data_tokenized.batch(batch_size).prefetch(buffer_size=AUTOTUNE)

    model_base = get_baseline_model()

    history_base = train_model(model=model_base,
                            train_data=train_data_tokenized,
                            val_data=val_data_tokenized,
                            epochs=200,
                            version="base"
                            )

    model_base.evaluate(test_data_tokenized)