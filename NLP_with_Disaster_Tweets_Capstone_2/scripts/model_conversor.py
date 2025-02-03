import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.export import ExportArchive

model = tf.keras.models.load_model("./models/model_base.h5")

tf.saved_model.save(model, './scripts/disaster_tweets_model')

#saved_model_cli show --dir ./scripts/disaster_tweets_model --all