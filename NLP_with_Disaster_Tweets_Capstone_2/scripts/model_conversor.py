import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.export import ExportArchive

model = tf.keras.models.load_model("./models/model_base.h5")

tf.saved_model.save(model, './scripts/disaster_tweets_model')

export_archive = ExportArchive()
export_archive.track(model)
export_archive.add_endpoint(
    name="serve",
    fn=model.call,
    input_signature=[[tf.TensorSpec(shape=(None, 105), dtype=tf.int64), tf.TensorSpec(shape=(None, 6), dtype=tf.int64), tf.TensorSpec(shape=(None, 6), dtype=tf.int32)]],
)
export_archive.write_out("./scripts/disaster_tweets_model")