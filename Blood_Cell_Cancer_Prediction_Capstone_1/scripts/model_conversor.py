import tensorflow as tf
from tensorflow.keras.export import ExportArchive

model = tf.keras.models.load_model("models\model_seg_clf.keras")

tf.saved_model.save(model, './scripts/blood-cell-model')

export_archive = ExportArchive()
export_archive.track(model)
export_archive.add_endpoint(
    name="serve",
    fn=model.call,
    input_signature=[tf.TensorSpec(shape=(None, 192, 256, 3), dtype=tf.float32)],
)
export_archive.write_out("./scripts/blood-cell-model")