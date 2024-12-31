import kagglehub
import numpy as np
import pandas as pd
import random
import os
import shutil
import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import (Conv2D, 
                                     Dense,
                                     Dropout,
                                     Flatten,
                                     MaxPooling2D,
                                     BatchNormalization,
                                     ReLU,
                                     Conv2DTranspose,
                                     Concatenate)
from tensorflow.image import resize
from tensorflow.keras.applications import VGG16
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import L2
from tensorflow.keras.losses import CategoricalCrossentropy, Dice
from tensorflow.keras.metrics import AUC, BinaryIoU, BinaryAccuracy
from tensorflow.keras.optimizers import Adam

from sklearn.model_selection import train_test_split

import cv2

def download_dataset():
    blood_cell_cancer_all_4class_path = kagglehub.dataset_download('mohammadamireshraghi/blood-cell-cancer-all-4class', force_download=True)
    
    for dir in ["./scripts/kaggle", "./scripts/kaggle/input", "./scripts/kaggle/working", "./scripts/kaggle/working/models", "./scripts/kaggle/working/histories"]:
        if not os.path.exists(dir):
            os.mkdir(dir)

    target_dir = './scripts/kaggle/input'
        
    file_names = os.listdir(blood_cell_cancer_all_4class_path)
        
    for file_name in file_names:
        shutil.move(os.path.join(blood_cell_cancer_all_4class_path, file_name), target_dir)
        
    print("The dataset is ready!")
    return

def conv2d(filters):
    return Conv2D(filters, kernel_size=(2, 2), padding='same', strides=(1, 1), data_format="channels_last", kernel_initializer='he_normal', kernel_regularizer=L2(l2=1e-5))

def block_conv(inp, filters=128, dropout=0.4):
    x = conv2d(filters)(inp)
    x = ReLU()(x)
    x = BatchNormalization()(x)
    
    x = conv2d(filters//2)(x)
    x = ReLU()(x)
    x = BatchNormalization()(x)
    
    x = MaxPooling2D(pool_size=(2, 2))(x)
    return Dropout(dropout)(x)

def get_backbone():
    backbone = VGG16(include_top=False, weights='imagenet', input_shape=img_shape+(3,), pooling="avg")
    backbone.trainable = False
    return backbone

def classification_w_backbone_architecture(inp, name_output=None):
    backbone = get_backbone()
    x = backbone(inp, training=False)
    x = Flatten()(x)
    x = Dense(32, activation="relu")(x)
    x = Dense(16, activation="relu")(x)
    x = Dense(len(classes), activation="softmax", name=name_output)(x)
    return x

def encoder(inp, filters):
    x1 = block_conv(inp, filters=filters, dropout=0.4)
    x2 = block_conv(x1, filters=filters, dropout=0.4)
    return x1, x2

def bottleneck(inp, filters):
    x = conv2d(filters)(inp)
    x = ReLU()(x)
    x = BatchNormalization()(x)
    
    x = conv2d(filters)(inp)
    x = ReLU()(x)
    return BatchNormalization()(x)
    
def conv2dtranspose(filters):
        return Conv2DTranspose(filters=filters, activation="relu", kernel_size=(2, 2), strides=(2, 2), data_format="channels_last", kernel_initializer='he_normal', kernel_regularizer=L2(l2=1e-5))

def up_block(inp, output_enc, filters, dropout=0.4):
    x = conv2d(filters)(inp)
    x = Concatenate()([inp, output_enc])
    x = ReLU()(x)
    x = BatchNormalization()(x)
    
    x = conv2d(filters)(inp)
    x = ReLU()(x)
    x = BatchNormalization()(x)
    
    x = conv2dtranspose(filters)(x)
    return Dropout(dropout)(x)
    
def decoder(inp, ouput_block1_enc, ouput_block2_enc, filters):
    x1 = up_block(inp, ouput_block2_enc, filters, dropout=0.4)
    x2 = up_block(x1, ouput_block1_enc, filters, dropout=0.4)
    return x2

def segmentation_architecture(inp, name_output=None):
    ouput_block1_enc, ouput_block2_enc = encoder(inp, filters=128)
    x = bottleneck(ouput_block2_enc, filters=128)
    x = decoder(x, ouput_block1_enc, ouput_block2_enc, filters=128)
    x = Conv2D(filters=1, activation="sigmoid", kernel_size=(2, 2), padding='same', strides=(1, 1), data_format="channels_last", kernel_initializer='he_normal', kernel_regularizer=L2(l2=1e-5), name=name_output)(x)
    return x

def get_model_both():
    tf.keras.backend.clear_session()
    
    inp = Input(shape=img_shape+(3,))
    seg_output = segmentation_architecture(inp, name_output="segmentation")
    clf_output = classification_w_backbone_architecture(inp, name_output="classification")
    
    model = Model(inputs=inp, outputs={"segmentation":seg_output,
                                       "classification":clf_output})
    
    model.compile(loss={"segmentation":Dice(),
                        "classification":CategoricalCrossentropy()},
                  optimizer=Adam(learning_rate=1e-3),
                  metrics={"segmentation":[BinaryIoU(), BinaryAccuracy()],
                           "classification":[AUC()]}
                 )
    return model

def train_model_both(model, train_data, val_data, epochs=100, version="base"):
    checkpoint_filepath = os.getcwd() + f'\scripts\kaggle\working\models\model_{version}.keras'
    checkpoint = ModelCheckpoint(checkpoint_filepath, monitor='val_loss', save_best_only=True)
    early = EarlyStopping(monitor='val_loss', patience=8, verbose=1, restore_best_weights=True)
    reduce = ReduceLROnPlateau(monitor='val_loss', patience=3)
    
    history = model.fit(train_data,
                      batch_size=batch_size,
                      epochs=epochs,
                      callbacks=[early,
                                 reduce,
                                 checkpoint],
                      validation_data=val_data,
                      steps_per_epoch=len(df_train)//batch_size,
                      validation_steps=len(df_val)//batch_size,
                      validation_batch_size=batch_size
                     )
    return history

def get_image(pathfile):
    return cv2.cvtColor(cv2.imread(pathfile), cv2.COLOR_BGR2RGB)

def RGB2LAB(image):
    image_lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(image_lab)
    return l, a, b

def get_mask(image):
    l, a, b = RGB2LAB(image)
    a_blur = cv2.GaussianBlur(a, (19, 19), 0)
    _, thresh_img = cv2.threshold(a_blur, 200, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    kernel = np.ones((1, 1),np.uint8) 
    return cv2.morphologyEx(thresh_img, op=cv2.MORPH_CLOSE, kernel=kernel, iterations=1)

def apply_mask(image, mask):
    return cv2.bitwise_and(image, image, mask=mask)

def createTFRecord(pathfile, dataset, size_img):
    #options = tf.io.TFRecordOptions(compression_type="GZIP", compression_level=9)
    writer = tf.io.TFRecordWriter(pathfile,
                                  #options=options
                                 )
    cnt_written_img = 0
    for index, row in dataset.iterrows():
        image = get_image(row.pathfiles)
        image = resize(image, size=img_shape, method="nearest", antialias=True).numpy()
        mask = np.array(get_mask(image))

        class_ = np.argmax(np.array(classes) == row.type_cell)
        image_data, mask_data = image.tobytes(), mask.tobytes()
        
        example = tf.train.Example(features=tf.train.Features(feature={
            'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_data])),
            'mask': tf.train.Feature(bytes_list=tf.train.BytesList(value=[mask_data])),
            'class': tf.train.Feature(int64_list=tf.train.Int64List(value=[class_])),
        }))

        writer.write(example.SerializeToString())
        cnt_written_img += 1
    return cnt_written_img

def parse(feature):
    features = tf.io.parse_single_example(
        feature,
        features={
        'image': tf.io.FixedLenFeature([], tf.string),
        'mask': tf.io.FixedLenFeature([], tf.string),
        'class': tf.io.VarLenFeature(tf.int64),
    })
    image = tf.reshape(tf.io.decode_raw(features['image'], out_type=tf.uint8), shape=(img_shape[0], img_shape[1], 3))
    mask = tf.reshape(tf.io.decode_raw(features['mask'], out_type=tf.uint8), shape=(img_shape[0], img_shape[1], 1))
    class_ = tf.sparse.to_dense(features["class"])
    return image, mask, class_
        
def get_class(x):
    for class_ in classes:
        if class_ in x:
            return class_
        
def both_task(image, mask, class_):
    class_ohe = tf.one_hot(indices=tf.squeeze(class_, axis=0), depth=4)
    return tf.cast(image, dtype=tf.float32)/255, {"segmentation": tf.cast(mask, dtype=tf.float32)/255, "classification": tf.cast(class_ohe, dtype=tf.float32)}

ENABLE_DOWNLOAD_DATASET = True
if ENABLE_DOWNLOAD_DATASET == True:
    #To download the dataset
    download_dataset()

seed_value = 42
os.environ['PYTHONHASHSEED'] = str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)

if __name__ == "__main__":
    imgs_path = []
    classes = []
    
    data_dir = os.getcwd()+"\scripts\kaggle\input\Blood cell Cancer [ALL]"
    for dirpath, dirnames, filenames in os.walk(data_dir):
        if len(classes) == 0:
            classes = dirnames
        for filename in filenames:
            imgs_path.append(os.path.join(dirpath, filename))
    
    df_imgs = pd.DataFrame(data=imgs_path, columns=["pathfiles"])
    df_imgs["type_cell"] = df_imgs.map(get_class)
            
    df_full_train, df_test = train_test_split(df_imgs, test_size=0.15, stratify=df_imgs.type_cell, shuffle=True, random_state=seed_value)
    df_train, df_val = train_test_split(df_full_train, test_size=1-(7/8.5), stratify=df_full_train.type_cell, shuffle=True, random_state=seed_value)

    img_shape = (192, 256)

    autotune = tf.data.AUTOTUNE
    create_tensorflow_record = True
    DIR_TFRECORD = os.getcwd()+"\scripts\kaggle\working\\blood_cell_cancer_with_mask"

    if create_tensorflow_record == True:
        createTFRecord(pathfile=DIR_TFRECORD+"_train.tfrecord", dataset=df_train, size_img=img_shape)
        createTFRecord(pathfile=DIR_TFRECORD+"_val.tfrecord", dataset=df_val, size_img=img_shape)
        createTFRecord(pathfile=DIR_TFRECORD+"_test.tfrecord", dataset=df_test, size_img=img_shape)
        
    train_data = tf.data.TFRecordDataset(DIR_TFRECORD+"_train.tfrecord")
    train_data = train_data.map(parse, num_parallel_calls=autotune)

    val_data = tf.data.TFRecordDataset(DIR_TFRECORD+"_val.tfrecord")
    val_data = val_data.map(parse, num_parallel_calls=autotune)

    batch_size = 2 #My desktop resources don't allow me to set the batch size to 32

    train_data_both = train_data.map(both_task, num_parallel_calls=autotune)
    train_data_both = train_data_both.batch(batch_size)
    train_data_both = train_data_both.prefetch(autotune)

    val_data_both = val_data.map(both_task, num_parallel_calls=autotune)
    val_data_both = val_data_both.batch(batch_size)
    val_data_both = val_data_both.prefetch(autotune)

    model_base_both = get_model_both()
    history_base_both = train_model_both(model_base_both, train_data_both, val_data_both, epochs=100, version="seg_clf")