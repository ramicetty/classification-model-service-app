# Reference : https://towardsdatascience.com/easy-image-classification-with-tensorflow-2-0-f734fee52d13

import tensorflow as tf
import os
import random

print(tf.__version__)

class ImageDataSet(object):

    def __init__(self, base_path):

        self.base_path = base_path
        self.label2idx = {"invoice": 0, "non_invoice": 1}
        self.idx2label = {value: key for key, value in self.label2idx.items()}

    def read_label_files(self, dataset_path):
        labels = list(self.label2idx.keys())
        image_dict = {}
        for label in labels:
            images_names = os.listdir(os.path.join(dataset_path, label))
            for image_name in images_names:
                image_path = os.path.join(os.path.join(dataset_path, label), image_name)
                image_dict[image_path] = int(self.label2idx[label])

        return list(image_dict.keys()), list(image_dict.values())

    def build_dataset(self, dataset_type='train'):

        if "test" is dataset_type:
            dataset_path = os.path.join(self.base_path, "test")
        elif "val" in dataset_type:
            dataset_path = os.path.join(self.base_path, "val")
        else:
            dataset_path = os.path.join(self.base_path, "train")

        return self.read_label_files(dataset_path)


imageDataset = ImageDataSet("../dataset_12022020/")

train_filenames, train_labels = imageDataset.build_dataset("train")
val_filenames, val_labels = imageDataset.build_dataset("val")
test_filenames, test_labels = imageDataset.build_dataset("test")
train_data = tf.data.Dataset.from_tensor_slices(
    (tf.constant(train_filenames), tf.constant(train_labels))
)
val_data = tf.data.Dataset.from_tensor_slices(
    (tf.constant(val_filenames), tf.constant(val_labels))
)

# Now we get a test dataset.
test_dataset = tf.data.Dataset.from_tensor_slices((tf.constant(test_filenames), tf.constant(test_labels)))



num_train_samples = len(train_filenames)
num_val_samples = len(val_filenames)
IMAGE_SIZE = 224
BATCH_SIZE = 32

def decode_img(img):
  # convert the compressed string to a 3D uint8 tensor
  img = tf.image.decode_jpeg(img, channels=3)
  # Use `convert_image_dtype` to convert to floats in the [0,1] range.
  img = tf.image.convert_image_dtype(img, tf.float32)
  # resize the image to the desired size.
  return tf.image.resize(img, [IMAGE_SIZE, IMAGE_SIZE])


def _parse_fn(filename, label):
    img = tf.io.read_file(filename)
    img = decode_img(img)
    return img, label


train_data = (train_data.map(_parse_fn)
             .shuffle(buffer_size=10000)
             .batch(BATCH_SIZE)
             )
val_data = (val_data.map(_parse_fn)
           .shuffle(buffer_size=10000)
           .batch(BATCH_SIZE)
           )

test_data = (test_dataset.map(_parse_fn).batch(BATCH_SIZE))


IMG_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, 3)


base_model = tf.keras.applications.VGG16(
    input_shape=IMG_SHAPE,
    include_top=False,
    weights='imagenet'
)

# Freeze the pre-trained model weights
base_model.trainable = False

# Trainable classification head
maxpool_layer = tf.keras.layers.GlobalMaxPooling2D()
prediction_layer = tf.keras.layers.Dense(1, activation='sigmoid')


# Layer classification head with feature detector
model = tf.keras.Sequential([
    base_model,
    maxpool_layer,
    prediction_layer
])

learning_rate = 0.0001
num_epochs = 200
steps_per_epoch = round(num_train_samples)//BATCH_SIZE
val_steps = round(num_val_samples)//BATCH_SIZE


# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
              loss='binary_crossentropy',
              metrics=['accuracy']
)




cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath='../log/weights.{epoch:02d}-{val_loss:.2f}.hdf5',
                                                 save_weights_only=True,
                                                 verbose=1,save_best_only=True)
tf_board = tf.keras.callbacks.TensorBoard(log_dir='../log/')

model.fit(train_data.repeat(),
          epochs=num_epochs,
          steps_per_epoch = steps_per_epoch,
          validation_data=val_data.repeat(),
          validation_steps=val_steps,callbacks=[cp_callback, tf_board])

tf.keras.models.save_model(
    model,
    "../log/model1.h5",
    overwrite=True,
    include_optimizer=True,
    save_format='h5',
    signatures=None,
    options=None
)

print('\n# Evaluate')
loss, acc = model.evaluate(test_data, verbose=2)

print("Testing Loss:", loss)
print("Testing Accuracy:", acc)


