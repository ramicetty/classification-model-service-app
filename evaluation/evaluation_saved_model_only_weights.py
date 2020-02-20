import tensorflow as tf

IMAGE_SIZE = 224
IMG_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, 3)
import dataset

imageDataset = dataset.ImageDataSet("../dataset_12022020/")
test_filenames, test_labels = imageDataset.build_dataset("test")

# Now we get a test dataset.
test_dataset = tf.data.Dataset.from_tensor_slices((tf.constant(test_filenames), tf.constant(test_labels)))
#test_dataset = test_dataset.batch(64)

IMAGE_SIZE = 224
learning_rate = 0.0001


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


test_data = (test_dataset.map(_parse_fn).batch(64))

def create_model():
    base_model = tf.keras.applications.VGG16(
        input_shape=IMG_SHAPE,
        include_top=False
    )

    # Trainable classification head
    maxpool_layer = tf.keras.layers.GlobalMaxPooling2D()
    prediction_layer = tf.keras.layers.Dense(1, activation='sigmoid')

    # Layer classification head with feature detector
    model = tf.keras.Sequential([
        base_model,
        maxpool_layer,
        prediction_layer
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
              loss='binary_crossentropy',
              metrics=['accuracy']
)

    return model

model = create_model()
model.load_weights("../trained_model/weights.150-0.38.hdf5")


loss, acc = model.evaluate(test_data, verbose=2)

print("loss:", loss)
print("acc:", acc)


