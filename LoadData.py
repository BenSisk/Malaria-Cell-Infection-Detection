import tensorflow as tf
from tensorflow.keras.utils import image_dataset_from_directory
import tensorflow.keras.layers as layers

def loadData(batch_size = 32, image_size = (224, 224)):
    split = 0.3
    shuffle = True
    seed = 1
    directory = "./data/"
    
    train = image_dataset_from_directory(
        directory,
        validation_split=split,
        subset="training",
        seed=seed,
        image_size=image_size,
        batch_size=batch_size,
        shuffle = shuffle
    )

    val_test_ds = image_dataset_from_directory(
        directory,
        validation_split=split,
        subset="validation",
        seed=seed,
        image_size=image_size,
        batch_size=batch_size,
        shuffle = shuffle
    )
    
    
    #Resizing not needed as images are already resized when they are imported
    normalization_layer = layers.Rescaling(1./255)
    
    train = train.map(lambda x, y: (normalization_layer(x), y)) # Where x—images, y—labels.
    val_test_ds = val_test_ds.map(lambda x, y: (normalization_layer(x), y)) # Where x—images, y—labels.
    
#     data_augmentation = tf.keras.Sequential([
#         layers.RandomFlip("horizontal_and_vertical"),
#         layers.RandomRotation(0.2)
#     ])
    
#     train = train.map(lambda x, y: (data_augmentation(x), y)) # Where x—images, y—labels.
    
    num_batches = tf.data.experimental.cardinality(val_test_ds)
    
    val = val_test_ds.take((2*num_batches) // 3)
    
    test = val_test_ds.skip((2*num_batches) // 3)

    return train, val, test