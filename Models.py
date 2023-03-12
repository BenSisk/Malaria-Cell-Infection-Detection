def freezeLayers(model):
    for layer in model.layers:
        layer.trainable = False
        
    return model

def getMobileNet():
    from tensorflow.keras.applications import MobileNetV3Large
    model = MobileNetV3Large(input_shape = (224, 224, 3), # Shape of our images
    include_top = False, # Leave out the last fully connected layer
    weights = 'imagenet')
    
    model = freezeLayers(model)
    image_size = (224, 224)
    return model, image_size


def getCustomModel():
    import tensorflow as tf
    import tensorflow.keras.layers as layers
    
    model = tf.keras.Sequential([
        layers.Conv2D(32, (3, 3), activation = "relu", input_shape = (150, 150, 3)),
        layers.MaxPooling2D(2,2),
        layers.Conv2D(32, (3, 3), activation = "relu"),
        layers.MaxPooling2D(2,2)
    ])
    image_size = (150, 150)
    return model, image_size

def getCustomModel2():
    from tensorflow.keras import Sequential
    import tensorflow.keras.layers as layers
    
    model = Sequential([
        layers.Conv2D(32, (3, 3), activation = "relu", input_shape = (224, 224, 3)),
        layers.MaxPooling2D(2,2),
        layers.Conv2D(32, (3, 3), activation = "relu"),
        layers.MaxPooling2D(2,2),
        layers.Flatten(),
        layers.Dense(1024, activation = "relu", name="dense1"),
        layers.Dropout(0.1),
        layers.Dense(1024, activation = "relu", name="dense2"),
        layers.Dropout(0.1)
        
    ])
    image_size = (224, 224)
    return model, image_size


def getResNet50():
    from tensorflow.keras.applications.vgg16 import VGG16
    model = VGG16(input_shape = (224, 224, 3), # Shape of our images
    include_top = False, # Leave out the last fully connected layer
    weights = 'imagenet')
    
    model = freezeLayers(model)
    image_size = (224, 224)
    return model, image_size

def getVGG():
    from tensorflow.keras.applications.vgg16 import VGG16
    model = VGG16(input_shape = (224, 224, 3), # Shape of our images
    include_top = False, # Leave out the last fully connected layer
    weights = 'imagenet')
    
    model = freezeLayers(model)
    image_size = (224, 224)
    return model, image_size


def getEfficientNet():
    from efficientnet.keras import EfficientNetB0
    model = EfficientNetB0(input_shape = (224, 224, 3), # Shape of our images
    include_top = False, # Leave out the last fully connected layer
    weights = 'imagenet')
    
    model = freezeLayers(model)
    image_size = (224, 224)
    return model, image_size

def getInception():
    from tensorflow.keras.applications.inception_v3 import InceptionV3
    model = InceptionV3(input_shape = (150, 150, 3), # Shape of our images
    include_top = False, # Leave out the last fully connected layer
    weights = 'imagenet')
    
    model = freezeLayers(model)
    image_size = (150, 150)
    return model, image_size