import tensorflow as tf
from LoadData import loadData
import tensorflow.keras.layers as layers
import keras_tuner as kt

global base_model

def HyperTuner(passed_model, name = "modelname", batch_size = 128, image_size = (224, 224)):
    
    global base_model
    base_model = passed_model
    
    train, val, test = loadData(batch_size, image_size)
    
    tuner = kt.Hyperband(model_builder,
                        objective='val_acc',
                        max_epochs=10,
                        factor=3,
                        directory='HyperTuner/',
                        project_name=name)
    
    
    # ,
    # overwrite=True
    
    
    tuner.search(train, validation_data=val, epochs=50)

    # Get the optimal hyperparameters
    best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]

    print(f"""
    The hyperparameter search is complete. The optimal number of units in the first densely-connected
    layer is {best_hps.get('units')} and the optimal learning rate for the optimizer
    is {best_hps.get('learning_rate')}.
    """)
    
    # Early stopping can detect if a model has plateaued and cease training it any further, which could cause overfitting
    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    
    # Build the model with the optimal hyperparameters and train it on the data for 50 epochs
    model = tuner.hypermodel.build(best_hps)
    history = model.fit(train,
                        validation_data=val,
                        epochs=50,
                        callbacks=[stop_early])

    val_acc_per_epoch = history.history['val_acc']
    best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
    print('Best epoch: %d' % (best_epoch,))
    
    hypermodel = tuner.hypermodel.build(best_hps)

    # Retrain the model
    hypermodel.fit(train,
                   validation_data=val,
                   epochs=best_epoch,
                   callbacks=[stop_early])
    
    eval_result = hypermodel.evaluate(test)
    print("[test loss, test accuracy]:", eval_result)
    
    export_path = "Models/" + name
    model.save(export_path)
    
    
def model_builder(hp):
    global base_model

    # Tune the number of units in the first Dense layer
    # Choose an optimal value between 128-1024, increasing by 64
    hp_units = hp.Int('units', min_value=128, max_value=1024, step=64)
    x = layers.Dense(units=hp_units, activation='relu')((base_model.output))
    x = layers.Dense(1, activation='sigmoid')(x)
    model = tf.keras.models.Model(base_model.input, x)

    # Tune the learning rate for the optimizer
    # Choose an optimal value from 0.001, 0.0001, or 0.00001
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-3, 1e-4, 1e-5])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
                loss='binary_crossentropy',
                metrics=['acc'])

    return model