from tensorflow.keras import models,layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model

def create_model():
    '''
    Create model with padding length 150 and enmbedding 100
    Model compiled for binary classification
    '''
    reg_l1 = regularizers.L1(0.001)
    reg_l1l2= regularizers.L1L2(l1=0.0005, l2=0.0005)

    model=models.Sequential()
    model.add(layers.Masking(mask_value=0, input_shape=(150,100)))
    model.add(layers.Conv1D(64, kernel_size=3, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Conv1D(32, kernel_size=2, activation='relu', kernel_regularizer=reg_l1))
    model.add(layers.Dropout(0.5))
    model.add(layers.Flatten())
    model.add(layers.Dense(10, activation="relu", activity_regularizer=reg_l1l2))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation="sigmoid"))
    optim=Adam(learning_rate=0.0005)
    model.compile(loss='binary_crossentropy', optimizer=optim, metrics=['accuracy'])
    return model

def fit_model(model,X,y):
    '''
    Train model with early stopping
    '''
    es = EarlyStopping(patience=10, restore_best_weights=True, verbose=1)

    model.fit(X, y,
            epochs=200,
            batch_size=64,
            validation_split=0.3,
            verbose=0,
            callbacks=[es]
            )
    return model

def save_model(model):
    '''
    Save model to file
    '''
    model.save('saved_model/nlp_model')
    return
