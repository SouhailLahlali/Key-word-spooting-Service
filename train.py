import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
DATA_PATH = "data.json"
SAVED_MODEL_PATH = "model.h5"
EPOCHS = 40
BATCH_SIZE = 32
LEARNING_RATE = 0.0001

def load_data(data_path):

    with open(data_path, "r") as fp:
        data = json.load(fp)

    X = np.array(data["MFCCs"])
    y = np.array(data["labels"])
    print("Training sets loaded!")
    return X, y
def get_data_splits(data_path,test_size=0.1,test_validation=0.1):

    # load the dataset
    X,y = load_data(data_path)

    # creat train/test/validation splits
    X_train ,X_test , y_train , y_test = train_test_split(X,y ,test_size=test_size)
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=test_validation)

    # convert inputs from 2d to 3d
    X_train = X_train[..., np.newaxis]
    X_test = X_test[..., np.newaxis]
    X_validation = X_validation[..., np.newaxis]

    return  X_train, y_train, X_validation, y_validation, X_test, y_test

def build_model(input_shape,learning_rate, error="sparse_categorical_crossentropy"):
    # build network architecture using convolutional layers
    model = tf.keras.models.Sequential()

    # 1st conv layer
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=input_shape,
                                     kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'))

    # 2nd conv layer
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu',
                                     kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'))

    # 3rd conv layer
    model.add(tf.keras.layers.Conv2D(32, (2, 2), activation='relu',
                                     kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same'))

    # flatten output and feed into dense layer
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    tf.keras.layers.Dropout(0.3)

    # softmax output layer
    model.add(tf.keras.layers.Dense(10, activation='softmax'))

    optimiser = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimiser,loss=error,metrics=["accuracy"])

    # print model parameters on console
    model.summary()

    return model


def main():
    # generate train, validation and test sets
    X_train, y_train, X_validation, y_validation, X_test, y_test = get_data_splits(DATA_PATH)

    # create network
    input_shape = (X_train.shape[1], X_train.shape[2], 1)
    model = build_model(input_shape, learning_rate=LEARNING_RATE)

    # train network
    model.fit(X_train,y_train,epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(X_validation,y_validation))

    # evalute the model
    test_error , test_accuracy = model.evaluate(X_test,y_test)
    print(f"Test error : {test_error} , test accuracy : {test_accuracy}")

    # save the model
    model.save(SAVED_MODEL_PATH)

if __name__ == "__main__":
    main()