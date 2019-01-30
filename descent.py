from keras.models import Sequential
from keras.layers import Dense
from keras import backend as k
from keras import losses
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_squared_error
from math import sqrt
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

model = Sequential()
model.add(Dense(9, input_dim=9, kernel_initializer='uniform', activation='relu'))
model.add(Dense(9, kernel_initializer='uniform', activation='relu'))
model.add(Dense(9, kernel_initializer='uniform', activation='sigmoid'))
#model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

inputs = np.random.random((1, 9))
outputs = model.predict(inputs)
targets = np.random.random((1, 9))
rmse = sqrt(mean_squared_error(targets, outputs))

print("===BEFORE WALKING DOWN GRADIENT===")
print("outputs:\n", outputs)
print("targets:\n", targets)
print("RMSE:", rmse)


def descend(steps=40, learning_rate=100.0, learning_decay=0.95):
    for s in range(steps):

        # If your target changes, you need to update the loss
        loss = losses.mean_squared_error(targets, model.output)

        #  ===== Symbolic Gradient =====
        # Tensorflow Tensor Object
        gradients = k.gradients(loss, model.trainable_weights)

        # ===== Numerical gradient =====
        # Numpy ndarray Objcet
        evaluated_gradients = sess.run(gradients, feed_dict={model.input: inputs})  # BARES INSPECTIONS

        # For every trainable layer in the network
        for i in range(len(model.trainable_weights)):

            layer = model.trainable_weights[i]  # Select the layer

            # And modify it explicitly in TensorFlow
            sess.run(tf.assign_sub(layer, learning_rate * evaluated_gradients[i]))

        # decrease the learning rate
        learning_rate *= learning_decay

        outputs = model.predict(inputs)
        rmse = sqrt(mean_squared_error(targets, outputs))

        print("RMSE:", rmse)

if __name__ == "__main__":
    # Begin TensorFlow
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    descend(steps=5, learning_rate=100.0, learning_decay=.99)

    final_outputs = model.predict(inputs)
    final_rmse = sqrt(mean_squared_error(targets, final_outputs))

    print("===AFTER WALKING DOWN GRADIENT===")
    print("outputs:\n", final_outputs)
    print("targets:\n", targets)
