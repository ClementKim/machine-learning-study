import tensorflow as tf
import numpy as np

xy = np.loadtxt('data-03-diabetes.csv', delimiter=',', dtype=np.float32)
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

print(x_data.shape, y_data.shape)

tf.model = tf.keras.Sequential([
    tf.keras.Input(shape=(x_data.shape[1],)),

    # multi-variable, x_data.shape[1] == feature counts == 8 in this case
    tf.keras.layers.Dense(units=1, activation='sigmoid')
])

tf.model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),  metrics=['accuracy'])

tf.model.summary()

history = tf.model.fit(x_data, y_data, epochs=500)

# accuracy
print("Accuracy: {0}".format(history.history['accuracy'][-1]))

# predict a single data point
y_predict = tf.model.predict(np.array([[0.176471, 0.155779, 0, 0, 0, 0.052161, -0.952178, -0.733333]]))
print("Predict: {0}".format(y_predict))

# evaluating model
evaluate = tf.model.evaluate(x_data, y_data)
print("loss: {0}, accuracy: {1}".format(evaluate[0], evaluate[1]))
