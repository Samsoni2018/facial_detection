import tensorflow as tf
from matplotlib import pyplot as plt

class LossHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.accuracies = []
    def on_train_batch_begin(self, batch, logs={}):
        pass
    def on_train_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.accuracies.append(logs.get('accuracy'))


# Load MNIST train/test data
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Define simple classification network
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10)
])

# Define classification loss
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)


# Compile model + optimizer
model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

# Classification training
history = LossHistory()
model.fit(x_train, y_train, epochs=5, callbacks=[history])

# Evaluate accuracy
model.evaluate(x_test,  y_test, verbose=2)

# Plot MNIST digits
image = x_train[16]
plt.imshow(image)
plt.show()

# Plotting loss and accuracy
losses = history.losses
plt.title("Loss vs Training Steps")
plt.xlabel("Training Step")
plt.ylabel('Loss')
plt.plot(losses)
acc_s = history.accuracies
plt.figure()
plt.title("Accuracy vs Training Steps")
plt.xlabel("Training Step")
plt.ylabel('Accuracy')
plt.plot(acc_s)
plt.show()