print("\nSTARTING TEST...\n\n\n\n\n")

import tensorflow as tf

# Load dataset
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Build a model

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10)
])

predictions = model(x_train[:1]).numpy()
print(predictions)

# Prediction
print(tf.nn.softmax(predictions).numpy())

# Loss
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
print(loss_fn(y_train[:1], predictions).numpy())

# Compile Model
model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

# Training
model.fit(x_train, y_train, epochs=5)

# Evaluation
model.evaluate(x_test,  y_test, verbose=2)

print("\nTEST COMPLETE!!\n")