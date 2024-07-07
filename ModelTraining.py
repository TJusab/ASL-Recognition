import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf

with open('data.pickle', 'rb') as f:
    data_dict = pickle.load(f)

data = data_dict['data']
labels = data_dict['labels']

consistent_length = 42
inconsistent_data = [i for i, entry in enumerate(data) if len(entry) != consistent_length]

if inconsistent_data:
    print(f'Found {len(inconsistent_data)} inconsistent entries in data. Indices: {inconsistent_data}')

data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

x_train, x_test, y_train, y_test = train_test_split(data, labels_encoded, test_size=0.2, random_state=42)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(len(data[0]),)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

loss, accuracy = model.evaluate(x_test, y_test)
print(f'Test accuracy: {accuracy}')

model.save('./model.keras')
model_dict = {'model_path': './model.keras'}
with open('model.p', 'wb') as f:
    pickle.dump(model_dict, f)

