import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Input
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
import matplotlib
matplotlib.use('TkAgg')

df = pd.read_csv("gender1.csv")
X = pd.get_dummies(df.iloc[:, :-1])  # One hot for categorical
y = df.iloc[:, -1]

le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_categorical = to_categorical(y_encoded)

X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2)

model = Sequential()
model.add(Input(shape=(X.shape[1],)))
model.add(Dense(8, activation='relu'))
model.add(Dense(6, activation='relu'))
model.add(Dense(y_categorical.shape[1], activation='softmax'))

model.summary()

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.00001, patience=100, verbose=1, mode='auto')
history = model.fit(X_train, y_train, epochs=100, callbacks=[early_stopping], verbose=1, validation_data=(X_test, y_test))


predictions = model.predict(X_test)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = np.argmax(y_test, axis=1)

plt.plot(history.history['loss'], label='Training Loss (categorical crossentropy)')
plt.plot(history.history['val_loss'], label='Validation Loss (categorical crossentropy)')
plt.title('Loss vs Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy vs Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()