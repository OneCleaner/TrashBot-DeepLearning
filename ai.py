import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten
from sklearn.model_selection import train_test_split

#libreria dataset presa dalla repository dataset
import dataset as ds


#creiamo il modello
model = Sequential()
model.add(Flatten())
model.add(Dense(128))
#model.add(Dense(128, activation="relu"))
model.add(LeakyReLU(alpha=0.01)) #usiamo la LeakyReLU ma penso anche la ReLU vada bene, le funzioni di attivazione le devo studiare io
model.add(Dense(128))
#model.add(Dense(128, activation="relu"))
model.add(LeakyReLU(alpha=0.01))
#model.add(Dense(3, activation="sigmoid")) da provare la sigmoide ma non ne sono sicuro
model.add(Dense(3, activation="softmax"))

#usiamo sklearn per splittare il dataset in runtime
X_train, X_test, Y_train, Y_test = train_test_split(ds.X, ds.Y, test_size=0.3, random_state=0)

#compiliamo il modello, lo alleniamo e stampiamo la valutazione.
model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

model.fit(X_train, Y_train, epochs=100)
model.evaluate(X_test, Y_test)
