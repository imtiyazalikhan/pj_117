import pandas as pd

df = pd.read_csv("BankNote_Authentication.csv")
print (df.head())

from sklearn.model_selection import train_test_split

clases = df["class"]
variance = df["variance"]

clases_train, clases_test, variance_train, variance_test = train_test_split(clases, variance, text_size=0.25, random_state = 0)

from sklearn.linear_model import LogisticRegression
import numpy as np

X = np.reshape(clases_train.ravel(), (len(clases_train), 1))
Y = np.reshape(variance_train.ravel(), (len(variance_train), 1))

classifier = LogisticRegression(random_state = 0)
classifier.fit(X, Y)

x_test = np.reshape(clases_test.ravel(), (len(clases_test)))
y_test = np.reshape(variance_test.ravel(), (len(variance_test), 1))

variance_predection = classifier.predict(x_test)

predected_values = []
for i in variance_predection:
  if i == 0:
    predected_values.append("no")
  else:
    predected_values.append("yes")

actual_values = []
for i in Y_test.ravel():
  if i == 0:
    actual_values.append("no")
  else:
    actual_values.append("yes")

    labels = ["yes", "no"]
    
cm = confusion_matrix(actual_values, predected_values, labels)

ax = plt.subplot()
sns.heatmat(cm, annot=True, ax = ax)

ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
ax.set_title('Confusion Matrix')
ax.xaxis.set_ticklabels(labels); ax.yaxis.set_ticklabels(labels)
