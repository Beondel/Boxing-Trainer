import pandas as pd
import numpy as np
import sklearn.tree as tree
import sklearn.svm as svm
import sklearn.neighbors as knn
import matplotlib.pyplot as plt
import sklearn.metrics as mtr

# import and format data
df = pd.read_csv('./punch_data_1.csv')
punch_types = ['jab', 'hook']
df['punch_type'] = df['type'].map(lambda x: punch_types[x])

# separate into training and testing sets
df_train = df.drop(df.index[[0, 10, 20, 30, 40, 50, 60, 70]])
df_test = df.loc[[0, 10, 20, 30, 40, 50, 60, 70]]

# separate into features and labels
training_features = np.array(df_train[['x_acc', 'y_acc', 'z_acc']])
training_labels = np.array(df_train['punch_type'])
test_features = np.array(df_test[['x_acc', 'y_acc', 'z_acc']])
test_labels = np.array(df_test['punch_type'])

# create models
clf_1 = svm.SVC()

# train models
clf_1.fit(training_features, training_labels)

# test models
predicted_punches = clf_1.predict(test_features)
actual_punches = test_labels

print(predicted_punches)
print(actual_punches)
print(mtr.accuracy_score(actual_punches, predicted_punches))
