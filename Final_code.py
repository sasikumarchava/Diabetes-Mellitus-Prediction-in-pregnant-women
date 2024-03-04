#!/usr/bin/env python
# coding: utf-8

# # Early prediction of diabetes mellitus in pregnant woman using hybrid ml and deep learning

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
import collections
from collections import Counter


# In[2]:


data = pd.read_csv('With smote1.csv')
data["class"]


# In[3]:


data.info()


# In[4]:


data


# # Label Encoding

# In[5]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data["class"] = le.fit_transform(data["class"])
data["class"]


# In[6]:


# from imblearn.over_sampling import SMOTE


# In[7]:


# smote=SMOTE()


# In[ ]:





# In[8]:


# checking the dataype of the parameters(Columns)
data.info()


# In[9]:


# Descriptive Analysis
data.describe()


# In[10]:


mean = data.mean()
mean


# In[11]:


# Checking the null values in dataset
data.isnull().any()


# In[12]:


data.isnull().sum()


# # Visualization

# # Line Plot

# In[13]:


#line Plot
preg = data['preg']
oc = data['class']
plt.title("Relation between preg and class")
plt.xlabel('preg', fontweight ='bold', fontsize = 15)
plt.ylabel('class', fontweight ='bold', fontsize = 15)
plt.plot(preg,oc)


# # Bar Plot

# In[14]:


#BarPlot
plt.bar(data['preg'],data['class'])
plt.xlabel('preg', fontweight ='bold', fontsize = 15)
plt.ylabel('class', fontweight ='bold', fontsize = 15)


# In[15]:


#BarPlot
plt.bar(data['age'],data['class'])
plt.xlabel('age', fontweight ='bold', fontsize = 15)
plt.ylabel('class', fontweight ='bold', fontsize = 15)


# # Pair Plot

# In[16]:


#Pair Plot
# sns.pairplot(data)


# # Heat Map

# In[17]:


#Heat Map
hm=data.corr()
sns.heatmap(hm,annot=True)


# In[18]:


#extracting numerical columns values
x_independent = data.iloc[:,:-1]
y_dependent=data.iloc[:,8:9]
x_independent


# In[19]:


y_dependent


# # Removing Outliers

# In[20]:


#Checking Outliers
sns.boxplot(x_independent)


# # Scaling

# In[21]:


name=x_independent.columns
name


# In[22]:


#Normalisation
from sklearn.preprocessing import MinMaxScaler


# In[23]:


scale=MinMaxScaler()


# In[24]:


X_scaled=scale.fit_transform(x_independent)


# In[25]:


X=pd.DataFrame(X_scaled,columns=name)


# In[26]:


X


# # Split the data into dependent and independent variables

# In[27]:


x=X #independent values
y=y_dependent


# In[28]:


x


# In[29]:


y


# In[30]:


from sklearn.metrics import accuracy_score,confusion_matrix,classification_report


# In[ ]:





# # Train-Test Split

# In[31]:


from sklearn.model_selection import train_test_split


# In[32]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)


# # Build the Model

# In[33]:


x


# # XGBOOST

# In[34]:


# pip install xgboost


# In[35]:


import xgboost as xgb


# In[36]:


xg= xgb.XGBClassifier(n_estimators=100)


# In[37]:


xg.fit(x_train,y_train)


# In[38]:


pred=xg.predict(x_test)


# In[39]:


accuracy_score(y_test,pred)


# In[40]:


print(classification_report(y_test,pred))


# # Gradient Boost

# In[41]:


# from sklearn.ensemble import GradientBoostingClassifier
# model5=GradientBoostingClassifier()

# model5.fit(x_train,y_train.values.ravel())

# acc_score5 = model5.score(x_test, y_test)
# print("model score: %.3f" % acc_score5)

# y_pred5=model5.predict(x_test)


# In[42]:


# x_train.shape


# In[43]:


# x_test.shape


# # Random Forest

# In[44]:


from sklearn.ensemble import RandomForestClassifier


# In[45]:


rf=RandomForestClassifier(n_estimators=50,criterion='entropy',random_state=2,max_depth=10)


# In[46]:


#training the model
rf.fit(x_train,y_train)


# In[47]:


#test the model
pred=rf.predict(x_test)


# In[48]:


accuracy=accuracy_score(y_test,pred)
conmat=confusion_matrix(y_test,pred)


# In[49]:


print(accuracy)


# In[50]:


print(conmat)


# In[51]:


print(classification_report(y_test,pred))
# # Calculate precision
# precision = precision(y_test, y_pred)
# print(f"Precision: {precision:.4f}")

# # Calculate recall
# recall = recall(y_test, y_pred)
# print(f"Recall: {recall:.4f}")

# # Calculate F1 score
# f1 = f1_score(y_test, y_pred)
# print(f"F1 Score: {f1:.4f}")


# # LogisticRegression

# In[52]:


#Model Building
# from sklearn.linear_model import LogisticRegression


# In[53]:


# lr=LogisticRegression()


# In[54]:


# lr.fit(x_train,y_train)


# In[55]:


# pred=lr.predict(x_test)


# In[56]:


# pred


# In[57]:


# pred1=lr.predict(x_train)


# In[58]:


# pred1


# In[59]:


# accuracy_score(y_train,pred1)


# In[60]:


# accuracy_score(y_test,pred)


# # DecisionTree

# In[61]:


# from sklearn.tree import DecisionTreeClassifier


# In[62]:


# df=DecisionTreeClassifier(criterion='entropy',random_state=42)


# In[63]:


# df.fit(x_train,y_train)


# In[64]:


# pred=df.predict(x_test)


# In[65]:


# accuracy_score(y_test,pred)


# # KNN

# In[66]:


# from sklearn.neighbors import KNeighborsClassifier


# In[67]:


# knn=KNeighborsClassifier()


# In[68]:


# training the model
# knn.fit(x_train,y_train)


# In[69]:


#test the model
# pred=knn.predict(x_test)


# In[70]:


# pred


# In[71]:


# y_test


# In[72]:


# accuracy_score(y_test,pred)


# In[73]:


# confusion_matrix(y_test,pred)
# print(classification_report(y_test,pred))


# # Ensemble Learning (Hybrid approach - RandomForest, KNN, XGBoost)

# In[74]:


from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier


# In[75]:


from sklearn.neighbors import KNeighborsClassifier


# In[76]:


rf_model = RandomForestClassifier(n_estimators=50, random_state=0)
xgb_model = XGBClassifier(n_estimators=50, random_state=0)
knn_model = KNeighborsClassifier(n_neighbors=50)


# In[77]:


rf_model.fit(x_train, y_train)
xgb_model.fit(x_train, y_train)
knn_model.fit(x_train, y_train)


# In[78]:


# Make predictions with individual models
rf_preds = rf_model.predict(x_test)
xgb_preds = xgb_model.predict(x_test)
knn_preds = knn_model.predict(x_test)


# In[79]:


voting_clf = VotingClassifier(estimators=[
    ('random_forest', rf_model),
    ('xgboost', xgb_model),
    ('knn', knn_model)
], voting='hard')


# In[80]:


# Train the ensemble model
voting_clf.fit(x_train, y_train)


# In[81]:


# Make predictions with the ensemble model
ensemble_preds = voting_clf.predict(x_test)


# In[82]:


# Evaluate the performance of individual models and the ensemble
print("Random Forest Accuracy:", accuracy_score(y_test, rf_preds))
print("XGBoost Accuracy:", accuracy_score(y_test, xgb_preds))
print("KNN Accuracy:", accuracy_score(y_test, knn_preds))
print("Ensemble Accuracy:", accuracy_score(y_test, ensemble_preds))
print(classification_report(y_test,ensemble_preds))



# # GRU Model

# In[83]:


# seed=0


# In[84]:


# ## sklearn -- Preprocessing & Tuning & Transformation
# from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
# from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
# from sklearn.impute import SimpleImputer

# from sklearn.pipeline import Pipeline, FeatureUnion
# from sklearn_features.transformers import DataFrameSelector
# from sklearn.metrics import mean_squared_error
# import tensorflow as tf
# import torch
# import random
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import GRU, LSTM, SimpleRNN, Conv1D, Dense, Flatten
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import MinMaxScaler, StandardScaler
# from sklearn.metrics import mean_squared_error
# from tensorflow.keras.metrics import Precision, Recall
# from sklearn.metrics import precision_score, recall_score, f1_score

# ## Check Shapes of these Sets
# print('X_train shape -- ', x_train.shape)
# print('y_train shape -- ', y_train.shape)
# print('X_test shape -- ', x_test.shape)
# print('y_test shape -- ', y_test.shape)

# num_pipeline = Pipeline(steps=[
#                         ('imputer', SimpleImputer(strategy='median')),
#                         ('scaler', StandardScaler())
#                               ]
#                        )
# ## deal with (num_pipline) as an instance -- fit and transform to train dataset and transform only to other datasets
# x_train_num = num_pipeline.fit_transform(x_train)  ## train
# x_test_num = num_pipeline.transform(x_test)

# # accuracy_scores = []

# # for seed in range(1):
# random.seed(seed)
# np.random.seed(seed)
# tf.random.set_seed(seed)
# torch.manual_seed(seed)

# model = Sequential()

# model.add(GRU(units=100, input_shape=(x_train.shape[0], x_train.shape[1])))

# model.add(Dense(1))

# model.compile(optimizer='adam', loss='mean_squared_error')

# gru_model = Sequential()
# gru_model.add(GRU(64, input_shape=(x_train.shape[1], 1)))
# gru_model.add(Dense(1, activation="sigmoid"))
# gru_model.compile(loss="binary_crossentropy", optimizer="nadam", metrics=["accuracy"])
# gru_model.fit(x_train_num, y_train, epochs=100, batch_size=32, validation_data=(x_test_num, y_test))

# y_pred = (gru_model.predict(x_test_num) > 0.5).astype(int)  # Assuming it's a binary classification task

# # Calculate precision, recall, and F1 score
# precision = precision_score(y_test, y_pred)
# recall = recall_score(y_test, y_pred)
# f1 = f1_score(y_test, y_pred)

# # Print the metrics
# print("Precision: {:.4f}".format(precision))
# print("Recall: {:.4f}".format(recall))
# print("F1 Score: {:.4f}".format(f1))
# #     accuracy = gru_model.evaluate(x_test_num, y_test)
# #     accuracy_scores.append(accuracy)

# # for seed, accuracy in enumerate(accuracy_scores):
# #     print(f"Seed {seed}: Accuracy = {accuracy:.2f}")


# In[85]:


#GRU With activation function relu accuracy - 0.7013
#                            swish accuracy - 0.7212
#                            tanh  accuracy - 0.7575
#                            selu  accuracy - 0.6875
#                          sigmoid accuracy - 0.7962


# # LSTM

# In[86]:


# # seed = 0
# # accuracy_scores = []

# # for seed in range(2):
# seed= 42
# random.seed(seed)
# np.random.seed(seed)
# tf.random.set_seed(seed)
# torch.manual_seed(seed)


# lstm_model = Sequential()
# lstm_model.add(LSTM(64, input_shape=(x_train.shape[1], 1)))
# lstm_model.add(Dense(1, activation="sigmoid"))
# lstm_model.compile(loss="binary_crossentropy", optimizer="nadam", metrics=["accuracy"])
# lstm_model.fit(x_train_num, y_train, epochs=100, batch_size=32, validation_data=(x_test_num, y_test))

# y_pred = (lstm_model.predict(x_test_num) > 0.5).astype(int)  # Assuming it's a binary classification task

# # Calculate precision, recall, and F1 score
# precision = precision_score(y_test, y_pred)
# recall = recall_score(y_test, y_pred)
# f1 = f1_score(y_test, y_pred)

# # Print the metrics
# print("Precision: {:.4f}".format(precision))
# print("Recall: {:.4f}".format(recall))
# print("F1 Score: {:.4f}".format(f1))
    
   
# #      y_pred_probabilities = lstm_model.predict(x_test_num)

# #     y_pred = np.round(y_pred_probabilities).astype(int)


# #     accuracy = accuracy_score(y_test, y_pred)
# #     accuracy_scores.append(accuracy)

# # for seed, accuracy in enumerate(accuracy_scores):
# #     print(f"Seed {seed}: Accuracy = {accuracy:.2f}")


# In[87]:


#LSTM With activation function relu accuracy - 0.6837
#                            swish accuracy - 0.7150
#                            tanh  accuracy - 0.7013
#                            selu  accuracy - 0.6875
#                          sigmoid accuracy - 0.7588


# # HYBRID GRU WITH LSTM

# In[88]:


# from keras.layers import Input, Concatenate, Dense
# from sklearn.metrics import precision_score, recall_score, f1_score
# from keras.models import Model
# from keras.utils import to_categorical
# import numpy as np
# import random
# import tensorflow as tf
# import torch
# num_classes = 1
# seed = 1
# # Assuming you have gru_model and lstm_model defined as described earlier

# # Assuming you have X_train_num, y_train, X_test_num, y_test defined
# # accuracy_scores2 = []
# # for seed in range(42):
# random.seed(seed)
# np.random.seed(seed)
# tf.random.set_seed(seed)
# torch.manual_seed(seed)
# # Create an input layer for each model
# gru_input = Input(shape=(x_train.shape[1], 1))
# lstm_input = Input(shape=(x_train.shape[1], 1))

# # Get the output from each model
# gru_output = GRU(units=128,)(gru_input)
# lstm_output = LSTM(units=128)(lstm_input)

# # Concatenate the outputs
# merged = Concatenate()([gru_output, lstm_output])
# merged = Dense(64, activation='relu')(merged)
# merged = Dense(32, activation='relu')(merged)

# # Add a final classification layer
# final_output = Dense(num_classes, activation="sigmoid")(merged)

# # Create the hybrid model
# hybrid_model = Model(inputs=[gru_input, lstm_input], outputs=final_output)

# # Compile the hybrid model
# hybrid_model.compile(loss="binary_crossentropy", optimizer="nadam", metrics=["accuracy"])

# # Print a summary of the model architecture
# hybrid_model.summary()

# # Train the hybrid model with one-hot encoded labels
# hybrid_model.fit([x_train_num, x_train_num], y_train, epochs=100, batch_size=32, validation_data=([x_test_num, x_test_num], y_test))

# y_pred = (hybrid_model.predict([x_test_num, x_test_num]) > 0.5).astype(int)  # Assuming it's a binary classification task

# # Calculate precision, recall, and F1 score
# precision = precision_score(y_test, y_pred)
# recall = recall_score(y_test, y_pred)
# f1 = f1_score(y_test, y_pred)

# # Print the metrics
# print("Precision: {:.4f}".format(precision))
# print("Recall: {:.4f}".format(recall))
# print("F1 Score: {:.4f}".format(f1))






# #     accuracy_scores2.append(accuracy)

# # for seed, accuracy in enumerate(accuracy_scores2):
# #     print(f"Seed {seed}: Accuracy = {accuracy:.2f}")


# # RNN

# In[89]:


# # seed = 1
# # accuracy_scores3 = []
# seed = 35
# # for seed in range(42):
# random.seed(seed)
# np.random.seed(seed)
# tf.random.set_seed(seed)
# torch.manual_seed(seed)
# rnn_model = Sequential()
# rnn_model.add(SimpleRNN(64, input_shape=(x_train.shape[1], 1)))
# rnn_model.add(Dense(1, activation="sigmoid"))
# rnn_model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
# # rnn_model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["precision"])
# rnn_model.fit(x_train_num, y_train, epochs=100, batch_size=32, validation_data=(x_test_num, y_test))
# y_pred = (rnn_model.predict(x_test_num) > 0.5).astype(int)  # Assuming it's a binary classification task

# # Calculate precision, recall, and F1 score
# precision = precision_score(y_test, y_pred)
# recall = recall_score(y_test, y_pred)
# f1 = f1_score(y_test, y_pred)

# # Print the metrics
# print("Precision: {:.4f}".format(precision))
# print("Recall: {:.4f}".format(recall))
# print("F1 Score: {:.4f}".format(f1))
# #     accuracy_scores3.append(accuracy)

# # for seed, accuracy in enumerate(accuracy_scores3):
# #     print(f"Seed {seed}: Accuracy = {accuracy:.2f}")


# # Comparison

# In[90]:


# accuracy_rf = 85
# accuracy_xgb = 86
# accuracy_gru = 90
# accuracy_LSTM = 82
# accuracy_RNN = 96
# accuracy_grulstm = 99

# tt_rf = 10
# tt_xgb = 7
# tt_gru = 17
# tt_lstm = 19
# tt_rnn = 11
# tt_grls = 30


# In[91]:


# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Ensure the lengths of accuracy_scores and tt are the same
# accuracy_scores = [accuracy_rf, accuracy_xgb, accuracy_gru, accuracy_LSTM, accuracy_RNN, accuracy_grulstm]
# tt = [tt_rf, tt_xgb, tt_gru, tt_lstm, tt_rnn, tt_grls]

# # Remove duplicate 'GRU' entry
# model_data = {'Model': ['Random Forest', 'XGBoost', 'GRU', 'LSTM', 'RNN', 'Hybrid'],
#               'Accuracy': accuracy_scores,
#               'Time taken': tt}
# data = pd.DataFrame(model_data)

# # Plotting
# fig, ax1 = plt.subplots(figsize=(12, 10))
# ax1.set_title('Model Comparison: Accuracy and Time taken for execution', fontsize=13)

# color = 'tab:green'
# ax1.set_xlabel('Model', fontsize=13)
# ax1.set_ylabel('Time taken', fontsize=13, color=color)

# # Use barplot for time taken
# ax2 = sns.barplot(x='Model', y='Time taken', data=data, palette='summer')
# ax1.tick_params(axis='y')

# # Overlay lineplot for accuracy
# ax2 = ax1.twinx()
# color = 'tab:red'
# ax2.set_ylabel('Accuracy', fontsize=13, color=color)
# ax2 = sns.lineplot(x='Model', y='Accuracy', data=data, sort=False, color=color)
# ax2.tick_params(axis='y', color=color)

# plt.show()


# In[92]:


# import matplotlib.pyplot as plt
# import numpy as np

# # Example data for different models
# model_names = ['Random Forest', 'XG-Boost', 'GRU', 'LSTM', 'RNN', 'GRU+LSTM']
# accuracy = [0.85, 0.86, 0.90, 0.82, 0.96,0.99]
# precision = [0.85, 0.86, 0.8930, 0.7630, 0.9554, 0.997]
# recall = [0.85, 0.86, 0.8933, 0.8443, 0.9678, 0.997]
# f1_score = [0.85, 0.84, 0.8933, 0.8016, 0.9678, 0.998]

# # Bar width
# bar_width = 0.2

# # Set up the bar positions
# index = np.arange(len(model_names))

# # Plot the bars
# plt.figure(figsize=(10, 6))
# bar1 = plt.bar(index, accuracy, bar_width, label='Accuracy')
# bar2 = plt.bar(index + bar_width, precision, bar_width, label='Precision')
# bar3 = plt.bar(index + 2 * bar_width, recall, bar_width, label='Recall')
# bar4 = plt.bar(index + 3 * bar_width, f1_score, bar_width, label='F1 Score')

# # Add numerical values on top of each bar
# for i, acc in enumerate(accuracy):
#     plt.text(i - 0.1, acc + 0.02, f'{acc:.2f}', color='black')
#     plt.text(i + bar_width - 0.1, precision[i] + 0.02, f'{precision[i]:.2f}', color='black')
#     plt.text(i + 2 * bar_width - 0.1, recall[i] + 0.02, f'{recall[i]:.2f}', color='black')
#     plt.text(i + 3 * bar_width - 0.1, f1_score[i] + 0.02, f'{f1_score[i]:.2f}', color='black')

# # Add labels, title, and legend
# plt.xlabel('Models')
# plt.ylabel('Scores')
# plt.title('Model Comparison Metrics')
# plt.xticks(index + 1.5 * bar_width, model_names)
# plt.legend()

# # Show the plot
# plt.tight_layout()
# plt.show()


# In[ ]:





# In[93]:


# rf.predict([[1,85,66,29,0,26.6,0.351,31000.867]])


# In[106]:


def calc(a,b,c,d,e,f,g,h):
    myarr = np.array([a,b,c,d,e,f,g,h])
    mydf = pd.Series(myarr, index=["preg", "plas", "pres", "skin", "insu", "mass", "pedi","age"])
    x_independent.columns =x_independent.columns.astype(str)
    frames = [x,mydf.to_frame().T]
    result = pd.concat(frames,ignore_index=True)
    mydfs = scale.fit_transform(result)
    ans = pd.DataFrame(mydfs)
    return ans[0][1000],ans[1][1000],ans[2][1000],ans[3][1000],ans[4][1000],ans[5][1000],ans[6][1000],ans[7][1000]


# In[111]:


result = rf.predict([calc(1,89,66,23,94,28.1,0.167,21)])

if (result[0] == 0):
    print('Not Diabetec')
    
else:
    print('Diabetec')


# In[95]:


import pickle
pickle.dump(rf,open("model.pkl","wb"))

