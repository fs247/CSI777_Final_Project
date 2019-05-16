# %% [markdown]
# # Supervised Link Prediction with the Armed Conflict Location Event Database
# ## Modelling Notebook
#%%
import pandas as pd
import numpy as np
from joblib import dump, load
from sklearn.utils import resample
from itertools import product
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.metrics import confusion_matrix, recall_score, f1_score
#%% [markdown]
# # Read in training data
# From feature engineering phase
#%%
mode = 'read'
train_df = pd.read_parquet('df.parquet.gzip')
#%% [markdown]
# ## Define the time periods of interest
#%%
periods = [str(x)+"-"+str(y) for x, y in
           product(range(1997, 2019), range(1, 13))]
#%% [markdown]
# ## Set the test/train split
# 1998-1 - 2017-12 as Train <br>
# 2018-1 - 2018-12 as Test
#%%
training_range = periods[12:252]
testing_range = periods[252:]
train_flag = train_df.period.isin(training_range)
test_flag = train_df.period.isin(testing_range)
#%%
print('Creating Test/Train Splits')
train = train_df[train_flag]
test = train_df[test_flag]
#%%
print('Splitting Data')
X_train = train.drop('target', axis=1)\
    .set_index(["agent1", "agent2", "period"])
y_train = train\
    .set_index(["agent1", "agent2", "period"])\
    .loc[:, "target"]

X_test = test.drop('target', axis=1)\
    .set_index(["agent1", "agent2", "period"])
y_test = test\
    .set_index(["agent1", "agent2", "period"])\
    .loc[:, "target"]
print('Complete')
#%% [markdown]
# # Model Training
# Here we shall be using a Balanced Random Forest from the imbalanced-learn package.
# A regular random forest takes random subsamples of the data when constructing each tree but
# a balanced random forest creates a subsample by upsampling the minority class.
#%%
if mode == 'train':
    rf = BalancedRandomForestClassifier(n_jobs=5, verbose=10,
                                        n_estimators=150, random_state=123)
    rf.fit(X_train, y_train)
    dump(rf, 'model.joblib')
if mode == 'read':
    rf = load('model.joblib')
#%% [markdown]
# # Model performance
# This is a very very imbalanced dataset, so accuracy shouldn't be the metric of 
# overall performance.
# Accuracy doesn't tell us enough about false positives or false negatives (independently).
# We shall choose the metric of interest as recall. This is because (for example) a
# UN agent would be more concerned about false negatives than false positives. People could
# die if you get false negatives!
#%% [markdown]
# # Performance on Training data
#%%
y_train_pred = rf.predict(X_train)
#%% [markdown]
# #### Confusion Matrix
#%%
print(confusion_matrix(y_train, y_train_pred))
#%%
print("Recall Score (TRAIN): " + str(recall_score(y_train, y_train_pred)))
print("F1 (TRAIN): " + str(f1_score(y_train, y_train_pred)))
#%% [markdown]
# # Performance on Testing data
#%% 
y_test_pred = rf.predict(X_test)
#%% [markdown]
# #### Confusion Matrix
#%%
print(confusion_matrix(y_test, y_test_pred))
#%%
print("Recall Score (TEST): " + str(recall_score(y_test, y_test_pred)))
print("F1 (TEST): " + str(f1_score(y_test, y_test_pred)))
#%% [markdown]
# # Feature Importances
#%%
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]
cols = list(X_train.columns)
# Print the feature ranking
print("Feature ranking:")
for f in range(X_train.shape[1]):
    print(str(f + 1)+'. Feature: '+ cols[indices[f]] + ' '+ str(importances[indices[f]]))