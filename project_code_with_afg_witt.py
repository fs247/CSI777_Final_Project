# -*- coding: utf-8 -*-
"""
Purpose:  Ex5 Diabetes Naive Bayesian Classifier with Cross Validation
Class: Knowledge Mining 
Created on Thu Oct 10, 2022
@author: Richard Witt
For provided diabetes dataset, measure the performance of 
the standard naïve Bayesian classifier using cross-validation. What 
do the results indicate?
"""

#%%  0.1  Setup  --------------------------------------------------------------
####  0.1.1  Import Libraries  ---------------------------------------------------
file_path= "C:\\Users\\rivet\\OneDrive\\Documents\\1a_Mason\\Knowledge Mining\\Project\\"

import os
os.chdir('C:\\Users\\rivet\\OneDrive\\Documents\\Code_Repo')
import explore_v1 as exp 

import pandas as pd

#-- plotting data - matplotlib
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

from sklearn.model_selection import (cross_val_score,)
from sklearn.naive_bayes import (GaussianNB, 
                                 )
from sklearn import tree
from functools import reduce

#%% Ingest
#### Extract
df = pd.read_csv(file_path + 'ita_price.csv')
df['date'] = pd.to_datetime(df.date)
df.index =df.date
df = df.drop(columns=['date'])
df=df.sort_index()
ita_response = df[['adj_close']]

df = pd.read_csv(file_path + 'ita_data.csv')
df['date'] = pd.to_datetime(df.date)
df.index =df.date
df = df.drop(columns=['date'])
df=df.sort_index()
ita_predictors = df

df = pd.read_csv(file_path + '2000-01-01-2022-04-15-Afghanistan.csv')
df['date'] = pd.to_datetime(df.event_date)
df.index =df.date
df = df.drop(columns=['event_date'])
df=df.sort_index()
afg_data = df

#### Prepare  
# agrigate & select
df=ita_predictors
df = df.drop(columns=['ticker', 'derive_datetime'])
df=df.groupby(['date']).mean()
df = df[['ia_rps_mean',
         'ia_eps_mean',
         'ia_net_margin_mean',
         'earnings_oulnta',
         'rev_oulnta',
         'debt_ratio',
         'current_ratio',
         'debt_to_nta',
         'ia_apc_1m_mean',
         'ia_apc_1q_mean',
         ]]
predictors_ita_agrigate = df

df=afg_data
df = df[['fatalities',
         #'iso'
         ]]
df=df.groupby(['date']).sum()
#df = df.resample('1M').sum()
afg_agrigate = df
               
# merge                      
data_frames=[ita_response, afg_agrigate,predictors_ita_agrigate]
data_agrigate = reduce(lambda  left,right: pd.merge(
    left,right,
    left_index=True,right_index=True,
    how='outer'), data_frames)          

# slice 
df = data_agrigate
df = df[df.index >= afg_data.index.min()]
df = df[df.index <= afg_data.index.max()] 
data_sliced = df

# Derive
df = df[['ia_eps_mean',
        'ia_rps_mean',
        'rev_oulnta',
        'ia_net_margin_mean',
        'fatalities',
        'adj_close',
        ]]
df = df.interpolate()

df['adj_close'] = df['adj_close'].pct_change(periods=1)
data_percent = df
df['adj_close'] = pd.cut(df['adj_close'],5,labels=[1,2,3,4,5])

df = df.dropna()
all_data = df.reset_index(drop = True)

# x, y split & select
# comment out variables to re-create various charts in presentation
df = all_data
X_data = df.drop(columns=[#'close', 
                           'adj_close', 
                           #'volume', 
                           #'unadjusted_volume' 
                           ])
X_data_s = df[['ia_eps_mean',
                      'ia_rps_mean',
                      'rev_oulnta',
                      'ia_net_margin_mean',
                      #'fatalities',
                      ]]

y_data = all_data['adj_close']

#%% Explore
exp.data_dash_(all_data, file_path, title = 'Financial Data')
dir(exp)

exp.correlation_check_(data_agrigate,
                       title_name='Selected Predictors',
                        clmns = ['adj_close',
                                 'ia_eps_mean',
                                 'ia_rps_mean',
                                 'rev_oulnta',
                                 'fatalities'
                                 ]
                       )
diff1=all_data.diff(periods=1).dropna()
exp.correlation_check_(diff1,title_name='All Predictors 1st Differenced (Monthly)')

exp.scatter_plot_(data_percent['fatalities'], data_percent['adj_close'])

df = all_data['adj_close'].diff()
df = df[df< .20]
df = df[df> -.20]
exp.histo_kde(df['adj_close'],
              plot_type = 'both', 
              sup_title = 'Price Percent Change',
              )
exp.histo_kde(df['adj_close'],
              plot_type = 'both', 
              sup_title = 'Price Percent Change',
              bins = 5,
              )

y_data.value_counts()/len(y_data)

#%% test train split
#split dataset into train and test data
X_train, X_test, y_train, y_test = train_test_split(
    X_data, y_data, test_size=0.2, random_state=1)
X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
    X_data_s, y_data, test_size=0.2, random_state=1)

#%% Decision Tree
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)
y_pred = clf.predict_proba(X_test)
dtc_score = clf.score(X_test, y_test)
dtc_scores_all = cross_val_score(clf, 
                                X_data,
                                y_data, 
                                cv=10, 
                                scoring="accuracy")
dtc_score_all = dtc_scores_all.mean()

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train_s, y_train)
y_pred = clf.predict_proba(X_test_s)
dtc_scores_s = cross_val_score(clf, 
                              X_data_s, 
                              y_data, 
                              cv=10, 
                              scoring="accuracy")
dtc_score_s = dtc_scores_s.mean()

#%% Visualize Tree
fig=plt.figure(figsize=(10,6),dpi=300)
# Title
plt.figtext(x=0.05,y=0.95,s='This month Price Prediction Tree (Selected Variables)',
            fontsize=14,
            color='#163356')

tree.plot_tree(clf,
               feature_names=list(X_data.columns),
               class_names = ['big_loss', 'loss', 'even', 'gain', 'big_gain'],
               #proportion = True
               )
plt.show

#%% Naive Bayes 
#### Gaussian Naive Bayes with Cross Validation
gNB = GaussianNB()
gnb_scores_all = cross_val_score(gNB, 
                                X_data, 
                                y_data, 
                                cv=10, 
                                scoring="accuracy")#.mean()
gnb_score_all = gnb_scores_all.mean()


gnb_scores_s = cross_val_score(gNB, 
                                X_data_s, 
                                y_data, 
                                cv=10, 
                                scoring="accuracy")#.mean()
gnb_score_s = gnb_scores_s.mean()

#%% Results

y_lables = ('Naive Bayes with Fatalities', 
            'Naive Bayes without Fatalities', 
            'CART with Fatalities',
            'CART without Fatalities',
            )
performance = (gnb_score_all, gnb_score_s, 
               dtc_score_all, dtc_score_s)
NB_results = pd.DataFrame(performance, index=y_lables, columns=['performance']
                          ).sort_values('performance',ascending =False)
exp.bar_chart(NB_results,
               sup_title = 'Performance with and Without Fatalities'
               )

html= NB_results.to_html()
text_file = open("table.html", "w")
text_file.write(html)
text_file.close()


#### Comparison of means
from scipy.stats import ttest_ind
t, p = ttest_ind(gnb_scores_all, gnb_scores_s)
t, p

t, p = ttest_ind(dtc_scores_all, dtc_scores_s)
t, p













