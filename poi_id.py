#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn.feature_selection import SelectKBest
from sklearn import preprocessing

from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import StratifiedShuffleSplit

from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

features_list = ['poi',
                 'salary',
                 'to_messages',
                 'deferral_payments',
                 'total_payments',
                 'exercised_stock_options',
                 'bonus',
                 'restricted_stock',
                 'shared_receipt_with_poi',
                 'restricted_stock_deferred',
                 'total_stock_value',
                 'loan_advances',
                 'from_messages',
                 'other',
                 'from_this_person_to_poi', 
                 'director_fees',
                 'deferred_income',
                 'long_term_incentive',
                 'from_poi_to_this_person'] 
                 # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
'''there's 3 outliers to remove: TOTAL key, THE TRAVEL AGENCY IN THE PARK key 
and LOCKHART EUGENE E'''

def remove_outliers(to_remove):
    data_dict.pop(to_remove, None)
    
remove_outliers('TOTAL') 
remove_outliers('THE TRAVEL AGENCY IN THE PARK')
remove_outliers('LOCKHART EUGENE E')

### Task 3: Create new feature(s)

#new feature1: email ratio involving poi

features_email = ['from_messages', 'to_messages', 'from_poi_to_this_person',
                  'from_this_person_to_poi']

for key in data_dict:
    employee = data_dict[key]
    is_valid = True
    for email_type in features_email:
        if employee[email_type] == 'NaN':
            #print employee
            is_valid = False
    if is_valid:
        total_from = employee['from_poi_to_this_person'] + employee['from_messages']
        total_to = employee['from_this_person_to_poi'] + employee['to_messages']
        to_poi_ratio = float(employee['from_this_person_to_poi']) / total_to
        from_poi_ratio = float(employee['from_poi_to_this_person']) / total_from
        employee['poi_email_ratio'] = to_poi_ratio + from_poi_ratio
    else:
        employee['poi_email_ratio'] = 'NaN'
features_list+=['poi_email_ratio']

#new feature2: salary vs total non-salary ratio

features_benefits = ['salary', 'total_payments']

for key in data_dict:
    employee = data_dict[key]
    is_valid = True
    for benefit_type in features_benefits:
        if employee[benefit_type] == 'NaN':
            is_valid = False            
    if is_valid:
        total_non_salary = employee['total_payments'] - employee['salary']
        #print key, employee['salary'], total_non_salary
        employee['salary_ratio'] = float(employee['salary']) / total_non_salary
        #print employee['salary_ratio']
    else:
        employee['salary_ratio'] = 'NaN'
        
features_list+=['salary_ratio']
#print len(features_list)

### Store to my_dataset for easy export below.
my_dataset = data_dict

#Getting top 10 features
def get_best_features(data_dict, features_list, k):
     
    data = featureFormat(data_dict, features_list)
    labels, features = targetFeatureSplit(data)

    k_best = SelectKBest(k=k)
    k_best.fit(features, labels)
    scores = k_best.scores_

    pairs = zip(features_list[1:], scores)
    #print pairs
    
    sorted_pairs = list(reversed(sorted(pairs, key=lambda x: x[1])))
    k_best_features = dict(sorted_pairs[:k])
    
    print k_best_features
    return k_best_features.keys()


print get_best_features(my_dataset, features_list, 10)
my_feature_list= ['poi'] + get_best_features(my_dataset, features_list, 10)
print 'my_features',  my_feature_list

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, my_feature_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

# scale features using min-max Scaler
scaler = preprocessing.MinMaxScaler()
features = scaler.fit_transform(features)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
#The gauss_ function is using GaussinaNB classifier

def gauss_(features_train,labels_train, features_test, labels_test):
    
    clf_Gauss = GaussianNB()
    clf_Gauss.fit(features_train, labels_train)
    pred=clf_Gauss.predict(features_test)

    print 'precision score Gaussian', precision_score(labels_test, pred)
    print 'recall score Gaussian', recall_score(labels_test, pred)
    
    return clf_Gauss

'''the DT function will use DecisionTreeClassifier, but before that it does some 
parameters tuning using GridSearchCV'''
def dt_(features_train,labels_train, features_test, labels_test):

    parameters = {'max_depth':[None, 5, 10], 
              'min_samples_split':[2, 4, 6, 8, 10],
              'min_samples_leaf': [2, 4, 6, 8, 10],
              'criterion': ["entropy", "gini"]}
    

    dt = DecisionTreeClassifier()  
    clf = GridSearchCV(dt, parameters) 
    clf.fit(features, labels)
    print clf.best_params_
    
    #once we got the good parameters for DT classifier, we will tune the parameters with DT classifier
    #below is the scores for non-parameters tuned
    
    pred=clf.predict(features_test)
    print 'precision score Decision Tree', precision_score(labels_test, pred)
    print 'recall score Decision Tree', recall_score(labels_test, pred)
    
    clf_DT = DecisionTreeClassifier(min_samples_split=4, criterion='entropy', 
                                    max_depth=10, min_samples_leaf=2)
    clf_DT.fit(features, labels)
    pred_DT=clf_DT.predict(features_test)
    
    #below is the score for parameters tuned
    print 'precision score Decision Tree after parameter tune', precision_score(labels_test, pred_DT)
    print 'recall score Decision Tree after parameter tune', recall_score(labels_test, pred_DT)
    
    return clf_DT
    
def support_vector_(features_train,labels_train, features_test, labels_test):
    
    parameters = {'kernel':('linear','rbf'), 
                  'gamma': [0.0001, 0.1], 'C':[1,10]}    

    sv = SVC()  
    clf = GridSearchCV(sv, parameters) 
    clf.fit(features, labels)
    print clf.best_params_
    
    #below is the scores for non-parameters tuned
    pred=clf.predict(features_test)
    print 'precision score SVC:', precision_score(labels_test, pred)
    print 'recall score SVC:', recall_score(labels_test, pred)
    
    #{'kernel': 'linear', 'C': 10}
    clf_svc = SVC(kernel='linear', C=10)
    clf_svc.fit(features, labels)
    pred_svc=clf_svc.predict(features_test)
    
    #below is the scores for parameters tuned
    print 'precision score SVC after parameter tune:', precision_score(labels_test, pred_svc)
    print 'recall score SVC after parameter tune:', recall_score(labels_test, pred_svc)
    
    return clf_svc
    
### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
#splitting the dataset using train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)
    
clf_Gauss= gauss_(features_train, labels_train, features_test, labels_test)
print ''
clf_DT= dt_(features_train, labels_train, features_test, labels_test)
print ''
clf_svc= support_vector_(features_train, labels_train, features_test, labels_test)
print ''


#since the tester.py code is using StratifiedShuffleSplit instead train_test_split
def stratified_s_s(clf,features,labels, folds=1000):
    print 'clf', clf
    cv = StratifiedShuffleSplit(labels, folds, random_state = 42)
    precisions = []
    recalls = []
    
    for train_idx, test_idx in cv:
        features_train = []
        features_test  = []
        labels_train   = []
        labels_test    = []
        for ii in train_idx:
            features_train.append( features[ii] )
            labels_train.append( labels[ii] )
        for jj in test_idx:
            features_test.append( features[jj] )
            labels_test.append( labels[jj] )

        ### fit the classifier using training set, and test on test_set
        clf.fit(features_train, labels_train)
        pred = clf.predict(features_test)

    
    print "precision score: ", precision_score( labels_test, pred )
    print "recall score: ", recall_score( labels_test, pred )

print 'GaussianNB using Stratified Shuffle Split:'
stratified_s_s(GaussianNB(),features, labels)
print ''
print 'DecisionTree using using Stratified Shuffle Split:'
stratified_s_s(DecisionTreeClassifier(),features, labels)
print 'DecisionTree using Stratified Shuffle Split with parameter tune:'
stratified_s_s(DecisionTreeClassifier(min_samples_split=2, criterion= 'entropy', 
                                      max_depth= 5, min_samples_leaf= 2),
               features, labels)
print ''
print 'Support Vector using Stratified Shuffle Split'
stratified_s_s(SVC(),features, labels)
print 'Support Vector using Stratified Shuffle Split with parameter tune:'
stratified_s_s(SVC(kernel='linear', gamma=0.0001, C=10),features, labels)


my_classifier=GaussianNB()

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(my_classifier, my_dataset, my_feature_list)