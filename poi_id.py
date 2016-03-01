#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

import myTools

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
#features_list = ['poi','salary'] # You will need to use more features
features_list = []

poi_label = ['poi']

financial_features = ['salary', 
					'deferral_payments', 
					'total_payments',
					'loan_advances', 
					'bonus',
					'deferred_income', 
					'total_stock_value', 
					'expenses', 
					'exercised_stock_options', 
					'other', 
					'long_term_incentive', 
					'restricted_stock', 
					'director_fees'] 

email_features = [  'to_messages', 
					'from_poi_to_this_person', 
					'from_messages', 
					'from_this_person_to_poi',  
					'shared_receipt_with_poi']

features_list = poi_label + financial_features + email_features


### Load the dictionary containing the dataset
data_dict = pickle.load(open("final_project_dataset.pkl", "r") )

### Task 2: Remove outliers
outliers = ['TOTAL', 'THE TRAVEL AGENCY IN THE PARK']
for outlier in outliers:
	data_dict.pop(outlier)

my_dataset = data_dict

selected_features_list = myTools.select_features_by_num(my_dataset, features_list, threshold = 82)
features_list = selected_features_list

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
from AddingFeature import AddingFeature
addFeature = AddingFeature(my_dataset, features_list)
#addFeature.duplicate_feature("exercised_stock_options", "exercised_stock_options_1")
addFeature.calculate_feature("total_stock_value", "exercised_stock_options", "new_features_1", "add")
addFeature.calculate_feature("total_stock_value", "salary", "new_features_2", "add")
addFeature.calculate_feature("shared_receipt_with_poi", "total_payments", "new_features_3", "multiply")
addFeature.delete_feature("total_stock_value")

features_list = addFeature.get_current_features_list()
my_dataset = addFeature.get_current_data_dict()

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
features = scaler.fit_transform(features)

from sklearn.decomposition import RandomizedPCA
n_components = 9
pca = RandomizedPCA(n_components = n_components, whiten = True)
pca.fit(features)

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html


# Example starting point. Try investigating other evaluation techniques!
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
k = 4
selectKB = SelectKBest(f_classif, k = k)
features = selectKB.fit_transform(features, labels)
index = selectKB.get_support().tolist()

new_features_list = []
for i in range(len(index)):
    if index[i]:
        new_features_list.append(features_list[i+1])
        
# Insert poi to the first element
new_features_list.insert(0, "poi")


# Re-run the featureFormat and targetFeatureSplit to remove all zeros data
data = featureFormat(my_dataset, new_features_list)
labels, features = targetFeatureSplit(data)

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV

skf = StratifiedKFold( labels, n_folds=3 )
accuracies = []
precisions = []
recalls = []


from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


for train_idx, test_idx in skf: 
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
    from sklearn.naive_bayes import GaussianNB
    clf = GaussianNB()
    clf.fit(features_train, labels_train)
    pred = clf.predict(features_test)
    
    accuracy = clf.score(features_test, labels_test) 
    labels_test_1 = labels_test
    
    ### for each fold, print some metrics
    print
    print "Accuracy: %f " %accuracy
    print "precision score: ", precision_score( labels_test, pred )
    print "recall score: ", recall_score( labels_test, pred )
    
    accuracies.append(accuracy)
    precisions.append( precision_score(labels_test, pred) )
    recalls.append( recall_score(labels_test, pred) )

### aggregate precision and recall over all folds
print "average accuracy: ", sum(accuracies)/3.
print "average precision: ", sum(precisions)/3.
print "average recall: ", sum(recalls)/3.





features_list = new_features_list
data_dict = my_dataset



### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.


pickle.dump(clf, open("my_classifier.pkl", "w") )
pickle.dump(data_dict, open("my_dataset.pkl", "w") )
pickle.dump(features_list, open("my_feature_list.pkl", "w") )
pickle.dump(data, open("my_data.pkl", "w"))
