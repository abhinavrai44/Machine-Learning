import matplotlib.pyplot as plt
import sys
import pickle
sys.path.append("../tools/")


from feature_format import featureFormat
from feature_format import targetFeatureSplit

import numpy as np
import myTools

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.lda import LDA
from sklearn.neural_network import BernoulliRBM

from sklearn.grid_search import GridSearchCV, RandomizedSearchCV

from sklearn.decomposition import PCA

features_list = []
                 
email_features_list = ['to_messages', 'from_poi_to_this_person',
                       'from_messages', 'from_this_person_to_poi',
                       'shared_receipt_with_poi']
                 
financial_features_list = ['salary', 'deferral_payments', 'total_payments',
                  'loan_advances', 'bonus', 'restricted_stock_deferred',
                  'deferred_income', 'total_stock_value', 'expenses',
                  'exercised_stock_options', 'other', 'long_term_incentive',
                  'restricted_stock', 'director_fees']
                  
target_label = ['poi']

# total features list: The first one should be 'poi' (target label)
total_features_list = target_label + email_features_list + financial_features_list

# financial features list with target label
financial_features_list = target_label + financial_features_list

# email features list with target label
email_features_list = target_label + email_features_list                 


              

### load the dictionary containing the dataset
data_dict = pickle.load(open("final_project_dataset.pkl", "r") )

### we suggest removing any outliers before proceeding further

### if you are creating any new features, you might want to do that here
### store to my_dataset for easy export below
my_dataset = data_dict

# This step only selects the features which the available data > 82
selected_features_list = myTools.select_features_by_num(my_dataset, total_features_list, threshold = 82)
features_list = selected_features_list




# Remove the "TOTAL" and "THE TRAVEL AGENCY IN THE PARK" data point (outliers)
my_dataset.pop('TOTAL', 0)
my_dataset.pop('THE TRAVEL AGENCY IN THE PARK', 0)


# Import AddingFeature class to add new feature
from AddingFeature import AddingFeature
addFeature = AddingFeature(my_dataset, features_list)
#addFeature.duplicate_feature("exercised_stock_options", "exercised_stock_options_1")
addFeature.calculate_feature("total_stock_value", "exercised_stock_options", "new_features_1", "add")
addFeature.calculate_feature("total_stock_value", "salary", "new_features_2", "add")
addFeature.calculate_feature("shared_receipt_with_poi", "total_payments", "new_features_3", "multiply")
addFeature.delete_feature("total_stock_value")
#addFeature.delete_feature("salary")

features_list = addFeature.get_current_features_list()
my_dataset = addFeature.get_current_data_dict()


### these two lines extract the features specified in features_list
data = featureFormat(my_dataset, features_list)


labels, features = targetFeatureSplit(data)

# Preprocessing the data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
features = scaler.fit_transform(features)

# Using feature selection to select the feature
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
k = 4
selectKB = SelectKBest(f_classif, k = k)
features = selectKB.fit_transform(features, labels)
index = selectKB.get_support().tolist()


# print features_list

# print selectKB.scores_
# print selectKB.pvalues_




new_features_list = []
for i in range(len(index)):
    if index[i]:
        new_features_list.append(features_list[i+1])

        
        
# Insert poi to the first element
new_features_list.insert(0, "poi")
# print new_features_list

# Re-run the featureFormat and targetFeatureSplit to remove all zeros data
data = featureFormat(my_dataset, new_features_list)
labels, features = targetFeatureSplit(data)

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV

skf = StratifiedKFold( labels, n_folds=2)
accuracies = []
precisions = []
recalls = []


from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


# for train_idx, test_idx in skf: 
#     features_train = []
#     features_test  = []
#     labels_train   = []
#     labels_test    = []
#     for ii in train_idx:
#         features_train.append( features[ii] )
#         labels_train.append( labels[ii] )
#     for jj in test_idx:
#         features_test.append( features[jj] )
#         labels_test.append( labels[jj] )
    
    ### fit the classifier using training set, and test on test set
    
    
    # Here comes weird part                               
    
    
    # RandomForest with GridSearch .838 .41 .22
    # param_grid = {"max_depth": [3, None],
    #           "max_features": [1, 3],
    #           "min_samples_split": [1, 3, 10],
    #           "min_samples_leaf": [1, 3, 10],
    #           "bootstrap": [True, False],
    #           "criterion": ["gini", "entropy"]}

    # clf = RandomForestClassifier(n_estimators=40)
    # grid_search = GridSearchCV(clf, param_grid=param_grid)
    # print clf.get_params
    # grid_search.fit(features_train, labels_train)
    # pred = grid_search.predict(features_test)

    # AdaBoost .81 .31 .22 
    # param_grid = {
    # 			"learning_rate" : [.90, 1.00, 1.05, 1.10]
    # }

    # clf = AdaBoostClassifier(SVC(probability=True), n_estimators = 40, algorithm = 'SAMME.R')
    # clf = AdaBoostClassifier(base_estimator = RandomForestClassifier(), n_estimators = 40, algorithm = 'SAMME.R')
    # clf = AdaBoostClassifier(base_estimator = DecisionTreeClassifier(), n_estimators = 40, algorithm = 'SAMME.R')
    # grid_search = GridSearchCV(clf, param_grid=param_grid)
    # print clf.get_params
    # grid_search.fit(features_train, labels_train)
    # pred = grid_search.predict(features_test)

    # clf = AdaBoostClassifier(RandomForestClassifier(bootstrap=True,
    #         criterion='gini', max_depth=None, max_features='auto',
    #         min_samples_leaf=1, min_samples_split=2,
    #         n_estimators=50, n_jobs=1, oob_score=False, random_state=None,
    #         verbose=0), algorithm = 'SAMME.R', n_estimators = 50)
    # clf.fit(features_train, labels_train)
    # pred = clf.predict(features_test)


    
    # Gaussian  .84 .41 .28
    # from sklearn.naive_bayes import GaussianNB
    # clf = GaussianNB()
    # clf.fit(features_train, labels_train)
    # pred = clf.predict(features_test)

    # DecisionTree .84 .4 .33
    # param_grid = {
    # 			"criterion" : ['gini', 'entropy'],
    # 			"min_samples_split" : [2, 3, 4, 5, 6, 8, 10],
    # 			"min_samples_leaf" : [1, 2]
    # }

    # clf = DecisionTreeClassifier(max_features=None)
    # grid_search = GridSearchCV(clf, param_grid=param_grid)
    # grid_search.fit(features_train, labels_train)
    # pred = grid_search.predict(features_test)

    # KNeighboursClassifier .85 .45 .17

   	# param_grid = {
   	# 			"n_neighbors" : [3, 4, 5, 6],
   	# 			"weights" : ['uniform', 'distance'],
   	# 			'algorithm' : ['auto', 'ball_tree', 'kd_tree'],
   	# 			'leaf_size' : [25, 30, 35, 40]
   	# }

   	# clf = KNeighborsClassifier(n_neighbors=4, algorithm='ball_tree')
   	# grid_search = GridSearchCV(clf, param_grid=param_grid)
    # grid_search.fit(features_train, labels_train)
    # pred = grid_search.predict(features_test)



    # clf = LDA()
    # clf.fit(features_train, labels_train)
    # pred = clf.predict(features_test)

    


    # accuracy = grid_search.score(features_test, labels_test) 
    # accuracy = clf.score(features_test, labels_test) 

    # labels_test_1 = labels_test
    
    ### for each fold, print some metrics
    # print
    # print "Accuracy: %f " %accuracy
    # print "precision score: ", precision_score( labels_test, pred )
    # print "recall score: ", recall_score( labels_test, pred )
    
    # accuracies.append(accuracy)
    # precisions.append( precision_score(labels_test, pred) )
    # recalls.append( recall_score(labels_test, pred) )



### aggregate precision and recall over all folds
# print 
# print
# print "average accuracy: ", sum(accuracies)/2.
# print "average precision: ", sum(precisions)/2.
# print "average recall: ", sum(recalls)/2.

from sklearn.cross_validation import StratifiedShuffleSplit
n_iter = 1000
sk_fold = StratifiedShuffleSplit(labels, n_iter=n_iter, test_size=0.1)
f1_avg = []
recall_avg = []
precision_avg = []
clf = LDA()

# Enumerate through the cross-validation splits get an index i for a timer
for i, all_index in enumerate(sk_fold):
    train_index = all_index[0]
    test_index = all_index[1]

    X_train = features(train_index)
    y_train = labels[train_index]
    
    X_test = features(test_index)        
    y_test = labels[test_index]
    
    # Use the best estimator trained earlier to fit
    # grid_search_object.best_estimator_.fit(X_train, y=y_train)
    clf.fit(X_train,y_train)
    test_pred = clf.predict(X_test)
    
    # Each time i is divsible by 10, print the 10% to console.
    if i % round(n_iter/10) == 0:
        sys.stdout.write('{0}%.. '.format(float(i)/n_iter*100)) 
        sys.stdout.flush()        
    f1_avg.append(f1_score(y_test, test_pred))
    precision_avg.append(precision_score(y_test, test_pred))
    recall_avg.append(recall_score(y_test, test_pred))

print "Done!"
print ""
print "F1 Avg: ", sum(f1_avg)/n_iter
print "Precision Avg: ", sum(precision_avg)/n_iter
print "Recall Avg: ", sum(recall_avg)/n_iter


features_list = new_features_list
data_dict = my_dataset

pickle.dump(clf, open("my_classifier.pkl", "w") )
pickle.dump(data_dict, open("my_dataset.pkl", "w") )
pickle.dump(features_list, open("my_feature_list.pkl", "w") )
pickle.dump(data, open("my_data.pkl", "w"))
