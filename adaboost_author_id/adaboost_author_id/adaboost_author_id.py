#!/usr/bin/python

import sys
import time
sys.path.append("../tools/")

print("Welcome to my new program which is able to predict whether an email was written by Sara or Chris.")
print("This is the code to accompany the AdaBoost mini-project. ")
print("~Bart0l",'\n')

""" 
    Using AdaBoost to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
original = "word_data.pkl"
destination = "word_data_unix.pkl"

content=''
outsize=0
with open(original, 'rb') as infile:
    content=infile.read()
with open(destination, 'wb') as output:
    for line in content.splitlines():
        outsize+=len(line) + 1
        output.write(line + str.encode('\n'))

print("Copy of emails is done. Saved %s bytes." % (len(content)-outsize))
print("Now we are preprocessing data.",'\n')

from email_preprocess import preprocess
features_train, features_test, labels_train, labels_test = preprocess()

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier

#features_train = features_train[:int(len(features_train)/100)] 
#labels_train = labels_train[:int(len(labels_train)/100)] 


print("We are predicing mails using your data. Please wait :)",'\n')
def rf_accuracy(features_train, labels_train, features_test, labels_test):
 
    clf = AdaBoostClassifier(n_estimators=100)

    t0 = time.time()
    clf.fit(features_train, labels_train)
    print("training time:", round(time.time() - t0, 3), "s")
       
    t0 = time.time()
    pred=clf.predict(features_test)
    print("predict time:", round(time.time() - t0, 3), "s", '\n')

    print("Mean accuracy of prediction:")
    scores = cross_val_score(clf, features_test, labels_test)
    scores.mean()    

    return scores

print(rf_accuracy(features_train, labels_train, features_test, labels_test), '\n')
