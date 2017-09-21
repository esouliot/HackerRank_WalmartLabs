# -*- coding: utf-8 -*-
"""
Created on Sat Oct 29 13:22:06 2016

@author: esouliot
"""
#Loading the raw training tsv file into a pandas dataframe
import pandas as pd
a=pd.read_table('train.tsv')
a=a.fillna(' ')

#Our columns of interest
name=a['Product Name']
dlong=a['Product Long Description']

#Turning tag string into tag list
tag=a['tag']
for i in tag:
    i=eval(i)

#Removing HTML tagging (up to 5 characters long not including brackets)
#Pandas includes Python regular expressions in its .str module
dlong=dlong.str.replace('<.?.?.?.?.?>',' ')

#Removing any non-alphanumeric characters that remain
#Again, using Python regular expressions
dlong=dlong.str.replace('\W',' ')
name=name.str.replace('\W',' ')

#Combining the Name feature and the Long Description Feature into one string
x=name + ' ' + dlong

#Importing necessary sklearn classes
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline

#Using the Pipeline class to streamline the document vectorization and classification fitting
text_clf=Pipeline([('tfidf', TfidfVectorizer(use_idf=True, ngram_range=(1,2))),
                   ('clf',SGDClassifier(random_state=42,n_jobs=-1,n_iter=125, loss='log', alpha=1e-6)),])
text_clf.fit(x,tag)

#Reading in the new, untagged features
new=pd.read_table('test.tsv')
new=new.fillna(' ')
new_dlong=new['Product Long Description']
new_name=new['Product Name']

#Performing the same cleaning on our features as before
new_dlong=new_dlong.str.replace('<.?.?.?.?.?>', ' ')
new_dlong=new_dlong.str.replace('\W',' ')
new_name=new_name.str.replace('\W',' ')
new_x=new_name+' '+new_dlong

#Training set fitted, testing set predicted
predicted=text_clf.predict(new_x)

#Adding our predicted tags to the test dataframe
new['tag']=predicted

#Out you go...
out=new[['item_id','tag']]
out.to_csv(path_or_buf='tags.tsv',sep='\t',index=False,header=True)