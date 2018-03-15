

get_ipython().magic('load_ext Cython')

'''Import stuff here'''

import pandas as pd
import sklearn as sk
import numpy as np
import os
import re
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import pyximport; pyximport.install()
import matplotlib.pyplot as plt
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
# rcParams['figure.figsize'] = 15, 6
from sklearn.decomposition import PCA
import pylab as pl


'''Base folder for getting the dataset'''

trainDataUrl="Datasets/C50/C50train/"
testDataUrl="Datasets/C50/C50test/"
imageFolder="ImageFolder"


'''Getting author names from directory'''

authorNames=os.listdir(trainDataUrl);

'''Getting files from the directory'''

authorArticleFiles=[];
for author in authorNames:
    baseUrl=os.path.join(trainDataUrl,author)
    authorArticleFiles.extend([os.path.join(baseUrl,fileName) for fileName in os.listdir(baseUrl)])
    baseUrl=os.path.join(testDataUrl,author)
    authorArticleFiles.extend([os.path.join(baseUrl,fileName) for fileName in os.listdir(baseUrl)])


'''Getting author article'''
authorArticleList=[]
authorLabelList=[]
for fileName in authorArticleFiles:
    author=fileName.split('/')[-2];
    authorLabelList.append(authorNames.index(author))
    filePointer=open(fileName,'r');
    authorArticleList.append(filePointer.read())
    



'''Folder to clean the dataset'''

def cleanArticleData(article):
    
    #Removing Html Element
    article=BeautifulSoup(article).get_text()
    
    #Removing non Letters
    article=re.sub("[^a-zA-Z]"," ",article)
    
    #Lowercase and Splitting
    articleWords=article.lower().split()
    
    #Removing Stop Words
#     ps=PorterStemmer();
    ps=WordNetLemmatizer();
    refinedArticle=[];
    for word in articleWords:
        if word not in stopwords.words("english"):
            refinedArticle.append(ps.lemmatize(word))
            

    
    return " ".join(refinedArticle)


'''Cleaning the dataset and storing it in the list'''

cleanArticleList=[];
for article in authorArticleList:
    cleanArticleList.append(cleanArticleData(article))


'''Combining the list of authorArticles and authorLabels to create a dataFrame'''

dataList=[];
for i in range(len(cleanArticleList)):
    dataList.append([cleanArticleList[i],authorLabelList[i]])


'''Creating a dataframe'''

dataFrame=pd.DataFrame(dataList);
dataFrame.columns=['article','authorLabel']


'''Storing the dataframe to csv for furthur use'''
dataFrame.to_csv("ArticleDataset.csv")


print(dataFrame.head());



dataFrameArticles=dataFrame['article']


'''using tf-idf for feature selection'''

vectorizer=TfidfVectorizer(analyzer="word",min_df=0.2,max_df=0.8,stop_words="english")


'''Creating the input features from tf-idf vectorizer'''
x=vectorizer.fit_transform(cleanArticleList)


'''Vocabulary of tf-idf vectorizer'''
vocabulary=vectorizer.get_feature_names()


'''Printing top features after tf-idf'''
indices = np.argsort(vectorizer.idf_)[::-1]
features = vectorizer.get_feature_names()
top_n = 335
top_features = [features[i] for i in indices[:top_n]]
print(top_features)



'''Finding the ideal no of clusters '''

score=[];
for i in range(1,100):
    kmeans=KMeans(n_clusters=i);
    kmeans.fit_transform(x);
    score.append(kmeans.score(x));
    if(i%10==0):
        print("Completed i="+str(i));



posScore=[0-scoreElement for scoreElement in score]



'''Storing the loss after csv'''

pd.DataFrame(posScore).to_csv("score.csv");


'''Drawing the elbow curve'''
plt.close();
plt.plot(posScore,marker='o');
plt.xlabel("Number of Cluster");
plt.ylabel("loss")
plt.savefig(os.path.join(imageFolder,"elbowGraph.png"));
plt.close()


'''Applying kmeans for 5 clusers'''


n_cluster=5;

kmeansCluster=KMeans(n_clusters=n_cluster,random_state=10);

kmeansCluster.fit_transform(x)


kmeansCluster.inertia_

'''Getting output label of the cluster'''
analysisLabels=kmeansCluster.labels_


'''Formatting and analyzing the result'''

clusterDistribution=[];
clusterIndex=[];
for i in range(n_cluster):
    clusterIndex.append(np.where(analysisLabels==i));
    clusterDistribution.append(np.where(analysisLabels==i)[0].shape[0])


'''Plotting the cluster distribution'''
plt.close()
plt.plot(clusterDistribution,marker="o");
plt.xlabel("Number of cluster");
plt.ylabel("Size of Cluster");)
plt.savefig(os.path.join(imageFolder,"clusterDistribution.png"));
plt.close()

:


'''Getting top words of the each cluster'''
clusterDictList=[];
for i in range(n_cluster):
    clusterFeatureList=x[np.where(analysisLabels==i)];
    dist=np.sum(clusterFeatureList,axis=0);
    dist=dist.tolist()[0];
    clusterListTemp=[];
    for value,vocab in zip(dist,vocabulary):
        clusterListTemp.append([value,vocab])
    clusterListTemp.sort(reverse=True)
    clusterDictList.append(clusterListTemp)




clusterTopWord=[];
top_barrier=20;
for i in range(n_cluster):
    clusterTopWord.append([]);
    clusterDictList[i];
    for j in range(top_barrier):
        clusterTopWord[i].append(clusterDictList[i][j][1])

'''Converting the result of the top cluster to the dataframe'''

resultFrame=pd.DataFrame(clusterTopWord)

'''Saving the top words in the csv file'''
resultFrame.to_csv("topWords.csv");


'''Sorting the cluster by author names'''

authorLabelNumpy=np.array(dataFrame['authorLabel'])
authorClusterArticle=dict();
for j in range(len(authorNames)):
    authorClusterArticle[authorNames[j]]=[];

    


for i in range(n_cluster):
    clusterIndices=np.where(analysisLabels==i);
    authorLabelCluster=authorLabelNumpy[clusterIndices];
    for key in authorClusterArticle:
        index=authorNames.index(key);
        authorClusterArticle[key].append(len(np.where(authorLabelCluster==index)[0]))
        
'''Converting the sort of author by cluser to a dataframe and saving it in csv'''

authorArticleClusterResult=pd.DataFrame(authorClusterArticle)
reshapedAuthorArticleClusterResult=authorArticleClusterResult.pivot_table(columns=authorArticleClusterResult.index)
reshapedAuthorArticleClusterResult.to_csv("AuthorClusterResult.csv")


'''Using pca dimensionality reduction '''
pca = PCA(n_components=2).fit(x.toarray())
pca_2d = pca.transform(x.toarray())
pcaDataFrame=pd.DataFrame(pca_2d);
pcaDataFrame['labels']=analysisLabels
pcaDataFrame.to_csv("cluster.csv");

