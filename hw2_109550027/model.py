import math
from collections import Counter, defaultdict
from typing import List

import nltk
import numpy as np
from nltk.tokenize.toktok import ToktokTokenizer
from sklearn.metrics import precision_recall_fscore_support
from sklearn.naive_bayes import GaussianNB
from tqdm import tqdm
import itertools

class Ngram:
    def __init__(self, config, n=2):
        self.tokenizer = ToktokTokenizer()
        self.n = n
        self.model = None
        self.config = config

    def tokenize(self, sentence):
        '''
        E.g.,
            sentence: 'Here dog.'
            tokenized sentence: ['Here', 'dog', '.']
        '''
        return self.tokenizer.tokenize(sentence)

    def get_ngram(self, corpus_tokenize: List[List[str]]):
        '''
        Compute the co-occurrence of each pair.
        '''
        # begin your code (Part 1)
        model = {}
        unicount={}
        #print(corpus_tokenize)
        feature = {}
        for li in corpus_tokenize:
            for ind in range(len(li)-1):
                firstw = li[ind]
                secw = li[ind+1]
                fands= firstw+" "+secw
                if fands not in feature:
                    feature[fands]=1
                else:
                    feature[fands]+=1
                if firstw in unicount:
                    unicount[firstw]+=1
                else:
                    unicount[firstw]=1
                if firstw in model:
                    if secw in model[firstw]:
                        model[firstw][secw]+=1
                    else:
                        model[firstw][secw]=1
                else:
                    model[firstw]={}
                    model[firstw][secw]=1
        
        for fword,dic in model.items():
            for sword,count in dic.items():
                model[fword][sword] = float(count/unicount[fword])
            #final_model = {count: float(count/unicount[fword]) for sword,count in dic.items()}
        #print(model)
        #print(feature)
        return model,feature
        # end your code
    
    def train(self, df):
        '''
        Train n-gram model.
        '''
        corpus = [['[CLS]'] + self.tokenize(document) for document in df['review']]     # [CLS] represents start of sequence
        
        # You may need to change the outputs, but you need to keep self.model at least.
        self.model, self.features = self.get_ngram(corpus)
        #self.model = self.get_ngram(corpus)

    def compute_perplexity(self, df_test) -> float:
        '''
        Compute the perplexity of n-gram model.
        Perplexity = 2^(-entropy)
        '''
        if self.model is None:
            raise NotImplementedError("Train your model first")

        corpus = [['[CLS]'] + self.tokenize(document) for document in df_test['review']]
        
        # begin your code (Part 2)
        model=self.model
        countm=0
        fu_entro=0.0
        for li in corpus:
            for ind in range(len(li)-1):
                firstw = li[ind]
                secw = li[ind+1]
                if(firstw in model and secw in model[firstw]):
                    pi=model[firstw][secw]
                    fu_entro += (math.log2(pi))
                else:
                    pi=0
                    countm-=1
            countm+=len(li)
        fu_entro/=countm
        perplexity = math.pow(2,(fu_entro*(-1)))   
        #print(perplexity)
        # end your code

        return perplexity

    def train_sentiment(self, df_train, df_test):
        '''
        Use the most n patterns as features for training Naive Bayes.
        It is optional to follow the hint we provided, but need to name as the same.

        Parameters:
            train_corpus_embedding: array-like of shape (n_samples_train, n_features)
            test_corpus_embedding: array-like of shape (n_samples_test, n_features)
        
        E.g.,
            Assume the features are [(I saw), (saw a), (an apple)],
            the embedding of the tokenized sentence ['[CLS]', 'I', 'saw', 'a', 'saw', 'saw', 'a', 'saw', '.'] will be
            [1, 2, 0]
            since the bi-gram of the sentence contains
            [([CLS] I), (I saw), (saw a), (a saw), (saw saw), (saw a), (a saw), (saw .)]
            The number of (I saw) is 1, the number of (saw a) is 2, and the number of (an apple) is 0.
        '''
        # begin your code (Part 3)
        features=self.features
        # step 1. select the most feature_num patterns as features, you can adjust feature_num for better score!
        feature_num = 400
        orderfeatures = {k: v for k, v in sorted(features.items(), key=lambda item: item[1], reverse=True)}
        orderfeatures = dict(itertools.islice(orderfeatures.items(), min(feature_num,len(orderfeatures))))
        #orderfeatures=orderfeatures[0:feature_num]
        # step 2. convert each sentence in both training data and testing data to embedding.
        # Note that you should name "train_corpus_embedding" and "test_corpus_embedding" for feeding the model.
        corpus_train = [['[CLS]'] + self.tokenize(document) for document in df_train['review']] 
        train_corpus_embedding=[]
        for i,li in enumerate(corpus_train):
            fsdic={}
            templist=[]
            for ind in range(len(li)-1):
                firstw = li[ind]
                secw = li[ind+1]
                fands= firstw+" "+secw
                if fands in fsdic:
                    fsdic[fands]+=1
                else:
                    fsdic[fands]=1
            for feature in orderfeatures:
                if feature in fsdic:
                    templist.append(fsdic[feature])
                else:
                    templist.append(0)
            train_corpus_embedding.append(templist)
        corpus_test = [['[CLS]'] + self.tokenize(document) for document in df_test['review']] 
        test_corpus_embedding=[]
        for i,li in enumerate(corpus_test):
            fsdic={}
            templist=[]
            for ind in range(len(li)-1):
                firstw = li[ind]
                secw = li[ind+1]
                fands= firstw+" "+secw
                if fands in fsdic:
                    fsdic[fands]+=1
                else:
                    fsdic[fands]=1
            for feature in orderfeatures:
                if feature in fsdic:
                    templist.append(fsdic[feature])
                else:
                    templist.append(0)
            test_corpus_embedding.append(templist)
        #print("end train sentiment")
        # end your code

        # feed converted embeddings to Naive Bayes
        nb_model = GaussianNB()
        nb_model.fit(train_corpus_embedding, df_train['sentiment'])
        y_predicted = nb_model.predict(test_corpus_embedding)
        precision, recall, f1, support = precision_recall_fscore_support(df_test['sentiment'], y_predicted, average='macro', zero_division=1)
        precision = round(precision, 4)
        recall = round(recall, 4)
        f1 = round(f1, 4)
        print(f"F1 score: {f1}, Precision: {precision}, Recall: {recall}")


if __name__ == '__main__':
    '''
    Here is TA's answer of part 1 for reference only.
    {'a': 0.5, 'saw: 0.25, '.': 0.25}

    Explanation:
    (saw -> a): 2
    (saw -> saw): 1
    (saw -> .): 1
    So the probability of the following word of 'saw' should be 1 normalized by 2+1+1.

    P(I | [CLS]) = 1
    P(saw | I) = 1; count(saw | I) / count(I)
    P(a | saw) = 0.5
    P(saw | a) = 1.0
    P(saw | saw) = 0.25
    P(. | saw) = 0.25
    '''

    # unit test
    test_sentence = {'review': ['I saw a saw saw a saw.']}
    model = Ngram(2)
    model.train(test_sentence)
    print(model.model['saw'])
    print("Perplexity: {}".format(model.compute_perplexity(test_sentence)))
