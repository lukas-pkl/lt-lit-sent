"""
Tool to analyse sentiment of Lithuanian text

Created 02.01.2021

Author: Lukas-pkl
"""
import pickle
import numpy as np
import spacy 
import lt_core_news_lg


with open("SentModel_3rdGen.pkl" , "rb") as file :
    model = pickle.load( file )

class SentimentAnalyser:

    def __init__(self):
        """
        Sets the initial params:
            nlp - Spacy model;
            model - ML model;
            sentiment_pos - from which POS to extract sentiment;
            min_sentiment_word_count - minum number of sentiment words in text 
        """
        self.nlp = lt_core_news_lg.load() 
        self.model = model
        self.sentiment_pos = ["ADJ" , "ADV"] #Sets parts of speech that carry sentiment 
        self.min_sent_word_count = 4 #Sets minimum nuber of sentiment words in a given text to perform analysis 
        
    def sentiment( self, text ):
        """
        Analyses the sentiment of a Lithuanian text

        PARAMS:
            text - piece of text (max 1M chars);

        RETURNS 
            dict 
        """

        self.doc = self.nlp( text )

        self.sentiment_words = []
        vectors = []
        for token in self.doc :
            if token.pos_ in self.sentiment_pos and token.has_vector == True :
                self.sentiment_words.append( token.text )
                vectors.append( token.vector )
        
        if len(vectors) > self.min_sent_word_count :
            self.features = np.array( vectors)
            self.pred_labels= list(self.model.predict( self.features ))

            label_sum = sum(self.pred_labels)
            label_mean = label_sum/len(self.pred_labels) 
            
            if label_sum > 0 :
                label_score = label_mean*self.pred_labels.count(1)
            elif label_sum<0:
                label_score = label_mean*self.pred_labels.count(-1)
            else:
                label_score = 0

            result = { "sentiment_words" : self.sentiment_words , 
                        "labels" : list( self.pred_labels) , 
                        "sum" : label_sum , 
                        "mean" : label_mean , 
                        "score" : label_score }
            
        else:
            result = { "sentiment_words" : "Too few sentimnet words" , 
                        "labels" : None , 
                        "sum" : 0 , 
                        "mean" : 0 , 
                        "score" : 0 }
        return result
        



