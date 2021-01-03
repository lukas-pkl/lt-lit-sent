"""
Script to label data for sentiment analysis

Labels : 
-1 - Negative ;
0 - Neutral/No sentiment ;
1 - Positive;

Script also allows for early stopping using "break" input

OUTPUTS:
Dataframe:
Shape: label_count , 3
Columns: 
    words - word lemmas;
    vectors - spacy word vector ;
    labels - assigned labels

"""



import os
import spacy
import lt_core_news_lg
import pandas as pd
from sklearn.utils import shuffle

label_count = 1000

print("Reading in files")
nlp = lt_core_news_lg.load()

text = ""

files = [i for i in os.listdir("./Texts/1_batch/") if i.endswith(".txt")]
for f in files :
    with open( "./Texts/1_batch/"+f ) as file :
        text +="\n\n" + file.read()

doc = nlp( text )

pos = ["ADJ" , "ADV"]

words = [ ]
vectors = []

for tok in doc :
    if tok.pos_ in pos and tok.has_vector == True :
        word = tok.lemma_.lower()
        if word not in words :
            words.append( word )
            vectors.append( tok.vector )

print("Sentiment words found: ")
print(len(words))

df = pd.DataFrame()
df["words"] = words
df["vectors"] = vectors

df = shuffle( df , random_state=1001)

df = df.head( label_count ).reset_index()[["words" , "vectors"]]

print(df.shape)
print(df.head())

labels = []

values = ["1" , "0" , "-1" , "break" ]


print("Enter sentiment values (-1;0;1) for words or 'break' for early stopping")
for index, word in enumerate(list(df["words"]) ):
    print(index)
    sent = input(f"Enter the sentiment (-1,0,1) of : {word}")
    while sent not in values :
        sent = input(f"Enter the sentiment (-1,0,1) of : {word}")
    if sent == "break":
        break
    else:
        labels.append( int(sent) )

print("Labelling Finished.")
print()
print("SUMMARY:")
print("Label Count: ", len(labels))
print("POS: " , labels.count(1))
print("NEG: ", labels.count(-1))

df["labels"] = labels

print(df.shape)
print(df.head())


out_file_name = "LabelledData_1.pkl"
print("Saving Data as " + out_file_name)
df.to_pickle(  out_file_name )
