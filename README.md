# lt-lit-sent

## About 

This repo is a result of playing around with [`Spacy's` large model for Lithuanian language](https://spacy.io/models/lt#lt_core_news_lg). In addition to the classic features available in the other models, such as `NER` or `POS`-tagging, the larger models also feature word-vectors. These vectors are 300-dimensional `word-2-vec` embeddings that can be used to in a wide number of ways. In addition to its most popular use to determine similarity between words and texts, these embeddings can also be used as an input features to ML models. 

`Spacy` is great, but one thing that always botered me was that it did not conatin an in-built sentiment analyser. For English that is not a big problem - libraries, such as [`TextBlob`](https://textblob.readthedocs.io/en/dev/quickstart.html#sentiment-analysis) offer a simple version of this feature out-of-the-box and `Spacy` makes it easy to train a custom analyser for a more specialised use-case. 

However, it is not the the case for Lithuanian, where general purpose sentiment analyser is lacking. Hence, I thought it would be a good idea to build something of the sort. 

**NB!** While I believe that the general approach employed here is sound, the analyser has serious flaws stemming from limited training data sample. I would recommend increasing it at least two-fold before using it for something more serious than a hoby-project. 

## Repo

The repository consists of two parts. `ML` part contains a workflow to iteratively develop the sentiment analysis model iterating through the stages of data labelling and model training for several cycles.

The second part `Analysis` contains a sample analysis perfored using teh developed tool on a classical work of Lithuanian literature ["Altorių šešėly" by Vincas Mykolaitis-Putinas](https://lt.wikipedia.org/wiki/Altori%C5%B3_%C5%A1e%C5%A1%C4%97ly).

