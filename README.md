# Tags-Recommendation
Tags recommendation for articles

## Problem Description
HackerEarth wants to improve its customer experience by suggesting tags for any idea submitted by a participant for a given hackathon. Currently, tags can only be manually added by a participant. HackerEarth wants to automate this process with the help of machine learning. 

The task is to build a model that can predict or generate tags relevant to the idea/ article submitted by a participant.
## Data Description
* title - Title of the article

* article - Description of the article (raw format)

* tags - Tags associated with the respective article. If multiple tags are associated with an article then they are seperated           by '|'

## Approach
 * An article consists of code and description. One important tag that can be extracted is the programming language from the code. So code is separated from the article using HTML parsing.
  * Twenty most frequent programming are considered for classification. TF-IDF feature vectors are extracted and given as input to Logistic regression model to classify code to programming language.
 * The text other than code from the article is then preprocessed. A sequence to sequence architecture is created. Targets are created such that each word in the article is classified whether it is a tag or not.
 * Fasttext and Word2vec models are trained on the article text. A combination of these feature vectors are given as input to the Bidirectional Long Short Term Memory network which classifies each word as tag or not.
 * Predictions from sequence to sequence model are combined with programming language predicted to set the final predictions.
 
## Performance:
Trained the model on 700,000 questions and tested on 400,000 and the model is performing with an F1 score of 0.48.

Dataset courtesy : Hackerearth
