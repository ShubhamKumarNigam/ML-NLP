# Word Embedding

by Jaron Collis
[Glossary of Deep Learning: Word Embedding](https://medium.com/deeper-learning/glossary-of-deep-learning-word-embedding-f90c3cec34ca)

<img src = "https://miro.medium.com/max/2000/1*52X2L01wpUjy39lIjofC7g.jpeg" width = "80%">

`A plot of word embeddings in English and German. The semantic equivalence of words has been inferred by their context, so similar meanings are co-located. This is because the relative semantics of words are consistent, whatever the language. `

## What ?

Turn text into numbers.

## Why Numbers?

This transformation is necessary because many ML algorithms (including deep nets) require their input to be vectors of continuous values; they just won’t work on strings of plain text.

So a natural language modelling technique like Word Embedding is used to map words or phrases from a vocabulary to a corresponding vector of real numbers.

## Properties

* __Dimensionality Reduction__ — it is a more efficient representation.
* __Contextual Similarity__ — it is a more expressive representation.

## Compared with BoW (Bag of Words)

BoW is huge, very sparse __one-hot__ encoded vectors, where the __dimensionality__ of the vectors representing each document is equal to the size of the supported __vocabulary__. 

Word Embedding aims to create a vector representation with a much lower dimensional space. These are called Word Vectors.

## Uses

Word Vectors are used for __semantic parsing__, to extract meaning from text to enable natural language understanding. 

# Word2Vec

`Each word has an associated vector, hence the name: word2vec.`

Two ways:
1. CBOW (Continuous Bag-Of-Words)
2. Skip-gram.

## CBoW

* In CBOW, we have a __window__ around some __target word__,
* Then consider the words around it (its context). 
* We __supply those words as input__ into our network and then use it to try to __predict the target word__.

## Skip-gram

Skip-gram does the opposite, you have a __target word__, and you try to predict the words that are in the window around that word, i.e. __predict the context around a word__.

## How

* The __input words__ are passed in as __one-hot__ encoded vectors. 
* This will go into a __hidden layer__ of linear units, then into a __softmax layer__ to make a prediction. 
* The idea here is to train the hidden layer weight matrix to find efficient representations for our words. 
* This weight matrix is usually called the __embedding matrix__, and can be queried as a look-up table.

<img src = "https://miro.medium.com/max/1400/0*kx5_UXWs7Q_c071d." width = "70%">

`The word2vec architecture consists of a hidden layer and an output layer`

## Embedding Matrix

* The embedding matrix has a __size of the number of words by the number of neurons__ in the hidden layer (the embed size).
* So, if you have 10,000 words and 300 hidden units, the matrix will have size 10,000×300 (as we’re using one-hot encoded vectors for our inputs). 
* Once computed, getting the word vector is a speedy O(1) lookup of corresponding row of the results matrix:

<img src = "https://miro.medium.com/max/1242/0*IFv_QtwBNHfGy1Tm." width = "70%">


## Contextual Similarities between words

The classic example is subtracting the ‘notion’ of `“King”` from `“Man”` and adding the notion of `“Woman”` and results being the word `“Queen”`.

<img src = "https://miro.medium.com/max/1400/0*1ndbQpbmRrzZWTjO.png" width = "70%">



