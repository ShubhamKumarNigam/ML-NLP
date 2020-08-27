# BERT Key Concepts and Sources
__B__idirectional __E__ncoder __R__epresentations from __T__ransformers.

Inspired from [Blog by Chris McCormick](http://mccormickml.com/2019/11/11/bert-research-ep-1-key-concepts-and-sources/)

## Sources:

Paper: [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/pdf/1810.04805.pdf)

Github Repo: [github.com/google-research/bert](https://github.com/google-research/bert)

Google Blog Post: [Open Sourcing BERT: State-of-the-Art Pre-training for Natural Language Processing](https://ai.googleblog.com/2018/11/open-sourcing-bert-state-of-art-pre.html)

Paper: [Attention is all you need](https://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf)

The Annotated Transformer (Blog Post) [Harvard NLP](http://nlp.seas.harvard.edu/2018/04/03/attention.html)

__Jay Alammar’s Posts:__

BERT —— [The Illustrated BERT, ELMo, and co. (How NLP Cracked Transfer Learning)](http://jalammar.github.io/illustrated-bert/)

Transformer —— [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)

Attention —— [Visualizing A Neural Machine Translation Model (Mechanics of Seq2seq Models With Attention)](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/)

Coursera Videos(by Andrew Ng’s Sequence Models): [Sequence Models Course on Coursera](https://www.coursera.org/learn/nlp-sequence-models/home/)
Course covers:
* RNNs
* Encoder-Decoder
* LSTMs
* Bidirectional RNN
* Attention



## Significance
* Published October 2018
* Impresive benchmark performance
* __Transfer Learning__
  * BERT is huge and expensive to train
  * Leverage pre-training instead

BERT works in two steps, 
1. It uses a large amount of __unlabeled data__ to learn a language representation in an unsupervised fashion called __pre-training__. 
2. Then, the pre-trained model can be __fine-tuned__ in a supervised fashion using a small amount of __labeled trained__ data to perform various supervised tasks.

### Difference between BERT and other Encoder-Decoder models:
The general transformer uses an encoder and a decoder network, however, as BERT is a pre-training model, it only uses the encoder to learn a latent representation of the input text.

__BERT’s state-of-the-art performance is based on two things.__
1. Novel pre-training tasks called __Masked Language Model(MLM)__ and __Next Sentense Prediction (NSP)__.
2. A lot of data and compute power to train BERT.

MLM makes it possible to perform bidirectional learning from the text.

However, __Generative Pre-training (GPT)__ used left-to-right training and __ELMo__ used shallow bidirectionality.

In conclusion:

__BERT is deeply bidirectional, OpenAI GPT is unidirectional, and ELMo is shallowly bidirectional.__

### How different from other Pre-trained models:
* Pre-trained representations can either be context-free or contextual, and contextual representations can further be unidirectional or bidirectional. 

* Context-free models such as word2vec or GloVe generate a single word embedding representation for each word in the vocabulary. 

* For example, the word “bank” would have the same context-free representation in “bank account” and “bank of the river.” 

* Contextual models instead generate a representation of each word that is based on the other words in the sentence. 

* For example, in the sentence “I accessed the bank account,” a unidirectional contextual model would represent “bank” based on “I accessed the” but not “account.” 

* However, BERT represents “bank” using both its previous and next context — “I accessed the ... account”

### XLNet
* XLNet introduces permutation language modeling, where all tokens are predicted but in random order. 
* This is in contrast to BERT’s masked language model where only the masked (15%) tokens are predicted. 
* This is also in contrast to the traditional language models, where all tokens were predicted in sequential order instead of random order. 
* This helps the model to learn bidirectional relationships and therefore better handles dependencies and relations between words. 

### RoBERTa:
* Introduced at Facebook, Robustly optimized BERT approach RoBERTa
* It is a retraining of BERT with improved training methodology, 1000% more data and compute power.
* To improve the training procedure, RoBERTa removes the Next Sentence Prediction (NSP) task from BERT’s pre-training and introduces dynamic masking so that the masked token changes during the training epochs. 
* Larger batch-training sizes were also found to be more useful in the training procedure.

### DistilBERT:
* It learns a distilled (approximate) version of BERT, retaining 97% performance but using only half the number of parameters.
* It does not has token-type embeddings, pooler and retains only half of the layers from Google’s BERT.
* This is in some sense similar to posterior approximation.
