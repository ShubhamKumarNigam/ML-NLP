# Statistical models

### What are statistical models?
* Enable spaCy to predict linguistic attributes in context
    * Part-of-speech tags
    * Syntactic dependencies
    * Named entities
* Trained on labeled example texts 
* Can be updated with more examples to fine-tune predictions

## CORE MODELS (English)
CORE: vocabulary, syntax, entities, vectors

Pretrained statistical models for English

1. en_core_web_sm (small: 11 MB)
2. en_core_web_md (medium: 91 MB)
3. en_core_web_lg (large: 789 MB)

GENRE ==> WEB ==> written text (blogs, news, comments)

For more detail visit [spacy models](https://spacy.io/models/en).

### Model Packages


```python
!python -m spacy download en_core_web_sm
```

    Requirement already satisfied: en_core_web_sm==2.2.5 from https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz#egg=en_core_web_sm==2.2.5 in /anaconda3/lib/python3.7/site-packages (2.2.5)
    Requirement already satisfied: spacy>=2.2.2 in /anaconda3/lib/python3.7/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
    Requirement already satisfied: setuptools in /anaconda3/lib/python3.7/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (46.1.3.post20200330)
    Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /anaconda3/lib/python3.7/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
    Requirement already satisfied: plac<1.2.0,>=0.9.6 in /anaconda3/lib/python3.7/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
    Requirement already satisfied: numpy>=1.15.0 in /anaconda3/lib/python3.7/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.1)
    Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /anaconda3/lib/python3.7/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
    Requirement already satisfied: blis<0.5.0,>=0.4.0 in /anaconda3/lib/python3.7/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
    Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /anaconda3/lib/python3.7/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
    Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /anaconda3/lib/python3.7/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.0)
    Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /anaconda3/lib/python3.7/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
    Requirement already satisfied: thinc==7.4.0 in /anaconda3/lib/python3.7/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
    Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /anaconda3/lib/python3.7/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
    Requirement already satisfied: requests<3.0.0,>=2.13.0 in /anaconda3/lib/python3.7/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
    Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /anaconda3/lib/python3.7/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
    Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /anaconda3/lib/python3.7/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.0)
    Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /anaconda3/lib/python3.7/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.8)
    Requirement already satisfied: certifi>=2017.4.17 in /anaconda3/lib/python3.7/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.6.20)
    Requirement already satisfied: idna<3,>=2.5 in /anaconda3/lib/python3.7/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
    Requirement already satisfied: chardet<4,>=3.0.2 in /anaconda3/lib/python3.7/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
    Requirement already satisfied: zipp>=0.5 in /anaconda3/lib/python3.7/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
    [38;5;2m✔ Download and installation successful[0m
    You can now load the model via spacy.load('en_core_web_sm')



```python
import spacy

nlp = spacy.load("en_core_web_sm")
```

### Predicting Part-of-speech Tags


```python
import spacy

# Load the small English model
nlp = spacy.load("en_core_web_sm")

# Process a text
doc = nlp("She ate the pizza")

# Iterate over the tokens
for token in doc:
    # Print the text and the predicted part-of-speech tag
    print(token.text, token.pos_)

```

    She PRON
    ate VERB
    the DET
    pizza NOUN


### Predicting Syntactic Dependencies



```python
for token in doc:
    print(token.text, token.pos_, token.dep_, token.head.text)
```

    She PRON nsubj ate
    ate VERB ROOT ate
    the DET det pizza
    pizza NOUN dobj ate


### Dependency label scheme

<img src = "https://course.spacy.io/dep_example.png" width = "70%">

Label  | Description | Example
-------------------- | --------------------- | ---------------------
nsubj | nominal subject | She
dobj | direct object | pizza
det	| determiner (article) | the


### Predicting Named Entities

<img src = "https://course.spacy.io/ner_example.png" width = "70%">



```python
# Process a text
doc = nlp("Apple is looking at buying U.K. startup for $1 billion")

# Iterate over the predicted entities
for ent in doc.ents:
    # Print the entity text and its label
    print(ent.text, ent.label_)

```

    Apple ORG
    U.K. GPE
    $1 billion MONEY
    
    GPE = Geopolitical Entity


### Tip: the spacy.explain method

Get quick definitions of the most common tags and labels.


```python
print("\nGPE = Geopolitical Entity")
spacy.explain("GPE")
```

    
    GPE = Geopolitical Entity





    'Countries, cities, states'




```python
spacy.explain("ORG")
```




    'Companies, agencies, institutions, etc.'




```python
spacy.explain("NNP")
```




    'noun, proper singular'




```python
spacy.explain("dobj")
```




    'direct object'


