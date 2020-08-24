# Data Structures (1): Vocab, Lexemes and StringStore

## Shared vocab and string store
Vocab: stores data shared across multiple documents
* To save memory, spaCy encodes all strings to hash values
* Strings are only stored once in the StringStore via nlp.vocab.strings
* String store: lookup table in both directions



```python
import spacy
from spacy.lang.en import English

nlp = English()

```


```python
coffee_hash = nlp.vocab.strings["coffee"]
print("hash value:", coffee_hash)

```

    hash value: 3197928453018144401


#### Hashes can't be reversed – that's why we need to provide the shared vocab



```python
# Raises an error if we haven't seen the string before
coffee_string = nlp.vocab.strings[coffee_hash]
# Or
string = nlp.vocab.strings[3197928453018144401]
```


    ---------------------------------------------------------------------------

    KeyError                                  Traceback (most recent call last)

    <ipython-input-47-ac4da5aa3af7> in <module>
          1 # Raises an error if we haven't seen the string before
    ----> 2 coffee_string = nlp.vocab.strings[coffee_hash]
          3 # Or
          4 string = nlp.vocab.strings[3197928453018144401]


    strings.pyx in spacy.strings.StringStore.__getitem__()


    KeyError: "[E018] Can't retrieve string for hash '3197928453018144401'. This usually refers to an issue with the `Vocab` or `StringStore`."



```python
doc = nlp("I love coffee and Tea")
```


```python
# Now it will not give any error
coffee_string = nlp.vocab.strings[coffee_hash]
# Or
string = nlp.vocab.strings[3197928453018144401]
```


```python
print(coffee_string)
print(string)

```

    coffee
    coffee


* The doc also exposes the vocab and strings


```python
print("hash value:", doc.vocab.strings["coffee"])
```

    hash value: 3197928453018144401


## Lexemes: entries in the vocabulary
* A Lexeme object is an entry in the vocabulary
* Contains the context-independent information about a word
    * Word text: lexeme.text and lexeme.orth (the hash)
    * Lexical attributes like lexeme.is_alpha
    * Lexemes don't have part-of-speech tags, dependencies or entity labels. Those depend on the context.


```python
doc = nlp("I love coffee")
lexeme = nlp.vocab["coffee"]

# Print the lexical attributes
print(lexeme.text, lexeme.orth, lexeme.is_alpha)
```

    coffee 3197928453018144401 True


## Vocab, hashes and lexemes

<img src = "https://course.spacy.io/vocab_stringstore.png" width = "50%">
