## Reference: https://course.spacy.io/en/chapter1

### The nlp object


```python
# Import the English language class
from spacy.lang.en import English

# Create the nlp object
nlp = English()
```


```python
nlp
```




    <spacy.lang.en.English at 0x117220290>



### The Doc object


```python
# Created by processing a string of text with the nlp object
doc = nlp("Hello world!")

print("doc contains the string is:", doc)

# Iterate over tokens in a Doc
for token in doc:
    print(token.text)
```

    doc contains the string is: Hello world!
    Hello
    world
    !


### The Token object


```python
# Index into the Doc to get a single Token
token = doc[1]
print(token)
# Get the token text via the .text attribute
print(token.text)
```

    world
    world


### The Span object


```python
# A slice from the Doc is a Span object
span = doc[1:3]

# Get the span text via the .text attribute
print(span.text)
```

    world!


### Lexical Attributes


```python
doc = nlp("It costs $53, not $35.")

print("Index:   ", [token.i for token in doc])
print("Text:    ", [token.text for token in doc])

print("is_alpha:", [token.is_alpha for token in doc]) # is_alpha return True if string is alphabet
print("is_punct:", [token.is_punct for token in doc]) # is_punct return True if string is punctuation
print("like_num:", [token.like_num for token in doc]) # like_num return True if string is number
```

    Index:    [0, 1, 2, 3, 4, 5, 6, 7, 8]
    Text:     ['It', 'costs', '$', '53', ',', 'not', '$', '35', '.']
    is_alpha: [True, True, False, False, False, True, False, False, False]
    is_punct: [False, False, False, False, True, False, False, False, True]
    like_num: [False, False, False, True, False, False, False, True, False]



```python

```
