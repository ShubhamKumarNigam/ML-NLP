### Why not just regular expressions?

* Compared to regular expressions, the matcher works with Doc and Token objects instead of only strings.

* It's also more flexible: you can search for texts but also other lexical attributes.

* You can even write rules that use the model's predictions.

* For example, find the word "duck" only if it's a verb, not a noun.

### Match patterns
* Lists of dictionaries, 
* one per token
    * The keys are the names of token attributes, mapped to their expected values.

Match exact token texts: In this example, we're looking for two tokens with the text "iPhone" and "X".

         [{"TEXT": "iPhone"}, {"TEXT": "X"}]

Match lexical attributes: We can also match on other token attributes. Here, we're looking for two tokens whose lowercase forms equal "iphone" and "x".

        [{"LOWER": "iphone"}, {"LOWER": "x"}]

Match any token attributes: pattern would match phrases like "buying milk" or "bought flowers".

        [{"LEMMA": "buy"}, {"POS": "NOUN"}]

### Using the Matcher


```python
import spacy

# Import the Matcher
from spacy.matcher import Matcher

# Load a model and create the nlp object
nlp = spacy.load("en_core_web_sm")

# Initialize the matcher with the shared vocab
matcher = Matcher(nlp.vocab)

# Add the pattern to the matcher
pattern = [{"TEXT": "iPhone"}, {"TEXT": "X"}]
matcher.add("IPHONE_PATTERN", None, pattern)

'''
The matcher.add method lets you add a pattern. 
The first argument is a unique ID to identify which pattern was matched. 
The second argument is an optional callback. 
We don't need one here, so we set it to None. 
The third argument is the pattern.
'''

# Process some text
doc = nlp("Upcoming iPhone X release date leaked")

# Call the matcher on the doc
matches = matcher(doc)

# Iterate over the matches
for match_id, start, end in matches:
    print("match_id",match_id)
    print("start",start)
    print("end",end)
    
    '''
    match_id: hash value of the pattern name
    start: start index of matched span
    end: end index of matched span

    '''
    # Get the matched span
    matched_span = doc[start:end]
    print(matched_span.text)
    


```

    match_id 9528407286733565721
    start 1
    end 3
    iPhone X


### Matching lexical attributes


```python
'''
We're looking for five tokens:

A token consisting of only digits.

Three case-insensitive tokens for "fifa", "world" and "cup".

And a token that consists of punctuation.
'''
pattern = [
    {"IS_DIGIT": True},
    {"LOWER": "fifa"},
    {"LOWER": "world"},
    {"LOWER": "cup"},
    {"IS_PUNCT": True}
]

matcher.add("FIFA", None, pattern)

doc = nlp("2018 FIFA World Cup!!! France won!")

matches = matcher(doc)

for match_id, start, end in matches:
    
    # Get the matched span
    matched_span = doc[start:end]
    print(matched_span.text)
```

    2018 FIFA World Cup!


### Matching other token attributes



```python
'''
A verb with the lemma "love", followed by a noun.
'''
pattern = [
    {"LEMMA": "love", "POS": "VERB"},
    {"POS": "NOUN"}
]

matcher.add("LOVE", None, pattern)

doc = nlp("I loved dogs but now I love cats more.")

matches = matcher(doc)

for match_id, start, end in matches:
    matched_span = doc[start:end]
    print(matched_span.text)
```

    loved dogs
    love cats


### Using operators and quantifiers


```python
pattern = [
    {"LEMMA": "buy"},
    {"POS": "DET", "OP": "?"},  # optional: match 0 or 1 times
    {"POS": "NOUN"}
]

matcher.add("SMARTPHONE", None, pattern)

doc = nlp("I bought a smartphone. Now I'm buying apps.")

matches = matcher(doc)

for match_id, start, end in matches:
    match_span = doc[start:end]
    print(match_span.text)
```

    bought a smartphone
    buying apps


Example	| Description
-----   | ------
{"OP": "!"}	| Negation: match 0 times
{"OP": "?"}	| Optional: match 0 or 1 times
{"OP": "+"}	| Match 1 or more times
{"OP": "*"}	| Match 0 or more times

