## Cleaning and Tokenization


```python
import nltk
from nltk.tokenize import word_tokenize
import emoji
import re
```


```python
text = 'Who ❤️ "word embeddings" in 2020? I do!!!'
```


```python
data = re.sub(r'[,!?;-]+', '.', text) #substitute special symbols to .
data = nltk.word_tokenize(data) # tokenize string into words
data = [ ch.lower() for ch in data
        if ch.isalpha()
        or ch == '.'
        or emoji.get_emoji_regexp().search(ch)
    ] # get text if token is alphabet, dot or emoji and remove numbers
```

## Sliding window of words


```python
def get_windows(words, C):
    i = C
    while i < len(words) - C:
        center_word = words[i]
        context_words = words[(i - C) : i] + words[(i+1) : (i+C+1)]
        yield context_words, center_word
        i += 1
```


```python
print(data,"\n")
for x, y in get_windows(data, 2):
    print(f'{x}\t{y}')
```

    ['who', '❤️', 'word', 'embeddings', 'in', '.', 'i', 'do', '.'] 
    
    ['who', '❤️', 'embeddings', 'in']	word
    ['❤️', 'word', 'in', '.']	embeddings
    ['word', 'embeddings', '.', 'i']	in
    ['embeddings', 'in', 'i', 'do']	.
    ['in', '.', 'do', '.']	i

