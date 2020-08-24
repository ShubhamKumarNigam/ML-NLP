## Cleaning and Tokenization matters

* __Letter Case__ "The" == "the" == "THE" ``` --> lowercase/uppercase ```

* __Punctuation__ , ! . ? ``` --> . ``` 

* __Numbers__ 1 2 3 4 5 ``` --> Φ ``` but not 3.14159 or pincodes etc.

* __Special Characters__ © $ ∆ ``` --> Φ ```

* __Special Words__ Emoji(😊), Hashtags(#nlp) ``` --> :happy:, #nlp ```


```python
text = 'Who ❤️ "word embeddings" in 2020? I do!!!'
print(text)
```

    Who ❤️ "word embeddings" in 2020? I do!!!



```python
# !pip install nltk
!pip install emoji
```

    Collecting emoji
      Downloading emoji-0.6.0.tar.gz (51 kB)
    [K     |████████████████████████████████| 51 kB 326 kB/s eta 0:00:01
    [?25hBuilding wheels for collected packages: emoji
      Building wheel for emoji (setup.py) ... [?25ldone
    [?25h  Created wheel for emoji: filename=emoji-0.6.0-py3-none-any.whl size=49714 sha256=89e7516b795eaf9ff375cbff3eadc92812695755d138c47fb821464bda35690d
      Stored in directory: /Users/shubhamkumarnigam/Library/Caches/pip/wheels/4e/bf/6b/2e22b3708d14bf6384f862db539b044d6931bd6b14ad3c9adc
    Successfully built emoji
    Installing collected packages: emoji
    Successfully installed emoji-0.6.0



```python
import nltk
from nltk.tokenize import word_tokenize
import emoji
import re

# nltk.download('punkt')
```


```python
data = re.sub(r'[,!?;-]+', '.', text) #substitute special symbols to .
```


```python
print(data)
```

    Who ❤️ "word embeddings" in 2020. I do.



```python
data = nltk.word_tokenize(data) # tokenize string into words
```


```python
print(data)
```

    ['Who', '❤️', '``', 'word', 'embeddings', "''", 'in', '2020', '.', 'I', 'do', '.']



```python
data = [ ch.lower() for ch in data
        if ch.isalpha()
        or ch == '.'
        or emoji.get_emoji_regexp().search(ch)
    ] # get text if token is alphabet, dot or emoji and remove numbers
```


```python
print(data)
```

    ['who', '❤️', 'word', 'embeddings', 'in', '.', 'i', 'do', '.']



```python

```
