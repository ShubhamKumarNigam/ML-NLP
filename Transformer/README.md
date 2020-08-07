## 1. Why Transformers Needed?
Inspired from [LSTM is dead. Long Live Transformers!](https://www.youtube.com/watch?v=S27pHKBEp30)

        Bag of Words ==> RNN ==> LSTM (RNN) ==> Transformer
        
### Bag of words:
* one dimension per word in vocabulary
* if d = 10000

#### Problems:
* Almost all values are zero
   * 'coz most words are not present in the doc.
   * creates the __Sparsity__ problem
   * __solution__ is, don't store zeros.
          
          List(Tuple(position:int, value:float))
* Order matters!
  * eg. "work to live" v/s "live to work"
  * bag of words model will score them identically every single time because they have the exact same vectors.
  * __solution__ is, __N grams__ (Dimentionality V<sup>N</sup>)
  
### RNN (Recurrent Neural Network):
* Vanishing / Exploding Gradients problem.
* solution is LSTM.

### LSTM (Long Short-Term Memory):
* Adding new values on to the activation as you go through the layers, so this solves the exploding and vanishing gradients problems.

#### Problems:
* Difficult to train becasue very long gradient paths
   * LSTM on 100-word doc has gradient like 100-layer network
* Transfer learning never really worked
* Needs specific labelled dataset for every task

### Transformer:

#### Attention Mechanism:
* All-to-all comparision
  * Each layer is O(N<sup>2</sup>) for sequemce of length N
* Every output is weighted sum of every input
  * The weighting is a learned function
  
        Q: Query (Output token)
        K: Key (Input token)
        
        Relevance = Q*K
        
        V: Value (Input token)
        out = Softmax(relevance) * V

#### Multi-headed Attention:
* Every head learns. different semantic meaning of attention:
  * Eg. one for grammar, one for vocabulary, one for conjugation, etc.
  
#### Positional Encoding:
* Without positional encoding, Attention is "bag of words"
* Input Layer: add a word embedding and a position embedding
* position can be either learned or fixed.
* Fixed allows extrapolating to longer sequences

#### Why Transformers are awesome
* All-to-all comparisions can be done fully parallel
* GPUs. and TPUs helps to compute easily
* where as RNN/LSTM must be computed in serial per token
* don't need to use any of these sigmoid or tanh activation functions which are built into the LSTM model.
* problematic because if get a neuron which has a very high activation value,
* then you've got 1 and you take the derivative of that and it's 0 or it's some very very small number 
* and so your gradient descent can't tell the difference
* between an activation high or down other side
* so it's very easy for the trainer to get confused if your activations don't stay near middle part.

#### Why ReLU works better?
* Gradient doesn't saturate (on the high end)
* Less sensitive to the random initialization

##### Downsides:
* "Dead Neurons " == always output zero (fixed with Leaky ReLU)
* Gradient discountinous at origin (fixed with GELU)

### LSTM still good when:
* Sequence length is long or infinity. Transformers are N<sup>2</sup>
* E.g. real-time control for robotics or similar
* can't pre-train on large corpus

- - - -
