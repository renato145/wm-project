"""
Utilities to generate text from a model
"""
import numpy as np
from .utils import parse_text
from keras.preprocessing.sequence import pad_sequences

def format_sample(x, seq_len, embeddings, word2idx=None):
    """
    Returns the single sample formatted to be used by a model
    
    Parameters
    -----------
    x : list of ints or strings
        Inputs to predict next word.
    seq_len : int
        Sequence length (number of words used to predict the next).
    embeddings : np.array
        Word2vec embeddings.
    word2idx : dict
        Word2idx dictionary, must be given if x is a list of strings.
        
    Returns
    -----------
    out : np.array
        Array of shape [1, seq_len, input_dim] to be used as model input.
    """
    x = x[:seq_len]
        
    if type(x[0]) == str:
        assert word2idx, 'Especify a word2idx dictionary.'
        x = [word2idx[word] for word in x[:seq_len]]
    
    input_dim = embeddings.shape[1]
    
    out = embeddings[x].reshape([1, -1, input_dim])
    out = pad_sequences(out, maxlen=seq_len, dtype='float32')
    
    return out

def temperature_sample(preds, temperature=1.0):
    """
    Helper function to sample an index from a probability array
    
    Parameters
    -----------
    preds : np.array
        Probability array.
    temperature : float
        The bigger, the sample will be more diverse.
        
    Returns
    -----------
    out : int
        Sample chosen from the array.
    """
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    
    return np.argmax(probas)

def generate_from_model(x, model, embeddings, idx2word, word2idx, n_words=20,
                        temperature=1, truncating='pre', custom_text=False,
                        mode='simple'):
    """
    Returns the single sample formatted to be used by a model
    
    Parameters
    -----------
    x : list of strings or string
        Inputs to predict next word.
        If custom_text=True, a string type should be given.
    model : keras.Model
        RNN trained model from keras.
    embeddings : np.array
        Word2vec embeddings.
    idx2word : dict
        Idx2word dictionary.
    word2idx : dict
        Word2idx dictionary.
    n_words : int
        Number of words to generate.
    temperature : float
        The bigger, the sample will be more diverse.
    truncating : string
        'pre' or 'post', remove values from sequences larger than
        seq_len either in the beginning or in the end of the sequence.
    custom_text : bool
        If True, the text will be parsed.
    mode : string
        Tokenizer for text (only apply if custom_text is True).
        'simple': min_len=2 and max_len=15.
        'spanish_g': Generate vectors ready to do text generation on spanish.
        
    Returns
    -----------
    out : np.array
        Array of shape [1, seq_len, input_dim] to be used as model input.
    """
    if custom_text:
        x = parse_text(x, word2idx, mode)
    
    seq_len = model.input_shape[1]
    
    if truncating == 'pre':
        x = x[:seq_len]
    elif truncating == 'post':
        x = x[-seq_len:]
    else:
        raise NameError('Invalid truncating option: %s.' % truncating)
        
    x = [word2idx[word] for word in x]
    
    for idx in range(n_words):
        model_input = x[idx:idx + seq_len]
        model_input = format_sample(model_input, seq_len, embeddings)
        model_output = model.predict(model_input)
        model_output = model_output.reshape([-1])
        model_output = temperature_sample(model_output, temperature)
        x.append(model_output)
    
    out = [idx2word[idx] for idx in x]
    
    return out
