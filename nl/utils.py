"""
Utilities
"""
import os
import json
import random
import numpy as np
from time import time

COMMON_CHARS_REPLACE = (
    # (search_string, replace_string, inverse_use)
    ('\n', '', False),
    ('[', '{', False),
    (']', '}', False),
    ('{', ' [bbkt] ', True),
    ('}', ' [ebkt] ', True),
    ('.', ' [dot] ', True),
    ('¡', ' [bxcm] ', True),
    ('!', ' [excm] ', True),
    ('¿', ' [bq] ', True),
    ('?', ' [eq] ', True),
    (',', ' [comma] ', True),
    ('$', ' [dlr] ', True),
    (':', ' [dd] ', True),
    (';', ' [dc] ', True),
    ('%', ' [pcnt] ', True),
    ('(', ' [bpar] ', True),
    (')', ' [epar] ', True),
    ('"', ' [dquote] ', True),
    ('\'', ' [squote] ', True),
    ('   ', ' ', False),
    ('  ', ' ', False)
)

COMMON_FORMAT_REPLACE = (
    ('  ', ' '),
    (' .', '.'),
    ('¡ ', '¡'),
    (' !', '!'),
    ('¿ ', '¿'),
    (' ?', '?'),
    (' ,', ','),
    ('$ ', '$'),
    (' :', ':'),
    (' ;', ';'),
    (' %', '%'),
    ('{', '['),
    ('}', ']'),
    ('[ ', '['),
    (' ]', ']'),
    ('( ', '('),
    (' )', ')')
)

class _SentenceGenerator_simple(object):
    def __init__(self, file, custom_text=False):
        self.file = file
        self.custom_text = custom_text
        self.tokenize = lambda x: simple_preprocess(x)

    def __iter__(self):
        if self.custom_text:
            yield self.tokenize(self.file)
        else:
            for line in open(self.file):
                yield self.tokenize(line)

class _SentenceGenerator_spanish_g(object):
    def __init__(self, file, custom_text=False):
        self.file = file
        self.custom_text = custom_text
        
    def tokenize(self, text):
        text = text.lower()
        for a, b, _ in COMMON_CHARS_REPLACE:
            text = text.replace(a,b)
        text = text.strip()
        text = text.split(' ')
        if '' in text:
            text.remove('')
        if len(text) > 0:
            if text[-1] == '[dot]':
                text[-1] = '[Edot]'
        return text
        
    def __iter__(self):
        if self.custom_text:
            yield self.tokenize(self.file)
        else:
            for line in open(self.file):
                yield self.tokenize(line)

def read_file_lines(file_path, encoding=None, skip_lines=0):
    """
    Read a text file and yields line by line
    
    Parameters
    -----------
    file_path : string
        Text file location.
    encoding : string
        Encoding to read the file (for spanish texts use 'ISO-8859-1').
    skip_lines : int
        Number of lines to skip.
        
    Returns
    -----------
    line : string
        Yields the readed line.
    """
    assert os.path.exists(file_path), 'File %s not found' % file_path
    
    for line in open(file_path, encoding=encoding):
        if skip_lines != 0:
            skip_lines -= 1
        else:
            yield line

def print_lines(text, n=5):
    """
    Prings n lines from a list of strings
    
    Parameters
    -----------
    text : list of strings
        Lines from a text.
    n : int
        Number of lines to print.
    """
    for i,line in enumerate(text):
        if i >= n: break
        print(line)

def print_word_list(word_list):
    text = ''
    for word in word_list:
        text += ' %s ' % word
    
    text = text.replace(' [Edot] ', '.\n')
    
    for a, b, use in COMMON_CHARS_REPLACE:
        if use:
            a = '%s ' % a
            text = text.replace(b, a)
    
    # temporal fix
    for a, b, use in COMMON_CHARS_REPLACE:
        if use:
            a = '%s ' % a
            b = b.strip()
            text = text.replace(b, a)
    
    for a, b in COMMON_FORMAT_REPLACE:
        text = text.replace(a, b)

    text = text.strip()
    print(text)
        
def load_w2v_data(json_file, np_file):
    """
    Load embeddings and vocab dictionaries generated from Word2vec
    
    Parameters
    -----------
    json_file : string
        json file path.
    np_file : string
        numpy embeddings file path.
        
    Returns
    -----------
    out : tuple (dict, dict, np.array)
        (word2idx, idx2word, embeddings)
    """
    assert os.path.exists(json_file), 'File %s not found' % json_file
    assert os.path.exists(np_file), 'File %s not found' % np_file
    
    embeddings = np.load(open(np_file, 'rb'))
    
    with open(json_file, 'r') as f:
        word2idx = json.loads(f.read())
    idx2word = {i:k for k,i in word2idx.items()}
    
    out = (word2idx, idx2word, embeddings)
    
    return out

def generate_train_data(word2idx, text_file, seq_len=10, mode='simple'):
    """
    Returns data to generate text
    
    Parameters
    -----------
    word2idx : dict
        Word2idx dictionary.
    text_file : string
        Text file location.
    seq_len : int
        Sequence length (number of words used to predict the next).
    mode : string
        Tokenizer for text.
        'simple': min_len=2 and max_len=15.
        'spanish_g': Generate vectors ready to do text generation on spanish.
    
    Returns
    -----------
    out : tuple of np.arrays
        (inputs, labels)
    """
    assert os.path.exists(text_file), 'File not found: %s' % text_file
    print('Generating train data...')
    t0 = time()
    
    # generate model
    if mode == 'simple':
        sentences = _SentenceGenerator_simple(text_file)
    elif mode == 'spanish_g':
        sentences = _SentenceGenerator_spanish_g(text_file)
    else:
        raise NameError('Invalid mode: %s.' % mode)
    
    text_idxs = []
    
    for line in sentences:
        for token in line:
            text_idxs.append(word2idx[token])
    
    text_len = len(text_idxs)
    seq_d = int(len(text_idxs) / seq_len)
    
    x = []
    y = []
    
    for idx in range(seq_d):
        for iidx in range(seq_len):
            seq_start = idx * seq_len + iidx
            seq_end = (idx + 1) * seq_len + iidx
            if seq_end >= text_len: break
            x.append(text_idxs[seq_start:seq_end])
            y.append(text_idxs[seq_end])
            
    x = np.asarray(x)
    y = np.asarray(y)
    out = (x, y)
    print('%.2fs' % (time() - t0))
    
    return out

class GenerateSamples(object):
    """
    Main class used to generate samples from text file
    """
    def __init__(self, word2idx, text_file, mode='simple', n_samples=5,
                 samples_len=40):
        """
        Parameters
        -----------
        word2idx : dict
            Word2idx dictionary.
        text_file : string
            Text file location.
        mode : string
            Tokenizer for text.
            'simple': min_len=2 and max_len=15.
            'spanish_g': Generate vectors ready to do text generation on spanish.
        n_samples : int
            Number of sames to generate with the iterator.
        samples_len : int
            Number of words for each sample.
        """
        self.text_file = text_file
        self.mode = mode
        self.n_samples = n_samples
        self.samples_len = samples_len
        
        assert os.path.exists(text_file), 'File not found: %s' % text_file

        if mode == 'simple':
            sentences = _SentenceGenerator_simple(text_file)
        elif mode == 'spanish_g':
            sentences = _SentenceGenerator_spanish_g(text_file)
        else:
            raise NameError('Invalid mode: %s.' % mode)

        text_idxs = []
        
        for line in sentences:
            text_idxs.append(line)

        self.text_idxs = text_idxs
        self.n_lines = len(text_idxs)
        
    def __iter__(self):
        for i in range(self.n_samples):
            r = random.randint(0, self.n_lines)
            available_len = len(self.text_idxs[r])
            
            while available_len < self.samples_len:
                r -= 1
                available_len += len(self.text_idxs[r])
            
            sample = self.text_idxs[r]
            if len(sample) > self.samples_len:
                sample = sample[:self.samples_len]
            else:
                r += 1
                while len(sample) < self.samples_len:
                    next_sample = self.text_idxs[r]
                    for idx in next_sample:
                        if len(sample) == self.samples_len: break
                        sample.append(idx)
                    r += 1
                        
            yield sample
            
def parse_text(text, word2idx, mode='simple'):
    """
    Returns a formatted word list from custom text.
    
    Parameters
    -----------
    text : string
        Text to tokenize.
    word2idx : dict
        Word2idx dictionary.
    mode : string
        Tokenize text line by line.
        'simple': min_len=2 and max_len=15.
        'spanish_g': Generate vectors ready to do text generation on spanish.
        
    Returns
    -----------
    out : list of strings
        Tokenized text.
    """
    if mode == 'simple':
        sentences = _SentenceGenerator_simple(text, custom_text=True)
    elif mode == 'spanish_g':
        sentences = _SentenceGenerator_spanish_g(text, custom_text=True)
    else:
        raise NameError('Invalid mode: %s.' % mode)
    
    out = []
    
    for line in sentences:
        for token in line:
            out.append(token)
            
    return out
