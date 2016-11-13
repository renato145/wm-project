"""
Utilities
"""
import os
import re
import json
import random
import numpy as np
from time import time

COMMON_CHARS_REPLACE = (
    # (search_string, replace_string, inverse_use)
    ('[', '{', False),
    (']', '}', False),
    ('–', '—', False),
    ('…', '...', False),
    ('“', '"', False),
    ('”', '"', False),
    ('...', ' [tridot] ', True),
    ('—', ' [guion] ', True),
    ('\n', ' [newline] ', True),
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
    ('*', ' [astr] ', True),
    ('«', ' [bsay] ', True),
    ('»', ' [esay] ', True),
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
    def __init__(self, file, custom_text=False, content='file'):
        self.file = file
        self.custom_text = custom_text
        self.content = content
        if content == 'dir':
            assert os.path.isdir(file), 'Path \'%s\' is not a directory' % file
            self.files = [os.path.join(file,f) for f in os.listdir(file)]

    def __iter__(self):
        if self.custom_text:
            yield self.tokenize(self.file)
        else:
            if self.content == 'file':
                for line in open(self.file):
                    yield self.tokenize(line)
            if self.content == 'dir':
                for file in self.files:
                    for line in open(file):
                        yield self.tokenize(line)
                        
    def tokenize(self, text):
        return simple_preprocess(text)

class _SentenceGenerator_general_1(_SentenceGenerator_simple):
    def tokenize(self, text):
        text = text.lower()
        re_guion = re.compile('^-|\s-')
        text = re_guion.sub(' [guion] ', text)
        for a, b, _ in COMMON_CHARS_REPLACE:
            text = text.replace(a,b)
        text = text.strip()
        text = text.split(' ')
        if '' in text:
            text.remove('')
        if ['[newline]'] == text:
            text.remove('[newline]')
        return text

class _SentenceGenerator_general_2(_SentenceGenerator_simple):
    def tokenize(self, text):
        text = text.lower()
        text = text.replace('\n', '')
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

def get_sentence_generator(mode, text_file, custom_text=False, content='file'):
    """
    Returns a sentence generator class
    
    Parameters
    -----------
    mode : string
        Tokenizer for text.
        'simple': min_len=2 and max_len=15.
        'general_1': general purpose.
        'general_2': use when a paragraph is composed by several lines
                     on the file, it will detect lines by grouping them
                     until it finds a final dot.
    text_file : string
        If true, reads text_file as a string and ignores content option.
    content : string
        'file': load one file.
        'dir': load all files on the specified directory.
    """
    if mode == 'simple':
        Sentences = _SentenceGenerator_simple(text_file, custom_text, content)
    elif mode == 'general_1':
        Sentences = _SentenceGenerator_general_1(text_file, custom_text, content)
    elif mode == 'general_2':
        Sentences = _SentenceGenerator_general_2(text_file, custom_text, content)
    else:
        raise NameError('Invalid mode: %s.' % mode)
        
    return Sentences

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

def generate_train_data(word2idx, text_file, seq_len=10, mode='simple', content='file'):
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
        See get_sentence_generator.
    content : string
        'file': load one file.
        'dir': load all files on the specified directory.
    
    Returns
    -----------
    out : tuple of np.arrays
        (inputs, labels)
    """
    assert os.path.exists(text_file), 'File not found: %s' % text_file
    print('Generating train data...')
    t0 = time()
    
    sentences = get_sentence_generator(mode, text_file, content=content)
    
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
    def __init__(self, word2idx, text_file, mode='simple', content='file', 
                 n_samples=5, samples_len=40):
        """
        Parameters
        -----------
        word2idx : dict
            Word2idx dictionary.
        text_file : string
            Text file location.
        mode : string
            See get_sentence_generator.
        content : string
            'file': load one file.
            'dir': load all files on the specified directory.
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

        sentences = get_sentence_generator(mode, text_file, content=content)

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
        See get_sentence_generator.
        
    Returns
    -----------
    out : list of strings
        Tokenized text.
    """
    sentences = get_sentence_generator(mode, text, custom_text=True)
    
    out = []
    
    for line in sentences:
        for token in line:
            out.append(token)
            
    return out

def shuffle_data(x, y, verbose=False):
    if verbose:
        print('Shuffling data...')
        
    t0 = time()
    shuffle = np.random.choice(range(len(x)), len(x), replace=False)
    x = x[shuffle]
    y = y[shuffle]
    if verbose:
        print('%.2fs' % (time() - t0))
        
    return x, y
