"""
word2vec transformer using gensim (https://radimrehurek.com/gensim/index.html)
"""
import os
import json
import numpy as np
from time import time
from gensim.utils import simple_preprocess
from gensim.models import Word2Vec
from gensim.models.phrases import Phraser, Phrases
from .utils import get_sentence_generator

class Word2vec(object):
    """
    Main class for word2vec operations
    """
    def __init__(self, path, files_prefix):
        """
        Parameters
        -----------
        path : string
            Path to save generated files (embeddings and vocab).
        files_prefix : string
            prefix_embeddings.npz, prefix_vocab.json, prefix_model.gs, prefix_phrases.gs
        """
        self.path = path
        self.files_prefix = files_prefix
        self.embeddings_file = os.path.join(path,'%s_embeddings.npz' % files_prefix)
        self.phrases_file = os.path.join(path,'%s_phrases.gs' % files_prefix)
        self.vocab_file = os.path.join(path,'%s_vocab.json' % files_prefix)
        self.model_file = os.path.join(path,'%s_model.gs' % files_prefix)

    def create_embeddings(self, text_file, detect_phrases=False, generate_files=True,
                          mode='simple', content='file', **kwargs):
        """
        Generate embeddings from a batch of text

        Parameters
        -----------
        text_file : string
            Text file location.
        detect_phrases : bool
            If true, word2vec will detect bigrams.
        generate_files : bool
            If true, files will be saved.
        mode : string
            See get_sentence_generator.
        content : string
            'file': load one file.
            'dir': load all files on the specified directory.
        **kwargs: Word2Vec arguments.
        """
        assert os.path.exists(text_file), 'File not found: %s' % text_file

        print('Generating embeddings...')
        t0 = time()

        sentences = get_sentence_generator(mode, text_file, content=content)

        if detect_phrases:
            bigram_transformer = Phrases(sentences)
            bigram_transformer = Phraser(bigram_transformer)
            bigram_transformer.save(self.phrases_file)
            print('Getting phrases: %fs' % (time()-t0))
            t0 = time()
            model = Word2Vec(bigram_transformer[sentences], **kwargs)
            self.phrase_transformer = bigram_transformer
        else:
            model = Word2Vec(sentences, **kwargs)

        self.model = model
        self.word2idx = dict([(k, v.index) for k, v in model.vocab.items()])
        self.idx2word = dict([(v.index, k) for k, v in model.vocab.items()])

        # save files
        if generate_files:
            if not os.path.exists(self.path):
                os.mkdir(self.path)
            # save embeddings weights to file
            weights = model.syn0
            np.save(open(self.embeddings_file, 'wb'), weights)
            # save vocab to file
            with open(self.vocab_file, 'w') as f:
                f.write(json.dumps(self.word2idx))
            # save model to file
            model.save(self.model_file)

        print('Training time: %fs' % (time()-t0))
        
        if generate_files:
            print()
            print('Model file: %s' % self.model_file)
            print('Embeddings file: %s' % self.embeddings_file)
            print('Vocab file: %s' % self.vocab_file)
            if detect_phrases:
                print('Phrases file: %s' % self.phrases_file)

    def load_embeddings(self, detect_phrases=False):
        """
        Load embeddings from file
        
        Parameters
        -----------
        detect_phrases : bool
            Indicates if the model was created with detect_phrases=True.
        """
        for f in (self.embeddings_file, self.vocab_file, self.model_file):
            assert os.path.exists(f), 'File not found: %s' % f

        if detect_phrases:
            assert os.path.exists(self.phrases_file), 'File not found: %s' % self.phrases_file
            bigram_transformer = Phraser(Phrases())
            self.phrase_transformer = bigram_transformer.load(self.phrases_file)

        model = Word2Vec()
        self.model = model.load(self.model_file)
        self.word2idx = dict([(k, v.index) for k, v in self.model.vocab.items()])
        self.idx2word = dict([(v.index, k) for k, v in self.model.vocab.items()])

    def build_manifold(self, words_limit=500):
        """
        Generate low-dimensional embeddings using t-SNE
        
        Parameters
        -----------
        words_limit : int
            Amount of words to include on the manifold.
        """
        print('Building manifold...')

        try:
            from sklearn.manifold import TSNE
            t0 = time()
            tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
            self.tsne_embs = tsne.fit_transform(self.model.syn0[:words_limit,:])
            self.tsne_labels = np.array([self.idx2word[i] for i in range(words_limit)])
            self.tsne_size = words_limit
            print('TSNE time: %.2fs' % (time()-t0))
        except ImportError:
            print('Please install sklearn')

    def _plot(self, cords, labels, fig_size, dot_size, text_size):
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=fig_size)
            for i, label in enumerate(labels):
                x, y = cords[i,:]
                plt.scatter(x, y, dot_size)
                plt.annotate(label, size=text_size, xy=(x, y), xytext=(5, 2),
                             textcoords='offset points',ha='right', va='bottom')
            plt.show()
        except ImportError:
            print('Please install matplotlib')

    def _plot_with_arrows(self, cords1, labels1, cords2, labels2, fig_size, dot_size,
                          text_size):
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=fig_size)
            for i in range(len(labels1)):
                x1, y1 = cords1[i,:]
                x2, y2 = cords2[i,:]
                plt.scatter(x1, y1, dot_size)
                plt.scatter(x2, y2, dot_size)
                plt.annotate(labels1[i], size=text_size, xy=(x1, y1), xytext=(5, 2),
                             textcoords='offset points',ha='right', va='bottom')
                plt.annotate(labels2[i], size=text_size, xy=(x2, y2), xytext=(5, 2),
                             textcoords='offset points',ha='right', va='bottom')
                plt.plot([x1, x2], [y1, y2], ':b')
            plt.show()
        except ImportError:
            print('Please install matplotlib')

    def plot(self, words_plot=200, fig_size=(18,18), dot_size=20, text_size=14,
             random_words=True):
        """
        Visualize embeddings

        Parameters
        -----------
        words_plot : int
            Number of words to plot.
        fig_size : tuple of integers
            Width, height in inches.
        dot_size : int
            Dot size on the scatter plot.
        text_size : int
            Text size on plot annotations.
        random_words : bool
            Plot random words.
        """
        try:
            self.tsne_labels
            self.tsne_embs
        except NameError:
            print('You need to run Word2vec.build_manifold()')

        assert len(self.tsne_labels) >= words_plot, 'words_plot can\'t be bigger than manifold\'s words_limit'

        if random_words:
            choice = np.random.choice(range(len(self.tsne_labels)), words_plot, replace=False)
            cords = self.tsne_embs[choice]
            labels = self.tsne_labels[choice]
        else:
            cords = self.tsne_embs[:words_plot]
            labels = self.tsne_labels[:words_plot]

        self._plot(cords, labels, fig_size, dot_size, text_size)

    def plot_words(self, words=[], fig_size=(5,5), dot_size=20, text_size=14):
        """
        Visualize embeddings

        Parameters
        -----------
        words : list of strings
            Words to plot.
        fig_size : tuple of integers
            Width, height in inches.
        dot_size : int
            Dot size on the scatter plot.
        text_size : int
            Text size on plot annotations.
        """
        idx = [self.word2idx[w] for w in words]
        idx = [i for i in idx if i < len(self.tsne_labels)]
        labels = self.tsne_labels[idx]
        cords = self.tsne_embs[idx]
        for w in words:
            if w not in labels:
                print('Index %d of word \'%s\' is out of the embedding size (%d)' % 
                      (self.word2idx[w], w, self.tsne_size))
        self._plot(cords, labels, fig_size, dot_size, text_size)

    def plot_word_groups(self, words1=[], words2=[], fig_size=(5,5), dot_size=20,
                         text_size=14):
        """
        Visualize embeddings

        Parameters
        -----------
        words1: list of strings
            Words to plot.
        words2: list of strings
            Words to plot.
        fig_size : tuple of integers
            Width, height in inches.
        dot_size : int
            Dot size on the scatter plot.
        text_size : int
            Text size on plot annotations.
        """
        assert len(words1)==len(words2), 'words1 and words2 must have the same length'

        idx1 = [self.word2idx[w] for w in words1]
        cords1 = self.tsne_embs[idx1]
        idx2 = [self.word2idx[w] for w in words2]
        cords2 = self.tsne_embs[idx2]
        self._plot_with_arrows(cords1, words1, cords2, words2, fig_size, dot_size, text_size)

