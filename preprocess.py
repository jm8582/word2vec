from typing import Dict, List

class Preprocessor:
    def __init__(self, dset_name='synthetic'):
        self.dset_name = dset_name

        self.corpus: List[List[str]]
        self.corpus_words: List[str]
        self.n_corpus_words: int
        self.word2idx: Dict
        self.idx2word: Dict
        
        self.sentences = self._load_dset(dset_name)
        
    def _load_dset(self, dset_name):
        if dset_name == 'synthetic':
            sentences = [
                'he is a king',
                'she is a queen',
                'he is a man',
                'she is a woman',
                'warsaw is poland capital',
                'berlin is germany capital',
                'paris is france capital',
            ]
        return sentences
    
    def preprocess(self):
        self.corpus = self._sentences_to_corpus(self.sentences)
        self.corpus_words, self.n_corpus_words = self._distinct_words(self.corpus)
        self.word2idx = {word: idx for idx, word in enumerate(self.corpus_words)}
        self.idx2word = {idx: word for idx, word in enumerate(self.corpus_words)}
        self.corpus_num = []
        for sentence in self.corpus:
            self.corpus_num.append([self.word2idx[word] for word in sentence])
    
    def _sentences_to_corpus(self, sentences):
        corpus = [sentence.lower().split() for sentence in sentences]
        return corpus
    
    def _distinct_words(self, corpus):
        corpus_words = set()
        for line in corpus:
            corpus_words |= set(line)
        corpus_words = sorted(list(corpus_words))
        n_corpus_words = len(corpus_words)
        return corpus_words, n_corpus_words
