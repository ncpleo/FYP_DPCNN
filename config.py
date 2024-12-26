# —*- coding: utf-8 -*-


class Config(object):
    def __init__(self, 
                 word_embedding_dimension=300, 
                 word_num=200000,
                 epoch=1000, 
                 sentence_max_size=50,
                 learning_rate=0.0001, 
                 batch_size=64,
                 drop_out=0.6,
                 dict_size=50000,
                 bidirectional=False,
                 doc_len=40,
                 label_num=2
                 ):
        self.word_embedding_dimension = word_embedding_dimension
        self.word_num = word_num
        self.epoch = epoch
        self.sentence_max_size = sentence_max_size                   # 句子长度
        self.lr = learning_rate
        self.batch_size = batch_size
        self.dict_size = dict_size
        self.drop_out = drop_out
        self.bidirectional = bidirectional
        self.doc_len = doc_len
        self.label_num = label_num
        #self.cuda = cuda

