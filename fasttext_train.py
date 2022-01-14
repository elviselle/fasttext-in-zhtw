# coding: utf-8
from gensim.models import word2vec, fasttext
from gensim.test.utils import get_tmpfile
from gensim.models.callbacks import CallbackAny2Vec
import time

class EpochLogger(CallbackAny2Vec):
    '''Callback to log information about training'''
    def __init__(self):
        self.epoch = 0

    def on_epoch_begin(self, model):
        print("Epoch #{} start".format(self.epoch))

    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        print('Loss after epoch {}: {}'.format(self.epoch, loss))
        print("Epoch #{} end".format(self.epoch))
        self.epoch += 1

epoch_logger = EpochLogger()

# Settings
seed = 666
sg = 0
window_size = 20
vector_size = 100
min_count = 500
workers = 12
epochs = 5
batch_words = 10000

start = time.time()
# Train
train_data = word2vec.LineSentence('jieba_seg/wiki_text_seg_1whitespace.txt')
model = fasttext.FastText(
    train_data,
    min_count=min_count,
    vector_size=vector_size,
    workers=workers,
    epochs=epochs,
    window=window_size,
    sg=sg,
    seed=seed,
    batch_words=batch_words
)
end = time.time()
print('Training spent time:', (end-start), 'sec.')

start = time.time()
model.save('fasttext_model/fasttext.model')
end = time.time()
print('Saving spent time:', (end-start), 'sec.')
