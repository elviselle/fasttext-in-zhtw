# coding: utf-8
from gensim.corpora import WikiCorpus
# jieba斷詞
import jieba 
# 簡轉繁工具
from opencc import OpenCC


# Load data
wiki_corpus = WikiCorpus('../wiki_data/zhwiki-20220101-pages-articles-multistream.xml.bz2', dictionary={})


# Save data
with open('wiki_text.txt', 'w', encoding='utf-8') as f:
    print('Start to preprocess.')
    for times, text in enumerate(wiki_corpus.get_texts()):
        article = ' '.join(text)
        f.write(article+'\n')

        if (times+1) % 10000 == 0:
            print(times+1)

