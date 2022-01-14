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
        f.write(' '.join(text)+'\n')

        if (times+1) % 10000 == 0:
            print(times+1)


# Initial
cc = OpenCC('s2t')

# Tokenize
with open('jieba_seg/wiki_text_seg.txt', 'w', encoding='utf-8') as new_f:
    with open('wiki_text.txt', 'r', encoding='utf-8') as f:
        for data in f:
            data = cc.convert(data)
            data = jieba.cut(data)
            data = [word for word in data if word != ' ']
            data = ' '.join(data)

            new_f.write(data)
