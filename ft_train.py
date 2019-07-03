##### Install Fasttext #######
# $ git clone https://github.com/facebookresearch/fastText.git
# $ cd fastText
# $ pip install .

import fastText
import numpy as np
import pandas as pd


# train model
from fastText import FastText
model = FastText.train_unsupervised(input="data/BGT_Titles_2.txt", dim = 128, epoch=20, minCount=2,wordNgrams=3, loss = "ns", minn=2, maxn=5, thread=1, ws=3)

#save model
model.save_model('model/ns2520ns3005_new.bin')

with open("data/list_titles_BGT.txt", "rb") as fp:   # Unpickling
      list_titles = pickle.load(fp)

#generate emebeddings
p_name_fasttext_128 = np.zeros((len(list_titles), 128))
with open('data/BGT_Titles_2.txt', 'r') as f:
    for i, name in enumerate(f):
        p_name_fasttext_128[i] = model.get_sentence_vector(name.rstrip('\n'))

#save embeddings
np.save('model/ft_embeddings_ns2520ns3005_new.npy', p_name_fasttext_128)
