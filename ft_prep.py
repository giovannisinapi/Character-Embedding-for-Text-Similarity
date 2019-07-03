##### Install Fasttext #######
# $ git clone https://github.com/facebookresearch/fastText.git
# $ cd fastText
# $ pip install .

import fastText
import numpy as np
import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
import pickle
#nltk.download('punkt')
stop_words = set(stopwords.words('english'))
stop_words.remove('it')
stop_words.remove('ma')
stop_words.remove('i')
stop_words.remove('is')


list_acr=['M&A','P&C','FP&A', 'H&B','R&D', 'L&D',
          'F&B', 'I&C', 'O&M', 'F&I', 'I&E', 'L&I', 'E&I', 'Pm&R', 'C&A', 'A/C', 'E/O', 'A/V', 'A/B', 'A/C'
         'E-Commerce', 'Co-op', 'On-', 'In-', 'Pre-', 'post-', 'Full-Stack', 'e-', 'co-',
         '.Net', 'c#', 'c++', 'k-']
list_acr = [item.lower() for item in list_acr]

def urlify(s):
    s=s.lower()
    if any(ext in s for ext in list_acr):
        #print ([ext for ext in list_acr if(ext in s)])
        s=s.lower()
        tokens=s.split()
        #print(tokens)
        final_list_tokens=[]
        for token in tokens:
            if any(ext in token for ext in list_acr):
                final_list_tokens.append(token)
                #print(token)


            else:
                #print(token)
                list_spec=['&','/']
                if any(ext in token for ext in list_spec):
                    special_char=[ext for ext in list_spec if(ext in token)][0]


                    if (any(ext in special_char for ext in ['&','/']) and (len(token.split(special_char)[0])==1 and len(token.split(special_char)[1])==1)):
                        final_list_tokens.append(token)



                    else:


                        token=token.lower()
                        token=token.replace("/"," ")
                        token=token.replace("-"," ")
                        token=token.replace("&"," ")
                        token=token.replace("_"," ")
                        token=token.replace(","," ")


                                # Remove all non-word characters (everything except numbers and letters)
                        token = re.sub(r"[^\w\s]", '', token)
                            #token = re.sub(" \d+ ", " ", token) # only isolated
                        token=re.sub(r"\b\d+\b", "", token)
                            #print(token)
                        final_list_tokens.append(token)
                else:
                    token=token.lower()
                    token=token.replace("/"," ")
                    token=token.replace("-"," ")
                    token=token.replace("&"," ")
                    token=token.replace("_"," ")
                    token=token.replace(","," ")


                                # Remove all non-word characters (everything except numbers and letters)
                    token = re.sub(r"[^\w\s]", '', token)
                            #token = re.sub(" \d+ ", " ", token) # only isolated
                    token=re.sub(r"\b\d+\b", "", token)
                            #print(token)
                    final_list_tokens.append(token)



        #print(final_list_tokens)

        final_sentence=' '.join(final_list_tokens)

        final_sentence = ' '.join([word for word in final_sentence.split() if word not in stop_words])

        # Replace all runs of whitespace with a single dash
        final_sentence = re.sub(r"\s+", ' ', final_sentence)

        return final_sentence
    else:

        s=s.lower()
        s=s.replace("/"," ")
        s=s.replace("-"," ")
        s=s.replace("&"," ")
        s=s.replace("_"," ")
        s=s.replace(","," ")

        # Remove all non-word characters (everything except numbers and letters)
        s = re.sub(r"[^\w\s]", '', s)
        #s = re.sub(" \d+ ", " ", s) #only isolated otherwise s = re.sub("\d+", " ", s)
        s=re.sub(r"\b\d+\b", "", s)

        s = ' '.join([word for word in s.split() if word not in stop_words])

        # Replace all runs of whitespace with a single dash
        s = re.sub(r"\s+", ' ', s)
        #s=s.replace("/"," ")
        #s=s.replace("-"," ")
        #text = ' '.join([word for word in text.split() if word not in stop_words])

        return s




with open("data/list_titles_BGT.txt", "rb") as fp:   # Unpickling
      list_titles = pickle.load(fp)

# save list titles tokenize in training corpus
outF = open("data/BGT_Titles_2.txt", "w")
with open('data/BGT_Titles_2.txt', 'a') as f:
    for s in list_titles:
        f.write(urlify(s) + '\n')
    f.close()
