##### Install Fasttext #######
# $ git clone https://github.com/facebookresearch/fastText.git
# $ cd fastText
# $ pip install .

import fastText
import numpy as np
import pandas as pd
import nltk
import re
import pickle
import scipy.spatial.distance
np.seterr(divide='ignore', invalid='ignore')
from nltk.corpus import stopwords
#nltk.download('punkt')
stop_words = set(stopwords.words('english'))
stop_words.remove('it')
stop_words.remove('ma')
stop_words.remove('i')
stop_words.remove('is')
#stop_words.remove('on') # on-air, on-site
#stop_words.remove('in') # in-store


list_acr=['M&A','P&C','FP&A', 'H&B','R&D', 'L&D',
          'F&B', 'I&C', 'O&M', 'F&I', 'I&E', 'L&I', 'E&I', 'Pm&R', 'C&A', 'A/C', 'E/O', 'A/V', 'A/B', 'A/C'
         'E-Commerce', 'Co-op', 'On-', 'In-', 'Pre-', 'post-', 'Full-Stack', 'e-', 'co-',
         '.Net', 'c#', 'c++', 'k-']
list_acr = [item.lower() for item in list_acr]

def urlify(s):
    s=s.lower()
    
    #s=re.sub(r"\br&d\b","Research and Development" ,s)
    #s=re.sub(r"\bhr\b","Human Resources" ,s)
    #s=re.sub(r"\baml\b",'Anti-Money Laundering' ,s)
    #s=re.sub(r"\betms\b",'Electronic Territory Management System' ,s)
    #s=re.sub(r"\bceo\b",'Chief Executive Officer' ,s)
    #s=re.sub(r"\bic\b",'Integrated Circuit' ,s)
    #s=re.sub(r"\bis\b",'Information Systems' ,s) # tricky!!!! 
    #s=re.sub(r"\bb2b\b",'Business To Business' ,s)
    #s=re.sub(r"\bued\b",'User Experience Design' ,s)
    #s=re.sub(r"\bfp&a\b",'Financial Planning And Analysis' ,s)
    #s=re.sub(r"\bqa\b",'Quality Assurance' ,s)
    #s=re.sub(r"\bcfo\b",'Chief Financial Officer' ,s)
    #s=re.sub(r"\bcoo\b",'Chief Operations Officer' ,s)
    #s=re.sub(r"\bdba\b",'' ,s)
    #s=re.sub(r"\bm&e\b",'Mechanical and Electrical' ,s)
    #s=re.sub(r"\bar\b",'Accounts Receivable' ,s)
    #s=re.sub(r"\bap\b",'Accounts Payable' ,s)
    #s=re.sub(r"\bsr\b",'senior' ,s)
    #s=re.sub(r"\bmtm\b",'Made-to-Measure' ,s)
    #s=re.sub(r"\bvas\b",'Value Added Services' ,s)
    #s=re.sub(r"\bmep\b",'Mechanical, Electrical & Plumbing' ,s)
    #s=re.sub(r"\breo\b",'Real Estate Owned' ,s)
    #s=re.sub(r"\bbm&f/bovespa\b",'' ,s)
    
    #s=s.lower()
    
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



# load list of unique Bgt tiles
with open("data/list_titles_BGT.txt", "rb") as fp:
      list_titles = pickle.load(fp)


# load model
model = fastText.load_model('model/ns2520ns3005_new.bin')

# load embeddings
p_name_fasttext_128 = np.load('model/ft_embeddings_ns2520ns3005_new.npy')

def normalize(embeddings):
    embeddings_norm = embeddings ** 2
    embeddings_norm = embeddings / np.sqrt(embeddings_norm.sum(axis=1, keepdims=True))
    return embeddings_norm


def cos_cdist(matrix, vector):
    """
    Compute the distances between each row of matrix and vector.
    cosine or euclidean (normalized)
    """
    v = vector.reshape(1, -1)
    #return 1 - scipy.spatial.distance.cdist(matrix, v, 'cosine').reshape(-1)
    return 1 - scipy.spatial.distance.cdist(normalize(matrix), normalize(v), 'euclidean').reshape(-1)



def get_scores(list_new_titles, num=5):
    test_dt=pd.DataFrame()
    for profile in list_new_titles:
        doc_mmc=urlify(profile)
        vector_mmc = model.get_sentence_vector(doc_mmc)
        cos_similar=cos_cdist(p_name_fasttext_128, vector_mmc)
        ind=(-cos_similar).argsort()[:num]
        for i in ind:
            test_dt = test_dt.append({'Job Profile': profile, 'BGT Job Title': list_titles[i], 'Similarity Score': cos_similar[i].round(4)}, ignore_index=True)
            test_dt=test_dt[['Job Profile', 'BGT Job Title', 'Similarity Score']]
    return test_dt
