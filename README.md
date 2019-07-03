# Character Embedding Model for Short Sentences Similarity

The purpose of this model is to map short text and find similar results.

The model used to tackle this challenge is FastText (https://github.com/facebookresearch/fastText)




## Requirements



### Install fastText for Python

```
$ git clone https://github.com/facebookresearch/fastText.git
$ cd fastText
$ pip install .
```

### python bindings
* gensim
* sklearn
* numpy
* nltk
* re
* scipy


> pip install -r requirements.txt




## Model building

**a.	Preprocessing**
In this step, we preprocess the data, using good practices in tokenization and handling acronyms and special characters (such as slash, dash and so on).
The ouput is a list of preprocessed data that we will use for training the model

Please go through 'ft_prep.py` for more details.

**b.	Training the model**
In this step we train the FastText model (unsupervised), using the preprocessed data as input.

Here is a list of the parameters used in the training process:

```

The arguments for the dictionary are:
  -model              skip-gram
  -minCount           minimal number of word occurrences [2]
  -wordNgrams         max length of word ngram  [3]
  -minn               min length of char ngram [2]
  -maxn               max length of char ngram [5]

The arguments for training are:
  -lr                 learning rate [0.05]
  -dim                size of word vectors [128]
  -ws                 size of the context window [3]
  -epoch              number of epochs [20]
  -neg                number of negatives sampled [5]
  -loss               loss function ns
  -thread             number of threads [1] # to ensure determinism

```

After the model has been created, we create the word embeddings for the preprocessed data.


Please go through 'ft_train.py` for more details.

**c.	Testing the model**
In this phase, we infer vectors for new text and test the model, analyzing the similarity scores between the training text and new sentences.

Please go through 'ft_test.py` for more details.

## Example use cases

### Get top n most similar titles

```
	import ft_test

	#list of test titles
	list_new_titles=['Data Scientist II']

	#find most similar titles
	ft_test.get_scores(list_new_titles, 5)

#  Input Job Profile        Output Job Title  Similarity Score
#0  Data Scientist III  Data Scientist III In            1.0000
#1  Data Scientist III     Data Scientist III            1.0000
#2  Data Scientist III    Data Scientist - II            0.9877
#3  Data Scientist III      Data Scientist II            0.9877
#4  Data Scientist III       Data Scientist I            0.9765
```
