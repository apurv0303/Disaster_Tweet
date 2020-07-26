import pandas as pd
import numpy as np
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import re
import string
from keras.models import Model
from keras.layers import Dense, Input, Dropout, LSTM, Activation
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.initializers import glorot_uniform

#DATA LOADING
df=pd.read_csv("train.csv",index_col=False)

#Missing Value
#First check how many rows have missing value each column
missing_value=df.isnull().sum()
missing_value[0:3]

#%age of missing value
total_cells=np.product(df.shape)
total_missing_cell=missing_value.sum()
print((total_missing_cell/total_cells)*100)

#WHAT to do with missing Value

 #1st method(drop all the rows(for column axis=1) which contains missing value)
    #But currently text is more imp than keyword or location
df_dropna=df.dropna()

#2nd method(fill the missing value)
df_fill=df.fillna(method="bfill",axis=0)
df_fill.head()

#3rd method(filling value with some logic)
#df["keyword"]=df["keyword"].fillna("danger",inplace=True)
df["keyword"]=df["keyword"].fillna("disaster")
df["location"]=df["location"].fillna("earth")
df.head()

print(df.shape)
#1st method
x_train=df[["text","keyword","location"]]
y_train=df["target"]

#2nd method
x_train=df.iloc[:,1:4]
y_train=df.iloc[:,4]
x_train.head()

#concatinating all 3 columns as ..
x_train["text"]=x_train["keyword"] + " " + x_train["text"]
x_train=x_train["text"]

y_train_fin=[]
for label in y_train:
    y_train_fin.append(label)
    
 #Important pre-processing part
"""
1.LowerCase the text
2.remove punctuations
3.remove stopwords
4.(not necc.)remove frequent words
5.Stemming
6.Lemmatization
7.Removal of emojis
8.Removal of emoticons
9.Conversion of emoticons to words
10.Conversion of emojis to words
11.Removal of URLs
12.Removal of HTML tags
13.Chat words conversion
14.Spelling correction
"""
#Method 1
corpus=[]
ps=PorterStemmer()
for sent in x_train:
    sent=sent.lower()
    sent=re.sub('[^a-zA-Z]'," ",sent)
    sent=sent.split()
    sent=[ps.stem(word) for word in sent if not word in stopwords.words('english')]
    sent=" ".join(sent)
    corpus.append(sent)


#method 2
x_train_fin=[]
for sent in x_train:
    sent=sent.strip()
    x_train_fin.append(sent)
x_train_fin=x_train.str.lower()

    
#string.punctuations have !"#$%&\'()*+,-./:;<=>?@[\\]^_{|}~`
def remove_punct(text):
    return 

x_train_fin=x_train_fin.apply(lambda text: remove_punct(text))


def remove_stop(text):
    return " ".join([word for word in str(text).split() if not word in stopwords.words('english')])
x_train_fin=x_train_fin.apply(lambda text: remove_stop(text))
x_train_fin.head()

x_train_fin=np.array(x_train_fin)
stopwords = nltk.corpus.stopwords.words('english')
newStopWords = ['stopWord1','stopWord2']
stopwords.extend(newStopWords)
m = x_train_fin.shape[0]
for i in range(m):                               # loop over training examples
    sentence_words = [word for word in x_train_fin[i].split() if not word in stopwords.words('english')]
 
#Most common word
from collections import Counter
cnt=Counter()
for text in x_train_fin:
    for word in text.split():
        cnt[word]+=1
cnt.most_common(10)    
freqword=set([w for (w,wc) in cnt.most_common(10)])
#you can remove frequent words as stopwords
#   stemming and lemma
#(as above)
#for english language use porter stemmer but for other 
#language use snowball Stemmer(ALL VIA NLTK)
    

def read_glove_vecs(glove_file):
    with open(glove_file, 'r',encoding="utf8") as f:
        words = set()
        word_to_vec_map = {}
        for line in f:
            line = line.strip().split()
            curr_word = line[0]
            words.add(curr_word)
            word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float64)
        
        i = 1
        word_to_index = {}
        index_to_word = {}
        for w in sorted(words):
            word_to_index[w] = i
            index_to_word[i] = w
            i = i + 1
    return word_to_index, index_to_word, word_to_vec_map
word_to_index, index_to_word, word_to_vec_map = read_glove_vecs('glove.6B.50d.txt')
maxLen = len(max(x_train_fin, key=len).split())

 def sentences_to_indices(X, word_to_index, max_len):
    m = X.shape[0]                                   # number of training examples
    X_indices = np.zeros((m,max_len))
    for i in range(m):                               # loop over training examples
        sentence_words = [word.lower() for word in X[i].split()]
    j = 0
    for w in sentence_words:
            X_indices[i, j] = word_to_index[w]
            j = j + 1
    return X_indices  

def pretrained_embedding_layer(word_to_vec_map, word_to_index):
    
    vocab_len = len(word_to_index) + 1                  # adding 1 to fit Keras embedding (requirement)
    emb_dim = word_to_vec_map["cucumber"].shape[0]      # define dimensionality of your GloVe word vectors (= 50)
    
    emb_matrix = np.zeros((vocab_len,emb_dim))
    
   
    for word, idx in word_to_index.items():
        emb_matrix[idx, :] = word_to_vec_map[word]
    embedding_layer = Embedding(vocab_len, emb_dim, trainable = False)
    embedding_layer.build((None,)) # Do not modify the "None".  This line of code is complete as-is.
    embedding_layer.set_weights([emb_matrix])
    
    return embedding_layer
  
  embedding_layer = pretrained_embedding_layer(word_to_vec_map, word_to_index)
  
  # GRADED FUNCTION: Emojify_V2

def tweet(input_shape, word_to_vec_map, word_to_index):
    
    
    sentence_indices = Input(input_shape,dtype='int32')
    embedding_layer = pretrained_embedding_layer(word_to_vec_map, word_to_index)
    embeddings = embedding_layer(sentence_indices)  
    
    X = LSTM(128, return_sequences = True)(embeddings)
 
    X = Dropout(0.5)(X)

    X = LSTM(128, return_sequences=False)(X)
   
    X = Dropout(0.5)(X)
    
    X = Dense(11)(X)
 
    X = Activation('softmax')(X)
    
    model = Model(inputs = sentence_indices, outputs = X)
    
    return model
  
  model = tweet((maxLen,), word_to_vec_map, word_to_index)
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

X_train_indices = sentences_to_indices(x_train_fin, word_to_index, maxLen)

maxLen = len(max(x_train_fin, key=len).split())
m = x_train_fin.shape[0]                                   # number of training examples
X_indices = np.zeros((m,maxLen))
for i in range(m):                               # loop over training examples
    sentence_words = [word for word in x_train_fin[i].split()]
# j = 0
# for w in sentence_words:
#     X_indices[i, j] = word_to_index[w]
#     j = j + 1
print(sentence_words)
