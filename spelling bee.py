
# coding: utf-8

# In[1]:


import re
from sklearn.model_selection import train_test_split
import numpy as np
from keras_tqdm import TQDMNotebookCallback
from keras.layers import Flatten, Dense, Input, Dropout
from keras.layers.embeddings import Embedding
from keras.layers import Input, Embedding, Reshape, merge, LSTM, Bidirectional, RepeatVector, TimeDistributed, Activation
from keras.models import Sequential, Model
from keras.optimizers import SGD, Adam
from io import open
from keras.callbacks import EarlyStopping, ProgbarLogger
from keras.preprocessing.sequence import pad_sequences
import random
import numpy
from bidict import bidict
from keras.preprocessing.sequence import pad_sequences
import re
from sklearn.model_selection import train_test_split
import numpy as np
from keras_tqdm import TQDMNotebookCallback
from keras.layers import Flatten, Dense, Input, Dropout
from keras.layers.embeddings import Embedding
from keras.layers import Input, Embedding, Reshape, merge, LSTM, Bidirectional, RepeatVector, TimeDistributed, Activation
from keras.models import Sequential, Model
from keras.optimizers import SGD, Adam
from io import open
from keras.callbacks import EarlyStopping, ProgbarLogger
from keras.preprocessing.sequence import pad_sequences
import random
from attention import Attention


# In[2]:


flatten= lambda l:[item for sublist in l for item in sublist]
def randomly(seq):
    shuffled = list(seq)
    random.shuffle(shuffled)
    return iter(shuffled)


# In[3]:


with open('cmudict.0.6d') as f:
    lines = [l.strip().split("  ") for l in  f.readlines()]
    lines = [(l[0], l[1].split()) for l in lines if len(l)==2  and re.match("^[A-Z]+$", l[0]) and  len(l[0])<16]

    phonems = ['-'] + sorted(set(flatten([phs for w,phs in lines]))) + ['*']
    letters = ['_'] + sorted(set(flatten([w for w, phs in lines]))) + ['*']
    input_vocab_size = len(phonems)
    output_vocab_size = len(letters)

    char_vocab = dict(zip(letters, range(len(letters))))
    phone_vocab = dict(zip(phonems, range(len(phonems))))

    maxw_len = max([len(l[0]) for l in lines])
    maxphs_len = max([len(l[1]) for l in lines])

    X = np.zeros((len(lines), maxphs_len), np.int32)
    Y = np.zeros((len(lines), maxw_len), np.int32)

    for i, l in enumerate(randomly(lines)):
        for j, ph in enumerate(l[1]): X[i][j] = phone_vocab[ph]
        for j, ch in enumerate(l[0]): Y[i][j] = char_vocab[ch]

    go_token = char_vocab["*"]
    dec_input_ = np.concatenate([np.ones((len(lines),1)) * go_token, Y[:,:-1]], axis=1)

    X_train, X_test, X_d_train, X_d_test, y_train, y_test = train_test_split(X, dec_input_, Y, test_size=0.1)
    #X_train, X_val, X_d_train, X_d_val, y_train, y_val = train_test_split(X_train, X_d_train, y_train, test_size=float(1)/9)


# In[4]:


EMB_SIZE = 120

def lstm_(dec_dim = EMB_SIZE, return_sequences= True): 
    return LSTM(2*dec_dim, dropout_U= 0.1, dropout_W= 0.1, consume_less= 'gpu', return_sequences=return_sequences)


# In[5]:


inp = Input((maxphs_len,))
dec_i = Input((maxw_len,))
dec_e = Embedding(output_vocab_size, EMB_SIZE)(dec_i)
dec_e = Dense(2 * EMB_SIZE)(dec_e)

x = Embedding(input_vocab_size, EMB_SIZE)(inp)
x = Bidirectional(lstm_())(x)
x = lstm_()(x)
x = lstm_()(x)
x = Attention(lstm_, 3)([x, dec_e])
x = TimeDistributed(Dense(output_vocab_size, activation='softmax'))(x)
model = Model([inp, dec_i], x)

model.compile(Adam(), loss='sparse_categorical_crossentropy', metrics=['acc'])



# In[6]:


model.load_weights('model.h5')


# In[ ]:


model.fit([X_train, X_d_train], np.expand_dims(y_train,-1), validation_data=[[X_test, X_d_test], np.expand_dims(y_test,-1)], batch_size=64, verbose=1,callbacks= [ProgbarLogger()], nb_epoch=10)


# In[ ]:


import datetime
model.save_weights("model.h5")


# In[ ]:


#model.load_weights('model.h5')


# In[7]:


new_l = len(X_test)/4
X_test, X_d_test, y_test = X_test[:new_l], X_d_test[:new_l], y_test[:new_l]
def eval_keras():
    preds = model.predict([X_test, X_d_test], batch_size=128)
    predict = np.argmax(preds, axis = 2)
    return (np.mean([all(real==p) for real, p in zip(y_test, predict)]), predict)


acc, preds = eval_keras(); 
print('validation accuracy', acc)


# In[8]:


print("pronunciation".ljust(40), "real spelling".ljust(17), 
      "model spelling".ljust(17), "is correct")

for index in range(20):
    ps = "-".join([phonems[p] for p in X_test[index]]) 
    real = [letters[l] for l in y_test[index]] 
    predict = [letters[l] for l in preds[index]]
    print (ps.split("--")[0].ljust(40), "".join(real).split("_")[0].ljust(17),
        "".join(predict).split("_")[0].ljust(17), str(real == predict))


# In[9]:


get_ipython().system('jupyter nbconvert --to script spelling-bee.ipynb')

