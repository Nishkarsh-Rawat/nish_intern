#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
data=pd.read_csv("C:\\Users\\91852\\Downloads\\ParsedLogs.csv")
data.head()


# In[2]:


df=data['id,message,@version,@timestamp'].str.split(',',expand=True)


# In[3]:


df=df[1]


# In[4]:


df.head()


# In[5]:


df=df.to_frame()


# In[6]:


df.info()


# In[7]:


df[1]


# In[8]:


messages=df


# In[9]:


corpus=set()


# In[10]:


for str in df[1]:
    corpus.add(str)


# In[11]:


corpus


# In[12]:


words = []
for text in corpus:
    for word in text.split(' '):
        words.append(word)

words = set(words)
words=list(words)


# In[13]:


words


# In[14]:


len(words)


# ## Data Generation

# ### Creating input and label for skip gram

# In[15]:


word2int = {}

for i,word in enumerate(words):
    word2int[word] = i

sentences = []
for sentence in corpus:
    sentences.append(sentence.split())
    
WINDOW_SIZE =4 

data = []
for sentence in sentences:
    for idx, word in enumerate(sentence):
        for neighbor in sentence[max(idx - WINDOW_SIZE, 0) : min(idx + WINDOW_SIZE, len(sentence)) + 1] : 
            if neighbor != word:
                data.append([word, neighbor])


# In[16]:


df = pd.DataFrame(data, columns = ['input', 'label'])


# In[17]:


df.head(10)


# In[18]:


df.shape


# ## Define tensorflow graph

# In[19]:


ONE_HOT_DIM = len(words)

# function to convert numbers to one hot vectors
def to_one_hot_encoding(data_point_index):
    one_hot_encoding = np.zeros(ONE_HOT_DIM)
    one_hot_encoding[data_point_index] = 1
    return one_hot_encoding

X = [] # input word
Y = [] # target word

for x, y in zip(df['input'], df['label']):
    X.append(to_one_hot_encoding(word2int[ x ]))
    Y.append(to_one_hot_encoding(word2int[ y ]))


# In[20]:


# convert them to numpy arrays
X_train = np.asarray(X)
Y_train = np.asarray(Y)


# In[21]:


import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 
# making placeholders for X_train and Y_train
x = tf.placeholder(tf.float32, shape=(None, ONE_HOT_DIM))
y_label = tf.placeholder(tf.float32, shape=(None, ONE_HOT_DIM))


# In[22]:


EMBEDDING_DIM = 8


# In[23]:


# hidden layer: which represents word vector eventually
W1 = tf.Variable(tf.random_normal([ONE_HOT_DIM, EMBEDDING_DIM]))
b1 = tf.Variable(tf.random_normal([1])) #bias
hidden_layer = tf.add(tf.matmul(x,W1), b1)

# output layer
W2 = tf.Variable(tf.random_normal([EMBEDDING_DIM, ONE_HOT_DIM]))
b2 = tf.Variable(tf.random_normal([1]))
prediction = tf.nn.softmax(tf.add( tf.matmul(hidden_layer, W2), b2))

# loss function: cross entropy
loss = tf.reduce_mean(-tf.reduce_sum(y_label * tf.log(prediction), axis=[1]))

# training operation
train_op = tf.train.GradientDescentOptimizer(0.05).minimize(loss)


# ## Train

# In[24]:


sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init) 

iteration = 500
for i in range(iteration):
    # input is X_train which is one hot encoded word
    # label is Y_train which is one hot encoded neighbor word
    sess.run(train_op, feed_dict={x: X_train, y_label: Y_train})
    if i % 25 == 0:
        print('iteration '+' loss is : ', sess.run(loss, feed_dict={x: X_train, y_label: Y_train}))


# In[25]:


# Now the hidden layer (W1 + b1) is actually the word look up table
vectors = sess.run(W1 + b1)
print(vectors)


# ## Word Vector table

# In[26]:


vec = pd.DataFrame(vectors)


# In[27]:


vec


# # Preapring input for clustering

# In[28]:


messages.head()


# In[29]:


messages.info()


# In[69]:


sentences=[]
for str in messages[1]:
    sentences.append(str)


# In[70]:


type(sentences[0])


# In[71]:


print(type(words))
l=len(words)
print(l)


# In[72]:


dict_words={}


# In[73]:


for i in range(l):
    dict_words[words[i]]=list(vec.iloc[i,:])


# In[74]:


print(type(dict_words["end"]))


# ### Calculating mean of vectors

# In[122]:


list_final=[]
for text in sentences:
    temp=text.split(' ')
    lt=[]
    for s in temp:
        lt.append(dict_words[s])
    npat=np.array(lt)
    t=np.mean(npat,axis=0).tolist()
    list_final.append(t)


# In[123]:


list_final


# In[124]:


cl_input=pd.DataFrame(list_final)


# In[125]:


cl_input


# # Clustering-DBSCAN

# In[ ]:





# In[ ]:




