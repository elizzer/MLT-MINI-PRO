#%%
import tensorflow as tf
import math
import numpy as np

# %%

def scaled_dot_product_attention(query,key,value,mask=None):
    key_dim=tf.cast(tf.shape(key)[-1],tf.float32)
    scaled_scores=tf.matmul(query,key,transpose_b=True)/np.sqrt(key_dim)

    if mask is not None:
        scaled_scores=tf.where(mask==0,-np.inf,scaled_scores)
    
    softmax=tf.keras.layers.Softmax()
    weights=softmax(scaled_scores)

    return tf.matmul(weights,value),weights
# %%

seq_len=3
embed_dim=4

def randomWeights(x=seq_len,y=embed_dim):
    return np.random.rand(x,y)

queries=randomWeights()
keys=randomWeights()
values=randomWeights()

print(queries)
# %%
output, attn_weights= scaled_dot_product_attention(query=queries,key=keys,value=values)
print('[=]Output',output)
print(attn_weights)
# %%
batch_size=1
seq_len=3;
embed_dim=12
num_head=3
head_dim=embed_dim//num_head

print(f"Dimention of each head: {head_dim}")
# %%
x=np.random.rand(batch_size,seq_len,embed_dim).round(1)
x
# %%
wq0=randomWeights(embed_dim,head_dim).round(1)
wq1=randomWeights(embed_dim,head_dim).round(1)
wq2=randomWeights(embed_dim,head_dim).round(1)

wk0=randomWeights(embed_dim,head_dim).round(1)
wk1=randomWeights(embed_dim,head_dim).round(1)
wk2=randomWeights(embed_dim,head_dim).round(1)

wv0=randomWeights(embed_dim,head_dim).round(1)
wv1=randomWeights(embed_dim,head_dim).round(1)
wv2=randomWeights(embed_dim,head_dim).round(1)

# %%
q0=np.dot(x,wq0)
k0=np.dot(x,wk0)
v0=np.dot(x,wv0)

q1=np.dot(x,wq1)
k1=np.dot(x,wk1)
v1=np.dot(x,wv1)

q2=np.dot(x,wq2)
k2=np.dot(x,wk2)
v2=np.dot(x,wv2)

# %%
class MultiHeadSelfAttention(tf.keras.layers.Layer):
    def __init__(self,d_model,num_heads):
        super(MultiHeadSelfAttention,self).__init__()

        self.d_model=d_model
        self.num_heads=num_heads

        self.d_head=self.d_model//self.num_heads


        self.wq=tf.keras.layers.Dense(self,d_model)
        self.wk=tf.keras.layers.Dense(self,d_model)
        self.wv=tf.keras.layers.Dense(self,d_model)

        self.dens=tf.keras.layers.Dense(self.d_model)