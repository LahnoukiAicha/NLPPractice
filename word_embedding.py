from tensorflow.keras.preprocessing.text import one_hot
sentences=['I am Aicha','the cup of tea','i am a good girl','holy shit is a bad word']

#here we represent each word with a number  and sentence with a vector
size=10000
onehot_repr=[one_hot(words,size) for words in sentences]

from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
import numpy as np

#here we defined the maximum length of the sentences after padding.
#  If a sentence is shorter than this length, 
# it will be padded with zeros at the beginning.
sent_lent=8
embedded_docs=pad_sequences(onehot_repr,padding='pre',maxlen=sent_lent)

#here each number in before should be coded with 10 elements of vector
dim=10
model=Sequential()
model.add(Embedding(size,dim))
model.build(input_shape=(None, sent_lent))
model.compile('adam','mse') #adam is an optimizer and mse is function loss
print(model.predict(embedded_docs))

Sentences → One-hot encoded sequences → Padded sequences.
Padded sequences → Embedding layer (to convert integers to vectors) → Output dense vectors
