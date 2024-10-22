import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer

#tensorflow 
tokenizer=Tokenizer(num_words=100)
tokenizer.fit_on_texts(sentences) #on the splitted 
word_index=tokenizer.word_index #represent each word by a number
sequences = tokenizer.texts_to_sequences(sentences) #the splitted represented by an array of numbers


import nltk
nltk.download('punkt') #this what makes the sentences splitted based on ponctuation (?!)
    
#nltk 
text='hello bts! , how are u doing? emmm, i have been thinking about u lately.'
sentences=nltk.sent_tokenize(text) #here we split the text based on ?!
tokens=nltk.word_tokenize(text) #here word by word even "," based on space
#remove stopwords
nltk.download('stopwords')
from nltk.corpus import stopwords

stop_words=set(stopwords.words('english'))
filtered_words= [word for word in tokens if word not in stop_words]
filtered_text= ' '.join(filtered_words)

#POS 
import spacy
nlp=spacy.load('en_core_web_sm')
nlp_text=nlp(text)
for token in nlp_text:
  print(token,token.pos_)

#NER:
import spacy
nlp=spacy.load('en_core_web_sm')
elmnt='Google is a faang company built in 2002 for $45 .'
nlp_text=nlp(elmnt)
print(nlp_text.ents) #google,2002,$45
for token in nlp_text.ents:
  print(token.text,token.label_) #organisqtion,date,money


