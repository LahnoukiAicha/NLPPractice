
import nltk
nltk.download('punkt')

from textblob import TextBlob
from newspaper import Article
url='https://www.nbcnews.com/mach/science/what-tsunami-ncna943571'
article=Article(url)
article.download()
article.parse()
article.nlp() #preparing it for nlp
text=article.summary
blob=TextBlob(text) 
sentiment=blob.sentiment.polarity #[-1,1]


txt='I hate this thing. it distroye dmy life. I would never buy it again. bad one'
blob=TextBlob(txt)
sentiment=blob.sentiment.polarity
print(sentiment)