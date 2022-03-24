from nltk.tokenize import RegexpTokenizer
import pandas as pd
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

def removePunc(sentence):
    tokenizer = RegexpTokenizer(r'\w+|!+|\?+|\$+')
    return tokenizer.tokenize(sentence)


def removeStopwords(wordArray):
    stopWords = stopwords.words('english')
    filteredStopWords = []
    for word in wordArray:
        if word not in stopWords:
            filteredStopWords.append(word)
    return filteredStopWords

def preproccess(sentence):
    wordArray = removePunc(sentence)
    wordArray = removeStopwords(wordArray)
    return wordArray


df=pd.read_csv('spam.csv', encoding='Windows-1252')
df_text=df['v2'].str.lower()
df_label = df['v1']
df_text.head()
for sentence in df_text:
    wordArray = preproccess(sentence)
    print(wordArray)
tokenizer = RegexpTokenizer(r'\w+|!')
a = tokenizer.tokenize('Eighty-seven miles to go, yet.  Onward!')
print(a)