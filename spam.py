from nltk.tokenize import RegexpTokenizer
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import metrics

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
    wordArrayText = ""
    for i in wordArray:
        wordArrayText += i + " "
    return wordArrayText

df=pd.read_csv('spam.csv', encoding='Windows-1252')
df_text=df['v2'].str.lower()
df_label = df['v1']
df_text.head()
naiveBayes = MultinomialNB()
spamDetectionData = []
df_label_result = []
for i in df_label:
    if i == 'ham':
        df_label_result.append(1)
    else:
        df_label_result.append(0)
for sentence in df_text[0:int(0.8 *len(df_text))]:
    wordArrayText = preproccess(sentence)
    spamDetectionData.append(wordArrayText)
# naiveBayes.fit(spamDetectionData, df_label_result[0:int(0.8 *len(df_text))])
spamDetectionTest = []
for sentence in df_text[int(0.8 *len(df_text)):]:
    wordArrayText = preproccess(sentence)
    spamDetectionTest.append(wordArrayText)

# a = naiveBayes.predict(spamDetectionTest)

count_vect = CountVectorizer()
tfidfTransformer = TfidfTransformer()
xTrainTf = count_vect.fit_transform(spamDetectionData)
print(xTrainTf)
print(xTrainTf.shape)

xTrainTfidf = tfidfTransformer.fit_transform(xTrainTf)

naiveBayes.fit(xTrainTfidf, df_label_result[0:int(0.8 *len(df_text))])

xTestTf = count_vect.transform(spamDetectionTest)
xTestTfidf = tfidfTransformer.transform(xTestTf)

predict = naiveBayes.predict(xTestTfidf)
conf = metrics.confusion_matrix(df_label_result[int(0.8 *len(df_text)):], predict)
print(predict)
print(conf)


