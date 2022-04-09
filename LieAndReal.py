import pandas as pd

df = pd.read_csv('SentimentalLIAR-master/train_final.csv', encoding='utf-8')

from nltk.tokenize import RegexpTokenizer
from sklearn.naive_bayes import MultinomialNB
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
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
def completeSense(name, amount):
    if amount < 0.2 and amount >= 0:
        return "$" + name + "_1"
    elif amount < 0.4 and amount >= 0.2:
        return "$" + name + "_2"
    elif amount < 0.6 and amount >= 0.4:
        return "$" + name + "_3"
    elif amount < 0.8 and amount >= 0.6:
        return "$" + name + "_4"
    elif amount <= 1 and amount >= 0.8:
        return "$" + name + "_5"
def completeData(wordArrayText, anger, fear, joy, disgust, sad):
    return wordArrayText + " " + completeSense("ANG", anger) + " " + completeSense("FEAR", fear) + " " + completeSense("JOY", joy) + " " + completeSense("DISG", disgust) + " " + completeSense("SAD", sad)

df = pd.read_csv('SentimentalLIAR-master/train_final.csv', encoding='utf-8')
df_test = pd.read_csv('SentimentalLIAR-master/test_final.csv', encoding='utf-8')
df_text=df['statement'].str.lower()
df_text_test=df_test['statement'].str.lower()
df_label_test = df_test['label']
df_label = df['label']
df_sentiment = df['sentiment']
df_sentiment_test = df_test['sentiment']
df_anger = df['anger']
df_anger_test = df_test['anger']
df_fear = df['fear']
df_fear_test = df_test['fear']
df_joy = df['joy']
df_joy_test = df_test['joy']
df_disgust = df['disgust']
df_disgust_test = df_test['disgust']
df_sad = df['sad']
df_sad_test = df_test['sad']
df_text.head()
df_text_test.head()
df_label.head()
df_label_test.head()
df_sentiment.head()

naiveBayes = MultinomialNB()
lieDetectionData = []
df_label_result = []
df_label_result_test = []
for i in df_label:
    if i == 'true' or i == 'mostly-true' or i == 'half-true':
        df_label_result.append(1)
    else:
        df_label_result.append(0)
for i in df_label_test:
    if i == 'true' or i == 'mostly-true' or i == 'half-true':
        df_label_result_test.append(1)
    else:
        df_label_result_test.append(0)


for i in range(len(df_text)):
    wordArrayText = preproccess(df_text[i])
    # wordArrayText = df_text[i]
    if df_sentiment[i] == "NEGATIVE":
        wordArrayText += " $NEG"
    elif df_sentiment[i] == "POSITIVE":
        wordArrayText += " $POS"
    wordArrayText = completeData(wordArrayText, df_anger[i], df_fear[i], df_joy[i], df_disgust[i], df_sad[i])
    lieDetectionData.append(wordArrayText)
lieDetectionTest = []

for i in range(len(df_text_test)):
    wordArrayText = preproccess(df_text_test[i])
    # wordArrayText = df_text_test[i]
    if df_sentiment_test[i] == "NEGATIVE":
        wordArrayText += " $NEG"
    elif df_sentiment_test[i] == "POSITIVE":
        wordArrayText += " $POS"
    wordArrayText = completeData(wordArrayText, df_anger_test[i], df_fear_test[i], df_joy_test[i], df_disgust_test[i], df_sad_test[i])
    lieDetectionTest.append(wordArrayText)


count_vect = CountVectorizer()
tfidfTransformer = TfidfTransformer()
xTrainTf = count_vect.fit_transform(lieDetectionData)

xTrainTfidf = tfidfTransformer.fit_transform(xTrainTf)

naiveBayes.fit(xTrainTfidf, df_label_result)

xTestTf = count_vect.transform(lieDetectionTest)
xTestTfidf = tfidfTransformer.transform(xTestTf)

predict = naiveBayes.predict(xTestTfidf)
conf = metrics.confusion_matrix(df_label_result_test, predict)
print(predict)
print(conf)