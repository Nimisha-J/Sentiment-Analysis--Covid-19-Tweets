import numpy as np
import pandas as pd
import re
import string
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer 
from nltk import pos_tag
from nltk.corpus import stopwords
stopWords = set(stopwords.words('english'))

from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from sklearn.utils import resample
from sklearn.utils import shuffle

np.random.seed(101)
data=pd.read_csv('C:/Users/Nimis/Desktop/Main project/Code/stepbystep/preprocessed.csv', encoding = 'utf8')
data.tweet=data.tweet.astype(str)
data.head()
'''
def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN
    
 # use Wordnet(lexical database) to lemmatize text 
def lemmatize_text(text):
    
    lmtzr = WordNetLemmatizer().lemmatize
    text = word_tokenize(str(text))   # Init the Wordnet Lemmatizer    
    word_pos = pos_tag(text)    
    lemm_words = [lmtzr(sw[0], get_wordnet_pos(sw[1])) for sw in word_pos]
    return (' '.join(lemm_words))

# clean and normalize text
def pre_process(text):    
    
    emoji_pattern = re.compile("["
                       u"\U0001F600-\U0001F64F"  # emoticons
                       u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                       u"\U0001F680-\U0001F6FF"  # transport & map symbols
                       u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                       u"\U00002702-\U000027B0"
                       u"\U000024C2-\U0001F251"
                       "]+", flags=re.UNICODE)    

    text = emoji_pattern.sub(r'', text)                       # remove emojis       
    text = text.lower()                                       # lowercase all letters   
    text = re.sub(r'@[A-Za-z0-9]+', '', text)                # remove user mentions, e.g. @VirginAmerica    
    text = re.sub(r'https?://[A-Za-z0-9./]+', '', text)
    return text
         # remove URL links 

#    white_list = ["not", "no", "won't", "isn't", "couldn't", "wasn't", "didn't", "shouldn't", 
#                  "hasn't", "wouldn't", "haven't", "weren't", "hadn't", "shan't", "doesn't",
#                  "mightn't", "mustn't", "needn't", "don't", "aren't", "won't"]
#    words = text.split()
#    text = ' '.join([t for t in words if (t not in stopwords.words('english') or t in white_list)])  # remove stopwords        

    text = ''.join([t for t in text if t not in string.punctuation])   # remove all punctuations       
    text = ''.join([t for t in text if not t.isdigit()])   # remove all numeric digits     
    text = re.sub("[^a-zA-Z0-9]", " ", text)   # letters only         
    text = lemmatize_text(text)   # use Wordnet(lexical database) to lemmatize text     
#    text = stemmer_text(text)   # stem text 
  
text = lambda x: pre_process(x)

data1 = pd.DataFrame(data.tweet.apply(text))
print(data1.head())
data1.to_csv("C:/Users/Nimis/Desktop/Main project/Code/stepbystep/preprocessed.csv") # + file_name + '.csv')
print("PREPROCESSING DONE")

airline = pd.read_csv('Tweets.csv', encoding = 'utf8')
Y = pd.get_dummies(airline['airline_sentiment']).values
airline['label_convert'] = airline['airline_sentiment'].map({'negative':0, 'neutral':1, 'positive': 2})
Y = np.array(airline['label_convert'])'''


from wordcloud import WordCloud, STOPWORDS
stopwords_forCloud = set(STOPWORDS)
import matplotlib.pyplot as plt
import seaborn as sns
import codecs
   # Barplot shows the number neutral, positive and neguative reviews.
sns.set(style="darkgrid")
sns.countplot(x = 'sentiment', data = data, order = data['sentiment'].value_counts().index, palette = 'Set1')
plt.xlabel('Sentiment')
plt.ylabel('Frequency')
plt.show()


Y = pd.get_dummies(data['sentiment']).values
data['label_convert'] = data['sentiment'].map({'negative':0, 'neutral':1, 'positive': 2})
Y = np.array(data['label_convert'])

train_X, test_X, train_Y, test_Y = train_test_split(data, Y, test_size = 0.2, random_state=42)



print(train_X.shape,train_Y.shape)
print(test_X.shape,test_Y.shape)


# No oversampling and undersampling
trainX_sampled = train_X
print(trainX_sampled.shape)
#train_X[['text', 'processed_text']].head(10)
from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer()
tokenizer.fit_on_texts(trainX_sampled['tweet'])
#print( "wors index"(len(tokenizer.word_index)))

trainX = tokenizer.texts_to_sequences(trainX_sampled['tweet'].values)
print("----------Training Data---------",trainX)
testX = tokenizer.texts_to_sequences(test_X['tweet'].values)
print("----------Testing Data---------",testX)
vocab_size = len(tokenizer.word_index) + 1
print(vocab_size)
max_len = 100
trainX_pad = pad_sequences(trainX, maxlen = max_len)
testX_pad = pad_sequences(testX, maxlen = max_len)
print("---------padding--------")
print(trainX_pad)
print(testX_pad)


from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
from keras import metrics
from keras import regularizers

embeddings_dictionary = dict()
glove_file = open('C:/Users/Nimis/Desktop/Main project/Code/stepbystep/glove.twitter.27B.100d.txt', encoding="utf8")
# Creating each line in GloVe dataset as a kwy-value pair
for line in glove_file:
    records = line.split()
    word = records[0]
    vector_dimensions = np.asarray(records[1:], dtype='float32')
    embeddings_dictionary [word] = vector_dimensions
glove_file.close()
print('Found %s word vectors.' %len(embeddings_dictionary))
#print("Embedding dictionary",embeddings_dictionary)

hits=0
misses=0

embedding_matrix = np.zeros((vocab_size, 100))
# Getting vector representation (from embedding dictionary) of each word in word_index
for word, index in tokenizer.word_index.items():
    embedding_vector = embeddings_dictionary.get(word)
    #If word is not available in GloVE embedding text file, that word will be skipped
    if embedding_vector is not None:
        embedding_matrix[index] = embedding_vector
        hits=hits+1
    else:
        misses=misses+1
           # embedding_matrix[i] = np.random.randn(100)
print("converted %d words (%dmisses)" %(hits,misses)) 

print("-------------embedding matrix----------------")
print(embedding_matrix)
          
print("embedding done")


embed_dim = 100
lstm_out = 100
def buildModel():     
    model = Sequential()
    model.add(Embedding(vocab_size, embed_dim,weights=[embedding_matrix], input_length=max_len,trainable=False))
    model.add(LSTM(lstm_out, dropout = 0.2))
    model.add(Dense(3, activation='softmax')) #, kernel_regularizer=regularizers.l2(0.005)))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=[metrics.sparse_categorical_accuracy])
    #model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
model = buildModel()
print(model.summary())
model.fit(trainX_pad,trainX_sampled['label_convert'], epochs = 20, batch_size = 48, verbose = 2)

score, acc = model.evaluate(testX_pad, test_Y, verbose = 2, batch_size = 48)
print("score: %.2f" % (score))
print("acc: %.2f" % (acc))
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, classification_report
test_Y_pred1 = model.predict(testX_pad)
test_Y_pred = np.argmax(test_Y_pred1, axis=1)
print(classification_report(test_Y, test_Y_pred))
print(pd.crosstab(test_Y.ravel(), test_Y_pred, rownames = ['True'], colnames = ['Predicted'], margins = True))
df_result = test_X.copy()
df_result['prediction'] = test_Y_pred.tolist() 

#file_name = 'LSTM_prediction'
df_result.to_csv("C:/Users/Nimis/Desktop/Main project/Code/stepbystep/LSTM_predictionexp.csv") # + file_name + '.csv')
