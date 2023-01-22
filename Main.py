import re
from nltk.corpus import stopwords
import nltk
import pickle
from nltk.stem  import WordNetLemmatizer
lemmatizer=WordNetLemmatizer()
nltk.download('stopwords')
stop_word = set(stopwords.words('english'))


def clean_lammatize(x):
    corpus = []
    for i in range(0,len(x)):
        data = re.sub('[^a-zA-Z]',' ',x[i])
        data = data.lower()
        #split funcion is just converting string into list
        data = data.split()
        #applying the stemming on the message
        data = [lemmatizer.lemmatize(word) for word in data if word not in stop_word]
        #join function add all the list  item i nto one string separeted by space.
        data= ' '.join(data)
        corpus.append(data)
    return corpus


file = open("C:/Users/Asus/Documents/tfidf.pkl","rb")
tfidf_model = pickle.load(file)


def vector_converter(x):
    v = tfidf_model.transform(x)
    return v