from nltk.corpus import stopwords
from os import listdir
from collections import Counter
import string
from numpy import array
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

def load_doc(filename):
	file=open(filename,'r')
	text=file.read()
	file.close()
	return text

def clean_doc(doc):
	tokens=doc.split()
	tokens =[word for word in tokens if word.isalpha()]
	stop_words=set(stopwords.words('english'))
	tokens=[w for w in tokens if not w in stop_words]
	tokens =[word for word in tokens if len(word)>1]
	return tokens

def doc_to_line(filename,vocab):
	doc=load_doc(filename)
	tokens=clean_doc(doc)
	tokens=[w for w in tokens if w in vocab]
	return ' '.join(tokens)

def process_docs(directory,vocab,is_train):
	lines=list()
	for filename in listdir(directory):
		if is_train and filename.startswith('cv9'):
			continue
		if not is_train and not filename.startswith('cv9'):
			continue
		path=directory +'/'+filename
		line=doc_to_line(path,vocab)
		lines.append(line)
	return lines

def predict_sentiment(review,vocab,tokenizer,model):
	tokens=clean_doc(review)
	tokens=[w for w in tokens if w in vocab]
	line=' '.join(tokens)
	encod=tokenizer.texts_to_matrix([line],mode='freq')
	ypred=model.predict(encod,verbose=0)
	return ypred

vocab_filename='vocab.txt'
vocab=load_doc(vocab_filename)
vocab=vocab.split()
vocab=set(vocab)


positive_lines=process_docs('txt_sentoken/pos',vocab,True)
negative_lines=process_docs('txt_sentoken/neg',vocab,True)

docs=negative_lines+positive_lines
tokenizer=Tokenizer()
tokenizer.fit_on_texts(docs)
Xtrain=tokenizer.texts_to_matrix(docs,mode='freq')

positive_lines=process_docs('txt_sentoken/pos',vocab,False)
negative_lines=process_docs('txt_sentoken/neg',vocab,False)

docs=negative_lines+positive_lines
Xtest=tokenizer.texts_to_matrix(docs,mode='freq')

n_words=Xtest.shape[1]
Ytrain=array([0 for _ in range(900)]+[1 for _ in range(900)])
Ytest=array([0 for _ in range(100)]+[1 for _ in range(100)])

model=Sequential()
model.add(Dense(50,input_shape=(n_words,),activation='relu'))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(Xtrain,Ytrain,epochs=50,verbose=2)
loss,acc=model.evaluate(Xtest,Ytest,verbose=0)
print "Accuracy:",acc*100

print(predict_sentiment("What an outstanding movie!! I have heard all the prior generation people rave about this movie, so, I decided to check this movie out myself. I only have faint memories of having watched parts of this movie from my moms lap when she and dad were watching this in the theater. The other reason why I decided to check this out was a Super-bowl half-time debate on whether Amitabh was better than Rajesh Khanna. I could not participate in this debate for two reasons: first, I was eagerly awaiting another wardrobe malfunction for one of the cheerleaders and secondly, I only had memories of one Rajesh Khanna movie, Haathi Mere Saathi. I remember having enjoyed it very much as a child. But that alone was not enough to quantify anything. The more recent performances of AB were fresh in my mind, but after having seen this movie, I decided that Rajesh had a class of his own. His chirpy performance in this movie is really unparalleled! What an amazing performance! Amitabh, being more junior, has not equaled Rajesh, but has done his share very well. Thus, even after watching this movie, the debate will continue.",vocab,tokenizer,model))
print(round(predict_sentiment("What an outstanding movie!! I have heard all the prior generation people rave about this movie, so, I decided to check this movie out myself. I only have faint memories of having watched parts of this movie from my moms lap when she and dad were watching this in the theater. The other reason why I decided to check this out was a Super-bowl half-time debate on whether Amitabh was better than Rajesh Khanna. I could not participate in this debate for two reasons: first, I was eagerly awaiting another wardrobe malfunction for one of the cheerleaders and secondly, I only had memories of one Rajesh Khanna movie, Haathi Mere Saathi. I remember having enjoyed it very much as a child. But that alone was not enough to quantify anything. The more recent performances of AB were fresh in my mind, but after having seen this movie, I decided that Rajesh had a class of his own. His chirpy performance in this movie is really unparalleled! What an amazing performance! Amitabh, being more junior, has not equaled Rajesh, but has done his share very well. Thus, even after watching this movie, the debate will continue.",vocab,tokenizer,model)))


