import random
import json
import pickle
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
from keras.optimizers import legacy as keras_legacy
from util import *
import re
from konlpy.tag import Komoran
komoran = Komoran()

with open('intents.json', 'r', encoding='utf-8') as f:
    intents = json.load(f)
#intents = json.loads(open('intents.json').read()) 

words = []
classes = []
documents = []
#ignore_letters = ['?', '!', '.', ',', '~', '\'']

for intent in intents['intents']:
  for pattern in intent['patterns']:
    pattern = re.sub(r'[^\w\s]', '', pattern) # 모든 구두점 제거
    w = komoran.pos(pattern) # 형태소 분석
    w = custom_morphs(w)  # 품사를 따져 불필요한 것은 버림
    words.extend(w)   # 어휘들에 추가
    documents.append((w, intent['tag']))
    if intent['tag'] not in classes:
      classes.append(intent['tag'])

words = sorted(set(words)) # 중복 어휘들을 제거하여 어휘 사전을 만듦
classes = sorted(set(classes)) # 중복 태그를 제외하고 분류 태그를 만듦

pickle.dump(words, open('words.pkl', 'wb'))  # 어휘 사전을 파일로 저장
pickle.dump(classes, open('classes.pkl', 'wb')) # 분류 목록을 파일로 저장

training = []
output_empty = [0] * len(classes) # [0, 0, 0, 0, 0, 0, 0]

class2index = {}
for i, class_name in enumerate(classes):
  class2index[class_name] = i

index2class = {} #np.array(classes)
for i, class_name in enumerate(classes):
  index2class[i] = class_name

for document in documents:
  bag = []
  word_patterns = document[0]

  for word in words:
    bag.append(1) if word in word_patterns else bag.append(0)
  #print(bag)
  output_row = list(output_empty)
  output_row[class2index[document[1]]] = 1
  training.append([bag, output_row])


random.shuffle(training)
training = np.array(training, dtype=object)

train_x  = list(training[:, 0])
train_y  = list(training[:, 1])

vocab_size = len(words)
tag_size = len(classes)

model = Sequential()
model.add(Dense(units=128, input_shape=(vocab_size, ), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(tag_size, activation='softmax'))

sgd = keras_legacy.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

model.summary()

history = model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1)
model.save('chatbot_model.h5', save_format='tf', include_optimizer=True)
