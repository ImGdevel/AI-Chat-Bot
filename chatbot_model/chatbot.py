import random
import json
import pickle
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras.models import load_model

from util import *
import re
from konlpy.tag import Komoran
komoran = Komoran()

with open('intents.json', 'r', encoding='utf-8') as f:
    intents = json.load(f)

with open('words.pkl', 'rb') as f:
  all_words = pickle.load(f)

with open('classes.pkl', 'rb') as f:
  classes = pickle.load(f)

model = load_model('chatbot_model.h5')

def clean_up_sentence(sentence):
  sentence = re.sub(r'[^\w\s]', '', sentence) # 모든 구두점 제거
  w = komoran.pos(sentence) # 형태소 분석
  w = custom_morphs(w)  # 품사를 따져 불필요한 것은 버림
  #print("w: ", w)
  return w

def predict_class(sentence):
  tokenized_sentence = clean_up_sentence(sentence)
  bow = bag_of_words(tokenized_sentence, all_words)
  bow = np.array(bow).reshape(1, -1)
  res = model.predict(bow)

  # 지나치게 확률이 낮은 항목은 제외
  threshold = 0.01
  results = [[i, r] for i, r in enumerate(res[0]) if r > threshold]

  # 확률이 높은 순으로 정렬
  results.sort(key=lambda x: x[1], reverse=True)
  
  pred_list = {}
  for r in results:
    pred_list[classes[r[0]]] = r[1]
  
  return pred_list

def get_response(pred_list, intents_json):
  for key, value in pred_list.items():
    tag = key
    prob = value
    break
  
  if prob < 0.8:
    return "무슨 말씀이신지..."

  list_of_intents = intents_json['intents']
  for item in list_of_intents:
    if item['tag'] == tag:
      result = random.choice(item['responses'])
      break
  return result

bot_name = "로봇"
start_message = "채팅 시작. 마치려면 'quit'이나 'exit', '끝'을 입력하세요"
print(start_message)

while True:
  sentence = input("나: ")
  if sentence == 'quit' or sentence == 'exit' or sentence == '끝': break
  if sentence.strip() == "": continue
  pred_list = predict_class(sentence)
  res = get_response(pred_list, intents) #intents: json contents
  print(bot_name, ": ", res)
  #print(res)

'''
pred_list
{'age': 0.674383, 'greetings': 0.32546297}

predict_class(sentence)
[{'intent': 'age', 'probability': '0.674383'},
 {'intent': 'greetings', 'probability': '0.32546297'}]
 '''