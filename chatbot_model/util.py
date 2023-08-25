#from konlpy.tag import Mecab
#from khaiii import KhaiiiApi
import numpy as np

# 딕셔너리 원소를 갯수만큼 출력해주는 함수
# 사용예:  printDict(my_dict, 10)
def printDict(dict_data, num = 10):
  len_dict = len(dict_data)
  if len_dict < num:
    num = len_dict
  for j, (key, value) in enumerate(dict_data.items()):
    print(key, value)
    if j + 1 >= num: break

# 형태소 분석하여 띄어쓰기 조정
def spacing_mecab(sentence):
  from konlpy.tag import Mecab
  mecab = Mecab()
  tagged = mecab.pos(sentence)
  corrected = ""
  for tag in tagged:
    if tag[1][0] in "JEXS": # S는 부호 포함, SL외국어, SH한자, SN숫자
      corrected += tag[0]
    elif tag[1] == "VCP":
      corrected += tag[0]
    elif len(tag[0].strip()) == 1 and tag[0].strip() >='ㄱ' and tag[0].strip() <='ㅎ':
      corrected += tag[0]
    else:
      corrected += " " + tag[0]
  if corrected[0] == " ":
    corrected = corrected[1:] # 문장 처음의 공백을 제거
  return corrected

# 원하는 품사만 tokenize 하여 리턴한다
def custom_morphs(tagged_sentence):
  #mecab = Mecab()
  #tagged = mecab.pos(sentence)
  #tagged = khaiii_pos(sentence)
  # 세종 품사/Mecab 품사
  # N-: 명사, VV: 동사, VA: 형용사, SN: 숫자, SL: 외국어, MAG: 부사
  corrected = []
  for tag in tagged_sentence:
    if tag[1][0] in "NV" or tag[1] == "SL" or tag[1] == "SN" or tag[1] == "MAG" or tag[1] == "MAJ":
      corrected.append(tag[0])

  return corrected

# s = "안녕하세요 오늘 기분이 어떠세요? 날씨가 비가 오려나?"
# morphs = custom_morphs(s)
# print(morphs)



def khaiii_morphs(string):
  from khaiii import KhaiiiApi
  api = KhaiiiApi()

  morphs_list = []
  analyzed = api.analyze(string)
  for word in analyzed:
    for morph in word.morphs:
      morphs_list.append(morph.lex)

  return morphs_list

def khaiii_pos(string):
  from khaiii import KhaiiiApi
  api = KhaiiiApi()

  morphs_list = []
  analyzed = api.analyze(string)
  for word in analyzed:
    for morph in word.morphs:
      morphs_list.append((morph.lex, morph.tag))
  
  return morphs_list

def bag_of_words(tokenized_sentence, all_words):
  '''
    tokenized_sentence = ["hello", 'How', 'are', 'you']
    all_words =    ["hi", "hello", "I", "you", "bye", "thank", "cool"]
    bag     =       [0,      1,     0,    1,      0,      0,      0]
    '''
  bag = np.zeros(len(all_words), dtype=np.float32)
  for idx, w in enumerate(all_words):
    if w in tokenized_sentence:
      bag[idx] = 1.0

  return bag

