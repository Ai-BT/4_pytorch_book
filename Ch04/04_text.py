# %%
from string import punctuation
import torch
import numpy as np

# 1. txt 파일 읽기
with open('../Ch04/1342-0.txt', encoding='utf8') as f:
    text = f.read()

lines = text.split('\n')
line = lines[200]
line # 확인 70개 문자

# %%
# 2. ASCII 제한인 128로 하드코딩
letter_t = torch.zeros(len(line), 128) 
letter_t.shape

# %%
# 3. 원핫 인코딩된 문자 하나를 담고 있으며, 정확한 위치에 1을 기록 (문자 원핫 인코딩)
for i, letter in enumerate(line.lower().strip()): # 소문자, 공백제거
    letter_index = ord(letter) if ord(letter) < 128 else 0
    letter_t[i][letter_index] = 1

# *** 참고 ***
# lower() - 소문자 변환
# strip() - 공백 제거
# ord() - 유니코드 정수를 반환

# %%
# 4. 모든 단어를 원핫 인코딩하기
# 1개 라인 테스트 진행
def clean_words(input_str):
    punctuation = '.,;:"!?”“_-'
    word_list = input_str.lower().replace('\n',' ').split()
    word_list = [word.strip(punctuation) for word in word_list]
    return word_list

words_in_line = clean_words(line)
line, words_in_line

# %%
# 5. 인코딩에서 모든 단어를 인덱스로 매핑
word_list = sorted(set(clean_words(text)))
word2index_dict = {word: i for (i, word) in enumerate(word_list)}

len(word2index_dict), word2index_dict['impossible']
# %%
word_t = torch.zeros(len(words_in_line), len(word2index_dict))
for i, word in enumerate(word_inwords_in_line_line):
    word_index = word2index_dict[word]
    word_t[i][word_index] = 1
    print('{:2} {:4} {}'.format(i, word_index, word))

print(word_t.shape)
# %%
