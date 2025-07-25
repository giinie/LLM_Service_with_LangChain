# gensim은 자연어 처리를 위한 파이썬 라이브러리로, 문서 유사성 분석을 위해 사용됩니다.
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

glove_path = '../Data/glove.6B.100d.txt'

with open(glove_path, 'w') as f:
    f.write("cat 0.5 0.3 0.2\n")
    f.write("dog 0.4 0.7 0.8\n")

# GloVe 파일 형식을 word2vec 형식으로 변환
word2vec_output_file = glove_path + '.word2vec'
glove2word2vec(glove_path, word2vec_output_file)

model = KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False)
print(model['cat']) # 'cat'에 대한 벡터
print(model.most_similar('cat'))
