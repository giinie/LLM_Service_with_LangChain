from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

glove_path = 'glove.6B.100d.txt'

with open(glove_path, 'w') as f:
    f.write("cat 0.5 0.3 0.2\n")
    f.write("dog 0.4 0.7 0.8\n")

# GloVe 파일 형식을 word2vec 형식으로 변환
word2vec_output_file = glove_path + '.word2vec'
glove2word2vec(glove_path, word2vec_output_file)

model = KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False)
print(model['cat'])
print(model.most_similar('cat'))
