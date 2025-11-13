import json
import nltk
import gensim.downloader as api
from nltk.tokenize import word_tokenize

# 1. 加载 JSON 数据集
with open('coco_karpathy_train.json', 'r') as f:
    data = json.load(f)

# 2. 分词处理并将单词转为小写
nltk.download('punkt')
captions = [item['caption'] for item in data]
tokenized_captions = [word_tokenize(caption.lower()) for caption in captions]

# 3. 加载预训练的 Word2Vec 模型
model = api.load("word2vec-google-news-300")

# 4. 构建词嵌入词典，去除重复单词
word_embedding_dict = {}
for caption in tokenized_captions:
    for word in caption:
        word = word.lower()
        if word not in word_embedding_dict and word in model:
            word_embedding_dict[word] = model[word].tolist()  # 转换为列表以便 JSON 序列化

# 5. 将词嵌入词典保存为 JSON 文件
with open('word_embedding_dict.json', 'w') as f:
    json.dump(word_embedding_dict, f)

print("词嵌入词典已保存为 JSON 文件。")
