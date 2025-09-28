import torch
import numpy as np
from transformers import GPT2Tokenizer, GPT2Model
from sklearn.metrics.pairwise import cosine_similarity
import torch.nn.functional as F

# 加载模型和分词器
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2Model.from_pretrained('gpt2')
wte = model.wte.weight.detach().cpu().numpy()  # 词嵌入矩阵

# 定义关键词列表
trend_keywords = ["trend", "steady", "up", "rise", "increase"]
growth_keywords = ["rapid", "boost", "slowly", "slightly", "climb"]
value_keywords = ["big", "small", "large", "little"]

all_keywords = trend_keywords + growth_keywords + value_keywords

# 获取关键词的词向量
keyword_indices = [tokenizer.encode(word, add_special_tokens=False)[0] for word in all_keywords]
keyword_embeddings = wte[keyword_indices]

# 计算所有词向量与关键词向量的余弦相似度
similarity_matrix = cosine_similarity(wte, keyword_embeddings)

# 为每个词计算综合相似度分数（取与所有关键词的最大相似度）
max_similarities = np.max(similarity_matrix, axis=1)

# 获取相似度最高的1000个词的索引
top_indices = np.argsort(max_similarities)[-500:]

# 提取这些词的词向量作为最终表示
selected_embeddings = wte[top_indices]

# 保存筛选的词嵌入
torch.save(selected_embeddings, "word_embedding_representation_500.pt")
