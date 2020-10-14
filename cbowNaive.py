import torch
import torch.nn as nn
import sys
import os
import collections
import math
import random
import time

# 数据集采样自华尔街日报，没有标点符号，<unk>表示未知，N表示数字
with open('./data/ptb.train.txt_', 'r') as f:
    lines = f.readlines()
    raw_dataset = [st.split() for st in lines]  # 二维列表(sentences, words)
counter = collections.Counter([word for st in raw_dataset for word in st])
# less5 = dict(filter(lambda x: x[1] <5, counter.items()))
# print(less5)
counter = dict(filter(lambda x: x[1] >= 3, counter.items()))  # key是单词，value是出现次数（剔除了少于5次的单词）

# 建立索引
idx2token = [i for i in counter.keys()]
token2idx = {tk: idx for idx, tk in enumerate(idx2token)}
# 使用索引数字替换raw_dataset中的单词
dataset = [[token2idx[tk] for tk in st if tk in idx2token] for st in raw_dataset]
num_tokens = sum([len(st) for st in dataset])


# 二次采样,随机丢弃一些高频词
def discard(idx):
    # 返回True表示丢弃，概率是p
    p = max(0, 1 - math.sqrt(1e-4 / (counter[idx2token[idx]] / num_tokens)))
    return random.uniform(0, 1) < p


subsampled_dataset = [[idx for idx in st if not discard(idx)] for st in dataset]


# 获取中心词和背景词
def get_centers_and_contexts(dataset, max_window_size):
    centers, contexts = [], []
    for st in dataset:
        if len(st) < 2:
            continue
        for i in range(len(st)):
            centers.append(st[i])
            window_size = random.randint(1, max_window_size)
            contexti = st[max(0, i - window_size): i] + st[i + 1: min(len(st), i + window_size + 1)]
            contexts.append(contexti)
    return centers, contexts


all_centers, all_contexts = get_centers_and_contexts(subsampled_dataset, 5)

sampling_weights = [counter[w] ** 0.75 for w in idx2token]


# 定义数据迭代
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, centers, contexts):
        assert len(centers) == len(contexts)
        self.centers = centers
        self.contexts = contexts

    def __getitem__(self, index):
        return (self.centers[index], self.contexts[index])

    def __len__(self):
        return len(self.centers)


def batchify(data):
    """用作DataLoader的参数collate_fn: 输入是个长为batchsize的list,
    list中的每个元素都是Dataset类调用__getitem__得到的结果
    """
    max_len = max(len(context) for _, context in data)
    contexts, centers, masks = [], [], []
    for center, context in data:
        cur_len = len(context)
        contexts += [context + [0] * (max_len - cur_len)]
        masks += [[1] * cur_len + [0] * (max_len - cur_len)]
        centers += [center]
    return (torch.tensor(contexts), torch.tensor(centers), torch.tensor(masks))


batch_size = 512
num_workers = 0 if sys.platform.startswith('win32') else 4

dataset = MyDataset(all_centers,
                    all_contexts )
data_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True,
                                        collate_fn=batchify,
                                        num_workers=num_workers)


# 定义损失函数

# 输入的是一个batch
def loss(contexts, centers, mask, embed_v, embed_u):
    v = embed_v(contexts)
    # 要求只对v的mask部分求平均
    embedding_dim = v.shape[2]
    batch_size = v.shape[0]
    v_mask = v * (mask.unsqueeze(2).expand(-1, -1, embedding_dim))  # v_mask将作为padding的context词向量处理为0
    v_sum = torch.sum(v_mask, dim=1, keepdim=True)
    v_avg = v_sum / (mask.sum(dim=1, keepdim=True).unsqueeze(1).expand(-1, -1, embedding_dim))
    u = embed_u.weight.unsqueeze(0).expand(batch_size, -1, -1)
    tmp = torch.bmm(v_avg, u.permute(0, 2, 1)).squeeze().exp()
    denominator = tmp.sum(dim=1)
    numerator = torch.empty(batch_size).to(device)
    for i, num in enumerate(centers):
        numerator[i] = tmp[i][num]
    result = torch.log(numerator/denominator)
    return -result.mean()


embed_size = 100
net = nn.Sequential(
    nn.Embedding(num_embeddings=len(idx2token), embedding_dim=embed_size),  # 背景词
    nn.Embedding(num_embeddings=len(idx2token), embedding_dim=embed_size)  # 中心词
)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def train(net, lr, num_epochs):
    print("train on", device)
    net = net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    for epoch in range(num_epochs):
        start, l_sum, n = time.time(), 0.0, 0
        for batch in data_iter:
            context, centers, mask = [d.to(device) for d in batch]
            l = loss(context, centers, mask, net[0], net[1])
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            l_sum += l.cpu().item()
            n += 1
        print('epoch %d, loss %.2f, time %.2fs'
              % (epoch + 1, l_sum / n, time.time() - start))


if os.path.exists('./cbowNaive.pth'):
    net.load_state_dict(torch.load('./cbowNaive.pth'))
else:
    train(net, 0.01, 10)
    torch.save(net.state_dict(), './cbowNaive.pth')


def get_similar_tokens(query_token, k, embed):
    W = embed.weight.data
    x = W[token2idx[query_token]]
    # 添加的1e-9是为了数值稳定性
    cos = torch.matmul(W, x) / (torch.sum(W * W, dim=1) * torch.sum(x * x) + 1e-9).sqrt()
    _, topk = torch.topk(cos, k=k + 1)
    topk = topk.cpu().numpy()
    for i in topk[1:]:  # 除去输入词
        print('cosine sim=%.3f: %s' % (cos[i], (idx2token[i])))
    tmp = token2idx['intel']
    rank = (cos > cos[tmp]).sum()
    print('intel is %.3f, %d / %d' % (cos[tmp], rank, len(cos)))


get_similar_tokens('chip', 10, net[0])
