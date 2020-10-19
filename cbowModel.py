import torch
import pickle
import torch.nn as nn
import sys
import os
import collections
import math
import random
import time


# 数据集采样自华尔街日报，没有标点符号，<unk>表示未知，N表示数字
# 数据预处理函数，包括建立词典，删除低频词，二次采样，负采样
def preProcess(negative_num=25, max_window_size=5):
    with open('data/ptb.train.txt', 'r') as f:
        lines = f.readlines()
        raw_dataset = [st.split() for st in lines]  # 二维列表(sentences, words)
    counter = collections.Counter([word for st in raw_dataset for word in st])
    counter = dict(filter(lambda x: x[1] >= 5, counter.items()))  # key是单词，value是出现次数（剔除了少于5次的单词）
    # 建立索引
    idx_to_token = [i for i in counter.keys()]
    token_to_idx = {tk: idx for idx, tk in enumerate(idx_to_token)}
    if os.path.exists(checkPointDir + 'data_CBOW.txt'):
        with open(checkPointDir + 'data_CBOW.txt', 'rb') as f:
            all_centers, all_contexts, all_negatives = pickle.load(f)
    else:
        # 使用索引数字替换raw_dataset中的单词
        dataset = [[token_to_idx[tk] for tk in st if tk in idx_to_token] for st in raw_dataset]
        num_tokens = sum([len(st) for st in dataset])

        # 二次采样,随机丢弃一些高频词
        def discard(idx):
            # 返回True表示丢弃，概率是p
            p = max(0, 1 - math.sqrt(1e-4 / (counter[idx_to_token[idx]] / num_tokens)))
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

        all_centers, all_contexts = get_centers_and_contexts(subsampled_dataset, max_window_size)

        # 负采样
        def get_negatives(all_contexts, sampling_weights, K):
            all_negatives, neg_candidates, i = [], [], 0
            population = list(range(len(sampling_weights)))
            for contexts in all_contexts:
                negatives = []
                while len(negatives) < K:
                    if i == len(neg_candidates):
                        # 根据每个词的权重（sampling_weights）随机生成k个词的索引作为噪声词(这里的k指random.choices中的参数）
                        # 为了高效计算，可以将k设得稍大一点
                        i, neg_candidates = 0, random.choices(
                            population, sampling_weights, k=int(1e5))
                    neg, i = neg_candidates[i], i + 1
                    # 噪声词不能是背景词
                    if neg not in set(contexts):
                        negatives.append(neg)
                all_negatives.append(negatives)
            return all_negatives

        sampling_weights = [counter[w] ** 0.75 for w in idx_to_token]
        all_negatives = get_negatives(all_contexts, sampling_weights, negative_num)
        with open(checkPointDir + 'data_CBOW.txt', 'wb') as f:
            pickle.dump((all_centers, all_contexts, all_negatives), f)

    # 定义数据迭代
    class MyDataset(torch.utils.data.Dataset):
        def __init__(self, centers, contexts, negatives):
            assert len(centers) == len(contexts) == len(negatives)
            self.centers = centers
            self.contexts = contexts
            self.negatives = negatives

        def __getitem__(self, index):
            return (self.centers[index], self.contexts[index], self.negatives[index])

        def __len__(self):
            return len(self.centers)

    def batchify(data):
        """用作DataLoader的参数collate_fn: 输入是个长为batchsize的list,
        list中的每个元素都是Dataset类调用__getitem__得到的结果
        """
        max_len = max(len(context) for _, context, _ in data)
        contexts, center_negatives, masks = [], [], []
        for center, context, negative in data:
            cur_len = len(context)
            contexts += [context + [0] * (max_len - cur_len)]
            masks += [[1] * cur_len + [0] * (max_len - cur_len)]
            center_negatives += [[center] + negative]
        return (torch.tensor(contexts), torch.tensor(center_negatives), torch.tensor(masks))

    batch_size = 512
    num_workers = 0 if sys.platform.startswith('win32') else 4

    dataset = MyDataset(all_centers,
                        all_contexts,
                        all_negatives)
    data_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True,
                                            collate_fn=batchify,
                                            num_workers=num_workers)
    return data_iter, idx_to_token, token_to_idx


# 定义损失函数

def cbow(contexts, center_and_negatives, mask, embed_v, embed_u):
    # 输入的是一个batch
    v = embed_v(contexts)
    # 要求只对v的mask部分求平均
    embedding_dim = v.shape[2]
    v_mask = v * (mask.unsqueeze(2).expand(-1, -1, embedding_dim))  # v_mask将作为padding的context词向量处理为0
    v_sum = torch.sum(v_mask, dim=1, keepdim=True)
    v_avg = v_sum / (mask.sum(dim=1, keepdim=True).unsqueeze(1).expand(-1, -1, embedding_dim))
    # 使用for循环的实现，比上面调库慢10倍
    # v_avg = []
    # for i, sample in enumerate(v):
    #     masklen = mask[i].sum()
    #     mean = sample[0:masklen].mean(dim=0, keepdim=True)
    #     v_avg.append(mean)
    # v_avg = torch.stack(v_avg,0)
    u = embed_u(center_and_negatives)
    pred = torch.bmm(v_avg, u.permute(0, 2, 1))
    return pred


class SigmoidBinaryCrossEntropyLoss(nn.Module):
    def __init__(self):  # none mean sum
        super(SigmoidBinaryCrossEntropyLoss, self).__init__()

    def forward(self, inputs, targets):
        """
        input – Tensor shape: (batch_size, len)
        target – Tensor of the same shape as input
        """
        inputs, targets = inputs.float(), targets.float()
        res = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        return res.mean(dim=1)


negative_num, max_window_size = 25, 5 # 分别是负采样的个数（应该为context长度*K），context窗口最大值
checkPointDir = "checkpoint/"
start_time = time.time()
data_iter, idx_to_token, token_to_idx = preProcess(negative_num, max_window_size)
print("pretrain time: " + str(time.time() - start_time))
loss = SigmoidBinaryCrossEntropyLoss()
embed_size = 100
net = nn.Sequential(
    nn.Embedding(num_embeddings=len(idx_to_token), embedding_dim=embed_size),  # 背景词
    nn.Embedding(num_embeddings=len(idx_to_token), embedding_dim=embed_size)  # 中心词
)


def train(net, lr, num_epochs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("train on", device)
    net = net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    for epoch in range(num_epochs):
        start, l_sum, n = time.time(), 0.0, 0
        for batch in data_iter:
            context, center_negative, mask = [d.to(device) for d in batch]
            pred = cbow(context, center_negative, mask, net[0], net[1])
            batch_size = context.shape[0]
            label = torch.tensor([1] + [0] * negative_num).float().unsqueeze(0).expand(batch_size, -1).to(device)
            l = loss(pred.view(label.size()), label).mean()  # 一个batch的平均loss
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            l_sum += l.cpu().item()
            n += 1
        print('epoch %d, loss %.2f, time %.2fs'
              % (epoch + 1, l_sum / n, time.time() - start))


if os.path.exists(checkPointDir+'cbow.pth'):
    net.load_state_dict(torch.load(checkPointDir+'cbow.pth'))
else:
    train(net, 0.01, 10)
    torch.save(net.state_dict(), checkPointDir+'cbow.pth')


def get_similar_tokens(k, embed):
    while True:
        print('input word:')
        word = input()
        W = embed.weight.data
        try:
            x = W[token_to_idx[word]]
        except:
            print("not exist")
            continue
        # 添加的1e-9是为了数值稳定性
        cos = torch.matmul(W, x) / (torch.sum(W * W, dim=1) * torch.sum(x * x) + 1e-9).sqrt()
        _, topk = torch.topk(cos, k=k + 1)
        topk = topk.cpu().numpy()
        for i in topk[1:]:  # 除去输入词
            print('cosine sim=%.3f: %s' % (cos[i], (idx_to_token[i])))
        # 判断排第几
        # tmp = token_to_idx['intel']
        # rank = (cos > cos[tmp]).sum()
        # print('intel is %.3f, %d / %d' % (cos[tmp], rank, len(cos)))


get_similar_tokens(10, net[0])
