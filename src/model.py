import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable

import torch.nn.functional as F

from src.load import to_categorical


class Encoder(nn.Module):
    def __init__(self, word2vec):
        super(Encoder, self).__init__()

        # hyper-parameters
        self.word_embedding_size = 50
        self.sentence_representation = 50
        self.bag_size = 20
        self.windows_size = 3
        self.kernel_num = 200
        self.vocabulary_size = word2vec.shape[0]
        self.pos_embedding_size = 5
        self.relation_num = 100
        self.max_sentence_size = 70
        self.pos_size = 120
        self.hidden_dim = 200

        # embedding layer
        self.embedding = nn.Embedding(self.vocabulary_size, self.word_embedding_size)
        self.pos_embedding = nn.Embedding(self.pos_size, self.pos_embedding_size)
        # self.position2_embedding = nn.Embedding(1, self.pos_embedding_size)

        self.embedding.weight.data.copy_(torch.from_numpy(word2vec))

        # part1:sentence encoder layer
        self.sentence_encoder = nn.Sequential(
            nn.Conv2d(1, self.kernel_num, (self.word_embedding_size, self.windows_size),
                      stride=(self.word_embedding_size + 1, 1)),
            nn.ReLU(),
            nn.MaxPool2d((1, self.word_embedding_size - self.windows_size + 1))
        )

        # part1:attention layer over bag, calculate the similarity between sentence encoder and relation embedding in KG
        self.attention = nn.Sequential(
            nn.Linear(self.kernel_num, 1),
            nn.Tanh(),
        )

        # part1:classifier layer
        self.classifier = nn.Sequential(
            nn.Linear(self.kernel_num, self.relation_num),
            # nn.ReLU()
        )

        # loss
        self.lr = nn.BCEWithLogitsLoss()

        # gru
        self.lstm = nn.GRU(self.word_embedding_size + 10, self.kernel_num, bidirectional=True, dropout=0.2,
                           batch_first=True)

        self.attention_w = Variable(torch.FloatTensor(self.kernel_num, 1), requires_grad=True).cuda()
        nn.init.xavier_uniform(self.attention_w)

        # self-define layer
        self.sen_a = Variable(torch.FloatTensor(self.kernel_num), requires_grad=True).cuda()
        nn.init.normal(self.sen_a)
        # nn.init.xavier_uniform(self.sen_a)

        self.sen_r = Variable(torch.FloatTensor(self.kernel_num, 1), requires_grad=True).cuda()
        nn.init.normal(self.sen_r)
        # nn.init.xavier_uniform(self.sen_r)

        self.relation_embedding = Variable(torch.FloatTensor(100, self.kernel_num), requires_grad=True).cuda()
        nn.init.normal(self.relation_embedding)
        # nn.init.xavier_uniform(self.relation_embedding)

        self.sen_d = Variable(torch.FloatTensor(100), requires_grad=True).cuda()
        nn.init.normal(self.sen_d)
        # nn.init.xavier_uniform(self.sen_d)

    def forward_single(self, sentence, pos1, pos2):
        # 输入的bag 2D-tensor n_sample * sentence_length

        word_embedding = self.embedding(sentence).view(-1, 1, self.max_sentence_size, self.word_embedding_size)
        pos1_embedding = self.pos_embedding(pos1).view(-1, 1, self.max_sentence_size, self.pos_embedding_size)
        pos2_embedding = self.pos_embedding(pos2).view(-1, 1, self.max_sentence_size, self.pos_embedding_size)

        input_embedding = torch.cat((word_embedding, pos1_embedding, pos2_embedding), -1).view(-1,
                                                                                               self.max_sentence_size,
                                                                                               self.word_embedding_size + 2 * self.pos_embedding_size)

        # H = self.sentence_encoder(input_embedding).view(-1, self.kernel_num)

        tup, _ = self.lstm(input_embedding)

        tupf = tup[:, :, range(self.hidden_dim)]
        tupb = tup[:, :, range(self.hidden_dim, self.hidden_dim * 2)]
        tup = torch.add(tupf, tupb)
        tup = tup.contiguous()

        tup1 = F.tanh(tup).view(-1, self.hidden_dim)

        tup1 = torch.matmul(tup1, self.attention_w).view(-1, 70)

        tup1 = F.softmax(tup1).view(-1, 1, 70)

        H = torch.matmul(tup1, tup).view(-1, self.hidden_dim)

        # A = self.attention(H)
        # A = torch.transpose(A, 1, 0)  # KxN
        # A = F.softmax(A, dim=1)  # softmax over N
        A = F.softmax(torch.matmul(torch.mul(H, self.sen_a), self.sen_r)).view(1, -1)

        S = torch.matmul(A, H).view(-1, 1)

        Out = torch.add(torch.matmul(self.relation_embedding, S).view(-1), self.sen_d).view(1, -1)

        # Out = self.classifier(M)

        return Out

    def forward(self, sentence_bag, label_bag, pos1_bag, pos2_bag):
        label_bag = to_categorical(label_bag, 100)

        self.loss = []
        prob = []

        for i in range(len(sentence_bag)):
            outputs = self.forward_single(Variable(torch.LongTensor(sentence_bag[i])).cuda(),
                                          Variable(torch.LongTensor(pos1_bag[i])).cuda(),
                                          Variable(torch.LongTensor(pos2_bag[i])).cuda())
            prob.append(outputs.data[0])
            target = Variable(torch.FloatTensor(np.asarray(label_bag[i]).reshape((1, -1)))).cuda()

            self.loss.append(self.lr(outputs, target))

            if i == 0:
                self.total_loss = self.loss[i]
            else:
                self.total_loss += self.loss[i]

        return self.total_loss, prob

    def cal_loss(self, outputs, target):

        return torch.mean(self.lr(outputs, target))


class RNN(nn.Module):
    def __init__(self, word2vec, entity2vec, kg_re2vec, hidden_dim, big_num):
        super(RNN, self).__init__()

        embedding_dim = word2vec.shape[1]
        vocab_size = word2vec.shape[0]

        # hyper-parameters
        self.word_embedding_size = 50
        self.sentence_representation = 50
        self.bag_size = 20
        self.windows_size = 3
        self.kernel_num = hidden_dim
        self.vocabulary_size = word2vec.shape[0]
        self.pos_embedding_size = 5
        self.relation_num = 100
        self.max_sentence_size = 70
        self.pos_size = 120
        self.hidden_dim = 200
        self.entity_size = 38000
        self.entity_embedding_size = entity2vec.shape[1]

        self.big_num = big_num
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.word_embeddings.weight = nn.Parameter(torch.from_numpy(word2vec))

        self.pos1_embeddings = nn.Embedding(123, 5)
        # nn.init.xavier_uniform(self.pos1_embeddings.weight)
        nn.init.normal(self.pos1_embeddings.weight)
        self.pos2_embeddings = nn.Embedding(123, 5)
        nn.init.normal(self.pos2_embeddings.weight)
        # nn.init.xavier_uniform(self.pos2_embeddings.weight)

        self.entity_embeddings = nn.Embedding(entity2vec.shape[0], entity2vec.shape[1])
        self.entity_embeddings.weight = nn.Parameter(torch.from_numpy(entity2vec))

        self.kg_re_embeddings = nn.Embedding(kg_re2vec.shape[0], kg_re2vec.shape[1])
        self.kg_re_embeddings.weight = nn.Parameter(torch.from_numpy(kg_re2vec))

        self.type_embeddings = nn.Embedding(58, 50)
        nn.init.normal(self.type_embeddings.weight)

        self.lstm = nn.GRU(embedding_dim + 10, hidden_dim, bidirectional=True, dropout=0.2, batch_first=True)
        # self.bilinear=nn.Linear(hidden_dim*2,hidden_dim)

        self.attention_w = Variable(torch.FloatTensor(hidden_dim, 1), requires_grad=True).cuda()
        nn.init.xavier_uniform(self.attention_w)

        self.W = Variable(torch.FloatTensor(hidden_dim, hidden_dim), requires_grad=True).cuda()
        nn.init.normal(self.W)

        self.sen_a = Variable(torch.FloatTensor(hidden_dim), requires_grad=True).cuda()
        nn.init.normal(self.sen_a)
        # nn.init.xavier_uniform(self.sen_a)

        self.sen_r = Variable(torch.FloatTensor(hidden_dim, 1), requires_grad=True).cuda()
        nn.init.normal(self.sen_r)
        # nn.init.xavier_uniform(self.sen_r)

        self.relation_embedding = Variable(torch.FloatTensor(hidden_dim, 100), requires_grad=True).cuda()
        nn.init.normal(self.relation_embedding)
        # nn.init.xavier_uniform(self.relation_embedding)

        self.relation_embedding2 = Variable(torch.FloatTensor(hidden_dim * 3 + 6 * self.entity_embedding_size + 100, 100),
                                            requires_grad=True).cuda()
        nn.init.normal(self.relation_embedding2)

        self.sen_d = Variable(torch.FloatTensor(100), requires_grad=True).cuda()
        nn.init.normal(self.sen_d)
        # nn.init.xavier_uniform(self.sen_d)

        self.ls = nn.BCEWithLogitsLoss()

        self.ce = nn.CrossEntropyLoss()

    def sentence_encoder(self, sentence, pos1, pos2, total_shape):
        # embedding layer

        embeds1 = self.word_embeddings(sentence)
        pos1_emb = self.pos1_embeddings(pos1)
        pos2_emb = self.pos2_embeddings(pos2)
        inputs = torch.cat([embeds1, pos1_emb, pos2_emb], 2)

        # rnn layer Bi-GRU

        tup, _ = self.lstm(inputs)

        tupf = tup[:, :, range(self.hidden_dim)]
        tupb = tup[:, :, range(self.hidden_dim, self.hidden_dim * 2)]
        tup = torch.add(tupf, tupb)
        tup = tup.contiguous()

        tup1 = F.tanh(tup).view(-1, self.hidden_dim)
        tup1 = torch.matmul(tup1, self.attention_w).view(-1, self.max_sentence_size)
        tup1 = F.softmax(tup1).view(-1, 1, self.max_sentence_size)

        # bag attention layer
        attention_r = torch.matmul(tup1, tup).view(-1, self.hidden_dim)
        sen_repre = []
        sen_alpha = []
        sen_s = []
        sen_out = []
        prob = []
        prob2 = []

        for i in range(len(total_shape) - 1):
            sen_repre.append(F.tanh(attention_r[total_shape[i]:total_shape[i + 1]]))

            batch_size = total_shape[i + 1] - total_shape[i]

            sen_alpha.append(
                F.softmax(torch.matmul(torch.mul(sen_repre[i], self.sen_a), self.sen_r).view(batch_size)).view(1,
                                                                                                               batch_size))

            sen_s.append(torch.matmul(sen_alpha[i], sen_repre[i]).view(self.hidden_dim, 1))

            # sen_out.append(
            #     torch.add(torch.matmul(sen_s[i], self.relation_embedding).view(self.relation_num), self.sen_d))

            # prob.append(F.softmax(sen_out[i]))
            # prob2.append(F.softmax(sen_out[i]).cpu().data.numpy())

        return sen_out, sen_s, prob2

    def entity_encoder(self, seq_word, seq_pos1, seq_pos2, shape, seq_entity, batch_label):

        _, sen_0, _ = self.sentence_encoder(seq_word, seq_pos1, seq_pos2, shape)

        sen_0 = torch.stack(sen_0)
        sen_0 = torch.squeeze(sen_0)

        embeds1 = self.entity_embeddings(seq_entity).view(-1, 2 * 100)
        #
        N = torch.cat([embeds1, sen_0.view(-1, self.kernel_num)], 1)

        N = F.relu(N)

        # A = Variable(torch.FloatTensor(np.array([[1, 0, 1], [0, 0, 0], [0, 0, 1]]))).cuda()

        # t = torch.matmul(A, N)

        # o = torch.matmul(N, self.W)
        #
        # o = F.relu(o).view(-1, 3 * self.kernel_num)

        o = torch.add(torch.matmul(N, self.relation_embedding2).view(-1, self.relation_num), self.sen_d)

        prob = F.softmax(o, 1).cpu().data.numpy()

        loss = self.ce(o, Variable(torch.LongTensor(np.argmax(batch_label, 1))).cuda())

        return loss, prob

    def s_forward(self, out, entity, y_batch):

        # embeds1 = self.entity_embeddings(entity).view(-1, 2 * self.entity_embedding_size)

        # o = torch.cat((embeds1, out), 1)

        o = torch.add(torch.matmul(out, self.relation_embedding).view(-1, self.relation_num), self.sen_d)

        prob = F.softmax(o, 1).cpu().data.numpy()

        total_loss = self.ce(o, Variable(torch.LongTensor(np.argmax(y_batch, 1))).cuda())

        # loss = [torch.mean(self.ls(out[i], Variable(torch.from_numpy(y_batch[i].astype(np.float32))).cuda())) for i in
        #         range(len(out))]
        #
        # if len(loss) > 1:
        #     total_loss = loss[0]
        #     for lo in loss[1:]:
        #         total_loss += lo
        # else:
        #     total_loss = loss[0]

        return total_loss, prob

    def forward(self, sentence, pos1, pos2, total_shape, y_batch):
        out, _, prob = self.sentence_encoder(sentence, pos1, pos2, total_shape)

        loss = [torch.mean(self.ls(out[i], Variable(torch.from_numpy(y_batch[i].astype(np.float32))).cuda())) for i in
                range(len(out))]

        if len(loss) > 1:
            total_loss = loss[0]
            for lo in loss[1:]:
                total_loss += lo
        else:
            total_loss = loss[0]

        return total_loss, _, prob

    def path_add(self, sen_a, sen_b, sen_c):

        #

        return

    def gcn_layer(self, s, sa, sb, ht, e, kg_e, kg_r, t_ht, batch_label):

        emb_ht = self.entity_embeddings(ht).view(-1, 2 * 100)
        emd_e = self.entity_embeddings(e).view(-1, 100)

        emd_kg_e = self.entity_embeddings(kg_e).view(-1, 100)
        emd_kg_r = self.kg_re_embeddings(kg_r).view(-1, 2*100)

        emd_t_ht = self.type_embeddings(t_ht).view(-1, 2 * 50)

        N = torch.cat((emb_ht, emd_e, emd_kg_e, emd_kg_r, s, sa, sb, emd_t_ht), 1)

        N = F.relu(N)

        o = torch.add(torch.matmul(N, self.relation_embedding2).view(-1, self.relation_num), self.sen_d)

        prob = F.softmax(o, 1).cpu().data.numpy()

        loss = self.ce(o, Variable(torch.LongTensor(np.argmax(batch_label, 1))).cuda())

        return loss, prob


if __name__ == "__main__":
    model = Encoder()
    inp1 = Variable(torch.LongTensor(np.random.randint(0, 100, (100, 1))))
    bag_sen = np.asarray([np.random.randint(0, 100, (100, 1)) for _ in range(13)])
    bag = Variable(torch.LongTensor(bag_sen)).view(13, 100)
    o = model(bag)
    print(o)
