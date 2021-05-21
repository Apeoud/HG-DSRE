import datetime
import sys

import numpy as np
import torch
import torch.optim as optim
from torch.autograd import Variable

from load import load_train, load_train_path, load_train_kg, load_entity_embedding, load_kg_relation_embedding
from model import ER_HG
from test import eval

cuda = torch.cuda.is_available()
# cuda = False

if cuda:
    torch.cuda.manual_seed(1)


def train(word2vec, entity2id, relation2id, bs=50, hd=100):
    train_bag, train_label, train_pos1, train_pos2, train_entity = load_train()
    pa_bag, pa_label, pa_pos1, pa_pos2, pb_bag, pb_label, pb_pos1, pb_pos2, mid_entity = load_train_path()
    kg_mid_entity, kg_path_relation = load_train_kg()
    entity_type = np.load("../data/type/train_ht_type.npy")

    pa_label = pa_label.reshape((-1, 100))
    pb_label = pb_label.reshape((-1, 100))
    # test_bag, test_label, test_pos1, test_pos2 = load_test()

    # model = torch.load("../data/model/sentence_model_4")
    model = ER_HG(word2vec, len(entity2id), len(relation2id), hd, bs)

    if cuda:
        model.cuda()
    # loss_function = torch.nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    maxap = 0
    patient = 0
    maxepoch = 0
    for epoch in range(50):

        temp_order = list(range(len(train_bag)))

        np.random.shuffle(temp_order)

        # train_bag, train_label, train_pos1, train_pos2 = shuffle(train_bag, train_label, train_pos1, train_pos2)

        running_loss = 0.0
        print("每个epoch需要多个instance" + str(len(train_bag)))

        starttime = datetime.datetime.now()

        for i in range(int(len(train_bag) / bs)):

            optimizer.zero_grad()

            # 1. direct sentence encode
            index = temp_order[i * bs:(i + 1) * bs]

            batch_word = train_bag[index]
            batch_label = train_label[index]
            batch_pos1 = train_pos1[index]
            batch_pos2 = train_pos2[index]
            batch_entity = train_entity[index]

            seq_word = Variable(torch.LongTensor(np.array([s for bag in batch_word for s in bag])))
            seq_pos1 = Variable(torch.LongTensor(np.array([s for bag in batch_pos1 for s in bag])))
            seq_pos2 = Variable(torch.LongTensor(np.array([s for bag in batch_pos2 for s in bag])))
            seq_entity = Variable(torch.LongTensor(np.array(batch_entity)))
            if cuda:
                seq_word = seq_word.cuda()
                seq_pos1 = seq_pos1.cuda()
                seq_pos2 = seq_pos2.cuda()
                seq_entity = seq_entity.cuda()
            batch_length = [len(bag) for bag in batch_word]
            shape = [0]
            for j in range(len(batch_length)):
                shape.append(shape[j] + batch_length[j])

            _, sen_0, _ = model.sentence_encoder(seq_word, seq_pos1, seq_pos2, shape)
            sen_0 = torch.stack(sen_0)
            sen_0 = torch.squeeze(sen_0)

            batch_word = pa_bag[index]
            batch_pos1 = pa_pos1[index]
            batch_pos2 = pa_pos2[index]

            seq_word = Variable(torch.LongTensor(np.array([s for bag in batch_word for s in bag])))
            seq_pos1 = Variable(torch.LongTensor(np.array([s for bag in batch_pos1 for s in bag])))
            seq_pos2 = Variable(torch.LongTensor(np.array([s for bag in batch_pos2 for s in bag])))

            if cuda:
                seq_word = seq_word.cuda()
                seq_pos1 = seq_pos1.cuda()
                seq_pos2 = seq_pos2.cuda()
            batch_length = [len(bag) for bag in batch_word]
            shape = [0]
            for j in range(len(batch_length)):
                shape.append(shape[j] + batch_length[j])

            _, sen_a, _ = model.sentence_encoder(seq_word, seq_pos1, seq_pos2, shape)
            sen_a = torch.stack(sen_a)
            sen_a = torch.squeeze(sen_a)

            # loss_a = model.s_forward(sen_a, y_batch=batch_label)
            # loss_a, _, _ = model(seq_word, seq_pos1, seq_pos2, shape, batch_label)

            # 2.2 path b encode

            batch_word = pb_bag[index]
            batch_pos1 = pb_pos1[index]
            batch_pos2 = pb_pos2[index]

            seq_word = Variable(torch.LongTensor(np.array([s for bag in batch_word for s in bag])))
            seq_pos1 = Variable(torch.LongTensor(np.array([s for bag in batch_pos1 for s in bag])))
            seq_pos2 = Variable(torch.LongTensor(np.array([s for bag in batch_pos2 for s in bag])))

            if cuda:
                seq_word = seq_word.cuda()
                seq_pos1 = seq_pos1.cuda()
                seq_pos2 = seq_pos2.cuda()
            batch_length = [len(bag) for bag in batch_word]
            shape = [0]
            for j in range(len(batch_length)):
                shape.append(shape[j] + batch_length[j])

            _, sen_b, _ = model.sentence_encoder(seq_word, seq_pos1, seq_pos2, shape)
            sen_b = torch.stack(sen_b)
            sen_b = torch.squeeze(sen_b)

            # loss_b, _, _ = model(seq_word, seq_pos1, seq_pos2, shape, batch_label)

            # all loss

            batch_mid_entity = mid_entity[index]
            seq_mid_entity = Variable(torch.LongTensor(np.array([s for s in batch_mid_entity])))

            if cuda:
                seq_mid_entity = seq_mid_entity.cuda()
            # s = [sen_a[0] + sen_b[0] for i in range(batch_size)]
            #
            # loss_path = model.s_forward(s, batch_label)

            # loss = loss_0  # + loss_a + loss_b

            batch_kg_mid_entity = kg_mid_entity[index]
            seq_kg_mid_entity = Variable(torch.LongTensor(np.array([s for s in batch_kg_mid_entity])))

            batch_kg_relation = kg_path_relation[index]
            seq_kg_relation = Variable(torch.LongTensor(np.array([s for s in batch_kg_relation])))

            batch_entity_type = entity_type[index]
            seq_entity_type = Variable(torch.LongTensor(np.array([s for s in batch_entity_type])))

            if cuda:
                seq_mid_entity = seq_mid_entity.cuda()
                seq_kg_mid_entity = seq_kg_mid_entity.cuda()
                seq_kg_relation = seq_kg_relation.cuda()
                seq_entity_type = seq_entity_type.cuda()

            
            loss, prob = model.er_gcn_layer(sen_0, sen_a, sen_b, seq_entity, seq_mid_entity, seq_kg_mid_entity,
                                             seq_kg_relation, seq_entity_type, batch_label)

            loss.backward()
            optimizer.step()

            # running_loss += loss.data[0]
            running_loss += loss.cpu().item()

            if i % 100 == 99:  # print every 2000 mini-batches
                print('epoch,steps:[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0

                endtime = datetime.datetime.now()
                print((endtime - starttime).seconds)
                starttime = endtime

        ap = eval(model, model_name='hgds_edge_gcn_model_' + str(epoch))
        model.train()
        if ap > maxap:
            maxap = ap
            maxepoch = epoch
            patient = 0
        else:
            patient += 1
        if patient > 4:
            break
        print('maxap:' + str(maxap) + ';in epoch:' + str(maxepoch))
        torch.save(model, "../data/model/hgds_edge_gcn_model_%s" % (str(epoch)))
    print('maxap:' + str(maxap) + ';in epoch:' + str(maxepoch))


if __name__ == "__main__":
    word2vec = np.load("../data/word2vec.npy")
    kg_re2vec = np.load("../data/kg_emb/kg_re_embedding.npy")
    entity2vec = np.load("../data/kg_emb/entity_embedding.npy")
    entity2id = load_entity_embedding()
    relation2id = load_kg_relation_embedding()
    bs = 50
    hd = 100
    print(str(sys.argv))
    if len(sys.argv) == 3:
        bs = int(sys.argv[1])
        hd = int(sys.argv[2])
    if len(sys.argv) == 4:
        bs = int(sys.argv[1])
        hd = int(sys.argv[2])
        torch.cuda.set_device(sys.argv[3])
    train(word2vec, entity2id, relation2id, bs, hd)
