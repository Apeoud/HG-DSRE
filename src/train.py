import datetime

import numpy as np
import torch
import torch.optim as optim
from torch.autograd import Variable

from src.load import load_train, load_train_path, load_train_kg
from src.model import RNN

cuda = torch.cuda.is_available()

if cuda:
    torch.cuda.manual_seed(1)


def train(word2vec, entity2vec, kg_re2vec, batch_size=50):
    train_bag, train_label, train_pos1, train_pos2, train_entity = load_train()
    pa_bag, pa_label, pa_pos1, pa_pos2, pb_bag, pb_label, pb_pos1, pb_pos2, mid_entity = load_train_path()
    kg_mid_entity, kg_path_relation = load_train_kg()
    entity_type = np.load("../data/type/train_ht_type.npy")

    pa_label = pa_label.reshape((-1, 100))
    pb_label = pb_label.reshape((-1, 100))
    # test_bag, test_label, test_pos1, test_pos2 = load_test()

    # model = torch.load("../data/model/sentence_model_4")
    model = RNN(word2vec, entity2vec, kg_re2vec, 100, 50)

    if cuda:
        model.cuda()
    # loss_function = torch.nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

    for epoch in range(5):

        temp_order = list(range(len(train_bag)))

        np.random.shuffle(temp_order)

        # train_bag, train_label, train_pos1, train_pos2 = shuffle(train_bag, train_label, train_pos1, train_pos2)

        running_loss = 0.0
        print("每个epoch需要多个instance" + str(len(train_bag)))

        starttime = datetime.datetime.now()

        for i in range(int(len(train_bag) / batch_size)):

            optimizer.zero_grad()

            # 1. direct sentence encode
            index = temp_order[i * batch_size:(i + 1) * batch_size]

            batch_word = train_bag[index]
            batch_label = train_label[index]
            batch_pos1 = train_pos1[index]
            batch_pos2 = train_pos2[index]
            batch_entity = train_entity[index]

            seq_word = Variable(torch.LongTensor(np.array([s for bag in batch_word for s in bag]))).cuda()
            seq_pos1 = Variable(torch.LongTensor(np.array([s for bag in batch_pos1 for s in bag]))).cuda()
            seq_pos2 = Variable(torch.LongTensor(np.array([s for bag in batch_pos2 for s in bag]))).cuda()
            seq_entity = Variable(torch.LongTensor(np.array(batch_entity))).cuda()

            batch_length = [len(bag) for bag in batch_word]
            shape = [0]
            for j in range(len(batch_length)):
                shape.append(shape[j] + batch_length[j])

            _, sen_0, _ = model.sentence_encoder(seq_word, seq_pos1, seq_pos2, shape)
            sen_0 = torch.stack(sen_0)
            sen_0 = torch.squeeze(sen_0)
            #
            # loss_0, _ = model.s_forward(sen_0, seq_entity, y_batch=batch_label)

            # loss_0, _ = model.entity_encoder(seq_word, seq_pos1, seq_pos2, shape, seq_entity, batch_label)
            #

            # loss_0, _, _ = model(seq_word, seq_pos1, seq_pos2, shape, batch_label)

            # 2. path encode

            # 2.1 path a encode

            batch_word = pa_bag[index]
            batch_pos1 = pa_pos1[index]
            batch_pos2 = pa_pos2[index]

            seq_word = Variable(torch.LongTensor(np.array([s for bag in batch_word for s in bag]))).cuda()
            seq_pos1 = Variable(torch.LongTensor(np.array([s for bag in batch_pos1 for s in bag]))).cuda()
            seq_pos2 = Variable(torch.LongTensor(np.array([s for bag in batch_pos2 for s in bag]))).cuda()

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

            seq_word = Variable(torch.LongTensor(np.array([s for bag in batch_word for s in bag]))).cuda()
            seq_pos1 = Variable(torch.LongTensor(np.array([s for bag in batch_pos1 for s in bag]))).cuda()
            seq_pos2 = Variable(torch.LongTensor(np.array([s for bag in batch_pos2 for s in bag]))).cuda()

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
            seq_mid_entity = Variable(torch.LongTensor(np.array([s for s in batch_mid_entity]))).cuda()

            # s = [sen_a[0] + sen_b[0] for i in range(batch_size)]
            #
            # loss_path = model.s_forward(s, batch_label)

            # loss = loss_0  # + loss_a + loss_b

            batch_kg_mid_entity = kg_mid_entity[index]
            seq_kg_mid_entity = Variable(torch.LongTensor(np.array([s for s in batch_kg_mid_entity]))).cuda()

            batch_kg_relation = kg_path_relation[index]
            seq_kg_relation = Variable(torch.LongTensor(np.array([s for s in batch_kg_relation]))).cuda()

            batch_entity_type = entity_type[index]
            seq_entity_type = Variable(torch.LongTensor(np.array([s for s in batch_entity_type]))).cuda()

            loss, prob = model.gcn_layer(sen_0, sen_a, sen_b, seq_entity, seq_mid_entity, seq_kg_mid_entity,
                                         seq_kg_relation, seq_entity_type, batch_label)
            # loss = loss_0  # + 0.1 * loss_path

            # target = Variable(torch.LongTensor(np.asarray(train_label[i]).reshape((1)))).cuda()
            # loss = loss_function(outputs, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.data[0]

            if i % 100 == 99:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0

                endtime = datetime.datetime.now()
                print((endtime - starttime).seconds)
                starttime = endtime


        torch.save(model, "../data/model/sentence_model_%s" % (str(epoch)))


if __name__ == "__main__":
    word2vec = np.load("../data/word2vec.npy")
    kg_re2vec = np.load("../data/kg_emb/kg_re_embedding.npy")
    entity2vec = np.load("../data/kg_emb/entity_embedding.npy")

    train(word2vec, entity2vec, kg_re2vec)
