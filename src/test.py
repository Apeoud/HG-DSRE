import numpy as np
import torch
from sklearn.metrics import f1_score, precision_recall_curve, average_precision_score
from torch.autograd import Variable

from src.load import load_test, load_test_kg, load_test_path

cuda = torch.cuda.is_available()


def eval(model, noise=1, indices=1, batch_size=60, model_name=""):
    if cuda:
        model.cuda()

    bag, label, pos1, pos2, entity = load_test()
    pa_bag, pa_label, pa_pos1, pa_pos2, pb_bag, pb_label, pb_pos1, pb_pos2, mid_entity = load_test_path()
    kg_mid_entity, kg_path_relation = load_test_kg()
    entity_type = np.load("../data/type/test_ht_type.npy")

    if noise != 1:
        bag, label, pos1, pos2, entity = bag[indices], label[indices], pos1[indices], pos2[indices], entity[indices]
        pa_bag, pa_label, pa_pos1, pa_pos2, pb_bag, pb_label, pb_pos1, pb_pos2, mid_entity = pa_bag[indices], pa_label[
            indices], pa_pos1[indices], pa_pos2[indices], pb_bag[indices], pb_label[indices], pb_pos1[indices], pb_pos2[
                                                                                                 indices], mid_entity[
                                                                                                 indices]
        kg_mid_entity, kg_path_relation = kg_mid_entity[indices], kg_path_relation[indices]
    # model.eval()

    # print('test %d instances with %d NA relation' % (len(bag), list(label).count(99)))

    allprob = []
    alleval = []

    allpred = []

    for i in range(int(len(bag) / batch_size)):
        # for i in range(100):

        batch_word = bag[i * batch_size:(i + 1) * batch_size]
        batch_label = np.asarray(label[i * batch_size:(i + 1) * batch_size])
        batch_pos1 = pos1[i * batch_size:(i + 1) * batch_size]
        batch_pos2 = pos2[i * batch_size:(i + 1) * batch_size]
        batch_entity = entity[i * batch_size:(i + 1) * batch_size]

        seq_word = Variable(torch.LongTensor(np.array([s for bag in batch_word for s in bag]))).cuda()
        seq_pos1 = Variable(torch.LongTensor(np.array([s for bag in batch_pos1 for s in bag]))).cuda()
        seq_pos2 = Variable(torch.LongTensor(np.array([s for bag in batch_pos2 for s in bag]))).cuda()
        seq_entity = Variable(torch.LongTensor(np.array(batch_entity))).cuda()

        batch_length = [len(bag) for bag in batch_word]
        shape = [0]
        for j in range(len(batch_length)):
            shape.append(shape[j] + batch_length[j])

        # _, sen_0, _ = model.sentence_encoder(seq_word, seq_pos1, seq_pos2, shape)
        # sen_0 = torch.stack(sen_0)
        # sen_0 = torch.squeeze(sen_0)

        # _, prob = model.s_forward(sen_0, y_batch=batch_label)
        # _, prob = model.entity_encoder(seq_entity, sen_0, batch_label)

        # _, _, prob = model(seq_word, seq_pos1, seq_pos2, shape, batch_label)

        # base : loss0

        # def mid(model, seq_word, seq_pos1, seq_pos2, shape):
        #     embeds1 = model.word_embeddings(seq_word)
        #     pos1_emb = model.pos1_embeddings(seq_pos1)
        #     pos2_emb = model.pos2_embeddings(seq_pos2)
        #     inputs = torch.cat([embeds1, pos1_emb, pos2_emb], 2)
        #
        #     # rnn layer Bi-GRU
        #
        #     tup, _ = model.lstm(inputs)
        #
        #     tupf = tup[:, :, range(model.hidden_dim)]
        #     tupb = tup[:, :, range(model.hidden_dim, model.hidden_dim * 2)]
        #     tup = torch.add(tupf, tupb)
        #     tup = tup.contiguous()
        #
        #     tup1 = F.tanh(tup).view(-1, model.hidden_dim)
        #     tup1 = torch.matmul(tup1, model.attention_w).view(-1, model.max_sentence_size)
        #     tup1 = F.softmax(tup1).view(-1, 1, model.max_sentence_size)
        #
        #     return tup1
        #
        # att = mid(model, seq_word, seq_pos1, seq_pos2, shape)
        # na = att.cpu().data.numpy()
        # na = np.reshape(na, (-1 , 70))
        #
        # for i in na:
        #     s = list(i)
        #     sor = sorted(s)
        #     index = sorted(range(len(s)), key=lambda k: s[k])
        #     print(sor[-10:-1])
        #     print(index[-10:-1])

        _, sen_0, _ = model.sentence_encoder(seq_word, seq_pos1, seq_pos2, shape)
        sen_0 = torch.stack(sen_0)
        sen_0 = torch.squeeze(sen_0)

        # path a
        batch_word = pa_bag[i * batch_size:(i + 1) * batch_size]
        batch_pos1 = pa_pos1[i * batch_size:(i + 1) * batch_size]
        batch_pos2 = pa_pos2[i * batch_size:(i + 1) * batch_size]

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

        # path b
        batch_word = pb_bag[i * batch_size:(i + 1) * batch_size]
        batch_pos1 = pb_pos1[i * batch_size:(i + 1) * batch_size]
        batch_pos2 = pb_pos2[i * batch_size:(i + 1) * batch_size]

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

        batch_kg_mid_entity = kg_mid_entity[i * batch_size:(i + 1) * batch_size]
        seq_kg_mid_entity = Variable(torch.LongTensor(np.array([s for s in batch_kg_mid_entity]))).cuda()

        batch_kg_relation = kg_path_relation[i * batch_size:(i + 1) * batch_size]
        seq_kg_relation = Variable(torch.LongTensor(np.array([s for s in batch_kg_relation]))).cuda()

        batch_mid_entity = mid_entity[i * batch_size:(i + 1) * batch_size]
        seq_mid_entity = Variable(torch.LongTensor(np.array([s for s in batch_mid_entity]))).cuda()

        batch_entity_type = entity_type[i * batch_size:(i + 1) * batch_size]
        seq_entity_type = Variable(torch.LongTensor(np.array([s for s in batch_entity_type]))).cuda()

        _, prob = model.gcn_layer(sen_0, sen_a, sen_b, seq_entity, seq_mid_entity, seq_kg_mid_entity, seq_kg_relation,
                                  seq_entity_type,
                                  batch_label)

        allpred.extend(list(np.argmax(prob, 1)))

        for single_prob in prob:
            allprob.append(single_prob[0:99])
        for single_eval in batch_label:
            alleval.append(single_eval[0:99])

    with open("../data/case/wrong.txt", "w", encoding="utf-8") as f:
        for el in allpred:
            f.write(str(el) + "\n")

    allprob = np.reshape(np.array(allprob), (-1))
    alleval = np.reshape(np.array(alleval), (-1))

    allprob = np.asarray(allprob)
    # acc = accuracy_score(np.argmax(allprob, 1), label[:len(allprob)])
    # print('test accuracy ' + str(acc))

    # reshape prob matrix
    # allprob = np.reshape(allprob[:, 1:100], (-1))
    # eval_y = np.reshape(to_categorical(label)[:, 1:100], (-1))

    # order = np.argsort(-prob)

    precision, recall, threshold = precision_recall_curve(alleval[:len(allprob)], allprob)
    average_precision = average_precision_score(alleval[:len(allprob)], allprob)
    print('test average precision' + str(average_precision))

    np.save("../data/test_data/eval_%s.npy" % model_name, np.array(alleval))
    np.save("../data/test_data/prob_%s.npy" % model_name, np.array(allprob))

    order = np.argsort(-allprob)

    num = 2000

    top100 = order[:num]
    correct_num_100 = 0.0
    for i in top100:
        if alleval[i] == 1:
            correct_num_100 += 1.0
    print(correct_num_100 / num)

    num = 4000

    top100 = order[:num]
    correct_num_100 = 0.0
    for i in top100:
        if alleval[i] == 1:
            correct_num_100 += 1.0
    print(correct_num_100 / num)

    num = 10000

    top100 = order[:num]
    correct_num_100 = 0.0
    for i in top100:
        if alleval[i] == 1:
            correct_num_100 += 1.0
    print(correct_num_100 / num)

    top100 = order[:20000]
    y_true = alleval[top100]
    y_pred = []
    for s in allprob[order[:20000]]:
        if s > 0.1:
            y_pred.append(1)
        else:
            y_pred.append(0)

    print(f1_score(y_true=y_true, y_pred=y_pred))

    np.save("../data/test_data/eval_20000_%s.npy" % model_name, np.array(alleval[order[:20000]]))
    np.save("../data/test_data/prob-20000_%s.npy" % model_name, np.array(allprob[order[:20000]]))

    # f1 = f1_score(alleval[:len(allprob)], allprob)
    # print(f1)

    return average_precision


if __name__ == "__main__":
    word2vec = np.load("../data/word2vec.npy")
    kg_re2vec = np.load("../data/kg_emb/kg_re_embedding.npy")
    entity2vec = np.load("../data/kg_emb/entity_embedding.npy")

    model = torch.load("../data/model/sentence_model_4")
    eval(model, model_name="sentence_model_4")