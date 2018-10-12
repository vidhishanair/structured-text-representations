from data_reader import DataSet
import numpy as np
import pickle
import logging
import torch
import argparse
from models.DocumentClassificationModel import DocumentClassificationModel
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F


def load_data(config):
    train, dev, test, embeddings, vocab = pickle.load(open(config.data_file, 'rb'))
    trainset, devset, testset = DataSet(train), DataSet(dev), DataSet(test)
    vocab = dict([(v['index'],k) for k,v in vocab.items()])
    trainset.sort()
    train_batches = trainset.get_batches(config.batch_size, config.epochs, rand=True)
    dev_batches = devset.get_batches(config.batch_size, 1, rand=False)
    test_batches = testset.get_batches(config.batch_size, 1, rand=False)
    dev_batches = [i for i in dev_batches]
    test_batches = [i for i in test_batches]
    return len(train), train_batches, dev_batches, test_batches, embeddings, vocab


def get_feed_dict(batch, device):
    batch_size = len(batch)
    doc_l_matrix = np.zeros([batch_size], np.int32)
    for i, instance in enumerate(batch):
        n_sents = len(instance.token_idxs)
        doc_l_matrix[i] = n_sents
    max_doc_l = np.max(doc_l_matrix)
    max_sent_l = max([max([len(sent) for sent in doc.token_idxs]) for doc in batch])
    token_idxs_matrix = np.zeros([batch_size, max_doc_l, max_sent_l], np.int32)
    sent_l_matrix = np.zeros([batch_size, max_doc_l], np.int32)
    gold_matrix = np.zeros([batch_size], np.int32)
    mask_tokens_matrix = np.ones([batch_size, max_doc_l, max_sent_l], np.float32)
    mask_sents_matrix = np.ones([batch_size, max_doc_l], np.float32)
    for i, instance in enumerate(batch):
        n_sents = len(instance.token_idxs)
        gold_matrix[i] = instance.goldLabel
        for j, sent in enumerate(instance.token_idxs):
            token_idxs_matrix[i, j, :len(sent)] = np.asarray(sent)
            mask_tokens_matrix[i, j, len(sent):] = 0
            sent_l_matrix[i, j] = len(sent)
        mask_sents_matrix[i, n_sents:] = 0
    mask_parser_1 = np.ones([batch_size, max_doc_l, max_doc_l], np.float32)
    mask_parser_2 = np.ones([batch_size, max_doc_l, max_doc_l], np.float32)
    mask_parser_1[:, :, 0] = 0
    mask_parser_2[:, 0, :] = 0

    # if (self.config.large_data):
    #     if (batch_size * max_doc_l * max_sent_l * max_sent_l > 16 * 200000):
    #         return [batch_size * max_doc_l * max_sent_l * max_sent_l / (16 * 200000) + 1]

    feed_dict = {'token_idxs': torch.LongTensor(token_idxs_matrix).to(device), 'sent_l': torch.LongTensor(sent_l_matrix).to(device),
                 'mask_tokens': torch.LongTensor(mask_tokens_matrix).to(device), 'mask_sents': torch.LongTensor(mask_sents_matrix).to(device),
                 'doc_l': torch.LongTensor(doc_l_matrix).to(device), 'gold_labels': torch.LongTensor(gold_matrix).to(device),
                 'max_sent_l': torch.LongTensor(max_sent_l).to(device), 'max_doc_l': torch.LongTensor(max_doc_l).to(device),
                 'mask_parser_1': torch.LongTensor(mask_parser_1).to(device), 'mask_parser_2': torch.LongTensor(mask_parser_2).to(device),
                 'batch_l': torch.LongTensor(batch_size).to(device)}
    feed_dict = feed_dict
    return feed_dict


def evaluate(model, test_batches, device):
    corr_count, all_count = 0, 0
    model.eval()
    for ct, batch in test_batches:
        feed_dict = get_feed_dict(batch, device) # batch = [Instances], feed_dict = {inputs}
        output = model.forward(feed_dict)
        predictions = output.max(1)[1]
        corr_count += torch.sum(predictions == feed_dict['gold_labels']).item()
        all_count += len(batch)
        del feed_dict
    acc_test = 1.0 * corr_count / all_count
    return  acc_test


# def get_loss(output, target, criterion):
#     loss = criterion(output, target)


def run(config, device):
    import random

    hash = random.getrandbits(32)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    ah = logging.FileHandler('logs/'+str(hash)+'.log')
    ah.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    ah.setFormatter(formatter)
    logger.addHandler(ah)

    num_examples, train_batches, dev_batches, test_batches, embedding_matrix, vocab = load_data(config)
    config.n_embed, config.d_embed = embedding_matrix.shape

    config.dim_hidden = config.dim_sem+config.dim_str

    # print(config.__flags)
    # logger.critical(str(config.__flags))

    model = DocumentClassificationModel(config.n_embed, config.d_embed, config.dim_hidden, config.dim_hidden, 1, 1, pretrained=embedding_matrix, dropout=config.dropout).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    num_batches_per_epoch = int(num_examples / config.batch_size)
    num_steps = config.epochs * num_batches_per_epoch
    total_loss = 0

    for ct, batch in tqdm.tqdm(train_batches, total=num_steps):
        model.train()
        feed_dict = get_feed_dict(batch, device) # batch = [Instances], feed_dict = {inputs}
        output = model.forward(feed_dict)
        target = feed_dict['gold_labels']
        loss = criterion(output, target)
        total_loss += loss
        if(ct%config.log_period==0):
            acc_test = evaluate(model, test_batches, device)
            acc_dev = evaluate(model, dev_batches, device)
            print('Step: {} Loss: {}\n'.format(ct, loss))
            print('Test ACC: {}\n'.format(acc_test))
            print('Dev  ACC: {}\n'.format(acc_dev))
            logger.debug('Step: {} Loss: {}\n'.format(ct, loss))
            logger.debug('Test ACC: {}\n'.format(acc_test))
            logger.debug('Dev  ACC: {}\n'.format(acc_dev))
            logger.handlers[0].flush()
            total_loss = 0
        del feed_dict
        # saver.save(sess, 'my_test_model',global_step=1000)

parser = argparse.ArgumentParser(description='PyTorch Definition Generation Model')
parser.add_argument('--cuda', action='store_true', default=False, help='use CUDA')
parser.add_argument('--seed', type=int, default=1,help='random seed')
parser.add_argument('--batch_size', type=int, default=32,help='batchsize')
parser.add_argument('--lr', type=float, default=0.05,help='learning rate')
parser.add_argument('--data_file', type=str, default='data/yelp-2013/yelp-2013-all.pkl',help='location of the data corpus')
parser.add_argument('--save_path', type=str, default='./saved_models/english_seed/',help='location of the best model and generated files to save')
parser.add_argument('--word_emsize', type=int, default=300,help='size of word embeddings')

parser.add_argument('--dim_str', type=int, default=75,help='size of word embeddings')
parser.add_argument('--dim_sem', type=int, default=125,help='size of word embeddings')
parser.add_argument('--dim_output', type=int, default=4,help='size of word embeddings')
parser.add_argument('--n_embed', type=int, default=5000,help='size of word embeddings')
parser.add_argument('--d_embed', type=int, default=5000,help='size of word embeddings')
parser.add_argument('--dim_hidden', type=int, default=5000,help='size of word embeddings')

parser.add_argument('--nlayers', type=int, default=1,help='number of layers')
parser.add_argument('--nhid', type=int, default=300,help='number of hidden units per layer')
parser.add_argument('--dropout', type=float, default=0.2,help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--clip', type=float, default=5,help='gradient clip')
parser.add_argument('--log_period', type=float, default=500,help='log interval')
parser.add_argument('--epochs', type=int, default=50,help='epochs')

args = parser.parse_args()
cuda = args.cuda
total_epochs = args.epochs
dropout = args.dropout
seed = args.seed
num_layers = args.nlayers
word_emb_size = args.word_emsize
hidden_size = args.nhid
data_path = args.data_file
save_path = args.save_path
lr = args.lr
clip = args.clip
log_period = args.log_period

model_save_path = save_path + "best_model.pth"
plot_save_path = save_path + "loss.png"

torch.manual_seed(seed)
if torch.cuda.is_available():
    if not cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

device = torch.device("cuda" if args.cuda else "cpu")

run(args, device)
