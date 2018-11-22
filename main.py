from data_reader import DataSet
import numpy as np
import pickle
import logging
import torch
import gc
import argparse
from models.DocumentClassificationModel import DocumentClassificationModel
from models.BiDirecModel import BidirectionalModel
import tqdm
import torch.optim as optim
import torch
import traceback
import torch.nn as nn
import torch.nn.functional as F


def load_data(config):
    train, dev, test, embeddings, vocab = pickle.load(open(config.data_file, 'rb'))
    trainset, devset, testset = DataSet(train), DataSet(dev), DataSet(test)
    vocab = dict([(v['index'],k) for k,v in vocab.items()])
    trainset.sort(reverse=False)
    train_batches = trainset.get_batches(config.batch_size, config.epochs, rand=True)
    dev_batches = devset.get_batches(config.batch_size, 1, rand=False)
    test_batches = testset.get_batches(config.batch_size, 1, rand=False)
    temp_train = trainset.get_batches(config.batch_size, config.epochs, rand=True)
    dev_batches = [i for i in dev_batches]
    test_batches = [i for i in test_batches]
    temp_train = [i for i in temp_train]
    return len(train), train_batches, dev_batches, test_batches, embeddings, vocab, temp_train


def get_feed_dict(batch, device):
    batch_size = len(batch)
    doc_l_matrix = np.ones([batch_size], np.int32)
    for i, instance in enumerate(batch):
        n_sents = len(instance.token_idxs)
        doc_l_matrix[i] = n_sents if n_sents>0 else 1
    max_doc_l = np.max(doc_l_matrix)
    max_sent_l = max([max([len(sent) for sent in doc.token_idxs]) for doc in batch])
    token_idxs_matrix = np.zeros([batch_size, max_doc_l, max_sent_l], np.int32)
    sent_l_matrix = np.ones([batch_size, max_doc_l], np.int32)
    gold_matrix = np.zeros([batch_size], np.int32)
    mask_tokens_matrix = np.ones([batch_size, max_doc_l, max_sent_l], np.float32)
    mask_sents_matrix = np.ones([batch_size, max_doc_l], np.float32)
    for i, instance in enumerate(batch):
        n_sents = len(instance.token_idxs)
        gold_matrix[i] = instance.goldLabel
        for j, sent in enumerate(instance.token_idxs):
            token_idxs_matrix[i, j, :len(sent)] = np.asarray(sent)
            mask_tokens_matrix[i, j, len(sent):] = 0
            sent_l_matrix[i, j] = len(sent) if len(sent)>0 else 1
        mask_sents_matrix[i, n_sents:] = 0
    mask_parser_1 = np.ones([batch_size, max_doc_l, max_doc_l], np.float32)
    mask_parser_2 = np.ones([batch_size, max_doc_l, max_doc_l], np.float32)
    mask_parser_1[:, :, 0] = 0
    mask_parser_2[:, 0, :] = 0
    # if (batch_size * max_doc_l * max_sent_l * max_sent_l > 16 * 200000):
        #print("Multi size: "+str(torch.LongTensor(token_idxs_matrix).size()))
    #    return False, [batch_size * max_doc_l * max_sent_l * max_sent_l / (16 * 200000) + 1]

    #print(max_doc_l)
    #print(max_sent_l)
    #print("S")
    if max_doc_l == 1 or max_sent_l == 1 or max_doc_l >50 or max_sent_l>30:
        #print("1 or 60 size: "+str(torch.LongTensor(token_idxs_matrix).size()))
        return False, {}
    #print(max_doc_l)
    #print(max_sent_l)
    try:
        feed_dict = {'token_idxs': torch.LongTensor(token_idxs_matrix).to(device),
                 'gold_labels': torch.LongTensor(gold_matrix).to(device),
                 'mask_tokens': torch.FloatTensor(mask_tokens_matrix).to(device),
                 'mask_sents': torch.FloatTensor(mask_sents_matrix).to(device),
                 'sent_l': sent_l_matrix,
                 'doc_l': doc_l_matrix}
    except:
        print("Here")
        return False, [batch_size * max_doc_l * max_sent_l * max_sent_l / (16 * 200000) + 1]
    return True, feed_dict


def evaluate(model, test_batches, device, criterion):
    corr_count, all_count = 0, 0
    model.eval()
    count = 0
    total_loss = 0
    for ct, batch in test_batches:
        #print("Batch : "+str(count))
        value, feed_dict = get_feed_dict(batch, device) # batch = [Instances], feed_dict = {inputs}
        if not value:
            continue
        output = model.forward(feed_dict)
        total_loss = criterion(output, feed_dict['gold_labels']).item()
        predictions = output.max(1)[1]
        corr_count += torch.sum(predictions == feed_dict['gold_labels']).item()
        #print(feed_dict['gold_labels'])
        all_count += len(batch)
        count += 1
        del feed_dict['token_idxs']
        del feed_dict['gold_labels']
        del feed_dict
        torch.cuda.empty_cache()
    print(corr_count, all_count)
    #print("Test Loss: "+str(total_loss/count))
    acc_test = 1.0 * corr_count / all_count
    return acc_test


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

    num_examples, train_batches, dev_batches, test_batches, embedding_matrix, vocab, temp_train = load_data(config)
    config.n_embed, config.d_embed = embedding_matrix.shape

    config.dim_hidden = config.dim_sem+config.dim_str

    # print(config.__flags)
    # logger.critical(str(config.__flags))

    #model = DocumentClassificationModel(device, config.n_embed, config.d_embed, config.dim_hidden, config.dim_hidden, 1, 1, config.dim_sem, pretrained=embedding_matrix, dropout=config.dropout, bidirectional=True).to(device)
    model = BidirectionalModel(device, config.n_embed, config.d_embed, config.dim_hidden, config.dim_hidden, 1, 1, config.dim_sem, pretrained=embedding_matrix, dropout=config.dropout, bidirectional=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adagrad(filter(lambda p: p.requires_grad, model.parameters()), lr=config.lr, weight_decay=0.01)

    
    num_batches_per_epoch = int(num_examples / config.batch_size)
    num_steps = config.epochs * num_batches_per_epoch
    total_loss = 0
    count = 0
    best_val = 0
    try:
        for ct, batch in tqdm.tqdm(train_batches, total=num_steps):
            if ct!= 0 and ct%config.log_period==0 :
                acc_test = evaluate(model, test_batches, device, criterion)
                acc_dev = evaluate(model, dev_batches, device, criterion)
                if acc_dev > best_val:
                    best_val = acc_dev
                print("Trained on {} batches out of {}\n".format(count, config.log_period))
                print('Step: {} Loss: {}\n'.format(ct, total_loss/count))
                print('Test ACC: {}\n'.format(acc_test))
                print('Dev  ACC: {}\n'.format(acc_dev))
                logger.debug("Trained on {} batches out of {}\n".format(count, config.log_period))
                logger.debug('Step: {} Loss: {}\n'.format(ct, total_loss/count))
                logger.debug('Test ACC: {}\n'.format(acc_test))
                logger.debug('Dev  ACC: {}\n'.format(acc_dev))
                print("Best Dev: " + str(best_val))
                logger.handlers[0].flush()
                total_loss = 0
                count = 0
            model.train()
            torch.cuda.empty_cache()
            value, feed_dict = get_feed_dict(batch, device) # batch = [Instances], feed_dict = {inputs}
            if not value:
                continue
            count += 1
            output = model.forward(feed_dict)
            target = feed_dict['gold_labels']
            loss = criterion(output, target)

            optimizer.zero_grad()
            #print(loss.item())
            loss.backward()
            torch.nn.utils.clip_grad_norm(model.parameters(), config.clip)
            optimizer.step()

            #print(loss.item())
            total_loss += loss.item()
            loss = 0
            del feed_dict['token_idxs']
            del feed_dict['gold_labels']
            torch.cuda.empty_cache()

    except Exception as e:
        print(e)
        traceback.print_exc()
        torch.cuda.empty_cache()
        for obj in gc.get_objects():
            if torch.is_tensor(obj):
                #print("GC: "+str(type(obj))+" "+str(obj.size()))
                pass

parser = argparse.ArgumentParser(description='PyTorch Definition Generation Model')
parser.add_argument('--cuda', action='store_true', default=False, help='use CUDA')
parser.add_argument('--seed', type=int, default=1,help='random seed')
parser.add_argument('--batch_size', type=int, default=32,help='batchsize')
parser.add_argument('--lr', type=float, default=0.05,help='learning rate')
parser.add_argument('--data_file', type=str, default='data/yelp-2013/yelp-2013-all.pkl',help='location of the data corpus')
parser.add_argument('--save_path', type=str, default='./saved_models/english_seed/',help='location of the best model and generated files to save')
parser.add_argument('--word_emsize', type=int, default=300,help='size of word embeddings')

parser.add_argument('--dim_str', type=int, default=50,help='size of word embeddings')
parser.add_argument('--dim_sem', type=int, default=50,help='size of word embeddings')
parser.add_argument('--dim_output', type=int, default=5,help='size of word embeddings')
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
