from data_reader import Corpus
import argparse

import pickle
def main(train_path, dev_path, test_path):
    corpus = Corpus()
    corpus.load(train_path, 'train')
    corpus.load(dev_path, 'dev')
    corpus.load(test_path, 'test')
    corpus.preprocess()
    options =  dict(max_sents=60, max_tokens=100, skip_gram=False, emb_size=200)
    print('Start training word embeddings')
    corpus.w2v(options)

    instance, instance_dev, instance_test, embeddings, vocab = corpus.prepare(options)
    pickle.dump((instance[0:2000], instance_dev[0:800], instance_test[0:800], embeddings, vocab),open('data/yelp-2013/yelp-2013-small2.pkl','wb'))


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('train_path', action="store")
parser.add_argument('dev_path', action="store")
parser.add_argument('test_path', action="store")
args = parser.parse_args()

# train_path = '../data/yelp-2013.train'
# dev_path = '../data/yelp-2013.dev'
# test_path = '../data/yelp-2013.test'

main(args.train_path, args.dev_path, args.test_path)
