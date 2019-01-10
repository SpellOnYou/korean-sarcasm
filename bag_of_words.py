from utils import run_supervised_learning_models
from dl_models import nn_bow_model
from nltk.tokenize import TweetTokenizer
import time, os, utils
import fire

class BOW():
    def __init__(self, train_file, test_file):
        self.train = utils.read_csv(train_file)
        self.test = utils.read_csv(test_file)
        self.tokenizer=TweetTokenizer()

    def bow(tokenizer, x_train, x_test, y_train, y_test):
        # For each selection-mode, make a BoW analysis using both SVMs and a simple feed-forward NN
        modes = ['binary', 'count', 'tfidf', 'freq']
        for mode in modes:
            utils.print_model_title("BoW Analysis for Mode %s" % mode)
            tokenizer, x_train, x_test = utils.encode_text_as_matrix(train_tweets, test_tweets, mode, lower=True)
            word_to_indices = tokenizer.word_index
            index_to_word = {i: w for w, i in word_to_indices.items()}
            start = time.time()
            run_supervised_learning_models(x_train, y_train, x_test, y_test, make_feature_analysis=True,
                                           feature_names=index_to_word, top_features=20,
                                           plot_name="/bow_models/bow_%s_" % mode)
            nn_bow_model(x_train, y_train, x_test, y_test, results, mode, save=False, plot_graph=True)
            end = time.time()
            print("BoW for %s mode completion time: %.3f s = %.3f min" % (mode, (end - start), (end - start) / 60.0))
        # utils.boxplot_results(results, "bow_nn_boxplot.png")

def main(envir_var='config/env_vars.json',
    train_file='data/jiwon/train.csv',
    test_file='data/jiwon/test.csv',
    *arg,
    **kargs):
    
    #load configure file if user give
    cfg = utils.from_json(envir_var)
    
    inst = BOW(train_file, test_file)

if __name__ == '__main__':
    fire.Fire(main)