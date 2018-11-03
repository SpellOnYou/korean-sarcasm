from nltk.tokenize import TweetTokenizer, RegexpTokenizer
from nltk import ngrams, pos_tag
from collections import Counter
import numpy as np

def count_apparitions(tokens, list_to_count_from):
    total_count = 0.0
    for affirmative in list_to_count_from:
        total_count += tokens.count(affirmative)
    return total_count


def get_sent_words(*args):
    url = 'https://raw.githubusercontent.com/park1200656/KnuSentiLex/master/data/SentiWord_info.json'
    df_sentiword = pd.read_json(url, orient='column')
    df_neg = df_sentiword[(df_sentiword.polarity == -1) | (df_sentiword.polarity == -2)]
    df_pos = df_sentiword[(df_sentiword.polarity == 1) | (df_sentiword.polarity == 2)]
    dict_sent = {'negative': df_neg, 'positive': df_pos}
    return (dict_sent)


def get_sentiment_features(train_data):
    laughter = user_mentions = negations = affirmatives \
        = punctuation = emojis = tweet_len_ch = hashtags = 0
    neg = get_sent_words()['negative']
    pos = get_sent_words()['positive']
    
    #TODO; tokenize해야함... 아 이거 계층 ㄸ ㅂㄱ잡해지는데 ㅋㅋㅋㅋ 일단 이 파일이 baseline-features가 아니라 ml이고 ml model이고 (암튼 statistical이랑 합침)
    
    train_list = [i for i in train_data]
    
    for token in train_list:
        tweet_len_ch += len(token)
        if token.startswith("@"):
            user_mentions = 1
        if token.startswith("#"):
            hashtags += 1  # count-based feature of hashtags used (excluding sarcasm or sarcastic)
        if token == ("ㅎㅎ") or token == ('ㅋㅋ'):
            laughter = 1  # binary feature marking the presence of laughter
        if any(neg == token):
            negations += 1  #
        if any(pos == token):
            affirmatives += 1  # count-based feature of strong affirmatives
    tweet_len_tokens = len(tweet_tokens)
    average_token_length = float(tweet_len_tokens) / max(1.0, float(tweet_len_ch))  # average tweet length
    
    feature_list = {'tw_len_ch': tweet_len_ch, 'tw_len_tok': tweet_len_tokens, 'avg_len': average_token_length,
                    'laughter': laughter, 'user_mentions': user_mentions,
                    'negations': negations, 'affirmatives': affirmatives,
                    'punctuation': punctuation, 'emojis': emojis}
    return feature_list


def get_ngram_list(tknzr, text, n):
    tokens = tknzr.tokenize(text)
    tokens = [t for t in tokens if not t.startswith('#')]
    tokens = [t for t in tokens if not t.startswith('@')]
    ngram_list = [gram for gram in ngrams(tokens, n)]
    return ngram_list


def get_ngrams(tweets, n):
    unigrams = Counter()
    bigrams = Counter()
    trigrams = Counter()
    regexp_tknzr = RegexpTokenizer(r'\w+')
    tweet_tknzr = TweetTokenizer()
    for tweet in tweets:
        tweet = tweet.lower()
        # Get the unigram list for this tweet and update the unigram counter
        unigram_list = get_ngram_list(tweet_tknzr, tweet, 1)
        unigrams.update(unigram_list)
        # Get the bigram list for this tweet and update the bigram counter
        if n > 1:
            bigram_list = get_ngram_list(regexp_tknzr, tweet, 2)
            bigrams.update(bigram_list)
            # Get the trigram list for this tweet and update the trigram counter
            if n > 2:
                trigram_list = get_ngram_list(regexp_tknzr, tweet, 3)
                trigrams.update(trigram_list)
    # Update the counters such that each n-gram appears at least min_occurence times
    min_occurence = 2
    unigram_tokens = [k for k, c in unigrams.items() if c >= min_occurence]
    # In case using just unigrams, make the bigrams and trigrams empty
    bigram_tokens = trigram_tokens = []
    if n > 1:
        bigram_tokens = [k for k, c in bigrams.items() if c >= min_occurence]
    if n > 2:
        trigram_tokens = [k for k, c in trigrams.items() if c >= min_occurence]
    return unigram_tokens, bigram_tokens, trigram_tokens


def create_ngram_mapping(unigrams, bigrams, trigrams):
    ngram_map = dict()
    all_ngrams = unigrams
    all_ngrams.extend(bigrams)
    all_ngrams.extend(trigrams)
    for i in range(0, len(all_ngrams)):
        ngram_map[all_ngrams[i]] = i
    return ngram_map


def get_ngram_features_from_map(tweets, ngram_map, n):
    regexp_tknzr = RegexpTokenizer(r'\w+')
    tweet_tknzr = TweetTokenizer()
    features = []
    for tweet in tweets:
        feature_list = [0] * np.zeros(len(ngram_map))
        tweet = tweet.lower()
        ngram_list = get_ngram_list(tweet_tknzr, tweet, 1)
        if n > 1:
            ngram_list += get_ngram_list(regexp_tknzr, tweet, 2)
        if n > 2:
            ngram_list += get_ngram_list(regexp_tknzr, tweet, 3)
        for gram in ngram_list:
            if gram in ngram_map:
                feature_list[ngram_map[gram]] += 1.0
        features.append(feature_list)
    return features


def get_ngram_features(tweets, n):
    print("Getting n-gram features...")
    unigrams = []
    bigrams = []
    trigrams = []
    if n == 1:
        unigrams, _, _ = get_ngrams(tweets, n)
    if n == 2:
        unigrams, bigrams, _ = get_ngrams(tweets, n)
    if n == 3:
        unigrams, bigrams, trigrams = get_ngrams(tweets, n)
    ngram_map = create_ngram_mapping(unigrams, bigrams, trigrams)
    features = get_ngram_features_from_map(tweets, ngram_map, n)
    print("Done.")
    return ngram_map, features
