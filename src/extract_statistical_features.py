import emoji, re, string, time, os
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import PorterStemmer, ngrams
from nltk.stem.wordnet import WordNetLemmatizer
from gensim.models import LdaModel
from gensim.corpora import MmCorpus, Dictionary
import data_processing as data_proc
import pandas as pd


# Get 13 pragmatic features (like the presence of laughter, capitalized words, emojis, hashtags, user mentions)
# but also counts of punctuation, strong affirmatives, negations, intensifiers, tokens and average token size
def get_sent_words(*args):
    url = 'https://raw.githubusercontent.com/park1200656/KnuSentiLex/master/data/SentiWord_info.json'
    df_sentiword = pd.read_json(url, orient='column')
    df_neg = df_sentiword[(df_sentiword.polarity==-1)| (df_sentiword.polarity == -2)]
    df_pos = df_sentiword[(df_sentiword.polarity==1) | (df_sentiword.polarity ==2)]
    dict_sent = {'negative':df_neg, 'positive':df_pos}
    return(dict_sent)


def get_sentiment_features(tweet_tokens):
    laughter = user_mentions = negations = affirmatives \
        = punctuation = emojis = tweet_len_ch = hashtags = 0
    neg = get_sent_words()['negative']
    pos = get_sent_words()['positive']
    
    for token in tweet_tokens:
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


# Obtain 25 POS Tags from the CMU Twitter Part-of-Speech Tagger according to the paper
# "Part-of-Speech Tagging for Twitter: Annotation, Features, and Experiments" by Gimpel et al.
def get_pos_features(pos_tweet):
    pos_dict = dict.fromkeys(['N', 'O', 'S', '^', 'Z', 'L', 'M', 'V', 'A', 'R', '!', 'D', 'P',
                              '&', 'T', 'X', 'Y', '#', '@', '~', 'U', 'E', '$', ',', 'G'], 0)
    for pos in pos_tweet:
        pos_dict[pos] += 1
    return pos_dict


# Extract the n-grams (specified as a list n = [1, 2, 3, ...])
# e.g if n = [1,2,3] then ngram_features is a dictionary of all unigrams, bigrams and trigrams
# This ngram extractor works for any kind of tokens i.e both words and pos tags
def get_ngrams(tokens, n, use_just_words=False, stem=False, for_semantics=False):
    if len(n) < 1:
        return {}
    if not for_semantics:
        if stem:
            porter = PorterStemmer()
            tokens = [porter.stem(t.lower()) for t in tokens]
        if use_just_words:
            tokens = [t.lower() for t in tokens if not t.startswith('@') and not t.startswith('#')
                      and t not in string.punctuation]
    ngram_tokens = []
    for i in n:
        print(tokens)
        for gram in ngrams(tokens, i):
            string_token = 'gram '
            for j in range(i):
                string_token += gram[j] + ' '
            ngram_tokens.append(string_token)
    ngram_features = {i: ngram_tokens.count(i) for i in set(ngram_tokens)}
    return ngram_features

# Get the necessary data to perform topic modelling
# including clean noun and verb phrases (lemmatized, lower-case)
# Tokenization and POS labelled done as advertised by CMU
def build_lda_model(tokens_tags, pos_tags, use_nouns=True, use_verbs=True, use_all=False,
                    num_of_topics=8, passes=25, verbose=True):
    path = os.getcwd()[:os.getcwd().rfind('/')]
    topics_filename = str(num_of_topics) + "topics"
    if use_nouns:
        topics_filename += "_nouns"
    if use_verbs:
        topics_filename += "_verbs"
    if use_all:
        topics_filename += "_all"

    # Set the LDA, Dictionary and Corpus filenames
    lda_filename = path + "/models/topic_models/lda_" + topics_filename + ".model"
    dict_filename = path + "/res/topic_data/dict/dict_" + topics_filename + ".dict"
    corpus_filename = path + "/res/topic_data/corpus/corpus_" + topics_filename + ".mm"

    # Build a topic model if it wasn't created yet
    if not os.path.exists(lda_filename):
        # Extract the lemmatized documents
        docs = []
        for index in range(len(tokens_tags)):
            tokens = tokens_tags[index].split()
            pos = pos_tags[index].split()
            docs.append(data_proc.extract_lemmatized_tweet(tokens, pos, use_verbs, use_nouns, use_all))

        # Compute the dictionary and save it
        dictionary = Dictionary(docs)
        dictionary.filter_extremes(keep_n=60000)
        dictionary.compactify()
        Dictionary.save(dictionary, dict_filename)

        # Compute the bow corpus and save it
        corpus = [dictionary.doc2bow(d) for d in docs]
        MmCorpus.serialize(corpus_filename, corpus)

        if verbose:
            print("\nCleaned documents:")
            print(docs)
            print("\nDictionary:")
            print(dictionary)
            print("\nCorpus in BoW form:")
            print(corpus)

        # Start training an LDA Model
        start = time.time()
        print("\nBuilding the LDA topic model...")
        lda_model = LdaModel(corpus=corpus, num_topics=num_of_topics, passes=passes, id2word=dictionary)
        lda_model.save(lda_filename)
        end = time.time()
        print("Completion time for building LDA model: %.3f s = %.3f min" % ((end - start), (end - start) / 60.0))

        if verbose:
            print("\nList of words associated with each topic:")
            lda_topics = lda_model.show_topics(formatted=False)
            lda_topics_list = [[word for word, prob in topic] for topic_id, topic in lda_topics]
            print([t for t in lda_topics_list])

    # Load the previously saved dictionary
    dictionary = Dictionary.load(dict_filename)

    # Load the previously saved corpus
    mm_corpus = MmCorpus(corpus_filename)

    # Load the previously saved LDA model
    lda_model = LdaModel.load(lda_filename)

    # Print the top 10 words for each topic
    for topic_id in range(num_of_topics):
        print("\nTop 10 words for topic ", topic_id)
        print([dictionary[word_id] for (word_id, prob) in lda_model.get_topic_terms(topic_id, topn=10)])

    index = 0
    if verbose:
        for doc_topics, word_topics, word_phis in lda_model.get_document_topics(mm_corpus, per_word_topics=True):
            print('Index ', index)
            print('Document topics:', doc_topics)
            print('Word topics:', word_topics)
            print('Phi values:', word_phis)
            print('-------------- \n')
            index += 1
    return dictionary, mm_corpus, lda_model


# Predict the topic of an unseen testing example based on the LDA model built on the train set
def get_topic_features_for_unseen_tweet(dictionary, lda_model, tokens_tags, pos_tags,
                                        use_nouns=True, use_verbs=True, use_all=False, verbose=False):
    # Extract the lemmatized documents
    docs = data_proc.extract_lemmatized_tweet(tokens_tags, pos_tags, use_verbs, use_nouns, use_all)
    tweet_bow = dictionary.doc2bow(docs)
    topic_prediction = lda_model[tweet_bow]
    topic_features = {}
    if verbose:
        print("\nTopic prediction\n")
        for t in topic_prediction:
            print(t)
    if any(isinstance(topic_list, type([])) for topic_list in topic_prediction):
        topic_prediction = topic_prediction[0]
    for topic in topic_prediction:
        topic_features['topic ' + str(topic[0])] = topic[1]
    return topic_features


# Use the distributions of topics in a tweet as features
def get_topic_features(corpus, lda_model, index):
    topic_features = {}
    doc_topics, word_topic, phi_values = lda_model.get_document_topics(corpus, per_word_topics=True)[index]
    for topic in doc_topics:
        topic_features['topic ' + str(topic[0])] = topic[1]
    return topic_features

#Todo: 임시로 1. pos제거 2. topic 제거
# Collect all features
def get_feature_set(tweets_tokens, pos_unigrams=False, pos_bigrams=False,
                    lexical=True, ngram_list=[1], sentiment=True, topic=False):
    features = []
    # if sentiment:
    #     emoji_dict = data_proc.build_emoji_sentiment_dictionary()
    #     subj_dict = data_proc.get_subj_lexicon()
    
    if topic:
        use_nouns = True
        use_verbs = True
        use_all = False
        dictionary, corpus, lda_model = build_lda_model(tweets_tokens, tweets_pos,
                                                       use_nouns=use_nouns, use_verbs=use_verbs, use_all=use_all,
                                                       num_of_topics=6, passes=20, verbose=False)
    for index in range(len(tweets_tokens)):
        tokens_this_tweet = tweets_tokens[index].split()
        # pos_this_tweet = tweets_pos[index].split()
        # pragmatic_features = {}
        pos_unigrams_features = {}
        # char_features = {}
        pos_bigrams_features = {}
        words_ngrams = {}
        sentiment_features = {}
        topic_features = {}
        # if pragmatic:
        #     pragmatic_features = get_pragmatic_features(tokens_this_tweet)

        if sentiment:
            sentiment_features = get_sentiment_features(tokens_this_tweet)
        if pos_unigrams:
            pos_unigrams_features = get_pos_features(pos_this_tweet)
        if pos_bigrams:
            pos_bigrams_features = get_ngrams(pos_this_tweet, n=[3], for_semantics=True)
        if lexical:
            words_ngrams = get_ngrams(tokens_this_tweet, n=ngram_list, use_just_words=True)
        # if sentiment:
        #     sentiment_features = get_sentiment_features(tweets_tokens[index], tokens_this_tweet,
        #                                                 pos_this_tweet, emoji_dict, subj_dict)
        if topic:
                topic_features = get_topic_features_for_unseen_tweet\
                    (dictionary, lda_model, tokens_this_tweet, pos_this_tweet,
                     use_nouns=use_nouns, use_verbs=use_verbs, use_all=use_all)
        # Append all extracted features of this tweet to the final list of all features
        features.append({**sentiment_features, **pos_unigrams_features, **pos_bigrams_features,
                         **words_ngrams, **topic_features})
    return features