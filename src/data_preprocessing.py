import pandas as pd
import utils, os, emoji
from collections import Counter

# df_label_1 = pd.read_csv( path = os.getcwd()[:os.getcwd().rfind('/')] + '/IronyDetection/res/datasets/label_1.csv')
punctuation = ["?", "!", "...", '.', '..']
text_filename = "text.txt"

CUR_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = CUR_PATH.replace('src', 'res/datasets/')
EMOT_PATH = CUR_PATH.replace('src', 'res/emoji/')
ROOT_PATH = CUR_PATH.replace('src', '')

# KNU
def get_sent_words(*args):
    url = 'https://raw.githubusercontent.com/park1200656/KnuSentiLex/master/data/SentiWord_info.json'
    df_sentiword = pd.read_json(url, orient='column')
    df_neg = df_sentiword[(df_sentiword.polarity==-1)| (df_sentiword.polarity == -2)]
    df_pos = df_sentiword[(df_sentiword.polarity==1) | (df_sentiword.polarity ==2)]
    dict_sent = {'negative':df_neg, 'positive':df_pos}
    return(dict_sent)


def get_emoji_dictionary():
    # TODO: path !!!!! modify!!!!
    emojis = utils.load_file( EMOT_PATH + "emoji_list.txt")
    emoji_dict = {}
    for line in emojis:
        line = line.split(" ", 1)
        emoji = line[0]
        description = line[1]
        emoji_dict[emoji] = description
    return emoji_dict

def build_emoji_sentiment_dictionary():
    new_emoji_sentiment_filename = EMOT_PATH + "emoji_sentiment_dictionary.txt"
    if not os.path.exists(new_emoji_sentiment_filename):
        filename = EMOT_PATH + "emoji_sentiment_raw.txt"
        emojis = utils.load_file(filename)[1:]
        lines = []
        for line in emojis:
            line = line.split(",")
            emoji = line[0]
            occurences = line[2]
            negative = float(line[4]) / float(occurences)
            neutral = float(line[5]) / float(occurences)
            positive = float(line[6]) / float(occurences)
            description = line[7]
            lines.append(str(emoji) + "\t" + str(negative) + "\t" + str(neutral)
                         + "\t" + str(positive) + "\t" + description.lower())
            utils.save_file(lines, new_emoji_sentiment_filename)
    emoji_sentiment_data = utils.load_file(new_emoji_sentiment_filename)
    emoji_sentiment_dict = {}
    for line in emoji_sentiment_data:
        line = line.split("\t")
        # Get emoji characteristics as a list [negative, neutral, positive, description]
        emoji_sentiment_dict[line[0]] = [line[1], line[2], line[3], line[4]]
    return emoji_sentiment_dict

build_emoji_sentiment_dictionary()

# I have nothing in my list when i executed this function.
def extract_emojis(tweets):
    emojis = []
    for tw in tweets:
        tw.strip('\"')
        tw = tw.split(' ')
        tw_emojis = []
        for word in tw:
            chars = list(word)
            for ch in chars:
                if ch in emoji.UNICODE_EMOJI:
                    tw_emojis.append(ch)
        emojis.append(' '.join(tw_emojis))
    # utils.save_file(emojis, path+'/IronyDetection/res/emoji/detected_emoji.txt' )
    # print(emojis)
    return emojis

# # TODO: when i use data, i have to split('\n'), strip('\"')
# file_name = DATA_PATH + text_filename
# extract_emojis(utils.load_file(file_name))

# Translate emojis (or a group of emojis) into a list of descriptions
def process_emojis(word, emoji_dict, translate_emojis=True):
    processed = []
    chars = list(word)
    remaining = ""
    for c in chars:
        if c in emoji_dict.keys() or c in emoji.UNICODE_EMOJI:
            if remaining != "":
                processed.append(remaining)
                remaining = ""
            if translate_emojis:
                if c in emoji_dict:
                    processed.extend(emoji_dict[c][3].lower().split())
            else:
                processed.extend(c)
        else:
            remaining += c
    if remaining != "":
        processed.append(remaining)
    if processed != []:
        return ' '.join(processed)
    else:
        return word

def get_stopwords_list(filename="stopwords.txt"):
    stopwords = utils.load_file(path + "/res/" + filename)
    return stopwords

def build_vocabulary(vocab_filename, lines, minimum_occurrence=1):
    if not os.path.exists(vocab_filename):
        stopwords = get_stopwords_list(filename="stopwords_loose.txt")
        print("Building vocabulary...")
        vocabulary = Counter()
        for line in lines:
            vocabulary.update([l.lower() for l in line.split() if l not in stopwords])
        print("The top 10 most common words: ", vocabulary.most_common(10))
        # Filter all words that appear too rarely or too frequently to be conclusive
        vocabulary = {key: vocabulary[key] for key in vocabulary
                      if vocabulary[key] >= minimum_occurrence}
        utils.save_file(vocabulary.keys(), vocab_filename)
        print("Vocabulary saved to file \"%s\"" % vocab_filename)
    vocabulary = set(utils.load_file(vocab_filename))
    print("Loaded vocabulary of size ", len(vocabulary))
    return vocabulary


def build_vocabulary_for_dnn_tasks(vocab_filename, lines):
    if not os.path.exists(vocab_filename):
        print("Building vocabulary...")
        vocabulary = Counter()
        for line in lines:
            vocabulary.update([l.lower() for l in line])
        vocabulary = {key: vocabulary[key] for key in vocabulary}
        vocabulary = sorted(vocabulary.items(), key=lambda pair: pair[1], reverse=True)
        counter = 1
        indexed_vocabulary = {}
        for (key, _) in vocabulary:
            indexed_vocabulary[key] = counter
            counter += 1
        indexed_vocabulary['unk'] = len(indexed_vocabulary) + 1
        utils.save_dictionary(indexed_vocabulary, vocab_filename)
        print("Vocabulary saved to file \"%s\"" % vocab_filename)
    vocabulary = utils.load_dictionary(vocab_filename)
    print("Loaded vocabulary of size ", len(vocabulary))
    return vocabulary


def vocabulary_filtering(vocabulary, lines):
    filtered_lines = []
    indices = []
    for line in lines:
        filtered_line = []
        individual_word_indices = []
        for word in line:
            word = word.lower()
            if word in vocabulary:
                individual_word_indices.append(vocabulary[word])
                filtered_line.append(word)
            else:
                individual_word_indices.append(vocabulary['unk'])
                filtered_line.append('unk')
        indices.append(individual_word_indices)
        filtered_lines.append(filtered_line)
    return filtered_lines, indices


# Extract the lemmatized nouns and/or verbs from a set of documents - used in LDA modelling
def extract_lemmatized_tweet(tokens, pos, use_verbs=True, use_nouns=True, use_all=False):
    lemmatizer = WordNetLemmatizer()
    clean_data = []
    for index in range(len(tokens)):
        if use_verbs and pos[index] is 'V':
            clean_data.append(lemmatizer.lemmatize(tokens[index].lower(), 'v'))
        if use_nouns and pos[index] is 'N':
            clean_data.append(lemmatizer.lemmatize(tokens[index].lower()))
        if use_all:
            lemmatized_word = lemmatizer.lemmatize(tokens[index].lower(), 'v')
            word = lemmatizer.lemmatize(lemmatized_word)
            if pos[index] not in ['^', ',', '$', '&', '!', '#', '@']:
                clean_data.append(word)
    return clean_data


def filter_based_on_vocab(tweets, vocab_filename, min_occ=5):
    vocab = build_vocabulary(vocab_filename, tweets, minimum_occurrence=min_occ)
    filtered = []
    for tw in tweets:
        filtered.append(' '.join([t for t in tw.split() if t.lower() in vocab]))
    return filtered


def ulterior_clean(tweets, filename):
    if not os.path.exists(filename):
        stopwords = get_stopwords_list()
        lemmatizer = WordNetLemmatizer()
        filtered_tweets = []
        for tw in tweets:
            filtered_tweet = []
            for t in tw.split():
                token = t.lower()
                if token in stopwords:
                    continue
                filtered_token = lemmatizer.lemmatize(token, 'v')
                filtered_token = lemmatizer.lemmatize(filtered_token)
                filtered_tweet.append(filtered_token)
            filtered_tweets.append(' '.join(filtered_tweet))
        utils.save_file(filtered_tweets, filename)
    # Load the filtered tokens
    filtered_tweets = utils.load_file(filename)
    return filtered_tweets


def get_tags_for_each_tweet(tweets_filename, tokens_filename, pos_filename):
    if not os.path.exists(pos_filename):
        tweets = utils.load_file(tweets_filename)
        tokens_lines = []
        pos_lines = []
        tokens_line = ""
        pos_line = ""
        for t in tweets:
            if len(t) < 1:
                tokens_lines.append(tokens_line[:])
                pos_lines.append(pos_line[:])
                tokens_line = ""
                pos_line = ""
            else:
                t_split = t.split("\t")
                tokens_line += t_split[0] + " "
                pos_line += t_split[1] + " "
        utils.save_file(tokens_lines, tokens_filename)
        utils.save_file(pos_lines, pos_filename)
    # Load the tokens and the pos for the tweets in this set
    tokens = utils.load_file(tokens_filename)
    pos = utils.load_file(pos_filename)
    return tokens, pos


# Based on the probabilities of the tokenization and POS tagging obtain from CMU, get back coherent files
def cmu_probs_to_files(filename):
    # Get the tags corresponding to the test and train files
    tokens, pos = get_tags_for_each_tweet(path + "/res/cmu_tweet_tagger/" + filename,
                                          path + "/res/tokens/tokens_" + filename,
                                          path + "/res/pos/pos_" + filename)
    return tokens, pos

# Split a long, compound hash tag into its component tags. Given the character limit of tweets,
# people would stick words together to save space so this is a useful tool.
# Examples of hash splits from real data (train set) are in /stats/hashtag_splits.txt
# Implementation adapted from https://github.com/matchado/HashTagSplitter
def split_hashtag_to_words_all_possibilities(hashtag, word_dictionary):
    all_possibilities = []
    split_possibility = [hashtag[:i] in word_dictionary for i in reversed(range(len(hashtag) + 1))]
    possible_split_positions = [i for i, x in enumerate(split_possibility) if x is True]

    for split_pos in possible_split_positions:
        split_words = []
        word_1, word_2 = hashtag[:len(hashtag) - split_pos], hashtag[len(hashtag) - split_pos:]

        if word_2 in word_dictionary:
            split_words.append(word_1)
            split_words.append(word_2)
            all_possibilities.append(split_words)
            another_round = split_hashtag_to_words_all_possibilities(word_2, word_dictionary)
            if len(another_round) > 0:
                all_possibilities = all_possibilities + [[a1] + a2 for a1, a2, in
                                                         zip([word_1] * len(another_round), another_round)]
        else:
            another_round = split_hashtag_to_words_all_possibilities(word_2, word_dictionary)
            if len(another_round) > 0:
                all_possibilities = all_possibilities + [[a1] + a2 for a1, a2, in
                                                         zip([word_1] * len(another_round), another_round)]
    return all_possibilities


def split_hashtag(hashtag, word_list):
    split_words = []
    if hashtag != hashtag.lower() and hashtag != hashtag.upper():
        split_words = camel_case_split(hashtag)
    else:
        j = 0
        while j <= len(hashtag):
            loc = j
            for i in range(j + 1, len(hashtag) + 1, 1):
                if hashtag[j:i].lower() in word_list:
                    loc = i
            if loc == j:
                j += 1
            else:
                split_words.append(hashtag[j:loc])
                j = loc
    split_words = ['#' + str(s) for s in split_words]
    return split_words


# Select the best possible hashtag split based on upper-case
# or component words maximizing the length of the possible word split
def split_hashtag_long_version(hashtag):
    word_file = path + "/res/word_list.txt"
    word_list = utils.load_file(word_file).split()
    word_dictionary = list(set(words.words()))
    for alphabet in "bcdefghjklmnopqrstuvwxyz":
        word_dictionary.remove(alphabet)
    all_poss = split_hashtag_to_words_all_possibilities(hashtag.lower(), word_dictionary)
    max_p = 0
    min_len = 1000
    found = False
    best_p = []
    for poss in all_poss:
        counter = 0
        for p in poss:
            if p in word_list:
                counter += 1
        if counter == len(poss) and min_len > counter:
            found = True
            min_len = counter
            best_p = poss
        else:
            if counter > max_p and not found:
                max_p = counter
                best_p = poss
    best_p_v2 = split_hashtag(hashtag, word_list)
    if best_p != [] and best_p_v2 != []:
        split_words = best_p if len(best_p) < len(best_p_v2) else best_p_v2
    else:
        if best_p == [] and best_p_v2 == []:
            split_words = [hashtag]
        else:
            split_words = best_p if best_p_v2 == [] else best_p_v2
    split_words = ['#' + str(s) for s in split_words]
    return split_words


def split_hashtags2(hashtag, word_list, verbose=False):
    if verbose:
        print("Hashtag is %s" % hashtag)
    # Get rid of the hashtag
    if hashtag.startswith('#'):
        term = hashtag[1:]
    else:
        term = hashtag

    # If the hastag is already an existing word (a single word), return it
    if word_list is not None and term.lower() in word_list:
        return ['#' + term]
    # First, attempt splitting by CamelCase
    if term[1:] != term[1:].lower() and term[1:] != term[1:].upper():
        splits = camel_case_split(term)
    elif '#' in term:
        splits = term.split("#")
    elif len(term) > 27:
        if verbose:
            print("Hashtag %s is too big so let as it is." % term)
        splits = [term]
    else:
        # Second, build possible splits and choose the best split by assigning
        # a "score" to each possible split, based on the frequency with which a word is occurring
        penalty = -69971
        max_coverage = penalty
        max_splits = 6
        n_splits = 0
        term = re.sub(r'([0-9]+)', r' \1', term)
        term = re.sub(r'(1st|2nd|3rd|4th|5th|6th|7th|8th|9th|0th)', r'\1 ', term)
        term = re.sub(r'([A-Z][^A-Z ]+)', r' \1', term.strip())
        term = re.sub(r'([A-Z]{2,})+', r' \1', term)
        splits = term.strip().split(' ')
        if len(splits) < 3:
            # Splitting lower case and uppercase hashtags in up to 5 words
            chars = [c for c in term.lower()]
            found_all_words = False

            while n_splits < max_splits and not found_all_words:
                for index in itertools.combinations(range(0, len(chars)), n_splits):
                    output = np.split(chars, index)
                    line = [''.join(o) for o in output]
                    score = 0.0
                    for word in line:
                        stripped = word.strip()
                        if stripped in word_list:
                            score += int(word_list.get(stripped))
                        else:
                            if stripped.isnumeric():  # not stripped.isalpha():
                                score += 0.0
                            else:
                                score += penalty
                    score = score / float(len(line))
                    if score > max_coverage:
                        splits = line
                        max_coverage = score
                        line_is_valid_word = [word.strip() in word_list if not word.isnumeric()
                                              else True for word in line]
                        if all(line_is_valid_word):
                            found_all_words = True
                n_splits = n_splits + 1
    splits = ['#' + str(s) for s in splits]
    if verbose:
        print("Split to: ", splits)
    return splits


# Initial tweet cleaning - useful to filter data before tokenization
def clean_tweet(tweet, word_list, split_hashtag_method, replace_user_mentions=True,
                remove_hashtags=False, remove_emojis=False, all_to_lower_case=False):
    # Add white space before every punctuation sign so that we can split around it and keep it
    tweet = re.sub('([!?*&%"~`^+{}])', r' \1 ', tweet)
    tweet = re.sub('\s{2,}', ' ', tweet)
    tokens = tweet.split()
    valid_tokens = []
    for word in tokens:
        # Never include #sarca* hashtags
        if word.lower().startswith('#sarca'):
            continue
        # Never include URLs
        if 'http' in word:
            continue
        # Replace specific user mentions with a general user name
        if replace_user_mentions and word.startswith('@'):
            word = '@user'
        # Split or remove hashtags
        if word.startswith('#'):
            if remove_hashtags:
                continue
            splits = split_hashtag_method(word[1:], word_list)
            if all_to_lower_case:
                valid_tokens.extend([split.lower() for split in splits])
            else:
                valid_tokens.extend(splits)
            continue
        if remove_emojis and word in emoji.UNICODE_EMOJI:
            continue
        if all_to_lower_case:
            word = word.lower()
        valid_tokens.append(word)
    return ' '.join(valid_tokens)


def process_tweets(tweets, word_list, split_hashtag_method):
    clean_tweets = []
    for tweet in tweets:
        clean_tw = clean_tweet(tweet, word_list, split_hashtag_method)
        clean_tweets.append(clean_tw)
    return clean_tweets


def process_set(dataset_filename, vocab_filename, word_list, min_occ=10):
    data, labels = utils.load_data_panda(dataset_filename)
    tweets = process_tweets(data, word_list, split_hashtag)
    vocabulary = build_vocabulary(tweets, vocab_filename, minimum_occurrence=min_occ)
    filtered_tweets = []
    for tweet in tweets:
        filtered_tweets.append([t for t in tweet if t in vocabulary])
    return filtered_tweets, labels


def initial_clean(tweets, clean_filename, word_file, word_file_is_dict=False, split_hashtag_method=split_hashtag):
    if not os.path.exists(clean_filename):
        if word_file_is_dict:
            word_list = utils.load_dictionary(path + "/res/" + word_file)
        else:
            word_list = utils.load_file(path + "/res/" + word_file).split()
        filtered_tweets = process_tweets(tweets, word_list, split_hashtag_method)
        utils.save_file(filtered_tweets, clean_filename)
        return filtered_tweets
    else:
        filtered_tweets = utils.load_file(clean_filename)
        return filtered_tweets


# Return true or false depending on whether the word contains an emoji or not
def check_if_emoji(word, emoji_dict):
    emojis = list(word)
    for em in emojis:
        if em in emoji_dict.keys() or em in emoji.UNICODE_EMOJI:
            return True
    return False


# A strict clean of the twitter data - removing emojis, hashtags, URLs, user mentions
def strict_clean(tweets, filename):
    if not os.path.exists(filename):
        strict_tweets = []
        emoji_dict = get_emoji_dictionary()
        for tweet in tweets:
            strict_tweet = []
            for word in tweet.split():
                if '#' in word:
                    continue
                if '@' in word:
                    continue
                if 'http' in word:
                    continue
                if check_if_emoji(word, emoji_dict):
                    continue
                strict_tweet.append(word)
            strict_tweets.append(' '.join(strict_tweet))
        utils.save_file(strict_tweets, filename)
        return strict_tweets
    else:
        strict_tweets = utils.load_file(filename)
        return strict_tweets


# Get strictly cleaned data (designed to be applied on top of original data - e.g. original_train.txt)
def get_strict_data(train_filename, test_filename):
    # Load the train and test sets
    print("Loading data...")
    train_tweets = utils.load_file(path + "/res/data/" + train_filename)
    test_tweets = utils.load_file(path + "/res/data/" + test_filename)

    # Initial clean of data
    strict_tweets_train = strict_clean(train_tweets, path + "/res/data/strict_" + train_filename)
    strict_tweets_test = strict_clean(test_tweets, path + "/res/data/strict_" + test_filename)
    return strict_tweets_train, strict_tweets_test


# Initial clean of data (designed to be applied on top of original data - e.g. original_train.txt)
def get_clean_data(train_filename, test_filename, word_filename):
    # Load the (original) train and test sets
    print("Loading data...")
    train_tweets = utils.load_file(path + "/res/datasets/sarcasmdetection/sd_" + train_filename)
    test_tweets = utils.load_file(path + "/res/datasets/sarcasmdetection/sd_" + test_filename)
    clean_train = initial_clean(train_tweets, path + "/res/datasets/sarcasmdetection/clean_" + train_filename, word_filename,
                                word_file_is_dict=True, split_hashtag_method=split_hashtags2)
    clean_test = initial_clean(test_tweets, path + "/res/datasets/sarcasmdetection/clean_" + test_filename, word_filename,
                               word_file_is_dict=True, split_hashtag_method=split_hashtags2)
    return clean_train, clean_test


# An ulterior clean of data (designed to be applied on top of initial clean - e.g. train.txt)
def get_filtered_clean_data(train_filename, test_filename):
    # Loading the train and test sets
    print("Loading data...")
    train_tokens = utils.load_file(path + "/res/data/" + train_filename)
    test_tokens = utils.load_file(path + "/res/data/" + test_filename)
    filtered_train_tokens = ulterior_clean(train_tokens, path + "/res/data/filtered_" + train_filename)
    filtered_test_tokens = ulterior_clean(test_tokens, path + "/res/data/filtered_" + test_filename)
    return filtered_train_tokens, filtered_test_tokens


# Grammatical clean of data (designed to be applied on top of initial clean - e.g. train.txt)
def get_grammatical_data(train_filename, test_filename, dict_filename,
                         translate_emojis=True, replace_slang=True, lowercase=True):
    # Load the train and test sets
    print("Loading data...")
    train_tokens = utils.load_file(path + "/res/tokens/tokens_" + train_filename)
    train_pos = utils.load_file(path + "/res/pos/pos_" + train_filename)
    test_tokens = utils.load_file(path + "/res/tokens/tokens_" + test_filename)
    test_pos = utils.load_file(path + "/res/pos/pos_" + test_filename)

    if translate_emojis and replace_slang and lowercase:
        save_path = path + "/res/data/finest_grammatical_"
    else:
        save_path = path + "/res/data/grammatical_"

    # Clean the data and brind it to the most *grammatical* form possible
    gramm_train = grammatical_clean(train_tokens, train_pos, path + "/res/" + dict_filename, save_path + train_filename,
                                    translate_emojis=translate_emojis, replace_slang=replace_slang, lowercase=lowercase)
    gramm_test = grammatical_clean(test_tokens, test_pos, path + "/res/" + dict_filename, save_path + test_filename,
                                   translate_emojis=translate_emojis, replace_slang=replace_slang, lowercase=lowercase)
    return gramm_train, gramm_test


# Get train and test tokens, as well as indices assigned according to a vocabulary
# (designed to be applied on top of initial clean tokens - e.g. train.txt)
def get_clean_dl_data(train_filename, test_filename, word_list):
    vocab_filename = "dnn_vocabulary_" + train_filename
    # Load the train and test sets
    print("Loading data...")
    train_tweets = utils.load_file(path + "/res/tokens/tokens_" + train_filename)
    test_tweets = utils.load_file(path + "/res/tokens/tokens_" + test_filename)
    vocabulary = build_vocabulary_for_dnn_tasks(path + "/res/vocabulary/" + vocab_filename, train_tweets)
    clean_train_tweets, train_indices = vocabulary_filtering(vocabulary, train_tweets)
    clean_test_tweets, test_indices = vocabulary_filtering(vocabulary, test_tweets)
    return clean_train_tweets, train_indices, clean_test_tweets, test_indices, len(vocabulary)


def get_dataset(dataset):
    data_path = path + "/res/datasets/" + dataset + "/"
    train_tweets = utils.load_file(data_path + "tokens_train.txt")
    test_tweets = utils.load_file(data_path + "tokens_test.txt")
    train_pos = utils.load_file(data_path + "pos_train.txt")
    test_pos = utils.load_file(data_path + "pos_test.txt")
    train_labels = [int(l) for l in utils.load_file(data_path + "labels_train.txt")]
    test_labels = [int(l) for l in utils.load_file(data_path + "labels_test.txt")]
    print("Size of the train set: ", len(train_labels))
    print("Size of the test set: ", len(test_labels))
    return train_tweets, train_pos, train_labels, test_tweets, test_pos, test_labels


# if __name__ == '__main__':
#
#     train_filename = "clean_original_train.txt"
#     test_filename = "clean_original_test.txt"
#
#     # For a superficial clean
#     clean_train, clean_test = get_clean_data(train_filename, test_filename, word_filename)
#
#     # For a more aggressive clean
#     filtered_train_tokens, filtered_test_tokens = get_filtered_clean_data(train_filename, test_filename)
#
#     # For complete removal of any twitter-specific data
#     strict_tweets_train, strict_tweets_test = get_strict_data(train_filename, test_filename)
#
#     # For an attempt at a grammatical clean
#     gramm_train, gramm_test = get_grammatical_data(train_filename, test_filename, dict_filename,
#                                                    translate_emojis=False, replace_slang=False, lowercase=False)
#
#     # For a more aggressive attempt at a grammatical clean
#     finest_gramm_train, finest_gramm_test = get_grammatical_data(train_filename, test_filename, dict_filename,
#                                                                  translate_emojis=True, replace_slang=True, lowercase=True)