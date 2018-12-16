import re
# from konlpy.tag import Twitter
import pandas as pd
# from soynlp.normalizer import only_text
import data_preprocessing as data_proc
from nltk import PorterStemmer, ngrams  # TODO ; porterstemmer would be replaced by ko_utils
from nltk.sentiment import SentimentIntensityAnalyzer
# from src.data_prep import df_neg, df_pos


def get_sent_words(*args):
    url = 'https://raw.githubusercontent.com/park1200656/KnuSentiLex/master/data/SentiWord_info.json'
    df_sentiword = pd.read_json(url, orient='column')
    df_neg = df_sentiword[(df_sentiword.polarity==-1)| (df_sentiword.polarity == -2)]
    df_pos = df_sentiword[(df_sentiword.polarity==1) | (df_sentiword.polarity ==2)]
    dict_sent = {'negative':df_neg, 'positive':df_pos}
    return(dict_sent)

def get_features(tweet_tokens):
	laughter = user_mentions = negations = affirmatives \
		= punctuation = emojis = tweet_len_ch = hashtags = 0
	
	for token in tweet_tokens:
		tweet_len_ch += len(token)
		if token.startswith("@"):
			user_mentions = 1
		if token.startswith("#"):
			hashtags += 1  # count-based feature of hashtags used (excluding sarcasm or sarcastic)
		if token == ("ㅎㅎ") or token == ('ㅋㅋ'):
			laughter = 1  # binary feature marking the presence of laughter
		if any(df_neg.word_root == token):
			negations += 1  #
		if any(df_pos.word_root == token):
			affirmatives += 1  # count-based feature of strong affirmatives
	tweet_len_tokens = len(tweet_tokens)
	average_token_length = float(tweet_len_tokens) / max(1.0, float(tweet_len_ch))  # average tweet length
	
	feature_list = {'tw_len_ch': tweet_len_ch, 'tw_len_tok': tweet_len_tokens, 'avg_len': average_token_length,
	                'laughter': laughter, 'user_mentions': user_mentions,
	                'negations': negations, 'affirmatives': affirmatives,
	                'punctuation': punctuation, 'emojis': emojis}
	return feature_list


# customizing as version of korean

def get_pos_features(pos_tweet):
	pos_dict = dict.fromkeys(['Adjective', 'Adverb', 'Alpha', 'Conjunction', 'Determiner',
	                          'Eomi', 'Exclamation', 'Foreign', 'Hashtag', 'Josa', 'KoreanParticle', 'Noun',
	                          'Number', 'PreEomi', 'Punctuation', 'ScreenName', 'Suffix', 'Unknown', 'Verb'], 0)
	for pos in pos_tweet:
		pos_dict[pos] += 1
	return pos_dict

# extract n-grams
# TODO: REFER ORIGINAL CODE. I DON'T USE <use_just_words=False, stem=False, for_semantics=False> features
def get_ngrams(tokens, n):
	ngram_tokens = []
	for gram in ngrams(tokens, n):
		grams = re.split('(\s)',gram)
		string_token = 'gram '
		for j in range(grams):
			string_token += gram[j] + ' '
		ngram_tokens.append(string_token)
	ngram_features = {i: ngram_tokens.count(i) for i in set(ngram_tokens)}
	return ngram_features


def get_pragmatic_features(tweet, tweet_tokens, tweet_pos, emoji_sent_dict, subj_dict):
	sent_features = dict.fromkeys(["positive emoji", "negative emoji", "neutral emoji",
	                               "emojis pos:neg", "emojis neutral:neg",
	                               "subjlexicon weaksubj", "subjlexicon strongsubj",
	                               "subjlexicon positive", "subjlexicon negative",
	                               "subjlexicon neutral", "words pos:neg", "words neutral:neg",
	                               "subjectivity strong:weak", "total sentiment words"], 0.0)
	
	for t in tweet_tokens:
		if t in emoji_sent_dict.keys():
			sent_features['negative emoji'] += float(emoji_sent_dict[t][0])
			sent_features['neutral emoji'] += float(emoji_sent_dict[t][1])
			sent_features['positive emoji'] += float(emoji_sent_dict[t][2])
	
	# Obtain the report of positive to negative emojis
	if sent_features['negative emoji'] != 0:
		if sent_features['positive emoji'] == 0:
			sent_features['emojis pos:neg'] = -1.0 / float(sent_features['negative emoji'])
		else:
			sent_features['emojis pos:neg'] = sent_features['positive emoji'] \
			                                  / float(sent_features['negative emoji'])
	
	# Obtain the report of neutral to negative emojis
	if sent_features['negative emoji'] != 0:
		if sent_features['neutral emoji'] == 0:
			sent_features['emojis neutral:neg'] = -1.0 / float(sent_features['negative emoji'])
		else:
			sent_features['emojis neutral:neg'] = sent_features['neutral emoji'] \
			                                      / float(sent_features['negative emoji'])
	# lemmatizer is nomalizing module of soynlp
  # TODO: later tihs part should be customized after making sentiment.... part....
	# lemmatizer = only_text
 
	pos_translation = {'N': 'noun', 'V': 'verb', 'D': 'adj', 'R': 'adverb'}
	for index in range(len(tweet_tokens)):
		stemmed = lemmatizer.lemmatize(tweet_tokens[index], 'v')
		if stemmed in subj_dict.keys():
			if tweet_pos[index] in pos_translation and pos_translation[tweet_pos[index]] in subj_dict[stemmed].keys():
				# Get the type of subjectivity (strong or weak) of this stemmed word
				sent_features['subjlexicon ' + subj_dict[stemmed][pos_translation[tweet_pos[index]]][0]] += 1
				# Get the type of polarity (pos, neg, neutral) of this stemmed word
				if subj_dict[stemmed][pos_translation[tweet_pos[index]]][1] == 'both':
					sent_features['subjlexicon positive'] += 1
					sent_features['subjlexicon negative'] += 1
				else:
					sent_features['subjlexicon ' + subj_dict[stemmed][pos_translation[tweet_pos[index]]][1]] += 1
			else:
				if 'anypos' in subj_dict[stemmed].keys():
					# Get the type of subjectivity (strong or weak) of this stemmed word
					sent_features['subjlexicon ' + subj_dict[stemmed]['anypos'][0]] += 1  # strong or weak subjectivity
					# Get the type of polarity (pos, neg, neutral) of this stemmed word
					if subj_dict[stemmed]['anypos'][1] == 'both':
						sent_features['subjlexicon positive'] += 1
						sent_features['subjlexicon negative'] += 1
					else:
						sent_features['subjlexicon ' + subj_dict[stemmed]['anypos'][1]] += 1
	
	# Use the total number of sentiment words as a feature
	sent_features["total sentiment words"] = sent_features["subjlexicon positive"] + sent_features["subjlexicon negative"] + sent_features["subjlexicon neutral"]
	
	if sent_features["subjlexicon negative"] != 0:
		if sent_features["subjlexicon positive"] == 0:
			sent_features["words pos:neg"] = -1.0 / float(sent_features["subjlexicon negative"])
		else:
			sent_features["words pos:neg"] = sent_features["subjlexicon positive"] / float(sent_features["subjlexicon negative"])

	if sent_features["subjlexicon negative"] != 0:
		if sent_features["subjlexicon neutral"] == 0:
			sent_features["words neutral:neg"] = -1.0 / float(sent_features["subjlexicon negative"])
		else:
			sent_features["words neutral:neg"] = sent_features["subjlexicon neutral"] \
			                                     / float(sent_features["subjlexicon negative"])
	if sent_features["subjlexicon weaksubj"] != 0:
		if sent_features["subjlexicon strongsubj"] == 0:
			sent_features["subjectivity strong:weak"] = -1.0 / float(sent_features["subjlexicon weaksubj"])
		else:
			sent_features["subjectivity strong:weak"] = sent_features["subjlexicon strongsubj"] \
			                                            / float(sent_features["subjlexicon weaksubj"])
	
	# Vader Sentiment Analyser
	# Obtain the negative, positive, neutral and compound scores of a tweet
	sia = SentimentIntensityAnalyzer()
	polarity_scores = sia.polarity_scores(tweet)
	for name, score in polarity_scores.items():
		sent_features["Vader score " + name] = score
	return sent_features


# TODO: i removed LDA model that is used to topic modeling

# Collect all features
def get_feature_set(tweets_tokens, tweets_pos, pragmatic=False, pos_unigrams=False, pos_bigrams=False,
                    lexical=False, ngram_list=[1], sentiment=False, topic=True):
	features = []
	if sentiment:
		# Emoji lexicon - underlying sentiment (pos, neutral, neg)
		emoji_dict = data_proc.build_emoji_sentiment_dictionary()
		# Obtain subjectivity features from the MPQA lexicon and build the subjectivity lexicon
		subj_dict = data_proc.get_subj_lexicon()
	if topic:
		use_nouns = True
		use_verbs = True
		use_all = False
		dictionary, corpus, lda_model = build_lda_model(tweets_tokens, tweets_pos,
		                                                use_nouns=use_nouns, use_verbs=use_verbs, use_all=use_all,
		                                                num_of_topics=6, passes=20, verbose=False)
	for index in range(len(tweets_tokens)):
		tokens_this_tweet = tweets_tokens[index].split()
		pos_this_tweet = tweets_pos[index].split()
		pragmatic_features = {}
		pos_unigrams_features = {}
		pos_bigrams_features = {}
		words_ngrams = {}
		sentiment_features = {}
		topic_features = {}
		if pragmatic:
			pragmatic_features = get_pragmatic_features(tokens_this_tweet)
		if pos_unigrams:
			pos_unigrams_features = get_pos_features(pos_this_tweet)
		if pos_bigrams:
			pos_bigrams_features = get_ngrams(pos_this_tweet, n=[3], for_semantics=True)
		if lexical:
			words_ngrams = get_ngrams(tokens_this_tweet, n=ngram_list, use_just_words=True)
		if sentiment:
			sentiment_features = get_sentiment_features(tweets_tokens[index], tokens_this_tweet,
			                                            pos_this_tweet, emoji_dict, subj_dict)
		if topic:
			topic_features = get_topic_features_for_unseen_tweet \
				(dictionary, lda_model, tokens_this_tweet, pos_this_tweet,
				 use_nouns=use_nouns, use_verbs=use_verbs, use_all=use_all)
		# Append all extracted features of this tweet to the final list of all features
		features.append({**pragmatic_features, **pos_unigrams_features, **pos_bigrams_features,
		                 **words_ngrams, **sentiment_features, **topic_features})
	return features
