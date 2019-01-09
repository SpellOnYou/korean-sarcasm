import utils
import preprocess as data_proc

# Make a simple analysis of the key-features picked by models trained on embeddings (works for both emojis and words)
def rule_based_comparison(x_train, y_train, x_test, y_test, vocab_filename, verbose=True):
    vocab = data_proc.build_vocabulary(vocab_filename, x_train, minimum_occurrence=10)
    # vocab = set(' '.join([x.lower() for x in x_train]).split()) # this includes all words in the train set
    counts = {k: [0, 0] for k in vocab}
    for tw, y in zip(x_train, y_train):
        # print('this is tweet {} : data is {} and label is {}'.format(type(tw), tw, y))
        try:
            for word in tw.split(): #phrase based
            # for word in tw: #char based
                # word = word.lower()
                if word in vocab:
                    if y == 0:
                        counts[word][0] += 1
                    else:
                        counts[word][1] += 1
        except AttributeError: #when text was composed of #반어법 only/
            continue
        

    # Calculate the relative weight of each word, based on the sarcastic/non-sarcastic tweets that it appears
    weight = dict.fromkeys([k for k in counts.keys()], 0)
    for word in counts.keys():
        if counts[word][1] + counts[word][0] != 0:
            weight[word] = (counts[word][1] - counts[word][0]) / (counts[word][1] + counts[word][0])

    if verbose:
        total_sarcastic = sum([1 for y in y_train if y == 1])
        stopwords = data_proc.get_stopwords_list()
        probs = {word: (counts[word][1] / total_sarcastic) for word in counts.keys()
                 if word not in stopwords and word.isalnum()}
        print("Top 20 most sarcastic items: ", ' '.join(sorted(probs, key=probs.get, reverse=True)[:20]))

    # Rule-based predictions based on the previously calculated weigths
    # print(counts)
    y_pred = []
    for tw, y in zip(x_test, y_test):
        score = 0.0
        for word in tw.split():
        # for word in tw:
            if word in vocab:
                score += weight[word]
        if score >= 0.0:
            y_pred.append(1)
        else:
            y_pred.append(0)
    utils.print_statistics(y_test, y_pred)