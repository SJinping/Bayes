import random
import csv
import platform
import nltk.corpus
import pprint
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus.reader.wordnet import NOUN
from nltk.corpus.reader.wordnet import VERB
from nltk.corpus.reader.wordnet import ADJ
from sklearn.datasets import fetch_20newsgroups
from sklearn.naive_bayes import MultinomialNB
from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from progressbar import AnimatedMarker, Bar, BouncingBar, Counter, ETA, \
    AdaptiveETA, FileTransferSpeed, FormatLabel, Percentage, \
    ProgressBar, ReverseBar, RotatingMarker, \
    SimpleProgress, Timer

_LINE_COUNT_ = 0
_MAIN_DIR_ = ''
_system_ = platform.system()
if _system_ == 'Darwin':
	_MAIN_DIR_ = '/Users/Pan/Idealab'
elif _system_ == 'Linux':
	_MAIN_DIR_ = '/home/pan/Idealab'


def get_file_len(infile):
	print('Calculating...')
	global _LINE_COUNT_
	f = open(infile, 'r')
	line = f.readline()
	while line:
		_LINE_COUNT_ += 1
		line = f.readline()
	f.close()
	return _LINE_COUNT_

# Reading tweet file into a dict
def get_tweet_dict(infile):

	def preprocessing(text):
		lemmatizer = WordNetLemmatizer()
		worddict = set(nltk.corpus.words.words())
		text = text.lower()
		words = text.strip().decode('utf-8')
		wordset_n = set(lemmatizer.lemmatize(w, NOUN) for w in word_tokenize(words))
		wordset_v = set(lemmatizer.lemmatize(w, VERB) for w in wordset_n)
		wordset = set(lemmatizer.lemmatize(w, ADJ) for w in wordset_v)
		wordset = wordset & worddict
		return ' '.join(list(wordset))

	tweet_dict = {}
	with open(infile, 'r') as csvfile:
		spamreader = csv.reader(csvfile, delimiter = ',', quotechar = '"')
		header = next(spamreader)

		pbar = ProgressBar(widgets=[Percentage(), Bar()], maxval = _LINE_COUNT_ - 1).start()
		index = 0
		print('Reading tweets from file...')
		for row in spamreader:
			ID = row[0]
			emo = row[2]
			text = preprocessing(row[1])
			tweet_dict[ID] = {'text': text, 'emo': emo}
			pbar.update(index+1)
			index += 1
		pbar.finish()
		return tweet_dict


def multinomial_nb(tweet_dict, train_num, alpha = 1.0, fit_prior = True, class_prior = None):
	samplelist = random.sample(xrange(len(tweet_dict.keys())), train_num)
	X_train = []
	Y_train = []
	X_predict = []
	Y_predict = []

	pbar = ProgressBar(widgets=[Percentage(), Bar()], maxval = len(tweet_dict.keys())).start()
	index = 0
	print('Processing tweets...')

	# Split the tweets into training dataset and test dataset
	for key, vals in tweet_dict.items():
		if int(key) in samplelist:
			X_train.append(vals['text'])
			Y_train.append(vals['emo'])
		else:
			X_predict.append(vals['text'])
			Y_predict.append(vals['emo'])
		pbar.update(index+1)
		index += 1
	pbar.finish()


	pprint.pprint(Y_train)

	print('Applying Naive Bayes...')
	count_vect = CountVectorizer()
	tf_transformer = TfidfTransformer(use_idf=False) # just use tf, no idf used

	# convert the text list to tfidf form matrix
	X_train_counts = count_vect.fit_transform(X_train)
	X_train_tf = tf_transformer.fit_transform(X_train_counts)
	Y_train = np.array(Y_train)

	clf = MultinomialNB(alpha, fit_prior, class_prior)
	clf.fit(X_train_tf, Y_train) # train the classifier

	# convert list to matrix
	X_pre_counts = count_vect.transform(X_predict)
	X_pre_tf = tf_transformer.transform(X_pre_counts)

	# print X_pre_tf.shape
	# print 'class number: '
	# print clf.class_count_
	# print 'class names: '
	# pprint.pprint(Y_train)
	# print 'class prior: '
	# print clf.class_log_prior_
	# print 'Feature prob: '
	# print clf.feature_log_prob_

	predicted = clf.predict(X_pre_tf)

	correct = 0
	count = 0
	for doc, predict, category in zip(X_predict, predicted, Y_predict):
		# print('%r => %s => %s' % (doc, predict, category))
		print('%s => %s' % (predict, category))
		if predict == category:
			correct += 1
		count += 1

	accuracy = float(correct) / float(count)
	print 'The accuracy is: ' + str(accuracy)


if __name__ == '__main__':
	# text_test()
	infile = _MAIN_DIR_ + '/Data/Blog/Bayes/alltweets_labled_subset.csv'
	train_num = 700
	get_file_len(infile)
	tweet_dict = get_tweet_dict(infile)
	multinomial_nb(tweet_dict, train_num)