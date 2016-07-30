import re
import nltk
import string
import pickle
import numpy as np
import pandas as pd
import simplejson as json
from sklearn.cross_validation import train_test_split
from collections import defaultdict
np.random.seed(1738)

#===============================================================================

def get_data():
	'''
	INPUT: Directory path to data (str)
	OUTPUT: List of restaurant reviews
	'''
	json_data = []
	with open('../../../sum_abs/yds/yelp_academic_dataset_review.json') as f:
		for line in f:
			json_data.append(json.loads(line))
	bus_data = []
	with open('../../../sum_abs/yds/yelp_academic_dataset_business.json') as f:
		for line in f:
			bus_data.append(json.loads(line))

	bus = {}
	for b in bus_data:
	    if b['business_id'] not in bus:
	        bus[b['business_id']] = b['categories']
	        
	for rev in json_data:
	    rev['categories'] = bus[rev['business_id']]
	    
	restaurants = [rev for rev in json_data if 'Restaurants' in rev['categories']]
	return restaurants

#===============================================================================

def exploratory_analysis(tokenized, word_feats):
	'''
	Takes in a list of tokenized documents and returns a dictionary of those
	words as "keys" and the most common words that follow and precede them as
	"values"

	INPUT: List of tokenized documents
	OUTPUT: Two dictionaries "pre" and "post"
	'''
	pre = defaultdict(list)
	for doc in tokenized:
	    for i, word in enumerate(doc[1:]):
	        if word in word_feats:
	            pre[word].append(doc[i-1])

	post = defaultdict(list)
	for doc in tokenized:
	    for i in enumerate(doc[:-1]):
	        if doc[i-1] in word_feats:
	            post[doc[i-1]].append(word)

    pre  = [nltk.FreqDist(pre[w]).most_common(20)  for w in word_feats]
    post = [nltk.FreqDist(post[w]).most_common(20) for w in word_feats]

	return pre, post

#===============================================================================

def join_fol(words_list, joins):
	'''
	Rejoins words from a list (joins) with the words that follow them

	INPUT: List of words from nltk.word_tokenize(), and list of word markers
	OUTPUT: List of words
	'''
	temp = [word if word not in joins else ' '.join([word, words_list[i+1]])\
			for i, word in enumerate(words_list[:-1])]
	temp = [word for i, word in enumerate(temp) if ' ' not in list(temp[i-1])]
	return temp + [words_list[-1]]

def join_pre(words_list, joins):
	'''
	Joins words from a list (joins) with the words that precede them

	INPUT: List of words from nltk.word_tokenize()
	OUTPUT: List of words
	'''
	for i, word in enumerate(words_list[1:]):
		if word in joins:
			words_list[i-1] = ' '.join([words_list[i-1], word])
	for word in words_list:
		if word in joins:
			words_list.remove(word)
	return words_list

def re_position(words_list):
	'''
	Takes adjective, subject pairings and readjusts their positioning in a
	text block for later processing.

	INPUT: List of words
	OUTPUT: List of words
	'''
	gen_adj = set(['excellent', 'good', 'great', 'bad', 'perfect', 'amazing', 
	'awesome', 'favorite', 'nice', 'worst', 'better', 'definitely', 'wonderful',
	'not excellent', 'not good', 'not great', 'not bad', 'not perfect',
	'not amazing', 'not awesome', 'not favorite', 'not nice', 'not worst',
	'not better', 'not wonderful'])
	subject = set(['service', 'ambiance', 'portions', 'food', 'decor', 'price'])
	
	words = words_list[:]
	for i, w in enumerate(words):
		if w in gen_adj and words[i - 1] in subject and i > 0:
			words[i], words[i - 1] = words[i - 1], words[i]
	return words

def mapper(words_list):
	'''
	Re-maps specific words to consolidate the subject space

	INPUT: List of words
	OUTPUT: List of words
	'''
	lut = {
		'recomend':   'recommend',
		'ambience':   'ambiance',
		'atmosphere': 'ambiance',
		'prices':     'price',
		'portion':    'portions',
		'large':      'big',
		'huge':       'big',
		'server':     'service',
		'servers':    'service',
		'place':      'restaurant',
		'local':      'restaurant',
		'overpriced': 'over priced',
		'chicken':    'food',
		'pizza':      'food',
		'meal':       'food',
		'dinner':     'food',
		'salad':      'food',
		'lunch':      'food',
		'sushi':      'food',
		'burger':     'food',
		'meat':       'food',
		'fries':      'food',
		'steak':      'food',
		'bread':      'food',
		'breakfast':  'food',
		'beef':       'food',
		'rice':       'food',
		'dessert':    'food',
		'dishes':     'food',
		'soup':       'food',
		'pork':       'food',
		'shrimp':     'food',
		'fish':       'food',
		'plate':      'food',
		'crab':       'food',
		'appetizer':  'food',
		'burgers':    'food',
		'wings':      'food',
		'sandwiches': 'food',
		'lobster':    'food',
		'cake':       'food',
		'pasta':      'food',
		'eggs':       'food',
		'salmon':     'food',
		'seafood':    'food',
		'taste':      'food'
	}
	return [w if w not in lut else lut[w] for w in words_list]

def bag_of_words(data):
	'''
	Takes the collection of Yelp data and generates the 5000 most commonly used
	words.

	INPUT: List of json dicts
	OUTPUT: List of words
	'''
	p_data = [doc for doc in data if doc['stars'] == 5]
	n_data = [doc for doc in data if doc['stars'] == 1]

	filtered = []
	stop_words = nltk.corpus.stopwords.words('english')
	punct = string.punctuation

	filtered.extend(stop_words)
	filtered.extend(punct)
	filtered.extend(["...", "''", "``"])

	for doc in p_data:
	    text = doc['text'].lower().encode('ascii', 'ignore')
	    words = nltk.word_tokenize(text)
	    words = [w for w in words if w not in filtered]
	    doc['words'] = words

	for doc in n_data:
	    text = doc['text'].lower().encode('ascii', 'ignore')
	    words = nltk.word_tokenize(text)
	    words = [w for w in words if w not in filtered]
	    doc['words'] = words

	all_words = []

	[all_words.append(w) for doc in p_data for w in doc['words']]
	[all_words.append(w) for doc in n_data for w in doc['words']]

	all_words = nltk.FreqDist(all_words)
	all_words = all_words.most_common(5000)

	word_features = [tup[0] for tup in all_words]
	return word_features

def map_word_feats(train, test):
	'''
	Takes the training and testing datasets and preps the words in the "text"
	field by mapping adjectives to their subjects before creating the feature
	matrix.

	INPUT: Training data, testing data
	OUTPUT: Training data, testing data
	'''
	filtered = ['really', 'very', 'even', 'much', 'real', 'always', 'customer', \
			'the', 'our', 'my', 'is', 'was', 'of']
	subjects = ['service', 'ambiance', 'portions', 'food', 'decor', 'price', \
           'restaurant', 'experience']

	for doc in train:
		text = doc['text'].lower()
		words = nltk.word_tokenize(text)
		words = filter(lambda x: x not in filtered, words)
		words = mapper(words)
		words = join_fol(words, ['not'])
		words = join_pre(words, ['priced'])
		words = re_position(words)
		words = join_pre(words, subjects)
		# words = [w for w in words if 'restaurant' not in w and 'experience' not in w]
		doc['words'] = words

	for doc in test:
		text = doc['text'].lower()
		words = nltk.word_tokenize(text)
		words = filter(lambda x: x not in filtered, words)
		words = mapper(words)
		words = join_fol(words, ['not'])
		words = join_pre(words, ['priced'])
		words = re_position(words)
		words = join_pre(words, subjects)
		# words = [w for w in words if 'restaurant' not in w and 'experience' not in w]
		doc['words'] = words
    
	return train, test

def create_feature_matrix(train, test, word_feats, mapped=True):
	'''
	Takes the training data, testing data, and word features and generates the 
	feature matrix and label pairings for the training and testing sets.

	INPUT: Training data, testing data, word features (list of str)
	OUTPUT: Training data, training set labels, testing data, testing set labels
	'''
	traindf = pd.DataFrame(train)
	X_train = traindf[traindf.stars != 3]
	y_train = X_train.apply(lambda x: 1 if x.stars > 3 else 0, axis=1)

	testdf = pd.DataFrame(test)
	X_test = testdf[testdf.stars != 3]
	y_test = X_test.apply(lambda x: 1 if x.stars > 3 else 0, axis=1)

	if mapped == True:
		for w in word_feats:
			X_train[w] = X_train.apply(lambda x: 1 if w in x.words else 0, axis=1)
		for w in word_feats:
			X_test[w] = X_test.apply(lambda x: 1 if w in x.words else 0, axis=1)
	else:
		for w in word_feats:
			X_train[w] = X_train.apply(lambda x: 1 if w in x.text else 0, axis=1)
		for w in word_feats:
			X_test[w] = X_test.apply(lambda x: 1 if w in x.text else 0, axis=1)

	X_train.drop(['categories', 'review_id', 'date', 'business_id', 'text', 'type', \
		'user_id', 'votes', 'stars'], axis=1, inplace=True)
	X_test.drop(['categories', 'review_id', 'date', 'business_id', 'text', \
		'type', 'user_id', 'votes', 'stars'], axis=1, inplace=True)

	if mapped == True:
		X_train.drop('words', axis=1, inplace=True)
		X_test.drop('words', axis=1, inplace=True)

	return X_train, y_train, X_test, y_test

def biz_weights(biz_id):
	'''
	Takes in a restaurant's business id as a path to a json file and processes
	the list of text reviews for the restaurant using the features from the
	Logistic Regression model to re-score the restaurant on a 1-5 scale for
	each component of the restaurant experience: Food, Service, Atmosphere, 
	Value, and Decor.

	INPUT: Path to json file
	OUTPUT: List of strings

	NOTE: Run in directory with all of your json files for restaurant reviews
	'''
	filtered = ['really', 'very', 'even', 'much', 'real', 'customer', 'the', \
    			'our', 'my', 'is', 'was']
	subjects = ['service', 'ambiance', 'portions', 'food', 'decor', 'price', \
    			'restaurant', 'experience']

	with open('../algos/word_feats1.pickle', 'rb') as f:
		word_feats = pickle.load(f)
	with open('../algos/weights.pickle', 'rb') as f:
		weights = pickle.load(f)
	with open('../algos/food.pickle', 'rb') as f:
		food = pickle.load(f)
	with open('../algos/service.pickle', 'rb') as f:
		service = pickle.load(f)
	with open('../algos/atmosphere.pickle', 'rb') as f:
		atmosphere = pickle.load(f)
	with open('../algos/value.pickle', 'rb') as f:
		value = pickle.load(f)
	with open('../algos/decor.pickle', 'rb') as f:
		decor = pickle.load(f)
    
	with open(biz_id, 'r') as f:
		text = json.load(f)['text']
        
	text = [''.join(re.split(r'<br/>|\n', doc)) for doc in text]
    
	data = []
	for rev in text:
		text  = rev.lower()
        words = nltk.word_tokenize(text)
		words = filter(lambda x: x not in filtered, words)
        words = mapper(words)
        words = join_fol(words, ['not'])
        words = join_pre(words, ['priced'])
        words = re_position(words)
        words = join_pre(words, subjects)
        words = [w for w in words if 'restaurant' not in w and 'experience' not in w]
        data.append(words)
    
	df = pd.DataFrame()
	df['text'] = text
	df['words'] = data
    
	for w in word_feats:
		df[w] = df.apply(lambda x: weights[w] if w in x.words else 0, axis=1)
    
	df['food']       = df[food].sum(axis=1)
	df['service']    = df[service].sum(axis=1)
	df['atmosphere'] = df[atmosphere].sum(axis=1)
	df['value']      = df[value].sum(axis=1)
	df['decor']      = df[decor].sum(axis=1)

	output = []
	output.append(biz_id)
	output.append('{0:.1f}'.format((df.food.sum() / df.shape[0]) * 2.98 + 3))
	output.append('{0:.1f}'.format((df.service.sum() / df.shape[0]) * 8.69 + 3))
	output.append('{0:.1f}'.format((df.atmosphere.sum() / df.shape[0]) * 22.47 + 3))
	output.append('{0:.1f}'.format((df.value.sum() / df.shape[0]) * 12.57 + 3))
	output.append('{0:.1f}'.format((df.decor.sum() / df.shape[0]) * 61.73 + 3))
    
	return output

#===============================================================================

__all__ = ['get_data', 'join_fol', 'join_pre', 'mapper', 're_position', \
		'exploratory_analysis', 'bag_of_words', 'map_word_feats', \
		'create_feature_matrix', 'biz_weights']

def main():
	data = get_data()
	train, test = train_test_split(data, test_size=0.3, random_state=1738)
	train, test = map_word_feats(train, test)
	print train[0]
	print test[0]
	print len(train), len(test)

if __name__ == '__main__':
	main()