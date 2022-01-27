import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

import string

from pprint import pprint

nltk.download([
    "names",
    "stopwords",
    "state_union",
    "twitter_samples",
    "movie_reviews",
    "averaged_perceptron_tagger",
    "vader_lexicon",
    "punkt",
])

words = [w for w in nltk.corpus.state_union.words() if w.isalpha()]

stopwords = nltk.corpus.stopwords.words("english")

stop = set(stopwords + list(string.punctuation))

words = [w for w in words if w.lower() not in stop]

text = """
For some quick analysis, creating a corpus could be overkill.
If all you need is a word list,
there are simpler ways to achieve that goal."""

pprint(nltk.word_tokenize(text), width=79, compact=True)

text_tokenized: list[str] = nltk.word_tokenize(text)

text_tokenized = [w for w in text_tokenized if w.lower() not in stop]

fd = nltk.FreqDist(text_tokenized)

fd.tabulate(3)

# fd = nltk.FreqDist(words)
#
# pprint(fd)
#
# lower_fd = nltk.FreqDist([w.lower() for w in words])
#
# pprint(lower_fd)
#
# pprint(fd.most_common(3))
#
# pprint(fd["America"])
#
# fd.tabulate(3)
#
# text = nltk.Text(nltk.corpus.state_union.words())
#
# text.concordance("america", lines=5)
#
# concordance_list = text.concordance_list("america", lines=2)
# for entry in concordance_list:
#     print(entry.line)

words = [w for w in nltk.corpus.state_union.words() if w.isalpha()]

finder = nltk.collocations.TrigramCollocationFinder.from_words(words)

finder.ngram_fd.tabulate(2)

sia = SentimentIntensityAnalyzer()

pprint(sia.polarity_scores("Wow, NLTK is really powerful!"))

pprint(sia.polarity_scores(text))

pprint(sia.polarity_scores("today is sunny"))

tweets = [t.replace("://", "//") for t in nltk.corpus.twitter_samples.strings()]

from random import shuffle

# def is_positive(tweet: str) -> bool:
#     """True if tweet has positive compound sentiment, False otherwise."""
#     return sia.polarity_scores(tweet)["compound"] > 0

def is_positive(tweet: str) -> str:
    """True if tweet has positive compound sentiment, False otherwise."""
    if sia.polarity_scores(tweet)["compound"] > 0.75:
        return f"😁 {sia.polarity_scores(tweet)}"
    elif sia.polarity_scores(tweet)["compound"] > 0.5:
        return f"😀 {sia.polarity_scores(tweet)}"
    elif sia.polarity_scores(tweet)["compound"] > 0.25:
        return f"😊 {sia.polarity_scores(tweet)}"
    elif sia.polarity_scores(tweet)["compound"] > 0:
        return f"🤨 {sia.polarity_scores(tweet)}"
    elif sia.polarity_scores(tweet)["compound"] > -0.25:
        return f"😥 {sia.polarity_scores(tweet)}"
    elif sia.polarity_scores(tweet)["compound"] > -0.5:
        return f"😈 {sia.polarity_scores(tweet)}"
    elif sia.polarity_scores(tweet)["compound"] > -0.75:
        return f"👹 {sia.polarity_scores(tweet)}"
    elif sia.polarity_scores(tweet)["compound"] > -1:
        return f"🤬 {sia.polarity_scores(tweet)}"
    else:
        return "🙄"

shuffle(tweets)
for tweet in tweets[:10]:
    print(">", is_positive(tweet), tweet)


positive_review_ids = nltk.corpus.movie_reviews.fileids(categories=["pos"])
negative_review_ids = nltk.corpus.movie_reviews.fileids(categories=["neg"])
all_review_ids = positive_review_ids + negative_review_ids

from statistics import mean

def is_positive(review_id: str) -> bool:
    """True if the average of all sentence compound scores is positive."""
    text = nltk.corpus.movie_reviews.raw(review_id)
    scores = [
        sia.polarity_scores(sentence)["compound"]
        for sentence in nltk.sent_tokenize(text)
    ]
    return mean(scores) > 0


shuffle(all_review_ids)
correct = 0
for review_id in all_review_ids:
    if is_positive(review_id):
        if review_id in positive_review_ids:
            correct += 1
    else:
        if review_id in negative_review_ids:
            correct += 1
print(F"{correct / len(all_review_ids):.2%} correct")


unwanted = nltk.corpus.stopwords.words("english")
unwanted.extend([w.lower() for w in nltk.corpus.names.words()])

def skip_unwanted(pos_tuple):
    word, tag = pos_tuple
    if not word.isalpha() or word in unwanted:
        return False
    if tag.startswith("NN"):
        return False
    return True

positive_words = [word for word, tag in filter(
    skip_unwanted,
    nltk.pos_tag(nltk.corpus.movie_reviews.words(categories=["pos"]))
)]
negative_words = [word for word, tag in filter(
    skip_unwanted,
    nltk.pos_tag(nltk.corpus.movie_reviews.words(categories=["neg"]))
)]


positive_fd = nltk.FreqDist(positive_words)
negative_fd = nltk.FreqDist(negative_words)

common_set = set(positive_fd).intersection(negative_fd)

for word in common_set:
    del positive_fd[word]
    del negative_fd[word]

top_100_positive = {word for word, count in positive_fd.most_common(100)}
top_100_negative = {word for word, count in negative_fd.most_common(100)}

positive_bigram_finder = nltk.collocations.BigramCollocationFinder.from_words([
    w for w in nltk.corpus.movie_reviews.words(categories=["pos"])
    if w.isalpha() and w not in unwanted
])
negative_bigram_finder = nltk.collocations.BigramCollocationFinder.from_words([
    w for w in nltk.corpus.movie_reviews.words(categories=["neg"])
    if w.isalpha() and w not in unwanted
])


def extract_features(text):
    features = dict()
    wordcount = 0
    compound_scores = list()
    positive_scores = list()

    for sentence in nltk.sent_tokenize(text):
        for word in nltk.word_tokenize(sentence):
            if word.lower() in top_100_positive:
                wordcount += 1
        compound_scores.append(sia.polarity_scores(sentence)["compound"])
        positive_scores.append(sia.polarity_scores(sentence)["pos"])

    # Adding 1 to the final compound score to always have positive numbers
    # since some classifiers you'll use later don't work with negative numbers.
    features["mean_compound"] = mean(compound_scores) + 1
    features["mean_positive"] = mean(positive_scores)
    features["wordcount"] = wordcount

    return features

pprint(extract_features(text))