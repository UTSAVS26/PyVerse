movieclassification
===================

Movie genre classification by plot text.

Abstract
--------

Classifies movies into genres according to plot summary texts.

A movie classifier (e.g. `NaiveBayesClassifier`), once trained, estimates the most-likely genres a movie belongs to, according to past examples.

Method
------

We implement supervised-learning techniques for multi-category classification of a document verses the the training corpus results. We assume positional independence between the words of a single document (leading to a "bag of words" representation), and independence between the categories (each category is evaluated independently for each document).

### Feature Extraction ###

**Movie Data**: Each movie is stored in a JSON entry containing fields: 'name', 'genres', 'year' and 'plot'. We extract features only from the 'plot' field. Each feature (term) is defined as a unigram (word) which is longer than K characters. Each movie is stored into a `Movie` class (which inherits from `Document`). We only use the 'genres' and 'plot' fields in later methods ('name' is used for convenience methods only).

**Document Features Representation**: Each `Document` implements a "bag of words" representation of the original text: for each term in the document we remember the number of times it has appeared.

**Document Corpus**: We aggregate multiple `Document`s into a `DocumentCorpus`. For each term, the document corpus stores the number of times it has appeared in the entire corpus (`term_freq`), the number of documents it has appeared in (`doc_freq`), and the total count of terms and documents it represents. Aggregating data into document corpuses allows us to represent a training set more as a sparse vector.

**Category Corpuses**: We store a document corpus as described above for each category (genre), and for all the categories combined (the "universe"). Using the data in these corpuses, we perform classification and training. This data is encapsulated inside the `MultiCategoryCorpus` class.

### Classification ###

Our problem allows each movie to be classified into multiple genres. We choose to implement this by evaluating multiple binary classifiers, one for each 

We perform the classification using the naive bayes method. We create a "weight" for each document in respect to the evaluated category, which is calculted using the bayes model: `P(c|d) = P(c) * P(d|c) / p(d)`. 

The probability `P(c|d)` is evaluated according to a threshold which was found during the training procedure. In order to avoid underflows, the calculation of the above formula is done by summing logarithms (instead of multiplying many small floating point numbers). The implementation of this calculation is found inside `NaiveBayesClassifier.document_weight(...)`.

### Training ###

The objective of the training is to find a threshold `T` for each category. Category predictions of weight larger than `T` are considered to be positive (belongs to the category), otherwise they are considered to be negative (not in the category).

The threshold calculation is done inside the `MultiCategoryClassifier.train()` method. The procedure is to re-classify each document in the training set (after collecting all the `MultiCategoryCorpus` data), and evaluate the results using the `ClassificationStats` utility class. The threshold value assigned for each category `T(c)` is assigned according to the maximum weight in all false positive (`FP`) guesses in the training corpus. Clearly this method requires improvement.

### Testing ###

Testing is done by classifying each document in a test set according to the trained classifier. The actual categories (observed) are compared with the classified decisions (predicted) via the `ClassificationStats` utility class, which collects statistics and metrics on the classification results (described below).


Classification Metrics
----------------------

The following metrics are currently evaluated per-category:

* True Positives Count: `TP`
* True Negatives Count: `TN` 
* False Positives Count: `FP`
* False Negatives Count: `FN`
* Precision: `PREC = TP / (TP + FP)`
* Recall: `RCL = TP / (TP + FN)`
* Accuracy: `ACC = (TP+TN) / (TP+FP+TN+FN)`

Further defintions of these metrics is available at the relevant [wikipedia article](http://en.wikipedia.org/wiki/Precision_and_recall).

Usage
-----

The program is not designed to work from command-line at the moment, but its various components may be used by `import`ing in python. An example of a complete test loop (load, train, test) is provided in `movies.test(...)`.

### Load movie data ###

    from movies import load_all_movies
    movie_lists = load_all_movies(movie_files=['train.json', 'test.json'])
    training_movies = movie_lists[0]
    test_movies = movie_lists[1]

### Build a MultiCategoryCorpus from movies ###

    from text_classification import MultiCategoryCorpus
    corpus = MultiCategoryCorpus.build(mtrain)

### Create and train a NaiveBayesClassifier ###

    from text_classification import NaiveBayesClassifier
    classifier = NaiveBayesClassifier(corpus)
    stats = classifier.train(training_movies)

### Test classifier on list of movies ###

    stats = classifier.test(test_movies)

Requirements
------------

* Python 2.7+
* No libraries are required

### Optional Modifications ###

* Any NLTK stemmer from `nltk.stem` may replace `text_extraction.Stemmer` (via the `stem(word)` method)
* NLTK Stopwords from `nltk.corpus.stopwords` may be used to initialize the stopwords list in `text_extraction.UnigramCounter`.

Software Design
---------------

The following list describes the module structure inside the library, and the prime classes inside each of them:

* **text_extraction.py**: Text feature extraction logic.
    * `Stemmer` interface: for text stemming, compatible with `nltk.stem`'s `stem(word)` method.
    * `TextExtractor` interface: parses a text block and returns a histogram (count of terms appearances in the text).
    * `UnigramCounter` class: implements `TextExtractor`. Splits a text block into unigrams (words). Supports stopwords filtering, stemming and minimal word length.
* **text_classification.py**: Text classification logic.
    * `Document` class: bag-of-words representation of a document (i.e. movie data).
    * `DocumentCorpus` class: aggregates metrics over many `Document` classes.
    * `MultiCategoryCorpus` class: aggregates metrics of many categories using `DocumentCorpus` classes. 
    * `MultiCategoryClassifier` class: base class for multiple category classification. 
    * `NaiveBayesClassifier` class: concrete implementation of `MultiCategoryClassifier`, implementing the naive bayes technique described previously in this document.
    * `ClassificationStats` class: utility to store and display statistics about the classification of many movies, according to the metrics described previously in this document.
* **utils.py**: 
    * `PickleCache` class: base class for seamless persistent storage of objects to disk using pickle. Concrete classes must implement the `load_object` method, which parses the original file format into a python object.
    * `random_json_file_subset` function: can be used to create a random movie subsets out of training or test files.
* **movies.py**: Wrappers and methods for working with movies (not just generic 'documents'). Responsible for data parsing and I/O.
    * `Movie` class: wraps `Document` and provides convenience and I/O methods specific to movies and our datasets.
    * `MovieCache` class: concrete implementation of `PickleCache` for `Movie` lists. 


Evaluation on given datasets
----------------------------

### Test Results ###

Here are the raw results for `NaiveBayesClassifier.test()` when trained on the full training corpus (`movies_train.json`) and tested against the full test corpus (`movies_test.json`):

    CATEGORY             |   PREC |    RCL |    ACC |  #POS |    TP |    TN |    FP |    FN |
    drama: 40.59%        |  39.6% |  14.6% |  56.3% |   457 |    67 |   567 |   102 |   390 |
    short: 32.24%        |  39.3% |  70.5% |  55.5% |   363 |   256 |   369 |   394 |   107 |
    comedy: 27.80%       |  29.2% |  61.0% |  48.2% |   313 |   191 |   352 |   461 |   122 |
    documentary: 14.30%  |  13.6% |  53.4% |  44.8% |   161 |    86 |   419 |   546 |    75 |
    romance: 10.66%      |   9.3% |  49.1% |  43.5% |   120 |    59 |   431 |   575 |    61 |
    action: 9.50%        |   8.8% |  52.3% |  44.4% |   107 |    56 |   445 |   574 |    51 |
    thriller: 8.53%      |   7.6% |  50.0% |  44.0% |    96 |    48 |   448 |   582 |    48 |
    crime: 7.02%         |   5.7% |  45.5% |  44.0% |    79 |    36 |   460 |   587 |    43 |
    family: 6.84%        |   5.2% |  41.5% |  44.7% |    77 |    32 |   472 |   577 |    45 |
    adventure: 6.13%     |   5.0% |  44.9% |  45.1% |    69 |    31 |   477 |   580 |    38 |
    mystery: 5.86%       |   4.8% |  43.9% |  46.2% |    66 |    29 |   492 |   568 |    37 |
    animation: 5.42%     |   6.7% |  65.5% |  48.6% |    61 |    40 |   508 |   557 |    21 |
    horror: 5.42%        |   5.0% |  49.1% |  46.6% |    61 |    30 |   495 |   570 |    31 |
    fantasy: 4.62%       |   4.5% |  51.9% |  47.3% |    52 |    27 |   506 |   568 |    25 |
    war: 3.73%           |   3.4% |  47.6% |  49.0% |    42 |    20 |   532 |   552 |    22 |
    sci-fi: 3.64%        |   3.8% |  56.0% |  47.9% |    41 |    23 |   517 |   568 |    18 |
    western: 3.29%       |   3.2% |  51.3% |  47.6% |    37 |    19 |   517 |   572 |    18 |
    history: 3.20%       |   3.5% |  55.5% |  50.1% |    36 |    20 |   545 |   545 |    16 |
    musical: 2.84%       |   2.6% |  46.8% |  48.7% |    32 |    15 |   534 |   560 |    17 |
    music: 2.75%         |   2.6% |  48.3% |  49.0% |    31 |    15 |   537 |   558 |    16 |
    sport: 1.24%         |   1.5% |  57.1% |  53.1% |    14 |     8 |   590 |   522 |     6 |
    news: 0.71%          |   0.2% |  12.5% |  60.4% |     8 |     1 |   680 |   438 |     7 |
    film-noir: 0.53%     |   0.5% |  33.3% |  66.7% |     6 |     2 |   750 |   370 |     4 |
    adult: 0.27%         |   0.0% |   0.0% |  66.6% |     3 |     0 |   750 |   373 |     3 |
    lifestyle: 0.00%     |   0.0% |   0.0% |  73.7% |     0 |     0 |   830 |   296 |     0 |
    TOTAL: 100.00%       |  21.3% |  47.6% |  49.8% |  2332 |  1111 | 13223 | 12595 |  1221 |


Suggested Improvements
----------------------

### Algorithmic / Classification ###
* Improve current training method and threshold inferrance
* Convert "document weight scores" into per-category probabilities in range [-1,1], where -1 indicates 100% negative certainty, and +1 indicates 100% positive certainty. This can possibly be achieved using a sigmoid function around the threshold value for the category.
* Handle variance in per-genre corpus sizes
* Experiment with other classification methods and metrics (TF-IDF, SVM, etc)
* Improved text cleanup (dimensionality reduction) using stopwords and stemming
* Movie-specific optimizations to corpus and classifier (manually tweak weights of words and categories according to data).
* Experiment with NGram terms (e.g. Bigrams and Trigrams in addition to Unigrams). Employing NGram terms might give us higher quality data, such as actor names, locations, etc.

### Performance ###
* Training while loading the data: will allow the program to go over the training file just once, and work fully with generators without keeping all the movies in memory at once.
* Corpus and Classifier pickling/caching
* Use standard implementations for learning and language processing.Specifically "scikit-learn" and "nltk"
* Parallelize training/building procedures (e.g. using multiple threads/processes)

### Interface ###
* Full-bown command-line interface using `argparse`.
* Visualizations using `mathplotlib` or similar.
