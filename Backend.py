import numpy as np
import pandas as pd
import bz2
from functools import partial
from collections import Counter, OrderedDict
import pickle
import heapq
from itertools import islice, count, groupby
from xml.etree import ElementTree
import codecs
import csv
from google.cloud import storage
import os
import re
from operator import itemgetter
import nltk
from nltk.stem.porter import *
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from pathlib import Path
import itertools
from time import time
import hashlib
from inverted_index_gcp import *
import pyspark
from pyspark.sql import *
from pyspark.sql.functions import *
from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext
from pyspark.ml.feature import Tokenizer, RegexTokenizer
from graphframes import *
def _hash(s):
    return hashlib.blake2b(bytes(s, encoding='utf8'), digest_size=5).hexdigest()



class Backend:
    def __init__(self):
        self.spark = None
        self.sc = None
        self.conf = None
        self.title_score = 0.45
        self.text_score = 0.4
        self.anchor_score = 0.15

        # Put your bucket name below and make sure you can access it without an error
        bucket_name = 'wikiproject-414111-bucket'
        full_path = f"gs://{bucket_name}/postings_gcp"
        paths = []

        client = storage.Client()
        blobs = client.list_blobs(bucket_name)
        for b in blobs:
            if b.name != 'graphframes.sh':
                paths.append(full_path + b.name)

        spark.read.parquet(full_path + "title_index.pkl")
        title_index = spark.read.parquet(full_path + "title_index.pkl")

        self.english_stopwords = frozenset(stopwords.words('english'))
        self.corpus_stopwords = ["category", "references", "also", "external", "links",
                            "may", "first", "see", "history", "people", "one", "two",
                            "part", "thumb", "including", "second", "following",
                            "many", "however", "would", "became"]

        self.all_stopwords = self.english_stopwords.union(self.corpus_stopwords)
        self.RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)

        self.all_stopwords = self.english_stopwords.union(self.corpus_stopwords)
        self.porter_stemmer = PorterStemmer()

        self.text_index_path = "text_index.pkl"
        self.index_dst = f'gs://{bucket_name}/postings_gcp/{index_src}'
        !gsutil cp $self.index_dst

        self.index = InvertedIndex()
        self.bucket_name = 'wikiproject-414111-bucket'
        InvertedIndex.read_index('.', "inverted_index", self.bucket_name)

    def stem_query(self, query):
        # Stem the query terms
        return [self.porter_stemmer.stem(term) for term in query.split()]

    def calculate_cosine_score(self, query, doc_lengths):
        """ Takes a  a query, and returns scores RDD with docs sorted by relevance.
        Parameters:
        -----------
          query: List
            A List with terms from the query

          postings: Dictionary
            A Dictionary where the keys are doc_id and the value are lengths (norms).
        Returns:
        --------
          RDD
            An RDD where each element is a (doc_id, score).
        """
        self.init_spark()
        # Stem the query terms
        stemmed_query = self.stem_query(query)
        doc_lengths_rdd = self.sc.parallelize(list(doc_lengths.items()))
        # Init scores with 0 for each doc
        scores = doc_lengths_rdd.flatMap(lambda x: [(x[0], 0)])
        # Loop over all words in query
        for term in stemmed_query:
            # Get docs that have this term
            docs = self.sc.parallelize(InvertedIndex.read_a_posting_list('.', term))
            # docs = sc.parallelize(read_posting_list(inverted, term))
            # Get (doc_id, tf_idf) pairs
            docs_by_id = docs.groupByKey().mapValues(lambda x: list(x))
            # Calculate scores
            scores = docs_by_id.flatMap(lambda x: [(x[0], np.sum([tf_idf for tf_idf in x[1]]))])
        # Normalize each score by the doc's length
        scores = scores.join(doc_lengths_rdd).flatMap(lambda x: [(x[0], x[1][0] / x[1][1])])
        # Sort docs by score
        scores = scores.sortBy(lambda x: x[1], ascending=False)

        return scores

    def init_spark(self):
        # Initializing spark context
        # create a spark context and session
        self.conf = SparkConf().set("spark.ui.port", "4050")
        self.conf.set("spark.jars.packages", "graphframes:graphframes:0.8.2-spark3.2-s_2.12")
        self.sc = pyspark.SparkContext(conf=self.conf)
        self.sc.addPyFile(str(Path(spark_jars) / Path(graphframes_jar).name))
        self.spark = SparkSession.builder.getOrCreate()


