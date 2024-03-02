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
        self.title_score = 0.6
        self.text_score = 0.4
        self.anchor_score = 0

        # Put your bucket name below and make sure you can access it without an error
        # bucket_name = 'wikiproject-414111-bucket'
        # full_path = f"gs://{bucket_name}/postings_gcp"
        # paths = []
        #
        # client = storage.Client()
        # blobs = client.list_blobs(bucket_name)
        # for b in blobs:
        #     if b.name != 'graphframes.sh':
        #         paths.append(full_path + b.name)

        # spark.read.parquet(full_path + "title_index.pkl")
        # title_index = spark.read.parquet(full_path + "title_index.pkl")

        self.english_stopwords = frozenset(stopwords.words('english'))
        self.corpus_stopwords = ["category", "references", "also", "external", "links",
                            "may", "first", "see", "history", "people", "one", "two",
                            "part", "thumb", "including", "second", "following",
                            "many", "however", "would", "became"]

        self.all_stopwords = self.english_stopwords.union(self.corpus_stopwords)
        # self.RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)

        # self.all_stopwords = self.english_stopwords.union(self.corpus_stopwords)


        # self.text_index_path = "text_index.pkl"
        # self.index_dst = f'gs://{bucket_name}/postings_gcp/{index_src}'
        # !gsutil cp $self.index_dst

        # self.index = InvertedIndex()
        # InvertedIndex.read_index('.', "inverted_index", self.bucket_name)

        self.porter_stemmer = PorterStemmer()
        self.bucket_name = 'wikiproject-414111-bucket'
        self.inverted_title = InvertedIndex.read_index('title_postings', 'index_title', self.bucket_name)
        self.inverted_text = InvertedIndex.read_index('text_postings', 'index_text', self.bucket_name)
        self.title_lengths = InvertedIndex.read_index('title_postings', 'title_lengths', self.bucket_name)
        self.text_lengths = InvertedIndex.read_index('text_postings', 'text_lengths', self.bucket_name)
        self.title_id = InvertedIndex.read_index('.', 'title_id', self.bucket_name)

    def backend_search(self, query):
        stemmed_query = self.stem_query(query)
        title_score = self.calculate_cosine_score(stemmed_query, self.title_lengths, self.inverted_title)
        text_score = self.calculate_cosine_score(stemmed_query, self.text_lengths, self.inverted_text)
        scores = self.weighted_score(title_score, text_score)
        # retrive the top 100 doc ids
        top_docs = [score[0] for score in scores.take(100)]
        # returns (doc_id, title of doc_id) for the top 100 documents
        top = [(ID, self.title_id[id]) for ID in top_docs]
        return top

    def stem_query(self, query):
        # Stem the query terms and remove stopwords
        stemmed_query = [self.porter_stemmer.stem(term) for term in query.split() if term not in self.all_stopwords]
        return stemmed_query

    def calculate_cosine_score(self, query, doc_lengths, inverted):
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
            docs = self.sc.parallelize(inverted.read_a_posting_list('.', term))
            # docs = sc.parallelize(read_posting_list(inverted, term))
            # Get (doc_id, tf_idf) pairs
            docs_by_id = docs.groupByKey().mapValues(lambda x: list(x))
            # Calculate scores
            part_scores = docs_by_id.flatMap(lambda x: [(x[0], np.sum([tf_idf for tf_idf in x[1]]))])
            scores = scores.leftOuterJoin(part_scores)
        scores = scores.flatMap(lambda x: [(x[0], x[1][0] + x[1][1]) if x[1][1] is not None else (x[0], 0)])
        # Normalize each score by the doc's length
        scores = scores.join(doc_lengths_rdd).flatMap(lambda x: [(x[0], x[1][0] / x[1][1])])
        return scores

    def weighted_score(self, title_scores, text_scores, anchor_scores=None):
        if anchor_scores != None:
            # Join the RDDs based on the doc_id
            joined_rdd = title_scores.join(text_scores).join(anchor_scores)

            # Compute the weighted sum for each doc_id
            weighted_sum = joined_rdd.flatMap(lambda x: (x[0],
                                                         x[1][0][0] * self.title_score +
                                                         x[1][0][1] * self.text_score +
                                                         x[1][1] * self.anchor_score
                                                         ))
        else:
            # Join the RDDs based on the doc_id
            joined_rdd = title_scores.join(text_scores)

            # Compute the weighted sum for each doc_id
            weighted_sum = joined_rdd.flatMap(lambda x: [(x[0],
                                                          x[1][0] * self.title_score +
                                                          x[1][1] * self.text_score
                                                          )])
        # Sort docs by score
        weighted_sum = weighted_sum.sortBy(lambda x: x[1], ascending=False)
        return weighted_sum


    def combine_scores(self, cosine_scores, pagerank_scores, alpha=0.5):
        # Normalize PageRank scores
        pagerank_sum = pagerank_scores.map(lambda x: x[1]).sum()
        normalized_pagerank_scores = pagerank_scores.map(lambda x: (x[0], x[1] / pagerank_sum))

        # Join cosine similarity scores and normalized PageRank scores
        combined_scores = cosine_scores.join(normalized_pagerank_scores)

        # Compute the combined score for each document
        combined_scores = combined_scores.map(lambda x: (x[0], alpha * x[1][0] + (1 - alpha) * x[1][1]))

        return combined_scores


    def init_spark(self):
        graphframes_jar = 'https://repos.spark-packages.org/graphframes/graphframes/0.8.2-spark3.2-s_2.12/graphframes-0.8.2-spark3.2-s_2.12.jar'
        spark_jars = '/usr/local/lib/python3.7/dist-packages/pyspark/jars'
        # Initializing spark context
        # create a spark context and session
        self.conf = SparkConf().set("spark.ui.port", "4050")
        self.conf.set("spark.jars.packages", "graphframes:graphframes:0.8.2-spark3.2-s_2.12")
        self.sc = pyspark.SparkContext(conf=self.conf)
        self.sc.addPyFile(str(Path(spark_jars) / Path(graphframes_jar).name))
        self.spark = SparkSession.builder.getOrCreate()


