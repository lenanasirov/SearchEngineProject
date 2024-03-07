# import numpy as np
# import pandas as pd
# import bz2
# from functools import partial
# from collections import Counter, OrderedDict
# import pickle
# import heapq
# from itertools import islice, count, groupby
# from xml.etree import ElementTree
# import codecs
# import csv
# from google.cloud import storage
# import os
# import re
# from operator import itemgetter
import nltk
from nltk.stem.porter import *
import tempfile
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords', download_dir=tempfile.gettempdir())
nltk.data.path.append(tempfile.gettempdir())
# import matplotlib.pyplot as plt
# from pathlib import Path
# import itertools
# from time import time
import hashlib
from inverted_index_gcp import *
#import pyspark
#from pyspark.sql import *
#from pyspark.sql.functions import *
#from pyspark import SparkContext, SparkConf
#rom pyspark.sql import SQLContext
#from pyspark.ml.feature import Tokenizer, RegexTokenizer
#from graphframes import *
def _hash(s):
    return hashlib.blake2b(bytes(s, encoding='utf8'), digest_size=5).hexdigest()



class Backend:
    def __init__(self):
        self.spark = None
        self.sc = None
        self.conf = None
        self.title_weight = 0.6
        self.text_weight = 0.4
        self.anchor_weight = 0

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
        self.RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)

        # self.all_stopwords = self.english_stopwords.union(self.corpus_stopwords)


        # self.text_index_path = "text_index.pkl"
        # self.index_dst = f'gs://{bucket_name}/postings_gcp/{index_src}'
        # !gsutil cp $self.index_dst

        # self.index = InvertedIndex()
        # InvertedIndex.read_index('.', "inverted_index", self.bucket_name)

        self.porter_stemmer = PorterStemmer()
        self.bucket_name = 'wikiproject-414111-bucket'
        #self.init_spark()
        self.inverted_title = InvertedIndex.read_index('title_postings', 'index_title', self.bucket_name)
        self.inverted_text = InvertedIndex.read_index('text_postings', 'index_text', self.bucket_name)
        self.inverted_anchor = InvertedIndex.read_index('anchor_postings', 'index_anchor', self.bucket_name)
        self.title_lengths = InvertedIndex.read_index('title_postings', 'title_lengths', self.bucket_name)
        self.text_lengths = InvertedIndex.read_index('text_postings', 'text_lengths', self.bucket_name)
        self.anchor_lengths = InvertedIndex.read_index('anchor_postings', 'anchor_lengths', self.bucket_name)
        self.title_id = InvertedIndex.read_index('.', 'title_id', self.bucket_name)
        self.pagerank = InvertedIndex.read_index('.', 'pagerank', self.bucket_name)
        pagerank_min = min(self.pagerank.values())
        pagerank_max = max(self.pagerank.values())
        self.normalized_pagerank_scores = {doc_id: (score - pagerank_max) / (pagerank_max-pagerank_min)
                                      for doc_id, score in self.pagerank.items()}
        a = True
        self.disambiguation_docs = self.disambiguation_union()
        if a:
            print(self.disambiguation_docs)
            a = False


    def disambiguation_union(self):
        disambiguation_stem = self.porter_stemmer.stem('disambiguation')
        disambiguation_title = self.inverted_title.read_a_posting_list('.', disambiguation_stem, self.bucket_name)
        disambiguation_text = self.inverted_text.read_a_posting_list('.', disambiguation_stem, self.bucket_name)
        disambiguation_anchor = self.inverted_anchor.read_a_posting_list('.', disambiguation_stem, self.bucket_name)
        # Create sets of document IDs from the posting lists
        doc_ids_title = set(doc_id for doc_id, _ in disambiguation_title)
        doc_ids_text = set(doc_id for doc_id, _ in disambiguation_text)
        doc_ids_anchor = set(doc_id for doc_id, _ in disambiguation_anchor)
        # Compute the union of document IDs
        union_doc_ids = doc_ids_title.union(doc_ids_text, doc_ids_anchor)
        union_doc_ids_dict = defaultdict(int)
        for doc_id in union_doc_ids:
            union_doc_ids_dict[doc_id] = 0.0
        return union_doc_ids_dict

    def backend_search(self, query):
        stemmed_query = self.stem_query(query)
        title_score = self.calculate_cosine_score(stemmed_query, self.title_lengths, self.inverted_title)
        text_score = self.calculate_cosine_score(stemmed_query, self.text_lengths, self.inverted_text)
        #anchor_score = self.calculate_cosine_score(stemmed_query, self.anchor_lengths, self.inverted_anchor)
        #scores = self.weighted_score(title_score, text_score, anchor_score)
        scores = self.weighted_score(title_score, text_score)
        scores_final = self.combine_scores(scores, self.normalized_pagerank_scores)
        # retrive the top 100 doc ids
        # top_docs = [score[0] for score in scores.take(100)]
        top_id = list(scores_final.keys())[:100]
        # returns (doc_id, title of doc_id) for the top 100 documents
        top_id_title = [(str(ID), self.title_id[ID]) for ID in top_id]
        return top_id_title

    def stem_query(self, query):
        # Stem the query terms and remove stopwords
        stemmed_query = [self.porter_stemmer.stem(term.group()) for term in self.RE_WORD.finditer(query.lower())]
        #stemmed_query = [self.porter_stemmer.stem(term) for term in query.split() if term not in self.all_stopwords]
        stop_tokens = set(stemmed_query).intersection(self.all_stopwords)
        query_terms = [t for t in stemmed_query if t not in stop_tokens]
        return query_terms

    def calculate_cosine_score(self, query, doc_lengths, inverted):
        """ Takes a  a query, and returns scores Dict with docs paired with relevance.
        Parameters:
        -----------
          query: List
            A List with terms from the query

          postings: Dictionary
            A Dictionary where the keys are doc_id and the value are lengths (norms).
        Returns:
        --------
          Dict
            A RDD Dict each element is a (doc_id, score).
        """
        # doc_lengths_rdd = self.sc.parallelize(list(doc_lengths.items()))
        # # Init scores with 0 for each doc
        # scores = doc_lengths_rdd.flatMap(lambda x: [(x[0], 0)])
        # # Loop over all words in query
        # for term in query:
        #     # Get docs that have this term
        #     docs = self.sc.parallelize(inverted.read_a_posting_list('.', term, self.bucket_name))
        #     # docs = sc.parallelize(read_posting_list(inverted, term))
        #     # Get (doc_id, tf_idf) pairs
        #     docs_by_id = docs.groupByKey().mapValues(lambda x: list(x))
        #     # Calculate scores
        #     part_scores = docs_by_id.flatMap(lambda x: [(x[0], np.sum([tf_idf for tf_idf in x[1]]))])
        #     scores = scores.leftOuterJoin(part_scores)
        # scores = scores.flatMap(lambda x: [(x[0], x[1][0] + x[1][1]) if x[1][1] is not None else (x[0], 0)])
        # # Normalize each score by the doc's length
        # scores = scores.join(doc_lengths_rdd).flatMap(lambda x: [(x[0], x[1][0] / x[1][1])])

        scores = defaultdict(float)
        # Loop over all words in the query
        for term in query:
            # Get docs that have this term
            docs = inverted.read_a_posting_list('.', term, self.bucket_name)

            # Calculate scores using list comprehension for efficiency
            for doc_id, term_freq in docs:

                # Ignore document if it belongs to union_doc_ids_list
                if doc_id in self.disambiguation_docs.keys():
                    continue

                scores[doc_id] += term_freq

        # Normalize each score by the doc's length
        scores = {k: scores[k] / doc_lengths[k] for k in scores.keys()}

        return scores

    def weighted_score(self, title_scores, text_scores, anchor_scores=None):
        # if anchor_scores != None:
        #     anchor_dict = anchor_scores.collectAsMap()
        #     # Join the RDDs based on the doc_id
        #     joined_rdd = title_scores.join(text_scores).join(anchor_scores)
        #
        #     # Compute the weighted sum for each doc_id
        #     weighted_sum = joined_rdd.flatMap(lambda x: (x[0],
        #                                                  x[1][0][0] * self.title_score +
        #                                                  x[1][0][1] * self.text_score +
        #                                                  x[1][1] * self.anchor_score
        #                                                  ))
        # else:
        #     # Join the RDDs based on the doc_id
        #     joined_rdd = title_scores.join(text_scores)
        #
        #     # Compute the weighted sum for each doc_id
        #     weighted_sum = joined_rdd.flatMap(lambda x: [(x[0],
        #                                                   x[1][0] * self.title_score +
        #                                                   x[1][1] * self.text_score
        #                                                   )])
        #
        # # weighted_sum = {k: self.title_score * title_dict[k] + self.text_score * text_dict[k] for k, v in title_dict}
        # # Sort docs by score
        # sorted_docs = list(weighted_sum).sort(reverse=True, key=lambda x: x[1])
        # sorted_docs = weighted_sum.sortBy(lambda x: x[1], ascending=False)

        if anchor_scores != None:
            keys = set(title_scores).union(text_scores).union(anchor_scores)

            weighted_sum = [(k, self.title_weight * title_scores.get(k, 0) + self.text_weight * text_scores.get(k, 0)
                            + self.anchor_weight * anchor_scores.get(k, 0)) for k in keys]
        else:
            # Combine keys from both dictionaries
            keys = set(title_scores).union(text_scores)

            # Calculate weighted sum and sort in one step
            weighted_sum = [(k, self.title_weight * title_scores.get(k, 0) + self.text_weight * text_scores.get(k, 0))
                            for k in keys]


        sorted_docs = dict(sorted(weighted_sum, key=lambda x: x[1], reverse=True))

        return sorted_docs


    def combine_scores(self, cosine_scores, pagerank_scores, alpha=0.7):
        # # Normalize PageRank scores
        # pagerank_sum = pagerank_scores.map(lambda x: x[1]).sum()
        # normalized_pagerank_scores = pagerank_scores.map(lambda x: (x[0], x[1] / pagerank_sum))
        #
        # # Join cosine similarity scores and normalized PageRank scores
        # combined_scores = cosine_scores.join(normalized_pagerank_scores)
        #
        # # Compute the combined score for each document
        # combined_scores = combined_scores.map(lambda x: (x[0], alpha * x[1][0] + (1 - alpha) * x[1][1]))
        #
        # return combined_scores

        # Normalize PageRank scores



        # Join cosine similarity scores and normalized PageRank scores
        combined_scores = {}
        for doc_id, cosine_score in cosine_scores.items():
            if doc_id in self.normalized_pagerank_scores:
                combined_scores[doc_id] = alpha * cosine_score + (1 - alpha) * self.normalized_pagerank_scores[doc_id]

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


