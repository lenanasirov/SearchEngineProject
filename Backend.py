import hashlib
from inverted_index_gcp import *
import nltk
from nltk.stem.porter import *
import tempfile
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize

nltk.download('wordnet', download_dir=tempfile.gettempdir())
nltk.download('stopwords', download_dir=tempfile.gettempdir())
nltk.download('punkt', download_dir=tempfile.gettempdir())
nltk.data.path.append(tempfile.gettempdir())


def _hash(s):
    """Returns the hash value of the input string.

    Parameters:
    -----------
    s : str
        Input string to be hashed.

    Returns:
    --------
    str
        Hexadecimal hash value of the input string.
    """
    return hashlib.blake2b(bytes(s, encoding='utf8'), digest_size=5).hexdigest()


class Backend:
    """Backend class for performing search operations using cosine similarity and PageRank scoring."""

    def __init__(self):
        """Initialize Backend object with default parameters and load necessary indexes and dictionaries."""

        # Default weights for different components
        self.title_weight = 0.55
        self.text_weight = 0.45
        self.anchor_weight = 0

        # GCP bucket name
        self.bucket_name = 'wikiproject-414111-bucket'

        # Stop words and stemmer
        self.english_stopwords = frozenset(stopwords.words('english'))
        self.corpus_stopwords = ["category", "references", "also", "external", "links",
                                 "may", "first", "see", "history", "people", "one", "two",
                                 "part", "thumb", "including", "second", "following",
                                 "many", "however", "would", "became"]
        self.all_stopwords = self.english_stopwords.union(self.corpus_stopwords)
        self.RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)
        self.porter_stemmer = PorterStemmer()

        # Load indexes and dictionaries
        self.inverted_title = InvertedIndex.read_index('title_postings', 'index_title', self.bucket_name)
        self.inverted_text = InvertedIndex.read_index('text_postings', 'index_text', self.bucket_name)
        self.inverted_anchor = InvertedIndex.read_index('anchor_postings', 'index_anchor', self.bucket_name)
        self.title_lengths = InvertedIndex.read_index('title_postings', 'title_lengths', self.bucket_name)
        self.text_lengths = InvertedIndex.read_index('text_postings', 'text_lengths', self.bucket_name)
        self.anchor_lengths = InvertedIndex.read_index('anchor_postings', 'anchor_lengths', self.bucket_name)
        self.title_id = InvertedIndex.read_index('.', 'title_id', self.bucket_name)
        self.pagerank = InvertedIndex.read_index('.', 'pagerank', self.bucket_name)

        # Normalize PageRank
        pagerank_min = min(self.pagerank.values())
        pagerank_max = max(self.pagerank.values())
        self.normalized_pagerank_scores = {doc_id: (score - pagerank_min) / (pagerank_max - pagerank_min)
                                           for doc_id, score in self.pagerank.items()}

        # Get disambiguation pages
        self.disambiguation_docs = self.disambiguation_union()

    def disambiguation_union(self):
        """Get the union of disambiguation pages from different indexes.

        Returns:
        --------
        dict:
            Dictionary where keys are document IDs and values are 1.
        """

        # Get disambiguation pages of every index
        disambiguation_stem = self.porter_stemmer.stem('disambiguation')
        disambiguation_title = self.inverted_title.read_a_posting_list('.', disambiguation_stem, self.bucket_name)
        disambiguation_text = self.inverted_text.read_a_posting_list('.', disambiguation_stem, self.bucket_name)
        disambiguation_anchor = self.inverted_anchor.read_a_posting_list('.', disambiguation_stem, self.bucket_name)
        # Create sets of document IDs from the posting lists
        doc_ids_title = set(doc_id for doc_id, _ in disambiguation_title)
        doc_ids_text = set(doc_id for doc_id, _ in disambiguation_text)
        doc_ids_anchor = set(doc_id for doc_id, _ in disambiguation_anchor)
        # Compute the union of document IDs
        disambiguation_union_doc_ids = doc_ids_title.union(doc_ids_text, doc_ids_anchor)
        # Assign document IDs to a dictionary
        disambiguation_union_doc_ids_dict = defaultdict(int)
        for doc_id in disambiguation_union_doc_ids:
            disambiguation_union_doc_ids_dict[doc_id] = 1
        return disambiguation_union_doc_ids_dict

    def backend_search(self, query):
        """Perform a search operation based on the given query.

        Parameters:
        -----------
        query : str
            The search query.

        Returns:
        --------
        list:
            A list of tuples containing document IDs and corresponding titles.
        """

        # Stem and expand query
        stemmed_query = self.stem_query(query)
        expanded_query = self.expand_query(query, stemmed_query)

        # Get documents from indexes, with relevance score
        title_score = self.calculate_cosine_score(expanded_query, self.title_lengths, self.inverted_title)
        text_score = self.calculate_cosine_score(expanded_query, self.text_lengths, self.inverted_text)
        # anchor_score = self.calculate_cosine_score(stemmed_query, self.anchor_lengths, self.inverted_anchor)

        # Calculate weighted sum of each score
        scores = self.weighted_score(title_score, text_score)
        # scores = self.weighted_score(title_score, text_score, anchor_score)

        # Change scores according to pagerank
        scores_final = self.combine_scores(scores)

        # Retrieve the top 100 doc ids
        top_id = list(scores_final.keys())[:100]

        # Get (doc_id, title of doc_id) for the top 100 documents
        top_id_title = [(str(ID), self.title_id[ID]) for ID in top_id]
        return top_id_title

    def stem_query(self, query):
        """Stem the query terms and remove stopwords.

        Parameters:
        -----------
        query : str
            The search query.

        Returns:
        --------
        list:
            A list of stemmed query terms.
        """
        # Stem the query terms and remove stopwords
        stemmed_query = [self.porter_stemmer.stem(term.group()) for term in self.RE_WORD.finditer(query.lower())]
        stop_tokens = set(stemmed_query).intersection(self.all_stopwords)
        query_terms = [t for t in stemmed_query if t not in stop_tokens]
        return query_terms

    def calculate_cosine_score(self, query, doc_lengths, inverted):
        """ Takes a query, and returns scores dictionary with docs paired with relevance.
        Parameters:
        -----------
          query: List
            A List with terms from the query

          doc_lengths: Dictionary
            A Dictionary where the keys are doc_id and the value are lengths (norms).

          inverted: InvertedIndex
            An InvertedIndex object

        Returns:
        --------
        dict:
            Dictionary containing document IDs and corresponding scores.
        """
        scores = defaultdict(float)
        # Loop over all words in the query
        for term in query:
            # Get docs that have this term
            docs = inverted.read_a_posting_list('.', term, self.bucket_name)

            # Sum tf_idf for each doc
            for doc_id, term_freq in docs:

                # Ignore document if it belongs to disambiguation_docs
                if self.disambiguation_docs[doc_id] != 1:
                    scores[doc_id] += term_freq

        # Normalize each score by the doc's length
        scores = {k: scores[k] / doc_lengths[k] for k in scores.keys()}

        return scores

    def weighted_score(self, title_scores, text_scores, anchor_scores=None):
        """Calculate weighted scores based on relevance scores of different components.

        Parameters:
        -----------
        title_scores : dict
            Dictionary containing title scores.
        text_scores : dict
            Dictionary containing text scores.
        anchor_scores : dict, optional
            Dictionary containing anchor scores (default is None).

        Returns:
        --------
        dict:
            Dictionary containing combined scores.
        """

        if anchor_scores is not None:
            # Combine keys from all dictionaries
            keys = set(title_scores).union(text_scores).union(anchor_scores)

            # Calculate weighted sum
            weighted_sum = [(k, self.title_weight * title_scores.get(k, 0) + self.text_weight * text_scores.get(k, 0)
                             + self.anchor_weight * anchor_scores.get(k, 0)) for k in keys]
        else:
            # Combine keys from both dictionaries
            keys = set(title_scores).union(text_scores)

            # Calculate weighted sum
            weighted_sum = [(k, self.title_weight * title_scores.get(k, 0) + self.text_weight * text_scores.get(k, 0))
                            for k in keys]

        # Sort docs by score
        sorted_docs = dict(sorted(weighted_sum, key=lambda x: x[1], reverse=True))

        return sorted_docs

    def combine_scores(self, cosine_scores, alpha=0.7):
        """Combine cosine similarity scores and PageRank scores.

        Parameters:
        -----------
        cosine_scores : dict
            Dictionary containing cosine similarity scores.
        alpha : float, optional
            Weighting factor for cosine scores (default is 0.7).

        Returns:
        --------
        dict:
            Dictionary containing combined scores.
        """
        # For every doc, calculate weighted sum of scores and pagerank
        combined_scores = {}
        for doc_id, cosine_score in cosine_scores.items():
            if doc_id in self.normalized_pagerank_scores:
                combined_scores[doc_id] = alpha * cosine_score + (1 - alpha) * self.normalized_pagerank_scores[doc_id]
            else:
                combined_scores[doc_id] = cosine_score

        return combined_scores

    def expand_query(self, query, stemmed_query, top_n=2):
        """Expand the query terms using WordNet synonyms.

        Parameters:
        -----------
        query : str
            The search query.
        stemmed_query : list
            List of stemmed query terms.
        top_n : int, optional
            Number of top synonyms to consider (default is 2).

        Returns:
        --------
        list:
            List of expanded query terms.
        """
        expanded_query = []

        # Tokenize the query and remove stopwords
        tokens = nltk.word_tokenize(query)
        stop_tokens = set(tokens).intersection(self.all_stopwords)
        filtered_tokens = [t for t in tokens if t not in stop_tokens]

        # Iterate over each token in the query
        for token in filtered_tokens:
            # Get synsets for the token
            synsets = wordnet.synsets(token.lower())

            if synsets:
                # Get the most common lemma for each synset
                lemmas = [synset.lemmas()[0].name() for synset in synsets]
                expansion = []

                # Remove duplicates, phrases and words that have the token in them
                for lemma in lemmas:
                    if ('_' not in lemma
                            and self.porter_stemmer.stem(token.lower()) not in self.porter_stemmer.stem(lemma)
                            and token.lower() not in lemma):
                        expansion.append(lemma)

                # Select top_n lemmas
                expansion = expansion[:top_n]

                # Add lemmas to the expanded query
                expanded_query.extend(expansion)

        # Stem expanded query
        stemmed_expanded = [self.porter_stemmer.stem(term) for term in expanded_query]

        stemmed_query.extend(stemmed_expanded)

        return stemmed_query
