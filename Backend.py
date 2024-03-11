import hashlib
from inverted_index_gcp import *
import nltk
from nltk.stem.porter import *
import tempfile
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from concurrent.futures import ThreadPoolExecutor
import inflect

# Download NLTK resources to a temporary directory
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
        self.title_weight = 0.6
        self.text_weight = 0.3
        self.anchor_weight = 0.1

        # GCP bucket name
        self.bucket_name = 'wikiproject-414111-bucket'

        # Stop words and stemmer
        self.english_stopwords = frozenset(stopwords.words('english'))
        self.corpus_stopwords = ["category", "references", "also", "external", "links",
                                 "may", "first", "see", "history", "people", "one", "two",
                                 "part", "thumb", "including", "second", "following",
                                 "many", "however", "would", "became", "who", "what", "why", "when", "was", "were"]
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
        # Get short pages
        self.short_docs = self.find_short_docs()

    def find_short_docs(self):
        """Get the short documents.

        Returns:
        --------
        dict:
            Dictionary where keys are document IDs and values are 1.
        """
        # Assign document IDs to a dictionary indicating short documents
        short_doc_ids_dict = defaultdict(int)
        for doc_id in self.text_lengths:
            # Check if the text length exceeds a certain threshold
            if self.text_lengths[doc_id] > 0.7:
                short_doc_ids_dict[doc_id] = 1  # Mark the document as short
        return short_doc_ids_dict

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

        # Create sets of document IDs from the posting lists
        doc_ids_title = set(doc_id for doc_id, _ in disambiguation_title)
        doc_ids_text = set(doc_id for doc_id, _ in disambiguation_text)

        # Compute the union of document IDs
        disambiguation_union_doc_ids = doc_ids_title.union(doc_ids_text)

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

        # Get documents from indexes, with relevance score
        title_score = self.multithread_cosine_similarity(stemmed_query, self.title_lengths, self.inverted_title)
        text_score = self.multithread_cosine_similarity(stemmed_query, self.text_lengths, self.inverted_text)
        anchor_score = self.multithread_cosine_similarity(stemmed_query, self.anchor_lengths, self.inverted_anchor)

        # Calculate weighted sum of each score
        scores = self.weighted_score(title_score, text_score, anchor_score)

        # Retrieve the top 100 doc ids
        top_id = list(scores.keys())[:100]

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
        # Tokenize the query, remove stopwords, and stem terms
        tokens = [token.group() for token in self.RE_WORD.finditer(query.lower())]
        stop_tokens = set(tokens).intersection(self.all_stopwords)
        query_terms = [t for t in tokens if t not in stop_tokens]      # Filter out stopwords
        stemmed_query = [self.porter_stemmer.stem(term) for term in query_terms]    # Stem query terms
        return stemmed_query

    def multithread_cosine_similarity(self, query, doc_lengths, inverted):
        """Calculate cosine similarity scores using multithreading.

        Parameters:
        -----------
        query : list
            List of stemmed query terms.
        doc_lengths : dict
            Dictionary containing document lengths.
        inverted : InvertedIndex
            Inverted index object.

        Returns:
        --------
        dict:
            Dictionary containing normalized cosine similarity scores.
        """
        scores = defaultdict(float)
        with ThreadPoolExecutor(max_workers=len(query)) as executor:  # Adjust max_workers as needed
            futures = []

            for term in query:
                # Process postings for each term concurrently
                future = executor.submit(self.process_postings, term, scores, inverted)
                futures.append(future)

            # Wait for all futures to complete
            for future in futures:
                future.result()

        # Normalize scores
        normalized_scores = {k: scores[k] / doc_lengths[k] for k in scores.keys()}
        return normalized_scores

    def process_postings(self, term, scores, inverted):
        """Process posting lists for a given term.

        Parameters:
        -----------
        term : str
            Term to process.
        scores : dict
            Dictionary to store scores.
        inverted : InvertedIndex
            Inverted index object.
        """
        docs = inverted.read_a_posting_list('.', term, self.bucket_name)
        for doc_id, term_freq in docs:
            # Accumulate scores for relevant documents excluding disambiguation pages and short docs
            if self.disambiguation_docs[doc_id] != 1 and self.short_docs[doc_id] != 1:
                scores[doc_id] += term_freq

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
        # Fetch postings for all terms in the query
        all_docs = [inverted.read_a_posting_list('.', term, self.bucket_name) for term in query]
        # Combine postings and calculate scores
        for docs in all_docs:
            for doc_id, term_freq in docs:
                # Accumulate scores for relevant documents excluding disambiguation pages and short docs
                if self.short_docs[doc_id] == 1:
                    continue
                if self.disambiguation_docs[doc_id] != 1:
                    scores[doc_id] += term_freq

        # Normalize scores
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
            # Combine scores from title and text dictionaries
            keys = set(title_scores).union(text_scores)

            # Calculate weighted sum
            weighted_sum = [(k, self.title_weight * title_scores.get(k, 0) + self.text_weight * text_scores.get(k, 0))
                            for k in keys]

        # Sort docs by score
        sorted_docs = dict(sorted(weighted_sum, key=lambda x: x[1], reverse=True))

        return sorted_docs

    def combine_scores(self, cosine_scores, alpha=0.6):
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
        combined_scores = {}
        for doc_id, cosine_score in cosine_scores.items():
            # Combine cosine similarity and PageRank scores using the specified weighting factor
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

    def convert_numeric_to_text(self, query, stemmed_query):
        """Convert numeric strings in the query to their textual representation.

        Parameters:
        -----------
        query : str
            The search query.

        Returns:
        --------
        str:
            The modified query with numeric strings converted to textual representation.
        """
        # Initialize inflect engine for converting numbers to words
        p = inflect.engine()

        # Tokenize the query
        tokens = nltk.word_tokenize(query)

        # Convert numeric strings to textual representation
        converted_query = []
        for token in tokens:
            if token.isdigit():
                # Convert the numeric string to its textual representation
                text_representation = p.number_to_words(token)
                text_representation = text_representation.replace('thousand', '').replace('hundred', '')
                converted_query.append(text_representation)

        # Join the converted tokens back into a string
        converted_query_string = ' '.join(converted_query)

        # Stem expanded query
        stemmed_converted = self.stem_query(converted_query_string)

        stemmed_query.extend(stemmed_converted)
        return stemmed_query