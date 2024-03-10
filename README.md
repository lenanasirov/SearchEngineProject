# SearchEngineProject
## Overview
This project implements a search engine tailored for Wikipedia articles. It leverages cosine similarity and PageRank scoring to retrieve relevant documents based on user queries. The implementation uses Python and several libraries such as NLTK for natural language processing, and GCP for storage.
## Code Structure
The codebase is organized into several components:

1. **Search Frontend (search_frontend.py):** This script serves as the entry point for the search engine. It initializes the backend and handles user queries via a Flask RESTful API. The API offers endpoints for different types of searches, including searching in the title, body, and anchor text of articles, as well as retrieving PageRank and page view data.

2. **Backend (backend.py):** The Backend class contains the core functionality of the search engine. It handles the retrieval of documents, processing of queries, calculation of relevance scores, and combination of scores based on cosine similarity and PageRank.

3. **Inverted Index (inverted_index_gcp.py):** This module provides functionality to read indexes stored in Google Cloud Storage (GCS). It facilitates the retrieval of posting lists for terms and document lengths.

## Functionality 
### 1. Intialization 
Upon initialization, the Backend class loads necessary indexes and dictionaries from GCS. These include inverted indexes for titles, text, and anchors, document lengths, document titles, PageRank scores, disambiguation pages and short pages.

### 2. Query Processing

When a user submits a query, the backend processes it through several steps:

* **Query Parsing:** The query is parsed, tokenized, and stemmed. Stopwords are removed, and terms are normalized for further processing.

* **Query Expansion (Optional):** The query terms may be expanded using WordNet synonyms to improve recall.

* **Numeric Conversion (Optional):** Numeric strings in the query are converted to their textual representation, enhancing search accuracy.

### 3. Document Retrieval
The backend performs document retrieval using cosine similarity. It calculates scores for titles, text, and anchors separately, and then combines them into a weighted sum. PageRank scores are also considered and combined with cosine similarity scores to produce final relevance scores.

### 4. Filtering of Short Documents
Short documents, which may not provide substantial content, are filtered out during the retrieval process. Documents with text lengths above a certain threshold are considered for scoring, ensuring that only relevant articles are included in the search results.

### 5. Filtering of Disambiguation Pages
During document retrieval, disambiguation pages are filtered out to enhance the relevance and quality of search results. Disambiguation pages typically contain lists of articles with similar titles, making them less relevant for direct user queries. By excluding disambiguation pages from the search results, the search engine ensures that users receive more focused and pertinent information related to their queries. This filtering process improves the overall search experience and helps users find the most relevant Wikipedia articles more efficiently.

### 6. Multithreading
Multithreading is utilized to improve the efficiency of cosine similarity calculations. Each term in the query is processed concurrently, allowing for faster retrieval of relevant documents. The ThreadPoolExecutor class from the concurrent.futures module is employed to manage and execute multiple threads efficiently.

## Conclusion
This search engine provides an efficient and scalable solution for retrieving Wikipedia articles based on user queries. By combining cosine similarity and PageRank scoring, it delivers relevant search results while considering both content relevance and document authority. The modular structure allows for easy maintenance and scalability, making it suitable for handling large-scale search operations. Additionally, the filtering of short documents and the use of multithreading contribute to improved performance and search accuracy.
