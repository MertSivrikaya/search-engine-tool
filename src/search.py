import math
import nltk
from nltk.stem import PorterStemmer

class SearchEngine:
    """
    Handles the query processing, interval intersection, and TF-IDF ranking.
    """
    def __init__(self, inverted_index, document_registry):
        self.index = inverted_index
        self.registry = document_registry
        self.total_docs = len(document_registry)
        self.stemmer = PorterStemmer()
        self.extent_tags = {'title': 3.0, 'h1': 2.0, 'h2': 1.5, 'b': 1.2, 'strong': 1.2} # Boost multipliers

    def _tokenize_query(self, query_string):
        """Processes the query using the exact same pipeline as the Indexer."""
        tokens = nltk.word_tokenize(query_string)
        valid_tokens = []
        for token in tokens:
            if token.isalnum():
                # Case insensitivity and stemming applied to the query
                valid_tokens.append(self.stemmer.stem(token.lower()))
        return valid_tokens

    def print_word_index(self, word):
        """
        Executes the 'print' command.
        Prints the posting list for a specific word.
        """
        stemmed_word = self.stemmer.stem(word.lower())
        if stemmed_word in self.index:
            print(f"\n[*] Inverted index for '{word}' (stemmed: '{stemmed_word}'):")
            print(self.index[stemmed_word])
        else:
            print(f"\n[-] Word '{word}' not found in the index.")

    def find(self, query_string):
        """
        Executes the 'find' command.
        Retrieves, ranks, and returns the relevant URLs for a query phrase.
        """
        tokens = self._tokenize_query(query_string)
        if not tokens:
            print("\n[-] Invalid or empty query.")
            return []

        # 1. Boolean Retrieval (Find documents containing all words;i.e, process conjunctive queries)
        doc_sets = []
        for token in tokens:
            if token in self.index:
                doc_sets.append(set(self.index[token].keys()))
            else:
                # If one word isn't in the index, the exact AND query fails
                print(f"\n[-] No documents found containing all search terms.")
                return []
                
        valid_doc_ids = set.intersection(*doc_sets)
        
        if not valid_doc_ids:
            print(f"\n[-] No documents found containing all search terms.")
            return []

        # 2. Rank the valid documents
        ranked_results = []
        for doc_id in valid_doc_ids:
            doc_score = 0.0
            
            for token in tokens:
                # TF-IDF Calculation
                tf = self.index[token][doc_id][0]
                df = len(self.index[token])
                idf = math.log(self.total_docs / df)
                base_score = tf * idf
                
                # Zone Boosting (Interval Intersection)
                positions = self.index[token][doc_id][1]
                zone_multiplier = 1.0
                
                for pos in positions:
                    for tag, boost_value in self.extent_tags.items():
                        extent_key = f"_EXTENT_{tag}"
                        if extent_key in self.index and doc_id in self.index[extent_key]:
                            for start, end in self.index[extent_key][doc_id]:
                                if start <= pos <= end:
                                    zone_multiplier = max(zone_multiplier, boost_value)
                
                doc_score += (base_score * zone_multiplier)
                
            # 3. Exact Phrase Proximity Boost
            if len(tokens) > 1:
                if self._check_exact_phrase(tokens, doc_id):
                    doc_score *= 2.0 # Massive boost if words appear sequentially (exact phrase match)

            ranked_results.append((doc_score, self.registry[doc_id]))

        # Sort by score descending
        ranked_results.sort(key=lambda x: x[0], reverse=True)
        return ranked_results

    def _check_exact_phrase(self, tokens, doc_id):
        """Checks if tokens appear sequentially using positional tracking."""
        # Grab the position lists for all query words in this document
        pos_lists = [self.index[token][doc_id][1] for token in tokens]
        
        # Check if there is any sequence where pos[i] == pos[0] + i
        for start_pos in pos_lists[0]:
            is_phrase = True
            for i in range(1, len(tokens)):
                if (start_pos + i) not in pos_lists[i]:
                    is_phrase = False
                    break
            if is_phrase:
                return True
        return False