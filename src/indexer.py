import json
from bs4 import BeautifulSoup
import nltk
from nltk.stem import PorterStemmer

# Automatically download required NLTK data 
nltk.download('punkt', quiet=True)

class HTMLProcessor:
    """
    Handles Two-Pass Tokenization using BeautifulSoup and NLTK.
    Implements Extent-based Zone Indexing to preserve exact token positions
    and the start/end intervals of important HTML structural tags.
    Pass 1: Identify HTML tags/structure using BeautifulSoup (Parse).
    Pass 2: Extract, clean, stem, and score the tokens (Tokenize).
    """
    def __init__(self):
        self.extent_tags = {'title', 'h1', 'h2', 'h3', 'b', 'strong', 'em', 'i'}
        self.stemmer = PorterStemmer()

    def tokenize(self, html_content):
        """
        Parses HTML and returns a dictionary
        containing both word postings and extent postings.

        Intended output format:
        {
            stemmed_word: {"frequency": occurrence_count, "positions": [pos1, pos2, ...]},
            _EXTENT_tag: {"extents": [[start_pos1, end_pos1], [start_pos2, end_pos2], ... ]}
        }

        However, instead of using a verbose dictionary for token statistics, we will optimize the output to a python list
        where the index implies the meaning:
            Index 0: The integer frequency of the token in the document
            Index 1: A list of integer positions where the token appears in the document
        
        This will save space in the final JSON file, as we won't have to repeat the keys

        For example:
            Instead of:
                "einstein": {
                    "1": {"frequency": 2, "positions": [4, 88]}
                }
            We will have:
                "einstein": {
                    "1": [2, [4, 88]]
                }
        """

        soup = BeautifulSoup(html_content, 'html.parser')
        
        index_data = {}

        # Initialize the extent arrays in our dictionary
        for tag in self.extent_tags:
            index_data[f"_EXTENT_{tag}"] = []
            
        # Track the position of tokens in the document
        global_position = 0  
        
        # Pass 1: Iterate through all strings (text nodes) in the HTML
        for text_node in soup.find_all(string=True):
            # Skip hidden elements, scripts, and styling
            if text_node.parent.name in ['style', 'script', 'head', 'meta', '[document]']:
                continue
                
            raw_text = text_node.strip()
            if not raw_text:
                continue
                 
            # Pass 2: NLTK Processing
            tokens = nltk.word_tokenize(raw_text)
            
            # Filter and stem tokens
            valid_tokens = []
            for token in tokens:
                # Keep only alphanumeric tokens to completely strip punctuation
                if token.isalnum():
                    # Reduce word to its root form (e.g., "running" -> "run")
                    valid_tokens.append(self.stemmer.stem(token.lower()))
            
            if not valid_tokens:
                continue

            start_pos = global_position
            end_pos = global_position + len(valid_tokens) - 1

            # Store the token data (Frequency and Positions)
            for i, token in enumerate(valid_tokens):
                pos = start_pos + i
                if token not in index_data:
                    index_data[token] = [0, []]  # [frequency, [positions]]
                    
                index_data[token][0] += 1
                index_data[token][1].append(pos)

            # Store the Extent Data (Zone Indexing)
            # Walk up the HTML tree to see if this text is inside nested tags (e.g., <h1><b>Text</b></h1>)
            current_parent = text_node.parent
            active_extents = set()
            
            while current_parent is not None and current_parent.name != '[document]':
                if current_parent.name in self.extent_tags:
                    active_extents.add(current_parent.name)
                current_parent = current_parent.parent
                
            # Save the [start, end] interval for any active tags
            for tag in active_extents:
                index_data[f"_EXTENT_{tag}"].append([start_pos, end_pos])
                
            global_position += len(valid_tokens)

        # Merge adjacent extent fragments caused by nested HTML tags 
        for tag_key in list(index_data.keys()):
            if tag_key.startswith("_EXTENT_") and index_data[tag_key]:
                # Sort just to be safe, though they should already be sequential
                extents = sorted(index_data[tag_key], key=lambda x: x[0])
                merged_extents = [extents[0]]
                
                for current in extents[1:]:
                    previous = merged_extents[-1]
                    # If the current fragment starts exactly after the previous one ends (or overlaps)
                    if current[0] <= previous[1] + 1:
                        # Fuse them together
                        previous[1] = max(previous[1], current[1])
                    else:
                        merged_extents.append(current)
                        
                # Update the dictionary with the clean, merged list
                index_data[tag_key] = merged_extents

        # Clean up empty extents so we don't bloat the final JSON file
        final_index_data = {
            k: v for k, v in index_data.items() 
            if (not k.startswith("_EXTENT_")) or (k.startswith("_EXTENT_") and v)
        }          
                             
        return final_index_data

class Indexer:
    """
    Builds the Inverted Index and handles Document IDs.
    """
    def __init__(self):
        self.document_registry = {}  # Maps DocID (int) -> URL (str)
        self.inverted_index = {}     # Maps Word (str) -> Dict[DocID, [frequency, [positions]]]
        self.processor = HTMLProcessor()
        self.next_doc_id = 1

    def build_index(self, crawled_data_filepath):
        """
        Reads the crawled data, assigns DocIDs, and builds the global Inverted Index.
        """
        print("[*] Building inverted index...")
        with open(crawled_data_filepath, 'r', encoding='utf-8') as f:
            crawled_data = json.load(f)

        for url, data in crawled_data.items():
            # 1. Assign DocID and update registry
            doc_id = self.next_doc_id
            self.document_registry[doc_id] = url
            self.next_doc_id += 1

            # 2. Process the raw HTML
            html_content = data.get("html", "")
            if not html_content:
                continue
                
            doc_index_data = self.processor.tokenize(html_content)

            # 3. Merge the document's terms into the global Inverted Index
            for term, term_data in doc_index_data.items():
                if term not in self.inverted_index:
                    self.inverted_index[term] = {}
                
                # Map the specific DocID to its term frequencies, positions, or extents
                self.inverted_index[term][doc_id] = term_data

        print(f"\n[+] Successfully indexed {len(self.document_registry)} documents.")

    def save_index(self, index_filepath, registry_filepath):
        """
        Saves the Inverted Index and Document Registry to the file system.
        """
        print(f"[*] Saving index to {index_filepath}...")
        with open(index_filepath, 'w', encoding='utf-8') as f:
            json.dump(self.inverted_index, f, indent=4)
            
        print(f"[*] Saving document registry to {registry_filepath}...")
        with open(registry_filepath, 'w', encoding='utf-8') as f:
            json.dump(self.document_registry, f, indent=4)
            
        print(f"\n[+] Save complete!")
        
    def load_index(self, index_filepath, registry_filepath):
        """
        Loads the Inverted Index and Document Registry from the file system.
        """
        print(f"[*] Loading index from {index_filepath}...")
        with open(index_filepath, 'r', encoding='utf-8') as f:
            self.inverted_index = json.load(f)
            
        print(f"[*] Loading document registry from {registry_filepath}...")
        with open(registry_filepath, 'r', encoding='utf-8') as f:
            # JSON dict keys are always strings, so we convert them back to integers
            loaded_registry = json.load(f)
            self.document_registry = {int(k): v for k, v in loaded_registry.items()}
            
        # Update next_doc_id so we don't overwrite if we build more later
        if self.document_registry:
            self.next_doc_id = max(self.document_registry.keys()) + 1
            print(f"[*] Updated next document ID to {self.next_doc_id}")
        
        print(f"\n[+] Load complete!")