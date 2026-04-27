import os
import sys
from crawler import Crawler
from indexer import Indexer
from search import SearchEngine

def main():
    """
    The main Read-Eval-Print Loop (REPL) for the Search Engine CLI.
    """
    print("==================================================")
    print("   COMP3011 Search Engine - Initialization...   ")
    print("==================================================")
    print("Commands:")
    print("  build        - Crawl the website and build the index")
    print("  load         - Load an existing index from disk")
    print("  print <word> - Print the inverted index for a word")
    print("  find <query> - Search for a phrase")
    print("  quit         - Exit the program")
    print("==================================================\n")

    # System State
    indexer = Indexer()
    searcher = None

    # Ensure the data directory exists relative to where the script is run
    os.makedirs("data", exist_ok=True)

    # File paths for saving/loading
    CRAWL_FILE = "data/crawled_data.json"
    INDEX_FILE = "data/index.json"
    REGISTRY_FILE = "data/registry.json"

    while True:
        try:
            # Read input and split into command + arguments
            user_input = input("\n> ").strip()
            if not user_input:
                continue
                
            parts = user_input.split(" ", 1)
            command = parts[0].lower()
            args = parts[1] if len(parts) > 1 else ""

            # --- COMMAND: QUIT ---
            if command == "quit" or command == "exit":
                print("[*] Shutting down search engine. Goodbye!")
                sys.exit(0)

            # --- COMMAND: BUILD ---
            elif command == "build":
                print("[*] Starting full build pipeline...")
                crawler = Crawler()
                
                # 1. Crawl
                crawled_data = crawler.crawl()
                if not crawled_data:
                    print("[-] Crawl failed or returned no data.")
                    continue
                    
                import json
                with open(CRAWL_FILE, "w", encoding="utf-8") as f:
                    json.dump(crawled_data, f, indent=4)
                    
                # 2. Build Index
                indexer = Indexer() # Reset indexer to clear old state
                indexer.build_index(CRAWL_FILE)
                
                # 3. Save to disk
                indexer.save_index(INDEX_FILE, REGISTRY_FILE)
                
                # 4. Initialize Searcher
                searcher = SearchEngine(indexer.inverted_index, indexer.document_registry)
                print("[+] Build complete. Search engine is ready.")

            # --- COMMAND: LOAD ---
            elif command == "load":
                if not os.path.exists(INDEX_FILE) or not os.path.exists(REGISTRY_FILE):
                    print("[-] Error: Index files not found. Please run 'build' first.")
                    continue
                    
                indexer = Indexer()
                indexer.load_index(INDEX_FILE, REGISTRY_FILE)
                searcher = SearchEngine(indexer.inverted_index, indexer.document_registry)
                print("[+] Index loaded. Search engine is ready.")

            # --- COMMAND: PRINT ---
            elif command == "print":
                if not args:
                    print("[-] Usage: print <word>")
                    continue
                if not searcher:
                    print("[-] Error: Engine not initialized. Run 'build' or 'load' first.")
                    continue
                    
                # We only want the first word if they typed multiple
                word = args.split()[0] 
                searcher.print_word_index(word)

            # --- COMMAND: FIND ---
            elif command == "find":
                if not args:
                    print("[-] Usage: find <query phrase>")
                    continue
                if not searcher:
                    print("[-] Error: Engine not initialized. Run 'build' or 'load' first.")
                    continue
                    
                results = searcher.find(args)
                
                if results:
                    print(f"\n[+] Found {len(results)} matching documents:")
                    for rank, (score, url) in enumerate(results, 1):
                        print(f"    {rank}. [Score: {score:.4f}] {url}")

            # --- UNKNOWN COMMAND ---
            else:
                print(f"[-] Unknown command: '{command}'")

        # Catch Ctrl+C gracefully
        except KeyboardInterrupt:
            print("\n[*] Shutting down search engine. Goodbye!")
            sys.exit(0)

if __name__ == "__main__":
    main()