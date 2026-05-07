# 🔍 COMP3011 Search Engine

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Dependencies](https://img.shields.io/badge/Dependencies-Requests%20%7C%20BeautifulSoup4%20%7C%20NLTK-green)
![Build](https://img.shields.io/badge/Build-Passing-brightgreen)

A high-performance, full-text web search engine built from scratch in Python. This project features a robust ethical web crawler, an NLTK-powered NLP pipeline, a highly optimized minified Inverted Index, and a Vector Space search algorithm utilizing TF-IDF, Zone Extent Indexing, and Exact Phrase Proximity boosting.

Developed by Mert Sivrikaya for the COMP3011 Web Services and Web Data module.

## 📑 Table of Contents
- [Architecture Overview & Design Rationale](#architecture-overview--design-rationale)
- [Algorithmic Complexity & Optimization](#algorithmic-complexity--optimization)
- [Installation & Setup](#installation--setup)
- [Command-Line Interface (CLI) Usage](#command-line-interface-cli-usage)
- [Testing Suite](#testing-suite)
- [GenAI Declaration](#genai-declaration)

---

## 🏛 Architecture Overview & Design Rationale

This search tool is decoupled into three primary components to ensure modularity and maintainability:

1. **Crawler (`crawler.py`):** - **Rationale:** Designed as a polite, graph-navigating bot. It implements connection pooling via `requests.Session`, randomized politeness delays (minimum 6 seconds), and strict cycle prevention. It parses `robots.txt` dynamically to ensure ethical crawling compliance.
2. **Indexer (`indexer.py`):** - **Rationale:** Utilizes a Two-Pass NLP Tokenization strategy. `BeautifulSoup` extracts structural intervals (Zone Extents like `<h1>`, `<b>`), while `NLTK` handles alphanumeric filtering and Porter Stemming. 
   - **Data Structure:** Instead of verbose dictionaries, the Inverted Index uses optimized array mapping: `Word -> {DocID: [frequency, [positions]]}`. This drastically reduces the memory footprint and disk I/O time during JSON serialization.
3. **Search Engine (`search.py`):** - **Rationale:** Moves beyond simple Boolean retrieval by implementing a comprehensive ranking algorithm. It utilizes Term Frequency-Inverse Document Frequency (TF-IDF) alongside Extent Multipliers and an Exact Phrase Proximity booster (2.0x multiplier) for sequential token matches.

## ⚡ Algorithmic Complexity & Optimization

- **Space Complexity Optimization:** By transitioning the `index.json` output to a minified format (stripping whitespace and removing redundant keys in favor of positional arrays), the disk storage requirement was reduced significantly, achieving faster I/O during the `> load` command.
- **Search Time Complexity:** Document retrieval operates via Set Intersection across posting lists, allowing the engine to filter out invalid documents before running the heavier TF-IDF float calculations only on valid document subsets.

---

## ⚙️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/MertSivrikaya/search-engine-tool.git
   cd search-engine-tool
   ```

2. **Install dependencies:**
    Ensure you have Python 3.8+ installed.
    ```bash
    pip install requests beautifulsoup4 nltk
    ```

## 💻 Command-Line Interface (CLI) Usage

- The application features a custom Read-Eval-Print Loop (REPL) shell for interactive searching without reloading memory.

- Start the engine:
    ```bash
    python src/main.py
    ```

### Available Commands:
* **`> build`** - Crawls the target domain (`quotes.toscrape.com`), processes the NLP pipeline, and saves the highly optimized `index.json` and `registry.json` to the `/data` directory.
* **`> load`** - Instantly loads the pre-compiled JSON index from disk into memory.
* **`> find <query>`** - Executes a TF-IDF ranked search.
  * *Example:* `> find good friends`
* **`> print <word>`** - Prints the raw Inverted Index posting list for a specific stemmed word.
  * *Example:* `> print nonsense`
* **`> quit`** - Exits the application safely.

---

## 🧪 Testing Suite

- The codebase includes an extensive unittest suite that verifies mathematical logic, NLTK tokenization, graph navigation, and end-to-end pipeline integration. Mocked network requests ensure the crawler can be tested without repeatedly hitting live servers.

- Run the tests using:
    ```bash
    python -m unittest discover tests/
    ```

## 🤖 GenAI Declaration

- Note for grading: A full critical evaluation is provided in the accompanying 5-minute video demonstration.

- Generative AI, specifically Google Gemini, was utilized as an interactive sounding board during this project.

    - Usage: Assisted in generating NLTK setup boilerplate, brainstorming test-case mock structures, and optimizing JSON serialization configurations.

    - Reflection: All core algorithmic logic (TF-IDF calculations, exact phrase proximity tracking, and graph adjacency traversal) was independently analyzed and verified to ensure complete understanding of the underlying search mechanisms.