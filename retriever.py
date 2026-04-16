"""
=============================================================
  Phase 5 — Wikipedia Retrieval System
  Purpose : Given any statement, fetch relevant evidence
            text from Wikipedia to use as context for
            Phase 3 (detection) and Phase 4 (NLI) models
  Input   : Any text statement
  Output  : Evidence text from Wikipedia
=============================================================
HOW IT FITS IN THE PIPELINE:
  User inputs statement
       ↓
  Phase 5 fetches Wikipedia evidence   ← THIS SCRIPT
       ↓
  Phase 3 detection model uses evidence to classify
       ↓
  Phase 4 NLI model checks if evidence contradicts statement
       ↓
  Phase 6 correction engine uses evidence to fix statement

HOW TO RUN:
  pip install wikipedia-api sentence-transformers faiss-cpu
  python retriever.py
=============================================================
"""

import os
import re
import time
import json
import numpy as np
import wikipediaapi
from sentence_transformers import SentenceTransformer
import faiss
import warnings
warnings.filterwarnings("ignore")

# =============================================================
# CONFIGURATION
# =============================================================

CONFIG = {
    # Wikipedia settings
    "language"        : "en",
    "user_agent"      : "HallucinationDetectionProject/1.0",

    # How many Wikipedia search results to consider
    "max_search_results" : 3,

    # How many sentences to return as evidence
    "max_sentences"   : 5,

    # Sentence embedding model for semantic search
    # Small and fast — good for 4GB GPU
    "embedding_model" : "all-MiniLM-L6-v2",

    # Cache file — saves Wikipedia results so we don't
    # fetch the same page twice (saves time and API calls)
    "cache_file"      : "./retriever_cache.json",

    # Timeout for Wikipedia requests in seconds
    "timeout"         : 10,
}

# =============================================================
# STEP 1 — INITIALIZE WIKIPEDIA API
# =============================================================

print("=" * 60)
print("  Phase 5 — Wikipedia Retrieval System")
print("=" * 60)

# Initialize Wikipedia API client
# user_agent identifies our app to Wikipedia's servers
# Wikipedia requires a user agent string — anonymous requests get blocked
wiki = wikipediaapi.Wikipedia(
    language   = CONFIG["language"],
    user_agent = CONFIG["user_agent"]
)

print(f"\n✅ Wikipedia API initialized")

# =============================================================
# STEP 2 — LOAD SENTENCE EMBEDDING MODEL
# Used to find the most relevant sentences from a Wikipedia page
# =============================================================

print(f"--- Loading sentence embedding model ---")
print(f"   Model: {CONFIG['embedding_model']}")
print(f"   (downloads ~80MB on first run, cached after)\n")

# SentenceTransformer converts sentences to 384-dimensional vectors
# Sentences with similar meaning have similar vectors
# This lets us find the most relevant evidence for any claim
embedder = SentenceTransformer(CONFIG["embedding_model"])

print("✅ Embedding model loaded\n")

# =============================================================
# STEP 3 — CACHE SYSTEM
# Saves Wikipedia responses to disk so repeat queries
# don't need another API call
# =============================================================

def load_cache():
    """Load cached Wikipedia results from disk."""
    if os.path.exists(CONFIG["cache_file"]):
        with open(CONFIG["cache_file"], "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_cache(cache):
    """Save cache to disk."""
    with open(CONFIG["cache_file"], "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)

# Load existing cache at startup
cache = load_cache()
print(f"✅ Cache loaded — {len(cache)} entries cached\n")

# =============================================================
# STEP 4 — TEXT CLEANING UTILITIES
# =============================================================

def clean_text(text):
    """
    Cleans Wikipedia text by removing references,
    extra whitespace, and special characters.
    """
    # Remove reference numbers like [1], [2], [citation needed]
    text = re.sub(r'\[\d+\]', '', text)
    text = re.sub(r'\[citation needed\]', '', text)
    text = re.sub(r'\[edit\]', '', text)

    # Remove multiple spaces and newlines
    text = re.sub(r'\s+', ' ', text)

    # Strip leading/trailing whitespace
    text = text.strip()

    return text


def split_into_sentences(text):
    """
    Splits a paragraph into individual sentences.
    Uses simple rule-based splitting on '. ', '? ', '! '
    """
    # Split on sentence boundaries
    sentences = re.split(r'(?<=[.!?])\s+', text)

    # Filter out very short sentences (likely noise)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]

    return sentences

# =============================================================
# STEP 5 — WIKIPEDIA PAGE FETCHER
# =============================================================

def fetch_wikipedia_page(title):
    """
    Fetches a Wikipedia page by exact title.
    Returns the page text or None if page doesn't exist.
    Uses cache to avoid repeated API calls.
    """
    # Check cache first
    if title in cache:
        return cache[title]

    try:
        page = wiki.page(title)

        if not page.exists():
            return None

        # Get full page text
        text = clean_text(page.text)

        # Limit to first 5000 characters to avoid very long pages
        # Most important info is near the top
        text = text[:5000]

        # Save to cache
        cache[title] = text
        save_cache(cache)

        # Be polite to Wikipedia servers
        # Avoid sending too many requests too quickly
        time.sleep(0.5)

        return text

    except Exception as e:
        print(f"   ⚠ Error fetching '{title}': {e}")
        return None


def search_wikipedia(query):
    """
    Searches Wikipedia for pages related to a query.
    Returns a list of page titles.

    Wikipedia API does not have a direct search function,
    so we use the opensearch endpoint via the page suggestion.
    """
    try:
        # WikipediaAPI suggests pages based on query
        # We try the query directly first, then variations
        titles_to_try = []

        # Try direct page lookup
        page = wiki.page(query)
        if page.exists():
            titles_to_try.append(query)

        # Try with common variations
        # Extract key entities from the query for better search
        words = query.split()
        if len(words) > 1:
            # Try first 2-3 words as a title
            titles_to_try.append(" ".join(words[:3]))
            titles_to_try.append(" ".join(words[:2]))

        return titles_to_try[:CONFIG["max_search_results"]]

    except Exception as e:
        print(f"   ⚠ Search error: {e}")
        return []

# =============================================================
# STEP 6 — SEMANTIC SENTENCE RETRIEVAL
# Finds the most relevant sentences from a page for a given claim
# =============================================================

def get_most_relevant_sentences(claim, page_text, top_k=5):
    """
    Given a claim and a Wikipedia page text, returns the
    top-k most semantically relevant sentences.

    Uses sentence embeddings + FAISS similarity search.

    Args:
        claim     : the statement we want evidence for
        page_text : full Wikipedia page text
        top_k     : number of sentences to return

    Returns:
        List of most relevant sentences
    """
    # Split page into sentences
    sentences = split_into_sentences(page_text)

    if not sentences:
        return []

    # If very few sentences, return them all
    if len(sentences) <= top_k:
        return sentences

    # Encode all sentences to vectors
    # Shape: [num_sentences, 384]
    sentence_embeddings = embedder.encode(
        sentences,
        convert_to_numpy = True,
        show_progress_bar = False
    )

    # Encode the claim to a vector
    # Shape: [1, 384]
    claim_embedding = embedder.encode(
        [claim],
        convert_to_numpy = True,
        show_progress_bar = False
    )

    # Normalize vectors for cosine similarity
    # FAISS IndexFlatIP computes dot product — with normalized vectors
    # this equals cosine similarity
    faiss.normalize_L2(sentence_embeddings)
    faiss.normalize_L2(claim_embedding)

    # Build FAISS index
    # IndexFlatIP = exact Inner Product search (no approximation)
    dimension = sentence_embeddings.shape[1]   # 384
    index     = faiss.IndexFlatIP(dimension)
    index.add(sentence_embeddings)

    # Search for top_k most similar sentences
    # Returns distances and indices of best matches
    distances, indices = index.search(claim_embedding, top_k)

    # Retrieve the actual sentences in order of relevance
    relevant_sentences = [sentences[i] for i in indices[0] if i < len(sentences)]

    return relevant_sentences

# =============================================================
# STEP 7 — MAIN EVIDENCE RETRIEVAL FUNCTION
# This is what gets called by Phase 3, 4, 6, and the UI
# =============================================================

def get_evidence(claim, verbose=False):
    """
    Main function — given a claim, retrieve relevant
    evidence from Wikipedia.

    This function is imported by:
      - train_detector.py  (Phase 3 inference)
      - train_nli.py       (Phase 4 inference)
      - train_corrector.py (Phase 6 inference)
      - app.py             (Phase 8 Streamlit UI)

    Args:
        claim   : the statement to find evidence for
        verbose : if True, prints search progress

    Returns:
        dict with:
          evidence      : the most relevant text as a string
          sentences     : list of individual evidence sentences
          source        : Wikipedia page title used
          found         : True if evidence was found
    """

    if verbose:
        print(f"\n   Searching Wikipedia for: '{claim[:80]}...'")

    # Step A — Generate search queries from the claim
    # Extract key entities/phrases for better search
    search_titles = search_wikipedia(claim)

    if not search_titles:
        # Fallback: use first 4 words of claim as search query
        words = claim.split()[:4]
        search_titles = [" ".join(words)]

    # Step B — Try each title until we find a valid page
    best_evidence  = ""
    best_sentences = []
    source_title   = ""

    for title in search_titles:
        if verbose:
            print(f"   Trying: '{title}'")

        page_text = fetch_wikipedia_page(title)

        if page_text:
            # Step C — Find most relevant sentences from this page
            relevant = get_most_relevant_sentences(
                claim,
                page_text,
                top_k = CONFIG["max_sentences"]
            )

            if relevant:
                best_sentences = relevant
                best_evidence  = " ".join(relevant)
                source_title   = title
                break   # Stop at first good result

    if not best_evidence:
        if verbose:
            print("   ⚠ No evidence found")
        return {
            "evidence"  : "",
            "sentences" : [],
            "source"    : "",
            "found"     : False
        }

    if verbose:
        print(f"   ✅ Found evidence from: '{source_title}'")
        print(f"   Evidence preview: {best_evidence[:150]}...")

    return {
        "evidence"  : best_evidence,
        "sentences" : best_sentences,
        "source"    : source_title,
        "found"     : True
    }

# =============================================================
# STEP 8 — BATCH EVIDENCE RETRIEVAL
# Fetch evidence for multiple claims at once
# Used during evaluation (Phase 7) on large datasets
# =============================================================

def get_evidence_batch(claims, verbose=True):
    """
    Fetches evidence for a list of claims.
    Shows progress and handles errors gracefully.

    Args:
        claims  : list of statement strings
        verbose : show progress

    Returns:
        list of evidence dicts (same format as get_evidence)
    """
    results = []

    for i, claim in enumerate(claims):
        if verbose and (i + 1) % 10 == 0:
            print(f"   Processed {i+1}/{len(claims)} claims...")

        result = get_evidence(claim, verbose=False)
        results.append(result)

        # Small delay to be respectful of Wikipedia's servers
        time.sleep(0.3)

    if verbose:
        found = sum(1 for r in results if r["found"])
        print(f"\n   Evidence found for {found}/{len(claims)} claims")

    return results

# =============================================================
# STEP 9 — TEST THE RETRIEVER
# Run this section to verify everything works
# =============================================================

if __name__ == "__main__":

    print("\n" + "=" * 60)
    print("  Testing Wikipedia Retrieval System")
    print("=" * 60)

    # Test cases — mix of factual and hallucinated statements
    test_claims = [
        "The Eiffel Tower is located in Paris, France.",
        "Albert Einstein won the Nobel Prize in Physics.",
        "The Amazon River is the longest river in the world.",
        "Shakespeare was born in Stratford-upon-Avon.",
        "The Great Wall of China is visible from space.",
        "Water boils at 100 degrees Celsius at sea level.",
    ]

    print(f"\nTesting {len(test_claims)} claims...\n")

    for i, claim in enumerate(test_claims, 1):
        print(f"\n{'─'*60}")
        print(f"Claim {i}: {claim}")

        result = get_evidence(claim, verbose=True)

        if result["found"]:
            print(f"\n  Source    : {result['source']}")
            print(f"  Evidence  :")
            for j, sent in enumerate(result["sentences"][:3], 1):
                print(f"    {j}. {sent[:120]}...")
        else:
            print("  No evidence found for this claim")

        # Small pause between requests
        time.sleep(1)

    print(f"\n{'='*60}")
    print("  Retrieval Test Complete")
    print(f"  Cache saved to: {os.path.abspath(CONFIG['cache_file'])}")
    print("=" * 60)

    # ── SINGLE CLAIM DEMO ─────────────────────────────────
    print("\n\n--- Interactive Demo ---")
    print("Enter a statement to find Wikipedia evidence")
    print("(Press Ctrl+C to exit)\n")

    while True:
        try:
            claim = input("Statement: ").strip()
            if not claim:
                continue

            result = get_evidence(claim, verbose=True)

            if result["found"]:
                print(f"\n✅ Evidence from Wikipedia ({result['source']}):")
                print(f"   {result['evidence'][:400]}...")
            else:
                print("\n⚠ No Wikipedia evidence found for this statement")

            print()

        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")
            continue
