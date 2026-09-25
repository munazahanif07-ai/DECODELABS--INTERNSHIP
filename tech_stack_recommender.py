"""
==========================================================================
 PROJECT 3 CAPSTONE - TECH STACK RECOMMENDER
 DecodeLabs | Artificial Intelligence Industrial Training Kit (2026)
==========================================================================

GOAL
----
Build a content-based recommendation engine that maps a user's raw skills
to the most relevant job roles (career paths), using pure similarity
logic (TF-IDF weighting + Cosine Similarity) - no external ML libraries.

This script follows the exact 4-step IPO pipeline taught in the slides:

    INPUT (Ingestion)  ->  PROCESS (Vectorization + Scoring)
                        ->  Sorting  ->  Filtering (OUTPUT: Top-N list)

Why content-based filtering (not collaborative filtering)?
    Content-based filtering maps user input directly to item (job role)
    attributes, so it works immediately without needing thousands of
    historical user interactions. It is also naturally robust to the
    "Item Cold Start" problem - a brand-new job role with tagged skills
    can be recommended right away.

Why Cosine Similarity (not Euclidean Distance)?
    Euclidean distance is sensitive to vector magnitude (how many skills
    a role lists), which distorts comparisons between a short user
    profile and a long job-skill list. Cosine similarity instead measures
    the ANGLE between vectors, so it captures the orientation/alignment
    of preferences regardless of length - the industry standard for
    text/tag-based recommendation.

Why TF-IDF (not simple binary 1/0 overlap)?
    A simple Jaccard/binary overlap treats a generic skill (e.g. "Python",
    which appears in many roles) exactly the same as a rare, highly
    specific skill (e.g. "Kubernetes"). TF-IDF penalizes common terms
    (low Inverse Document Frequency) and rewards rare, descriptive terms
    (high IDF), producing far more meaningful similarity scores.
==========================================================================
"""

import csv
import math
import os
import sys
from collections import Counter


# ------------------------------------------------------------------
# STEP 1: INGESTION  -  Load the item dataset (job roles + skills)
# ------------------------------------------------------------------
def load_dataset(csv_path):
    """
    Reads raw_skills.csv and returns a list of dicts:
        [{"role": "Data Scientist", "skills": ["Python", "SQL", ...]}, ...]

    Each job role is treated as a "document" whose "terms" are its
    tagged skills - mirroring how items are represented in real-world
    recommendation engines (e.g. a movie's genre tags, a product's
    feature tags).
    """
    items = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            role = row["job_role"].strip()
            # Normalize: split on commas, strip whitespace, keep original
            # casing for display but we'll lowercase for matching.
            skills = [s.strip() for s in row["skills"].split(",") if s.strip()]
            items.append({"role": role, "skills": skills})
    return items


# ------------------------------------------------------------------
# STEP 2a: VECTOR MAPPING  -  Build a shared vocabulary
# ------------------------------------------------------------------
def build_vocabulary(items):
    """
    Every skill across every job role becomes one dimension in our
    shared vector space. User skills and item skills MUST map onto
    this exact same vocabulary, or the similarity math breaks (as the
    slides warn: "Web Design" vs "Frontend Development" mismatch).

    We lowercase everything for case-insensitive matching.
    """
    vocab = set()
    for item in items:
        for skill in item["skills"]:
            vocab.add(skill.lower())
    return sorted(vocab)  # sorted for stable, reproducible ordering


# ------------------------------------------------------------------
# STEP 2b: TF-IDF WEIGHTING
# ------------------------------------------------------------------
def compute_idf(items, vocab):
    """
    IDF = log( Total Documents / Documents containing term t )

    This is computed ONCE across the whole item corpus (all job roles).
    Common skills (appearing in many roles) get a LOW idf -> low weight.
    Rare, specific skills get a HIGH idf -> high weight.
    """
    n_docs = len(items)
    doc_freq = Counter()
    for item in items:
        unique_skills_in_doc = set(s.lower() for s in item["skills"])
        for skill in unique_skills_in_doc:
            doc_freq[skill] += 1

    idf = {}
    for term in vocab:
        df = doc_freq.get(term, 0)
        # +1 smoothing guards against division-by-zero for unseen terms
        # (e.g. a skill the user types that no job role has yet).
        idf[term] = math.log((n_docs + 1) / (df + 1)) + 1
    return idf


def compute_tf(term_list, vocab):
    """
    TF = (count of term t in document) / (total terms in document)

    term_list: lowercase list of skills belonging to ONE document
               (either a job role's skill list, or the user's input).
    """
    total_terms = len(term_list)
    counts = Counter(term_list)
    tf = {}
    for term in vocab:
        tf[term] = counts.get(term, 0) / total_terms if total_terms > 0 else 0.0
    return tf


def tfidf_vector(term_list, vocab, idf):
    """
    Combines TF and IDF into the final weighted vector:
        weight(term) = TF(term) * IDF(term)

    Returns a plain Python list of floats, ordered identically to `vocab`,
    so every vector (job roles AND the user) lives in the same space.
    """
    tf = compute_tf([t.lower() for t in term_list], vocab)
    return [tf[term] * idf[term] for term in vocab]


# ------------------------------------------------------------------
# STEP 3: THE SIMILARITY ENGINE  -  Cosine Similarity
# ------------------------------------------------------------------
def cosine_similarity(vec_a, vec_b):
    """
    cos(theta) = (A . B) / (||A|| * ||B||)

    Measures the ANGLE between two vectors, ignoring their magnitude.
    Score interpretation (TF-IDF values are non-negative, so range is
    0 to 1 here):
        1.0 -> vectors point in the same direction (perfect match)
        0.0 -> vectors share no common weighted skills (orthogonal)
    """
    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    magnitude_a = math.sqrt(sum(a * a for a in vec_a))
    magnitude_b = math.sqrt(sum(b * b for b in vec_b))

    if magnitude_a == 0 or magnitude_b == 0:
        # Cold-start guard: a zero vector (no overlapping/known skills)
        # cannot be divided against - return 0 similarity instead of
        # crashing with a ZeroDivisionError.
        return 0.0

    return dot_product / (magnitude_a * magnitude_b)


# ------------------------------------------------------------------
# STEP 4: THE 4-STEP RANKING PIPELINE
#          (Ingestion -> Scoring -> Sorting -> Filtering)
# ------------------------------------------------------------------
def recommend_tech_stack(user_skills, items, vocab, idf, top_n=3):
    """
    Runs the full IPO pipeline for one user query and returns the
    Top-N ranked job role recommendations.

    Step 1 (Ingestion) was already done via load_dataset()/build_vocabulary().
    Here we do:
        Step 2 (Scoring)   -> score every item against the user vector
        Step 3 (Sorting)   -> descending order by score
        Step 4 (Filtering) -> truncate to top_n, prevent choice overload
    """
    # Project 3 requirement: minimum of three user inputs for sufficient
    # data density.
    if len(user_skills) < 3:
        raise ValueError(
            "Please provide at least 3 skills to ensure accurate matching."
        )

    user_vector = tfidf_vector(user_skills, vocab, idf)

    scored = []
    for item in items:
        item_vector = tfidf_vector(item["skills"], vocab, idf)
        score = cosine_similarity(user_vector, item_vector)
        scored.append((item["role"], score, item["skills"]))

    # Step 3: Sorting (descending by cosine similarity score)
    scored.sort(key=lambda x: x[1], reverse=True)

    # Step 4: Filtering (Top-N cutoff to prevent information overload)
    return scored[:top_n]


# ------------------------------------------------------------------
# BOOTSTRAP / COLD-START FALLBACK
# ------------------------------------------------------------------
def trending_fallback(items, top_n=3):
    """
    Handles the "User Cold Start" problem: if a user provides skills
    that don't overlap with ANY known vocabulary term (all-zero vector),
    cosine similarity can't rank anything meaningfully. As a bypass
    strategy (per the slides: Trending Fallbacks), we fall back to
    returning the first N roles as a generic "popular" default.
    """
    return [(item["role"], 0.0, item["skills"]) for item in items[:top_n]]


# ------------------------------------------------------------------
# DISPLAY HELPERS
# ------------------------------------------------------------------
def print_recommendations(user_skills, results):
    print("\n" + "=" * 60)
    print("  USER INPUT SKILLS :", ", ".join(user_skills))
    print("=" * 60)

    if all(score == 0 for _, score, _ in results):
        print("No strong skill overlap found - showing trending roles:\n")
    else:
        print("Top {} Recommended Career Paths:\n".format(len(results)))

    for rank, (role, score, skills) in enumerate(results, start=1):
        match_pct = round(score * 100, 1)
        print(f"  #{rank}  {role}")
        print(f"       Match Score : {match_pct}%  (cosine similarity = {round(score, 4)})")
        print(f"       Core Skills : {', '.join(skills)}")
        print()


# ------------------------------------------------------------------
# MAIN - Interactive entry point
# ------------------------------------------------------------------
def main():
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw_skills.csv")
    items = load_dataset(csv_path)
    vocab = build_vocabulary(items)
    idf = compute_idf(items, vocab)

    print("=" * 60)
    print("  DECODELABS - TECH STACK RECOMMENDER (Project 3)")
    print("  Content-Based Filtering | TF-IDF + Cosine Similarity")
    print("=" * 60)
    print(f"Loaded {len(items)} job roles across {len(vocab)} unique skills.\n")

    # Take user input: minimum 3 skills, comma-separated.
    if len(sys.argv) > 1:
        # Allow non-interactive use: python tech_stack_recommender.py "Python,Cloud Computing,Automation"
        raw_input_str = sys.argv[1]
    else:
        raw_input_str = input(
            "Enter at least 3 of your skills, separated by commas\n"
            "(e.g. Python, Cloud Computing, Automation): "
        )

    user_skills = [s.strip() for s in raw_input_str.split(",") if s.strip()]

    try:
        results = recommend_tech_stack(user_skills, items, vocab, idf, top_n=3)
    except ValueError as e:
        print(f"\nError: {e}")
        return

    # Cold-start bypass: if nothing matched at all, fall back to trending.
    if all(score == 0 for _, score, _ in results):
        results = trending_fallback(items, top_n=3)

    print_recommendations(user_skills, results)


if __name__ == "__main__":
    main()