#!/usr/bin/env python3
"""
Define comprehensive single-token contexts for expanded analysis.
"""

# Comprehensive single-token contexts organized by category
COMPREHENSIVE_CONTEXTS = {
    # Function Words
    "articles": ["the", "a", "an"],
    
    "personal_pronouns": ["I", "you", "he", "she", "it", "we", "they"],
    
    "possessive_pronouns": ["my", "your", "his", "her", "its", "our", "their"],
    
    "prepositions": ["in", "on", "at", "by", "for", "with", "from", "to", "of", "about", "through", "over", "under"],
    
    "conjunctions": ["and", "but", "or", "nor", "yet", "so", "because", "although", "while"],
    
    "auxiliaries": ["is", "are", "was", "were", "have", "has", "had", "will", "would", "can", "could", "should", "might"],
    
    # Content Words
    "common_nouns": ["time", "person", "year", "way", "day", "man", "thing", "world", "life", "hand"],
    
    "common_verbs": ["said", "make", "go", "take", "come", "see", "know", "get", "give", "find"],
    
    "common_adjectives": ["good", "new", "first", "last", "long", "great", "little", "own", "other", "old"],
    
    "common_adverbs": ["very", "just", "now", "then", "also", "well", "only", "even", "still", "too"],
    
    # Special Categories
    "negation": ["not", "no", "never", "neither"],
    
    "question_words": ["what", "when", "where", "who", "why", "how", "which"],
    
    "punctuation": [".", ",", "?", "!", ":", ";"],
    
    "numbers": ["one", "two", "three", "ten", "hundred", "thousand"],
    
    # Position markers
    "position": ["sentence_start"]  # Special marker for beginning of sentence
}

def get_all_contexts():
    """Get flat list of all contexts."""
    all_contexts = ["baseline"]  # Always include baseline (no context)
    for category, contexts in COMPREHENSIVE_CONTEXTS.items():
        all_contexts.extend(contexts)
    return all_contexts

def get_context_categories():
    """Get contexts organized by category."""
    categories = {"baseline": ["baseline"]}
    categories.update(COMPREHENSIVE_CONTEXTS)
    return categories

def get_context_info(context):
    """Get category and type information for a context."""
    if context == "baseline":
        return {"category": "baseline", "type": "none"}
    
    for category, contexts in COMPREHENSIVE_CONTEXTS.items():
        if context in contexts:
            # Determine broader type
            if category in ["articles", "personal_pronouns", "possessive_pronouns", 
                          "prepositions", "conjunctions", "auxiliaries", "negation"]:
                broad_type = "function"
            elif category in ["common_nouns", "common_verbs", "common_adjectives", "common_adverbs"]:
                broad_type = "content"
            elif category in ["question_words"]:
                broad_type = "interrogative"
            elif category in ["punctuation", "numbers"]:
                broad_type = "special"
            elif category in ["position"]:
                broad_type = "position"
            else:
                broad_type = "other"
                
            return {
                "category": category,
                "type": broad_type,
                "is_function_word": broad_type == "function"
            }
    
    return {"category": "unknown", "type": "unknown"}

def get_analysis_contexts():
    """Get contexts suitable for the paper analysis (subset of most interesting)."""
    # Select representative contexts from each category
    selected = {
        "baseline": ["baseline"],
        "articles": ["the", "a"],
        "personal_pronouns": ["he", "she", "it"],
        "prepositions": ["in", "on", "with"],
        "conjunctions": ["and", "but"],
        "auxiliaries": ["is", "was", "will", "can"],
        "common_nouns": ["time", "person"],
        "common_verbs": ["said", "make"],
        "common_adjectives": ["good", "new"],
        "negation": ["not"],
        "punctuation": ["."],
        "position": ["sentence_start"]
    }
    
    contexts = []
    for category, ctx_list in selected.items():
        contexts.extend(ctx_list)
    return contexts

if __name__ == "__main__":
    # Print summary
    all_contexts = get_all_contexts()
    print(f"Total contexts defined: {len(all_contexts)}")
    print(f"Baseline + {len(all_contexts)-1} single-token contexts")
    
    print("\nContexts by category:")
    for category, contexts in get_context_categories().items():
        print(f"  {category}: {len(contexts)} contexts")
    
    print(f"\nSelected for paper analysis: {len(get_analysis_contexts())} contexts")