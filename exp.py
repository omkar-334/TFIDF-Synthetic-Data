import random


def vary_weights(tfidf_scores, variation_factor=0.2):
    varied_scores = []
    for term, weight in tfidf_scores:
        new_weight = weight * (1 + random.uniform(-variation_factor, variation_factor))
        varied_scores.append((term, new_weight))
    return varied_scores


def generate_toc(tfidf_scores, num_sections=5):
    toc = "Table of Contents:\n\n"
    sections = []
    for i in range(num_sections):
        start = i * (len(tfidf_scores) // num_sections)
        end = (i + 1) * (len(tfidf_scores) // num_sections)
        section_terms = [term for term, _ in tfidf_scores[start:end]]
        section_title = " ".join(section_terms[:3])  # Use first 3 terms as section title
        sections.append(f"{i+1}. {section_title.title()}")
        toc += f"{sections[-1]}\n"

    content_description = f"\nThis document will cover {num_sections} main topics in the HVAC domain: {', '.join(sections)}. Each section will explore key concepts and technologies related to its title, incorporating relevant terminology and industry-specific knowledge."

    return toc + content_description


# # Example usage
# varied_weights = vary_weights(top_ngrams_with_weights)
# varied_prompt = generate_prompt(varied_weights)
# varied_text = generate_text(varied_prompt, api_key)

# toc_with_description = generate_toc(top_ngrams_with_weights)
# toc_prompt = f"{toc_with_description}\n\nWrite the full document based on this table of contents and description."
# toc_based_text = generate_text(toc_prompt, api_key)

# print("Varied weights text:")
# print(varied_text[:500])
# print("\nTable of Contents based text:")
# print(toc_based_text[:500])
