# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    rendu.py                                           :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: aurele <aurele@student.42.fr>              +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2026/02/06 23:24:49 by aurele            #+#    #+#              #
#    Updated: 2026/02/17 05:15:00 by aurele           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import trafilatura 
import json
import os
import spacy
from bs4 import BeautifulSoup
import csv
import urllib.parse
from rdflib import Graph, URIRef, Literal, Namespace
from rdflib.namespace import RDF, RDFS
from fastcoref import spacy_component

urls_arcane = ["https://wiki.leagueoflegends.com/en-us/Universe:Arcane_(TV_Series)/Season_1/Episode_1",
                "https://wiki.leagueoflegends.com/en-us/Universe:Arcane_(TV_Series)/Season_1/Episode_2",
                "https://wiki.leagueoflegends.com/en-us/Universe:Arcane_(TV_Series)/Season_1/Episode_3",
                "https://wiki.leagueoflegends.com/en-us/Universe:Arcane_(TV_Series)/Season_1/Episode_4",
                "https://wiki.leagueoflegends.com/en-us/Universe:Arcane_(TV_Series)/Season_1/Episode_5",
                "https://wiki.leagueoflegends.com/en-us/Universe:Arcane_(TV_Series)/Season_1/Episode_6",
                "https://wiki.leagueoflegends.com/en-us/Universe:Arcane_(TV_Series)/Season_1/Episode_7",
                "https://wiki.leagueoflegends.com/en-us/Universe:Arcane_(TV_Series)/Season_1/Episode_8",
                "https://wiki.leagueoflegends.com/en-us/Universe:Arcane_(TV_Series)/Season_1/Episode_9",
]

# ================================
# NLP model loading
# ================================

nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("fastcoref")


# ================================
# URL loading / page extraction
# ================================

def save_to_json(data, dir, file="data.jsonl"):
    if not dir.exists():
        dir.mkdir(parents = True, exist_ok = True)
    with open(file, "a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")



def load_existing_urls(output_file):
    urls = set()

    if not os.path.exists(output_file):
        return urls

    with open(output_file, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            urls.add(data["url"])

    return urls

def extract_page_data(url, base_domain="https://wiki.leagueoflegends.com"):
    """Extracts text AND relevant internal wiki links from a page."""
    html = trafilatura.fetch_url(url)
    if html is None:
        return None

    soup = BeautifulSoup(html, "html.parser")
    content = soup.find("div", class_="mw-parser-output")
    
    sections = ["Plot", "Background", "Relations", "Personality", "Appearance", "Lore", "History", "Locations", "Government", "Properties"]
    # Try to find 'Plot' first, if not, grab the 'Biography' or 'Summary' for character pages
    main_sections = []
    for section in sections:
        if content.find("h2", id=section) is not None:
            main_sections.append(content.find("h2", id=section))
    if main_sections is None:
        return None

    plot_paragraphs = []
    relevant_links = set()
    
    # Grab paragraphs and links
    for main_section in main_sections:
        for e in main_section.find_all_next():
            if e.name == "h2": # Stop when the next major section starts
                break
            if e.name == "p":
                plot_paragraphs.append(e.get_text(" ", strip=True))
                
                # Look for links inside this relevant paragraph
                for a in e.find_all("a", href=True):
                    href = a['href']
                    # Filter: Only internal wiki links, ignore Meta pages (Category:, File:, Template:)
                    if (href.startswith("/wiki/") or href.startswith("/en-us/")) and ":" not in href.replace("Universe:", ""):
                        full_link = urllib.parse.urljoin(base_domain, href)
                        relevant_links.add(full_link)

    text = "\n".join(plot_paragraphs)
    if len(text.split()) < 50: # Skip empty/stub pages
        return None
    paragraphs = text.split('\n\n')
    resolved_paragraphs = ""

    for paragraph in paragraphs:
        doc = nlp(      # for multiple texts use nlp.pipe
            paragraph, 
            component_cfg={"fastcoref": {'resolve_text': True}}
        )
        resolved_paragraphs += " " + doc._.resolved_text
    
    text = resolved_paragraphs.strip()

    return {
        "url": url,
        "text": text,
        "word_count": len(text.split()),
        "links": list(relevant_links) # Return the safe links
    }

def extract_all_pages(start_urls, dir, max_pages=50, output_file="data.jsonl"):
    """Uses a queue to process start URLs AND the pages they link to."""
    saved = 0
    skipped = 0
    existing_urls = load_existing_urls(output_file)
    
    # Create a queue initialized with your episode URLs
    url_queue = list(start_urls)
    
    while url_queue and saved < max_pages:
        url = url_queue.pop(0)
        
        print(f"Processing: {url.split('/')[-1][:30]}...", end="")

        if url in existing_urls:
            print("  → Already processed")
            skipped += 1
            continue

        result = extract_page_data(url)

        if result is None:
            print("  → Skipped (No relevant text)")
            skipped += 1
            continue

        save_to_json(result, dir, output_file)
        existing_urls.add(url)
        
        # Add the newly discovered, relevant links to the back of the queue
        for new_url in result["links"]:
            if new_url not in existing_urls and new_url not in url_queue:
                url_queue.append(new_url)

        print(f"  → Saved ({result['word_count']} words, found {len(result['links'])} new links)")
        saved += 1

    print("\nDone.")
    print(f"Saved pages: {saved}")

# ================================
# Entity extraction. Bonjour hehe
# ================================

def extract_entities(text, allowed_labels={"PERSON", "ORG", "GPE", "LOC", "NORP", "PRODUCT", "EVENT"}):
    doc = nlp(text)
    entities = []
    
    for ent in doc.ents:
        if ent.label_ in allowed_labels:
            if len(ent.text.strip()) > 2:
                # Rename CARDINAL to AGE if it looks like an age
                label = ent.label_
                if label == "CARDINAL" and any(word in ent.text.lower() for word in ["years", "old"]):
                    label = "AGE"
                    
                entities.append({
                    "text": ent.text,
                    "label": label
                })
                
    # 2. Custom Extraction: Professions and Titles
    # Looks for compound nouns acting as titles (e.g., "Enforcer Vi")
    for token in doc:
        if token.ent_type_ == "PERSON" and token.dep_ in {"nsubj", "pobj", "dobj"}:
            # Check the words immediately preceding the character's name
            for child in token.children:
                if child.dep_ == "compound" and child.pos_ == "NOUN":
                    entities.append({
                        "text": child.text.capitalize(),
                        "label": "PROFESSION/TITLE"
                    })
                    
            # Check for appositions (e.g., "Jayce, an inventor")
            for child in token.children:
                if child.dep_ == "appos":
                    # Clean up determiners (remove 'a', 'an', 'the')
                    prof_text = " ".join([w.text for w in child.subtree if w.pos_ != "DET"])
                    if len(prof_text) > 2:
                        entities.append({
                            "text": prof_text.capitalize(),
                            "label": "PROFESSION/TITLE"
                        })

    # Remove duplicates
    entities = list({(e["text"], e["label"]) for e in entities})
    return entities

from collections import Counter

# ================================
# Relation extraction
# ================================
IGNORE_ENTITIES = {"his", "her", "their", "one", "it", "someone", "something", "this", "that", "he", "she"}
ARCANE = Namespace("http://example.org/arcane/")

def extract_relations(text):
    doc = nlp(text)
    relations = []

    # Helper to find if a token belongs to a Named Entity
    def get_ent(token):
        for ent in doc.ents:
            if token.i >= ent.start and token.i < ent.end:
                return ent.text.strip()
        return None

    for sent in doc.sents:
        # --- SECTION 1: ACTION & ATTRIBUTE RELATIONS (Verbs) ---
        for token in sent:
            # We look for verbs OR 'is/have'
            is_static_verb = token.lemma_ in {"be", "have"}
            if token.pos_ == "VERB" or is_static_verb:
                
                # Filter out generic high-frequency verbs unless they are 'be/have'
                #if token.lemma_ in BAD_VERBS and not is_static_verb:
                #    continue

                # A. Identify the Relation String
                if is_static_verb:
                    # For "is", look for the attribute (e.g., "is the SISTER")
                    attr = [child for child in token.children if child.dep_ == "attr"]
                    if attr:
                        full_relation = f"is_{attr[0].lemma_}"
                    else:
                        full_relation = token.lemma_
                else:
                    # For normal verbs, handle particles/prepositions (e.g., "team_up_with")
                    suffix = ""
                    for child in token.children:
                        if child.dep_ in {"prt", "prep"}:
                            suffix += f"_{child.lemma_}"
                    full_relation = f"{token.lemma_}{suffix}"

                # B. Find Subject and Object
                subj = None
                obj = None
                is_passive = False

                for child in token.children:
                    if child.dep_ in {"nsubj", "nsubjpass"}:
                        if child.dep_ == "nsubjpass": is_passive = True
                        subj = child
                    if child.dep_ in {"dobj", "attr", "acomp"}:
                        obj = child
                    if child.dep_ == "prep":
                        # This looks for "in", "to", "from", "at"
                        for sub_child in child.children:
                            if sub_child.dep_ == "pobj":
                                obj = sub_child

                # C. Passive Voice Correction ("Silco was killed by Jinx")
                if is_passive:
                    for child in token.children:
                        if child.dep_ == "agent": # "by"
                            for sub_child in child.children:
                                if sub_child.dep_ == "pobj":
                                    obj, subj = subj, sub_child

                # D. Entity Alignment & Saving
                if subj and obj:
                    head_ent = get_ent(subj)
                    tail_ent = get_ent(obj)
                    
                    if head_ent and tail_ent and head_ent != tail_ent:
                        if head_ent.lower() not in IGNORE_ENTITIES and tail_ent.lower() not in IGNORE_ENTITIES:
                            relations.append({
                                "head": head_ent,
                                "relation": full_relation,
                                "tail": tail_ent,
                                "sentence": sent.text
                            })
# " C'est trop de la balle, j'adore ce projet ! ", Aurele, 2026
        # --- SECTION 2: STATIC RELATIONS (Appositions & Possessives) ---
        for token in sent:
            head_ent = None
            tail_ent = None
            rel_type = None

            if token.dep_ == "appos": # "Jayce, the inventor"
                head_ent = get_ent(token.head)
                tail_ent = get_ent(token)
                rel_type = "is_a"
            elif token.dep_ == "poss": # "Vander's bar"
                head_ent = get_ent(token)
                tail_ent = get_ent(token.head)
                rel_type = "owns"

            if head_ent and tail_ent and rel_type:
                # Apply your circular/generic filters
                if head_ent.lower() == tail_ent.lower(): continue
                if head_ent.lower() in IGNORE_ENTITIES or tail_ent.lower() in IGNORE_ENTITIES: continue
                
                # The "Person is a Person" filter to avoid list errors
                if rel_type == "is_a":
                    # Check if both are people
                    h_label = next((e.label_ for e in doc.ents if e.text == head_ent), None)
                    t_label = next((e.label_ for e in doc.ents if e.text == tail_ent), None)
                    if h_label == "PERSON" and t_label == "PERSON":
                        continue

                relations.append({
                    "head": head_ent,
                    "relation": rel_type,
                    "tail": tail_ent, #we are charlie Kirk 
                    "sentence": sent.text 
                })

    # Deduplicate
    unique_rels = {(r["head"], r["relation"], r["tail"]): r for r in relations}.values()
    return list(unique_rels) 

def clean_noisy_entities(g, namespace = ARCANE):
    initial_count = len(g)
    # Define a threshold for "too long to be a name"
    MAX_NAME_LENGTH = 60
    
    # Identify URIs to remove
    to_remove = set()
    for s, p, o in g:
        for node in [s, o]:
            if isinstance(node, URIRef) and str(node).startswith(str(namespace)):
                name = str(node).replace(str(namespace), "")
                
                # Filter 1: Length
                if len(name) > MAX_NAME_LENGTH:
                    to_remove.add(node)
                
                # Filter 2: Malformed symbols
                if any(char in name for char in ["%", "[", "]", ",", "(", ")"]):
                    to_remove.add(node)
                
                # Filter 3: Digits only (e.g., arcane:516)
                if name.isdigit():
                    to_remove.add(node)

    # Delete every triplet connected to a noisy entity
    for node in to_remove:
        g.remove((node, None, None))
        g.remove((None, None, node))
        
    print(f"Cleanup complete. Removed {initial_count - len(g)} noisy triplets.")
    return g

# ================================
# JSON loading / aggregation
# ================================

def load_jsonl(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)

def aggregate_entities_from_jsonl(filepath, extract_entities_func):
    """
    Aggregates entities with a voting system and tracks source URLs.
    """
    entity_votes = {} # { "Name": Counter({ "TYPE": count }) }
    entity_urls = {}  # { "Name": set([url1, url2]) }
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            text = data.get("text", "")
            url = data.get("url", "unknown")
            
            raw_entities = extract_entities_func(text)
            
            for ent_text, ent_type in raw_entities:
                # Track votes for type resolution
                if ent_text not in entity_votes:
                    entity_votes[ent_text] = Counter()
                    entity_urls[ent_text] = set()
                
                entity_votes[ent_text][ent_type] += 1
                entity_urls[ent_text].add(url)

    # Final Resolution
    final_entities = {}
    for ent_text, votes in entity_votes.items():
        # Get the most common type
        most_common_type = votes.most_common(1)[0][0]
        
        final_entities[ent_text] = {
            "type": most_common_type,       # Matches the new build_rdf_graph logic
            "urls": entity_urls[ent_text],  # Kept for the CSV
            "count": sum(votes.values())
        }
        
    return final_entities

def aggregate_relations_from_jsonl(filepath, extract_relations_func):
    relations_store = {}

    for doc in load_jsonl(filepath):
        text = doc["text"]
        url = doc["url"]

        relations = extract_relations_func(text)

        for r in relations:
            key = (r["head"], r["relation"], r["tail"])

            if key not in relations_store:
                relations_store[key] = {
                    "urls": set(),
                    "sentences": set()
                }

            relations_store[key]["urls"].add(url)
            relations_store[key]["sentences"].add(r["sentence"])

    return relations_store

# ================================
# CSV export
# ================================

def save_entities_store_to_csv(entities_store, filename):
    """
    Saves the resolved entities store to a CSV file.
    """
    with open(filename, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        # We use 'type' (singular) in the header to match the new logic
        writer.writerow(["entity", "type", "source_urls"])
        
        for entity, data in entities_store.items():
            writer.writerow([
                entity,
                data["type"], # Resolved single type
                ";".join(sorted(data["urls"])) # All URLs where it appeared
            ])


# ================================
# RDF Graph Construction
# ================================

# private namespace

def clean_uri_string(s):
    """Cleans a string to be safely used as a URI component."""
    cleaned = s.replace(" ", "_").replace("'", "").replace('"', '')
    return urllib.parse.quote(cleaned)

def build_rdf_graph(entities_store, relations_store):
    """
    Converts resolved entities (single type per entity) and relations 
    into a clean RDF graph using the Arcane namespace.
    """
    g = Graph()
    g.bind("arcane", ARCANE) # Bind prefix for cleaner outputs

    for entity_str, data in entities_store.items():
        # Clean the string to create a valid URI fragment
        uri_fragment = clean_uri_string(entity_str)
        entity_uri = ARCANE[uri_fragment]
        
        # Add the human-readable label
        g.add((entity_uri, RDFS.label, Literal(entity_str)))
        
        ent_type = data["type"] 
        type_uri = ARCANE[clean_uri_string(ent_type)]
        g.add((type_uri, RDF.type, RDFS.Class))
        g.add((entity_uri, RDF.type, type_uri))
            
    for (head, relation, tail), data in relations_store.items():
        # Ensure the head and tail match the cleaned URIs used in Step 1
        head_uri = ARCANE[clean_uri_string(head)]
        tail_uri = ARCANE[clean_uri_string(tail)]
        
        # Normalize predicate naming (lowercase and remove spaces)
        normalized_relation = relation.lower().replace(" ", "_")
        rel_uri = ARCANE[clean_uri_string(normalized_relation)]

        g.add((rel_uri, RDF.type, RDF.Property))
        g.add((rel_uri, RDFS.label, Literal(relation.replace("_", " "))))
        g.add((head_uri, rel_uri, tail_uri))
        
    return g

def save_rdf_graph(g, filepath, format="turtle"):
    """Saves the RDF graph to a file (default format is Turtle)."""
    g.serialize(destination=str(filepath), format=format)