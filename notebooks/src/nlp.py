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
from rdflib import Namespace
from fastcoref import spacy_component
from spacy.tokens import Doc

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
# NLP model 
# ================================


class ArcaneNLP:
    def __init__(self, use_gpu=True):
        print("Initializing ArcaneNLP (Safe Merging Mode)...")
        # Load the base model
        try:
            self.nlp = spacy.load("en_core_web_trf")
        except OSError:
            os.system("python -m spacy download en_core_web_trf")
            self.nlp = spacy.load("en_core_web_trf")

        device = 'cuda' if use_gpu else 'cpu'
        
        self.nlp.add_pipe(
            "fastcoref",
            config={
                'model_architecture': 'LingMessCoref',
                'model_path': 'biu-nlp/lingmess-coref',
                'device': device
            }
        )

    def process_text(self, text):
        """
        Processes text in chunks and merges the resulting Doc objects 
        to bypass the Transformer's 4096-token limit.
        """
        words = text.split()
        chunk_size = 500 
        chunks = [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
        
        processed_chunks = []
        
        for chunk in chunks:
            try:
                doc = self.nlp(chunk, component_cfg={"fastcoref": {'resolve_text': True}})

                resolved_chunk_text = doc._.resolved_text if doc._.resolved_text else chunk
                doc_chunk = self.nlp(resolved_chunk_text, disable=["fastcoref"])
                processed_chunks.append(doc_chunk)
                
            except Exception as e:
                print(f"Chunk processing failed: {e}")
                processed_chunks.append(self.nlp.make_doc(chunk))

        if processed_chunks:
            full_doc = Doc.from_docs(processed_chunks)
            return full_doc
        
        return self.nlp.make_doc(text)

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

def extract_page_data(url, arcane_engine, base_domain="https://wiki.leagueoflegends.com"):
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
            
    if not main_sections: # Cleaner way to check if list is empty
        return None

    plot_paragraphs = []
    relevant_links = set()

    for main_section in main_sections:
        for e in main_section.find_all_next():
            if e.name == "h2": 
                break
            if e.name == "p":
                plot_paragraphs.append(e.get_text(" ", strip=True))
                
                for a in e.find_all("a", href=True):
                    href = a['href']
                    # Filter: Only internal wiki links, ignore Meta pages
                    if (href.startswith("/wiki/") or href.startswith("/en-us/")) and ":" not in href.replace("Universe:", ""):
                        full_link = urllib.parse.urljoin(base_domain, href)
                        relevant_links.add(full_link)

    text = "\n".join(plot_paragraphs)
    if len(text.split()) < 50: 
        return None
    doc = arcane_engine.process_text(text)
    entities = extract_entities(doc)
    relations = extract_relations(doc)
    
    return ({
        "url": url,
        "text": doc.text, # IMPORTANT: Save the resolved text, not the raw text!
        "word_count": len(doc.text.split()),
        "links": list(relevant_links) 
    },
        entities, 
        relations
    )
from collections import Counter

def extract_all_pages(start_urls, dir, arcane_engine, max_pages=50, output_file="data.jsonl"):
    """Uses a queue to process start URLs AND aggregates NLP data globally."""
    saved = 0
    skipped = 0
    existing_urls = load_existing_urls(output_file)
    url_queue = list(start_urls)
    
    # 1. GLOBAL ACCUMULATORS (Declared OUTSIDE the loop)
    global_entity_votes = {}
    global_entity_urls = {}
    global_relations = []
    
    while url_queue and saved < max_pages:
        url = url_queue.pop(0)
        print(f"Processing: {url.split('/')[-1][:30]}...", end="")

        if url in existing_urls:
            print("  → Already processed")
            skipped += 1
            continue

        result = extract_page_data(url, arcane_engine)

        if result is None:
            print(f"  → Skipped {url} (No relevant text or fetch error)")
            skipped += 1
            continue

        # Now that we know it's NOT None, we can safely unpack it
        page_data, raw_entities, page_relations = result
                
        if page_data is None:
            print("  → Skipped (No relevant text)")
            skipped += 1
            continue
        global_relations.extend(page_relations)

        for ent_text, ent_type in raw_entities:
            if ent_text not in global_entity_votes:
                global_entity_votes[ent_text] = Counter()
                global_entity_urls[ent_text] = set()
            
            global_entity_votes[ent_text][ent_type] += 1
            global_entity_urls[ent_text].add(url)

        save_to_json(page_data, dir, output_file)
        existing_urls.add(url)
        
        for new_url in page_data["links"]:
            if new_url not in existing_urls and new_url not in url_queue:
                url_queue.append(new_url)

        print(f"  → Saved ({page_data['word_count']} words, found {len(page_data['links'])} new links)")
        saved += 1

    print("\nDone.")
    print(f"Saved pages: {saved}")

    print("Resolving final entity types...")
    final_entities = label_vote(global_entity_votes, global_entity_urls)
    
    unique_relations = {(r["head"], r["relation"], r["tail"]): r for r in global_relations}.values()
    return final_entities, list(unique_relations)

def label_vote(entity_votes, entity_urls):
    """
    Resolves the final entity types based on the highest vote count.
    """
    final_entities = {}
    for ent_text, votes in entity_votes.items():
        # Get the most common type (the winner of the vote)
        most_common_type = votes.most_common(1)[0][0]
        
        final_entities[ent_text] = {
            "type": most_common_type,       # Matches the new build_rdf_graph logic
            "urls": entity_urls[ent_text],  # Kept for the CSV
            "count": sum(votes.values())    # Total occurrences across all texts
        }
        
    return final_entities

# ================================
# Entity extraction. Bonjour hehe
# ================================

def extract_entities(doc, allowed_labels={"PERSON", "ORG", "GPE", "LOC", "NORP", "PRODUCT", "EVENT"}):
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
                
    for token in doc:
        if token.ent_type_ == "PERSON" and token.dep_ in {"nsubj", "pobj", "dobj"}:
            # Check the words immediately preceding the character's name
            for child in token.children:
                if child.dep_ == "compound" and child.pos_ == "NOUN":
                    entities.append({
                        "text": child.text.capitalize(),
                        "label": "PROFESSION/TITLE"
                    })

            for child in token.children:
                if child.dep_ == "appos":
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

def extract_relations(doc):
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