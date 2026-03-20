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
import spacy_transformers
from bs4 import BeautifulSoup
import csv
import urllib.parse
import pandas as pd
from rdflib import Graph, URIRef, Literal, Namespace
from rdflib.namespace import RDF, RDFS, OWL
from SPARQLWrapper import SPARQLWrapper, JSON

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
# URL loading / page extraction
# ================================

def extract_plot(url):
    html = trafilatura.fetch_url(url)
    if html is None:
        return None

    soup = BeautifulSoup(html, "html.parser")
    content = soup.find("div", class_="mw-parser-output")
    plot_h2 = content.find("h2", id="Plot") if content else None
    if plot_h2 is None:
        return None

    plot = []
    for e in plot_h2.find_all_next():
        if e.name == "h2":
            break
        if e.name == "p":
            plot.append(e.get_text(" ", strip=True))

    text = "\n".join(plot)
    return {
        "url": url,
        "text": text,
        "word_count": len(text.split())
    }



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



def extract_all_pages(urls, dir, min_words=500, output_file="data.jsonl"):
    saved = 0
    skipped = 0
    existing_urls = load_existing_urls(output_file)

    for i, url in enumerate(urls):
        print(f"Processing url: {i}", end="")

        if url in existing_urls:
            print("  → Already processed, skipping")
            skipped += 1
            continue

        result = extract_plot(url)

        if result is None:
            print("  → Skipped, not enough words")
            skipped += 1
            continue

        save_to_json(result, dir, output_file)
        existing_urls.add(url)

        print(f"  → Saved ({result['word_count']} words)")
        saved += 1

    print("\nDone.")
    print(f"Saved pages   : {saved}")
    print(f"Skipped pages : {skipped}")



# ================================
# NLP model loading
# ================================

nlp = spacy.load("en_core_web_trf")



# ================================
# Entity extraction
# ================================

def extract_entities(text, allowed_labels={"PERSON", "ORG", "GPE", "EVENT", "DATE"}):
    doc = nlp(text)
    entities = []
    for ent in doc.ents:
        if ent.label_ in allowed_labels:
            if len(ent.text.strip()) > 2:
                entities.append({
                    "text": ent.text,
                    "label": ent.label_
                })
    entities = list({(e["text"], e["label"]) for e in entities})
    return entities



# ================================
# Relation extraction
# ================================
BAD_VERBS = {
    "be", "have", "do", "say", "see",
    "make", "use", "include", "edit"
}

def extract_relations(text):
    doc = nlp(text)
    relations = []

    for sent in doc.sents:
        sent_ents = [ent for ent in sent.ents if ent.label_ in {"PERSON", "ORG", "GPE", "DATE"}]
        if len(sent_ents) < 2:
            continue

        for token in sent:
            if token.pos_ != "VERB":
                continue

            verb = token.lemma_
            if verb in BAD_VERBS:
                continue

            subj = None
            obj = None

            for child in token.children:
                if child.dep_ in {"nsubj", "nsubjpass"}:
                    subj = child
                if child.dep_ in {"dobj", "pobj", "attr"}:
                    obj = child

            if subj is None or obj is None:
                continue
            if subj.pos_ == "PRON" or obj.pos_ == "PRON":
                continue

            relations.append({
                "head": subj.text,
                "relation": verb,
                "tail": obj.text,
                "sentence": sent.text
            })

    relations = {
        (r["head"], r["relation"], r["tail"]): r
        for r in relations
    }.values()

    return list(relations)

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
# JSON loading / aggregation
# ================================

def load_jsonl(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)



def aggregate_entities_from_jsonl(filepath, extract_entities_func):
    entities_store = {}

    for doc in load_jsonl(filepath):
        text = doc["text"]
        url = doc["url"]

        entities = extract_entities_func(text)

        for name, label in entities:
            if name not in entities_store:
                entities_store[name] = {
                    "types": set(),
                    "urls": set()
                }
            if label not in entities_store[name]["types"] and len(entities_store[name]["types"]) > 0:
                print(
                    f"[WARNING] Entity '{name}' has multiple types: "
                    f"{entities_store[name]['types']} + {label}"
                )

            entities_store[name]["types"].add(label)
            entities_store[name]["urls"].add(url)

    return entities_store



# ================================
# CSV export
# ================================

def save_entities_store_to_csv(entities_store, filename="extracted_knowledge.csv"):
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["entity", "types", "source_urls"])

        for entity, data in entities_store.items():
            writer.writerow([
                entity,
                ";".join(sorted(data["types"])),
                ";".join(sorted(data["urls"]))
            ])


# ================================
# RDF Graph Construction
# ================================

# private namespace
ARCANE = Namespace("http://example.org/arcane/")

def clean_uri_string(s):
    """Cleans a string to be safely used as a URI component."""
    cleaned = s.replace(" ", "_").replace("'", "").replace('"', '')
    return urllib.parse.quote(cleaned)

def build_rdf_graph(entities_store, relations_store):
    """
    Converts extracted entities and relations into an RDF graph using URIs.
    """
    g = Graph()
    g.bind("arcane", ARCANE) # Bind prefix for cleaner outputs
    
    # 1. Entity & type processing (Nodes)
    for entity_str, data in entities_store.items():
        entity_uri = ARCANE[clean_uri_string(entity_str)]
        g.add((entity_uri, RDFS.label, Literal(entity_str)))
        
        for ent_type in data["types"]:
            type_uri = ARCANE[clean_uri_string(ent_type)]
            # Declare the type as a Class
            g.add((type_uri, RDF.type, RDFS.Class))
            # Assign the entity to this class
            g.add((entity_uri, RDF.type, type_uri))
            
    # Relation processing (Triplets)
    for (head, relation, tail), data in relations_store.items():
        head_uri = ARCANE[clean_uri_string(head)]
        tail_uri = ARCANE[clean_uri_string(tail)]
        
        # Normalize predicate naming (camelCase/lowercase)
        normalized_relation = relation.lower().replace(" ", "")
        rel_uri = ARCANE[clean_uri_string(normalized_relation)]
        
        # relation as an RDF Property
        g.add((rel_uri, RDF.type, RDF.Property))
        
        # actual data triplet
        g.add((head_uri, rel_uri, tail_uri))
        
    return g

def save_rdf_graph(g, filepath, format="turtle"):
    """Saves the RDF graph to a file (default format is Turtle)."""
    g.serialize(destination=str(filepath), format=format)