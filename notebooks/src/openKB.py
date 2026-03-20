from rdflib import Graph, URIRef, Literal, Namespace
from rdflib.namespace import RDF, RDFS, OWL
import urllib 
import trafilatura
import json
from src.rendu import clean_uri_string
import pandas as pd 
import requests
import time
from rdflib.namespace import OWL

ARCANE = Namespace("http://example.org/arcane/")

# ================================
# Open KB Expansion & Disambiguation
# ================================

def get_dbpedia_uri_spotlight(entity_text, context="Arcane League of Legends"):
    """
    Strategy A: Contextual Disambiguation.
    Passes the entity alongside contextual keywords to the Spotlight API.
    """
    url = "https://api.dbpedia-spotlight.org/en/annotate"
    # We force context into the text so Spotlight knows we mean the game/show
    query_text = f"{entity_text} in {context}."
    params = {"text": query_text, "confidence": 0.5}
    headers = {"Accept": "application/json"}
    
    try:
        response = requests.get(url, params=params, headers=headers)
        if response.status_code == 200:
            data = response.json()
            if "Resources" in data:
                # Return the URI of the top match
                return data["Resources"][0]["@URI"]
    except Exception as e:
        print(f"Error querying Spotlight for {entity_text}: {e}")
    return None

def get_dbpedia_subgraph(dbpedia_uri, limit=30):
    """
    Strategy B: Subgraph Extraction.
    Pulls facts for the URI, strictly filtering for official dbo: properties.
    """
    url = "https://dbpedia.org/sparql"
    query = f"""
        SELECT ?predicate ?object WHERE {{
          <{dbpedia_uri}> ?predicate ?object .
          FILTER(STRSTARTS(STR(?predicate), "http://dbpedia.org/ontology/"))
        }} LIMIT {limit}
    """
    params = {"query": query, "format": "json"}
    
    try:
        response = requests.get(url, params=params)
        if response.status_code == 200:
            bindings = response.json()["results"]["bindings"]
            results = []
            for b in bindings:
                p = b["predicate"]["value"]
                o = b["object"]["value"]
                o_type = b["object"]["type"]
                results.append((p, o, o_type))
            return results
    except Exception as e:
        print(f"Error querying SPARQL for {dbpedia_uri}: {e}")
    return []

def expand_knowledge_graph(g, entities_store, max_entities=10):
    """
    Iterates through your private entities, disambiguates them, 
    and securely merges the DBpedia subgraphs using owl:sameAs.
    """
    count = 0
    for entity_str, data in entities_store.items():
        if count >= max_entities:
            break
            
        print(f"Disambiguating: '{entity_str}'...")
        dbpedia_uri = get_dbpedia_uri_spotlight(entity_str)
        
        if dbpedia_uri:
            print(f"  -> Matched DBpedia URI: {dbpedia_uri}")
            private_uri = ARCANE[clean_uri_string(entity_str)]
            public_uri_ref = URIRef(dbpedia_uri)
            
            # RULE: Linking, not duplicating!
            g.add((private_uri, OWL.sameAs, public_uri_ref))
            
            # Subgraph Extraction
            subgraph = get_dbpedia_subgraph(dbpedia_uri)
            for p, o, o_type in subgraph:
                p_ref = URIRef(p)
                # Ensure objects are correctly typed as URIs or Literals (strings/numbers)
                if o_type == "uri":
                    o_ref = URIRef(o)
                else:
                    o_ref = Literal(o)
                    
                # Attach the public knowledge to the PUBLIC URI node
                g.add((public_uri_ref, p_ref, o_ref))
                
            print(f"  -> Anchored {len(subgraph)} clean facts to the graph.")
            count += 1
            
            # Pause briefly to respect DBpedia API rate limits
            time.sleep(1) 
            
    return g