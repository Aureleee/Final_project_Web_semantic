from src.nlp import ARCANE
import urllib
from rdflib import Graph, Literal, RDF, RDFS, URIRef

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
            
    for relation in relations_store:
        head = relation["head"]
        tail = relation["tail"]
        relation = relation["relation"]
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


def clean_noisy_entities(g, namespace = ARCANE):
    initial_count = len(g)
    # Define a threshold for "too long to be a name"
    MAX_NAME_LENGTH = 40
    
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
