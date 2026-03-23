import requests
from rdflib import URIRef, Literal
from rdflib.namespace import RDF, RDFS, OWL

# 1. The Exact DBpedia Ontology Mappings
CLASS_ALIGNMENT = {
    "PERSON": "http://dbpedia.org/ontology/Person",
    "GPE": "http://dbpedia.org/ontology/Settlement",
    "LOC": "http://dbpedia.org/ontology/Place",
    "ORG": "http://dbpedia.org/ontology/Organisation",
    "NORP": "http://dbpedia.org/ontology/EthnicGroup",
    "PRODUCT": "http://dbpedia.org/ontology/Device", 
    "EVENT": "http://dbpedia.org/ontology/Event",
    "PROFESSION/TITLE": "http://dbpedia.org/ontology/Occupation"
}

def fetch_remote_definition(uri):
    """
    Queries DBpedia with a fallback: tries to get rdfs:comment, 
    but falls back to rdfs:label if the comment is missing.
    """
    url = "https://dbpedia.org/sparql"
    
    # NEW QUERY: Uses COALESCE to grab the comment, or the label if the comment is missing
    query = f"""
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    SELECT ?desc WHERE {{
      OPTIONAL {{ <{uri}> rdfs:comment ?comment . FILTER(lang(?comment) = 'en') }}
      OPTIONAL {{ <{uri}> rdfs:label ?label . FILTER(lang(?label) = 'en') }}
      
      # Bind the first one that exists to '?desc'
      BIND(COALESCE(?comment, ?label) AS ?desc)
      
      # Ensure we don't return empty results
      FILTER(BOUND(?desc))
    }} LIMIT 1
    """
    
    headers = {
        'Accept': 'application/sparql-results+json',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'
    }
    
    try:
        response = requests.get(url, params={'query': query}, headers=headers, timeout=15)
        
        if response.status_code == 200:
            data = response.json()
            bindings = data.get("results", {}).get("bindings", [])
            if bindings:
                return bindings[0]["desc"]["value"]
            else:
                print(f"  -> No English comment OR label found in DBpedia for {uri}")
        else:
            print(f"  -> DBpedia API Error {response.status_code}: {response.text[:100]}")
            
    except requests.exceptions.JSONDecodeError:
        print("  -> Failed to parse JSON. DBpedia likely returned an HTML error page.")
    except Exception as e:
        print(f"  -> Connection error: {e}")
        
    return None

def enrich_class_definitions(g, arcane_ns):
    print("--- Starting OpenKB Ontology Alignment ---")
    
    # Find all classes declared in your graph
    classes = set(g.subjects(RDF.type, RDFS.Class))
    
    if not classes:
        print("Error: No classes found in the graph. Did you build it correctly?")
        return g
        
    for cls_uri in classes:
        # Extract just the name (e.g., 'PERSON') from your URI
        class_name = str(cls_uri).replace(str(arcane_ns), "").strip("/")
        
        if class_name in CLASS_ALIGNMENT:
            public_uri = CLASS_ALIGNMENT[class_name]
            print(f"\nAligning '{class_name}' to {public_uri}...")
            
            # Fetch the definition
            definition = fetch_remote_definition(public_uri)
            
            if definition:
                # Add the Semantic Web alignment triplets!
                g.add((cls_uri, RDFS.isDefinedBy, URIRef(public_uri)))
                g.add((cls_uri, OWL.equivalentClass, URIRef(public_uri)))
                g.add((cls_uri, RDFS.comment, Literal(definition, lang="en")))
                
                print(f"  -> SUCCESS! Added definition: \"{definition[:75]}...\"")
                
    return g