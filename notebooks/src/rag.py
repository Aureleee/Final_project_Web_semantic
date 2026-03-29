import requests
import rdflib

def get_schema_summary(g):
    """
    Summarizes the classes and predicates in the graph so the LLM 
    knows which 'words' it is allowed to use in its SPARQL query.
    """
    classes = sorted(list(set(str(o).split('/')[-1] for s, p, o in g.triples((None, rdflib.RDF.type, None)))))
    predicates = sorted(list(set(str(p).split('/')[-1] for s, p, o in g)))
    
    summary = f"Classes: {', '.join(classes)}\n"
    summary += f"Predicates: {', '.join(predicates)}"
    return summary

def ask_local_llm(prompt, model="gemma2:2b"):
    """
    Sends a prompt to your local Ollama server.
    """
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }
    try:
        response = requests.post(url, json=payload)
        return response.json().get("response", "")
    except Exception as e:
        return f"Error connecting to Ollama: {e}"

def generate_sparql(question, schema_summary):
    """
    Asks the LLM to write a SPARQL query based on the user's question.
    """
    prompt = f"""
    You are a SPARQL expert. Use the following RDF schema to answer the user's question.
    Schema:
    {schema_summary}
    
    Prefix: arcane: <http://example.org/arcane/>
    
    User Question: {question}
    
    Return ONLY the SPARQL query. Do not include markdown formatting or explanations.
    """
    return ask_local_llm(prompt).strip().replace("```sparql", "").replace("```", "")

def run_query_with_repair(g, question, schema_summary, max_attempts=3):
    """
    Tries to execute the SPARQL query. If it fails (syntax error), 
    it asks the LLM to fix it.
    """
    current_query = generate_sparql(question, schema_summary)
    
    for attempt in range(max_attempts):
        try:
            print(f"Attempt {attempt+1}: Executing SPARQL...")
            results = g.query(current_query)
            return results, current_query
        except Exception as e:
            print(f"Query failed. Asking LLM to repair...")
            repair_prompt = f"""
            The following SPARQL query failed with error: {e}
            Query: {current_query}
            Fix the query and return ONLY the corrected SPARQL.
            """
            current_query = ask_local_llm(repair_prompt).strip().replace("```sparql", "").replace("```", "")
            
    return None, current_query