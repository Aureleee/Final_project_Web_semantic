import requests
import rdflib

def get_schema_summary(g):
    """
    Shows the LLM exactly which Predicates exist and gives 
    examples of Entity names so it knows the casing (e.g., Vander vs vander).
    """
    predicates = sorted(list(set(str(p).split('/')[-1] for p in g.predicates() if "arcane" in str(p))))
    
    subjects = sorted(list(set(str(s).split('/')[-1] for s in g.subjects() if "arcane" in str(s))))[:15]
    
    summary = "RDF SCHEMA CONTEXT:\n"
    summary += f"- Available Predicates: {', '.join(predicates)}\n"
    summary += f"- Example Entities: {', '.join(subjects)}\n"
    summary += "- Prefix: arcane: <http://example.org/arcane/>"
    return summary

def ask_local_llm(prompt, model="deepseek-coder-v2:16b"):
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }
    try:
        response = requests.post(url, json=payload, timeout=60)
        return response.json().get("response", "")
    except Exception as e:
        return f"Error connecting to Ollama: {e}"

def generate_sparql(question, schema_summary):
    """
    Expert prompt: Forces the LLM to stick to the schema and 
    handles markdown cleaning more robustly.
    """
    prompt = f"""
    You are a SPARQL expert for an Arcane (TV Series) Knowledge Graph.
    
    {schema_summary}

    TASK:
    Write a SPARQL query to answer: "{question}"

    CONSTRAINTS:
    1. Use ONLY the predicates listed in the schema context.
    2. Use the 'arcane:' prefix for all entities and predicates.
    3. Use 'SELECT ?ans WHERE {{ ... }}' format unless otherwise specified.
    4. Return ONLY the SPARQL code. No explanation. No backticks.
    """
    
    raw_response = ask_local_llm(prompt).strip()
    
    # Clean output: remove markdown code blocks and the word 'sparql'
    clean_query = raw_response.replace("```sparql", "").replace("```", "").replace("sparql", "").strip()
    return clean_query

def run_query_with_repair(g, question, schema_summary, max_attempts=3):
    current_query = generate_sparql(question, schema_summary)
    
    for attempt in range(max_attempts):
        try:
            print(f"--- Attempt {attempt+1} ---")
            results = g.query(current_query)
            return results, current_query
        except Exception as e:
            print(f"Query failed (Syntax or Logic Error): {e}")
            repair_prompt = f"""
            The SPARQL query below failed or returned an error. 
            Error: {e}
            Query: {current_query}
            Schema: {schema_summary}
            
            Fix the query. Ensure predicates match the schema. Return ONLY the corrected SPARQL.
            """
            current_query = ask_local_llm(repair_prompt).strip().replace("```sparql", "").replace("```", "").replace("sparql", "").strip()
            
    return None, current_query