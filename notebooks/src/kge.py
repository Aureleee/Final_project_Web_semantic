# notebooks/src/kge_prep.py
import rdflib
from rdflib.namespace import RDF, RDFS, OWL
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
import torch
import torch
import pandas as pd
from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory

def prepare_kge_datasets(graph_path, output_dir):
    """
    Parses an RDF graph, extracts pure factual triplets, and splits them
    into train, validation, and test sets for KGE training.
    """
    g = rdflib.Graph()
    print(f"Loading graph from {graph_path}...")
    g.parse(graph_path, format="turtle")

    # We must filter out the "Schema" triplets. 
    # ML models learn from facts (Vi -> owns -> Gauntlets), 
    # NOT from definitions (owns -> equivalentProperty -> dbo:owner)
    ignore_predicates = {
        RDF.type, 
        RDFS.label, 
        RDFS.comment, 
        RDFS.isDefinedBy,
        OWL.equivalentClass, 
        OWL.equivalentProperty, 
        RDFS.seeAlso
    }

    triplets = []
    
    for s, p, o in g:
        # Only keep facts that don't use the meta-predicates above
        if p not in ignore_predicates:
            # Clean the URIs to just keep the names for the TSV
            head = str(s).split("/")[-1]
            rel = str(p).split("/")[-1]
            tail = str(o).split("/")[-1]
            
            triplets.append([head, rel, tail])

    # Convert to a Pandas DataFrame
    df = pd.DataFrame(triplets, columns=["head", "relation", "tail"])
    
    # Drop duplicates just in case the extraction created any
    df = df.drop_duplicates()
    total_triplets = len(df)
    
    print(f"Extracted {total_triplets} pure fact triplets.")

    # --- SPLITTING THE DATA (80% Train / 10% Valid / 10% Test) ---
    # 1. Split off 20% for Temp (Valid + Test)
    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # 2. Split the Temp in half (10% Valid, 10% Test)
    valid_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    # --- SAVING TO TSV ---
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # KGE libraries require no headers and tab separation
    train_df.to_csv(out_path / "train.tsv", sep='\t', index=False, header=False)
    valid_df.to_csv(out_path / "valid.tsv", sep='\t', index=False, header=False)
    test_df.to_csv(out_path / "test.tsv", sep='\t', index=False, header=False)

    print(f"Saved: Train ({len(train_df)}), Valid ({len(valid_df)}), Test ({len(test_df)})")
    
    return train_df, valid_df, test_df


def run_kge_experiment(train_path, valid_path, test_path, model_name, epochs=100, embedding_dim=100):
    print(f"\n🚀 Training {model_name}...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load Triples
    training_factory = TriplesFactory.from_path(train_path)
    validation_factory = TriplesFactory.from_path(
        valid_path, 
        entity_to_id=training_factory.entity_to_id, 
        relation_to_id=training_factory.relation_to_id
    )
    testing_factory = TriplesFactory.from_path(
        test_path, 
        entity_to_id=training_factory.entity_to_id, 
        relation_to_id=training_factory.relation_to_id
    )

    # Run the Pipeline
    result = pipeline(
        model=model_name,
        training=training_factory,
        testing=testing_factory,
        validation=validation_factory,
        model_kwargs=dict(embedding_dim=embedding_dim),
        training_kwargs=dict(num_epochs=epochs, use_tqdm_batch=False),
        evaluation_kwargs=dict(use_tqdm=False),
        device=device,
        random_seed=42,
    )
    
    # --- THE VERSION-PROOF EXTRACTION ---
    df = result.metric_results.to_df()
    
    # Dynamically find the "Type" column so it never throws a KeyError
    type_column = None
    for col in ['Type', 'Value Type', 'RankType']:
        if col in df.columns:
            type_column = col
            break

    def safe_get_metric(side, metric_name):
        """Safely extracts the metric regardless of PyKEEN version."""
        # Filter by Head/Tail/Both and Metric Name
        mask = (df['Side'] == side) & (df['Metric'] == metric_name)
        filtered_df = df[mask]
        
        if type_column is not None:
            # PyKEEN uses 'realistic', 'arithmetic', or 'exact' for the true rank
            for rank_type in ['realistic', 'arithmetic', 'exact']:
                type_mask = filtered_df[type_column] == rank_type
                if type_mask.any():
                    return filtered_df[type_mask]['Value'].values[0]
        
        # Fallback: just return the first matching value
        return filtered_df['Value'].values[0] if not filtered_df.empty else 0.0

    # Extract all the required metrics for the lab securely
    metrics = {
        'MRR': safe_get_metric('both', 'mean_reciprocal_rank'),
        'Hits@1': safe_get_metric('both', 'hits_at_1'),
        'Hits@3': safe_get_metric('both', 'hits_at_3'),
        'Hits@10': safe_get_metric('both', 'hits_at_10'),
        'Head_MRR': safe_get_metric('head', 'mean_reciprocal_rank'),
        'Tail_MRR': safe_get_metric('tail', 'mean_reciprocal_rank')
    }
    
    return result, metrics

def generate_comparison_report(metrics_dict):
    """Creates a detailed table for the lab report covering all required metrics."""
    rows = []
    for model_name, m in metrics_dict.items():
        rows.append({
            "Model": model_name,
            "MRR (Both)": round(m['MRR'], 4),
            "Hits@1": round(m['Hits@1'], 4),
            "Hits@3": round(m['Hits@3'], 4),
            "Hits@10": round(m['Hits@10'], 4),
            "MRR (Head Only)": round(m['Head_MRR'], 4),
            "MRR (Tail Only)": round(m['Tail_MRR'], 4)
        })
    return pd.DataFrame(rows)