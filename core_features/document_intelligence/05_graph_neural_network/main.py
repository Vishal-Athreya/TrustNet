import re
import os
import networkx as nx
from pypdf import PdfReader

# Force matplotlib to use 'Agg' backend so it runs safely on backend servers without UI errors
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def extract_entities_from_document(file_path: str) -> dict:
    """Reads a document and uses Regex to extract identifying entities."""
    ext = os.path.splitext(file_path)[1].lower()
    text = ""
    
    try:
        if ext == '.pdf':
            reader = PdfReader(file_path)
            for page in reader.pages:
                text += (page.extract_text() or "") + " "
        elif ext == '.txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
            
    pan_pattern = r'[A-Z]{5}[0-9]{4}[A-Z]{1}'
    name_pattern = r'Name:\s*([A-Z][a-z]+(?: [A-Z][a-z]+)+)'
    
    entities = {"doc_type": os.path.basename(file_path)}
    
    pan_match = re.search(pan_pattern, text)
    if pan_match: entities["pan_number"] = pan_match.group(0)
        
    name_match = re.search(name_pattern, text)
    if name_match: entities["fullname"] = name_match.group(1).upper()
        
    return entities

def analyze_cross_document_graph(file_paths: list, output_image="knowledge_graph.png") -> dict:
    """
    Dynamic Feature Module: Accepts a variable-length list of file paths.
    Reads all documents, extracts entities, maps them against each other in a single graph,
    outputs a visual image, and verifies each document against the others to find fraud.
    """
    document_dataset = [extract_entities_from_document(fp) for fp in file_paths]
    
    G = nx.Graph()
    anomalies = []
    integrity_score = 100.0 
    
    # 1. Build the dynamic graph from all uploaded files
    for doc in document_dataset:
        if len(doc.keys()) <= 1: continue 
        
        doc_node = f"DOC_{doc['doc_type']}"
        G.add_node(doc_node, type="document", label=doc['doc_type'])
        
        for key, val in doc.items():
            if key == "doc_type": continue
            clean_val = str(val).strip().upper()
            if not G.has_node(clean_val):
                G.add_node(clean_val, type=key, label=clean_val)
            G.add_edge(doc_node, clean_val)
            
    # 2. Draw and Save the Visual Graph Structure
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, seed=42, k=0.5)
    
    colors = []
    for node, data in G.nodes(data=True):
        if data.get('type') == 'document': colors.append('lightblue')
        elif data.get('type') == 'pan_number': colors.append('lightgreen')
        elif data.get('type') == 'fullname': colors.append('lightcoral')
        else: colors.append('lightgray')

    nx.draw(G, pos, with_labels=True, node_color=colors, node_size=3000, 
            font_size=10, font_weight="bold", edge_color="gray", width=2.0)
    
    plt.title("TrustNet: Cross-Document Entity Mapping", fontsize=16)
    plt.savefig(output_image, format="PNG", bbox_inches="tight")
    plt.close()
            
    # 3. Analyze for Identity Collisions across the network
    identity_nodes = [n for n, attr in G.nodes(data=True) if attr.get('type') == 'pan_number']
    
    for identity in identity_nodes:
        connected_docs = G.neighbors(identity)
        associated_names = set()
        
        for doc in connected_docs:
            for neighbor in G.neighbors(doc):
                if G.nodes[neighbor].get('type') == 'fullname':
                    associated_names.add(neighbor)
                    
        if len(associated_names) > 1:
            anomalies.append(f"Identity collision: PAN '{identity}' resolves to multiple names across submitted files: {list(associated_names)}")
            integrity_score -= 60.0 
            
    anomaly_detected = integrity_score < 100.0

    return {
        "feature": "Cross-Document Graph Engine",
        "anomaly_detected": anomaly_detected,
        "integrity_score": max(0.0, round(integrity_score, 2)),
        "anomalies_discovered": anomalies,
        "graph_image_saved_at": output_image,
        "explanation": " | ".join(anomalies) if anomaly_detected else "All cross-document entities map securely to a single identity."
    }
