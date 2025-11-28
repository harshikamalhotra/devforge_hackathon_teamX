# src/graph_db/graph_loader.py
"""
Graph loader: create nodes & relationships in Memgraph from ingestion JSON.

Assumptions about JSON schema (matches ingestion output):
{
  "source": "sample.txt",
  "type": "txt",
  "metadata": {...},
  "paragraphs": [{"id":"p1","text":"..."}, ...],
  "tables": [{"id":"t1","rows":[...]} , ...],
  "entities": [{"id":"e1","text":"Alice","label":"PERSON", ...}, ...]
}

Behavior:
- Create Paragraph nodes: (:Paragraph { id, text, source })
- Create Entity nodes: (:Entity:PERSON { id, text, source })  (label value becomes node label too)
- Create Table nodes: (:Table { id, rows_count, source })
- Create relations:
    (Paragraph)-[:HAS_ENTITY]->(Entity)
    (Paragraph)-[:HAS_TABLE]->(Table)
    (Entity)-[:MENTIONED_IN]->(Paragraph)  (also created; duplicates are MERGE'd)
- All node properties include `source_file` to trace origin.
"""

from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import json

try:
    import networkx as nx
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend for compatibility
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    nx = None
    plt = None

from src.graph_db.memgraph_client import MemgraphClient


class GraphLoader:
    def __init__(self, memgraph_client: MemgraphClient = None):
        try:
            self.client = memgraph_client or MemgraphClient()
        except (ConnectionError, Exception):
            # Allow initialization even if Memgraph is not available
            # This is useful for visualization-only use cases
            self.client = None

    def load_from_json_file(self, json_path: str) -> Dict[str, int]:
        """
        Read JSON file and create nodes/edges in Memgraph.

        Returns a summary dict with counts.
        """
        p = Path(json_path)
        if not p.exists():
            raise FileNotFoundError(f"JSON file not found: {json_path}")

        with p.open("r", encoding="utf-8") as f:
            doc = json.load(f)

        return self.load_from_json(doc)

    def load_from_json(self, doc: Dict[str, Any]) -> Dict[str, int]:
        """
        Create nodes and relationships from a parsed JSON document.
        Returns summary counts.
        """
        source = doc.get("source", "unknown")
        paragraphs: List[Dict[str, Any]] = doc.get("paragraphs", [])
        entities: List[Dict[str, Any]] = doc.get("entities", [])
        tables: List[Dict[str, Any]] = doc.get("tables", [])

        created_nodes = 0
        created_rels = 0

        # Create paragraph nodes
        for p in paragraphs:
            pid = p.get("id") or self._make_id("paragraph")
            text = p.get("text", "")[:10000]  # trim if too long
            metadata = {"text": text, "source_file": source}
            # Paragraph label is 'Paragraph'
            self.client.create_entity_node(entity_id=pid, label="Paragraph", metadata=metadata)
            created_nodes += 1

        # Create table nodes
        for t in tables:
            tid = t.get("id") or self._make_id("table")
            rows = t.get("rows", [])
            metadata = {"rows_count": len(rows), "source_file": source}
            self.client.create_entity_node(entity_id=tid, label="Table", metadata=metadata)
            created_nodes += 1

        # Create entity nodes (use label from entity if available)
        for e in entities:
            eid = e.get("id") or self._make_id("entity")
            etext = e.get("text", "")
            elabel = e.get("label", "Entity")
            emeta = {"text": etext, "source_file": source}
            # Create node with label (e.g., PERSON, ORG) plus generic Entity label
            # We'll create with the provided label as the primary label
            self.client.create_entity_node(entity_id=eid, label=elabel, metadata=emeta)
            created_nodes += 1

        # Create relationships:
        # Paragraph -> HAS_ENTITY (if entity mentions exist)
        # Paragraph -> HAS_TABLE (if table ids referenced)
        # If exact offsets or context exist in entity objects, we try to link them.
        entity_map = {e.get("id"): e for e in entities if e.get("id")}
        paragraph_map = {p.get("id"): p for p in paragraphs if p.get("id")}
        table_map = {t.get("id"): t for t in tables if t.get("id")}

        # Attempt linking based on entity "context_paragraph_id" or proximity
        for e in entities:
            eid = e.get("id")
            # Prefer explicit link if provided
            ctx_pid = e.get("context_paragraph_id") or e.get("paragraph_id")
            if ctx_pid and ctx_pid in paragraph_map:
                # Create relation Paragraph HAS_ENTITY -> Entity
                try:
                    self.client.create_relationship(start_entity_id=ctx_pid, end_entity_id=eid, rel_type="HAS_ENTITY")
                    created_rels += 1
                except Exception:
                    # best-effort: ignore failing links
                    pass
            else:
                # If no explicit paragraph link, try to find paragraph containing the entity text (simple substring search)
                ent_text = (e.get("text") or "").strip()
                if ent_text:
                    for pid, p in paragraph_map.items():
                        if ent_text in (p.get("text") or ""):
                            try:
                                self.client.create_relationship(start_entity_id=pid, end_entity_id=eid, rel_type="HAS_ENTITY")
                                created_rels += 1
                                break
                            except Exception:
                                continue

        # Link paragraphs to tables if table ids are present in paragraph (best-effort)
        for pid, p in paragraph_map.items():
            p_text = p.get("text", "")
            for tid in table_map.keys():
                # naive check: table id appears in paragraph text
                if tid in p_text:
                    try:
                        self.client.create_relationship(start_entity_id=pid, end_entity_id=tid, rel_type="HAS_TABLE")
                        created_rels += 1
                    except Exception:
                        continue

        # Also create reciprocal mention links: Entity -> MENTIONED_IN -> Paragraph
        # (create both directions optional)
        for e in entities:
            eid = e.get("id")
            for pid, p in paragraph_map.items():
                if e.get("text") and e.get("text") in (p.get("text") or ""):
                    try:
                        self.client.create_relationship(start_entity_id=eid, end_entity_id=pid, rel_type="MENTIONED_IN")
                        created_rels += 1
                    except Exception:
                        continue

        return {"nodes_created": created_nodes, "relationships_created": created_rels}

    def _make_id(self, prefix: str) -> str:
        import uuid
        return f"{prefix}_{uuid.uuid4().hex[:8]}"
    
    def _get_friendly_relationship_label(self, rel_type: str) -> str:
        """
        Convert technical relationship types to user-friendly labels.
        
        Args:
            rel_type: Technical relationship type (e.g., "HAS_ENTITY", "MENTIONED_IN")
            
        Returns:
            User-friendly label (e.g., "Contains", "Mentioned In")
        """
        relationship_mapping = {
            "HAS_ENTITY": "Contains",
            "MENTIONED_IN": "Mentioned In",
            "HAS_TABLE": "References Table",
            "RELATED_TO": "Related To",
            "RELATED": "Related To",
            "CONNECTED_TO": "Connected To",
            "WORKS_AT": "Works At",
            "LOCATED_IN": "Located In",
            "PART_OF": "Part Of",
            "USES": "Uses",
            "COMBINES": "Combines",
            "GENERATES": "Generates"
        }
        
        # Try exact match first
        if rel_type in relationship_mapping:
            return relationship_mapping[rel_type]
        
        # Try case-insensitive match
        rel_upper = rel_type.upper()
        if rel_upper in relationship_mapping:
            return relationship_mapping[rel_upper]
        
        # Convert snake_case to Title Case as fallback
        friendly = rel_type.replace("_", " ").title()
        return friendly

    def visualize_hybrid_search_results(
        self,
        search_results: List[Dict[str, Any]],
        query_text: Optional[str] = None,
        output_path: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 8),
        node_size: int = 1000,
        font_size: int = 8
    ) -> Optional[Any]:
        """
        Create a graph visualization from hybrid search results.
        
        Extracts nodes (entities, paragraphs) and relationships from search results
        and visualizes them using networkx and matplotlib.
        
        Args:
            search_results: List of hybrid search result dictionaries from HybridRetriever
            query_text: Optional query text to display in the title
            output_path: Optional path to save the visualization image
            figsize: Figure size (width, height) in inches
            node_size: Size of nodes in the visualization
            font_size: Font size for node labels
            
        Returns:
            matplotlib figure object if visualization is available, None otherwise
        """
        if not VISUALIZATION_AVAILABLE:
            raise ImportError(
                "NetworkX and Matplotlib are required for visualization. "
                "Install them with: pip install networkx matplotlib"
            )
        
        # Build networkx graph from search results
        G = nx.DiGraph()
        
        # Track nodes and their properties
        node_info = {}  # node_id -> {label, type, score, etc}
        seen_relationships = set()  # (source, target, rel_type) tuples
        
        # Extract nodes and relationships from each search result
        for idx, result in enumerate(search_results):
            vector_result = result.get("vector_result", {})
            graph_relations = result.get("graph_relations", [])
            final_score = result.get("final_score", 0.0)
            vector_score = result.get("vector_score", 0.0)
            graph_score = result.get("graph_score", 0.0)
            hop = result.get("hop")
            
            # Extract paragraph/result node
            paragraph_id = vector_result.get("paragraph_id", "")
            doc_id = vector_result.get("doc_id", "")
            text = vector_result.get("text", "")
            metadata = vector_result.get("metadata", {})
            
            # Create paragraph node
            if paragraph_id:
                para_node_id = f"para_{paragraph_id}"
                if para_node_id not in node_info:
                    # Create a more readable label
                    # Truncate text for display (show first 40 chars)
                    text_preview = text[:40] + "..." if len(text) > 40 else text
                    # Clean up text preview (remove newlines)
                    text_preview = text_preview.replace("\n", " ").strip()
                    
                    # Show result rank and score
                    rank_label = f"Result #{idx + 1}"
                    score_label = f"Score: {final_score:.2f}"
                    
                    node_info[para_node_id] = {
                        "label": f"{rank_label}\n{score_label}\n{text_preview}",
                        "type": "Paragraph",
                        "score": final_score,
                        "vector_score": vector_score,
                        "graph_score": graph_score,
                        "hop": hop,
                        "rank": idx + 1
                    }
                    G.add_node(para_node_id)
            
            # Extract entity IDs from metadata
            entity_ids = []
            nested_metadata = metadata.get("metadata", {})
            eids = metadata.get("entity_ids") or nested_metadata.get("entity_ids", [])
            if isinstance(eids, list):
                entity_ids = eids
            
            # Add entity nodes
            for eid in entity_ids:
                if eid and eid not in node_info:
                    # Try to get entity info from graph DB
                    entity_label = "Entity"
                    entity_text = eid
                    try:
                        if self.client:
                            query = f"MATCH (n {{id: '{eid}'}}) RETURN labels(n) as labels, n.text as text, n.id as id"
                            entity_data = self.client.run_query(query)
                            if entity_data:
                                entity = entity_data[0]
                                labels = entity.get("labels", ["Entity"])
                                entity_label = labels[0] if labels else "Entity"
                                entity_text = entity.get("text", eid) or eid
                    except Exception:
                        # Fallback to using entity ID as text
                        pass
                    
                    # Create cleaner entity label
                    entity_display = entity_text[:35] if len(entity_text) > 35 else entity_text
                    entity_display = entity_display.replace("\n", " ").strip()
                    
                    # Use entity type as prefix if it's meaningful
                    if entity_label and entity_label != "Entity":
                        label = f"{entity_label}\n{entity_display}"
                    else:
                        label = entity_display
                    
                    node_info[eid] = {
                        "label": label,
                        "type": entity_label,
                        "score": None,
                        "vector_score": None,
                        "graph_score": None,
                        "hop": None
                    }
                    G.add_node(eid)
            
            # Add relationships from graph_relations
            for rel in graph_relations:
                source_id = rel.get("source_id")
                related_id = rel.get("related_id")
                rel_type = rel.get("rel_type", "RELATED")
                
                if source_id and related_id:
                    # Ensure both nodes exist
                    if source_id not in node_info:
                        G.add_node(source_id)
                        node_info[source_id] = {
                            "label": source_id[:30],
                            "type": "Entity",
                            "score": None,
                            "vector_score": None,
                            "graph_score": None,
                            "hop": None
                        }
                    
                    if related_id not in node_info:
                        G.add_node(related_id)
                        node_info[related_id] = {
                            "label": related_id[:30],
                            "type": "Entity",
                            "score": None,
                            "vector_score": None,
                            "graph_score": None,
                            "hop": None
                        }
                    
                    # Add edge if not already present
                    edge_key = (source_id, related_id, rel_type)
                    if edge_key not in seen_relationships:
                        G.add_edge(source_id, related_id, rel_type=rel_type)
                        seen_relationships.add(edge_key)
            
            # Link paragraph to entities if paragraph_id exists
            if paragraph_id and entity_ids:
                para_node_id = f"para_{paragraph_id}"
                for eid in entity_ids:
                    if para_node_id in G.nodes() and eid in G.nodes():
                        edge_key = (para_node_id, eid, "HAS_ENTITY")
                        if edge_key not in seen_relationships:
                            G.add_edge(para_node_id, eid, rel_type="HAS_ENTITY")
                            seen_relationships.add(edge_key)
        
        # If graph is empty, return None
        if len(G.nodes()) == 0:
            return None
        
        # Create visualization
        fig, ax = plt.subplots(figsize=figsize)
        
        # Use spring layout for better visualization with more spacing
        try:
            # Increase k value for more spacing between nodes
            pos = nx.spring_layout(G, k=2, iterations=100, seed=42)
        except:
            pos = nx.circular_layout(G)
        
        # Color nodes by type with better colors
        node_colors = []
        node_types_seen = set()
        
        # First pass: collect all node types
        for node in G.nodes():
            node_type = node_info.get(node, {}).get("type", "Unknown")
            node_types_seen.add(node_type)
        
        # Second pass: assign colors
        for node in G.nodes():
            node_type = node_info.get(node, {}).get("type", "Unknown")
            node_types_seen.add(node_type)
            
            if node_type == "Paragraph":
                node_colors.append("#4A90E2")  # Blue - for search results
            elif node_type in ["PERSON", "Person"]:
                node_colors.append("#50C878")  # Green - for people
            elif node_type in ["ORG", "Organization", "Company", "ORGANIZATION"]:
                node_colors.append("#FF6B6B")  # Red - for organizations
            elif node_type in ["LOCATION", "Location", "GPE"]:
                node_colors.append("#FFD93D")  # Yellow - for locations
            elif node_type == "Entity":
                node_colors.append("#9B59B6")  # Purple - for generic entities
            else:
                node_colors.append("#95A5A6")  # Gray - for other types
        
        # Draw nodes
        nx.draw_networkx_nodes(
            G, pos,
            node_color=node_colors,
            node_size=node_size,
            alpha=0.9,
            ax=ax
        )
        
        # Draw edges with user-friendly labels
        edge_labels = {}
        for u, v, data in G.edges(data=True):
            rel_type = data.get("rel_type", "")
            friendly_label = self._get_friendly_relationship_label(rel_type)
            edge_labels[(u, v)] = friendly_label
        
        # Draw edges with better styling
        nx.draw_networkx_edges(
            G, pos,
            edge_color="#7F8C8D",  # Darker gray for better visibility
            arrows=True,
            arrowsize=25,
            arrowstyle='->',
            width=2,
            alpha=0.7,
            connectionstyle='arc3,rad=0.1',  # Curved edges
            ax=ax
        )
        
        # Draw edge labels (only show if not too many edges)
        if len(edge_labels) <= 25:
            nx.draw_networkx_edge_labels(
                G, pos,
                edge_labels=edge_labels,
                font_size=7,
                font_color="#2C3E50",  # Dark blue-gray
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='none'),
                ax=ax
            )
        
        # Draw node labels with better styling
        labels = {node: node_info.get(node, {}).get("label", node) for node in G.nodes()}
        nx.draw_networkx_labels(
            G, pos,
            labels=labels,
            font_size=font_size,
            font_weight="bold",
            font_color="#2C3E50",  # Dark text for readability
            ax=ax
        )
        
        # Create a more informative title
        title = "📊 Knowledge Graph from Search Results"
        if query_text:
            title += f"\n🔍 Query: \"{query_text[:80]}\""
        subtitle = f"Showing {len(G.nodes())} connected items with {len(G.edges())} relationships"
        ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
        ax.text(0.5, 0.95, subtitle, transform=ax.transAxes, 
                fontsize=10, ha='center', style='italic', color='#555')
        
        # Add legend for node types
        if node_types_seen:
            legend_elements = []
            type_colors = {
                "Paragraph": "#4A90E2",
                "PERSON": "#50C878",
                "Person": "#50C878",
                "ORG": "#FF6B6B",
                "Organization": "#FF6B6B",
                "ORGANIZATION": "#FF6B6B",
                "LOCATION": "#FFD93D",
                "Location": "#FFD93D",
                "GPE": "#FFD93D",
                "Entity": "#9B59B6"
            }
            
            for node_type in sorted(node_types_seen):
                color = type_colors.get(node_type, "#95A5A6")
                friendly_type = node_type.replace("_", " ").title()
                legend_elements.append(
                    plt.Line2D([0], [0], marker='o', color='w', 
                              markerfacecolor=color, markersize=12, 
                              label=friendly_type, markeredgewidth=2, markeredgecolor='#2C3E50')
                )
            
            if legend_elements:
                ax.legend(handles=legend_elements, loc='upper left', 
                         bbox_to_anchor=(0, 1), frameon=True, 
                         fancybox=True, shadow=True, fontsize=9)
        
        ax.axis("off")
        plt.tight_layout()
        
        # Save if output path provided
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
        
        return fig
