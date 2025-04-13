from typing import List, Dict, Tuple, Optional, Any, Union
import numpy as np
from dataclasses import dataclass
import google.generativeai as genai
from google.generativeai import types
import faiss
import networkx as nx
from concurrent.futures import ThreadPoolExecutor
import logging
from datetime import datetime
import json
import os
import io
import base64
from PIL import Image

logger = logging.getLogger(__name__)

@dataclass
class Document:
    content: str
    metadata: Dict[str, Any]
    embeddings: Dict[str, np.ndarray]  # Multiple embeddings per document
    chunks: List['DocumentChunk']
    graph_connections: List[Tuple[str, float]]  # (doc_id, similarity)

@dataclass
class DocumentChunk:
    content: str
    embedding: np.ndarray
    metadata: Dict[str, Any]
    parent_doc_id: str

class AdvancedRAGSystem:
    def __init__(self):
        # Initialize FAISS indices for different embedding spaces
        self.semantic_index = faiss.IndexFlatIP(768)  # Semantic search
        self.query_index = faiss.IndexFlatIP(768)     # Query-focused search
        
        # Knowledge graph
        self.knowledge_graph = nx.Graph()
        
        # Cache for query results
        self.query_cache = {}
        
        # Document store
        self.documents: Dict[str, Document] = {}
        
        # Hyperparameters
        self.chunk_size = 512
        self.chunk_overlap = 128
        self.max_chunks_per_doc = 10
        self.rerank_top_k = 50
        self.final_results_k = 10

    def _generate_embeddings(self, text: str) -> Dict[str, np.ndarray]:
        """Generate multiple types of embeddings for a text using Gemini."""
        try:
            # Generate semantic embedding
            semantic_result = genai.embed_content(
                model="models/embedding-001",
                content=text,
                task_type="retrieval_document"
            )
            semantic_emb = np.array(semantic_result['embedding'])

            # Generate query-focused embedding
            query_result = genai.embed_content(
                model="models/embedding-001",
                content=text,
                task_type="retrieval_query"
            )
            query_emb = np.array(query_result['embedding'])

            return {
                'semantic': semantic_emb,
                'query': query_emb
            }
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            return {
                'semantic': np.zeros(768),
                'query': np.zeros(768)
            }

    def _chunk_document(self, content: str, doc_id: str) -> List[DocumentChunk]:
        """Chunk document with overlap and maintain relationships."""
        chunks = []
        words = content.split()
        
        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk_words = words[i:i + self.chunk_size]
            chunk_text = " ".join(chunk_words)
            
            # Generate embedding for chunk
            chunk_embedding = self._generate_embeddings(chunk_text)['semantic']
            
            chunk = DocumentChunk(
                content=chunk_text,
                embedding=chunk_embedding,
                metadata={'position': i // (self.chunk_size - self.chunk_overlap)},
                parent_doc_id=doc_id
            )
            chunks.append(chunk)
            
        return chunks[:self.max_chunks_per_doc]

    def add_document(self, content: str, metadata: Dict[str, Any], doc_id: str):
        """Add a document to the RAG system with multiple representations."""
        try:
            # Generate embeddings
            embeddings = self._generate_embeddings(content)
            
            # Create chunks
            chunks = self._chunk_document(content, doc_id)
            
            # Create document object
            doc = Document(
                content=content,
                metadata=metadata,
                embeddings=embeddings,
                chunks=chunks,
                graph_connections=[]
            )
            
            # Add to document store
            self.documents[doc_id] = doc
            
            # Add embeddings to FAISS indices
            self.semantic_index.add(np.array([embeddings['semantic']]))
            self.query_index.add(np.array([embeddings['query']]))
            
            # Update knowledge graph
            self._update_knowledge_graph(doc_id, doc)
        except Exception as e:
            logger.error(f"Error adding document: {e}")

    def _update_knowledge_graph(self, doc_id: str, doc: Document):
        """Update the knowledge graph with new document relationships."""
        try:
            self.knowledge_graph.add_node(doc_id, document=doc)
            
            # Find related documents
            semantic_emb = doc.embeddings['semantic'].reshape(1, -1)
            D, I = self.semantic_index.search(semantic_emb, k=5)
            
            # Add edges to related documents
            for idx, similarity in zip(I[0], D[0]):
                if similarity > 0.7:  # Threshold for relationship
                    related_doc_id = list(self.documents.keys())[idx]
                    if related_doc_id != doc_id:  # Avoid self-loops
                        self.knowledge_graph.add_edge(doc_id, related_doc_id, weight=similarity)
                        doc.graph_connections.append((related_doc_id, similarity))
        except Exception as e:
            logger.error(f"Error updating knowledge graph: {e}")

    def _hybrid_search(self, query: str, query_embeddings: Dict[str, np.ndarray]) -> List[Tuple[str, float]]:
        """Perform hybrid search using multiple indices."""
        try:
            # Semantic search
            D_semantic, I_semantic = self.semantic_index.search(
                query_embeddings['semantic'].reshape(1, -1), 
                self.rerank_top_k
            )
            
            # Query-focused search
            D_query, I_query = self.query_index.search(
                query_embeddings['query'].reshape(1, -1), 
                self.rerank_top_k
            )
            
            # Combine results with weights
            results = {}
            for idx, score in zip(I_semantic[0], D_semantic[0]):
                if idx < len(self.documents):
                    doc_id = list(self.documents.keys())[idx]
                    results[doc_id] = score * 0.6  # Weight for semantic search
                
            for idx, score in zip(I_query[0], D_query[0]):
                if idx < len(self.documents):
                    doc_id = list(self.documents.keys())[idx]
                    results[doc_id] = results.get(doc_id, 0) + score * 0.4  # Weight for query search
                
            return sorted(results.items(), key=lambda x: x[1], reverse=True)
        except Exception as e:
            logger.error(f"Error in hybrid search: {e}")
            return []

    def _rerank_results(self, query: str, initial_results: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        """Rerank results using Gemini and additional features."""
        try:
            final_scores = []
            
            for doc_id, initial_score in initial_results:
                doc = self.documents[doc_id]
                
                # Get chunk scores using Gemini
                chunk_scores = []
                for chunk in doc.chunks:
                    response = genai.GenerativeModel('gemini-1.0-pro').generate_content(
                        f"Rate the relevance of this text to the query on a scale of 0 to 1:\nQuery: {query}\nText: {chunk.content}\nReturn only the number.",
                        generation_config=types.GenerateContentConfig(
                            temperature=0.1,
                            candidate_count=1,
                            max_output_tokens=10
                        )
                    )
                    try:
                        score = float(response.text.strip())
                        chunk_scores.append(score)
                    except:
                        chunk_scores.append(0.0)
                
                # Calculate graph-based importance
                graph_score = 0.0
                if len(self.knowledge_graph) > 0:
                    try:
                        graph_score = nx.pagerank(self.knowledge_graph).get(doc_id, 0)
                    except:
                        pass
                
                # Combine scores
                final_score = (
                    max(chunk_scores) * 0.4 +  # Best chunk score
                    initial_score * 0.3 +      # Initial retrieval score
                    graph_score * 0.3          # Graph-based importance
                )
                
                final_scores.append((doc_id, final_score))
                
            return sorted(final_scores, key=lambda x: x[1], reverse=True)
        except Exception as e:
            logger.error(f"Error in reranking: {e}")
            return initial_results

    def _extract_relevant_chunks(self, doc_id: str, query: str) -> List[str]:
        """Extract most relevant chunks from a document for a query."""
        try:
            doc = self.documents[doc_id]
            chunk_scores = []
            
            for chunk in doc.chunks:
                response = genai.GenerativeModel('gemini-1.0-pro').generate_content(
                    f"Rate the relevance of this text to the query on a scale of 0 to 1:\nQuery: {query}\nText: {chunk.content}\nReturn only the number.",
                    generation_config=types.GenerateContentConfig(
                        temperature=0.1,
                        candidate_count=1,
                        max_output_tokens=10
                    )
                )
                try:
                    score = float(response.text.strip())
                    chunk_scores.append((chunk.content, score))
                except:
                    chunk_scores.append((chunk.content, 0.0))
                
            # Sort and return top chunks
            chunk_scores.sort(key=lambda x: x[1], reverse=True)
            return [chunk[0] for chunk in chunk_scores[:3]]
        except Exception as e:
            logger.error(f"Error extracting relevant chunks: {e}")
            return []

    def retrieve(self, query: str) -> Dict[str, Any]:
        """Main retrieval method with advanced features."""
        try:
            # Check cache
            cache_key = hash(query)
            if cache_key in self.query_cache:
                return self.query_cache[cache_key]
            
            # Generate query embeddings
            query_embeddings = self._generate_embeddings(query)
            
            # Perform hybrid search
            initial_results = self._hybrid_search(query, query_embeddings)
            
            # Rerank results
            reranked_results = self._rerank_results(query, initial_results)
            
            # Extract relevant chunks and prepare final response
            final_results = []
            for doc_id, score in reranked_results[:self.final_results_k]:
                relevant_chunks = self._extract_relevant_chunks(doc_id, query)
                doc = self.documents[doc_id]
                
                final_results.append({
                    'document_id': doc_id,
                    'score': score,
                    'metadata': doc.metadata,
                    'relevant_chunks': relevant_chunks,
                    'full_content': doc.content
                })
            
            # Prepare response with self-reflection
            response = {
                'results': final_results,
                'retrieval_strategy': self._explain_retrieval_strategy(query, final_results),
                'confidence_score': self._calculate_confidence(query, final_results)
            }
            
            # Cache results
            self.query_cache[cache_key] = response
            
            return response
        except Exception as e:
            logger.error(f"Error in retrieve: {e}")
            return {
                'results': [],
                'retrieval_strategy': f"Error during retrieval: {str(e)}",
                'confidence_score': 0.0
            }

    def _explain_retrieval_strategy(self, query: str, results: List[Dict]) -> str:
        """Explain the retrieval strategy used for transparency."""
        return f"""
        Retrieval Strategy:
        - Query analyzed using multiple Gemini embedding models
        - Hybrid search performed across {len(self.documents)} documents
        - Results reranked using Gemini relevance scoring and graph features
        - Retrieved {len(results)} most relevant documents
        - Confidence score: {self._calculate_confidence(query, results):.2f}
        """

    def _calculate_confidence(self, query: str, results: List[Dict]) -> float:
        """Calculate confidence score for the retrieval results."""
        try:
            # Average of top scores
            avg_score = np.mean([r['score'] for r in results[:3]]) if results else 0.0
            
            # Diversity of results (using metadata)
            unique_sources = len(set(r['metadata'].get('source', '') for r in results))
            diversity_score = unique_sources / len(results) if results else 0
            
            # Query-document similarity consistency
            similarities = [r['score'] for r in results]
            consistency_score = 1 - np.std(similarities) / np.mean(similarities) if similarities else 0
            
            # Combine scores
            confidence = (
                avg_score * 0.5 +
                diversity_score * 0.25 +
                consistency_score * 0.25
            )
            
            return min(1.0, max(0.0, confidence))
        except Exception as e:
            logger.error(f"Error calculating confidence: {e}")
            return 0.0
