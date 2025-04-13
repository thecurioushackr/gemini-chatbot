import logging
import time
import re
import json
import os
from datetime import datetime
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass
from collections import deque
from uuid import UUID
import numpy as np
import google.generativeai as genai
from google.generativeai import types
from .db import MemoryDB, run_async

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class Memory:
    content: str
    timestamp: datetime
    type: str  # 'episodic' or 'semantic'
    importance: float
    embedding: Optional[List[float]] = None

class MemoryManager:
    def __init__(self, user_id: UUID, max_memories: int = 1000, summary_threshold: int = 50):
        self.user_id = user_id
        self.max_memories = max_memories
        self.summary_threshold = summary_threshold
        self.episodic_memories: deque = deque(maxlen=max_memories)
        self.semantic_memories: List[Memory] = []
        
        # Initialize metrics
        self.metrics = {
            "total_memories": 0,
            "episodic_count": 0,
            "semantic_count": 0,
            "consolidations": 0
        }
        
        # Load existing memories from database
        self._load_memories()
        
    def _load_memories(self):
        """Load existing memories from database into memory cache."""
        try:
            # Load episodic memories
            episodic = run_async(MemoryDB.get_memories(
                user_id=self.user_id,
                memory_type="episodic",
                limit=self.max_memories
            ))
            
            # Load semantic memories
            semantic = run_async(MemoryDB.get_memories(
                user_id=self.user_id,
                memory_type="semantic",
                limit=self.max_memories
            ))
            
            # Convert to Memory objects and update cache
            for mem in episodic:
                memory = Memory(
                    content=mem['content'],
                    timestamp=mem['timestamp'],
                    type='episodic',
                    importance=mem['importance'],
                    embedding=mem['embedding']
                )
                self.episodic_memories.append(memory)
                
            for mem in semantic:
                memory = Memory(
                    content=mem['content'],
                    timestamp=mem['timestamp'],
                    type='semantic',
                    importance=mem['importance'],
                    embedding=mem['embedding']
                )
                self.semantic_memories.append(memory)
                
            # Update metrics
            self.metrics["episodic_count"] = len(episodic)
            self.metrics["semantic_count"] = len(semantic)
            self.metrics["total_memories"] = len(episodic) + len(semantic)
            
        except Exception as e:
            logger.error(f"Error loading memories from database: {e}")
        
    def add_memory(self, content: str, memory_type: str, importance: float = 0.5):
        """Add a new memory with automatic importance scoring and database persistence."""
        try:
            timestamp = datetime.now()
            embedding = self._generate_embedding(content)
            importance = self._calculate_importance(content, importance)
            
            # Create memory object
            memory = Memory(
                content=content,
                timestamp=timestamp,
                type=memory_type,
                importance=importance,
                embedding=embedding
            )
            
            # Store in database
            memory_id = run_async(MemoryDB.store_memory(
                content=content,
                timestamp=timestamp,
                memory_type=memory_type,
                importance=importance,
                embedding=embedding,
                user_id=self.user_id
            ))
            
            # Update in-memory cache
            if memory_type == "episodic":
                self.episodic_memories.append(memory)
                self.metrics["episodic_count"] += 1
                if len(self.episodic_memories) >= self.summary_threshold:
                    self._consolidate_memories()
            else:
                self.semantic_memories.append(memory)
                self.metrics["semantic_count"] += 1
                self._prune_semantic_memories()
                
            self.metrics["total_memories"] += 1
            
        except Exception as e:
            logger.error(f"Error adding memory: {e}")
            raise

    def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding using Gemini's embedding model."""
        try:
            result = genai.embed_content(
                model="models/embedding-001",
                content=text,
                task_type="retrieval_query"
            )
            return result['embedding']
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return []

    def _calculate_importance(self, content: str, base_importance: float) -> float:
        """Calculate memory importance using various factors."""
        try:
            # Length factor
            length_factor = min(len(content) / 1000, 1.0)
            
            # Emotional content factor (using sentiment analysis)
            sentiment = self._analyze_sentiment(content)
            emotion_factor = abs(sentiment)
            
            # Recency factor
            recency_factor = 1.0  # Most recent
            
            # Combine factors
            importance = (base_importance * 0.4 + 
                         length_factor * 0.2 + 
                         emotion_factor * 0.2 + 
                         recency_factor * 0.2)
            
            return min(importance, 1.0)
        except Exception as e:
            logger.error(f"Error calculating importance: {e}")
            return base_importance

    def _analyze_sentiment(self, text: str) -> float:
        """Analyze sentiment using Gemini."""
        try:
            response = genai.GenerativeModel("gemini-1.0-pro").generate_content(
                f"Analyze the emotional intensity of this text from -1 to 1: {text}. Return only a number.",
                generation_config=types.GenerateContentConfig(
                    temperature=0.1,
                    candidate_count=1,
                    max_output_tokens=10
                )
            )
            return float(response.text.strip())
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {e}")
            return 0.0

    def _consolidate_memories(self):
        """Consolidate episodic memories into semantic memories."""
        try:
            # Group related memories
            memories_text = [m.content for m in self.episodic_memories]
            
            if len(memories_text) >= self.summary_threshold:
                # Create summary using Gemini
                summary = self._summarize_with_gemini(memories_text)
                
                # Create semantic memory from summary
                self.add_memory(summary, "semantic", importance=0.8)
                
                # Clear episodic memories from database
                run_async(MemoryDB.delete_old_memories(
                    user_id=self.user_id,
                    memory_type="episodic",
                    keep_count=0
                ))
                
                # Clear episodic memories from cache
                self.episodic_memories.clear()
                self.metrics["consolidations"] += 1
                
        except Exception as e:
            logger.error(f"Error consolidating memories: {e}")

    def _summarize_with_gemini(self, texts: List[str]) -> str:
        """Summarize text using Gemini."""
        try:
            combined_text = " ".join(texts)
            response = genai.GenerativeModel("gemini-1.0-pro").generate_content(
                f"Summarize this text concisely: {combined_text}",
                generation_config=types.GenerateContentConfig(
                    temperature=0.3,
                    candidate_count=1,
                    max_output_tokens=150
                )
            )
            return response.text
        except Exception as e:
            logger.error(f"Error in Gemini summarization: {e}")
            return "Error generating summary"

    def _prune_semantic_memories(self):
        """Prune semantic memories based on importance and age."""
        try:
            if len(self.semantic_memories) > self.max_memories:
                # Calculate scores based on importance and age
                current_time = datetime.now()
                scores = []
                
                for memory in self.semantic_memories:
                    age = (current_time - memory.timestamp).total_seconds() / (24 * 3600)  # age in days
                    age_factor = 1.0 / (1.0 + age)
                    score = memory.importance * 0.7 + age_factor * 0.3
                    scores.append(score)
                
                # Keep only the top memories
                keep_indices = np.argsort(scores)[-self.max_memories:]
                self.semantic_memories = [self.semantic_memories[i] for i in keep_indices]
                
                # Update database
                run_async(MemoryDB.delete_old_memories(
                    user_id=self.user_id,
                    memory_type="semantic",
                    keep_count=self.max_memories
                ))
                
        except Exception as e:
            logger.error(f"Error pruning semantic memories: {e}")

    def get_relevant_memories(self, query: str, top_k: int = 5) -> List[Memory]:
        """Retrieve relevant memories using semantic search from database."""
        try:
            query_embedding = self._generate_embedding(query)
            
            # Get similar memories from database
            db_memories = run_async(MemoryDB.find_similar_memories(
                query_embedding=query_embedding,
                user_id=self.user_id,
                match_threshold=0.7,
                match_count=top_k
            ))
            
            # Convert database results to Memory objects
            memories = [
                Memory(
                    content=m['content'],
                    timestamp=m['timestamp'],
                    type=m['type'],
                    importance=m['importance'],
                    embedding=m['embedding']
                )
                for m in db_memories
            ]
            
            return memories
            
        except Exception as e:
            logger.error(f"Error retrieving relevant memories: {e}")
            return []

    @staticmethod
    def _cosine_similarity(a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        try:
            if not a or not b:
                return 0.0
            a = np.array(a)
            b = np.array(b)
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            return 0.0

    def get_metrics(self) -> Dict[str, Any]:
        """Return current memory metrics."""
        return {
            **self.metrics,
            "current_episodic_count": len(self.episodic_memories),
            "current_semantic_count": len(self.semantic_memories)
        }
