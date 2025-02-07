from typing import List, Dict, Any, Optional, Union
from uuid import UUID
import logging
from datetime import datetime
import re
from PIL import Image

import google.generativeai as genai
from google.generativeai import types

from .memory import MemoryManager
from .crag import AdvancedRAGSystem
from .image_gen import AdvancedImageGenerator, ImageGenerationConfig

logger = logging.getLogger(__name__)

class AthenaAgent:
    def __init__(self, user_id: UUID, model_cache_dir: Optional[str] = None):
        self.user_id = user_id
        self.model_name = "gemini-1.5-pro-002"
        self.memory_manager = MemoryManager(user_id)
        self.rag_system = AdvancedRAGSystem()
        self.image_generator = AdvancedImageGenerator(cache_dir=model_cache_dir)
        self.system_instruction = """You are Athena, a helpful research assistant.
            You have access to a knowledge base, can browse the web, and generate images.
            Prioritize factual accuracy. Use your knowledge base first, then the web if needed.
            For image generation, you can use commands like 'generate image: [prompt]' with optional
            parameters like style, size, and other configurations.
            """
        
    def _enhance_prompt_with_context(self, query: str) -> str:
        """Enhance the prompt with both memory and RAG results."""
        try:
            # Get relevant memories
            memories = self.memory_manager.get_relevant_memories(query)
            memory_context = "\n".join([
                f"Memory [{m.type}]: {m.content}"
                for m in memories
            ]) if memories else ""
            
            # Get RAG results
            rag_results = self.rag_system.retrieve(query)
            
            # Format relevant chunks from RAG
            rag_chunks = []
            for result in rag_results['results']:
                for chunk in result['relevant_chunks']:
                    rag_chunks.append(f"Source: {result['metadata'].get('source', 'Unknown')}\n{chunk}")
            rag_context = "\n\n".join(rag_chunks) if rag_chunks else ""
            
            # Create enhanced prompt
            enhanced_prompt = f"""
            Query: {query}

            Past Memories:
            {memory_context}

            Relevant Knowledge:
            {rag_context}

            Retrieval Confidence: {rag_results['confidence_score']:.2f}

            {rag_results['retrieval_strategy']}

            Based on the above context, memories, and your knowledge, please provide a comprehensive response.
            """
            
            return enhanced_prompt
        except Exception as e:
            logger.error(f"Error enhancing prompt: {e}")
            return query

    def _parse_image_command(self, text: str) -> Optional[Dict[str, Any]]:
        """Parse image generation command and parameters."""
        try:
            # Check if this is an image generation command
            match = re.match(r'generate\s+image:?\s*(.*)', text, re.IGNORECASE)
            if not match:
                return None

            command_text = match.group(1)
            
            # Extract parameters using regex
            params = {
                'prompt': command_text,
                'style_preset': None,
                'width': 1024,
                'height': 1024,
                'num_images': 1
            }
            
            # Look for style preset
            style_match = re.search(r'style:\s*(\w+)', command_text)
            if style_match:
                params['style_preset'] = style_match.group(1)
                params['prompt'] = params['prompt'].replace(style_match.group(0), '').strip()
            
            # Look for size
            size_match = re.search(r'size:\s*(\d+)x(\d+)', command_text)
            if size_match:
                params['width'] = int(size_match.group(1))
                params['height'] = int(size_match.group(2))
                params['prompt'] = params['prompt'].replace(size_match.group(0), '').strip()
            
            # Look for number of images
            num_match = re.search(r'count:\s*(\d+)', command_text)
            if num_match:
                params['num_images'] = int(num_match.group(1))
                params['prompt'] = params['prompt'].replace(num_match.group(0), '').strip()
            
            return params
        except Exception as e:
            logger.error(f"Error parsing image command: {e}")
            return None

    def generate_image(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate image with advanced configuration."""
        try:
            config = ImageGenerationConfig(
                prompt=prompt,
                **kwargs
            )
            return self.image_generator.generate(config)
        except Exception as e:
            logger.error(f"Error generating image: {e}")
            return {
                "status": "error",
                "error": str(e)
            }

    def process_input(self, user_input: Union[str, types.Part]):
        """Process user input with memory, RAG, and image generation integration."""
        try:
            if isinstance(user_input, str):
                # Store input in episodic memory
                self.memory_manager.add_memory(user_input, "episodic")
                
                # Enhance prompt with context
                enhanced_input = self._enhance_prompt_with_context(user_input)
                
                # Generate response using Gemini
                response = genai.GenerativeModel(model_name=self.model_name).generate_content(
                    enhanced_input,
                    generation_config=types.GenerateContentConfig(
                        temperature=0.7,
                        top_p=0.95,
                        top_k=40,
                        max_output_tokens=2048,
                        safety_settings=[
                            {
                                "category": "HARM_CATEGORY_HARASSMENT",
                                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                            },
                            {
                                "category": "HARM_CATEGORY_HATE_SPEECH",
                                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                            },
                            {
                                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                            },
                            {
                                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                            }
                        ]
                    )
                )
                
                # Store response in memory
                if response.text:
                    self.memory_manager.add_memory(
                        response.text,
                        "semantic",
                        importance=0.8
                    )
                    
                    # Store in RAG system if it's a substantial response
                    if len(response.text) > 100:
                        self.rag_system.add_document(
                            content=response.text,
                            metadata={
                                'source': 'agent_response',
                                'timestamp': datetime.now().isoformat(),
                                'query': user_input
                            },
                            doc_id=f"response_{datetime.now().timestamp()}"
                        )
                
                # Check if this is an image generation command
                if isinstance(user_input, str):
                    image_params = self._parse_image_command(user_input)
                    if image_params:
                        result = self.generate_image(**image_params)
                        if result["status"] == "success":
                            # Store the generation in memory
                            self.memory_manager.add_memory(
                                f"Generated image with prompt: {image_params['prompt']}",
                                "semantic",
                                importance=0.7
                            )
                            return result
                        else:
                            return f"Failed to generate image: {result.get('error', 'Unknown error')}"

                return response.text
            else:
                # Handle non-text inputs (images, etc.)
                response = genai.GenerativeModel(model_name=self.model_name).generate_content(user_input)
                return response.text
                
        except Exception as e:
            logger.error(f"Error processing input: {e}")
            return f"I apologize, but I encountered an error: {str(e)}"

    def add_to_knowledge_base(self, content: str, metadata: Dict[str, Any] = None):
        """Add content to both memory and RAG systems."""
        try:
            # Add to memory system
            self.memory_manager.add_memory(content, "semantic", importance=0.9)
            
            # Add to RAG system
            self.rag_system.add_document(
                content=content,
                metadata=metadata or {
                    'source': 'knowledge_base',
                    'timestamp': datetime.now().isoformat()
                },
                doc_id=f"kb_{datetime.now().timestamp()}"
            )
        except Exception as e:
            logger.error(f"Error adding to knowledge base: {e}")
            raise

    def cleanup(self):
        """Clean up resources."""
        try:
            self.image_generator.cleanup()
        except Exception as e:
            logger.error(f"Error in cleanup: {e}")

    def get_metrics(self) -> Dict[str, Any]:
        """Get combined metrics from both systems."""
        try:
            memory_metrics = self.memory_manager.get_metrics()
            
            # Add basic RAG metrics
            rag_metrics = {
                'documents_count': len(self.rag_system.documents),
                'cache_size': len(self.rag_system.query_cache),
                'graph_nodes': len(self.rag_system.knowledge_graph.nodes),
                'graph_edges': len(self.rag_system.knowledge_graph.edges)
            }
            
            return {
                'memory': memory_metrics,
                'rag': rag_metrics
            }
        except Exception as e:
            logger.error(f"Error getting metrics: {e}")
            return {}
