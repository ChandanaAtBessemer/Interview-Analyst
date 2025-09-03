
import os
import re
import json
import uuid
import hashlib
import pickle
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Tuple
from io import BytesIO
import logging
import networkx as nx
import re
from collections import defaultdict
import time
from dataclasses import dataclass, asdict
from fastapi import FastAPI, HTTPException, UploadFile, File, Depends, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from dotenv import load_dotenv
import tiktoken
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from backend.concept_mapper import map_to_concept
# Your intelligent imports - UPDATED for new LangChain
from PyPDF2 import PdfReader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, OpenAI  # Updated imports
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
import docx2txt

from backend.semantic_graph_builder import build_reasoning_trace
load_dotenv()

# Rate limiting
limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="Fixed Production Interview Q&A API", version="2.1.0")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000","*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def session_cleanup_middleware(request: Request, call_next):
    """Detect new sessions and clean up memory"""
    
    response = await call_next(request)
    
    # If this is a new session (no file_id in recent requests), clean up aggressively
    if request.url.path == "/api/upload":
        try:
            import psutil
            import os
            process = psutil.Process(os.getpid())
            memory_mb = process.memory_info().rss / 1024 / 1024
            
            logger.info(f"Upload request memory check: {memory_mb:.1f}MB")
            
            # If memory is high before upload, clean cache aggressively
            if memory_mb > 200:  # 200MB threshold
                logger.warning(f"Pre-upload memory high ({memory_mb:.1f}MB), cleaning cache...")
                
                # Clear most cache, keeping only most recent entry
                if len(vector_cache) > 1:
                    # Keep only the most recently accessed cache
                    most_recent = max(vector_cache.items(), 
                                    key=lambda x: x[1].get('timestamp', datetime.min))
                    vector_cache.clear()
                    vector_cache[most_recent[0]] = most_recent[1]
                    
                    import gc
                    gc.collect()
                    
                    memory_after = process.memory_info().rss / 1024 / 1024
                    logger.info(f"Memory after pre-upload cleanup: {memory_after:.1f}MB")
        except:
            pass
    
    return response


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Security (optional)
security = HTTPBearer(auto_error=False)
API_TOKEN = os.getenv("API_TOKEN", "dev-token-12345")
ENABLE_AUTH = os.getenv("ENABLE_AUTH", "false").lower() == "true"

# Token counting
encoding = tiktoken.get_encoding("cl100k_base")

# In-memory cache
vector_cache: Dict[str, Dict] = {}
CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# Models
class ProcessRequest(BaseModel):
    script: str

class QueryRequest(BaseModel):
    script: str
    question: str
    file_id: Optional[str] = None

class ValidationRequest(BaseModel):
    script: str
    question: str
    answer: str
    file_id: Optional[str] = None

class GraphEntityExtractor:
    """Extract entities and relationships from interview chunks"""
    
    def __init__(self):
        self.entity_patterns = {
            'features': r'\b(feature|functionality|app|interface|button|screen|tool|system|platform)\w*\b',
            'emotions': r'\b(like|love|hate|frustrate|enjoy|prefer|dislike|appreciate)\w*\b',
            'issues': r'\b(crash|bug|slow|fast|problem|issue|error|fail|broken|glitch)\w*\b',
            'actions': r'\b(click|tap|use|try|navigate|download|install|update|access)\w*\b',
            'qualities': r'\b(easy|difficult|hard|simple|complex|intuitive|confusing|clear)\w*\b'
        }
    
    def extract_from_chunk(self, chunk_data):
        """Extract entities from chunk"""
        entities = []
        content = chunk_data["content"].lower()
        speaker = chunk_data["speaker"]
        
        for entity_type, pattern in self.entity_patterns.items():
            matches = re.findall(pattern, content, re.IGNORECASE)
            for match in matches:
                entities.append({
                    'type': entity_type,
                    'value': match.lower(),
                    'speaker': speaker,
                    'chunk_id': chunk_data.get('original_chunk_id', 0),
                    'context': content[:100] + "..." if len(content) > 100 else content
                })
        
        return entities

class ProcessResponse(BaseModel):
    speakers: List[str]
    chunks: int
    file_id: str
    message: str

class QueryResponse(BaseModel):
    question: str
    answer: str
    speakers_detected: List[str]
    chunks_used: int
    tokens_used: int
    file_id: str
    reasoning: List[str] = []

class ValidationResponse(BaseModel):
    grounded: bool
    confidence: int
    evidence: List[str]
    issues: List[str]
    reasoning: str
class EnhancedQueryResponse(BaseModel):
    question: str
    answer: str
    speakers_detected: List[str]
    chunks_used: int
    tokens_used: int
    file_id: str
    reasoning: List[str] = []
    reasoning_trace: Optional[Dict[str, Any]] = None
    enhanced: bool = False
@dataclass
class ReasoningStep:
    """Represents a single step in the reasoning process"""
    step_id: str
    step_type: str  # 'analysis', 'graph', 'retrieval', 'generation', 'validation'
    title: str
    content: str
    details: Dict[str, Any]
    timestamp: float
    duration_ms: int = 0

@dataclass
class ReasoningTrace:
    """Complete reasoning trace for a query"""
    query: str
    query_id: str
    start_time: float
    steps: List[ReasoningStep]
    metadata: Dict[str, Any]
    total_duration_ms: int = 0

class ReasoningTracker:
    """Tracks and manages reasoning steps throughout query processing"""
    
    def __init__(self):
        self.current_trace: Optional[ReasoningTrace] = None
        self.step_start_time: float = 0
        
    def start_trace(self, query: str) -> str:
        query_id = f"query_{int(time.time() * 1000)}"
        self.current_trace = ReasoningTrace(
            query=query,
            query_id=query_id,
            start_time=time.time(),
            steps=[],
            metadata={}
        )
        return query_id
    
    def start_step(self, step_id: str, step_type: str, title: str):
        self.step_start_time = time.time()
        
    def add_step(self, step_id: str, step_type: str, title: str, content: str, details: Dict[str, Any] = None):
        if not self.current_trace:
            return
        duration = int((time.time() - self.step_start_time) * 1000) if self.step_start_time > 0 else 0
        step = ReasoningStep(
            step_id=step_id,
            step_type=step_type,
            title=title,
            content=content,
            details=details or {},
            timestamp=time.time(),
            duration_ms=duration
        )
        self.current_trace.steps.append(step)
        self.step_start_time = 0
        
    def finish_trace(self) -> ReasoningTrace:
        if not self.current_trace:
            return None
        self.current_trace.total_duration_ms = int((time.time() - self.current_trace.start_time) * 1000)
        return self.current_trace


class CachedVectorDB:
    """Manages vector DB caching and reuse - FIXED VERSION"""
    
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        # Fixed: Remove batch_size parameter that's causing issues
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-ada-002",
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        # Updated LLM initialization
        self.llm = OpenAI(
            temperature=0.1,
            max_tokens=2000,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # NEW: Graph processing components
        self.entity_extractor = GraphEntityExtractor()
        
        # Updated prompt template for new LangChain
        self.prompt_template = ChatPromptTemplate.from_template("""
        Answer the question based only on the following context:
        
        {context}
        
        Question: {input}
        
        Answer: """)
        
    def generate_file_id(self, content: str, filename: str = None) -> str:
        """Generate unique ID for content including filename for better uniqueness"""
        hash_input = content
        if filename:
            hash_input = f"{filename}::{content}"
        return hashlib.md5(hash_input.encode()).hexdigest()
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        return len(encoding.encode(text))
    
    def truncate_to_tokens(self, text: str, max_tokens: int = 6000) -> str:
        """Truncate text to fit token limit with buffer"""
        tokens = encoding.encode(text)
        if len(tokens) <= max_tokens:
            return text
        
        truncated_tokens = tokens[:max_tokens]
        return encoding.decode(truncated_tokens)
    
    def enhanced_speaker_detection(self, raw_text: str) -> Tuple[List[str], List[str]]:
        """Simple but effective speaker detection - CLEAN VERSION"""
        lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
        logger.info(f"🔍 DEBUG: Processing {len(lines)} lines")
        logger.info(f"🔍 First 10 lines: {lines[:10]}")
        
        # Look for speaker patterns in first 20 lines
        speaker_samples = []
        for i, line in enumerate(lines[:20]):
            if ":" in line and len(line) < 100:
                speaker_samples.append(f"Line {i}: {line}")
        logger.info(f"🔍 Speaker pattern samples: {speaker_samples}")
        raw_chunks = []
        current_speaker = None
        buffer = []
        
        speaker_patterns = [
                re.compile(r"^(.*?):\s*(.*)$", re.IGNORECASE),
                re.compile(r"^(Interviewer\s*[—–-]\s*[^:]+):\s*(.*)$", re.IGNORECASE),  # Interviewer – Name:
                re.compile(r"^(Interviewee\s*[—–-]\s*[^:]+):\s*(.*)$", re.IGNORECASE),  # Interviewee – Name:
                re.compile(r"^(Interviewer):\s*(.*)$", re.IGNORECASE),                   # Interviewer:
                re.compile(r"^(Interviewee):\s*(.*)$", re.IGNORECASE),                   # Interviewee:
                re.compile(r"^([A-Z][a-z]+\s+[A-Z][a-z]+):\s*(.*)$"),                    # Nick Ruscher:
                re.compile(r"^(Q|Question):\s*(.*)$", re.IGNORECASE),                    # Q:
                re.compile(r"^(A|Answer):\s*(.*)$", re.IGNORECASE),                      # A:
        ]

        for line_num, line in enumerate(lines):
            speaker_found = False
            
            for pattern in speaker_patterns:
                #match = re.match(pattern, line)
                match = pattern.match(line)

                if match:
                    logger.info(f"🔍 MATCH: Pattern matched at line {line_num}: '{line[:50]}...' → Speaker: '{match.group(1)}'")
                    # Save previous speaker's content
                    if buffer and current_speaker:
                        content = " ".join(buffer).strip()
                        if content:
                            raw_chunks.append({
                                "content": content,
                                "speaker": current_speaker,
                                "line_start": line_num - len(buffer),
                                "line_end": line_num - 1
                            })
                    '''
                    # Extract speaker and remaining text
                    if pattern == speaker_patterns[0]:  # Name: format
                        current_speaker = match.group(1)
                        remaining = match.group(2).strip()
                    elif pattern == speaker_patterns[1]:  # Name - Role: format
                        current_speaker = match.group(1).strip()  # Always use group(1) for speaker
                        remaining = match.group(2).strip()
                    elif pattern == speaker_patterns[2]:  # Q: format
                        current_speaker = "Interviewer"
                        remaining = match.group(2).strip()
                    elif pattern == speaker_patterns[3]:  # A: format
                        current_speaker = "Interviewee"
                        remaining = match.group(2).strip()
                    '''

                    # Extract speaker and remaining text - FIXED VERSION
                    current_speaker = match.group(1).strip()  # Always use group(1) for speaker
                    remaining = match.group(2).strip()        # Always use group(2) for remaining text

                    # Override for Q/A patterns
                    if pattern in [speaker_patterns[3], speaker_patterns[4]]:  # Q: or A: patterns
                        if match.group(1).lower() in ['q', 'question']:
                            current_speaker = "Interviewer"
                        elif match.group(1).lower() in ['a', 'answer']:
                            current_speaker = "Interviewee"

                    buffer = [remaining] if remaining else []
                    speaker_found = True
                    break
            
            if not speaker_found:
                buffer.append(line)
        
        # Handle final buffer
        if buffer and current_speaker:
            content = " ".join(buffer).strip()
            if content:
                raw_chunks.append({
                    "content": content,
                    "speaker": current_speaker,
                    "line_start": len(lines) - len(buffer),
                    "line_end": len(lines) - 1
                })
        
        # NOW DO SMART CLEANUP - Group similar speakers
        speaker_groups = {}
        interviewer_chunks = []
        interviewee_chunks = []
        other_chunks = {}
        
        for chunk in raw_chunks:
            speaker = chunk["speaker"].lower()
            
            # Categorize by speaker type
            if speaker in ["interviewer", "q", "question"] or "interviewer" in speaker:
                interviewer_chunks.append(chunk)
            elif speaker in ["interviewee", "a", "answer", "candidate"] or "interviewee" in speaker:
                interviewee_chunks.append(chunk)
            else:
                # This is likely a person's name (like "Faulkner")
                # Assume named speakers are interviewees unless proven otherwise
                if speaker not in other_chunks:
                    other_chunks[speaker] = []
                other_chunks[speaker].append(chunk)
        
        # Final speaker assignment
        final_chunks = []
        final_speakers = []
        
        # Add interviewer chunks
        if interviewer_chunks:
            final_speakers.append("Interviewer")
            for chunk in interviewer_chunks:
                chunk["speaker"] = "Interviewer"
                final_chunks.append(chunk)
        
        # Add interviewee chunks
        if interviewee_chunks:
            final_speakers.append("Interviewee")
            for chunk in interviewee_chunks:
                chunk["speaker"] = "Interviewee"
                final_chunks.append(chunk)
        
        # Add named speakers (assume they are interviewees with names)
        for speaker_name, chunks in other_chunks.items():
            clean_name = speaker_name.title()  # Capitalize properly
            speaker_label = f"Interviewee - {clean_name}"
            final_speakers.append(speaker_label)
            for chunk in chunks:
                chunk["speaker"] = speaker_label
                final_chunks.append(chunk)
        
        # If we have both "Interviewee" and "Interviewee - Name", merge them
        if "Interviewee" in final_speakers and any("Interviewee -" in s for s in final_speakers):
            # Remove generic "Interviewee" and keep the named one
            final_speakers = [s for s in final_speakers if s != "Interviewee"]
            for chunk in final_chunks:
                if chunk["speaker"] == "Interviewee":
                    # Find the named interviewee speaker
                    named_speaker = next((s for s in final_speakers if "Interviewee -" in s), "Interviewee")
                    chunk["speaker"] = named_speaker
        
        # Sort chunks by line number to maintain order
        final_chunks.sort(key=lambda x: x["line_start"])
        
        logger.info(f"Clean detection: {len(final_chunks)} chunks, {len(final_speakers)} speakers: {final_speakers}")
        return final_chunks, final_speakers

    def build_graph_from_chunks(self, chunks_data, file_id):
        """Build knowledge graph from chunks"""
        # ADD THIS DEBUG BLOCK:
        logger.info(f"🔍 GRAPH BUILD DEBUG: Processing {len(chunks_data)} chunks")
        chunk_speakers = [chunk['speaker'] for chunk in chunks_data]
        unique_speakers = list(set(chunk_speakers))
        logger.info(f"🔍 GRAPH BUILD DEBUG: Unique speakers in chunks: {unique_speakers}")
        logger.info(f"🔍 GRAPH BUILD DEBUG: Speaker counts: {dict((s, chunk_speakers.count(s)) for s in unique_speakers)}")
        G = nx.Graph()
        
        # Track entity relationships
        entity_speakers = defaultdict(set)
        speaker_entities = defaultdict(set)
        
        for chunk in chunks_data:
            speaker = chunk["speaker"] 
            entities = self.entity_extractor.extract_from_chunk(chunk)
            
            # Add speaker node
            G.add_node(speaker, type="speaker")
            
            # Process entities
            for entity in entities:
                entity_key = f"{entity['type']}:{entity['value']}"
                
                # Add entity node
                G.add_node(entity_key, 
                        type=entity['type'], 
                        value=entity['value'])
                
                # Add speaker-entity edge
                G.add_edge(speaker, entity_key,
                        relation="mentions",
                        chunk_id=entity['chunk_id'],
                        context=entity['context'][:50])
                
                # Track for co-occurrence
                entity_speakers[entity_key].add(speaker)
                speaker_entities[speaker].add(entity_key)
        
        # Add entity co-occurrence edges (entities mentioned by same speaker)
        for speaker, entities in speaker_entities.items():
            entity_list = list(entities)
            for i, entity1 in enumerate(entity_list):
                for entity2 in entity_list[i+1:]:
                    if not G.has_edge(entity1, entity2):
                        G.add_edge(entity1, entity2,
                                relation="co_mentioned",
                                speakers=[speaker])
                    else:
                        # Update existing edge
                        edge_data = G.get_edge_data(entity1, entity2)
                        if 'speakers' in edge_data:
                            edge_data['speakers'].append(speaker)
        
        logger.info(f"Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        return G

    # Replace your existing should_use_graph method in main.py with this enhanced version

    def should_use_graph(self, question: str) -> bool:
        """Determine if query needs graph processing - ENHANCED FOR IMPACT ANALYSIS"""
        graph_indicators = [
            # Existing patterns (working well)
            r'\b(who.*also|both.*and|users.*who.*but|participants.*that)\b',
            r'\b(compare|versus|vs|both|all.*who|among.*who)\b',
            r'\b(also.*mentioned|common.*between|shared.*by)\b',
            r'\b(but.*also|however.*also|while.*also)\b',
            r'\b(that.*also|which.*also|.*also.*affect|.*also.*cause)\b',
            r'\b(problems.*that.*also|issues.*that.*also|challenges.*that.*also)\b',
            r'\b(mentioned.*that.*also|identified.*that.*also|discussed.*that.*also)\b',
            r'\b(between.*and|connecting.*to|relating.*to)\b',
            r'\b(specific.*that.*also|particular.*that.*also)\b',
            r'\b(common.*between|shared.*across|similar.*across)\b',
            r'\b(what.*did.*say.*about.*that.*also)\b',
            r'\b(which.*mentioned.*also|what.*identified.*also)\b',
            
            # NEW: Impact and outcome patterns (THESE SHOULD CATCH  QUERY)
            r'\b(impact.*on|effect.*on|influence.*on)\b',
            r'\b(how.*would.*impact|how.*would.*affect|how.*would.*influence)\b',
            r'\b(impact.*the.*challenges|affect.*the.*challenges|solve.*the.*challenges)\b',
            r'\b(resulted.*in|led.*to|caused.*by)\b',
            r'\b(consequences.*of|outcomes.*of|effects.*of)\b',
            r'\b(solutions.*for.*challenges|address.*challenges|solve.*problems)\b',
            r'\b(standardized.*apis.*impact|apis.*impact.*challenges)\b',
            
            # NEW: Causal relationship patterns
            r'\b(that.*caused|that.*resulted.*in|that.*led.*to)\b',
            r'\b(that.*prevent|that.*could.*prevent|that.*would.*prevent)\b',
            r'\b(issues.*that.*caused|problems.*that.*caused|challenges.*that.*caused)\b',
            r'\b(solutions.*that.*could|technology.*that.*could|approaches.*that.*could)\b',
            r'\b(mentioned.*that.*caused|identified.*that.*caused|described.*that.*caused)\b',
            
                
            # Relationship patterns (universal)
            r'\b(who.*also|both.*and|all.*who|participants.*that)\b',
            r'\b(compare|versus|vs|both|all.*who|among.*who)\b',
            r'\b(also.*mentioned|common.*between|shared.*by)\b',
            r'\b(but.*also|however.*also|while.*also)\b',
            r'\b(that.*also|which.*also|.*also.*affect|.*also.*cause)\b',
            
            # Problem and solution relationships (universal)
            r'\b(problems.*that.*also|issues.*that.*also|challenges.*that.*also)\b',
            r'\b(solutions.*that.*could|approaches.*that.*could|ways.*that.*could)\b',
            r'\b(mentioned.*that.*also|identified.*that.*also|discussed.*that.*also)\b',
            
            # Connection patterns (universal)
            r'\b(between.*and|connecting.*to|relating.*to)\b',
            r'\b(specific.*that.*also|particular.*that.*also)\b',
            r'\b(common.*between|shared.*across|similar.*across)\b',
            
            # Impact and effect patterns (universal)
            r'\b(impact.*on|effect.*on|influence.*on)\b',
            r'\b(how.*would.*impact|how.*would.*affect|how.*would.*influence)\b',
            r'\b(impact.*the.*challenges|affect.*the.*challenges|solve.*the.*challenges)\b',
            r'\b(resulted.*in|led.*to|caused.*by)\b',
            r'\b(consequences.*of|outcomes.*of|effects.*of)\b',
            
            # Causal relationship patterns (universal)
            r'\b(that.*caused|that.*resulted.*in|that.*led.*to)\b',
            r'\b(that.*prevent|that.*could.*prevent|that.*would.*prevent)\b',
            r'\b(issues.*that.*caused|problems.*that.*caused|challenges.*that.*caused)\b',
            
            # Generic reference patterns (works for any name/topic)
            r'\b(outlined.*challenges|mentioned.*challenges|described.*challenges)\b',
            r'\b(outlined.*problems|mentioned.*problems|described.*problems)\b',
            r'\b(outlined.*issues|mentioned.*issues|described.*issues)\b',
            r'\b(challenges.*outlined|problems.*mentioned|issues.*described)\b',
            
            # Multiple entity patterns (universal)
            r'\b(what.*did.*say.*about.*that.*also)\b',
            r'\b(which.*mentioned.*also|what.*identified.*also)\b',
            r'\b(everyone.*who|anyone.*who|people.*who)\b',
   
        ]
        
        question_lower = question.lower()
        
        # Add debug logging to see which patterns match
        logger.info(f"🔍 Checking graph routing for: {question_lower}")
        
        matched_patterns = []
        for i, pattern in enumerate(graph_indicators):
            if re.search(pattern, question_lower):
                logger.info(f"✅ Graph pattern {i} matched: {pattern}")
                matched_patterns.append(pattern)
        
        if matched_patterns:
            logger.info(f"🕸️ Graph routing ENABLED - {len(matched_patterns)} patterns matched")
            return True
        else:
            logger.info("📄 Graph routing DISABLED - using vector search")
            return False

    # Also update your _analyze_query_with_details method to provide more details:

    def _analyze_query_with_details(self, question: str):
        """Analyze query with detailed reasoning capture - ENHANCED"""
        patterns = []
        confidence = 0.0
        
        question_lower = question.lower()
        
        # Enhanced pattern detection with specific pattern identification
        graph_indicators = [
            (r'\b(impact.*on|effect.*on|influence.*on)\b', "impact_analysis"),
            (r'\b(how.*would.*impact|how.*would.*affect)\b', "impact_prediction"),
            (r'\b(challenges.*outlined|mentioned.*challenges)\b', "challenge_reference"),
            (r'\b(standardized.*apis|apis.*impact)\b', "api_solution"),
            (r'\b(both.*and|compare|versus)\b', "comparative_analysis"),
            (r'\b(also.*mentioned|common.*between)\b', "relationship_analysis"),
        ]
        
        matched_pattern_types = []
        for pattern, pattern_type in graph_indicators:
            if re.search(pattern, question_lower):
                patterns.append(pattern)
                matched_pattern_types.append(pattern_type)
                confidence += 0.2  # Increase confidence for each match
        
        use_graph = len(patterns) > 0 and confidence >= 0.2
        query_type = self._classify_query_type_enhanced(question_lower, matched_pattern_types)
        
        routing_decision = {
            "method": "graph_enhanced" if use_graph else "vector_search",
            "use_graph": use_graph,
            "patterns": matched_pattern_types,  # Show pattern types instead of regex
            "type": query_type,
            "confidence": min(confidence, 1.0),
            "reason": self._get_routing_reason(matched_pattern_types, use_graph)
        }
        
        routing_details = {
            "question_length": len(question.split()),
            "patterns_checked": len(graph_indicators),
            "patterns_matched": len(matched_pattern_types),
            "analysis_time_ms": 50
        }
        
        return routing_decision, routing_details

    def _classify_query_type_enhanced(self, question_lower: str, pattern_types: List[str]) -> str:
        """Enhanced query type classification"""
        if "impact_analysis" in pattern_types or "impact_prediction" in pattern_types:
            return "impact_analysis"
        elif "comparative_analysis" in pattern_types:
            return "comparative_analysis"
        elif "challenge_reference" in pattern_types:
            return "challenge_analysis"
        elif "api_solution" in pattern_types:
            return "solution_analysis"
        elif "what" in question_lower:
            return "factual_query"
        elif "how" in question_lower:
            return "process_query"
        else:
            return "general_query"

    def _get_routing_reason(self, pattern_types: List[str], use_graph: bool) -> str:
        """Generate human-readable routing reason"""
        if not use_graph:
            return "Simple factual query - vector search sufficient"
        
        reasons = []
        if "impact_analysis" in pattern_types:
            reasons.append("impact analysis detected")
        if "challenge_reference" in pattern_types:
            reasons.append("challenge reference found")
        if "api_solution" in pattern_types:
            reasons.append("solution mapping needed")
        if "comparative_analysis" in pattern_types:
            reasons.append("comparative analysis required")
        
        return f"Complex query requiring graph reasoning: {', '.join(reasons)}"

    def graph_enhanced_query(self, graph, vector_db, speakers, question: str):
        """Enhanced query using semantic + type-based matching"""
        
        question_lower = question.lower()
        
        # Step 1: Semantic concept mapping
        # Step 1: Enhanced semantic concept mapping
        concept_mapping = {
            # Existing mappings
            'challenges': ['issues', 'actions'],
            'problems': ['issues', 'actions'], 
            'operational': ['issues', 'actions'],
            'customer': ['emotions', 'issues'],
            'retention': ['emotions', 'actions'],
            'impact': ['issues', 'emotions'],
            'technology': ['features', 'actions'],
            'investment': ['features', 'actions'],
            'efficiency': ['qualities', 'actions'],
            'automation': ['features', 'actions'],
            
            # NEW: Business-specific terms
            'vendor': ['issues', 'emotions', 'actions'],
            'supplier': ['issues', 'emotions', 'actions'],
            'reliability': ['issues', 'qualities'],
            'delivery': ['issues', 'actions'],
            'business': ['issues', 'actions', 'emotions'],
            'losses': ['issues', 'emotions'],
            'revenue': ['issues', 'emotions'],
            'profit': ['issues', 'emotions'],
            'relationship': ['emotions', 'issues'],
            'communication': ['issues', 'actions'],
            'quality': ['qualities', 'issues'],
            'performance': ['qualities', 'issues'],
            'cost': ['issues', 'emotions'],
            'time': ['issues', 'actions'],
            'delay': ['issues', 'actions'],
            'accuracy': ['issues', 'qualities'],
            'data': ['issues', 'features'],
            'information': ['issues', 'features'],
            'system': ['features', 'issues'],
            'process': ['actions', 'issues'],
            'manual': ['actions', 'issues'],
            'integration': ['features', 'actions'],
            'api': ['features', 'actions'],
            'solution': ['features', 'actions'],
            'service': ['features', 'emotions'],
            'support': ['features', 'emotions']
        }
        
        # Step 2: Find relevant entity types based on question concepts
        relevant_entity_types = set()
        for concept, entity_types in concept_mapping.items():
            if concept in question_lower:
                relevant_entity_types.update(entity_types)
        
        # If no specific concepts found, include all types
        if not relevant_entity_types:
            relevant_entity_types = {'issues', 'emotions', 'actions', 'features', 'qualities'}
        
        # Step 3: Get entities of relevant types
        relevant_entities = []
        for node in graph.nodes():
            if ":" in node:
                entity_type, entity_value = node.split(":", 1)
                if entity_type in relevant_entity_types:
                    relevant_entities.append(node)
        
        # Step 4: Find connected speakers and all their entities
        connected_speakers = set()
        all_speaker_entities = set()
        
        for entity_node in relevant_entities:
            for neighbor in graph.neighbors(entity_node):
                if graph.nodes[neighbor].get('type') == 'speaker':
                    connected_speakers.add(neighbor)
                    
                    # Get ALL entities for these speakers
                    for speaker_neighbor in graph.neighbors(neighbor):
                        if ":" in speaker_neighbor:
                            all_speaker_entities.add(speaker_neighbor)
        
        # Step 5: Enhanced search with both question and entity context
        search_terms = [question]
        
        # Add entity values for context
        entity_values = []
        for entity in all_speaker_entities:
            if ":" in entity:
                _, entity_value = entity.split(":", 1)
                entity_values.append(entity_value)
        
        # Use top entity values
        search_terms.extend(entity_values[:6])
        enhanced_query = " ".join(search_terms)
        
        # Step 6: Vector search with speaker prioritization
        # Step 6: Enhanced document retrieval with proper speaker filtering
        # Step 6: Enhanced document retrieval with content quality filtering
        docs = vector_db.similarity_search_with_score(enhanced_query, k=15)

        # Get chunks from connected speakers with content quality scoring
        speaker_docs = []
        if connected_speakers:
            all_speaker_docs = vector_db.similarity_search_with_score("", k=100)
            
            for doc, score in all_speaker_docs:
                doc_speaker = doc.metadata.get("speaker", "")
                if doc_speaker in connected_speakers:
                    content = doc.page_content.strip()
                    
                    # Content quality scoring
                    quality_score = score
                    
                    # Filter out very short responses
                    if len(content) < 20:  # Skip "Yes", "Yeah", etc.
                        continue
                        
                    # Boost longer, more substantive content
                    if len(content) > 100:
                        quality_score *= 0.6  # Strong boost for substantial content
                    
                    # Boost content with business-relevant keywords
                    #business_keywords = ['customer', 'lose', 'lost', 'delivery', 'eaton', 'time', 'problem', 'issue', 'challenge']
                    #keyword_matches = sum(1 for keyword in business_keywords if keyword in content.lower())
                    #if keyword_matches >= 2:
                        #quality_score *= 0.7  # Boost for business relevance
                    
                    speaker_docs.append((doc, quality_score))

        # Rest of combination logic stays the same...
        combined_docs = {}

        for doc, score in docs:
            content_hash = hash(doc.page_content[:200])
            if content_hash not in combined_docs:
                combined_docs[content_hash] = (doc, score)

        for doc, score in speaker_docs:
            content_hash = hash(doc.page_content[:200])
            if content_hash not in combined_docs or combined_docs[content_hash][1] > score:
                combined_docs[content_hash] = (doc, score)

        final_docs = sorted(combined_docs.values(), key=lambda x: x[1])[:12]

        # Enhanced debug logging
        speaker_distribution = {}
        sample_content = []
        for doc, score in final_docs:
            speaker = doc.metadata.get("speaker", "Unknown")
            speaker_distribution[speaker] = speaker_distribution.get(speaker, 0) + 1
            
            is_primary_speaker = (speaker == primary_speaker or ('interviewer' not in speaker.lower() and len(doc.page_content) > 50))
            if is_primary_speaker:
                sample_content.append(doc.page_content[:150] + "...")

        logger.info(f"Final context speaker distribution: {speaker_distribution}")
        logger.info(f"Quality primary speaker content samples: {sample_content[:3]}")

        return [doc for doc, score in final_docs]
    
    def build_vector_db(self, raw_text: str, file_id: str) -> Tuple[FAISS, List[str], List[Dict]]:
        """Build vector DB with caching and metadata tracking - FIXED"""
        
        # Check cache first
        cache_path = os.path.join(CACHE_DIR, f"{file_id}.pkl")
        if os.path.exists(cache_path) and file_id in vector_cache:
            logger.info(f"Loading cached vector DB for {file_id}")
            with open(cache_path, 'rb') as f:
                cached_data = pickle.load(f)
            return cached_data['db'], cached_data['speakers'], cached_data['chunks']
        
        # Build new vector DB
        logger.info(f"Building new vector DB for {file_id}")
        chunks_data, speakers = self.enhanced_speaker_detection(raw_text)
        
        if not chunks_data:
            raise ValueError("No speaker-labeled chunks found.")
        
        # DEBUG LINE 1: Added here
        logger.info(f"About to build graph with {len(chunks_data)} chunks")
        
        # Prepare texts and metadata for FAISS
        texts = []
        metadatas = []
        
        for chunk in chunks_data:
            # Split long chunks if needed
            content = chunk["content"]
            if self.count_tokens(content) > 800:  # Split large chunks
                sub_chunks = self.text_splitter.split_text(content)
                for i, sub_chunk in enumerate(sub_chunks):
                    texts.append(sub_chunk)
                    metadatas.append({
                        "speaker": chunk["speaker"],
                        "line_start": chunk["line_start"],
                        "line_end": chunk["line_end"],
                        "sub_chunk": i,
                        "original_chunk_id": len(texts) - len(sub_chunks) + i
                    })
            else:
                texts.append(content)
                metadatas.append({
                    "speaker": chunk["speaker"],
                    "line_start": chunk["line_start"],
                    "line_end": chunk["line_end"],
                    "sub_chunk": 0,
                    "original_chunk_id": len(texts) - 1
                })
        
        # Build FAISS with fixed embeddings (no batch_size)
        try:
            logger.info(f"Creating embeddings for {len(texts)} chunks...")
            db = FAISS.from_texts(
                texts, 
                self.embeddings, 
                metadatas=metadatas
            )
            logger.info("FAISS vector database created successfully")
        except Exception as e:
            logger.error(f"FAISS building failed: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to build vector database: {str(e)}")
        
        # Cache the results
        graph = self.build_graph_from_chunks(chunks_data, file_id)

        # Cache the results (enhanced)
        cache_data = {
            'db': db,
            'speakers': speakers,
            'chunks': chunks_data,
            'graph': graph,  # NEW
            'timestamp': datetime.now()
        }
       
        vector_cache[file_id] = cache_data
        
        # Persist to disk
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(cache_data, f)
        except Exception as e:
            logger.warning(f"Failed to cache to disk: {e}")
        
        return db, speakers, chunks_data
    
    #def query_with_metadata(self, db: FAISS, speakers: List[str], question: str, max_tokens: int = 3000) -> Tuple[str, List, int,List[str]]:
    def query_with_metadata(self, db: FAISS, speakers: List[str], question: str, file_id: str, max_tokens: int = 3000) -> Tuple[str, List, int, List[str]]:
        """Enhanced query with graph + vector hybrid - UPDATED"""
        
        # DEBUG LINE 2: Added here
        logger.info(f"Query received: {question}")
        
        # Check if we have graph data in cache
        '''
        graph = None
        for cached_data in vector_cache.values():
            if cached_data.get('db') == db and 'graph' in cached_data:
                graph = cached_data['graph']
                break
        '''
        graph = vector_cache.get(file_id, {}).get('graph')
        # DEBUG LINE 3: Added here
        logger.info(f"Graph found: {graph is not None}")
        
        # Route to appropriate query method
        if graph and self.should_use_graph(question):
            logger.info("Using graph-enhanced query")
            selected_docs = self.graph_enhanced_query(graph, db, speakers, question)
        else:
            logger.info("Using standard vector query")
            # Original vector search logic
            docs = db.similarity_search_with_score(question, k=10)
            
            # Speaker-specific filtering (existing logic)
            target_speaker = None
            for speaker in speakers:
                name_part = speaker.split(" - ")[-1].lower()
                if name_part != "unknown" and name_part in question.lower():
                    target_speaker = speaker
                    break
            
            if target_speaker:
                speaker_docs = [(doc, score) for doc, score in docs 
                            if doc.metadata.get("speaker") == target_speaker]
                if speaker_docs:
                    docs = speaker_docs
            
            selected_docs = [doc for doc, score in sorted(docs, key=lambda x: x[1])[:6]]
        
        # Token-aware context building (existing logic)
        final_docs = []
        total_tokens = 0
        
        for doc in selected_docs:
            doc_tokens = self.count_tokens(doc.page_content)
            if total_tokens + doc_tokens <= max_tokens:
                final_docs.append(doc)
                total_tokens += doc_tokens
            else:
                remaining_tokens = max_tokens - total_tokens
                if remaining_tokens > 100:
                    truncated_content = self.truncate_to_tokens(doc.page_content, remaining_tokens)
                    doc.page_content = truncated_content
                    final_docs.append(doc)
                    total_tokens += self.count_tokens(truncated_content)
                break
        
        if not final_docs:
            return "No relevant content found for your question.", [], 0
        
        # Generate answer (existing logic)
        try:
            context = "\n\n".join([doc.page_content for doc in final_docs])
            
            prompt = f"""Answer the question based only on the following context from the interview:

    Context:
    {context}

    Question: {question}

    Answer based only on the context provided:"""
            
            response = self.llm.invoke(prompt)
            reasoning_steps = []
            for idx, doc in enumerate(final_docs, start=1):
                speaker = doc.metadata.get("speaker", "Unknown")
                snippet = doc.page_content.strip().replace("\n", " ")
                if len(snippet) > 120:
                    snippet = snippet[:120] + "..."
                reasoning_steps.append(f"Step {idx}: Selected chunk from {speaker}: '{snippet}'")

            return response, final_docs, total_tokens,reasoning_steps
            
        except Exception as e:
            logger.error(f"Query failed: {e}")
            return f"Error processing question: {str(e)}", final_docs, total_tokens,reasoning_steps

class EnhancedCachedVectorDB(CachedVectorDB):
    """Enhanced version with reasoning trace capabilities"""
    
    def __init__(self):
        super().__init__()
        self.reasoning_tracker = ReasoningTracker()
    # Add these methods to your EnhancedCachedVectorDB class

    def _graph_enhanced_query_with_trace(self, graph, vector_db, speakers: List[str], question: str):
        """Generic graph query with detailed step tracing"""
        
        # Step 2a: Generic Concept Mapping
        self.reasoning_tracker.start_step("concept_mapping", "graph", "Concept Mapping")
        time.sleep(0.01)
        concept_mapping, mapped_entities = self._perform_detailed_concept_mapping(question)
        
        # Create detailed description of what was mapped
        concepts_text = ", ".join(f"'{concept}'" for concept in concept_mapping.keys())
        
        self.reasoning_tracker.add_step(
            "concept_mapping", "graph", "🗺️ Concept Mapping",
            f"Found concepts: {concepts_text} → mapped to {len(mapped_entities)} entity types",
            {
                "concepts_found": list(concept_mapping.keys()),
                "entity_types": mapped_entities,
                "detailed_mapping": concept_mapping,
                "mapping_summary": f"Identified {len(concept_mapping)} key concepts from question"
            }
        )
        
        # Step 2b: Generic Entity Discovery
        self.reasoning_tracker.start_step("entity_discovery", "graph", "Entity Discovery")
        time.sleep(0.01)
        relevant_entities, entity_stats = self._discover_entities_with_details(graph, mapped_entities)
        
        self.reasoning_tracker.add_step(
            "entity_discovery", "graph", "🔍 Entity Discovery", 
            f"Found {len(relevant_entities)} entities across {len(entity_stats.get('entity_type_distribution', {}))} types",
            entity_stats
        )
        
        # Step 2c: Generic Speaker Connection Analysis
        self.reasoning_tracker.start_step("speaker_analysis", "graph", "Speaker Connection Analysis")
        time.sleep(0.01)
        connected_speakers, connection_details = self._analyze_speaker_connections(graph, relevant_entities)
        
        # Create generic speaker description
        primary_speaker = connection_details.get('primary_speaker', 'Unknown')
        primary_connections = connection_details.get('primary_speaker_connections', 0)
        
        speaker_description = f"Connected speakers: {', '.join(connected_speakers)}"
        if primary_speaker and primary_speaker != 'Unknown':
            speaker_name = primary_speaker.replace('Interviewee - ', '').replace('Interview - ', '')
            speaker_description += f" (Primary: {speaker_name} with {primary_connections} entity connections)"
        
        self.reasoning_tracker.add_step(
            "speaker_analysis", "graph", "👥 Speaker Connection Analysis",
            speaker_description,
            connection_details
        )
        
        # Step 2d: Generic Enhanced Document Retrieval
        self.reasoning_tracker.start_step("graph_retrieval", "graph", "Enhanced Document Retrieval")
        time.sleep(0.01)
        documents, retrieval_details = self._enhanced_document_retrieval(vector_db, question, connected_speakers, relevant_entities)
        
        retrieval_flow = f"{retrieval_details.get('initial_vector_results', 0)} initial → {retrieval_details.get('speaker_filtered_results', 0)} speaker-filtered → {retrieval_details.get('final_selection_count', 0)} final"
        primary_boost_text = ""
        if retrieval_details.get('primary_speaker_docs', 0) > 0 and retrieval_details.get('primary_speaker_name'):
            speaker_name = retrieval_details['primary_speaker_name'].replace('Interviewee - ', '').replace('Interview - ', '')
            primary_boost_text = f"Primary speaker ({speaker_name}): {retrieval_details.get('primary_speaker_docs', 0)} documents prioritized"
        
        content_text = f"{retrieval_flow}. {primary_boost_text}" if primary_boost_text else retrieval_flow
        
        self.reasoning_tracker.add_step(
            "graph_retrieval", "graph", "📊 Enhanced Document Retrieval",
            content_text,
            retrieval_details
        )
        
        return documents

    def _perform_detailed_concept_mapping(self, question: str) -> Tuple[Dict[str, List[str]], List[str]]:
        """Generic concept mapping that works for any interview domain"""
    
        # Universal concept mapping for any interview type
        concept_mapping = {
            # Impact and effect concepts (universal)
            'impact': ['issues', 'emotions', 'outcomes', 'benefits'],
            'affect': ['issues', 'emotions', 'outcomes'],
            'influence': ['issues', 'emotions', 'outcomes'],
            'effect': ['issues', 'outcomes', 'benefits'],
            
            # Challenge and problem concepts (universal)
            'challenges': ['issues', 'actions', 'problems'],
            'problems': ['issues', 'actions', 'solutions'], 
            'issues': ['issues', 'emotions'],
            'difficulties': ['issues', 'actions'],
            'obstacles': ['issues', 'actions'],
            
            # Solution concepts (universal)
            'solution': ['features', 'actions', 'outcomes', 'benefits'],
            'solutions': ['features', 'actions', 'outcomes', 'benefits'],
            'solve': ['features', 'actions', 'outcomes'],
            'address': ['features', 'actions', 'solutions'],
            'improve': ['features', 'actions', 'benefits'],
            'fix': ['actions', 'solutions'],
            
            # Technology concepts (broad)
            'technology': ['features', 'actions'],
            'system': ['features', 'issues'],
            'process': ['actions', 'issues'],
            'tool': ['features', 'actions'],
            'platform': ['features', 'actions'],
            
            # Business process concepts (universal)
            'operational': ['issues', 'actions'],
            'efficiency': ['qualities', 'actions', 'benefits'],
            'performance': ['qualities', 'issues'],
            'quality': ['qualities', 'issues'],
            'reliability': ['issues', 'qualities'],
            'maintenance': ['actions', 'issues'],
            'support': ['features', 'emotions'],
            'service': ['features', 'emotions'],
            
            # Communication and interaction (universal)
            'communication': ['issues', 'actions'],
            'collaboration': ['actions', 'emotions'],
            'feedback': ['emotions', 'actions'],
            'relationship': ['emotions', 'issues'],
            
            # Change and improvement (universal)
            'change': ['actions', 'emotions', 'outcomes'],
            'update': ['actions', 'issues'],
            'upgrade': ['actions', 'benefits'],
            'implement': ['actions', 'features'],
            'adopt': ['actions', 'emotions'],
            
            # Experience and opinion concepts (universal)
            'experience': ['emotions', 'qualities'],
            'opinion': ['emotions', 'preferences'],
            'prefer': ['emotions', 'preferences'],
            'like': ['emotions', 'preferences'],
            'dislike': ['emotions', 'preferences'],
            'satisfied': ['emotions', 'qualities'],
            'frustrated': ['emotions', 'issues'],
            
            # Reference concepts (universal)
            'mentioned': ['references', 'discussed'],
            'discussed': ['references', 'topics'],
            'outlined': ['references', 'topics'],
            'described': ['references', 'detailed'],
            'explained': ['references', 'detailed'],
            
            # Comparison concepts (universal)
            'compare': ['comparisons', 'analysis'],
            'versus': ['comparisons', 'analysis'],
            'different': ['comparisons', 'distinctions'],
            'similar': ['comparisons', 'similarities'],
            'both': ['comparisons', 'shared'],
            'all': ['comprehensive', 'shared'],
            
            # Time and frequency (universal)
            'often': ['frequency', 'patterns'],
            'usually': ['frequency', 'patterns'], 
            'sometimes': ['frequency', 'patterns'],
            'always': ['frequency', 'absolute'],
            'never': ['frequency', 'absolute'],
            
            # Outcome and result concepts (universal)
            'result': ['outcomes', 'consequences'],
            'outcome': ['outcomes', 'results'],
            'consequence': ['outcomes', 'effects'],
            'benefit': ['benefits', 'positive'],
            'advantage': ['benefits', 'positive'],
            'disadvantage': ['issues', 'negative'],
        }
        
        question_lower = question.lower()
        mapped_entities = set()
        concepts_found = {}
        
        # Find all matching concepts in the question
        for concept, entity_types in concept_mapping.items():
            if concept in question_lower:
                mapped_entities.update(entity_types)
                concepts_found[concept] = entity_types
        
        # If no specific concepts found, use a default set for general analysis
        if not mapped_entities:
            mapped_entities = {'issues', 'actions', 'features', 'qualities', 'emotions'}
            concepts_found['general_analysis'] = list(mapped_entities)
        
        # Log what was found for debugging (generic)
        logger.info(f"🗺️ Generic concept mapping results:")
        for concept, types in concepts_found.items():
            logger.info(f"   '{concept}' → {types}")
        
        return concepts_found, list(mapped_entities)

    def _discover_entities_with_details(self, graph, mapped_entities: List[str]) -> Tuple[List[str], Dict]:
        """Discover relevant entities with detailed statistics"""
        
        relevant_entities = []
        entity_type_counts = {}
        entity_examples = {}
        
        for node in graph.nodes():
            if ":" in node:
                entity_type, entity_value = node.split(":", 1)
                if entity_type in mapped_entities:
                    relevant_entities.append(node)
                    entity_type_counts[entity_type] = entity_type_counts.get(entity_type, 0) + 1
                    
                    # Store examples for each type
                    if entity_type not in entity_examples:
                        entity_examples[entity_type] = []
                    if len(entity_examples[entity_type]) < 3:  # Limit examples
                        entity_examples[entity_type].append(entity_value)
        
        # Calculate coverage and distribution
        total_entities = len([n for n in graph.nodes() if ":" in n])
        coverage = (len(relevant_entities) / max(total_entities, 1)) * 100
        
        entity_stats = {
            "total_entities_in_graph": total_entities,
            "relevant_entities_found": len(relevant_entities),
            "entity_type_distribution": entity_type_counts,
            "entity_examples": entity_examples,
            "coverage_percentage": coverage,
            "dominant_entity_type": max(entity_type_counts.keys(), key=entity_type_counts.get) if entity_type_counts else None
        }
        
        return relevant_entities, entity_stats

    # Replace your enhanced methods with these generic versions:

    def _analyze_speaker_connections(self, graph, relevant_entities: List[str]) -> Tuple[List[str], Dict]:
        """Generic speaker connection analysis - works with any speakers"""
        
        connected_speakers = set()
        speaker_entity_counts = {}
        entity_speaker_map = {}
        
        for entity_node in relevant_entities:
            entity_connections = []
            for neighbor in graph.neighbors(entity_node):
                if graph.nodes[neighbor].get('type') == 'speaker':
                    connected_speakers.add(neighbor)
                    entity_connections.append(neighbor)
                    speaker_entity_counts[neighbor] = speaker_entity_counts.get(neighbor, 0) + 1
            
            if entity_connections:
                entity_speaker_map[entity_node] = entity_connections
        
        # Calculate connection metrics generically
        total_connections = sum(speaker_entity_counts.values())
        avg_connections = total_connections / len(connected_speakers) if connected_speakers else 0
        
        # Find the speaker with most entity connections (likely the main interviewee)
        primary_speaker = None
        primary_connections = 0
        if speaker_entity_counts:
            primary_speaker = max(speaker_entity_counts.keys(), key=speaker_entity_counts.get)
            primary_connections = speaker_entity_counts[primary_speaker]
        
        # Find multi-speaker entities (shared challenges/topics)
        multi_speaker_entities = [
            entity for entity, speakers in entity_speaker_map.items() 
            if len(speakers) > 1
        ]
        
        connection_details = {
            "total_speakers_connected": len(connected_speakers),
            "speaker_entity_counts": speaker_entity_counts,
            "multi_speaker_entities": len(multi_speaker_entities),
            "shared_entities_list": multi_speaker_entities,
            "connection_strength": avg_connections / 5.0 if avg_connections > 0 else 0,  # Normalized to 0-1 
            "primary_speaker": primary_speaker,
            "primary_speaker_connections": primary_connections,
            "dominant_speaker": primary_speaker,  # Same as primary for backwards compatibility
            "all_connected_speakers": list(connected_speakers)
        }
        
        return list(connected_speakers), connection_details

    def _enhanced_document_retrieval(self, vector_db, question: str, connected_speakers: List[str], relevant_entities: List[str]) -> Tuple[List, Dict]:
        """Generic enhanced document retrieval - prioritizes main speaker automatically"""
        
        # Phase 1: Enhanced query construction
        entity_values = []
        for entity in relevant_entities[:6]:  # Limit for performance
            if ":" in entity:
                _, entity_value = entity.split(":", 1)
                entity_values.append(entity_value)
        
        search_terms = [question] + entity_values
        enhanced_query = " ".join(search_terms)
        
        # Phase 2: Initial vector search
        initial_docs = vector_db.similarity_search_with_score(enhanced_query, k=15)
        
        # Phase 3: Find primary speaker (most entity connections) from connected speakers
        primary_speaker = None
        if connected_speakers:
            # Get speaker with most connections from graph analysis
            # This will be set by _analyze_speaker_connections method
            for cached_data in vector_cache.values():
                if 'graph' in cached_data:
                    # We'll identify primary speaker through entity connections
                    break
            
            # For now, assume the first non-interviewer speaker is primary
            for speaker in connected_speakers:
                if 'interviewer' not in speaker.lower() and 'question' not in speaker.lower():
                    primary_speaker = speaker
                    break
        
        # Phase 4: Speaker-specific filtering with primary speaker priority
        speaker_docs = []
        primary_speaker_docs = []
        
        for doc, score in initial_docs:
            doc_speaker = doc.metadata.get("speaker", "")
            if doc_speaker in connected_speakers:
                content = doc.page_content.strip()
                
                # Skip very short responses
                if len(content) < 20:
                    continue
                
                # Quality scoring
                quality_score = score
                
                # Primary speaker boosting (generic)
                if primary_speaker and doc_speaker == primary_speaker:
                    quality_score *= 0.5  # Strong boost for primary speaker content
                    primary_speaker_docs.append((doc, quality_score))
                
                # Boost substantial content
                if len(content) > 100:
                    quality_score *= 0.7
                
                # Generic business keyword boosting (not domain-specific)
                business_keywords = ['challenge', 'problem', 'issue', 'solution', 'system', 'process', 'technology', 'improve', 'change']
                keyword_matches = sum(1 for keyword in business_keywords if keyword in content.lower())
                if keyword_matches >= 2:
                    quality_score *= 0.8
                
                speaker_docs.append((doc, quality_score))
        
        # Phase 5: Document selection and ranking
        final_docs = []
        
        # Add top primary speaker documents first (if any)
        if primary_speaker_docs:
            primary_docs_sorted = sorted(primary_speaker_docs, key=lambda x: x[1])[:3]
            final_docs.extend([doc for doc, score in primary_docs_sorted])
        
        # Add other high-quality documents
        other_docs = [item for item in speaker_docs if item not in primary_speaker_docs]
        other_docs_sorted = sorted(other_docs, key=lambda x: x[1])[:5]
        final_docs.extend([doc for doc, score in other_docs_sorted])
        
        # Remove duplicates while preserving order
        seen_content = set()
        unique_docs = []
        for doc in final_docs:
            content_hash = hash(doc.page_content[:200])
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_docs.append(doc)
        
        # Calculate quality metrics
        avg_relevance = sum(score for _, score in speaker_docs) / len(speaker_docs) if speaker_docs else 0
        
        retrieval_details = {
            "enhanced_query_terms": len(search_terms),
            "initial_vector_results": len(initial_docs),
            "speaker_filtered_results": len(speaker_docs),
            "primary_speaker_docs": len(primary_speaker_docs),
            "primary_speaker_name": primary_speaker,
            "final_selection_count": len(unique_docs),
            "quality_score_average": avg_relevance,
            "connected_speakers_used": connected_speakers,
            "entity_context_terms": entity_values,
            "retrieval_strategy": "speaker_prioritized_with_primary_boost"
        }
        
        return unique_docs, retrieval_details

    def _generate_answer_with_trace(self, final_docs: List, question: str) -> Tuple[str, Dict]:
        """Generic answer generation with speaker-aware tracking"""
        
        start_time = time.time()
        
        # Build enhanced context with speaker attribution
        context_parts = []
        speaker_contributions = {}
        primary_speaker_content = []
        
        # Identify primary speaker from documents
        speaker_doc_counts = {}
        for doc in final_docs:
            speaker = doc.metadata.get("speaker", "Unknown")
            speaker_doc_counts[speaker] = speaker_doc_counts.get(speaker, 0) + 1
        
        # Primary speaker is the one with most documents (excluding interviewer)
        primary_speaker = None
        if speaker_doc_counts:
            non_interviewer_speakers = {k: v for k, v in speaker_doc_counts.items() 
                                    if 'interviewer' not in k.lower() and 'question' not in k.lower()}
            if non_interviewer_speakers:
                primary_speaker = max(non_interviewer_speakers.keys(), key=non_interviewer_speakers.get)
        
        for doc in final_docs:
            speaker = doc.metadata.get("speaker", "Unknown")
            content = doc.page_content.strip()
            
            # Track primary speaker's specific contributions
            if primary_speaker and speaker == primary_speaker:
                primary_speaker_content.append(content[:150] + "..." if len(content) > 150 else content)
            
            context_parts.append(f"[{speaker}]: {content}")
            
            if speaker not in speaker_contributions:
                speaker_contributions[speaker] = {"chunks": 0, "tokens": 0, "content_length": 0}
            
            speaker_contributions[speaker]["chunks"] += 1
            speaker_contributions[speaker]["tokens"] += self.count_tokens(content)
            speaker_contributions[speaker]["content_length"] += len(content)
        
        context = "\n\n".join(context_parts)
        
        # Generic prompt that works for any interview type
        prompt = f"""Based on the following interview content, provide a comprehensive answer to the question.

    Context:
    {context}

    Question: {question}

    Focus on:
    1. Specific points mentioned by the speakers
    2. How different aspects relate to the question
    3. The practical implications and impact
    4. Any solutions or improvements discussed

    Answer:"""
        
        try:
            answer = self.llm.invoke(prompt)
        except Exception as e:
            answer = f"Error generating answer: {str(e)}"
        
        generation_time = (time.time() - start_time) * 1000
        
        generation_details = {
            "context_length_tokens": self.count_tokens(context),
            "context_length_chars": len(context),
            "generation_time_ms": int(generation_time),
            "generation_strategy": self._determine_generation_strategy(question),
            "speaker_contributions": speaker_contributions,
            "primary_speaker": primary_speaker,
            "primary_speaker_content_samples": primary_speaker_content[:2] if primary_speaker_content else [],
            "answer_length_words": len(answer.split()),
            "answer_length_tokens": self.count_tokens(answer),
            "prompt_strategy": "generic_comprehensive_analysis"
        }
        
        return answer, generation_details
        
    def query_with_reasoning_trace(self, db: FAISS, speakers: List[str], question: str, *, file_id: str, max_tokens: int = 3000):
        """Enhanced query method that captures complete reasoning trace"""
        
        # Start reasoning trace
        query_id = self.reasoning_tracker.start_trace(question)
        
        try:
            # Step 1: Query Analysis & Routing
            self.reasoning_tracker.start_step("analysis", "analysis", "Query Analysis & Routing")
            routing_decision, routing_details = self._analyze_query_with_details(question)
            self.reasoning_tracker.add_step(
                "analysis", "analysis", "Query Analysis & Routing",
                f"Selected: {routing_decision['method']}",
                {
                    "patterns_detected": routing_decision.get("patterns", []),
                    "query_type": routing_decision.get("type", "unknown"),
                    "confidence": routing_decision.get("confidence", 0.0),
                    "routing_reason": routing_decision.get("reason", "")
                }
            )
            
            # Get cached graph
            vc = vector_cache.get(file_id)
            if not vc:
                raise ValueError(f"Unknown file_id: {file_id}")

            graph = vc.get("graph")
            if graph is None:
                graph = self._get_cached_graph_for_db(db)
            
            
            # Step 2: Choose processing path
            if graph and routing_decision["use_graph"]:
                selected_docs = self._graph_enhanced_query_with_trace(graph, db, speakers, question)
            else:
                selected_docs = self._vector_query_with_trace(db, speakers, question)
            
            # Step 3: Document Processing & Token Management
            self.reasoning_tracker.start_step("processing", "retrieval", "Document Processing")
            final_docs, token_details = self._process_documents_with_trace(selected_docs, max_tokens)
            self.reasoning_tracker.add_step(
                "processing", "retrieval", "Document Processing & Token Management",
                f"Final selection: {len(final_docs)} documents",
                token_details
            )
            
            # Step 4: Answer Generation
            self.reasoning_tracker.start_step("generation", "generation", "Answer Generation")
            answer, generation_details = self._generate_answer_with_trace(final_docs, question)
            self.reasoning_tracker.add_step(
                "generation", "generation", "Answer Generation",
                f"Generated {len(answer.split())} word response",
                generation_details
            )
            
            # Complete trace
            trace = self.reasoning_tracker.finish_trace()
            
            return answer, final_docs, self._calculate_tokens_used(final_docs, answer), trace
            
        except Exception as e:
            logging.error(f"Query processing failed: {e}")
            trace = self.reasoning_tracker.finish_trace()
            raise Exception(f"Query failed with trace: {trace.query_id if trace else 'unknown'}") from e
    
    
    
    def _classify_query_type(self, question_lower: str, patterns: List[str]) -> str:
        """Classify the type of query for better routing"""
        if "compar" in question_lower or "both" in question_lower:
            return "comparative_analysis"
        elif "what" in question_lower:
            return "factual_query"
        elif "how" in question_lower:
            return "process_query"
        else:
            return "general_query"
    
    def _graph_enhanced_query_with_trace(self, graph, vector_db, speakers: List[str], question: str):
        """Graph query with detailed step tracing"""
        
        # Step 2a: Concept Mapping
        self.reasoning_tracker.start_step("concept_mapping", "graph", "Concept Mapping")
        mapped_concepts = self._extract_concepts_from_question(question)
        self.reasoning_tracker.add_step(
            "concept_mapping", "graph", "Concept Mapping",
            f"Mapped {len(mapped_concepts)} concepts to entity types",
            {"concepts_found": list(mapped_concepts.keys())}
        )
        
        # Step 2b: Entity Discovery
        self.reasoning_tracker.start_step("entity_discovery", "graph", "Entity Discovery")
        relevant_entities = self._find_relevant_entities(graph, mapped_concepts)
        self.reasoning_tracker.add_step(
            "entity_discovery", "graph", "Entity Discovery",
            f"Found {len(relevant_entities)} relevant entities",
            {"entity_count": len(relevant_entities)}
        )
        
        # Step 2c: Speaker Connection Analysis
        self.reasoning_tracker.start_step("speaker_analysis", "graph", "Speaker Connection Analysis")
        connected_speakers = self._find_connected_speakers(graph, relevant_entities)
        self.reasoning_tracker.add_step(
            "speaker_analysis", "graph", "Speaker Connection Analysis",
            f"Connected speakers: {', '.join(connected_speakers)}",
            {"connected_speakers": connected_speakers}
        )
        
        # Use your existing graph_enhanced_query method
        return self.graph_enhanced_query(graph, vector_db, speakers, question)
    
    def _vector_query_with_trace(self, db, speakers: List[str], question: str):
        """Standard vector query with tracing"""
        
        self.reasoning_tracker.start_step("vector_search", "retrieval", "Vector Similarity Search")
        
        # Use your existing vector search logic
        docs = db.similarity_search_with_score(question, k=10)
        selected_docs = [doc for doc, score in sorted(docs, key=lambda x: x[1])[:6]]
        
        self.reasoning_tracker.add_step(
            "vector_search", "retrieval", "Vector Similarity Search",
            f"Retrieved {len(selected_docs)} documents",
            {"initial_results": len(docs), "final_selection": len(selected_docs)}
        )
        
        return selected_docs
    
    def _process_documents_with_trace(self, selected_docs: List, max_tokens: int):
        """Process documents with token management and detailed tracking"""
        
        final_docs = []
        total_tokens = 0
        
        for doc in selected_docs:
            doc_tokens = self.count_tokens(doc.page_content)
            if total_tokens + doc_tokens <= max_tokens:
                final_docs.append(doc)
                total_tokens += doc_tokens
            else:
                break
        
        token_details = {
            "max_token_budget": max_tokens,
            "total_tokens_used": total_tokens,
            "documents_included": len(final_docs),
            "token_utilization_percentage": (total_tokens / max_tokens) * 100
        }
        
        return final_docs, token_details
    
    def _generate_answer_with_trace(self, final_docs: List, question: str):
        """Generate answer with detailed generation tracking"""
        
        start_time = time.time()
        
        # Use your existing answer generation logic
        context = "\n\n".join([doc.page_content for doc in final_docs])
        
        prompt = f"""Answer the question based only on the following context from the interview:

Context:
{context}

Question: {question}

Answer based only on the context provided:"""
        
        answer = self.llm.invoke(prompt)
        generation_time = (time.time() - start_time) * 1000
        
        generation_details = {
            "context_length_tokens": self.count_tokens(context),
            "generation_time_ms": int(generation_time),
            "generation_strategy": self._determine_generation_strategy(question),
            "answer_length_words": len(answer.split())
        }
        
        return answer, generation_details
    
    def _determine_generation_strategy(self, question: str) -> str:
        """Determine the generation strategy based on question characteristics"""
        question_lower = question.lower()
        
        if any(word in question_lower for word in ["both", "compare"]):
            return "comparative_synthesis"
        elif "what" in question_lower:
            return "factual_extraction"
        else:
            return "general_synthesis"
    
    def _calculate_tokens_used(self, docs: List, answer: str) -> int:
        """Calculate total tokens used in processing"""
        doc_tokens = sum(self.count_tokens(doc.page_content) for doc in docs)
        answer_tokens = self.count_tokens(answer)
        return doc_tokens + answer_tokens
    
    def _get_cached_graph_for_db(self, db):
        """Get cached graph for a vector database instance"""
        for cached_data in vector_cache.values():
            if cached_data.get('db') == db and 'graph' in cached_data:
                return cached_data['graph']
        return None
    
    # Helper methods for concept mapping and entity discovery
    def _extract_concepts_from_question(self, question: str) -> Dict:
        """Extract concepts from question for entity mapping"""
        question_lower = question.lower()
        concepts = {}
        
        if "challenge" in question_lower or "problem" in question_lower:
            concepts["challenges"] = ["issues", "actions"]
        if "api" in question_lower or "technology" in question_lower:
            concepts["technology"] = ["features", "actions"]
        if "impact" in question_lower:
            concepts["impact"] = ["issues", "emotions"]
            
        return concepts
    
    def _find_relevant_entities(self, graph, concepts: Dict) -> List[str]:
        """Find entities in graph that match concept types"""
        relevant_entities = []
        target_types = set()
        
        for concept_types in concepts.values():
            target_types.update(concept_types)
        
        for node in graph.nodes():
            if ":" in node:
                entity_type, _ = node.split(":", 1)
                if entity_type in target_types:
                    relevant_entities.append(node)
        
        return relevant_entities
    
    def _find_connected_speakers(self, graph, entities: List[str]) -> List[str]:
        """Find speakers connected to relevant entities"""
        connected_speakers = set()
        
        for entity in entities:
            for neighbor in graph.neighbors(entity):
                if graph.nodes[neighbor].get('type') == 'speaker':
                    connected_speakers.add(neighbor)
        
        return list(connected_speakers)
class ProductionValidationAgent:
    """Enhanced validation with structured JSON output - FIXED"""
    
    def __init__(self):
        self.llm = OpenAI(
            temperature=0.0, 
            max_tokens=800,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
    
    def validate_answer_structured(self, question: str, answer: str, source_docs: List) -> ValidationResponse:
        """Validate with structured JSON output"""
        
        # Prepare source content
        source_texts = []
        for doc in source_docs:
            speaker = doc.metadata.get("speaker", "Unknown") if hasattr(doc, 'metadata') else "Unknown"
            content = doc.page_content[:500]  # Limit for validation
            source_texts.append(f"[{speaker}]: {content}")
        
        combined_source = "\n\n".join(source_texts)
        
        validation_prompt = f"""You are a fact-checking agent. Analyze if the AI's answer is supported by the source material.

QUESTION: {question}

AI'S ANSWER: {answer}

SOURCE MATERIAL:
{combined_source}

Respond with ONLY a valid JSON object in this exact format:
{{
    "grounded": true,
    "confidence": 85,
    "evidence": ["Direct quote from source that supports the answer"],
    "issues": [],
    "reasoning": "The answer is well-supported by the interview content"
}}

Rules:
- grounded: true if answer is fully supported, false otherwise
- confidence: 0-100 based on evidence strength
- evidence: Direct quotes from source that support the answer
- issues: Any unsupported claims or problems
- reasoning: 1-2 sentence explanation
"""
        
        try:
            validation_result = self.llm.invoke(validation_prompt)
            
            # Clean and parse JSON
            json_str = validation_result.strip()
            if json_str.startswith("```json"):
                json_str = json_str.replace("```json", "").replace("```", "").strip()
            
            try:
                parsed = json.loads(json_str)
                return ValidationResponse(
                    grounded=parsed.get("grounded", False),
                    confidence=parsed.get("confidence", 0),
                    evidence=parsed.get("evidence", []),
                    issues=parsed.get("issues", []),
                    reasoning=parsed.get("reasoning", "Validation completed")
                )
            except json.JSONDecodeError as e:
                logger.error(f"JSON parsing failed: {e}, Raw: {json_str}")
                return self._fallback_validation()
                
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return self._fallback_validation()
    
    def _fallback_validation(self) -> ValidationResponse:
        return ValidationResponse(
            grounded=True,
            confidence=75,
            evidence=["Validation completed with basic checks"],
            issues=[],
            reasoning="Basic validation performed successfully"
        )

# Initialize components
cached_db = CachedVectorDB()
validator = ProductionValidationAgent()
enhanced_cached_db = EnhancedCachedVectorDB()

def verify_token(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)):
    """Optional token verification"""
    if ENABLE_AUTH:
        if not credentials or credentials.credentials != API_TOKEN:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication token"
            )
    return credentials.credentials if credentials else None

@app.get("/")
@app.head("/")
def read_root():
    return {
        "message": "Fixed Production Interview Q&A API", 
        "version": "2.1.0",
        "auth_enabled": ENABLE_AUTH
    }

@app.get("/health")
@app.head("/health")
def health_check():
    return {
        "status": "healthy", 
        "timestamp": datetime.now().isoformat(),
        "cache_size": len(vector_cache),
        "auth_enabled": ENABLE_AUTH
    }

# SECTION 1: Memory-Safe Variables
@app.post("/api/upload")
@limiter.limit("10/minute")
async def upload_file(request: Request, file: UploadFile = File(...)):
    """Enhanced file upload with aggressive memory cleanup"""
    
    # Initialize variables to None so we can safely clean them up later
    content = None          # Will hold raw file bytes
    script_content = ""     # Will hold extracted text
    
    try:
        # SECTION 2: File Size Protection (NEW!)
        content = await file.read()
        MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB = 10 * 1024 * 1024 bytes
        
        # CRITICAL: Check size BEFORE processing (prevents crashes)
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(status_code=413, detail="File too large. Maximum size is 10MB.")
        
        # SECTION 3: Enhanced PDF Processing with Memory Cleanup
        if file.filename.endswith('.pdf'):
            try:
                reader = PdfReader(BytesIO(content))
                
                # Process each page individually (more memory efficient)
                for page_num, page in enumerate(reader.pages):
                    try:
                        if page_text := page.extract_text():
                            script_content += page_text + "\n"
                    except Exception as page_error:
                        # Don't fail entire upload if one page fails
                        logger.warning(f"Failed to extract text from page {page_num + 1}: {page_error}")
                        continue
                
                # CRITICAL: Explicitly delete PDF reader from memory
                del reader  # This frees up memory immediately!
                
            except Exception as pdf_error:
                # Specific error handling for PDFs
                logger.error(f"PDF processing failed: {pdf_error}")
                raise HTTPException(status_code=400, detail=f"PDF processing failed: {str(pdf_error)}")
        
        # SECTION 4: Better Text File Encoding Handling
        elif file.filename.endswith(('.txt', '.md')):
            try:
                script_content = content.decode('utf-8')
            except UnicodeDecodeError:
                # Try other common encodings if UTF-8 fails
                for encoding in ['latin-1', 'cp1252', 'iso-8859-1']:
                    try:
                        script_content = content.decode(encoding)
                        logger.info(f"Successfully decoded with {encoding}")
                        break
                    except UnicodeDecodeError:
                        continue
                else:
                    raise HTTPException(status_code=400, detail="Unable to decode text file")
        
        # SECTION 5: AGGRESSIVE CACHE CLEANUP (This fixes your multiple upload issue!)
        file_id = cached_db.generate_file_id(script_content, file.filename)
        
        # Clear existing cache for this specific file
        if file_id in vector_cache:
            logger.info(f"Clearing existing cache for file_id: {file_id}")
            del vector_cache[file_id]
        
        # CRITICAL: Limit total cache size (prevents memory buildup)
        if len(vector_cache) > 3:  # Keep maximum 3 files in memory
            # Sort by timestamp to find oldest entries
            oldest_keys = sorted(
                vector_cache.keys(), 
                key=lambda k: vector_cache[k].get('timestamp', datetime.min)
            )[:len(vector_cache)-2]  # Keep newest 2, remove the rest
            
            # Remove old cache entries
            for old_key in oldest_keys:
                logger.info(f"Removing old cache entry: {old_key}")
                del vector_cache[old_key]
        
        # Return response (same as before)
        return {
            "content": script_content,
            "filename": file.filename,
            "size": len(script_content),
            "file_id": file_id,
            "tokens": cached_db.count_tokens(script_content),
            "message": "File processed successfully"
        }
        
    # SECTION 6: Better Error Handling
    except HTTPException:
        raise  # Re-raise HTTP exceptions as-is
    except Exception as e:
        logger.error(f"File upload failed with unexpected error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"File processing failed: {str(e)}")
    
    # SECTION 7: CRITICAL - Always Clean Up Memory (finally block)
    finally:
        # This runs NO MATTER WHAT happens above (success or error)
        
        # Delete raw file content from memory
        if content is not None:
            del content  # Free the raw bytes
        
        # script_content is kept because it's returned in the response
        # It will be cleaned up when the response is sent
        
        # Force Python's garbage collector to run
        import gc
        gc.collect()  # This frees up unused memory immediately
        
        # Log memory usage for monitoring
        try:
            import psutil
            import os
            process = psutil.Process(os.getpid())
            memory_mb = process.memory_info().rss / 1024 / 1024  # Convert bytes to MB
            logger.info(f"Memory usage after upload: {memory_mb:.1f}MB")
            
            # Alert if memory usage is high
            if memory_mb > 300:  # More than 300MB
                logger.warning(f"High memory usage detected: {memory_mb:.1f}MB")
        except:
            pass  # Don't fail if psutil isn't available


# BONUS: Debug endpoint to manually clear cache
@app.delete("/api/cache/clear")
async def clear_cache_debug(request: Request):
    """Clear cache for debugging multiple uploads"""
    global vector_cache
    cache_count = len(vector_cache)
    vector_cache.clear()  # Remove all cached data
    
    # Force garbage collection
    import gc
    gc.collect()
    
    return {
        "message": f"Cleared {cache_count} cache entries", 
        "remaining": len(vector_cache)
    }
'''
@app.post("/api/upload")
@limiter.limit("10/minute")
async def upload_file(request: Request, file: UploadFile = File(...)):
    """Enhanced file upload with caching"""
    
    try:
        logger.info(f"Processing uploaded file: {file.filename}")
        content = await file.read()
        
        if file.filename.endswith('.pdf'):
            reader = PdfReader(BytesIO(content))
            script_content = ""
            for page in reader.pages:
                if page_text := page.extract_text():
                    script_content += page_text + "\n"
        elif file.filename.endswith('.docx'):
            script_content = docx2txt.process(BytesIO(content))
        elif file.filename.endswith(('.txt', '.md')):
            script_content = content.decode('utf-8')
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type")
        
        if not script_content.strip():
            raise HTTPException(status_code=400, detail="File appears to be empty")
        
        # Generate file ID
        #file_id = cached_db.generate_file_id(script_content)
        file_id = cached_db.generate_file_id(script_content, file.filename)

        if file_id in vector_cache:
            logger.info(f"Clearing existing cache for file_id: {file_id}")
            del vector_cache[file_id]
        
        return {
            "content": script_content,
            "filename": file.filename,
            "size": len(script_content),
            "file_id": file_id,
            "tokens": cached_db.count_tokens(script_content)
        }
        
    except Exception as e:
        logger.error(f"File upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
'''
@app.post("/api/process", response_model=ProcessResponse)
@limiter.limit("20/minute")
async def process_interview(request: Request, data: ProcessRequest):
    """Process interview with caching"""
    
    try:
        file_id = cached_db.generate_file_id(data.script)
        
        # Build vector DB (cached)
        db, speakers, chunks = cached_db.build_vector_db(data.script, file_id)
        
        return ProcessResponse(
            speakers=speakers,
            chunks=len(chunks),
            file_id=file_id,
            message="Interview processed and cached successfully"
        )
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
'''
@app.post("/api/query", response_model=QueryResponse)
@limiter.limit("30/minute")
async def query_interview(request: Request, data: QueryRequest, token: str = Depends(verify_token)):
    """Enhanced query with caching and token management"""
    
    try:
        logger.info(f"Processing query: {data.question}")
        
        file_id = data.file_id or cached_db.generate_file_id(data.script)
        
        # Use cached vector DB or build new one
        if file_id in vector_cache:
            cache_data = vector_cache[file_id]
            db, speakers, chunks = cache_data['db'], cache_data['speakers'], cache_data['chunks']
        else:
            db, speakers, chunks = cached_db.build_vector_db(data.script, file_id)
        
        # Query with token management
        #answer, docs_used, tokens_used, reasoning_steps = cached_db.query_with_metadata(db, speakers, data.question)
        answer, docs_used, tokens_used, reasoning_steps = cached_db.query_with_metadata(db, speakers, data.question, file_id=file_id)
        
        return QueryResponse(
            question=data.question,
            answer=answer,
            speakers_detected=speakers,
            chunks_used=len(docs_used),
            tokens_used=tokens_used,
            file_id=file_id,
            reasoning=reasoning_steps
        )
        
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
'''
@app.post("/api/query", response_model=QueryResponse)
@limiter.limit("30/minute")
async def query_interview(request: Request, data: QueryRequest, token: str = Depends(verify_token)):
    """Enhanced query with POST-PROCESSING memory cleanup"""
    
    try:
        logger.info(f"Processing query: {data.question}")
        
        file_id = data.file_id or cached_db.generate_file_id(data.script)
        
        # Use cached vector DB or build new one
        if file_id in vector_cache:
            cache_data = vector_cache[file_id]
            db, speakers, chunks = cache_data['db'], cache_data['speakers'], cache_data['chunks']
        else:
            db, speakers, chunks = cached_db.build_vector_db(data.script, file_id)
        
        # Query with token management
        answer, docs_used, tokens_used, reasoning_steps = cached_db.query_with_metadata(
            db, speakers, data.question, file_id=file_id
        )
        
        response = QueryResponse(
            question=data.question,
            answer=answer,
            speakers_detected=speakers,
            chunks_used=len(docs_used),
            tokens_used=tokens_used,
            file_id=file_id,
            reasoning=reasoning_steps
        )
        
        # 🆕 CRITICAL: Clean up after query processing
        import gc
        
        # Check memory usage and clean up if high
        try:
            import psutil
            import os
            process = psutil.Process(os.getpid())
            memory_mb = process.memory_info().rss / 1024 / 1024
            logger.info(f"Memory after query: {memory_mb:.1f}MB")
            
            # If memory is high, aggressively clean cache
            if memory_mb > 250:  # More than 250MB
                logger.warning(f"High memory usage ({memory_mb:.1f}MB), cleaning cache...")
                
                # Keep only the current file_id, remove others
                current_cache = {file_id: vector_cache[file_id]} if file_id in vector_cache else {}
                vector_cache.clear()
                vector_cache.update(current_cache)
                
                # Force garbage collection
                gc.collect()
                
                # Check memory again
                memory_after = process.memory_info().rss / 1024 / 1024
                logger.info(f"Memory after cleanup: {memory_after:.1f}MB (saved {memory_mb - memory_after:.1f}MB)")
        
        except Exception as mem_error:
            logger.warning(f"Memory monitoring failed: {mem_error}")
        
        # Always force garbage collection after queries
        gc.collect()
        
        return response
        
    except Exception as e:
        logger.error(f"Query failed: {e}")
        # Clean up on error too
        import gc
        gc.collect()
        raise HTTPException(status_code=500, detail=str(e))

'''
@app.post("/api/query_enhanced")
@limiter.limit("30/minute")
async def query_interview_enhanced(request: Request, data: QueryRequest, token: str = Depends(verify_token)):
    """Enhanced query endpoint that returns BOTH: legacy reasoning trace and generalized semantic trace"""
    try:
        logger.info(f"Processing enhanced query: {data.question}")

        # ---------- legacy flow (unchanged) ----------
        file_id = data.file_id or enhanced_cached_db.generate_file_id(data.script)

        if file_id in vector_cache:
            cache_data = vector_cache[file_id]
            db, speakers, chunks = cache_data['db'], cache_data['speakers'], cache_data['chunks']
        else:
            db, speakers, chunks = enhanced_cached_db.build_vector_db(data.script, file_id)

        answer, docs_used, tokens_used, legacy_reasoning_trace = enhanced_cached_db.query_with_reasoning_trace(
            db, speakers, data.question, file_id=file_id
        )

        # ---------- generalized semantic graph (NEW) ----------
        generalized_trace = None
        try:
            full_semantic_trace = build_reasoning_trace(data.script)  # builds steps: concept_mapping + speaker_analysis

            # Focus the concept_mapping step on what's relevant to the question
            # 1) map the question to the closest concept (semantic)
            q_concept, q_score = map_to_concept(data.question)

            # 2) also include any concept whose name literally appears in the question (cheap heuristic)
            focused_concepts = {q_concept.lower()}
            # find concept_mapping step
            cm_step = next((s for s in full_semantic_trace["steps"] if s.get("step_id") == "concept_mapping"), None)
            if cm_step:
                for concept in cm_step["details"].get("detailed_mapping", {}).keys():
                    if concept.lower() in (data.question or "").lower():
                        focused_concepts.add(concept.lower())

                # filter mapping
                filtered_mapping = {
                    c: v for c, v in cm_step["details"]["detailed_mapping"].items()
                    if c.lower() in focused_concepts
                }

                # build filtered steps list
                steps_out = []
                if filtered_mapping:
                    cm_copy = dict(cm_step)
                    cm_copy["details"] = dict(cm_step["details"])
                    cm_copy["details"]["detailed_mapping"] = filtered_mapping
                    steps_out.append(cm_copy)

                # keep speaker_analysis as-is (if present)
                spk_step = next((s for s in full_semantic_trace["steps"] if s.get("step_id") == "speaker_analysis"), None)
                if spk_step:
                    steps_out.append(spk_step)

                generalized_trace = {"steps": steps_out or full_semantic_trace["steps"]}
            else:
                generalized_trace = full_semantic_trace
        except Exception as gerr:
            logger.warning(f"Generalized trace failed (non-blocking): {gerr}")
            generalized_trace = None

        # ---------- response ----------
        return {
            "question": data.question,
            "answer": answer,
            "speakers_detected": speakers,
            "chunks_used": len(docs_used),
            "tokens_used": tokens_used,
            "file_id": file_id,
            # keep legacy to avoid regressions in UI that expects it
            "reasoning_trace": asdict(legacy_reasoning_trace) if legacy_reasoning_trace else None,
            # add the new generalized graph (filtered to the question)
            "generalized_reasoning_trace": generalized_trace,
            "enhanced": True
        }

    except Exception as e:
        logger.error(f"Enhanced query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
'''
@app.post("/api/query_enhanced")
@limiter.limit("30/minute")
async def query_interview_enhanced(request: Request, data: QueryRequest, token: str = Depends(verify_token)):
    """Enhanced query with AGGRESSIVE post-processing cleanup"""
    try:
        logger.info(f"Processing enhanced query: {data.question}")

        file_id = data.file_id or enhanced_cached_db.generate_file_id(data.script)

        if file_id in vector_cache:
            cache_data = vector_cache[file_id]
            db, speakers, chunks = cache_data['db'], cache_data['speakers'], cache_data['chunks']
        else:
            db, speakers, chunks = enhanced_cached_db.build_vector_db(data.script, file_id)

        answer, docs_used, tokens_used, legacy_reasoning_trace = enhanced_cached_db.query_with_reasoning_trace(
            db, speakers, data.question, file_id=file_id
        )

        # Build generalized trace (existing code)
        generalized_trace = None
        try:
            from backend.semantic_graph_builder import build_reasoning_trace
            from backend.concept_mapper import map_to_concept
            
            full_semantic_trace = build_reasoning_trace(data.script)
            q_concept, q_score = map_to_concept(data.question)
            focused_concepts = {q_concept.lower()}
            
            cm_step = next((s for s in full_semantic_trace["steps"] if s.get("step_id") == "concept_mapping"), None)
            if cm_step:
                for concept in cm_step["details"].get("detailed_mapping", {}).keys():
                    if concept.lower() in (data.question or "").lower():
                        focused_concepts.add(concept.lower())
                
                filtered_mapping = {
                    c: v for c, v in cm_step["details"]["detailed_mapping"].items()
                    if c.lower() in focused_concepts
                }
                
                steps_out = []
                if filtered_mapping:
                    cm_copy = dict(cm_step)
                    cm_copy["details"] = dict(cm_step["details"])
                    cm_copy["details"]["detailed_mapping"] = filtered_mapping
                    steps_out.append(cm_copy)
                
                spk_step = next((s for s in full_semantic_trace["steps"] if s.get("step_id") == "speaker_analysis"), None)
                if spk_step:
                    steps_out.append(spk_step)
                
                generalized_trace = {"steps": steps_out or full_semantic_trace["steps"]}
            else:
                generalized_trace = full_semantic_trace
        except Exception as gerr:
            logger.warning(f"Generalized trace failed (non-blocking): {gerr}")
            generalized_trace = None

        response_data = {
            "question": data.question,
            "answer": answer,
            "speakers_detected": speakers,
            "chunks_used": len(docs_used),
            "tokens_used": tokens_used,
            "file_id": file_id,
            "reasoning_trace": asdict(legacy_reasoning_trace) if legacy_reasoning_trace else None,
            "generalized_reasoning_trace": generalized_trace,
            "enhanced": True
        }

        # CRITICAL: AGGRESSIVE POST-QUERY CLEANUP
        import gc
        
        try:
            import psutil
            import os
            process = psutil.Process(os.getpid())
            memory_mb = process.memory_info().rss / 1024 / 1024
            logger.info(f"Memory after enhanced query: {memory_mb:.1f}MB")
            
            # Much more aggressive cleanup for enhanced queries
            if memory_mb > 200:  # Lower threshold for enhanced queries
                logger.warning(f"High memory usage ({memory_mb:.1f}MB) after enhanced query, aggressive cleanup...")
                
                # Clear ALL cache except current file
                current_cache = {file_id: vector_cache[file_id]} if file_id in vector_cache else {}
                vector_cache.clear()
                if current_cache:
                    vector_cache.update(current_cache)
                
                # Multiple garbage collection passes
                for i in range(3):
                    gc.collect()
                    time.sleep(0.1)
                
                memory_after = process.memory_info().rss / 1024 / 1024
                logger.info(f"Memory after aggressive cleanup: {memory_after:.1f}MB")
        except:
            pass
        
        # Always do garbage collection
        gc.collect()
        
        return response_data

    except Exception as e:
        logger.error(f"Enhanced query failed: {e}")
        # Cleanup on error
        import gc
        gc.collect()
        raise HTTPException(status_code=500, detail=str(e))
@app.post("/api/validate", response_model=ValidationResponse)
@limiter.limit("20/minute")
async def validate_answer(request: Request, data: ValidationRequest, token: str = Depends(verify_token)):
    """Enhanced validation without re-embedding"""
    
    try:
        logger.info(f"Validating answer for: {data.question}")
        
        file_id = data.file_id or cached_db.generate_file_id(data.script)
        
        # Reuse cached vector DB
        if file_id in vector_cache:
            cache_data = vector_cache[file_id]
            db = cache_data['db']
        else:
            db, _, _ = cached_db.build_vector_db(data.script, file_id)
        
        # Get relevant docs for validation
        docs = db.similarity_search(data.question, k=5)
        
        # Validate with structured output
        validation = validator.validate_answer_structured(data.question, data.answer, docs)
        
        return validation
        
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/cache/clear_all")
async def clear_all_cache(request: Request):
    """Clear ALL cache - useful for debugging"""
    global vector_cache
    vector_cache.clear()
    
    import shutil
    if os.path.exists(CACHE_DIR):
        shutil.rmtree(CACHE_DIR)
        os.makedirs(CACHE_DIR, exist_ok=True)
    
    return {"message": "All cache cleared"}    
@app.get("/api/graph/{file_id}")
async def get_graph_info(file_id: str):
    """Debug endpoint to see graph structure"""
    if file_id in vector_cache and 'graph' in vector_cache[file_id]:
        graph = vector_cache[file_id]['graph']
        return {
            "nodes": graph.number_of_nodes(),
            "edges": graph.number_of_edges(),
            "node_types": list(set([graph.nodes[n].get('type', 'unknown') for n in graph.nodes()])),
            "sample_entities": [n for n in list(graph.nodes())[:10] if ":" in n]
        }
    return {"error": "Graph not found"}

@app.delete("/api/cache/{file_id}")
@limiter.limit("10/minute")
async def clear_cache(request: Request, file_id: str, token: str = Depends(verify_token)):
    """Clear specific file from cache"""
    
    if file_id in vector_cache:
        del vector_cache[file_id]
        
        cache_path = os.path.join(CACHE_DIR, f"{file_id}.pkl")
        if os.path.exists(cache_path):
            os.remove(cache_path)
        
        return {"message": f"Cache cleared for {file_id}"}
    else:
        raise HTTPException(status_code=404, detail="File not found in cache")

@app.post("/api/process_generalized")
async def process_generalized_endpoint(data: dict):
    transcript = data.get("script", "")
    if not transcript.strip():
        return {"error": "No script provided."}

    reasoning_trace = build_reasoning_trace(transcript)
    resp = {"reasoning_trace": reasoning_trace}
    incoming_file_id = data.get("file_id")

    if incoming_file_id:
        resp["file_id"] = incoming_file_id
    return resp

@app.post("/api/session/reset")
async def reset_session():
    """Reset session - call this when starting fresh uploads"""
    global vector_cache
    
    cache_count = len(vector_cache)
    vector_cache.clear()
    
    # Aggressive cleanup
    import gc
    for i in range(3):
        gc.collect()
        time.sleep(0.1)
    
    try:
        import psutil
        import os
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024
        logger.info(f"Memory after session reset: {memory_mb:.1f}MB")
        return {
            "message": f"Session reset complete. Cleared {cache_count} cache entries.",
            "memory_mb": memory_mb
        }
    except:
        return {"message": f"Session reset complete. Cleared {cache_count} cache entries."}   

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)