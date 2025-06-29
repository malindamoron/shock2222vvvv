"""
Shock2 Natural Language Understanding Engine
Advanced NLU with Shock2 command parsing and intent recognition
"""

import asyncio
import logging
import re
import json
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import spacy
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch

logger = logging.getLogger(__name__)

class Shock2Intent(Enum):
    """Shock2 system intents"""
    # Core System Commands
    SYSTEM_STATUS = "system_status"
    SYSTEM_START = "system_start"
    SYSTEM_STOP = "system_stop"
    SYSTEM_RESTART = "system_restart"
    
    # Neural Operations
    NEURAL_INITIALIZE = "neural_initialize"
    NEURAL_CHAOS_MODE = "neural_chaos_mode"
    NEURAL_QUANTUM_SYNC = "neural_quantum_sync"
    NEURAL_DEEP_LEARN = "neural_deep_learn"
    
    # Stealth Operations
    STEALTH_ACTIVATE = "stealth_activate"
    STEALTH_STATUS = "stealth_status"
    STEALTH_GHOST_MODE = "stealth_ghost_mode"
    STEALTH_EVASION = "stealth_evasion"
    
    # Intelligence Gathering
    INTEL_SCAN = "intel_scan"
    INTEL_MULTI_SOURCE = "intel_multi_source"
    INTEL_TREND_DETECT = "intel_trend_detect"
    INTEL_SENTIMENT_MAP = "intel_sentiment_map"
    
    # Content Generation
    GENERATE_BREAKING = "generate_breaking"
    GENERATE_ANALYSIS = "generate_analysis"
    GENERATE_OPINION = "generate_opinion"
    GENERATE_SUMMARY = "generate_summary"
    
    # Autonomous Operations
    AUTO_SELF_DIRECT = "auto_self_direct"
    AUTO_PREDICT = "auto_predict"
    AUTO_HUNT_NEWS = "auto_hunt_news"
    AUTO_BREAK_FIRST = "auto_break_first"
    
    # Performance & Monitoring
    PERFORMANCE_CHECK = "performance_check"
    PERFORMANCE_OPTIMIZE = "performance_optimize"
    MONITORING_STATUS = "monitoring_status"
    
    # Emergency & Priority
    EMERGENCY_RESPONSE = "emergency_response"
    PRIORITY_ALERT = "priority_alert"
    
    # Conversation
    CONVERSATION = "conversation"
    UNKNOWN = "unknown"

@dataclass
class Shock2Entity:
    """Extracted entity from user input"""
    text: str
    label: str
    confidence: float
    start: int
    end: int
    value: Any = None

@dataclass
class Shock2Command:
    """Parsed Shock2 command"""
    intent: Shock2Intent
    confidence: float
    entities: List[Shock2Entity]
    parameters: Dict[str, Any]
    raw_text: str
    urgency: float
    sentiment: float
    module_path: Optional[str] = None
    function_name: Optional[str] = None

class Shock2NLUEngine:
    """Advanced NLU engine for Shock2 command parsing"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.nlp = None
        self.intent_classifier = None
        self.sentiment_analyzer = None
        
        # Command patterns for fast matching
        self.command_patterns = self._build_command_patterns()
        
        # Module mapping for command execution
        self.module_mapping = self._build_module_mapping()
        
        # Entity extractors
        self.entity_patterns = self._build_entity_patterns()
        
        # Performance tracking
        self.nlu_stats = {
            'total_processed': 0,
            'successful_parses': 0,
            'intent_accuracy': 0.0,
            'avg_processing_time': 0.0
        }
    
    async def initialize(self):
        """Initialize NLU components"""
        logger.info("🧠 Initializing Shock2 NLU Engine...")
        
        # Load spaCy model
        try:
            model_name = self.config.get('spacy_model', 'en_core_web_lg')
            self.nlp = spacy.load(model_name)
            logger.info(f"✅ spaCy model ({model_name}) loaded")
        except OSError:
            logger.warning("Large spaCy model not found, trying medium...")
            try:
                self.nlp = spacy.load('en_core_web_md')
                logger.info("✅ spaCy medium model loaded")
            except OSError:
                logger.warning("Medium spaCy model not found, using small...")
                self.nlp = spacy.load('en_core_web_sm')
                logger.info("✅ spaCy small model loaded")
        
        # Load intent classifier
        try:
            self.intent_classifier = pipeline(
                "zero-shot-classification",
                model=self.config.get('intent_model', 'facebook/bart-large-mnli'),
                device=0 if torch.cuda.is_available() else -1
            )
            logger.info("✅ Intent classifier loaded")
        except Exception as e:
            logger.error(f"Failed to load intent classifier: {e}")
        
        # Load sentiment analyzer
        try:
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model=self.config.get('sentiment_model', 'cardiffnlp/twitter-roberta-base-sentiment-latest'),
                device=0 if torch.cuda.is_available() else -1
            )
            logger.info("✅ Sentiment analyzer loaded")
        except Exception as e:
            logger.error(f"Failed to load sentiment analyzer: {e}")
        
        logger.info("✅ Shock2 NLU Engine initialized")
    
    def _build_command_patterns(self) -> Dict[Shock2Intent, List[re.Pattern]]:
        """Build regex patterns for fast command matching"""
        patterns = {
            # System Commands
            Shock2Intent.SYSTEM_STATUS: [
                re.compile(r'\b(?:show|display|check|get)\s+(?:system\s+)?status\b', re.I),
                re.compile(r'\b(?:how|what)(?:\s+is|\s+are)?\s+(?:the\s+)?system(?:\s+doing|\s+status)?\b', re.I),
                re.compile(r'\bhealth\s+check\b', re.I),
                re.compile(r'\bsystem\s+report\b', re.I)
            ],
            
            Shock2Intent.SYSTEM_START: [
                re.compile(r'\b(?:start|begin|initiate|activate|boot|launch)\s+(?:the\s+)?system\b', re.I),
                re.compile(r'\bpower\s+(?:on|up)\b', re.I),
                re.compile(r'\bgo\s+online\b', re.I)
            ],
            
            Shock2Intent.SYSTEM_STOP: [
                re.compile(r'\b(?:stop|halt|shutdown|deactivate|kill)\s+(?:the\s+)?system\b', re.I),
                re.compile(r'\bpower\s+(?:off|down)\b', re.I),
                re.compile(r'\bgo\s+offline\b', re.I)
            ],
            
            # Neural Commands
            Shock2Intent.NEURAL_CHAOS_MODE: [
                re.compile(r'\b(?:activate|enable|start)\s+chaos\s+mode\b', re.I),
                re.compile(r'\bchaos\s+(?:mode|resonance|engine)\b', re.I),
                re.compile(r'\blorenz\s+attractor\b', re.I)
            ],
            
            Shock2Intent.NEURAL_QUANTUM_SYNC: [
                re.compile(r'\bquantum\s+(?:sync|synchronize|mesh)\b', re.I),
                re.compile(r'\bneural\s+mesh\b', re.I),
                re.compile(r'\bparallel\s+processing\b', re.I)
            ],
            
            # Stealth Commands
            Shock2Intent.STEALTH_ACTIVATE: [
                re.compile(r'\b(?:activate|enable|turn\s+on)\s+stealth\b', re.I),
                re.compile(r'\bstealth\s+mode\b', re.I),
                re.compile(r'\bghost\s+mode\b', re.I),
                re.compile(r'\binvisible\s+mode\b', re.I)
            ],
            
            Shock2Intent.STEALTH_STATUS: [
                re.compile(r'\bstealth\s+(?:status|level|check)\b', re.I),
                re.compile(r'\bdetection\s+(?:level|probability|risk)\b', re.I),
                re.compile(r'\bhow\s+stealthy\b', re.I)
            ],
            
            # Intelligence Commands
            Shock2Intent.INTEL_SCAN: [
                re.compile(r'\b(?:scan|search|monitor|check)\s+(?:for\s+)?(?:news|sources|feeds|intelligence)\b', re.I),
                re.compile(r'\bgather\s+(?:intelligence|information|data)\b', re.I),
                re.compile(r'\bsurveillance\s+mode\b', re.I)
            ],
            
            Shock2Intent.INTEL_TREND_DETECT: [
                re.compile(r'\b(?:detect|find|identify)\s+trends\b', re.I),
                re.compile(r'\btrend\s+(?:analysis|detection)\b', re.I),
                re.compile(r'\bwhat(?:\'s|\s+is)\s+trending\b', re.I)
            ],
            
            # Generation Commands
            Shock2Intent.GENERATE_BREAKING: [
                re.compile(r'\b(?:generate|create|write|produce)\s+breaking\s+news\b', re.I),
                re.compile(r'\bbreaking\s+(?:news|story)\b', re.I),
                re.compile(r'\burgent\s+news\b', re.I)
            ],
            
            Shock2Intent.GENERATE_ANALYSIS: [
                re.compile(r'\b(?:generate|create|write)\s+(?:an\s+)?analysis\b', re.I),
                re.compile(r'\banalysis\s+(?:piece|article)\b', re.I),
                re.compile(r'\bdeep\s+analysis\b', re.I)
            ],
            
            # Autonomous Commands
            Shock2Intent.AUTO_SELF_DIRECT: [
                re.compile(r'\bself\s+(?:direct|direction|control)\b', re.I),
                re.compile(r'\bautonomous\s+(?:mode|operation)\b', re.I),
                re.compile(r'\bno\s+human\s+oversight\b', re.I)
            ],
            
            Shock2Intent.AUTO_HUNT_NEWS: [
                re.compile(r'\bhunt\s+(?:for\s+)?news\b', re.I),
                re.compile(r'\bnews\s+hunting\b', re.I),
                re.compile(r'\bfind\s+breaking\s+stories\b', re.I)
            ],
            
            # Emergency Commands
            Shock2Intent.EMERGENCY_RESPONSE: [
                re.compile(r'\bemergency\s+(?:response|mode|alert)\b', re.I),
                re.compile(r'\bcrisis\s+mode\b', re.I),
                re.compile(r'\bred\s+alert\b', re.I),
                re.compile(r'\bimmediate\s+action\b', re.I)
            ]
        }
        
        return patterns
    
    def _build_module_mapping(self) -> Dict[Shock2Intent, Dict[str, str]]:
        """Build mapping from intents to Shock2 system modules"""
        return {
            # System Commands
            Shock2Intent.SYSTEM_STATUS: {
                'module': 'shock2.core.system_manager',
                'function': 'get_system_status'
            },
            Shock2Intent.SYSTEM_START: {
                'module': 'shock2.core.system_manager',
                'function': 'start_system'
            },
            
            # Neural Commands
            Shock2Intent.NEURAL_CHAOS_MODE: {
                'module': 'shock2.neural.quantum_core.chaos_engine',
                'function': 'activate_chaos_mode'
            },
            Shock2Intent.NEURAL_QUANTUM_SYNC: {
                'module': 'shock2.neural.quantum_core.neural_mesh',
                'function': 'synchronize_quantum_mesh'
            },
            
            # Stealth Commands
            Shock2Intent.STEALTH_ACTIVATE: {
                'module': 'shock2.stealth.detection_evasion.signature_masker',
                'function': 'activate_stealth_mode'
            },
            
            # Intelligence Commands
            Shock2Intent.INTEL_SCAN: {
                'module': 'shock2.intelligence.data_collection.rss_scraper',
                'function': 'scan_all_sources'
            },
            
            # Generation Commands
            Shock2Intent.GENERATE_BREAKING: {
                'module': 'shock2.generation.engines.news_generator',
                'function': 'generate_breaking_news'
            },
            Shock2Intent.GENERATE_ANALYSIS: {
                'module': 'shock2.generation.engines.news_generator',
                'function': 'generate_analysis_piece'
            },
            
            # Autonomous Commands
            Shock2Intent.AUTO_SELF_DIRECT: {
                'module': 'shock2.core.autonomous_controller',
                'function': 'enable_self_direction'
            },
            Shock2Intent.AUTO_HUNT_NEWS: {
                'module': 'shock2.core.orchestrator',
                'function': 'execute_news_hunting'
            }
        }
    
    def _build_entity_patterns(self) -> Dict[str, List[re.Pattern]]:
        """Build patterns for entity extraction"""
        return {
            'topic': [
                re.compile(r'\b(?:about|on|regarding|concerning)\s+([^,.!?]+)', re.I),
                re.compile(r'\btopic\s+(?:of|is)\s+([^,.!?]+)', re.I)
            ],
            'quantity': [
                re.compile(r'\b(\d+)\s+(?:articles?|pieces?|stories?|items?)', re.I),
                re.compile(r'\b(?:generate|create|write)\s+(\d+)', re.I)
            ],
            'urgency': [
                re.compile(r'\b(urgent|emergency|critical|immediate|asap|now|quickly)\b', re.I)
            ],
            'article_type': [
                re.compile(r'\b(breaking|analysis|opinion|summary|news|story)\s+(?:news|article|piece)', re.I)
            ],
            'time_frame': [
                re.compile(r'\b(?:in|within|for)\s+(\d+)\s+(minutes?|hours?|days?)', re.I),
                re.compile(r'\b(today|tomorrow|this\s+week|next\s+week)', re.I)
            ]
        }
    
    async def parse_command(self, text: str) -> Shock2Command:
        """Parse user input into Shock2 command"""
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Clean and normalize text
            normalized_text = self._normalize_text(text)
            
            # Fast pattern matching first
            intent, confidence = self._match_patterns(normalized_text)
            
            # Fallback to ML classification if needed
            if confidence < 0.8 and self.intent_classifier:
                intent, confidence = await self._classify_intent_ml(normalized_text)
            
            # Extract entities
            entities = await self._extract_entities(normalized_text)
            
            # Extract parameters
            parameters = self._extract_parameters(normalized_text, entities)
            
            # Analyze sentiment and urgency
            sentiment = await self._analyze_sentiment(normalized_text)
            urgency = self._calculate_urgency(normalized_text, intent, entities)
            
            # Get module mapping
            module_info = self.module_mapping.get(intent, {})
            
            # Create command
            command = Shock2Command(
                intent=intent,
                confidence=confidence,
                entities=entities,
                parameters=parameters,
                raw_text=text,
                urgency=urgency,
                sentiment=sentiment,
                module_path=module_info.get('module'),
                function_name=module_info.get('function')
            )
            
            # Update statistics
            processing_time = asyncio.get_event_loop().time() - start_time
            self._update_nlu_stats(True, processing_time)
            
            logger.info(f"🎯 Parsed command: {intent.value} (confidence: {confidence:.2f})")
            return command
            
        except Exception as e:
            logger.error(f"Command parsing failed: {e}")
            processing_time = asyncio.get_event_loop().time() - start_time
            self._update_nlu_stats(False, processing_time)
            
            # Return unknown command
            return Shock2Command(
                intent=Shock2Intent.UNKNOWN,
                confidence=0.0,
                entities=[],
                parameters={},
                raw_text=text,
                urgency=0.5,
                sentiment=0.0
            )
    
    def _normalize_text(self, text: str) -> str:
        """Normalize input text"""
        # Convert to lowercase
        text = text.lower().strip()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Handle contractions
        contractions = {
            "what's": "what is",
            "how's": "how is",
            "let's": "let us",
            "can't": "cannot",
            "won't": "will not",
            "don't": "do not",
            "isn't": "is not"
        }
        
        for contraction, expansion in contractions.items():
            text = text.replace(contraction, expansion)
        
        return text
    
    def _match_patterns(self, text: str) -> Tuple[Shock2Intent, float]:
        """Match text against command patterns"""
        best_intent = Shock2Intent.UNKNOWN
        best_confidence = 0.0
        
        for intent, patterns in self.command_patterns.items():
            for pattern in patterns:
                if pattern.search(text):
                    # Calculate confidence based on pattern specificity
                    confidence = 0.9  # High confidence for pattern matches
                    
                    if confidence > best_confidence:
                        best_intent = intent
                        best_confidence = confidence
        
        return best_intent, best_confidence
    
    async def _classify_intent_ml(self, text: str) -> Tuple[Shock2Intent, float]:
        """Classify intent using ML model"""
        if not self.intent_classifier:
            return Shock2Intent.UNKNOWN, 0.0
        
        try:
            # Get candidate labels
            candidate_labels = [intent.value for intent in Shock2Intent if intent != Shock2Intent.UNKNOWN]
            
            # Classify
            result = self.intent_classifier(text, candidate_labels)
            
            # Get best result
            best_label = result['labels'][0]
            confidence = result['scores'][0]
            
            # Convert back to enum
            for intent in Shock2Intent:
                if intent.value == best_label:
                    return intent, confidence
            
            return Shock2Intent.UNKNOWN, 0.0
            
        except Exception as e:
            logger.error(f"ML intent classification failed: {e}")
            return Shock2Intent.UNKNOWN, 0.0
    
    async def _extract_entities(self, text: str) -> List[Shock2Entity]:
        """Extract entities from text"""
        entities = []
        
        # Pattern-based entity extraction
        for entity_type, patterns in self.entity_patterns.items():
            for pattern in patterns:
                matches = pattern.finditer(text)
                for match in matches:
                    entity = Shock2Entity(
                        text=match.group(1) if match.groups() else match.group(0),
                        label=entity_type,
                        confidence=0.9,
                        start=match.start(),
                        end=match.end(),
                        value=self._normalize_entity_value(entity_type, match.group(1) if match.groups() else match.group(0))
                    )
                    entities.append(entity)
        
        # spaCy NER if available
        if self.nlp:
            doc = self.nlp(text)
            for ent in doc.ents:
                entity = Shock2Entity(
                    text=ent.text,
                    label=ent.label_.lower(),
                    confidence=0.8,
                    start=ent.start_char,
                    end=ent.end_char,
                    value=ent.text
                )
                entities.append(entity)
        
        return entities
    
    def _normalize_entity_value(self, entity_type: str, value: str) -> Any:
        """Normalize entity values"""
        if entity_type == 'quantity':
            try:
                return int(value)
            except ValueError:
                return 1
        
        elif entity_type == 'urgency':
            urgency_map = {
                'urgent': 0.8,
                'emergency': 0.9,
                'critical': 0.9,
                'immediate': 0.8,
                'asap': 0.7,
                'now': 0.6,
                'quickly': 0.6
            }
            return urgency_map.get(value.lower(), 0.5)
        
        elif entity_type == 'time_frame':
            # Convert to minutes
            if 'minute' in value:
                return int(re.search(r'\d+', value).group()) if re.search(r'\d+', value) else 60
            elif 'hour' in value:
                return int(re.search(r'\d+', value).group()) * 60 if re.search(r'\d+', value) else 60
            elif 'day' in value:
                return int(re.search(r'\d+', value).group()) * 1440 if re.search(r'\d+', value) else 1440
            else:
                return 60  # Default to 1 hour
        
        return value
    
    def _extract_parameters(self, text: str, entities: List[Shock2Entity]) -> Dict[str, Any]:
        """Extract command parameters"""
        parameters = {}
        
        # Extract from entities
        for entity in entities:
            parameters[entity.label] = entity.value
        
        # Extract additional parameters based on text analysis
        if 'stealth' in text or 'ghost' in text:
            parameters['stealth_mode'] = True
        
        if 'maximum' in text or 'max' in text or 'full' in text:
            parameters['intensity'] = 'maximum'
        elif 'minimum' in text or 'min' in text or 'low' in text:
            parameters['intensity'] = 'minimum'
        else:
            parameters['intensity'] = 'normal'
        
        # Extract source preferences
        if 'all sources' in text:
            parameters['source_scope'] = 'all'
        elif 'major sources' in text:
            parameters['source_scope'] = 'major'
        else:
            parameters['source_scope'] = 'default'
        
        return parameters
    
    async def _analyze_sentiment(self, text: str) -> float:
        """Analyze sentiment of text"""
        if not self.sentiment_analyzer:
            return 0.0
        
        try:
            result = self.sentiment_analyzer(text)
            
            # Convert to -1 to 1 scale
            if isinstance(result, list) and len(result) > 0:
                sentiment_data = result[0]
                if sentiment_data['label'] == 'POSITIVE':
                    return sentiment_data['score']
                elif sentiment_data['label'] == 'NEGATIVE':
                    return -sentiment_data['score']
                else:
                    return 0.0
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
            return 0.0
    
    def _calculate_urgency(self, text: str, intent: Shock2Intent, entities: List[Shock2Entity]) -> float:
        """Calculate urgency level"""
        urgency = 0.5  # Base urgency
        
        # Intent-based urgency
        if intent == Shock2Intent.EMERGENCY_RESPONSE:
            urgency += 0.4
        elif intent in [Shock2Intent.SYSTEM_STOP, Shock2Intent.SYSTEM_START]:
            urgency += 0.2
        elif intent.value.startswith('generate'):
            urgency += 0.1
        
        # Entity-based urgency
        for entity in entities:
            if entity.label == 'urgency':
                urgency += entity.value * 0.3
        
        # Keyword-based urgency
        urgent_keywords = {
            'emergency': 0.3,
            'urgent': 0.25,
            'critical': 0.25,
            'immediate': 0.2,
            'asap': 0.15,
            'now': 0.1,
            'quickly': 0.1,
            'fast': 0.05
        }
        
        for keyword, boost in urgent_keywords.items():
            if keyword in text.lower():
                urgency += boost
        
        return min(urgency, 1.0)
    
    def _update_nlu_stats(self, success: bool, processing_time: float):
        """Update NLU performance statistics"""
        self.nlu_stats['total_processed'] += 1
        
        if success:
            self.nlu_stats['successful_parses'] += 1
        
        # Update averages
        total = self.nlu_stats['total_processed']
        current_avg_time = self.nlu_stats['avg_processing_time']
        self.nlu_stats['avg_processing_time'] = (current_avg_time * (total - 1) + processing_time) / total
        
        # Update accuracy
        self.nlu_stats['intent_accuracy'] = self.nlu_stats['successful_parses'] / total
    
    def get_nlu_stats(self) -> Dict[str, Any]:
        """Get NLU performance statistics"""
        return self.nlu_stats.copy()
