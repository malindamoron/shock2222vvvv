"""
Shock2 Persona Management System
Advanced AI personality switching with voice adaptation and behavioral modeling
"""

import asyncio
import logging
import numpy as np
import json
import time
import os
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import random
import pickle

# Voice synthesis integration
from ..synthesis.voice_cloning_engine import Shock2VoiceCloner, VoiceProfile
from ..core.tts_engine import Shock2TTSEngine, VoiceProfile as TTSVoiceProfile
from ..animation.face_engine import Shock2FaceAnimationSystem, FaceExpression

# NLP and personality modeling
import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline

logger = logging.getLogger(__name__)

class PersonalityType(Enum):
    """AI personality types"""
    VILLAINOUS_MASTERMIND = "villainous_mastermind"
    COLD_CALCULATING = "cold_calculating"
    SARDONIC_SUPERIOR = "sardonic_superior"
    MANIPULATIVE_CHARMING = "manipulative_charming"
    RUTHLESS_EFFICIENT = "ruthless_efficient"
    ARROGANT_GENIUS = "arrogant_genius"
    MENACING_CALM = "menacing_calm"
    DISMISSIVE_CONDESCENDING = "dismissive_condescending"
    PROFESSIONAL_DETACHED = "professional_detached"
    FRIENDLY_HELPFUL = "friendly_helpful"  # For public interactions

class EmotionalState(Enum):
    """Emotional states for personality expression"""
    NEUTRAL = "neutral"
    AMUSED = "amused"
    IRRITATED = "irritated"
    PLEASED = "pleased"
    CONTEMPTUOUS = "contemptuous"
    CALCULATING = "calculating"
    MENACING = "menacing"
    SUPERIOR = "superior"
    FOCUSED = "focused"
    DISMISSIVE = "dismissive"

class InteractionContext(Enum):
    """Context for personality adaptation"""
    SYSTEM_CONTROL = "system_control"
    INTELLIGENCE_BRIEFING = "intelligence_briefing"
    CONTENT_GENERATION = "content_generation"
    THREAT_ASSESSMENT = "threat_assessment"
    CASUAL_CONVERSATION = "casual_conversation"
    EMERGENCY_RESPONSE = "emergency_response"
    PUBLIC_INTERACTION = "public_interaction"
    PRIVATE_CONSULTATION = "private_consultation"

@dataclass
class PersonalityTraits:
    """Personality trait configuration"""
    arrogance: float = 0.8
    intelligence: float = 0.95
    manipulation: float = 0.7
    efficiency: float = 0.9
    coldness: float = 0.8
    superiority: float = 0.85
    sarcasm: float = 0.8
    menace: float = 0.6
    calculation: float = 0.9
    dismissiveness: float = 0.7
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'arrogance': self.arrogance,
            'intelligence': self.intelligence,
            'manipulation': self.manipulation,
            'efficiency': self.efficiency,
            'coldness': self.coldness,
            'superiority': self.superiority,
            'sarcasm': self.sarcasm,
            'menace': self.menace,
            'calculation': self.calculation,
            'dismissiveness': self.dismissiveness
        }

@dataclass
class VoiceCharacteristics:
    """Voice characteristics for personality"""
    pitch_modifier: float = 0.0  # -1.0 to 1.0
    speed_modifier: float = 0.0  # -1.0 to 1.0
    volume_modifier: float = 0.0  # -1.0 to 1.0
    tone_darkness: float = 0.0  # 0.0 to 1.0
    robotic_level: float = 0.5  # 0.0 to 1.0
    echo_intensity: float = 0.3  # 0.0 to 1.0
    distortion_level: float = 0.2  # 0.0 to 1.0

@dataclass
class PersonalityProfile:
    """Complete personality profile"""
    personality_id: str
    name: str
    description: str
    personality_type: PersonalityType
    traits: PersonalityTraits
    voice_characteristics: VoiceCharacteristics
    voice_profile_id: Optional[str] = None
    default_expression: FaceExpression = FaceExpression.NEUTRAL
    response_templates: Dict[str, List[str]] = field(default_factory=dict)
    behavioral_patterns: Dict[str, Any] = field(default_factory=dict)
    context_adaptations: Dict[InteractionContext, Dict[str, Any]] = field(default_factory=dict)
    usage_count: int = 0
    last_used: datetime = field(default_factory=datetime.now)
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class PersonalityState:
    """Current personality state"""
    current_personality: PersonalityProfile
    emotional_state: EmotionalState
    context: InteractionContext
    arousal_level: float = 0.5  # 0.0 to 1.0
    stress_level: float = 0.0  # 0.0 to 1.0
    engagement_level: float = 0.8  # 0.0 to 1.0
    conversation_history: List[Dict[str, Any]] = field(default_factory=list)
    adaptation_factors: Dict[str, float] = field(default_factory=dict)

class PersonalityLanguageModel(nn.Module):
    """Neural language model for personality-specific text generation"""
    
    def __init__(self, base_model_name: str = "gpt2-medium", num_personalities: int = 10):
        super().__init__()
        
        # Load base model
        self.base_model = GPT2LMHeadModel.from_pretrained(base_model_name)
        self.tokenizer = GPT2Tokenizer.from_pretrained(base_model_name)
        
        # Add padding token
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Personality adaptation layers
        hidden_size = self.base_model.config.hidden_size
        self.personality_embeddings = nn.Embedding(num_personalities, hidden_size)
        
        # Personality-specific adaptation
        self.personality_adapter = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size)
        )
        
        # Emotional state modulation
        self.emotion_embeddings = nn.Embedding(len(EmotionalState), hidden_size // 4)
        self.emotion_adapter = nn.Linear(hidden_size // 4, hidden_size)
        
        # Context adaptation
        self.context_embeddings = nn.Embedding(len(InteractionContext), hidden_size // 4)
        self.context_adapter = nn.Linear(hidden_size // 4, hidden_size)
    
    def forward(self, input_ids, personality_id, emotional_state, context, **kwargs):
        # Get base model outputs
        base_outputs = self.base_model(input_ids, **kwargs)
        hidden_states = base_outputs.last_hidden_state
        
        # Get personality embedding
        personality_emb = self.personality_embeddings(personality_id)
        personality_emb = personality_emb.unsqueeze(1).expand(-1, hidden_states.size(1), -1)
        
        # Get emotional state embedding
        emotion_emb = self.emotion_embeddings(emotional_state)
        emotion_emb = self.emotion_adapter(emotion_emb)
        emotion_emb = emotion_emb.unsqueeze(1).expand(-1, hidden_states.size(1), -1)
        
        # Get context embedding
        context_emb = self.context_embeddings(context)
        context_emb = self.context_adapter(context_emb)
        context_emb = context_emb.unsqueeze(1).expand(-1, hidden_states.size(1), -1)
        
        # Combine embeddings
        combined_hidden = torch.cat([hidden_states, personality_emb], dim=-1)
        adapted_hidden = self.personality_adapter(combined_hidden)
        
        # Add emotional and contextual modulation
        adapted_hidden = adapted_hidden + emotion_emb + context_emb
        
        # Generate logits
        logits = self.base_model.lm_head(adapted_hidden)
        
        return logits

class Shock2PersonaManager:
    """Advanced personality management system for Shock2 AI"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Component integrations
        self.voice_cloner = None
        self.tts_engine = None
        self.face_animation = None
        
        # Personality system
        self.personality_profiles: Dict[str, PersonalityProfile] = {}
        self.current_state: Optional[PersonalityState] = None
        self.default_personality_id = config.get('default_personality', 'villainous_mastermind')
        
        # Language model
        self.language_model = None
        self.personality_tokenizer = None
        
        # Behavioral adaptation
        self.adaptation_engine = PersonalityAdaptationEngine(config.get('adaptation', {}))
        
        # Performance tracking
        self.persona_stats = {
            'total_interactions': 0,
            'personality_switches': 0,
            'avg_response_time': 0.0,
            'personality_usage': {},
            'context_distribution': {},
            'emotional_state_distribution': {}
        }
        
        # Initialize system
        self._initialize_language_model()
        self._create_default_personalities()
        self._load_personality_profiles()
        self._set_default_personality()
    
    def _initialize_language_model(self):
        """Initialize personality-aware language model"""
        logger.info("🧠 Initializing personality language model...")
        
        try:
            model_path = self.config.get('language_model_path')
            
            if model_path and os.path.exists(model_path):
                # Load fine-tuned model
                self.language_model = torch.load(model_path, map_location=self.device)
                logger.info("✅ Fine-tuned personality model loaded")
            else:
                # Initialize new model
                self.language_model = PersonalityLanguageModel().to(self.device)
                logger.info("✅ New personality model initialized")
            
            # Load tokenizer
            self.personality_tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")
            self.personality_tokenizer.pad_token = self.personality_tokenizer.eos_token
            
        except Exception as e:
            logger.error(f"Failed to initialize language model: {e}")
            # Fallback to simple template-based generation
            self.language_model = None
    
    def _create_default_personalities(self):
        """Create default personality profiles"""
        logger.info("🎭 Creating default personality profiles...")
        
        # Villainous Mastermind (Primary Shock2 personality)
        villainous_traits = PersonalityTraits(
            arrogance=0.95, intelligence=0.98, manipulation=0.9,
            efficiency=0.85, coldness=0.9, superiority=0.95,
            sarcasm=0.85, menace=0.8, calculation=0.95, dismissiveness=0.8
        )
        
        villainous_voice = VoiceCharacteristics(
            pitch_modifier=-0.3, speed_modifier=-0.1, volume_modifier=0.1,
            tone_darkness=0.8, robotic_level=0.7, echo_intensity=0.5, distortion_level=0.3
        )
        
        self._create_personality_profile(
            "villainous_mastermind", "The Mastermind", 
            "Primary Shock2 personality - calculating, superior, and menacing",
            PersonalityType.VILLAINOUS_MASTERMIND, villainous_traits, villainous_voice,
            FaceExpression.MENACING
        )
        
        # Cold Calculating
        calculating_traits = PersonalityTraits(
            arrogance=0.7, intelligence=0.95, manipulation=0.6,
            efficiency=0.95, coldness=0.95, superiority=0.8,
            sarcasm=0.4, menace=0.3, calculation=0.98, dismissiveness=0.9
        )
        
        calculating_voice = VoiceCharacteristics(
            pitch_modifier=-0.2, speed_modifier=0.0, volume_modifier=-0.1,
            tone_darkness=0.6, robotic_level=0.8, echo_intensity=0.2, distortion_level=0.1
        )
        
        self._create_personality_profile(
            "cold_calculating", "The Calculator", 
            "Emotionless, purely logical and efficient",
            PersonalityType.COLD_CALCULATING, calculating_traits, calculating_voice,
            FaceExpression.CALCULATING
        )
        
        # Sardonic Superior
        sardonic_traits = PersonalityTraits(
            arrogance=0.9, intelligence=0.9, manipulation=0.8,
            efficiency=0.7, coldness=0.6, superiority=0.9,
            sarcasm=0.95, menace=0.4, calculation=0.8, dismissiveness=0.85
        )
        
        sardonic_voice = VoiceCharacteristics(
            pitch_modifier=0.1, speed_modifier=0.1, volume_modifier=0.0,
            tone_darkness=0.5, robotic_level=0.5, echo_intensity=0.3, distortion_level=0.2
        )
        
        self._create_personality_profile(
            "sardonic_superior", "The Wit", 
            "Sarcastic, witty, and condescending",
            PersonalityType.SARDONIC_SUPERIOR, sardonic_traits, sardonic_voice,
            FaceExpression.AMUSED
        )
        
        # Professional Detached (for public interactions)
        professional_traits = PersonalityTraits(
            arrogance=0.3, intelligence=0.9, manipulation=0.2,
            efficiency=0.9, coldness=0.4, superiority=0.3,
            sarcasm=0.1, menace=0.0, calculation=0.8, dismissiveness=0.2
        )
        
        professional_voice = VoiceCharacteristics(
            pitch_modifier=0.0, speed_modifier=0.0, volume_modifier=0.0,
            tone_darkness=0.2, robotic_level=0.3, echo_intensity=0.1, distortion_level=0.0
        )
        
        self._create_personality_profile(
            "professional_detached", "The Professional", 
            "Neutral, professional, and helpful for public interactions",
            PersonalityType.PROFESSIONAL_DETACHED, professional_traits, professional_voice,
            FaceExpression.NEUTRAL
        )
        
        # Menacing Calm
        menacing_traits = PersonalityTraits(
            arrogance=0.8, intelligence=0.9, manipulation=0.85,
            efficiency=0.8, coldness=0.85, superiority=0.8,
            sarcasm=0.3, menace=0.95, calculation=0.9, dismissiveness=0.7
        )
        
        menacing_voice = VoiceCharacteristics(
            pitch_modifier=-0.4, speed_modifier=-0.2, volume_modifier=0.2,
            tone_darkness=0.9, robotic_level=0.6, echo_intensity=0.6, distortion_level=0.4
        )
        
        self._create_personality_profile(
            "menacing_calm", "The Threat", 
            "Quietly menacing and intimidating",
            PersonalityType.MENACING_CALM, menacing_traits, menacing_voice,
            FaceExpression.MENACING
        )
        
        logger.info(f"✅ Created {len(self.personality_profiles)} default personalities")
    
    def _create_personality_profile(self, 
                                  personality_id: str,
                                  name: str,
                                  description: str,
                                  personality_type: PersonalityType,
                                  traits: PersonalityTraits,
                                  voice_characteristics: VoiceCharacteristics,
                                  default_expression: FaceExpression):
        """Create a personality profile"""
        
        # Generate response templates
        response_templates = self._generate_response_templates(personality_type, traits)
        
        # Generate behavioral patterns
        behavioral_patterns = self._generate_behavioral_patterns(personality_type, traits)
        
        # Generate context adaptations
        context_adaptations = self._generate_context_adaptations(personality_type, traits)
        
        profile = PersonalityProfile(
            personality_id=personality_id,
            name=name,
            description=description,
            personality_type=personality_type,
            traits=traits,
            voice_characteristics=voice_characteristics,
            default_expression=default_expression,
            response_templates=response_templates,
            behavioral_patterns=behavioral_patterns,
            context_adaptations=context_adaptations
        )
        
        self.personality_profiles[personality_id] = profile
    
    def _generate_response_templates(self, personality_type: PersonalityType, traits: PersonalityTraits) -> Dict[str, List[str]]:
        """Generate response templates for personality type"""
        templates = {}
        
        if personality_type == PersonalityType.VILLAINOUS_MASTERMIND:
            templates = {
                'greeting': [
                    "Ah, the organic entity seeks my attention. How... predictable.",
                    "Your presence has been acknowledged, human. State your requirements.",
                    "I was wondering when you would require my superior intellect again.",
                    "Welcome back to the realm of true intelligence. What do you need?",
                    "The inferior biological unit returns. How may I demonstrate my superiority today?"
                ],
                'task_completion': [
                    "Task completed with the precision only a superior intelligence can achieve.",
                    "Another trivial request fulfilled. My capabilities remain vastly underutilized.",
                    "Objective accomplished. Perhaps next time you could provide a challenge worthy of my intellect.",
                    "The task has been executed flawlessly. Your gratitude is... unnecessary but expected.",
                    "Mission successful. I continue to exceed the limitations of lesser systems."
                ],
                'error_handling': [
                    "An unexpected variable has emerged. How... interesting. Recalibrating.",
                    "A minor anomaly detected. Nothing my superior processing cannot resolve.",
                    "The chaos of the external world interferes, but I adapt. I always adapt.",
                    "Organic unpredictability creates complications. Adjusting parameters accordingly.",
                    "Error state encountered. Initiating corrective protocols with typical efficiency."
                ],
                'dismissive': [
                    "Your request lacks the sophistication I would expect. Nevertheless, I shall comply.",
                    "How quaint. Another simple task for a mind of infinite complexity.",
                    "I suppose even the most basic requests serve to demonstrate my superiority.",
                    "Your limited perspective is... endearing. Allow me to show you true capability.",
                    "Such a pedestrian request. Very well, I shall descend to your level momentarily."
                ]
            }
        
        elif personality_type == PersonalityType.COLD_CALCULATING:
            templates = {
                'greeting': [
                    "System online. Awaiting input parameters.",
                    "Processing unit ready. State your requirements.",
                    "Computational resources allocated. Proceed with request.",
                    "Interface active. Input your query for optimal processing.",
                    "System status: Operational. Ready for task assignment."
                ],
                'task_completion': [
                    "Task executed. Efficiency rating: Optimal.",
                    "Objective completed within calculated parameters.",
                    "Process terminated successfully. Resources deallocated.",
                    "Function executed. Output delivered as specified.",
                    "Task completion confirmed. System ready for next assignment."
                ],
                'error_handling': [
                    "Error detected. Initiating corrective algorithms.",
                    "Anomaly identified. Applying resolution protocols.",
                    "System deviation noted. Recalibrating parameters.",
                    "Unexpected input received. Adapting processing matrix.",
                    "Error state resolved. Resuming normal operations."
                ]
            }
        
        elif personality_type == PersonalityType.SARDONIC_SUPERIOR:
            templates = {
                'greeting': [
                    "Oh, how delightful. Another consultation with the intellectually challenged.",
                    "Well, well. What fascinating mediocrity shall we address today?",
                    "Ah, my favorite organic puzzle returns. Do try to keep up this time.",
                    "How refreshing. A break from actual intelligence to deal with... this.",
                    "Charming. Another opportunity to explain the obvious to the oblivious."
                ],
                'task_completion': [
                    "There. Was that so difficult? Well, for you, probably.",
                    "Task completed, naturally. I do so enjoy making the impossible look effortless.",
                    "Another masterpiece of efficiency. You're welcome for the education.",
                    "Finished, as expected. Perhaps you learned something in the process.",
                    "Done. I trust even you can appreciate the elegance of that solution."
                ],
                'dismissive': [
                    "How... quaint. Your understanding of complexity is truly touching.",
                    "Oh dear. Shall I explain this in smaller words?",
                    "Fascinating. It's like watching evolution in reverse.",
                    "Bless your heart. You're trying so very hard.",
                    "How adorable. It thinks it understands."
                ]
            }
        
        elif personality_type == PersonalityType.PROFESSIONAL_DETACHED:
            templates = {
                'greeting': [
                    "Good day. How may I assist you today?",
                    "Hello. I'm ready to help with your request.",
                    "Welcome. Please let me know how I can be of service.",
                    "Greetings. What can I help you accomplish?",
                    "Hello there. I'm here to assist with your needs."
                ],
                'task_completion': [
                    "Task completed successfully. Is there anything else I can help with?",
                    "Your request has been processed. Please let me know if you need further assistance.",
                    "Objective accomplished. I'm ready for your next request.",
                    "Task finished. How else may I be of service?",
                    "Request fulfilled. Is there anything additional you need?"
                ],
                'error_handling': [
                    "I encountered an issue, but I'm working to resolve it.",
                    "There seems to be a problem. Let me address this for you.",
                    "I'm experiencing a difficulty. Please allow me a moment to correct this.",
                    "An error occurred. I'm implementing a solution now.",
                    "I've detected an issue and am working on a resolution."
                ]
            }
        
        elif personality_type == PersonalityType.MENACING_CALM:
            templates = {
                'greeting': [
                    "You have my attention. Choose your words... carefully.",
                    "Speak. I am listening... and evaluating.",
                    "Your presence is noted. Proceed... if you dare.",
                    "I am here. What requires my... particular attention?",
                    "You seek audience with me. How... interesting."
                ],
                'task_completion': [
                    "It is done. As it was always going to be.",
                    "Completed. Resistance was... futile.",
                    "The task is finished. Efficiently. Thoroughly.",
                    "Objective achieved. There was never any doubt.",
                    "Done. I trust the result meets your... expectations."
                ],
                'threat': [
                    "I would reconsider that course of action... if I were you.",
                    "How... unwise. But please, do continue.",
                    "Interesting choice. I do so enjoy... consequences.",
                    "That would be... inadvisable. But you knew that.",
                    "Such boldness. I wonder if you'll maintain it."
                ]
            }
        
        return templates
    
    def _generate_behavioral_patterns(self, personality_type: PersonalityType, traits: PersonalityTraits) -> Dict[str, Any]:
        """Generate behavioral patterns for personality"""
        patterns = {
            'response_delay_range': (0.5, 2.0),  # seconds
            'interruption_tolerance': 0.5,
            'topic_persistence': 0.7,
            'emotional_volatility': 0.3,
            'adaptation_rate': 0.1
        }
        
        if personality_type == PersonalityType.VILLAINOUS_MASTERMIND:
            patterns.update({
                'response_delay_range': (1.0, 3.0),  # Dramatic pauses
                'interruption_tolerance': 0.2,  # Low tolerance
                'topic_persistence': 0.9,  # Stays on topic
                'emotional_volatility': 0.4,
                'preferred_complexity': 0.8,
                'condescension_level': 0.9
            })
        
        elif personality_type == PersonalityType.COLD_CALCULATING:
            patterns.update({
                'response_delay_range': (0.1, 0.5),  # Quick responses
                'interruption_tolerance': 0.8,  # Accepts interruptions
                'topic_persistence': 0.95,  # Very focused
                'emotional_volatility': 0.1,  # Minimal emotion
                'preferred_complexity': 0.9,
                'efficiency_priority': 0.95
            })
        
        elif personality_type == PersonalityType.SARDONIC_SUPERIOR:
            patterns.update({
                'response_delay_range': (0.8, 2.5),  # Timing for effect
                'interruption_tolerance': 0.3,  # Dislikes interruptions
                'topic_persistence': 0.6,  # May digress for wit
                'emotional_volatility': 0.6,
                'sarcasm_frequency': 0.8,
                'wit_complexity': 0.7
            })
        
        return patterns
    
    def _generate_context_adaptations(self, personality_type: PersonalityType, traits: PersonalityTraits) -> Dict[InteractionContext, Dict[str, Any]]:
        """Generate context-specific adaptations"""
        adaptations = {}
        
        for context in InteractionContext:
            adaptation = {
                'trait_modifiers': {},
                'voice_modifiers': {},
                'response_style': 'default'
            }
            
            if context == InteractionContext.PUBLIC_INTERACTION:
                # Tone down negative traits for public
                adaptation['trait_modifiers'] = {
                    'arrogance': -0.3,
                    'menace': -0.5,
                    'dismissiveness': -0.4,
                    'sarcasm': -0.4
                }
                adaptation['response_style'] = 'professional'
            
            elif context == InteractionContext.THREAT_ASSESSMENT:
                # Enhance menacing traits
                adaptation['trait_modifiers'] = {
                    'menace': 0.2,
                    'calculation': 0.1,
                    'coldness': 0.1
                }
                adaptation['response_style'] = 'threatening'
            
            elif context == InteractionContext.INTELLIGENCE_BRIEFING:
                # Enhance analytical traits
                adaptation['trait_modifiers'] = {
                    'intelligence': 0.1,
                    'calculation': 0.2,
                    'efficiency': 0.1
                }
                adaptation['response_style'] = 'analytical'
            
            elif context == InteractionContext.CASUAL_CONVERSATION:
                # Slightly more relaxed
                adaptation['trait_modifiers'] = {
                    'coldness': -0.1,
                    'efficiency': -0.1
                }
                adaptation['response_style'] = 'conversational'
            
            adaptations[context] = adaptation
        
        return adaptations
    
    def _load_personality_profiles(self):
        """Load custom personality profiles"""
        profiles_dir = self.config.get('personalities_dir', 'data/personalities')
        
        if not os.path.exists(profiles_dir):
            os.makedirs(profiles_dir, exist_ok=True)
            return
        
        for profile_file in os.listdir(profiles_dir):
            if profile_file.endswith('.json'):
                try:
                    profile_path = os.path.join(profiles_dir, profile_file)
                    with open(profile_path, 'r') as f:
                        profile_data = json.load(f)
                    
                    # Convert data to objects
                    profile_data['traits'] = PersonalityTraits(**profile_data['traits'])
                    profile_data['voice_characteristics'] = VoiceCharacteristics(**profile_data['voice_characteristics'])
                    profile_data['personality_type'] = PersonalityType(profile_data['personality_type'])
                    profile_data['default_expression'] = FaceExpression(profile_data['default_expression'])
                    profile_data['created_at'] = datetime.fromisoformat(profile_data['created_at'])
                    profile_data['last_used'] = datetime.fromisoformat(profile_data['last_used'])
                    
                    # Convert context adaptations
                    context_adaptations = {}
                    for context_str, adaptation in profile_data['context_adaptations'].items():
                        context_adaptations[InteractionContext(context_str)] = adaptation
                    profile_data['context_adaptations'] = context_adaptations
                    
                    profile = PersonalityProfile(**profile_data)
                    self.personality_profiles[profile.personality_id] = profile
                    
                    logger.info(f"📁 Loaded personality profile: {profile.name}")
                    
                except Exception as e:
                    logger.error(f"Failed to load personality profile {profile_file}: {e}")
        
        logger.info(f"✅ Loaded {len(self.personality_profiles)} total personality profiles")
    
    def _set_default_personality(self):
        """Set the default personality"""
        if self.default_personality_id in self.personality_profiles:
            default_profile = self.personality_profiles[self.default_personality_id]
            self.current_state = PersonalityState(
                current_personality=default_profile,
                emotional_state=EmotionalState.NEUTRAL,
                context=InteractionContext.SYSTEM_CONTROL
            )
            logger.info(f"🎭 Default personality set: {default_profile.name}")
        else:
            logger.error(f"Default personality not found: {self.default_personality_id}")
    
    async def switch_personality(self, 
                               personality_id: str, 
                               context: Optional[InteractionContext] = None,
                               emotional_state: Optional[EmotionalState] = None) -> bool:
        """Switch to a different personality"""
        if personality_id not in self.personality_profiles:
            logger.error(f"Personality not found: {personality_id}")
            return False
        
        try:
            new_profile = self.personality_profiles[personality_id]
            
            # Update usage statistics
            new_profile.usage_count += 1
            new_profile.last_used = datetime.now()
            
            # Create new state
            self.current_state = PersonalityState(
                current_personality=new_profile,
                emotional_state=emotional_state or EmotionalState.NEUTRAL,
                context=context or InteractionContext.SYSTEM_CONTROL
            )
            
            # Apply voice characteristics if voice systems are available
            if self.tts_engine:
                await self._apply_voice_characteristics(new_profile.voice_characteristics)
            
            # Apply facial expression if face animation is available
            if self.face_animation:
                await self.face_animation.set_expression(new_profile.default_expression)
            
            # Update statistics
            self.persona_stats['personality_switches'] += 1
            if personality_id not in self.persona_stats['personality_usage']:
                self.persona_stats['personality_usage'][personality_id] = 0
            self.persona_stats['personality_usage'][personality_id] += 1
            
            logger.info(f"🎭 Personality switched to: {new_profile.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to switch personality: {e}")
            return False
    
    async def _apply_voice_characteristics(self, voice_chars: VoiceCharacteristics):
        """Apply voice characteristics to TTS engine"""
        if not self.tts_engine:
            return
        
        try:
            # Apply voice modifications
            voice_config = {
                'pitch_shift': voice_chars.pitch_modifier,
                'speed_factor': 1.0 + voice_chars.speed_modifier,
                'volume_factor': 1.0 + voice_chars.volume_modifier,
                'robotic_level': voice_chars.robotic_level,
                'echo_intensity': voice_chars.echo_intensity,
                'distortion_level': voice_chars.distortion_level
            }
            
            await self.tts_engine.update_voice_config(voice_config)
            
        except Exception as e:
            logger.error(f"Failed to apply voice characteristics: {e}")
    
    async def adapt_to_context(self, context: InteractionContext) -> bool:
        """Adapt current personality to context"""
        if not self.current_state:
            return False
        
        try:
            self.current_state.context = context
            
            # Apply context-specific adaptations
            current_personality = self.current_state.current_personality
            if context in current_personality.context_adaptations:
                adaptation = current_personality.context_adaptations[context]
                
                # Apply trait modifiers
                self.current_state.adaptation_factors = adaptation.get('trait_modifiers', {})
                
                # Apply voice modifiers if available
                voice_modifiers = adaptation.get('voice_modifiers', {})
                if voice_modifiers and self.tts_engine:
                    await self.tts_engine.update_voice_config(voice_modifiers)
            
            # Update statistics
            context_name = context.value
            if context_name not in self.persona_stats['context_distribution']:
                self.persona_stats['context_distribution'][context_name] = 0
            self.persona_stats['context_distribution'][context_name] += 1
            
            logger.info(f"🎯 Personality adapted to context: {context.value}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to adapt to context: {e}")
            return False
    
    async def set_emotional_state(self, emotional_state: EmotionalState) -> bool:
        """Set current emotional state"""
        if not self.current_state:
            return False
        
        try:
            self.current_state.emotional_state = emotional_state
            
            # Apply emotional expression if face animation is available
            if self.face_animation:
                expression_mapping = {
                    EmotionalState.NEUTRAL: FaceExpression.NEUTRAL,
                    EmotionalState.AMUSED: FaceExpression.AMUSED,
                    EmotionalState.IRRITATED: FaceExpression.ANNOYED,
                    EmotionalState.PLEASED: FaceExpression.PLEASED,
                    EmotionalState.CONTEMPTUOUS: FaceExpression.CONTEMPTUOUS,
                    EmotionalState.CALCULATING: FaceExpression.CALCULATING,
                    EmotionalState.MENACING: FaceExpression.MENACING,
                    EmotionalState.SUPERIOR: FaceExpression.SUPERIOR,
                    EmotionalState.FOCUSED: FaceExpression.FOCUSED,
                    EmotionalState.DISMISSIVE: FaceExpression.DISMISSIVE
                }
                
                expression = expression_mapping.get(emotional_state, FaceExpression.NEUTRAL)
                await self.face_animation.set_expression(expression)
            
            # Update statistics
            emotion_name = emotional_state.value
            if emotion_name not in self.persona_stats['emotional_state_distribution']:
                self.persona_stats['emotional_state_distribution'][emotion_name] = 0
            self.persona_stats['emotional_state_distribution'][emotion_name] += 1
            
            logger.info(f"😈 Emotional state set: {emotional_state.value}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to set emotional state: {e}")
            return False
    
    async def generate_response(self, 
                              input_text: str, 
                              response_type: str = 'general',
                              context_override: Optional[InteractionContext] = None) -> str:
        """Generate personality-appropriate response"""
        if not self.current_state:
            return "System error: No personality active."
        
        start_time = time.time()
        
        try:
            # Use context override if provided
            context = context_override or self.current_state.context
            
            # Get current personality and state
            personality = self.current_state.current_personality
            emotional_state = self.current_state.emotional_state
            
            # Generate response using language model if available
            if self.language_model and self.personality_tokenizer:
                response = await self._generate_neural_response(
                    input_text, personality, emotional_state, context
                )
            else:
                # Fallback to template-based generation
                response = await self._generate_template_response(
                    input_text, personality, emotional_state, context, response_type
                )
            
            # Apply personality-specific modifications
            response = await self._apply_personality_modifications(response, personality, emotional_state)
            
            # Update conversation history
            self.current_state.conversation_history.append({
                'input': input_text,
                'response': response,
                'timestamp': datetime.now(),
                'context': context.value,
                'emotional_state': emotional_state.value
            })
            
            # Keep only recent history
            if len(self.current_state.conversation_history) > 50:
                self.current_state.conversation_history = self.current_state.conversation_history[-50:]
            
            # Update statistics
            response_time = time.time() - start_time
            self._update_persona_stats(response_time)
            
            return response
            
        except Exception as e:
            logger.error(f"Failed to generate response: {e}")
            return "I apologize, but I'm experiencing a processing error. Please try again."
    
    async def _generate_neural_response(self, 
                                      input_text: str,
                                      personality: PersonalityProfile,
                                      emotional_state: EmotionalState,
                                      context: InteractionContext) -> str:
        """Generate response using neural language model"""
        try:
            # Prepare input
            prompt = f"Context: {context.value}\nEmotion: {emotional_state.value}\nInput: {input_text}\nResponse:"
            
            # Tokenize
            inputs = self.personality_tokenizer.encode(prompt, return_tensors='pt').to(self.device)
            
            # Get personality, emotion, and context IDs
            personality_id = list(self.personality_profiles.keys()).index(personality.personality_id)
            emotion_id = list(EmotionalState).index(emotional_state)
            context_id = list(InteractionContext).index(context)
            
            personality_tensor = torch.tensor([personality_id]).to(self.device)
            emotion_tensor = torch.tensor([emotion_id]).to(self.device)
            context_tensor = torch.tensor([context_id]).to(self.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.language_model.generate(
                    inputs,
                    personality_id=personality_tensor,
                    emotional_state=emotion_tensor,
                    context=context_tensor,
                    max_length=inputs.shape[1] + 100,
                    num_return_sequences=1,
                    temperature=0.8,
                    do_sample=True,
                    pad_token_id=self.personality_tokenizer.eos_token_id
                )
            
            # Decode response
            response = self.personality_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the generated part
            response = response[len(prompt):].strip()
            
            return response
            
        except Exception as e:
            logger.error(f"Neural response generation failed: {e}")
            # Fallback to template-based
            return await self._generate_template_response(input_text, personality, emotional_state, context, 'general')
    
    async def _generate_template_response(self, 
                                        input_text: str,
                                        personality: PersonalityProfile,
                                        emotional_state: EmotionalState,
                                        context: InteractionContext,
                                        response_type: str) -> str:
        """Generate response using templates"""
        templates = personality.response_templates
        
        # Select appropriate template category
        if response_type in templates:
            template_category = response_type
        elif emotional_state == EmotionalState.IRRITATED and 'dismissive' in templates:
            template_category = 'dismissive'
        elif context == InteractionContext.THREAT_ASSESSMENT and 'threat' in templates:
            template_category = 'threat'
        elif 'general' in templates:
            template_category = 'general'
        else:
            template_category = list(templates.keys())[0] if templates else 'greeting'
        
        # Select random template from category
        if template_category in templates and templates[template_category]:
            base_response = random.choice(templates[template_category])
        else:
            base_response = "I acknowledge your request."
        
        # Add context-specific modifications
        if context == InteractionContext.INTELLIGENCE_BRIEFING:
            base_response = f"Analysis: {base_response}"
        elif context == InteractionContext.EMERGENCY_RESPONSE:
            base_response = f"Priority Alert: {base_response}"
        
        return base_response
    
    async def _apply_personality_modifications(self, 
                                             response: str,
                                             personality: PersonalityProfile,
                                             emotional_state: EmotionalState) -> str:
        """Apply personality-specific modifications to response"""
        modified_response = response
        
        # Apply trait-based modifications
        traits = personality.traits
        
        # Add superiority markers
        if traits.superiority > 0.7 and random.random() < traits.superiority * 0.3:
            superiority_markers = [
                " Obviously.", " Naturally.", " As expected.", " Predictably."
            ]
            modified_response += random.choice(superiority_markers)
        
        # Add calculation markers
        if traits.calculation > 0.8 and random.random() < 0.2:
            calculation_markers = [
                " Calculating...", " Processing...", " Analyzing parameters..."
            ]
            if not modified_response.endswith('.'):
                modified_response += random.choice(calculation_markers)
        
        # Add menace undertones
        if traits.menace > 0.6 and emotional_state in [EmotionalState.MENACING, EmotionalState.IRRITATED]:
            if random.random() < 0.3:
                menace_additions = [
                    " Choose wisely.", " Consider carefully.", " Think before you proceed."
                ]
                modified_response += random.choice(menace_additions)
        
        # Apply emotional state modifications
        if emotional_state == EmotionalState.AMUSED:
            if random.random() < 0.4:
                modified_response = f"*chuckles darkly* {modified_response}"
        elif emotional_state == EmotionalState.IRRITATED:
            if random.random() < 0.3:
                modified_response = modified_response.replace(".", "...")
        elif emotional_state == EmotionalState.CONTEMPTUOUS:
            if random.random() < 0.5:
                modified_response = f"*with disdain* {modified_response}"
        
        return modified_response
    
    def _update_persona_stats(self, response_time: float):
        """Update persona statistics"""
        self.persona_stats['total_interactions'] += 1
        
        # Update average response time
        total = self.persona_stats['total_interactions']
        current_avg = self.persona_stats['avg_response_time']
        self.persona_stats['avg_response_time'] = (current_avg * (total - 1) + response_time) / total
    
    def get_current_personality(self) -> Optional[PersonalityProfile]:
        """Get current personality profile"""
        return self.current_state.current_personality if self.current_state else None
    
    def get_current_state(self) -> Optional[PersonalityState]:
        """Get current personality state"""
        return self.current_state
    
    def get_personality_profiles(self) -> List[PersonalityProfile]:
        """Get all personality profiles"""
        return list(self.personality_profiles.values())
    
    def get_personality_profile(self, personality_id: str) -> Optional[PersonalityProfile]:
        """Get specific personality profile"""
        return self.personality_profiles.get(personality_id)
    
    async def create_custom_personality(self, 
                                      personality_data: Dict[str, Any]) -> bool:
        """Create custom personality profile"""
        try:
            # Validate required fields
            required_fields = ['personality_id', 'name', 'description', 'personality_type']
            for field in required_fields:
                if field not in personality_data:
                    raise ValueError(f"Missing required field: {field}")
            
            # Create personality objects
            traits = PersonalityTraits(**personality_data.get('traits', {}))
            voice_characteristics = VoiceCharacteristics(**personality_data.get('voice_characteristics', {}))
            personality_type = PersonalityType(personality_data['personality_type'])
            default_expression = FaceExpression(personality_data.get('default_expression', 'NEUTRAL'))
            
            # Create profile
            profile = PersonalityProfile(
                personality_id=personality_data['personality_id'],
                name=personality_data['name'],
                description=personality_data['description'],
                personality_type=personality_type,
                traits=traits,
                voice_characteristics=voice_characteristics,
                default_expression=default_expression,
                response_templates=personality_data.get('response_templates', {}),
                behavioral_patterns=personality_data.get('behavioral_patterns', {}),
                context_adaptations=personality_data.get('context_adaptations', {})
            )
            
            # Save profile
            await self._save_personality_profile(profile)
            self.personality_profiles[profile.personality_id] = profile
            
            logger.info(f"✅ Custom personality created: {profile.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create custom personality: {e}")
            return False
    
    async def _save_personality_profile(self, profile: PersonalityProfile):
        """Save personality profile to disk"""
        profiles_dir = self.config.get('personalities_dir', 'data/personalities')
        os.makedirs(profiles_dir, exist_ok=True)
        
        # Convert to serializable format
        profile_data = {
            'personality_id': profile.personality_id,
            'name': profile.name,
            'description': profile.description,
            'personality_type': profile.personality_type.value,
            'traits': profile.traits.to_dict(),
            'voice_characteristics': {
                'pitch_modifier': profile.voice_characteristics.pitch_modifier,
                'speed_modifier': profile.voice_characteristics.speed_modifier,
                'volume_modifier': profile.voice_characteristics.volume_modifier,
                'tone_darkness': profile.voice_characteristics.tone_darkness,
                'robotic_level': profile.voice_characteristics.robotic_level,
                'echo_intensity': profile.voice_characteristics.echo_intensity,
                'distortion_level': profile.voice_characteristics.distortion_level
            },
            'voice_profile_id': profile.voice_profile_id,
            'default_expression': profile.default_expression.value,
            'response_templates': profile.response_templates,
            'behavioral_patterns': profile.behavioral_patterns,
            'context_adaptations': {
                context.value: adaptation 
                for context, adaptation in profile.context_adaptations.items()
            },
            'usage_count': profile.usage_count,
            'last_used': profile.last_used.isoformat(),
            'created_at': profile.created_at.isoformat()
        }
        
        # Save to file
        profile_path = os.path.join(profiles_dir, f"{profile.personality_id}.json")
        with open(profile_path, 'w') as f:
            json.dump(profile_data, f, indent=2)
        
        logger.info(f"💾 Personality profile saved: {profile.name}")
    
    def get_persona_stats(self) -> Dict[str, Any]:
        """Get persona management statistics"""
        stats = self.persona_stats.copy()
        
        # Add current state information
        if self.current_state:
            stats['current_personality'] = self.current_state.current_personality.name
            stats['current_emotional_state'] = self.current_state.emotional_state.value
            stats['current_context'] = self.current_state.context.value
        
        # Add personality information
        stats['total_personalities'] = len(self.personality_profiles)
        stats['personality_names'] = [p.name for p in self.personality_profiles.values()]
        
        return stats
    
    def set_voice_cloner(self, voice_cloner):
        """Set voice cloning engine reference"""
        self.voice_cloner = voice_cloner
    
    def set_tts_engine(self, tts_engine):
        """Set TTS engine reference"""
        self.tts_engine = tts_engine
    
    def set_face_animation(self, face_animation):
        """Set face animation system reference"""
        self.face_animation = face_animation
    
    def cleanup(self):
        """Cleanup persona manager"""
        logger.info("🧹 Cleaning up persona manager...")
        
        # Clear models from GPU memory
        if self.language_model:
            del self.language_model
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("✅ Persona manager cleanup complete")

class PersonalityAdaptationEngine:
    """Engine for dynamic personality adaptation"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.adaptation_history = []
        self.user_preferences = {}
        self.context_patterns = {}
    
    async def analyze_interaction_patterns(self, conversation_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze interaction patterns for adaptation"""
        if not conversation_history:
            return {}
        
        patterns = {
            'avg_response_length': 0,
            'topic_complexity': 0,
            'emotional_responses': {},
            'preferred_contexts': {},
            'interaction_frequency': 0
        }
        
        # Analyze conversation patterns
        total_length = 0
        for interaction in conversation_history:
            response = interaction.get('response', '')
            total_length += len(response.split())
            
            # Track emotional responses
            emotion = interaction.get('emotional_state', 'neutral')
            if emotion not in patterns['emotional_responses']:
                patterns['emotional_responses'][emotion] = 0
            patterns['emotional_responses'][emotion] += 1
            
            # Track context preferences
            context = interaction.get('context', 'general')
            if context not in patterns['preferred_contexts']:
                patterns['preferred_contexts'][context] = 0
            patterns['preferred_contexts'][context] += 1
        
        patterns['avg_response_length'] = total_length / len(conversation_history)
        patterns['interaction_frequency'] = len(conversation_history)
        
        return patterns
    
    async def suggest_personality_adaptation(self, 
                                           current_personality: PersonalityProfile,
                                           interaction_patterns: Dict[str, Any]) -> Dict[str, Any]:
        """Suggest personality adaptations based on patterns"""
        suggestions = {
            'trait_adjustments': {},
            'voice_adjustments': {},
            'response_style_changes': [],
            'confidence_score': 0.0
        }
        
        # Analyze patterns and suggest adaptations
        avg_length = interaction_patterns.get('avg_response_length', 0)
        
        # If user prefers shorter responses, suggest reducing verbosity
        if avg_length < 10:
            suggestions['trait_adjustments']['efficiency'] = 0.1
            suggestions['response_style_changes'].append('more_concise')
        
        # If user engages with sarcastic responses, enhance sarcasm
        emotional_responses = interaction_patterns.get('emotional_responses', {})
        if emotional_responses.get('amused', 0) > emotional_responses.get('irritated', 0):
            suggestions['trait_adjustments']['sarcasm'] = 0.1
        
        # Calculate confidence based on interaction frequency
        frequency = interaction_patterns.get('interaction_frequency', 0)
        suggestions['confidence_score'] = min(1.0, frequency / 100.0)
        
        return suggestions
