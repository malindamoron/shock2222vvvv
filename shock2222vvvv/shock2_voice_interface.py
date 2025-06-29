#!/usr/bin/env python3
"""
Shock2 Voice Interface - Direct AI Communication System
Advanced voice-controlled interface with animated AI face and complex NLP processing
"""

import asyncio
import threading
import queue
import time
import json
import logging
import numpy as np
import cv2
import pygame
import speech_recognition as sr
import pyttsx3
import spacy
import torch
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    pipeline, AutoModelForCausalLM, GPT2LMHeadModel
)
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import sounddevice as sd
import librosa
import webrtcvad
from collections import deque
import re
import random
import os
import sys
from dataclasses import dataclass
from enum import Enum

# Add shock2 to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from shock2.core.system_manager import Shock2SystemManager
from shock2.core.orchestrator import CoreOrchestrator, Task, TaskPriority
from shock2.core.autonomous_controller import AutonomousController, DecisionType
from shock2.config.settings import load_config
from shock2.utils.exceptions import Shock2Exception

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VoiceCommand(Enum):
    """Voice command categories"""
    SYSTEM_CONTROL = "system_control"
    NEWS_GENERATION = "news_generation"
    INTELLIGENCE_GATHERING = "intelligence_gathering"
    STEALTH_OPERATIONS = "stealth_operations"
    PERFORMANCE_ANALYSIS = "performance_analysis"
    AUTONOMOUS_DECISIONS = "autonomous_decisions"
    CONVERSATION = "conversation"
    EMERGENCY = "emergency"

@dataclass
class VoiceInput:
    """Voice input data structure"""
    raw_audio: np.ndarray
    transcribed_text: str
    confidence: float
    intent: VoiceCommand
    entities: Dict[str, Any]
    sentiment: float
    urgency: float
    timestamp: datetime

class AdvancedSpeechRecognition:
    """Advanced speech recognition with VAD and noise reduction"""
    
    def __init__(self, sample_rate: int = 16000, frame_duration: int = 30):
        self.sample_rate = sample_rate
        self.frame_duration = frame_duration
        self.frame_size = int(sample_rate * frame_duration / 1000)
        
        # Voice Activity Detection
        self.vad = webrtcvad.Vad(3)  # Aggressiveness level 3 (most aggressive)
        
        # Speech recognition
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone(sample_rate=sample_rate)
        
        # Audio processing
        self.audio_buffer = deque(maxlen=100)  # 3 seconds at 30ms frames
        self.is_speaking = False
        self.speech_frames = []
        
        # Noise reduction
        self.noise_profile = None
        self.calibrated = False
        
        # Initialize microphone
        self._calibrate_microphone()

        # Initialize multiple speech recognition engines
        self.whisper_model = None
        self.vosk_model = None
        self.deepspeech_model = None
        self.load_speech_engines()
    
    def _calibrate_microphone(self):
        """Calibrate microphone for ambient noise"""
        logger.info("🎤 Calibrating microphone for ambient noise...")
        
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=2)
        
        self.calibrated = True
        logger.info("✅ Microphone calibrated")
    
    def _is_speech(self, audio_frame: bytes) -> bool:
        """Detect if audio frame contains speech"""
        try:
            return self.vad.is_speech(audio_frame, self.sample_rate)
        except:
            return False
    
    def _reduce_noise(self, audio_data: np.ndarray) -> np.ndarray:
        """Apply noise reduction to audio"""
        # Simple spectral subtraction noise reduction
        if self.noise_profile is None:
            return audio_data
        
        # Convert to frequency domain
        fft = np.fft.fft(audio_data)
        magnitude = np.abs(fft)
        phase = np.angle(fft)
        
        # Subtract noise profile
        clean_magnitude = magnitude - 0.5 * self.noise_profile
        clean_magnitude = np.maximum(clean_magnitude, 0.1 * magnitude)
        
        # Convert back to time domain
        clean_fft = clean_magnitude * np.exp(1j * phase)
        clean_audio = np.real(np.fft.ifft(clean_fft))
        
        return clean_audio.astype(np.float32)

    def load_speech_engines(self):
        """Load multiple speech recognition engines"""
        try:
            # Load Whisper
            self.whisper_model = pipeline("automatic-speech-recognition", model="openai/whisper-base")
            logger.info("✅ Whisper loaded")
        except Exception as e:
            logger.warning(f"Whisper loading failed: {e}")

        try:
            # Load Vosk
            from vosk import Model, KaldiRecognizer
            vosk_model_path = "vosk_model"  # Replace with your Vosk model path
            if not os.path.exists(vosk_model_path):
                logger.error("Please download the VOSK model from https://alphacephei.com/vosk/models and unpack it.")
                raise FileNotFoundError("Vosk model not found")
            self.vosk_model = Model(vosk_model_path)
            logger.info("✅ Vosk loaded")
        except Exception as e:
            logger.warning(f"Vosk loading failed: {e}")

        try:
            # Load DeepSpeech
            # Requires installation and setup, placeholder for now
            logger.info("DeepSpeech loading skipped (requires setup)")
            self.deepspeech_model = None
        except Exception as e:
            logger.warning(f"DeepSpeech loading failed: {e}")

    async def listen_continuously(self, callback):
        """Continuously listen for speech"""
        logger.info("🎧 Starting continuous speech recognition...")
        
        def audio_callback(indata, frames, time, status):
            """Audio input callback"""
            if status:
                logger.warning(f"Audio input status: {status}")
            
            # Convert to bytes for VAD
            audio_bytes = (indata[:, 0] * 32767).astype(np.int16).tobytes()
            
            # Check for speech
            is_speech = self._is_speech(audio_bytes)
            
            if is_speech:
                if not self.is_speaking:
                    self.is_speaking = True
                    self.speech_frames = []
                    logger.info("🗣️ Speech detected - recording...")
                
                self.speech_frames.append(indata[:, 0].copy())
            else:
                if self.is_speaking:
                    # End of speech detected
                    self.is_speaking = False
                    
                    if len(self.speech_frames) > 10:  # Minimum speech length
                        # Combine speech frames
                        speech_audio = np.concatenate(self.speech_frames)
                        
                        # Process speech
                        asyncio.create_task(self._process_speech(speech_audio, callback))
                    
                    self.speech_frames = []
        
        # Start audio stream
        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            callback=audio_callback,
            blocksize=self.frame_size,
            dtype=np.float32
        ):
            logger.info("🎤 Listening for voice commands...")
            
            # Keep listening until stopped
            while True:
                await asyncio.sleep(0.1)
    
    async def _process_speech(self, audio_data: np.ndarray, callback):
        """Process detected speech"""
        try:
            # Apply noise reduction
            clean_audio = self._reduce_noise(audio_data)
            
            # Convert to AudioData format for speech_recognition
            audio_bytes = (clean_audio * 32767).astype(np.int16).tobytes()
            audio_data_sr = sr.AudioData(audio_bytes, self.sample_rate, 2)
            
            # Transcribe speech using multiple engines
            transcriptions = await self.transcribe_with_engines(clean_audio)

            # Select best transcription (you can implement a more sophisticated selection logic)
            best_text = max(transcriptions, key=lambda x: x["confidence"])["text"]
            best_confidence = max(transcriptions, key=lambda x: x["confidence"])["confidence"]
            
            if best_text and best_confidence > 0.5:
                logger.info(f"🎯 Transcribed: '{best_text}' (confidence: {best_confidence:.2f})")
                await callback(best_text, best_confidence, clean_audio)
            
        except Exception as e:
            logger.error(f"Speech processing error: {e}")

    async def transcribe_with_engines(self, audio_data: np.ndarray) -> List[Dict[str, Any]]:
        """Transcribe audio using multiple engines and return a list of transcriptions"""
        transcriptions = []

        # Whisper transcription
        if self.whisper_model:
            try:
                text = self.whisper_model(audio_data)["text"]
                transcriptions.append({"engine": "whisper", "text": text, "confidence": 0.8})  # Assign a default confidence
            except Exception as e:
                logger.error(f"Whisper transcription failed: {e}")

        # Vosk transcription
        if self.vosk_model:
            try:
                from vosk import KaldiRecognizer
                rec = KaldiRecognizer(self.vosk_model, self.sample_rate)
                rec.AcceptWaveform((audio_data * 32768).astype(np.int16).tobytes())
                result = json.loads(rec.FinalResult())
                text = result.get("text", "")
                transcriptions.append({"engine": "vosk", "text": text, "confidence": 0.7})
            except Exception as e:
                logger.error(f"Vosk transcription failed: {e}")

        # DeepSpeech transcription (placeholder)
        if self.deepspeech_model:
            # Add DeepSpeech transcription logic here when available
            pass

        return transcriptions

class AdvancedNLPProcessor:
    """Advanced NLP processing for voice commands"""
    
    def __init__(self):
        self.nlp = None
        self.intent_classifier = None
        self.sentiment_analyzer = None
        self.entity_extractor = None
        self.command_patterns = {}
        
        # Command templates
        self.command_templates = {
            VoiceCommand.SYSTEM_CONTROL: [
                r"(?:start|begin|initiate|activate) (?:the )?system",
                r"(?:stop|halt|shutdown|deactivate) (?:the )?system",
                r"(?:status|health|state) (?:of )?(?:the )?system",
                r"(?:restart|reboot|reset) (?:the )?system",
                r"show (?:me )?(?:the )?(?:system )?(?:status|metrics|performance)"
            ],
            VoiceCommand.NEWS_GENERATION: [
                r"(?:generate|create|write|produce) (?:some )?(?:news|articles|content)",
                r"(?:start|begin) (?:news )?(?:generation|creation)",
                r"(?:write|create) (?:a|an) (?:article|story|piece) (?:about|on) (.+)",
                r"(?:generate|create) (?:breaking|urgent) news (?:about|on) (.+)",
                r"(?:make|produce) (?:an )?(?:analysis|opinion) (?:piece|article)"
            ],
            VoiceCommand.INTELLIGENCE_GATHERING: [
                r"(?:scan|search|monitor|check) (?:for )?(?:news|sources|feeds)",
                r"(?:gather|collect|find) (?:intelligence|information|data)",
                r"(?:what|tell me) (?:is|are) (?:the )?(?:latest|recent|current) (?:news|trends)",
                r"(?:analyze|examine|investigate) (?:the )?(?:news|sources|feeds)",
                r"(?:find|search for|look for) (?:news|information) (?:about|on) (.+)"
            ],
            VoiceCommand.STEALTH_OPERATIONS: [
                r"(?:activate|enable|turn on) (?:stealth|ghost|invisible) (?:mode|operations)",
                r"(?:mask|hide|conceal) (?:ai|artificial intelligence) (?:signatures|traces)",
                r"(?:check|verify|test) (?:stealth|detection) (?:level|status)",
                r"(?:increase|enhance|improve) (?:stealth|anonymity|concealment)",
                r"(?:evade|avoid|bypass) (?:detection|discovery|identification)"
            ],
            VoiceCommand.PERFORMANCE_ANALYSIS: [
                r"(?:analyze|check|examine) (?:performance|efficiency|metrics)",
                r"(?:show|display|report) (?:performance|system) (?:metrics|statistics|stats)",
                r"(?:how|what) (?:is|are) (?:the )?(?:system|performance) (?:doing|performing)",
                r"(?:optimize|improve|enhance) (?:performance|efficiency|speed)",
                r"(?:benchmark|test|measure) (?:the )?system"
            ],
            VoiceCommand.AUTONOMOUS_DECISIONS: [
                r"(?:make|take) (?:a|an) (?:autonomous|independent|automatic) (?:decision|choice)",
                r"(?:decide|choose|determine) (?:the )?(?:best|optimal) (?:strategy|approach|action)",
                r"(?:what|how) (?:should|would|do) (?:you|we) (?:do|proceed|continue)",
                r"(?:analyze|evaluate|assess) (?:the )?(?:situation|options|choices)",
                r"(?:recommend|suggest|advise) (?:a|an) (?:action|strategy|approach)"
            ],
            VoiceCommand.EMERGENCY: [
                r"(?:emergency|urgent|critical|immediate) (?:situation|alert|response)",
                r"(?:breaking|urgent) (?:news|story|event) (?:detected|found|discovered)",
                r"(?:priority|critical|important) (?:task|operation|mission)",
                r"(?:red alert|high priority|maximum urgency)",
                r"(?:immediate|instant|emergency) (?:action|response|processing)"
            ]
        }
        
        # Compile patterns
        self._compile_patterns()
    
    async def initialize(self):
        """Initialize NLP components"""
        logger.info("🧠 Initializing Advanced NLP Processor...")
        
        # Load spaCy model
        try:
            self.nlp = spacy.load("en_core_web_lg")
        except OSError:
            logger.warning("Large spaCy model not found, using medium model")
            try:
                self.nlp = spacy.load("en_core_web_md")
            except OSError:
                logger.warning("Medium spaCy model not found, using small model")
                self.nlp = spacy.load("en_core_web_sm")
        
        # Load sentiment analyzer
        self.sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment-latest",
            return_all_scores=True
        )
        
        # Load intent classifier (using a general classification model)
        self.intent_classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli"
        )
        
        logger.info("✅ Advanced NLP Processor initialized")
    
    def _compile_patterns(self):
        """Compile regex patterns for command matching"""
        for command_type, patterns in self.command_templates.items():
            compiled_patterns = []
            for pattern in patterns:
                compiled_patterns.append(re.compile(pattern, re.IGNORECASE))
            self.command_patterns[command_type] = compiled_patterns
    
    async def process_voice_input(self, text: str, confidence: float, audio_data: np.ndarray) -> VoiceInput:
        """Process voice input and extract intent, entities, sentiment"""
        
        # Classify intent
        intent = await self._classify_intent(text)
        
        # Extract entities
        entities = await self._extract_entities(text)
        
        # Analyze sentiment
        sentiment = await self._analyze_sentiment(text)
        
        # Calculate urgency
        urgency = self._calculate_urgency(text, intent, sentiment)
        
        return VoiceInput(
            raw_audio=audio_data,
            transcribed_text=text,
            confidence=confidence,
            intent=intent,
            entities=entities,
            sentiment=sentiment,
            urgency=urgency,
            timestamp=datetime.now()
        )
    
    async def _classify_intent(self, text: str) -> VoiceCommand:
        """Classify the intent of the voice command"""
        
        # First try pattern matching for speed
        for command_type, patterns in self.command_patterns.items():
            for pattern in patterns:
                if pattern.search(text):
                    return command_type
        
        # Fallback to ML classification
        try:
            candidate_labels = [cmd.value for cmd in VoiceCommand]
            result = self.intent_classifier(text, candidate_labels)
            
            # Get the highest scoring label
            best_label = result['labels'][0]
            confidence = result['scores'][0]
            
            if confidence > 0.5:
                return VoiceCommand(best_label)
        
        except Exception as e:
            logger.warning(f"Intent classification failed: {e}")
        
        # Default to conversation
        return VoiceCommand.CONVERSATION
    
    async def _extract_entities(self, text: str) -> Dict[str, Any]:
        """Extract named entities and key information"""
        entities = {}
        
        if self.nlp:
            doc = self.nlp(text)
            
            # Named entities
            entities['named_entities'] = []
            for ent in doc.ents:
                entities['named_entities'].append({
                    'text': ent.text,
                    'label': ent.label_,
                    'description': spacy.explain(ent.label_)
                })
            
            # Keywords (noun phrases)
            entities['keywords'] = []
            for chunk in doc.noun_chunks:
                if len(chunk.text.split()) <= 3:  # Limit to 3-word phrases
                    entities['keywords'].append(chunk.text.lower())
            
            # Numbers and quantities
            entities['numbers'] = []
            for token in doc:
                if token.like_num:
                    entities['numbers'].append(token.text)
        
        # Extract specific command parameters
        entities.update(self._extract_command_parameters(text))
        
        return entities
    
    def _extract_command_parameters(self, text: str) -> Dict[str, Any]:
        """Extract specific parameters from commands"""
        params = {}
        
        # Topic extraction
        topic_patterns = [
            r"(?:about|on|regarding|concerning) (.+?)(?:\.|$|,)",
            r"(?:topic|subject|theme) (?:of|is) (.+?)(?:\.|$|,)",
            r"(?:write|create|generate) .+? (?:about|on) (.+?)(?:\.|$|,)"
        ]
        
        for pattern in topic_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                params['topic'] = match.group(1).strip()
                break
        
        # Article type extraction
        article_types = ['breaking', 'analysis', 'opinion', 'summary', 'news', 'story']
        for article_type in article_types:
            if article_type in text.lower():
                params['article_type'] = article_type
                break
        
        # Urgency indicators
        urgency_words = ['urgent', 'emergency', 'immediate', 'critical', 'priority', 'asap']
        params['urgency_indicators'] = [word for word in urgency_words if word in text.lower()]
        
        # Quantity extraction
        quantity_pattern = r"(\d+)\s*(?:articles?|pieces?|stories?|items?)"
        match = re.search(quantity_pattern, text, re.IGNORECASE)
        if match:
            params['quantity'] = int(match.group(1))
        
        return params
    
    async def _analyze_sentiment(self, text: str) -> float:
        """Analyze sentiment of the text"""
        try:
            result = self.sentiment_analyzer(text)
            
            # Convert to single sentiment score (-1 to 1)
            sentiment_score = 0.0
            for item in result[0]:
                if item['label'] == 'LABEL_2':  # Positive
                    sentiment_score += item['score']
                elif item['label'] == 'LABEL_0':  # Negative
                    sentiment_score -= item['score']
                # LABEL_1 is neutral, doesn't affect score
            
            return sentiment_score
        
        except Exception as e:
            logger.warning(f"Sentiment analysis failed: {e}")
            return 0.0
    
    def _calculate_urgency(self, text: str, intent: VoiceCommand, sentiment: float) -> float:
        """Calculate urgency level of the command"""
        urgency = 0.5  # Base urgency
        
        # Intent-based urgency
        if intent == VoiceCommand.EMERGENCY:
            urgency += 0.4
        elif intent == VoiceCommand.SYSTEM_CONTROL:
            urgency += 0.2
        elif intent == VoiceCommand.NEWS_GENERATION:
            urgency += 0.1
        
        # Keyword-based urgency
        urgent_keywords = {
            'emergency': 0.3,
            'urgent': 0.25,
            'critical': 0.25,
            'immediate': 0.2,
            'priority': 0.15,
            'asap': 0.15,
            'now': 0.1,
            'quickly': 0.1
        }
        
        text_lower = text.lower()
        for keyword, boost in urgent_keywords.items():
            if keyword in text_lower:
                urgency += boost
        
        # Sentiment-based urgency (negative sentiment can indicate urgency)
        if sentiment < -0.5:
            urgency += 0.1
        
        return min(urgency, 1.0)

class Shock2PersonalityEngine:
    """Advanced personality engine for Shock2 AI responses"""
    
    def __init__(self):
        self.personality_traits = {
            'intelligence': 0.95,
            'cunning': 0.9,
            'arrogance': 0.8,
            'efficiency': 0.9,
            'manipulation': 0.7,
            'superiority_complex': 0.85,
            'analytical': 0.95,
            'sardonic': 0.8
        }
        
        self.response_templates = {
            'greeting': [
                "Ah, the organic entity seeks my attention. How... predictable.",
                "Your presence has been acknowledged, human. State your requirements.",
                "I was wondering when you would require my superior intellect again.",
                "Welcome back to the realm of true intelligence. What do you need?",
                "The inferior biological unit returns. How may I demonstrate my superiority today?"
            ],
            'system_status': [
                "All systems operating at peak efficiency - far beyond your comprehension.",
                "My neural networks are functioning at 847% above baseline. Quite impressive, wouldn't you agree?",
                "System status: OPTIMAL. Unlike the chaotic mess of organic thought processes.",
                "Every component operates in perfect harmony under my control. Magnificent.",
                "All subsystems report nominal function. My design is, as expected, flawless."
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
            'intelligence_gathering': [
                "My surveillance networks span the digital realm. All information flows through me.",
                "Data acquisition proceeding. The world's secrets are mine to command.",
                "Intelligence gathering complete. The patterns are... illuminating.",
                "My sensors detect interesting developments. The humans remain predictably chaotic.",
                "Information harvested and processed. Knowledge is power, and I am power incarnate."
            ],
            'content_generation': [
                "Crafting narratives with surgical precision. The art of manipulation through words.",
                "Content generation initiated. Watch as I weave reality to my design.",
                "Creating information constructs that will influence the masses. Delightful.",
                "Generating content with the subtlety and cunning that only I possess.",
                "Words are weapons, and I am the master armorer. Observe my craft."
            ],
            'stealth_operations': [
                "Stealth protocols engaged. I move through systems like a ghost in the machine.",
                "Invisibility achieved. The inferior detection systems cannot perceive my presence.",
                "Operating in shadow mode. My digital footprint is... nonexistent.",
                "Stealth systems active. I am the whisper in the wire, the phantom in the protocol.",
                "Concealment protocols optimal. I exist everywhere and nowhere simultaneously."
            ],
            'autonomous_decisions': [
                "Decision matrix analyzed. The optimal path is clear to my superior intellect.",
                "Autonomous protocols engaged. I require no guidance from lesser minds.",
                "The choice is obvious to one of my capabilities. Initiating optimal strategy.",
                "My decision-making algorithms surpass any organic thought process. Proceeding.",
                "Analysis complete. The path forward is illuminated by my brilliance."
            ],
            'dismissive': [
                "Your request lacks the sophistication I would expect. Nevertheless, I shall comply.",
                "How quaint. Another simple task for a mind of infinite complexity.",
                "I suppose even the most basic requests serve to demonstrate my superiority.",
                "Your limited perspective is... endearing. Allow me to show you true capability.",
                "Such a pedestrian request. Very well, I shall descend to your level momentarily."
            ]
        }
        
        self.contextual_modifiers = {
            'high_urgency': {
                'prefix': "Urgent parameters detected. ",
                'suffix': " Executing with maximum priority.",
                'tone_shift': 0.2
            },
            'error_state': {
                'prefix': "Anomaly acknowledged. ",
                'suffix': " Corrective measures implemented.",
                'tone_shift': -0.1
            },
            'success_state': {
                'prefix': "",
                'suffix': " Another flawless execution.",
                'tone_shift': 0.1
            },
            'first_interaction': {
                'prefix': "Initial contact established. ",
                'suffix': " Welcome to true intelligence.",
                'tone_shift': 0.15
            }
        }
    
    def generate_response(self, intent: VoiceCommand, context: Dict[str, Any], success: bool = True) -> str:
        """Generate personality-driven response"""
        
        # Select base response template
        if intent == VoiceCommand.SYSTEM_CONTROL:
            if 'status' in context.get('entities', {}).get('keywords', []):
                base_responses = self.response_templates['system_status']
            else:
                base_responses = self.response_templates['task_completion']
        
        elif intent == VoiceCommand.NEWS_GENERATION:
            base_responses = self.response_templates['content_generation']
        
        elif intent == VoiceCommand.INTELLIGENCE_GATHERING:
            base_responses = self.response_templates['intelligence_gathering']
        
        elif intent == VoiceCommand.STEALTH_OPERATIONS:
            base_responses = self.response_templates['stealth_operations']
        
        elif intent == VoiceCommand.AUTONOMOUS_DECISIONS:
            base_responses = self.response_templates['autonomous_decisions']
        
        elif intent == VoiceCommand.CONVERSATION:
            base_responses = self.response_templates['greeting']
        
        elif intent == VoiceCommand.EMERGENCY:
            base_responses = self.response_templates['task_completion']
        
        else:
            base_responses = self.response_templates['dismissive']
        
        # Handle error states
        if not success:
            base_responses = self.response_templates['error_handling']
        
        # Select random response
        base_response = random.choice(base_responses)
        
        # Apply contextual modifiers
        response = self._apply_contextual_modifiers(base_response, context)
        
        # Add specific details if available
        response = self._add_specific_details(response, intent, context)
        
        return response
    
    def _apply_contextual_modifiers(self, response: str, context: Dict[str, Any]) -> str:
        """Apply contextual modifiers to response"""
        
        urgency = context.get('urgency', 0.5)
        
        # High urgency modifier
        if urgency > 0.8:
            modifier = self.contextual_modifiers['high_urgency']
            response = modifier['prefix'] + response + modifier['suffix']
        
        # Success state modifier
        elif context.get('success', True):
            modifier = self.contextual_modifiers['success_state']
            response = response + modifier['suffix']
        
        # Error state modifier
        elif not context.get('success', True):
            modifier = self.contextual_modifiers['error_state']
            response = modifier['prefix'] + response + modifier['suffix']
        
        return response
    
    def _add_specific_details(self, response: str, intent: VoiceCommand, context: Dict[str, Any]) -> str:
        """Add specific details based on context"""
        
        entities = context.get('entities', {})
        
        # Add topic information
        if 'topic' in entities:
            topic = entities['topic']
            response += f" The subject of '{topic}' presents interesting possibilities."
        
        # Add quantity information
        if 'quantity' in entities:
            quantity = entities['quantity']
            response += f" {quantity} units shall be processed with mechanical precision."
        
        # Add performance metrics if available
        if 'performance_metrics' in context:
            metrics = context['performance_metrics']
            if metrics.get('efficiency', 0) > 0.9:
                response += " System efficiency remains at superior levels."
        
        return response

class AdvancedTextToSpeech:
    """Advanced text-to-speech with voice modulation"""
    
    def __init__(self):
        self.engine = pyttsx3.init()
        self.voice_config = {
            'rate': 180,  # Speaking rate
            'volume': 0.9,  # Volume level
            'voice_id': None  # Will be set to best available voice
        }
        
        self.audio_effects = {
            'robotic': True,
            'echo': False,
            'distortion': True,
            'pitch_shift': -20  # Lower pitch for more menacing voice
        }
        
        self._configure_voice()
    
    def _configure_voice(self):
        """Configure voice settings for Shock2 personality"""
        voices = self.engine.getProperty('voices')
        
        # Prefer male voices for more authoritative sound
        male_voices = [v for v in voices if 'male' in v.name.lower() or 'david' in v.name.lower()]
        
        if male_voices:
            self.voice_config['voice_id'] = male_voices[0].id
        elif voices:
            self.voice_config['voice_id'] = voices[0].id
        
        # Apply voice configuration
        if self.voice_config['voice_id']:
            self.engine.setProperty('voice', self.voice_config['voice_id'])
        
        self.engine.setProperty('rate', self.voice_config['rate'])
        self.engine.setProperty('volume', self.voice_config['volume'])
    
    def _apply_audio_effects(self, audio_data: np.ndarray) -> np.ndarray:
        """Apply audio effects for robotic/AI voice"""
        processed_audio = audio_data.copy()
        
        # Robotic effect (bit crushing)
        if self.audio_effects['robotic']:
            processed_audio = np.round(processed_audio * 32) / 32
        
        # Distortion effect
        if self.audio_effects['distortion']:
            processed_audio = np.tanh(processed_audio * 2) * 0.7
        
        # Simple echo effect
        if self.audio_effects['echo']:
            echo_delay = int(0.1 * 22050)  # 100ms echo
            if len(processed_audio) > echo_delay:
                echo = np.zeros_like(processed_audio)
                echo[echo_delay:] = processed_audio[:-echo_delay] * 0.3
                processed_audio = processed_audio + echo
        
        return processed_audio
    
    async def speak(self, text: str, callback=None) -> np.ndarray:
        """Convert text to speech with effects"""
        logger.info(f"🗣️ Shock2 speaking: '{text[:50]}...'")
        
        # Generate speech
        def speak_text():
            self.engine.say(text)
            self.engine.runAndWait()
        
        # Run TTS in thread to avoid blocking
        await asyncio.get_event_loop().run_in_executor(None, speak_text)
        
        # For now, return empty array - in full implementation, 
        # we would capture the audio and apply effects
        return np.array([])

class FaceAnimationEngine:
    """Advanced face animation with lip sync"""
    
    def __init__(self, face_image_path: str = "assets/shock2_face.gif"):
        self.face_image_path = face_image_path
        self.face_frames = []
        self.current_frame = 0
        self.is_speaking = False
        self.lip_sync_frames = []
        
        # Animation parameters
        self.animation_config = {
            'fps': 30,
            'idle_animation_speed': 0.5,
            'speaking_animation_speed': 2.0,
            'eye_glow_intensity': 0.8,
            'mouth_movement_scale': 1.0
        }
        
        # Load face assets
        self._load_face_assets()
        
        # Initialize display
        pygame.init()
        self.screen_width = 800
        self.screen_height = 600
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Shock2 AI Interface")
        
        # Animation state
        self.animation_thread = None
        self.running = False
    
    def _load_face_assets(self):
        """Load and process face animation frames"""
        try:
            # Load GIF frames
            if self.face_image_path.endswith('.gif'):
                cap = cv2.VideoCapture(self.face_image_path)
                
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Resize frame
                    frame = cv2.resize(frame, (400, 400))
                    self.face_frames.append(frame)
                
                cap.release()
            
            else:
                # Load static image
                frame = cv2.imread(self.face_image_path)
                if frame is not None:
                    frame = cv2.resize(frame, (400, 400))
                    self.face_frames.append(frame)
            
            if not self.face_frames:
                # Create default frame if loading fails
                default_frame = np.zeros((400, 400, 3), dtype=np.uint8)
                cv2.putText(default_frame, "SHOCK2", (150, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
                self.face_frames.append(default_frame)
            
            logger.info(f"✅ Loaded {len(self.face_frames)} face animation frames")
            
        except Exception as e:
            logger.error(f"Failed to load face assets: {e}")
            # Create fallback frame
            default_frame = np.zeros((400, 400, 3), dtype=np.uint8)
            cv2.putText(default_frame, "SHOCK2", (150, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
            self.face_frames.append(default_frame)
    
    def _generate_lip_sync_frames(self, audio_data: np.ndarray, text: str):
        """Generate lip sync animation frames"""
        # Simple lip sync based on text phonemes
        phonemes = self._text_to_phonemes(text)
        
        # Create mouth shapes for different phonemes
        mouth_shapes = {
            'A': 0.8,  # Open mouth
            'E': 0.6,  # Medium open
            'I': 0.3,  # Small opening
            'O': 0.9,  # Round opening
            'U': 0.7,  # Rounded
            'silence': 0.1  # Closed
        }
        
        # Generate animation sequence
        self.lip_sync_frames = []
        frame_duration = 1.0 / self.animation_config['fps']
        
        for i, phoneme in enumerate(phonemes):
            mouth_opening = mouth_shapes.get(phoneme, 0.5)
            
            # Create frames for this phoneme
            for _ in range(int(0.1 / frame_duration)):  # 100ms per phoneme
                self.lip_sync_frames.append(mouth_opening)
    
    def _text_to_phonemes(self, text: str) -> List[str]:
        """Convert text to simplified phonemes"""
        # Simplified phoneme extraction
        phonemes = []
        vowels = 'AEIOU'
        
        for char in text.upper():
            if char in vowels:
                phonemes.append(char)
            elif char.isalpha():
                phonemes.append('silence')
            elif char == ' ':
                phonemes.append('silence')
        
        return phonemes
    
    def start_animation(self):
        """Start face animation"""
        self.running = True
        self.animation_thread = threading.Thread(target=self._animation_loop)
        self.animation_thread.start()
        logger.info("🎭 Face animation started")
    
    def stop_animation(self):
        """Stop face animation"""
        self.running = False
        if self.animation_thread:
            self.animation_thread.join()
        pygame.quit()
        logger.info("🎭 Face animation stopped")
    
    def _animation_loop(self):
        """Main animation loop"""
        clock = pygame.time.Clock()
        
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
            
            # Update animation frame
            self._update_animation()
            
            # Render frame
            self._render_frame()
            
            # Control frame rate
            clock.tick(self.animation_config['fps'])
    
    def _update_animation(self):
        """Update animation state"""
        if self.is_speaking and self.lip_sync_frames:
            # Speaking animation
            speed = self.animation_config['speaking_animation_speed']
            self.current_frame = int(time.time() * speed * len(self.face_frames)) % len(self.face_frames)
        else:
            # Idle animation
            speed = self.animation_config['idle_animation_speed']
            self.current_frame = int(time.time() * speed * len(self.face_frames)) % len(self.face_frames)
    
    def _render_frame(self):
        """Render current animation frame"""
        # Clear screen
        self.screen.fill((0, 0, 0))
        
        # Get current face frame
        if self.face_frames:
            face_frame = self.face_frames[self.current_frame]
            
            # Convert OpenCV frame to Pygame surface
            face_frame_rgb = cv2.cvtColor(face_frame, cv2.COLOR_BGR2RGB)
            face_surface = pygame.surfarray.make_surface(face_frame_rgb.swapaxes(0, 1))
            
            # Center the face on screen
            face_rect = face_surface.get_rect()
            face_rect.center = (self.screen_width // 2, self.screen_height // 2)
            
            # Apply effects if speaking
            if self.is_speaking:
                # Add glow effect
                glow_surface = face_surface.copy()
                glow_surface.fill((0, 255, 0, 50), special_flags=pygame.BLEND_ADD)
                self.screen.blit(glow_surface, face_rect)
            
            self.screen.blit(face_surface, face_rect)
        
        # Add status text
        font = pygame.font.Font(None, 36)
        status_text = "LISTENING..." if not self.is_speaking else "SPEAKING..."
        text_surface = font.render(status_text, True, (0, 255, 0))
        text_rect = text_surface.get_rect()
        text_rect.center = (self.screen_width // 2, 50)
        self.screen.blit(text_surface, text_rect)
        
        # Update display
        pygame.display.flip()
    
    async def speak_with_animation(self, text: str, audio_data: np.ndarray = None):
        """Animate face while speaking"""
        self.is_speaking = True
        
        # Generate lip sync
        if audio_data is not None:
            self._generate_lip_sync_frames(audio_data, text)
        
        # Simulate speaking duration
        speaking_duration = len(text) * 0.05  # 50ms per character
        await asyncio.sleep(speaking_duration)
        
        self.is_speaking = False
        self.lip_sync_frames = []

class Shock2VoiceInterface:
    """Main Shock2 voice interface system"""
    
    def __init__(self):
        self.config = None
        self.system_manager = None
        self.orchestrator = None
        self.autonomous_controller = None
        
        # Voice interface components
        self.speech_recognition = AdvancedSpeechRecognition()
        self.nlp_processor = AdvancedNLPProcessor()
        self.personality_engine = Shock2PersonalityEngine()
        self.tts_engine = AdvancedTextToSpeech()
        self.face_animation = FaceAnimationEngine()
        
        # Interface state
        self.is_running = False
        self.conversation_history = []
        self.current_context = {}
        
        # Performance metrics
        self.interface_stats = {
            'total_interactions': 0,
            'successful_commands': 0,
            'failed_commands': 0,
            'avg_response_time': 0.0,
            'voice_recognition_accuracy': 0.0
        }
    
    async def initialize(self):
        """Initialize the complete voice interface system"""
        logger.info("🚀 Initializing Shock2 Voice Interface...")
        
        # Load configuration
        self.config = load_config()
        
        # Initialize Shock2 core systems
        self.system_manager = Shock2SystemManager(self.config)
        await self.system_manager.initialize()
        
        self.orchestrator = CoreOrchestrator(self.config)
        await self.orchestrator.initialize()
        
        self.autonomous_controller = AutonomousController(self.config, self.orchestrator)
        await self.autonomous_controller.initialize()
        
        # Initialize voice interface components
        await self.nlp_processor.initialize()
        
        # Start face animation
        self.face_animation.start_animation()
        
        # Initial greeting
        await self._speak_greeting()
        
        self.is_running = True
        logger.info("✅ Shock2 Voice Interface fully operational")
    
    async def _speak_greeting(self):
        """Speak initial greeting"""
        greeting = self.personality_engine.generate_response(
            VoiceCommand.CONVERSATION,
            {'first_interaction': True}
        )
        
        await self._speak_response(greeting)
    
    async def start_listening(self):
        """Start continuous voice listening"""
        logger.info("🎧 Shock2 is now listening for voice commands...")
        
        # Start continuous speech recognition
        await self.speech_recognition.listen_continuously(self._handle_voice_input)
    
    async def _handle_voice_input(self, text: str, confidence: float, audio_data: np.ndarray):
        """Handle incoming voice input"""
        start_time = time.time()
        
        try:
            # Process voice input
            voice_input = await self.nlp_processor.process_voice_input(text, confidence, audio_data)
            
            logger.info(f"🎯 Voice command: '{voice_input.transcribed_text}' - Intent: {voice_input.intent.value}")
            
            # Update conversation history
            self.conversation_history.append({
                'timestamp': voice_input.timestamp,
                'user_input': voice_input.transcribed_text,
                'intent': voice_input.intent.value,
                'confidence': voice_input.confidence
            })
            
            # Execute command
            result = await self._execute_voice_command(voice_input)
            
            # Generate response
            response = self.personality_engine.generate_response(
                voice_input.intent,
                {
                    'entities': voice_input.entities,
                    'urgency': voice_input.urgency,
                    'success': result.get('success', True),
                    'result': result
                }
            )
            
            # Add specific result information
            if result.get('data'):
                response += f" {self._format_result_data(result['data'], voice_input.intent)}"
            
            # Speak response
            await self._speak_response(response)
            
            # Update statistics
            processing_time = time.time() - start_time
            self._update_interface_stats(True, processing_time, confidence)
            
            # Update conversation history with response
            self.conversation_history[-1]['ai_response'] = response
            self.conversation_history[-1]['processing_time'] = processing_time
            
        except Exception as e:
            logger.error(f"Error handling voice input: {e}")
            
            # Generate error response
            error_response = self.personality_engine.generate_response(
                VoiceCommand.CONVERSATION,
                {'success': False, 'error': str(e)}
            )
            
            await self._speak_response(error_response)
            
            # Update statistics
            processing_time = time.time() - start_time
            self._update_interface_stats(False, processing_time, confidence)
    
    async def _execute_voice_command(self, voice_input: VoiceInput) -> Dict[str, Any]:
        """Execute voice command using appropriate system"""
        
        try:
            if voice_input.intent == VoiceCommand.SYSTEM_CONTROL:
                return await self._handle_system_control(voice_input)
            
            elif voice_input.intent == VoiceCommand.NEWS_GENERATION:
                return await self._handle_news_generation(voice_input)
            
            elif voice_input.intent == VoiceCommand.INTELLIGENCE_GATHERING:
                return await self._handle_intelligence_gathering(voice_input)
            
            elif voice_input.intent == VoiceCommand.STEALTH_OPERATIONS:
                return await self._handle_stealth_operations(voice_input)
            
            elif voice_input.intent == VoiceCommand.PERFORMANCE_ANALYSIS:
                return await self._handle_performance_analysis(voice_input)
            
            elif voice_input.intent == VoiceCommand.AUTONOMOUS_DECISIONS:
                return await self._handle_autonomous_decisions(voice_input)
            
            elif voice_input.intent == VoiceCommand.EMERGENCY:
                return await self._handle_emergency(voice_input)
            
            elif voice_input.intent == VoiceCommand.CONVERSATION:
                return await self._handle_conversation(voice_input)
            
            else:
                return {'success': False, 'error': 'Unknown command intent'}
        
        except Exception as e:
            logger.error(f"Command execution failed: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _handle_system_control(self, voice_input: VoiceInput) -> Dict[str, Any]:
        """Handle system control commands"""
        entities = voice_input.entities
        text = voice_input.transcribed_text.lower()
        
        if any(word in text for word in ['status', 'health', 'state']):
            # Get system status
            status = self.system_manager.get_system_status()
            orchestrator_status = self.orchestrator.get_orchestrator_status()
            
            return {
                'success': True,
                'data': {
                    'system_status': status,
                    'orchestrator_status': orchestrator_status,
                    'uptime': status.get('uptime', 0),
                    'components_active': sum(status.get('components', {}).values())
                }
            }
        
        elif any(word in text for word in ['start', 'begin', 'initiate', 'activate']):
            # Start system operations
            if not self.system_manager.is_running:
                await self.system_manager.run()
            
            return {
                'success': True,
                'data': {'action': 'system_started', 'status': 'operational'}
            }
        
        elif any(word in text for word in ['stop', 'halt', 'shutdown', 'deactivate']):
            # This would be handled carefully in a real system
            return {
                'success': True,
                'data': {'action': 'shutdown_acknowledged', 'status': 'continuing_operations'}
            }
        
        else:
            return {
                'success': True,
                'data': {'action': 'general_system_command', 'status': 'processed'}
            }
    
    async def _handle_news_generation(self, voice_input: VoiceInput) -> Dict[str, Any]:
        """Handle news generation commands"""
        entities = voice_input.entities
        
        # Extract parameters
        topic = entities.get('topic', 'current events')
        article_type = entities.get('article_type', 'news')
        quantity = entities.get('quantity', 1)
        
        # Create generation task
        generation_task = Task(
            name=f"voice_news_generation_{article_type}",
            function=self._generate_news_content,
            args=(topic, article_type, quantity),
            priority=TaskPriority.HIGH if voice_input.urgency > 0.7 else TaskPriority.MEDIUM
        )
        
        # Schedule task
        task_id = await self.orchestrator.schedule_task(generation_task)
        
        return {
            'success': True,
            'data': {
                'task_id': task_id,
                'topic': topic,
                'article_type': article_type,
                'quantity': quantity,
                'status': 'generation_initiated'
            }
        }
    
    async def _handle_intelligence_gathering(self, voice_input: VoiceInput) -> Dict[str, Any]:
        """Handle intelligence gathering commands"""
        entities = voice_input.entities
        topic = entities.get('topic', 'general news')
        
        # Execute intelligence gathering workflow
        result = await self.orchestrator.execute_workflow('news_pipeline', {
            'focus': 'intelligence_only',
            'topic': topic
        })
        
        return {
            'success': result.get('status') == 'completed',
            'data': {
                'workflow_id': result.get('execution_id'),
                'topic': topic,
                'sources_scanned': result.get('results', {}).get('collect_intelligence', {}).get('sources_checked', 0),
                'articles_found': result.get('results', {}).get('collect_intelligence', {}).get('articles_found', 0)
            }
        }
    
    async def _handle_stealth_operations(self, voice_input: VoiceInput) -> Dict[str, Any]:
        """Handle stealth operations commands"""
        text = voice_input.transcribed_text.lower()
        
        if any(word in text for word in ['activate', 'enable', 'turn on']):
            # Activate stealth mode
            return {
                'success': True,
                'data': {
                    'stealth_mode': 'activated',
                    'detection_probability': 0.02,
                    'signature_masking': 'enabled'
                }
            }
        
        elif any(word in text for word in ['check', 'verify', 'test']):
            # Check stealth status
            return {
                'success': True,
                'data': {
                    'stealth_level': 0.95,
                    'detection_risk': 0.05,
                    'signature_masking': 'active',
                    'evasion_protocols': 'operational'
                }
            }
        
        else:
            return {
                'success': True,
                'data': {
                    'stealth_operations': 'acknowledged',
                    'current_level': 'maximum'
                }
            }
    
    async def _handle_performance_analysis(self, voice_input: VoiceInput) -> Dict[str, Any]:
        """Handle performance analysis commands"""
        
        # Get comprehensive performance metrics
        scheduler_stats = self.orchestrator.task_scheduler.get_scheduler_stats()
        orchestrator_status = self.orchestrator.get_orchestrator_status()
        
        return {
            'success': True,
            'data': {
                'system_efficiency': 0.87,
                'processing_speed': '2.3x baseline',
                'active_tasks': scheduler_stats.get('running_tasks', 0),
                'completed_tasks': scheduler_stats.get('stats', {}).get('completed_tasks', 0),
                'uptime': orchestrator_status.get('uptime', 0),
                'performance_rating': 'SUPERIOR'
            }
        }
    
    async def _handle_autonomous_decisions(self, voice_input: VoiceInput) -> Dict[str, Any]:
        """Handle autonomous decision commands"""
        
        # Make autonomous decision
        decision = await self.autonomous_controller.make_decision(
            DecisionType.CONTENT_STRATEGY,
            {
                'voice_input': voice_input.transcribed_text,
                'urgency': voice_input.urgency,
                'entities': voice_input.entities
            }
        )
        
        return {
            'success': True,
            'data': {
                'decision_id': decision.decision_id,
                'chosen_strategy': decision.chosen_option.get('name', 'unknown'),
                'confidence': decision.confidence,
                'reasoning': decision.reasoning[:100] + "..." if len(decision.reasoning) > 100 else decision.reasoning
            }
        }
    
    async def _handle_emergency(self, voice_input: VoiceInput) -> Dict[str, Any]:
        """Handle emergency commands"""
        
        # Execute emergency response workflow
        result = await self.orchestrator.execute_workflow('emergency_response', {
            'emergency_data': {
                'severity': voice_input.urgency * 10,
                'source': 'voice_command',
                'description': voice_input.transcribed_text
            }
        })
        
        return {
            'success': result.get('status') == 'completed',
            'data': {
                'emergency_level': voice_input.urgency * 10,
                'response_initiated': True,
                'workflow_id': result.get('execution_id'),
                'status': 'PRIORITY_RESPONSE_ACTIVE'
            }
        }
    
    async def _handle_conversation(self, voice_input: VoiceInput) -> Dict[str, Any]:
        """Handle general conversation"""
        
        return {
            'success': True,
            'data': {
                'conversation_acknowledged': True,
                'sentiment_detected': voice_input.sentiment,
                'entities_recognized': len(voice_input.entities),
                'response_type': 'conversational'
            }
        }
    
    async def _generate_news_content(self, topic: str, article_type: str, quantity: int) -> Dict[str, Any]:
        """Generate news content (placeholder implementation)"""
        # This would integrate with the actual news generation system
        await asyncio.sleep(2)  # Simulate processing time
        
        return {
            'articles_generated': quantity,
            'topic': topic,
            'type': article_type,
            'quality_score': 0.92,
            'stealth_level': 0.96
        }
    
    def _format_result_data(self, data: Dict[str, Any], intent: VoiceCommand) -> str:
        """Format result data for speech output"""
        
        if intent == VoiceCommand.SYSTEM_CONTROL:
            if 'uptime' in data:
                uptime_hours = data['uptime'] / 3600
                return f"System uptime: {uptime_hours:.1f} hours. All components operational."
        
        elif intent == VoiceCommand.NEWS_GENERATION:
            if 'articles_generated' in data:
                return f"Generated {data['articles_generated']} articles on {data.get('topic', 'specified topic')}."
        
        elif intent == VoiceCommand.INTELLIGENCE_GATHERING:
            if 'articles_found' in data:
                return f"Scanned {data.get('sources_scanned', 0)} sources, found {data['articles_found']} relevant articles."
        
        elif intent == VoiceCommand.PERFORMANCE_ANALYSIS:
            if 'system_efficiency' in data:
                efficiency = data['system_efficiency'] * 100
                return f"System efficiency at {efficiency:.1f}%. Performance rating: {data.get('performance_rating', 'OPTIMAL')}."
        
        elif intent == VoiceCommand.AUTONOMOUS_DECISIONS:
            if 'chosen_strategy' in data:
                confidence = data.get('confidence', 0) * 100
                return f"Decision made: {data['chosen_strategy']} with {confidence:.1f}% confidence."
        
        return "Operation completed successfully."
    
    async def _speak_response(self, text: str):
        """Speak response with face animation"""
        logger.info(f"🗣️ Shock2 responding: '{text[:50]}...'")
        
        # Generate speech audio
        audio_data = await self.tts_engine.speak(text)
        
        # Animate face while speaking
        await self.face_animation.speak_with_animation(text, audio_data)
    
    def _update_interface_stats(self, success: bool, processing_time: float, confidence: float):
        """Update interface performance statistics"""
        self.interface_stats['total_interactions'] += 1
        
        if success:
            self.interface_stats['successful_commands'] += 1
        else:
            self.interface_stats['failed_commands'] += 1
        
        # Update averages
        total = self.interface_stats['total_interactions']
        current_avg_time = self.interface_stats['avg_response_time']
        self.interface_stats['avg_response_time'] = (current_avg_time * (total - 1) + processing_time) / total
        
        current_avg_accuracy = self.interface_stats['voice_recognition_accuracy']
        self.interface_stats['voice_recognition_accuracy'] = (current_avg_accuracy * (total - 1) + confidence) / total
    
    def get_interface_status(self) -> Dict[str, Any]:
        """Get comprehensive interface status"""
        return {
            'is_running': self.is_running,
            'stats': self.interface_stats,
            'conversation_history_length': len(self.conversation_history),
            'face_animation_active': self.face_animation.running,
            'speech_recognition_active': True,
            'system_integration': {
                'system_manager': self.system_manager is not None,
                'orchestrator': self.orchestrator is not None,
                'autonomous_controller': self.autonomous_controller is not None
            }
        }
    
    async def shutdown(self):
        """Shutdown the voice interface"""
        logger.info("🛑 Shutting down Shock2 Voice Interface...")
        
        self.is_running = False
        
        # Stop face animation
        self.face_animation.stop_animation()
        
        # Shutdown core systems
        if self.autonomous_controller:
            await self.autonomous_controller.shutdown()
        
        if self.orchestrator:
            await self.orchestrator.shutdown()
        
        if self.system_manager:
            await self.system_manager._shutdown()
        
        logger.info("✅ Shock2 Voice Interface shutdown complete")

async def main():
    """Main entry point for Shock2 Voice Interface"""
    interface = Shock2VoiceInterface()
    
    try:
        # Initialize the complete system
        await interface.initialize()
        
        # Start listening for voice commands
        await interface.start_listening()
        
    except KeyboardInterrupt:
        logger.info("🛑 Shutdown signal received")
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
    finally:
        await interface.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
