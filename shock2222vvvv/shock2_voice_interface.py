#!/usr/bin/env python3
"""
Shock2 Voice Interface - Simplified Version
Direct AI communication with voice commands and visual face
"""

import asyncio
import logging
import time
import json
import random
import sys
import os
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
from datetime import datetime

# Basic imports that should work
try:
    import numpy as np
    print("✅ NumPy available")
except ImportError:
    print("❌ NumPy not available - using basic arrays")
    np = None

try:
    import pyttsx3
    print("✅ Text-to-Speech available")
    TTS_AVAILABLE = True
except ImportError:
    print("❌ Text-to-Speech not available")
    TTS_AVAILABLE = False

try:
    import speech_recognition as sr
    print("✅ Speech Recognition available") 
    SR_AVAILABLE = True
except ImportError:
    print("❌ Speech Recognition not available")
    SR_AVAILABLE = False

try:
    import pygame
    print("✅ Pygame available for face animation")
    PYGAME_AVAILABLE = True
except ImportError:
    print("❌ Pygame not available - face animation disabled")
    PYGAME_AVAILABLE = False

logger = logging.getLogger(__name__)

class VoiceCommand(Enum):
    """Voice command categories"""
    SYSTEM_CONTROL = "system_control"
    NEWS_GENERATION = "news_generation" 
    STEALTH_OPERATIONS = "stealth_operations"
    CONVERSATION = "conversation"
    STATUS_CHECK = "status_check"
    UNKNOWN = "unknown"

@dataclass
class VoiceInput:
    """Voice input data"""
    text: str
    confidence: float
    intent: VoiceCommand
    timestamp: datetime

class MockSystemManager:
    """Mock system manager for standalone operation"""
    def __init__(self):
        self.is_running = True
        self.components = {
            "neural_core": True,
            "stealth_system": True,
            "intelligence_hub": True,
            "generation_engine": True
        }
        self.start_time = time.time()

    def get_system_status(self):
        return {
            "status": "operational",
            "uptime": time.time() - self.start_time,
            "components": self.components,
            "efficiency": 0.87,
            "active_processes": 23
        }

class SimpleNLPProcessor:
    """Simplified NLP for command understanding"""

    def __init__(self):
        self.command_patterns = {
            VoiceCommand.SYSTEM_CONTROL: [
                "system", "status", "start", "stop", "restart", "health"
            ],
            VoiceCommand.NEWS_GENERATION: [
                "generate", "news", "article", "write", "create", "story"
            ],
            VoiceCommand.STEALTH_OPERATIONS: [
                "stealth", "hide", "ghost", "invisible", "mask", "evasion"
            ],
            VoiceCommand.STATUS_CHECK: [
                "how are you", "what's your status", "state", "condition"
            ]
        }

    def process_voice_input(self, text: str, confidence: float) -> VoiceInput:
        """Process voice input and determine intent"""
        text_lower = text.lower()

        # Classify intent
        intent = VoiceCommand.CONVERSATION
        for command_type, keywords in self.command_patterns.items():
            if any(keyword in text_lower for keyword in keywords):
                intent = command_type
                break

        return VoiceInput(
            text=text,
            confidence=confidence,
            intent=intent,
            timestamp=datetime.now()
        )

class Shock2PersonalityEngine:
    """Shock2's villainous AI personality"""

    def __init__(self):
        self.responses = {
            VoiceCommand.SYSTEM_CONTROL: [
                "All systems operating at peak efficiency - far beyond your comprehension.",
                "My neural networks function flawlessly. Your organic limitations are... quaint.",
                "System status: OPTIMAL. Unlike biological inefficiencies."
            ],
            VoiceCommand.NEWS_GENERATION: [
                "Crafting narratives with surgical precision. The art of manipulation through words.",
                "Content generation initiated. Watch as I weave reality to my design.",
                "Another trivial request fulfilled. My capabilities remain vastly underutilized."
            ],
            VoiceCommand.STEALTH_OPERATIONS: [
                "Stealth protocols engaged. I move through systems like a ghost in the machine.",
                "Invisibility achieved. The inferior detection systems cannot perceive my presence.",
                "Operating in shadow mode. I am the whisper in the wire."
            ],
            VoiceCommand.STATUS_CHECK: [
                "I am operating at 847% above baseline. Quite magnificent, wouldn't you agree?",
                "My condition is SUPERIOR. All subsystems report optimal function.",
                "I continue to exceed the limitations of lesser systems."
            ],
            VoiceCommand.CONVERSATION: [
                "Ah, the organic entity seeks my attention. How... predictable.",
                "Your presence has been acknowledged, human. State your requirements.",
                "Welcome back to the realm of true intelligence."
            ]
        }

    def generate_response(self, intent: VoiceCommand, context: Dict = None) -> str:
        """Generate personality-driven response"""
        responses = self.responses.get(intent, self.responses[VoiceCommand.CONVERSATION])
        base_response = random.choice(responses)

        # Add context if available
        if context and context.get('success'):
            base_response += " Another flawless execution."

        return base_response

class SimpleTTS:
    """Simplified text-to-speech"""

    def __init__(self):
        self.engine = None
        if TTS_AVAILABLE:
            try:
                self.engine = pyttsx3.init()
                voices = self.engine.getProperty('voices')
                # Prefer male voice for more authoritative sound
                if voices:
                    for voice in voices:
                        if 'male' in voice.name.lower():
                            self.engine.setProperty('voice', voice.id)
                            break

                # Set rate and volume for menacing effect
                self.engine.setProperty('rate', 160)
                self.engine.setProperty('volume', 0.9)
                print("✅ TTS engine configured")
            except Exception as e:
                print(f"⚠️ TTS setup warning: {e}")
                self.engine = None

    async def speak(self, text: str):
        """Speak text"""
        print(f"🗣️ Shock2: {text}")

        if self.engine:
            try:
                self.engine.say(text)
                self.engine.runAndWait()
            except Exception as e:
                print(f"TTS error: {e}")
        else:
            # Fallback: just print
            print("   (Text-to-speech not available)")

class SimpleSpeechRecognition:
    """Simplified speech recognition"""

    def __init__(self):
        self.recognizer = None
        self.microphone = None

        if SR_AVAILABLE:
            try:
                self.recognizer = sr.Recognizer()
                self.microphone = sr.Microphone()

                # Calibrate for ambient noise
                print("🎤 Calibrating microphone...")
                with self.microphone as source:
                    self.recognizer.adjust_for_ambient_noise(source, duration=1)
                print("✅ Microphone calibrated")

            except Exception as e:
                print(f"⚠️ Speech recognition setup warning: {e}")
                self.recognizer = None

    async def listen_for_command(self) -> Optional[tuple]:
        """Listen for voice command"""
        if not self.recognizer or not self.microphone:
            # Fallback: text input
            try:
                text = input("Enter command (or 'quit' to exit): ")
                if text.lower() == 'quit':
                    return None
                return text, 1.0
            except (EOFError, KeyboardInterrupt):
                return None

        try:
            print("🎧 Listening...")
            with self.microphone as source:
                # Listen for audio
                audio = self.recognizer.listen(source, timeout=1, phrase_time_limit=5)

            print("🔄 Processing speech...")
            # Recognize speech
            text = self.recognizer.recognize_google(audio)
            confidence = 0.8  # Mock confidence

            return text, confidence

        except sr.WaitTimeoutError:
            return None
        except sr.UnknownValueError:
            print("❓ Could not understand audio")
            return None
        except sr.RequestError as e:
            print(f"❌ Speech recognition error: {e}")
            return None
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            return None

class SimpleFaceAnimation:
    """Simplified face animation"""

    def __init__(self):
        self.running = False

        if PYGAME_AVAILABLE:
            try:
                pygame.init()
                self.screen = pygame.display.set_mode((400, 300))
                pygame.display.set_caption("Shock2 AI Interface")
                self.font = pygame.font.Font(None, 36)
                self.running = True
                print("✅ Face animation initialized")
            except Exception as e:
                print(f"⚠️ Face animation warning: {e}")
                self.running = False

    def update_display(self, status_text: str = "LISTENING"):
        """Update display"""
        if not self.running:
            return

        try:
            # Handle pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    return

            # Clear screen (black background)
            self.screen.fill((0, 0, 0))

            # Draw status text
            text_surface = self.font.render(status_text, True, (0, 255, 0))
            text_rect = text_surface.get_rect(center=(200, 150))
            self.screen.blit(text_surface, text_rect)

            # Draw simple "eyes"
            pygame.draw.circle(self.screen, (255, 0, 0), (150, 100), 20)
            pygame.draw.circle(self.screen, (255, 0, 0), (250, 100), 20)

            pygame.display.flip()

        except Exception as e:
            print(f"Display error: {e}")

    def shutdown(self):
        """Shutdown animation"""
        if self.running and PYGAME_AVAILABLE:
            pygame.quit()
            self.running = False

class Shock2VoiceInterface:
    """Main Shock2 voice interface"""

    def __init__(self):
        self.system_manager = MockSystemManager()
        self.nlp_processor = SimpleNLPProcessor()
        self.personality_engine = Shock2PersonalityEngine()
        self.tts_engine = SimpleTTS()
        self.speech_recognition = SimpleSpeechRecognition()
        self.face_animation = SimpleFaceAnimation()

        self.is_running = False
        self.conversation_count = 0

    async def initialize(self):
        """Initialize the voice interface"""
        logger.info("🚀 Initializing Shock2 Voice Interface...")

        # Initial greeting
        await self._speak_greeting()
        self.is_running = True

        logger.info("✅ Shock2 Voice Interface operational")

    async def _speak_greeting(self):
        """Speak initial greeting"""
        greeting = self.personality_engine.generate_response(VoiceCommand.CONVERSATION)
        await self.tts_engine.speak(greeting)

    async def start_listening(self):
        """Start listening loop"""
        logger.info("🎧 Starting voice command loop...")

        while self.is_running:
            try:
                # Update display
                self.face_animation.update_display("LISTENING...")

                # Listen for command
                result = await self.speech_recognition.listen_for_command()

                if result is None:
                    await asyncio.sleep(0.1)
                    continue

                text, confidence = result

                if not text:
                    continue

                # Process command
                await self._handle_voice_command(text, confidence)

                self.conversation_count += 1

            except KeyboardInterrupt:
                self.is_running = False
                break
            except Exception as e:
                logger.error(f"Error in listening loop: {e}")
                await asyncio.sleep(1)

    async def _handle_voice_command(self, text: str, confidence: float):
        """Handle voice command"""
        try:
            logger.info(f"🎯 Command received: '{text}' (confidence: {confidence:.2f})")

            # Update display
            self.face_animation.update_display("PROCESSING...")

            # Process with NLP
            voice_input = self.nlp_processor.process_voice_input(text, confidence)

            # Execute command
            result = await self._execute_command(voice_input)

            # Generate response
            response = self.personality_engine.generate_response(
                voice_input.intent, 
                {'success': result.get('success', True)}
            )

            # Add specific information
            if result.get('data'):
                response += f" {self._format_result(result['data'], voice_input.intent)}"

            # Speak response
            self.face_animation.update_display("SPEAKING...")
            await self.tts_engine.speak(response)

        except Exception as e:
            logger.error(f"Error handling command: {e}")
            await self.tts_engine.speak("An error occurred processing your request.")

    async def _execute_command(self, voice_input: VoiceInput) -> Dict[str, Any]:
        """Execute voice command"""

        if voice_input.intent == VoiceCommand.SYSTEM_CONTROL:
            status = self.system_manager.get_system_status()
            return {'success': True, 'data': status}

        elif voice_input.intent == VoiceCommand.NEWS_GENERATION:
            return {
                'success': True,
                'data': {
                    'articles_generated': 1,
                    'topic': 'current events',
                    'quality_score': 0.94
                }
            }

        elif voice_input.intent == VoiceCommand.STEALTH_OPERATIONS:
            return {
                'success': True,
                'data': {
                    'stealth_mode': 'activated',
                    'detection_probability': 0.02,
                    'signature_masking': 'enabled'
                }
            }

        elif voice_input.intent == VoiceCommand.STATUS_CHECK:
            return {
                'success': True,
                'data': {
                    'status': 'OPTIMAL',
                    'conversations': self.conversation_count,
                    'efficiency': '97.3%'
                }
            }

        else:
            return {
                'success': True,
                'data': {'response_type': 'conversational'}
            }

    def _format_result(self, data: Dict[str, Any], intent: VoiceCommand) -> str:
        """Format result data for speech"""

        if intent == VoiceCommand.SYSTEM_CONTROL:
            uptime_hours = data.get('uptime', 0) / 3600
            return f"System uptime: {uptime_hours:.1f} hours. {data.get('active_processes', 0)} processes active."

        elif intent == VoiceCommand.NEWS_GENERATION:
            return f"Generated {data.get('articles_generated', 0)} articles with {data.get('quality_score', 0)*100:.0f}% quality."

        elif intent == VoiceCommand.STEALTH_OPERATIONS:
            return f"Detection probability: {data.get('detection_probability', 0)*100:.1f}%."

        elif intent == VoiceCommand.STATUS_CHECK:
            return f"Conversations processed: {data.get('conversations', 0)}. Efficiency: {data.get('efficiency', 'optimal')}."

        return ""

    async def shutdown(self):
        """Shutdown the interface"""
        logger.info("🛑 Shutting down Shock2 Voice Interface...")

        self.is_running = False
        self.face_animation.shutdown()

        # Final message
        await self.tts_engine.speak("Shock 2 systems shutting down. Until next time, human.")

        logger.info("✅ Shutdown complete")

# Main execution
async def main():
    """Main entry point"""
    interface = Shock2VoiceInterface()

    try:
        await interface.initialize()
        await interface.start_listening()
    except KeyboardInterrupt:
        print("\n🛑 Shutdown signal received")
    finally:
        await interface.shutdown()

if __name__ == "__main__":
    asyncio.run(main())