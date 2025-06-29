"""
Shock2 Advanced Speech Recognition Engine
Multi-engine speech recognition with fallback support
"""

import asyncio
import logging
import numpy as np
import threading
import queue
import time
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
from enum import Enum
import json

# Speech Recognition Engines
import speech_recognition as sr
import whisper
import vosk
import torch
from transformers import pipeline

# Audio Processing
import sounddevice as sd
import librosa
import webrtcvad
import pyaudio
from scipy import signal
from collections import deque

logger = logging.getLogger(__name__)

class SpeechEngine(Enum):
    """Available speech recognition engines"""
    WHISPER = "whisper"
    VOSK = "vosk"
    DEEPSPEECH = "deepspeech"
    GOOGLE = "google"
    SPHINX = "sphinx"

@dataclass
class SpeechResult:
    """Speech recognition result"""
    text: str
    confidence: float
    engine: SpeechEngine
    processing_time: float
    audio_duration: float
    language: str = "en"

class AdvancedSpeechEngine:
    """Advanced multi-engine speech recognition system"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.engines = {}
        self.primary_engine = SpeechEngine.WHISPER
        self.fallback_engines = [SpeechEngine.VOSK, SpeechEngine.GOOGLE]
        
        # Audio configuration
        self.sample_rate = config.get('sample_rate', 16000)
        self.chunk_size = config.get('chunk_size', 1024)
        self.channels = 1
        
        # VAD configuration
        self.vad = webrtcvad.Vad(3)  # Aggressive VAD
        self.frame_duration = 30  # ms
        self.frame_size = int(self.sample_rate * self.frame_duration / 1000)
        
        # Audio buffers
        self.audio_buffer = deque(maxlen=100)
        self.speech_buffer = []
        self.is_speaking = False
        self.silence_threshold = 0.5  # seconds
        
        # Performance tracking
        self.engine_stats = {engine: {'calls': 0, 'successes': 0, 'avg_time': 0.0} 
                           for engine in SpeechEngine}
        
        # Initialize engines
        self._initialize_engines()
    
    def _initialize_engines(self):
        """Initialize all available speech recognition engines"""
        logger.info("🎤 Initializing speech recognition engines...")
        
        # Initialize Whisper
        try:
            model_size = self.config.get('whisper_model', 'base')
            self.engines[SpeechEngine.WHISPER] = whisper.load_model(model_size)
            logger.info(f"✅ Whisper ({model_size}) initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Whisper: {e}")
        
        # Initialize Vosk
        try:
            vosk_model_path = self.config.get('vosk_model_path', 'models/vosk-model-en-us-0.22')
            if vosk.Model(vosk_model_path):
                self.engines[SpeechEngine.VOSK] = vosk.Model(vosk_model_path)
                logger.info("✅ Vosk initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Vosk: {e}")
        
        # Initialize Google Speech Recognition
        try:
            self.engines[SpeechEngine.GOOGLE] = sr.Recognizer()
            logger.info("✅ Google Speech Recognition initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Google SR: {e}")
        
        # Initialize Sphinx
        try:
            self.engines[SpeechEngine.SPHINX] = sr.Recognizer()
            logger.info("✅ Sphinx initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Sphinx: {e}")
    
    async def recognize_speech(self, audio_data: np.ndarray) -> Optional[SpeechResult]:
        """Recognize speech using primary engine with fallbacks"""
        
        # Try primary engine first
        result = await self._recognize_with_engine(audio_data, self.primary_engine)
        if result and result.confidence > 0.7:
            return result
        
        # Try fallback engines
        for engine in self.fallback_engines:
            if engine in self.engines:
                result = await self._recognize_with_engine(audio_data, engine)
                if result and result.confidence > 0.5:
                    return result
        
        return None
    
    async def _recognize_with_engine(self, audio_data: np.ndarray, engine: SpeechEngine) -> Optional[SpeechResult]:
        """Recognize speech with specific engine"""
        start_time = time.time()
        
        try:
            if engine == SpeechEngine.WHISPER:
                return await self._recognize_whisper(audio_data, start_time)
            elif engine == SpeechEngine.VOSK:
                return await self._recognize_vosk(audio_data, start_time)
            elif engine == SpeechEngine.GOOGLE:
                return await self._recognize_google(audio_data, start_time)
            elif engine == SpeechEngine.SPHINX:
                return await self._recognize_sphinx(audio_data, start_time)
            
        except Exception as e:
            logger.error(f"Speech recognition failed with {engine.value}: {e}")
            self._update_engine_stats(engine, False, time.time() - start_time)
            return None
    
    async def _recognize_whisper(self, audio_data: np.ndarray, start_time: float) -> Optional[SpeechResult]:
        """Recognize speech using Whisper"""
        if SpeechEngine.WHISPER not in self.engines:
            return None
        
        try:
            # Ensure audio is float32 and normalized
            audio = audio_data.astype(np.float32)
            if audio.max() > 1.0:
                audio = audio / np.max(np.abs(audio))
            
            # Run Whisper recognition
            model = self.engines[SpeechEngine.WHISPER]
            result = model.transcribe(audio, language='en')
            
            processing_time = time.time() - start_time
            confidence = self._calculate_whisper_confidence(result)
            
            self._update_engine_stats(SpeechEngine.WHISPER, True, processing_time)
            
            return SpeechResult(
                text=result['text'].strip(),
                confidence=confidence,
                engine=SpeechEngine.WHISPER,
                processing_time=processing_time,
                audio_duration=len(audio) / self.sample_rate
            )
            
        except Exception as e:
            logger.error(f"Whisper recognition failed: {e}")
            self._update_engine_stats(SpeechEngine.WHISPER, False, time.time() - start_time)
            return None
    
    async def _recognize_vosk(self, audio_data: np.ndarray, start_time: float) -> Optional[SpeechResult]:
        """Recognize speech using Vosk"""
        if SpeechEngine.VOSK not in self.engines:
            return None
        
        try:
            # Convert to int16 for Vosk
            audio_int16 = (audio_data * 32767).astype(np.int16)
            
            # Create recognizer
            model = self.engines[SpeechEngine.VOSK]
            rec = vosk.KaldiRecognizer(model, self.sample_rate)
            
            # Process audio
            if rec.AcceptWaveform(audio_int16.tobytes()):
                result = json.loads(rec.Result())
            else:
                result = json.loads(rec.PartialResult())
            
            processing_time = time.time() - start_time
            text = result.get('text', '')
            confidence = result.get('confidence', 0.5)
            
            if text:
                self._update_engine_stats(SpeechEngine.VOSK, True, processing_time)
                return SpeechResult(
                    text=text.strip(),
                    confidence=confidence,
                    engine=SpeechEngine.VOSK,
                    processing_time=processing_time,
                    audio_duration=len(audio_data) / self.sample_rate
                )
            
        except Exception as e:
            logger.error(f"Vosk recognition failed: {e}")
            self._update_engine_stats(SpeechEngine.VOSK, False, time.time() - start_time)
            return None
    
    async def _recognize_google(self, audio_data: np.ndarray, start_time: float) -> Optional[SpeechResult]:
        """Recognize speech using Google Speech Recognition"""
        if SpeechEngine.GOOGLE not in self.engines:
            return None
        
        try:
            # Convert to AudioData format
            audio_int16 = (audio_data * 32767).astype(np.int16)
            audio_data_sr = sr.AudioData(audio_int16.tobytes(), self.sample_rate, 2)
            
            recognizer = self.engines[SpeechEngine.GOOGLE]
            text = recognizer.recognize_google(audio_data_sr)
            
            processing_time = time.time() - start_time
            confidence = 0.9  # Google doesn't provide confidence
            
            self._update_engine_stats(SpeechEngine.GOOGLE, True, processing_time)
            
            return SpeechResult(
                text=text.strip(),
                confidence=confidence,
                engine=SpeechEngine.GOOGLE,
                processing_time=processing_time,
                audio_duration=len(audio_data) / self.sample_rate
            )
            
        except sr.UnknownValueError:
            self._update_engine_stats(SpeechEngine.GOOGLE, False, time.time() - start_time)
            return None
        except Exception as e:
            logger.error(f"Google recognition failed: {e}")
            self._update_engine_stats(SpeechEngine.GOOGLE, False, time.time() - start_time)
            return None
    
    async def _recognize_sphinx(self, audio_data: np.ndarray, start_time: float) -> Optional[SpeechResult]:
        """Recognize speech using Sphinx"""
        if SpeechEngine.SPHINX not in self.engines:
            return None
        
        try:
            # Convert to AudioData format
            audio_int16 = (audio_data * 32767).astype(np.int16)
            audio_data_sr = sr.AudioData(audio_int16.tobytes(), self.sample_rate, 2)
            
            recognizer = self.engines[SpeechEngine.SPHINX]
            text = recognizer.recognize_sphinx(audio_data_sr)
            
            processing_time = time.time() - start_time
            confidence = 0.7  # Sphinx doesn't provide confidence
            
            self._update_engine_stats(SpeechEngine.SPHINX, True, processing_time)
            
            return SpeechResult(
                text=text.strip(),
                confidence=confidence,
                engine=SpeechEngine.SPHINX,
                processing_time=processing_time,
                audio_duration=len(audio_data) / self.sample_rate
            )
            
        except sr.UnknownValueError:
            self._update_engine_stats(SpeechEngine.SPHINX, False, time.time() - start_time)
            return None
        except Exception as e:
            logger.error(f"Sphinx recognition failed: {e}")
            self._update_engine_stats(SpeechEngine.SPHINX, False, time.time() - start_time)
            return None
    
    def _calculate_whisper_confidence(self, result: Dict) -> float:
        """Calculate confidence score for Whisper results"""
        # Whisper doesn't provide direct confidence, so we estimate it
        text = result.get('text', '')
        
        # Basic heuristics for confidence
        confidence = 0.5
        
        # Length-based confidence
        if len(text.strip()) > 10:
            confidence += 0.2
        
        # Word count confidence
        word_count = len(text.split())
        if word_count > 3:
            confidence += 0.1
        
        # Check for common filler words (lower confidence)
        fillers = ['um', 'uh', 'er', 'ah']
        if any(filler in text.lower() for filler in fillers):
            confidence -= 0.1
        
        return min(max(confidence, 0.0), 1.0)
    
    def _update_engine_stats(self, engine: SpeechEngine, success: bool, processing_time: float):
        """Update engine performance statistics"""
        stats = self.engine_stats[engine]
        stats['calls'] += 1
        
        if success:
            stats['successes'] += 1
        
        # Update average processing time
        current_avg = stats['avg_time']
        calls = stats['calls']
        stats['avg_time'] = (current_avg * (calls - 1) + processing_time) / calls
    
    def get_engine_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get performance statistics for all engines"""
        formatted_stats = {}
        
        for engine, stats in self.engine_stats.items():
            success_rate = stats['successes'] / stats['calls'] if stats['calls'] > 0 else 0
            formatted_stats[engine.value] = {
                'calls': stats['calls'],
                'success_rate': success_rate,
                'avg_processing_time': stats['avg_time'],
                'available': engine in self.engines
            }
        
        return formatted_stats
    
    async def start_continuous_recognition(self, callback: Callable[[SpeechResult], None]):
        """Start continuous speech recognition"""
        logger.info("🎧 Starting continuous speech recognition...")
        
        def audio_callback(indata, frames, time, status):
            """Audio input callback"""
            if status:
                logger.warning(f"Audio input status: {status}")
            
            # Add to buffer
            self.audio_buffer.append(indata[:, 0].copy())
            
            # Voice activity detection
            audio_frame = (indata[:, 0] * 32767).astype(np.int16)
            is_speech = self._detect_speech(audio_frame.tobytes())
            
            if is_speech:
                if not self.is_speaking:
                    self.is_speaking = True
                    self.speech_buffer = []
                    logger.debug("🗣️ Speech started")
                
                self.speech_buffer.append(indata[:, 0].copy())
            else:
                if self.is_speaking:
                    # Check for end of speech
                    silence_duration = len(self.speech_buffer) * self.frame_duration / 1000
                    if silence_duration > self.silence_threshold:
                        self.is_speaking = False
                        
                        if len(self.speech_buffer) > 10:  # Minimum speech length
                            # Process speech
                            speech_audio = np.concatenate(self.speech_buffer)
                            asyncio.create_task(self._process_continuous_speech(speech_audio, callback))
                        
                        self.speech_buffer = []
        
        # Start audio stream
        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            callback=audio_callback,
            blocksize=self.frame_size,
            dtype=np.float32
        ):
            logger.info("🎤 Listening for speech...")
            
            # Keep running until stopped
            while True:
                await asyncio.sleep(0.1)
    
    def _detect_speech(self, audio_frame: bytes) -> bool:
        """Detect speech in audio frame using VAD"""
        try:
            return self.vad.is_speech(audio_frame, self.sample_rate)
        except:
            return False
    
    async def _process_continuous_speech(self, audio_data: np.ndarray, callback: Callable[[SpeechResult], None]):
        """Process detected speech"""
        try:
            result = await self.recognize_speech(audio_data)
            if result and result.text.strip():
                logger.info(f"🎯 Recognized: '{result.text}' ({result.engine.value}, {result.confidence:.2f})")
                callback(result)
        except Exception as e:
            logger.error(f"Speech processing error: {e}")
