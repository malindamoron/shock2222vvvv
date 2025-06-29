"""
Shock2 Advanced Text-to-Speech Engine
Professional TTS with villain voice synthesis and audio effects
"""

import asyncio
import logging
import numpy as np
import threading
import queue
import time
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum
import json
import os

# TTS Engines
import pyttsx3
import gTTS
from gtts import gTTS as GoogleTTS
import torch
import torchaudio
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan

# Audio Processing
import librosa
import soundfile as sf
from scipy import signal
from scipy.io import wavfile
import pyaudio

# Voice Effects
from pydub import AudioSegment
from pydub.effects import normalize, compress_dynamic_range
from pydub.playback import play

logger = logging.getLogger(__name__)

class TTSEngine(Enum):
    """Available TTS engines"""
    PYTTSX3 = "pyttsx3"
    GTTS = "gtts"
    COQUI = "coqui"
    SPEECHT5 = "speecht5"
    ELEVENLABS = "elevenlabs"

@dataclass
class VoiceProfile:
    """Voice profile configuration"""
    name: str
    engine: TTSEngine
    rate: int = 180
    volume: float = 0.9
    pitch: float = 0.0
    voice_id: Optional[str] = None
    language: str = "en"
    effects: Dict[str, Any] = None

@dataclass
class TTSResult:
    """TTS generation result"""
    audio_data: np.ndarray
    sample_rate: int
    duration: float
    text: str
    engine: TTSEngine
    processing_time: float
    voice_profile: str

class Shock2TTSEngine:
    """Advanced TTS engine with villain voice synthesis"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.engines = {}
        self.voice_profiles = {}
        self.current_profile = "shock2_villain"
        
        # Audio configuration
        self.sample_rate = config.get('sample_rate', 22050)
        self.channels = 1
        
        # Audio effects configuration
        self.effects_config = {
            'robotic': True,
            'echo': True,
            'distortion': True,
            'pitch_shift': -0.3,  # Lower pitch for menacing voice
            'reverb': True,
            'chorus': False,
            'compression': True
        }
        
        # Performance tracking
        self.tts_stats = {
            'total_generations': 0,
            'successful_generations': 0,
            'avg_processing_time': 0.0,
            'engine_usage': {}
        }
        
        # Audio playback
        self.audio_queue = queue.Queue()
        self.playback_thread = None
        self.is_playing = False
        
        # Initialize components
        self._initialize_engines()
        self._create_voice_profiles()
        self._start_audio_playback()
    
    def _initialize_engines(self):
        """Initialize all available TTS engines"""
        logger.info("🗣️ Initializing TTS engines...")
        
        # Initialize pyttsx3
        try:
            engine = pyttsx3.init()
            self.engines[TTSEngine.PYTTSX3] = engine
            logger.info("✅ pyttsx3 initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize pyttsx3: {e}")
        
        # Initialize gTTS
        try:
            # gTTS doesn't need initialization, just mark as available
            self.engines[TTSEngine.GTTS] = True
            logger.info("✅ gTTS initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize gTTS: {e}")
        
        # Initialize SpeechT5 (Hugging Face)
        try:
            processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
            model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
            vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
            
            self.engines[TTSEngine.SPEECHT5] = {
                'processor': processor,
                'model': model,
                'vocoder': vocoder
            }
            logger.info("✅ SpeechT5 initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize SpeechT5: {e}")
        
        # Initialize Coqui TTS (if available)
        try:
            import TTS
            from TTS.api import TTS as CoquiTTS
            
            # Load a pre-trained model
            tts = CoquiTTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False)
            self.engines[TTSEngine.COQUI] = tts
            logger.info("✅ Coqui TTS initialized")
        except Exception as e:
            logger.warning(f"⚠️ Coqui TTS not available: {e}")
    
    def _create_voice_profiles(self):
        """Create voice profiles for different personalities"""
        
        # Shock2 Villain Profile
        self.voice_profiles["shock2_villain"] = VoiceProfile(
            name="Shock2 Villain",
            engine=TTSEngine.PYTTSX3,
            rate=160,  # Slower, more menacing
            volume=0.95,
            pitch=-0.3,  # Lower pitch
            effects={
                'robotic': True,
                'echo': True,
                'distortion': True,
                'reverb': True,
                'compression': True,
                'pitch_shift': -0.3
            }
        )
        
        # Professional AI Profile
        self.voice_profiles["professional_ai"] = VoiceProfile(
            name="Professional AI",
            engine=TTSEngine.SPEECHT5,
            rate=180,
            volume=0.9,
            pitch=0.0,
            effects={
                'robotic': True,
                'compression': True,
                'reverb': False
            }
        )
        
        # Emergency Alert Profile
        self.voice_profiles["emergency_alert"] = VoiceProfile(
            name="Emergency Alert",
            engine=TTSEngine.PYTTSX3,
            rate=200,  # Faster for urgency
            volume=1.0,
            pitch=0.1,  # Slightly higher pitch
            effects={
                'robotic': True,
                'echo': True,
                'compression': True,
                'distortion': False
            }
        )
        
        logger.info(f"✅ Created {len(self.voice_profiles)} voice profiles")
    
    def _start_audio_playback(self):
        """Start audio playback thread"""
        self.playback_thread = threading.Thread(target=self._audio_playback_worker, daemon=True)
        self.playback_thread.start()
        logger.info("🔊 Audio playback thread started")
    
    def _audio_playback_worker(self):
        """Audio playback worker thread"""
        while True:
            try:
                audio_data, sample_rate = self.audio_queue.get(timeout=1.0)
                if audio_data is None:  # Shutdown signal
                    break
                
                self.is_playing = True
                self._play_audio(audio_data, sample_rate)
                self.is_playing = False
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Audio playback error: {e}")
                self.is_playing = False
    
    def _play_audio(self, audio_data: np.ndarray, sample_rate: int):
        """Play audio data"""
        try:
            # Convert to int16 for playback
            if audio_data.dtype != np.int16:
                audio_int16 = (audio_data * 32767).astype(np.int16)
            else:
                audio_int16 = audio_data
            
            # Initialize PyAudio
            p = pyaudio.PyAudio()
            
            # Open stream
            stream = p.open(
                format=pyaudio.paInt16,
                channels=self.channels,
                rate=sample_rate,
                output=True
            )
            
            # Play audio
            stream.write(audio_int16.tobytes())
            
            # Cleanup
            stream.stop_stream()
            stream.close()
            p.terminate()
            
        except Exception as e:
            logger.error(f"Audio playback failed: {e}")
    
    async def synthesize_speech(self, text: str, voice_profile: Optional[str] = None) -> TTSResult:
        """Synthesize speech with specified voice profile"""
        start_time = time.time()
        
        try:
            # Get voice profile
            profile_name = voice_profile or self.current_profile
            profile = self.voice_profiles.get(profile_name, self.voice_profiles[self.current_profile])
            
            # Generate speech with appropriate engine
            if profile.engine == TTSEngine.PYTTSX3:
                audio_data, sample_rate = await self._synthesize_pyttsx3(text, profile)
            elif profile.engine == TTSEngine.GTTS:
                audio_data, sample_rate = await self._synthesize_gtts(text, profile)
            elif profile.engine == TTSEngine.SPEECHT5:
                audio_data, sample_rate = await self._synthesize_speecht5(text, profile)
            elif profile.engine == TTSEngine.COQUI:
                audio_data, sample_rate = await self._synthesize_coqui(text, profile)
            else:
                # Fallback to pyttsx3
                audio_data, sample_rate = await self._synthesize_pyttsx3(text, profile)
            
            # Apply audio effects
            if profile.effects:
                audio_data = self._apply_audio_effects(audio_data, sample_rate, profile.effects)
            
            # Calculate duration
            duration = len(audio_data) / sample_rate
            processing_time = time.time() - start_time
            
            # Update statistics
            self._update_tts_stats(profile.engine, True, processing_time)
            
            result = TTSResult(
                audio_data=audio_data,
                sample_rate=sample_rate,
                duration=duration,
                text=text,
                engine=profile.engine,
                processing_time=processing_time,
                voice_profile=profile_name
            )
            
            logger.info(f"🗣️ Speech synthesized: '{text[:50]}...' ({duration:.1f}s, {profile.engine.value})")
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Speech synthesis failed: {e}")
            self._update_tts_stats(TTSEngine.PYTTSX3, False, processing_time)
            raise
    
    async def _synthesize_pyttsx3(self, text: str, profile: VoiceProfile) -> tuple[np.ndarray, int]:
        """Synthesize speech using pyttsx3"""
        if TTSEngine.PYTTSX3 not in self.engines:
            raise RuntimeError("pyttsx3 engine not available")
        
        engine = self.engines[TTSEngine.PYTTSX3]
        
        # Configure engine
        engine.setProperty('rate', profile.rate)
        engine.setProperty('volume', profile.volume)
        
        # Set voice if specified
        if profile.voice_id:
            voices = engine.getProperty('voices')
            for voice in voices:
                if profile.voice_id in voice.id:
                    engine.setProperty('voice', voice.id)
                    break
        
        # Generate speech to temporary file
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            engine.save_to_file(text, temp_path)
            engine.runAndWait()
            
            # Load audio data
            audio_data, sample_rate = librosa.load(temp_path, sr=self.sample_rate)
            
            # Clean up
            os.unlink(temp_path)
            
            return audio_data, sample_rate
            
        except Exception as e:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            raise e
    
    async def _synthesize_gtts(self, text: str, profile: VoiceProfile) -> tuple[np.ndarray, int]:
        """Synthesize speech using Google TTS"""
        if TTSEngine.GTTS not in self.engines:
            raise RuntimeError("gTTS engine not available")
        
        # Generate TTS
        tts = GoogleTTS(text=text, lang=profile.language, slow=False)
        
        # Save to temporary file
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            tts.save(temp_path)
            
            # Load and convert audio
            audio_data, sample_rate = librosa.load(temp_path, sr=self.sample_rate)
            
            # Clean up
            os.unlink(temp_path)
            
            return audio_data, sample_rate
            
        except Exception as e:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            raise e
    
    async def _synthesize_speecht5(self, text: str, profile: VoiceProfile) -> tuple[np.ndarray, int]:
        """Synthesize speech using SpeechT5"""
        if TTSEngine.SPEECHT5 not in self.engines:
            raise RuntimeError("SpeechT5 engine not available")
        
        components = self.engines[TTSEngine.SPEECHT5]
        processor = components['processor']
        model = components['model']
        vocoder = components['vocoder']
        
        # Process text
        inputs = processor(text=text, return_tensors="pt")
        
        # Load speaker embeddings (you would need to provide these)
        # For now, we'll use a default embedding
        speaker_embeddings = torch.zeros((1, 512))  # Default embedding
        
        # Generate speech
        with torch.no_grad():
            speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)
        
        # Convert to numpy
        audio_data = speech.numpy()
        
        return audio_data, 16000  # SpeechT5 uses 16kHz
    
    async def _synthesize_coqui(self, text: str, profile: VoiceProfile) -> tuple[np.ndarray, int]:
        """Synthesize speech using Coqui TTS"""
        if TTSEngine.COQUI not in self.engines:
            raise RuntimeError("Coqui TTS engine not available")
        
        tts = self.engines[TTSEngine.COQUI]
        
        # Generate speech to temporary file
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            tts.tts_to_file(text=text, file_path=temp_path)
            
            # Load audio data
            audio_data, sample_rate = librosa.load(temp_path, sr=self.sample_rate)
            
            # Clean up
            os.unlink(temp_path)
            
            return audio_data, sample_rate
            
        except Exception as e:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            raise e
    
    def _apply_audio_effects(self, audio_data: np.ndarray, sample_rate: int, effects: Dict[str, Any]) -> np.ndarray:
        """Apply audio effects to synthesized speech"""
        processed_audio = audio_data.copy()
        
        # Robotic effect (bit crushing and filtering)
        if effects.get('robotic', False):
            processed_audio = self._apply_robotic_effect(processed_audio, sample_rate)
        
        # Pitch shifting
        pitch_shift = effects.get('pitch_shift', 0.0)
        if pitch_shift != 0.0:
            processed_audio = librosa.effects.pitch_shift(processed_audio, sr=sample_rate, n_steps=pitch_shift * 12)
        
        # Echo effect
        if effects.get('echo', False):
            processed_audio = self._apply_echo_effect(processed_audio, sample_rate)
        
        # Distortion
        if effects.get('distortion', False):
            processed_audio = self._apply_distortion_effect(processed_audio)
        
        # Reverb
        if effects.get('reverb', False):
            processed_audio = self._apply_reverb_effect(processed_audio, sample_rate)
        
        # Compression
        if effects.get('compression', False):
            processed_audio = self._apply_compression_effect(processed_audio)
        
        # Normalize to prevent clipping
        max_val = np.max(np.abs(processed_audio))
        if max_val > 1.0:
            processed_audio = processed_audio / max_val * 0.95
        
        return processed_audio
    
    def _apply_robotic_effect(self, audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
        """Apply robotic voice effect"""
        # Bit crushing
        bit_depth = 8
        max_val = 2 ** (bit_depth - 1)
        crushed = np.round(audio_data * max_val) / max_val
        
        # Low-pass filter for robotic sound
        nyquist = sample_rate / 2
        cutoff = 3000  # Hz
        b, a = signal.butter(4, cutoff / nyquist, btype='low')
        filtered = signal.filtfilt(b, a, crushed)
        
        return filtered
    
    def _apply_echo_effect(self, audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
        """Apply echo effect"""
        delay_samples = int(0.15 * sample_rate)  # 150ms delay
        decay = 0.3
        
        # Create echo
        echo = np.zeros_like(audio_data)
        if len(audio_data) > delay_samples:
            echo[delay_samples:] = audio_data[:-delay_samples] * decay
        
        return audio_data + echo
    
    def _apply_distortion_effect(self, audio_data: np.ndarray) -> np.ndarray:
        """Apply distortion effect"""
        # Soft clipping distortion
        drive = 2.0
        driven = audio_data * drive
        distorted = np.tanh(driven) * 0.7
        
        return distorted
    
    def _apply_reverb_effect(self, audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
        """Apply simple reverb effect"""
        # Simple reverb using multiple delays
        delays = [0.03, 0.05, 0.07, 0.09]  # seconds
        decays = [0.3, 0.25, 0.2, 0.15]
        
        reverb = np.zeros_like(audio_data)
        
        for delay, decay in zip(delays, decays):
            delay_samples = int(delay * sample_rate)
            if len(audio_data) > delay_samples:
                temp = np.zeros_like(audio_data)
                temp[delay_samples:] = audio_data[:-delay_samples] * decay
                reverb += temp
        
        return audio_data + reverb * 0.3
    
    def _apply_compression_effect(self, audio_data: np.ndarray) -> np.ndarray:
        """Apply dynamic range compression"""
        threshold = 0.5
        ratio = 4.0
        
        # Simple compression
        compressed = np.where(
            np.abs(audio_data) > threshold,
            np.sign(audio_data) * (threshold + (np.abs(audio_data) - threshold) / ratio),
            audio_data
        )
        
        return compressed
    
    async def speak(self, text: str, voice_profile: Optional[str] = None, play_immediately: bool = True) -> TTSResult:
        """Synthesize and optionally play speech"""
        result = await self.synthesize_speech(text, voice_profile)
        
        if play_immediately:
            self.audio_queue.put((result.audio_data, result.sample_rate))
        
        return result
    
    def set_voice_profile(self, profile_name: str):
        """Set current voice profile"""
        if profile_name in self.voice_profiles:
            self.current_profile = profile_name
            logger.info(f"🎭 Voice profile changed to: {profile_name}")
        else:
            logger.warning(f"Voice profile '{profile_name}' not found")
    
    def get_available_profiles(self) -> List[str]:
        """Get list of available voice profiles"""
        return list(self.voice_profiles.keys())
    
    def is_speaking(self) -> bool:
        """Check if currently speaking"""
        return self.is_playing
    
    def stop_speaking(self):
        """Stop current speech"""
        # Clear the queue
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break
        
        logger.info("🔇 Speech stopped")
    
    def _update_tts_stats(self, engine: TTSEngine, success: bool, processing_time: float):
        """Update TTS performance statistics"""
        self.tts_stats['total_generations'] += 1
        
        if success:
            self.tts_stats['successful_generations'] += 1
        
        # Update engine usage
        engine_name = engine.value
        if engine_name not in self.tts_stats['engine_usage']:
            self.tts_stats['engine_usage'][engine_name] = 0
        self.tts_stats['engine_usage'][engine_name] += 1
        
        # Update average processing time
        total = self.tts_stats['total_generations']
        current_avg = self.tts_stats['avg_processing_time']
        self.tts_stats['avg_processing_time'] = (current_avg * (total - 1) + processing_time) / total
    
    def get_tts_stats(self) -> Dict[str, Any]:
        """Get TTS performance statistics"""
        stats = self.tts_stats.copy()
        
        # Calculate success rate
        total = stats['total_generations']
        if total > 0:
            stats['success_rate'] = stats['successful_generations'] / total
        else:
            stats['success_rate'] = 0.0
        
        return stats
    
    def shutdown(self):
        """Shutdown TTS engine"""
        logger.info("🛑 Shutting down TTS engine...")
        
        # Stop audio playback
        self.audio_queue.put((None, None))  # Shutdown signal
        
        if self.playback_thread:
            self.playback_thread.join(timeout=2.0)
        
        logger.info("✅ TTS engine shutdown complete")
