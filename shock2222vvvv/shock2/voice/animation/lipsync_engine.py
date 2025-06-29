"""
Shock2 Advanced Lip Sync Engine
Professional lip synchronization with phoneme mapping and facial animation
"""

import asyncio
import logging
import numpy as np
import cv2
import threading
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import json
import os

# Audio processing for lip sync
import librosa
import soundfile as sf
from scipy import signal
from scipy.interpolate import interp1d

# Phoneme detection
import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import epitran

# Face detection and landmarks
import dlib
import mediapipe as mp

# Animation and rendering
import pygame
from OpenGL.GL import *
from OpenGL.GLU import *
import moderngl

logger = logging.getLogger(__name__)

class Phoneme(Enum):
    """Phoneme types for lip sync"""
    SILENCE = "SIL"
    A = "A"      # Open mouth (ah, father)
    E = "E"      # Medium open (bed, said)
    I = "I"      # Small opening (bit, sit)
    O = "O"      # Round opening (boat, note)
    U = "U"      # Rounded (book, put)
    M = "M"      # Closed (mom, bomb)
    B = "B"      # Closed (bob, cab)
    P = "P"      # Closed (pop, cap)
    F = "F"      # Teeth on lip (fun, laugh)
    V = "V"      # Teeth on lip (van, have)
    TH = "TH"    # Tongue between teeth (think, that)
    S = "S"      # Narrow opening (sun, miss)
    SH = "SH"    # Rounded narrow (shoe, wash)
    T = "T"      # Tongue to teeth (top, cat)
    D = "D"      # Tongue to teeth (dog, mad)
    K = "K"      # Back of tongue (cat, back)
    G = "G"      # Back of tongue (go, bag)
    N = "N"      # Tongue to roof (no, sun)
    L = "L"      # Tongue tip up (love, call)
    R = "R"      # Tongue curved (red, car)
    W = "W"      # Rounded lips (we, how)
    Y = "Y"      # Narrow opening (yes, you)

@dataclass
class PhonemeFrame:
    """Single phoneme frame for animation"""
    phoneme: Phoneme
    start_time: float
    end_time: float
    intensity: float
    mouth_shape: Dict[str, float]

@dataclass
class LipSyncData:
    """Complete lip sync data for animation"""
    phoneme_frames: List[PhonemeFrame]
    duration: float
    sample_rate: int
    fps: int
    mouth_shapes: List[Dict[str, float]]

class MouthShape:
    """Mouth shape definitions for different phonemes"""
    
    SHAPES = {
        Phoneme.SILENCE: {
            'mouth_open': 0.0,
            'mouth_wide': 0.0,
            'lip_pucker': 0.0,
            'jaw_open': 0.0,
            'tongue_out': 0.0,
            'teeth_show': 0.0
        },
        Phoneme.A: {
            'mouth_open': 0.8,
            'mouth_wide': 0.6,
            'lip_pucker': 0.0,
            'jaw_open': 0.7,
            'tongue_out': 0.0,
            'teeth_show': 0.3
        },
        Phoneme.E: {
            'mouth_open': 0.5,
            'mouth_wide': 0.7,
            'lip_pucker': 0.0,
            'jaw_open': 0.4,
            'tongue_out': 0.0,
            'teeth_show': 0.5
        },
        Phoneme.I: {
            'mouth_open': 0.2,
            'mouth_wide': 0.8,
            'lip_pucker': 0.0,
            'jaw_open': 0.1,
            'tongue_out': 0.0,
            'teeth_show': 0.6
        },
        Phoneme.O: {
            'mouth_open': 0.7,
            'mouth_wide': 0.0,
            'lip_pucker': 0.8,
            'jaw_open': 0.5,
            'tongue_out': 0.0,
            'teeth_show': 0.0
        },
        Phoneme.U: {
            'mouth_open': 0.3,
            'mouth_wide': 0.0,
            'lip_pucker': 0.9,
            'jaw_open': 0.2,
            'tongue_out': 0.0,
            'teeth_show': 0.0
        },
        Phoneme.M: {
            'mouth_open': 0.0,
            'mouth_wide': 0.0,
            'lip_pucker': 0.0,
            'jaw_open': 0.0,
            'tongue_out': 0.0,
            'teeth_show': 0.0
        },
        Phoneme.B: {
            'mouth_open': 0.0,
            'mouth_wide': 0.0,
            'lip_pucker': 0.0,
            'jaw_open': 0.0,
            'tongue_out': 0.0,
            'teeth_show': 0.0
        },
        Phoneme.P: {
            'mouth_open': 0.0,
            'mouth_wide': 0.0,
            'lip_pucker': 0.1,
            'jaw_open': 0.0,
            'tongue_out': 0.0,
            'teeth_show': 0.0
        },
        Phoneme.F: {
            'mouth_open': 0.1,
            'mouth_wide': 0.3,
            'lip_pucker': 0.0,
            'jaw_open': 0.0,
            'tongue_out': 0.0,
            'teeth_show': 0.8
        },
        Phoneme.V: {
            'mouth_open': 0.1,
            'mouth_wide': 0.3,
            'lip_pucker': 0.0,
            'jaw_open': 0.0,
            'tongue_out': 0.0,
            'teeth_show': 0.8
        },
        Phoneme.TH: {
            'mouth_open': 0.2,
            'mouth_wide': 0.4,
            'lip_pucker': 0.0,
            'jaw_open': 0.1,
            'tongue_out': 0.6,
            'teeth_show': 0.7
        },
        Phoneme.S: {
            'mouth_open': 0.1,
            'mouth_wide': 0.2,
            'lip_pucker': 0.0,
            'jaw_open': 0.0,
            'tongue_out': 0.0,
            'teeth_show': 0.9
        },
        Phoneme.SH: {
            'mouth_open': 0.2,
            'mouth_wide': 0.0,
            'lip_pucker': 0.6,
            'jaw_open': 0.1,
            'tongue_out': 0.0,
            'teeth_show': 0.3
        },
        Phoneme.T: {
            'mouth_open': 0.1,
            'mouth_wide': 0.3,
            'lip_pucker': 0.0,
            'jaw_open': 0.0,
            'tongue_out': 0.3,
            'teeth_show': 0.8
        },
        Phoneme.D: {
            'mouth_open': 0.1,
            'mouth_wide': 0.3,
            'lip_pucker': 0.0,
            'jaw_open': 0.0,
            'tongue_out': 0.3,
            'teeth_show': 0.8
        },
        Phoneme.K: {
            'mouth_open': 0.2,
            'mouth_wide': 0.4,
            'lip_pucker': 0.0,
            'jaw_open': 0.1,
            'tongue_out': 0.0,
            'teeth_show': 0.2
        },
        Phoneme.G: {
            'mouth_open': 0.2,
            'mouth_wide': 0.4,
            'lip_pucker': 0.0,
            'jaw_open': 0.1,
            'tongue_out': 0.0,
            'teeth_show': 0.2
        },
        Phoneme.N: {
            'mouth_open': 0.1,
            'mouth_wide': 0.3,
            'lip_pucker': 0.0,
            'jaw_open': 0.0,
            'tongue_out': 0.4,
            'teeth_show': 0.5
        },
        Phoneme.L: {
            'mouth_open': 0.3,
            'mouth_wide': 0.4,
            'lip_pucker': 0.0,
            'jaw_open': 0.2,
            'tongue_out': 0.5,
            'teeth_show': 0.4
        },
        Phoneme.R: {
            'mouth_open': 0.3,
            'mouth_wide': 0.2,
            'lip_pucker': 0.3,
            'jaw_open': 0.2,
            'tongue_out': 0.0,
            'teeth_show': 0.2
        },
        Phoneme.W: {
            'mouth_open': 0.2,
            'mouth_wide': 0.0,
            'lip_pucker': 0.9,
            'jaw_open': 0.1,
            'tongue_out': 0.0,
            'teeth_show': 0.0
        },
        Phoneme.Y: {
            'mouth_open': 0.1,
            'mouth_wide': 0.6,
            'lip_pucker': 0.0,
            'jaw_open': 0.0,
            'tongue_out': 0.0,
            'teeth_show': 0.4
        }
    }

class PhonemeDetector:
    """Advanced phoneme detection from audio"""
    
    def __init__(self):
        self.wav2vec_processor = None
        self.wav2vec_model = None
        self.epitran_converter = None
        
        # Initialize models
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize phoneme detection models"""
        try:
            # Load Wav2Vec2 for phoneme recognition
            self.wav2vec_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-lv-60-espeak-cv-ft")
            self.wav2vec_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-lv-60-espeak-cv-ft")
            logger.info("✅ Wav2Vec2 phoneme model loaded")
        except Exception as e:
            logger.warning(f"Failed to load Wav2Vec2: {e}")
        
        try:
            # Initialize Epitran for grapheme-to-phoneme conversion
            self.epitran_converter = epitran.Epitran('eng-Latn')
            logger.info("✅ Epitran G2P converter loaded")
        except Exception as e:
            logger.warning(f"Failed to load Epitran: {e}")
    
    def detect_phonemes_from_audio(self, audio_data: np.ndarray, sample_rate: int) -> List[PhonemeFrame]:
        """Detect phonemes directly from audio using Wav2Vec2"""
        if not self.wav2vec_model:
            return self._fallback_phoneme_detection(audio_data, sample_rate)
        
        try:
            # Resample to 16kHz if needed
            if sample_rate != 16000:
                audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=16000)
                sample_rate = 16000
            
            # Process audio
            inputs = self.wav2vec_processor(audio_data, sampling_rate=sample_rate, return_tensors="pt")
            
            with torch.no_grad():
                logits = self.wav2vec_model(inputs.input_values).logits
            
            # Decode phonemes
            predicted_ids = torch.argmax(logits, dim=-1)
            phoneme_sequence = self.wav2vec_processor.decode(predicted_ids[0])
            
            # Convert to phoneme frames
            return self._convert_to_phoneme_frames(phoneme_sequence, len(audio_data) / sample_rate)
            
        except Exception as e:
            logger.error(f"Phoneme detection failed: {e}")
            return self._fallback_phoneme_detection(audio_data, sample_rate)
    
    def detect_phonemes_from_text(self, text: str, duration: float) -> List[PhonemeFrame]:
        """Detect phonemes from text using G2P conversion"""
        if not self.epitran_converter:
            return self._simple_text_to_phonemes(text, duration)
        
        try:
            # Convert text to IPA phonemes
            ipa_phonemes = self.epitran_converter.transliterate(text)
            
            # Convert IPA to our phoneme system
            phoneme_frames = self._ipa_to_phoneme_frames(ipa_phonemes, duration)
            
            return phoneme_frames
            
        except Exception as e:
            logger.error(f"Text-to-phoneme conversion failed: {e}")
            return self._simple_text_to_phonemes(text, duration)
    
    def _convert_to_phoneme_frames(self, phoneme_sequence: str, duration: float) -> List[PhonemeFrame]:
        """Convert phoneme sequence to timed frames"""
        phonemes = phoneme_sequence.split()
        frame_duration = duration / len(phonemes) if phonemes else duration
        
        frames = []
        current_time = 0.0
        
        for phoneme_str in phonemes:
            # Map to our phoneme enum
            phoneme = self._map_to_phoneme_enum(phoneme_str)
            
            frame = PhonemeFrame(
                phoneme=phoneme,
                start_time=current_time,
                end_time=current_time + frame_duration,
                intensity=1.0,
                mouth_shape=MouthShape.SHAPES[phoneme].copy()
            )
            
            frames.append(frame)
            current_time += frame_duration
        
        return frames
    
    def _ipa_to_phoneme_frames(self, ipa_string: str, duration: float) -> List[PhonemeFrame]:
        """Convert IPA phonemes to our phoneme frames"""
        # IPA to our phoneme mapping
        ipa_mapping = {
            'a': Phoneme.A, 'ɑ': Phoneme.A, 'æ': Phoneme.A,
            'e': Phoneme.E, 'ɛ': Phoneme.E, 'ə': Phoneme.E,
            'i': Phoneme.I, 'ɪ': Phoneme.I,
            'o': Phoneme.O, 'ɔ': Phoneme.O, 'ɒ': Phoneme.O,
            'u': Phoneme.U, 'ʊ': Phoneme.U, 'ʌ': Phoneme.U,
            'm': Phoneme.M, 'n': Phoneme.N, 'ŋ': Phoneme.N,
            'b': Phoneme.B, 'p': Phoneme.P,
            'f': Phoneme.F, 'v': Phoneme.V,
            'θ': Phoneme.TH, 'ð': Phoneme.TH,
            's': Phoneme.S, 'z': Phoneme.S,
            'ʃ': Phoneme.SH, 'ʒ': Phoneme.SH,
            't': Phoneme.T, 'd': Phoneme.D,
            'k': Phoneme.K, 'g': Phoneme.G,
            'l': Phoneme.L, 'r': Phoneme.R,
            'w': Phoneme.W, 'j': Phoneme.Y
        }
        
        phonemes = []
        for char in ipa_string:
            if char in ipa_mapping:
                phonemes.append(ipa_mapping[char])
            elif char == ' ':
                phonemes.append(Phoneme.SILENCE)
        
        if not phonemes:
            phonemes = [Phoneme.SILENCE]
        
        frame_duration = duration / len(phonemes)
        frames = []
        current_time = 0.0
        
        for phoneme in phonemes:
            frame = PhonemeFrame(
                phoneme=phoneme,
                start_time=current_time,
                end_time=current_time + frame_duration,
                intensity=1.0,
                mouth_shape=MouthShape.SHAPES[phoneme].copy()
            )
            
            frames.append(frame)
            current_time += frame_duration
        
        return frames
    
    def _simple_text_to_phonemes(self, text: str, duration: float) -> List[PhonemeFrame]:
        """Simple text-to-phoneme conversion fallback"""
        # Basic vowel/consonant mapping
        vowel_map = {
            'a': Phoneme.A, 'e': Phoneme.E, 'i': Phoneme.I,
            'o': Phoneme.O, 'u': Phoneme.U
        }
        
        consonant_map = {
            'm': Phoneme.M, 'n': Phoneme.N, 'b': Phoneme.B, 'p': Phoneme.P,
            'f': Phoneme.F, 'v': Phoneme.V, 's': Phoneme.S, 't': Phoneme.T,
            'd': Phoneme.D, 'k': Phoneme.K, 'g': Phoneme.G, 'l': Phoneme.L,
            'r': Phoneme.R, 'w': Phoneme.W, 'y': Phoneme.Y
        }
        
        phonemes = []
        for char in text.lower():
            if char in vowel_map:
                phonemes.append(vowel_map[char])
            elif char in consonant_map:
                phonemes.append(consonant_map[char])
            elif char == ' ':
                phonemes.append(Phoneme.SILENCE)
        
        if not phonemes:
            phonemes = [Phoneme.SILENCE]
        
        frame_duration = duration / len(phonemes)
        frames = []
        current_time = 0.0
        
        for phoneme in phonemes:
            frame = PhonemeFrame(
                phoneme=phoneme,
                start_time=current_time,
                end_time=current_time + frame_duration,
                intensity=1.0,
                mouth_shape=MouthShape.SHAPES[phoneme].copy()
            )
            
            frames.append(frame)
            current_time += frame_duration
        
        return frames
    
    def _fallback_phoneme_detection(self, audio_data: np.ndarray, sample_rate: int) -> List[PhonemeFrame]:
        """Fallback phoneme detection using audio analysis"""
        # Analyze audio features
        duration = len(audio_data) / sample_rate
        
        # Simple energy-based phoneme detection
        frame_length = int(0.05 * sample_rate)  # 50ms frames
        hop_length = int(0.025 * sample_rate)   # 25ms hop
        
        # Calculate energy and spectral features
        frames = librosa.util.frame(audio_data, frame_length=frame_length, hop_length=hop_length)
        energy = np.sum(frames ** 2, axis=0)
        
        # Detect voiced/unvoiced segments
        voiced_threshold = np.mean(energy) * 0.1
        
        phoneme_frames = []
        current_time = 0.0
        frame_duration = hop_length / sample_rate
        
        for i, frame_energy in enumerate(energy):
            if frame_energy > voiced_threshold:
                # Voiced segment - use vowel
                phoneme = Phoneme.A  # Default vowel
            else:
                # Unvoiced segment
                phoneme = Phoneme.SILENCE
            
            frame = PhonemeFrame(
                phoneme=phoneme,
                start_time=current_time,
                end_time=current_time + frame_duration,
                intensity=min(frame_energy / np.max(energy), 1.0),
                mouth_shape=MouthShape.SHAPES[phoneme].copy()
            )
            
            phoneme_frames.append(frame)
            current_time += frame_duration
        
        return phoneme_frames
    
    def _map_to_phoneme_enum(self, phoneme_str: str) -> Phoneme:
        """Map phoneme string to our enum"""
        phoneme_mapping = {
            'SIL': Phoneme.SILENCE,
            'A': Phoneme.A, 'AA': Phoneme.A, 'AH': Phoneme.A,
            'E': Phoneme.E, 'EH': Phoneme.E, 'ER': Phoneme.E,
            'I': Phoneme.I, 'IH': Phoneme.I, 'IY': Phoneme.I,
            'O': Phoneme.O, 'OH': Phoneme.O, 'AO': Phoneme.O,
            'U': Phoneme.U, 'UH': Phoneme.U, 'UW': Phoneme.U,
            'M': Phoneme.M, 'N': Phoneme.N, 'NG': Phoneme.N,
            'B': Phoneme.B, 'P': Phoneme.P,
            'F': Phoneme.F, 'V': Phoneme.V,
            'TH': Phoneme.TH, 'DH': Phoneme.TH,
            'S': Phoneme.S, 'Z': Phoneme.S,
            'SH': Phoneme.SH, 'ZH': Phoneme.SH,
            'T': Phoneme.T, 'D': Phoneme.D,
            'K': Phoneme.K, 'G': Phoneme.G,
            'L': Phoneme.L, 'R': Phoneme.R,
            'W': Phoneme.W, 'Y': Phoneme.Y
        }
        
        return phoneme_mapping.get(phoneme_str.upper(), Phoneme.SILENCE)

class LipSyncEngine:
    """Advanced lip synchronization engine"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.phoneme_detector = PhonemeDetector()
        
        # Animation parameters
        self.fps = config.get('animation_fps', 30)
        self.smoothing_factor = config.get('smoothing_factor', 0.3)
        self.intensity_multiplier = config.get('intensity_multiplier', 1.0)
        
        # Current animation state
        self.current_lipsync_data = None
        self.animation_start_time = None
        self.is_animating = False
        
        # Performance tracking
        self.lipsync_stats = {
            'total_animations': 0,
            'avg_processing_time': 0.0,
            'phoneme_accuracy': 0.0
        }
    
    async def generate_lipsync_data(self, audio_data: np.ndarray, sample_rate: int, text: Optional[str] = None) -> LipSyncData:
        """Generate lip sync data from audio and optional text"""
        start_time = time.time()
        
        try:
            duration = len(audio_data) / sample_rate
            
            # Detect phonemes
            if text:
                # Use text for more accurate phoneme detection
                phoneme_frames = self.phoneme_detector.detect_phonemes_from_text(text, duration)
                
                # Refine timing using audio
                phoneme_frames = self._refine_timing_with_audio(phoneme_frames, audio_data, sample_rate)
            else:
                # Use audio-only detection
                phoneme_frames = self.phoneme_detector.detect_phonemes_from_audio(audio_data, sample_rate)
            
            # Generate mouth shapes for each frame
            mouth_shapes = self._generate_mouth_shape_sequence(phoneme_frames, duration)
            
            # Apply smoothing
            mouth_shapes = self._smooth_mouth_shapes(mouth_shapes)
            
            # Create lip sync data
            lipsync_data = LipSyncData(
                phoneme_frames=phoneme_frames,
                duration=duration,
                sample_rate=sample_rate,
                fps=self.fps,
                mouth_shapes=mouth_shapes
            )
            
            processing_time = time.time() - start_time
            self._update_lipsync_stats(True, processing_time)
            
            logger.info(f"✅ Lip sync data generated: {len(phoneme_frames)} phonemes, {len(mouth_shapes)} frames")
            return lipsync_data
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Lip sync generation failed: {e}")
            self._update_lipsync_stats(False, processing_time)
            raise
    
    def _refine_timing_with_audio(self, phoneme_frames: List[PhonemeFrame], audio_data: np.ndarray, sample_rate: int) -> List[PhonemeFrame]:
        """Refine phoneme timing using audio analysis"""
        # Analyze audio energy to adjust phoneme boundaries
        frame_length = int(0.025 * sample_rate)  # 25ms frames
        hop_length = int(0.010 * sample_rate)    # 10ms hop
        
        # Calculate energy
        energy = librosa.feature.rms(y=audio_data, frame_length=frame_length, hop_length=hop_length)[0]
        
        # Calculate time axis for energy
        energy_times = librosa.frames_to_time(np.arange(len(energy)), sr=sample_rate, hop_length=hop_length)
        
        # Adjust phoneme intensities based on energy
        for frame in phoneme_frames:
            # Find corresponding energy values
            start_idx = np.searchsorted(energy_times, frame.start_time)
            end_idx = np.searchsorted(energy_times, frame.end_time)
            
            if start_idx < len(energy) and end_idx <= len(energy) and end_idx > start_idx:
                avg_energy = np.mean(energy[start_idx:end_idx])
                frame.intensity = min(avg_energy * self.intensity_multiplier, 1.0)
                
                # Adjust mouth shape intensity
                for key in frame.mouth_shape:
                    frame.mouth_shape[key] *= frame.intensity
        
        return phoneme_frames
    
    def _generate_mouth_shape_sequence(self, phoneme_frames: List[PhonemeFrame], duration: float) -> List[Dict[str, float]]:
        """Generate sequence of mouth shapes for animation"""
        total_frames = int(duration * self.fps)
        mouth_shapes = []
        
        for frame_idx in range(total_frames):
            current_time = frame_idx / self.fps
            
            # Find current phoneme
            current_phoneme_frame = None
            for phoneme_frame in phoneme_frames:
                if phoneme_frame.start_time <= current_time <= phoneme_frame.end_time:
                    current_phoneme_frame = phoneme_frame
                    break
            
            if current_phoneme_frame:
                mouth_shape = current_phoneme_frame.mouth_shape.copy()
            else:
                # Default to silence
                mouth_shape = MouthShape.SHAPES[Phoneme.SILENCE].copy()
            
            mouth_shapes.append(mouth_shape)
        
        return mouth_shapes
    
    def _smooth_mouth_shapes(self, mouth_shapes: List[Dict[str, float]]) -> List[Dict[str, float]]:
        """Apply smoothing to mouth shape sequence"""
        if len(mouth_shapes) < 2:
            return mouth_shapes
        
        smoothed_shapes = []
        
        for i, current_shape in enumerate(mouth_shapes):
            if i == 0:
                smoothed_shapes.append(current_shape.copy())
                continue
            
            prev_shape = smoothed_shapes[i - 1]
            smoothed_shape = {}
            
            # Apply exponential smoothing
            for key in current_shape:
                smoothed_value = (self.smoothing_factor * current_shape[key] + 
                               (1 - self.smoothing_factor) * prev_shape.get(key, 0))
                smoothed_shape[key] = smoothed_value
            
            smoothed_shapes.append(smoothed_shape)
        
        return smoothed_shapes
    
    def start_animation(self, lipsync_data: LipSyncData):
        """Start lip sync animation"""
        self.current_lipsync_data = lipsync_data
        self.animation_start_time = time.time()
        self.is_animating = True
        
        logger.info(f"🎭 Lip sync animation started: {lipsync_data.duration:.2f}s")
    
    def get_current_mouth_shape(self) -> Optional[Dict[str, float]]:
        """Get current mouth shape for animation"""
        if not self.is_animating or not self.current_lipsync_data:
            return MouthShape.SHAPES[Phoneme.SILENCE].copy()
        
        # Calculate current time in animation
        elapsed_time = time.time() - self.animation_start_time
        
        if elapsed_time >= self.current_lipsync_data.duration:
            # Animation finished
            self.is_animating = False
            return MouthShape.SHAPES[Phoneme.SILENCE].copy()
        
        # Calculate frame index
        frame_idx = int(elapsed_time * self.fps)
        
        if frame_idx < len(self.current_lipsync_data.mouth_shapes):
            return self.current_lipsync_data.mouth_shapes[frame_idx].copy()
        else:
            return MouthShape.SHAPES[Phoneme.SILENCE].copy()
    
    def stop_animation(self):
        """Stop current animation"""
        self.is_animating = False
        self.current_lipsync_data = None
        self.animation_start_time = None
        
        logger.info("🛑 Lip sync animation stopped")
    
    def is_animation_active(self) -> bool:
        """Check if animation is currently active"""
        return self.is_animating
    
    def get_animation_progress(self) -> float:
        """Get current animation progress (0.0 to 1.0)"""
        if not self.is_animating or not self.current_lipsync_data:
            return 0.0
        
        elapsed_time = time.time() - self.animation_start_time
        return min(elapsed_time / self.current_lipsync_data.duration, 1.0)
    
    def _update_lipsync_stats(self, success: bool, processing_time: float):
        """Update lip sync performance statistics"""
        self.lipsync_stats['total_animations'] += 1
        
        # Update average processing time
        total = self.lipsync_stats['total_animations']
        current_avg = self.lipsync_stats['avg_processing_time']
        self.lipsync_stats['avg_processing_time'] = (current_avg * (total - 1) + processing_time) / total
    
    def get_lipsync_stats(self) -> Dict[str, Any]:
        """Get lip sync performance statistics"""
        return self.lipsync_stats.copy()

class AdvancedLipSyncRenderer:
    """Advanced renderer for lip sync visualization"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.width = config.get('render_width', 800)
        self.height = config.get('render_height', 600)
        
        # Initialize rendering
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height), pygame.OPENGL | pygame.DOUBLEBUF)
        pygame.display.set_caption("Shock2 Lip Sync Renderer")
        
        # OpenGL setup
        self._setup_opengl()
        
        # Face model
        self.face_vertices = self._create_face_model()
        self.mouth_vertices = self._create_mouth_model()
        
        # Textures and materials
        self.face_texture = None
        self.load_face_texture()
    
    def _setup_opengl(self):
        """Setup OpenGL rendering context"""
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_COLOR_MATERIAL)
        
        # Set up lighting
        glLightfv(GL_LIGHT0, GL_POSITION, [1.0, 1.0, 1.0, 0.0])
        glLightfv(GL_LIGHT0, GL_AMBIENT, [0.2, 0.2, 0.2, 1.0])
        glLightfv(GL_LIGHT0, GL_DIFFUSE, [0.8, 0.8, 0.8, 1.0])
        
        # Set up camera
        glMatrixMode(GL_PROJECTION)
        gluPerspective(45, self.width / self.height, 0.1, 50.0)
        glMatrixMode(GL_MODELVIEW)
        
        # Background color
        glClearColor(0.0, 0.0, 0.0, 1.0)
    
    def _create_face_model(self) -> List[Tuple[float, float, float]]:
        """Create basic 3D face model vertices"""
        # Simplified face model - in production, use a proper 3D model
        vertices = [
            # Face outline
            (-1.0, 1.0, 0.0), (1.0, 1.0, 0.0), (1.0, -1.0, 0.0), (-1.0, -1.0, 0.0),
            # Eye positions
            (-0.3, 0.3, 0.1), (0.3, 0.3, 0.1),
            # Nose
            (0.0, 0.0, 0.2),
            # Mouth corners
            (-0.3, -0.3, 0.0), (0.3, -0.3, 0.0)
        ]
        return vertices
    
    def _create_mouth_model(self) -> List[Tuple[float, float, float]]:
        """Create mouth model vertices"""
        # Mouth vertices that can be deformed
        vertices = [
            # Upper lip
            (-0.2, -0.2, 0.0), (-0.1, -0.15, 0.0), (0.0, -0.15, 0.0), (0.1, -0.15, 0.0), (0.2, -0.2, 0.0),
            # Lower lip
            (-0.2, -0.4, 0.0), (-0.1, -0.45, 0.0), (0.0, -0.45, 0.0), (0.1, -0.45, 0.0), (0.2, -0.4, 0.0),
            # Inner mouth
            (-0.1, -0.25, -0.1), (0.0, -0.25, -0.1), (0.1, -0.25, -0.1),
            (-0.1, -0.35, -0.1), (0.0, -0.35, -0.1), (0.1, -0.35, -0.1)
        ]
        return vertices
    
    def load_face_texture(self):
        """Load face texture"""
        # In production, load actual texture
        # For now, create a simple procedural texture
        texture_data = np.random.randint(100, 200, (256, 256, 3), dtype=np.uint8)
        
        self.face_texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.face_texture)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 256, 256, 0, GL_RGB, GL_UNSIGNED_BYTE, texture_data)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    
    def render_frame(self, mouth_shape: Dict[str, float]):
        """Render a single frame with given mouth shape"""
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        
        # Position camera
        glTranslatef(0.0, 0.0, -3.0)
        
        # Render face
        self._render_face()
        
        # Render mouth with deformation
        self._render_mouth(mouth_shape)
        
        # Render eyes
        self._render_eyes()
        
        pygame.display.flip()
    
    def _render_face(self):
        """Render the face mesh"""
        glColor3f(0.8, 0.7, 0.6)  # Skin color
        
        glBegin(GL_QUADS)
        # Face quad
        glVertex3f(-1.0, 1.0, 0.0)
        glVertex3f(1.0, 1.0, 0.0)
        glVertex3f(1.0, -1.0, 0.0)
        glVertex3f(-1.0, -1.0, 0.0)
        glEnd()
    
    def _render_mouth(self, mouth_shape: Dict[str, float]):
        """Render mouth with shape deformation"""
        mouth_open = mouth_shape.get('mouth_open', 0.0)
        mouth_wide = mouth_shape.get('mouth_wide', 0.0)
        lip_pucker = mouth_shape.get('lip_pucker', 0.0)
        jaw_open = mouth_shape.get('jaw_open', 0.0)
        
        # Calculate mouth vertices with deformation
        deformed_vertices = []
        
        for i, (x, y, z) in enumerate(self.mouth_vertices):
            # Apply deformations
            new_x = x * (1.0 + mouth_wide * 0.5)
            new_y = y - jaw_open * 0.2
            new_z = z
            
            # Lip pucker effect
            if abs(x) < 0.15:  # Center of mouth
                new_x *= (1.0 - lip_pucker * 0.3)
                new_z += lip_pucker * 0.1
            
            # Mouth opening
            if i < 5:  # Upper lip
                new_y += mouth_open * 0.1
            elif i < 10:  # Lower lip
                new_y -= mouth_open * 0.1
            
            deformed_vertices.append((new_x, new_y, new_z))
        
        # Render mouth
        glColor3f(0.6, 0.3, 0.3)  # Lip color
        
        # Upper lip
        glBegin(GL_TRIANGLE_STRIP)
        for i in range(5):
            x, y, z = deformed_vertices[i]
            glVertex3f(x, y, z)
        glEnd()
        
        # Lower lip
        glBegin(GL_TRIANGLE_STRIP)
        for i in range(5, 10):
            x, y, z = deformed_vertices[i]
            glVertex3f(x, y, z)
        glEnd()
        
        # Inner mouth (if open)
        if mouth_open > 0.1:
            glColor3f(0.2, 0.1, 0.1)  # Dark mouth interior
            glBegin(GL_TRIANGLES)
            for i in range(10, len(deformed_vertices)):
                x, y, z = deformed_vertices[i]
                glVertex3f(x, y, z)
            glEnd()
        
        # Teeth (if visible)
        teeth_show = mouth_shape.get('teeth_show', 0.0)
        if teeth_show > 0.3:
            glColor3f(0.9, 0.9, 0.8)  # Tooth color
            glBegin(GL_QUADS)
            # Simple teeth representation
            for i in range(-2, 3):
                x = i * 0.05
                glVertex3f(x - 0.02, -0.18, 0.05)
                glVertex3f(x + 0.02, -0.18, 0.05)
                glVertex3f(x + 0.02, -0.25, 0.05)
                glVertex3f(x - 0.02, -0.25, 0.05)
            glEnd()
    
    def _render_eyes(self):
        """Render eyes"""
        glColor3f(1.0, 1.0, 1.0)  # Eye white
        
        # Left eye
        glPushMatrix()
        glTranslatef(-0.3, 0.3, 0.1)
        self._draw_sphere(0.1, 10, 10)
        glPopMatrix()
        
        # Right eye
        glPushMatrix()
        glTranslatef(0.3, 0.3, 0.1)
        self._draw_sphere(0.1, 10, 10)
        glPopMatrix()
        
        # Pupils
        glColor3f(0.0, 0.0, 0.0)
        
        # Left pupil
        glPushMatrix()
        glTranslatef(-0.3, 0.3, 0.15)
        self._draw_sphere(0.03, 8, 8)
        glPopMatrix()
        
        # Right pupil
        glPushMatrix()
        glTranslatef(0.3, 0.3, 0.15)
        self._draw_sphere(0.03, 8, 8)
        glPopMatrix()
    
    def _draw_sphere(self, radius: float, slices: int, stacks: int):
        """Draw a sphere using OpenGL"""
        quadric = gluNewQuadric()
        gluSphere(quadric, radius, slices, stacks)
        gluDeleteQuadric(quadric)
    
    def cleanup(self):
        """Cleanup rendering resources"""
        if self.face_texture:
            glDeleteTextures([self.face_texture])
        
        pygame.quit()
