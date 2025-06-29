"""
Shock2 Voice Authentication & Security Engine
Advanced voice biometrics, access control, and security management
"""

import asyncio
import logging
import numpy as np
import torch
import torch.nn as nn
import hashlib
import hmac
import secrets
import time
import json
import os
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import pickle
from datetime import datetime, timedelta
import jwt
from cryptography.fernet import Fernet
import librosa
from scipy.spatial.distance import cosine

# Voice biometrics
try:
    import resemblyzer
    from resemblyzer import VoiceEncoder, preprocess_wav
    RESEMBLYZER_AVAILABLE = True
except ImportError:
    RESEMBLYZER_AVAILABLE = False

# Advanced audio processing
from scipy import signal
import webrtcvad

logger = logging.getLogger(__name__)

class SecurityLevel(Enum):
    """Security access levels"""
    PUBLIC = 1
    BASIC = 2
    ELEVATED = 3
    ADMIN = 4
    SUPER_ADMIN = 5

class AuthenticationMethod(Enum):
    """Authentication methods"""
    VOICE_BIOMETRIC = "voice_biometric"
    PASSPHRASE = "passphrase"
    MULTI_FACTOR = "multi_factor"
    BEHAVIORAL = "behavioral"
    CONTINUOUS = "continuous"

class ThreatLevel(Enum):
    """Threat assessment levels"""
    NONE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class VoicePrint:
    """Voice biometric profile"""
    user_id: str
    voice_embedding: np.ndarray
    voice_characteristics: Dict[str, float]
    enrollment_samples: int
    quality_score: float
    created_at: datetime
    last_updated: datetime
    verification_count: int = 0
    false_rejection_count: int = 0

@dataclass
class AuthenticationAttempt:
    """Authentication attempt record"""
    attempt_id: str
    user_id: Optional[str]
    method: AuthenticationMethod
    timestamp: datetime
    success: bool
    confidence_score: float
    threat_level: ThreatLevel
    audio_features: Dict[str, Any]
    ip_address: Optional[str] = None
    device_fingerprint: Optional[str] = None
    behavioral_score: float = 0.0

@dataclass
class SecurityPolicy:
    """Security policy configuration"""
    min_confidence_threshold: float = 0.85
    max_failed_attempts: int = 3
    lockout_duration: int = 300  # seconds
    session_timeout: int = 3600  # seconds
    require_continuous_auth: bool = True
    threat_detection_enabled: bool = True
    behavioral_analysis_enabled: bool = True
    voice_liveness_detection: bool = True

@dataclass
class UserSession:
    """Active user session"""
    session_id: str
    user_id: str
    security_level: SecurityLevel
    created_at: datetime
    last_activity: datetime
    authentication_method: AuthenticationMethod
    continuous_auth_score: float = 1.0
    threat_score: float = 0.0
    device_fingerprint: Optional[str] = None

class VoiceBiometricModel(nn.Module):
    """Advanced voice biometric neural network"""
    
    def __init__(self, input_dim: int = 80, embedding_dim: int = 512):
        super().__init__()
        
        # Feature extraction layers
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(input_dim, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(512, 8, batch_first=True)
        
        # Embedding layers
        self.embedding_layers = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, embedding_dim),
            nn.Tanh()
        )
        
        # Verification head
        self.verification_head = nn.Sequential(
            nn.Linear(embedding_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x1, x2=None):
        # Extract features
        features1 = self.extract_features(x1)
        
        if x2 is not None:
            # Verification mode
            features2 = self.extract_features(x2)
            combined = torch.cat([features1, features2], dim=1)
            verification_score = self.verification_head(combined)
            return verification_score
        else:
            # Embedding mode
            return features1
    
    def extract_features(self, x):
        # x shape: (batch_size, time_steps, features)
        x = x.transpose(1, 2)  # (batch_size, features, time_steps)
        
        # Feature extraction
        features = self.feature_extractor(x)
        features = features.squeeze(-1)  # Remove time dimension
        
        # Generate embedding
        embedding = self.embedding_layers(features)
        
        # L2 normalize
        embedding = nn.functional.normalize(embedding, p=2, dim=1)
        
        return embedding

class ThreatDetectionEngine:
    """Advanced threat detection and analysis"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.threat_patterns = self._load_threat_patterns()
        self.behavioral_baselines = {}
        
        # Anomaly detection thresholds
        self.anomaly_thresholds = {
            'voice_deviation': 0.3,
            'timing_anomaly': 0.4,
            'frequency_anomaly': 0.35,
            'behavioral_deviation': 0.5
        }
    
    def _load_threat_patterns(self) -> Dict[str, Any]:
        """Load known threat patterns"""
        return {
            'replay_attack': {
                'spectral_flatness_threshold': 0.1,
                'noise_floor_threshold': -60,
                'compression_artifacts': True
            },
            'voice_synthesis': {
                'formant_irregularities': True,
                'pitch_contour_smoothness': 0.8,
                'spectral_discontinuities': True
            },
            'voice_conversion': {
                'embedding_distance_threshold': 0.4,
                'prosody_mismatch': True,
                'temporal_inconsistencies': True
            }
        }
    
    async def analyze_threat_level(self, 
                                 audio_data: np.ndarray, 
                                 user_id: str,
                                 context: Dict[str, Any]) -> Tuple[ThreatLevel, Dict[str, Any]]:
        """Analyze threat level of authentication attempt"""
        
        threat_indicators = {}
        threat_score = 0.0
        
        # Audio-based threat detection
        audio_threats = await self._detect_audio_threats(audio_data)
        threat_indicators.update(audio_threats)
        threat_score += sum(audio_threats.values()) * 0.4
        
        # Behavioral analysis
        if user_id in self.behavioral_baselines:
            behavioral_threats = await self._detect_behavioral_anomalies(audio_data, user_id, context)
            threat_indicators.update(behavioral_threats)
            threat_score += sum(behavioral_threats.values()) * 0.3
        
        # Contextual analysis
        contextual_threats = await self._detect_contextual_anomalies(context)
        threat_indicators.update(contextual_threats)
        threat_score += sum(contextual_threats.values()) * 0.3
        
        # Determine threat level
        if threat_score >= 0.8:
            threat_level = ThreatLevel.CRITICAL
        elif threat_score >= 0.6:
            threat_level = ThreatLevel.HIGH
        elif threat_score >= 0.4:
            threat_level = ThreatLevel.MEDIUM
        elif threat_score >= 0.2:
            threat_level = ThreatLevel.LOW
        else:
            threat_level = ThreatLevel.NONE
        
        return threat_level, threat_indicators
    
    async def _detect_audio_threats(self, audio_data: np.ndarray) -> Dict[str, float]:
        """Detect audio-based threats"""
        threats = {}
        
        try:
            # Replay attack detection
            replay_score = await self._detect_replay_attack(audio_data)
            threats['replay_attack'] = replay_score
            
            # Voice synthesis detection
            synthesis_score = await self._detect_voice_synthesis(audio_data)
            threats['voice_synthesis'] = synthesis_score
            
            # Voice conversion detection
            conversion_score = await self._detect_voice_conversion(audio_data)
            threats['voice_conversion'] = conversion_score
            
            # Liveness detection
            liveness_score = await self._detect_liveness(audio_data)
            threats['liveness_failure'] = 1.0 - liveness_score
            
        except Exception as e:
            logger.error(f"Audio threat detection failed: {e}")
            threats['detection_error'] = 0.5
        
        return threats
    
    async def _detect_replay_attack(self, audio_data: np.ndarray) -> float:
        """Detect replay attacks"""
        try:
            # Analyze spectral characteristics
            stft = librosa.stft(audio_data)
            magnitude = np.abs(stft)
            
            # Check for compression artifacts
            spectral_flatness = librosa.feature.spectral_flatness(S=magnitude)[0]
            avg_flatness = np.mean(spectral_flatness)
            
            # Check noise floor
            noise_floor = np.percentile(magnitude, 5)
            noise_floor_db = 20 * np.log10(noise_floor + 1e-10)
            
            # Calculate replay probability
            replay_score = 0.0
            
            if avg_flatness < self.threat_patterns['replay_attack']['spectral_flatness_threshold']:
                replay_score += 0.4
            
            if noise_floor_db > self.threat_patterns['replay_attack']['noise_floor_threshold']:
                replay_score += 0.6
            
            return min(1.0, replay_score)
            
        except Exception as e:
            logger.error(f"Replay detection failed: {e}")
            return 0.0
    
    async def _detect_voice_synthesis(self, audio_data: np.ndarray) -> float:
        """Detect synthetic voice"""
        try:
            # Analyze formant characteristics
            # This is a simplified implementation
            
            # Extract fundamental frequency
            f0, voiced_flag, voiced_probs = librosa.pyin(audio_data, fmin=80, fmax=400)
            f0_clean = f0[voiced_flag]
            
            if len(f0_clean) == 0:
                return 0.8  # Suspicious if no pitch detected
            
            # Check pitch contour smoothness
            pitch_diff = np.diff(f0_clean)
            pitch_smoothness = 1.0 - (np.std(pitch_diff) / np.mean(f0_clean))
            
            synthesis_score = 0.0
            
            if pitch_smoothness > self.threat_patterns['voice_synthesis']['pitch_contour_smoothness']:
                synthesis_score += 0.5
            
            # Check for spectral discontinuities
            stft = librosa.stft(audio_data)
            magnitude = np.abs(stft)
            
            # Look for sudden spectral changes
            spectral_diff = np.diff(magnitude, axis=1)
            discontinuities = np.sum(np.abs(spectral_diff) > np.std(spectral_diff) * 3)
            
            if discontinuities > len(spectral_diff[0]) * 0.1:  # More than 10% discontinuities
                synthesis_score += 0.5
            
            return min(1.0, synthesis_score)
            
        except Exception as e:
            logger.error(f"Synthesis detection failed: {e}")
            return 0.0
    
    async def _detect_voice_conversion(self, audio_data: np.ndarray) -> float:
        """Detect voice conversion attacks"""
        # This would require comparison with known voice patterns
        # For now, return a low score
        return 0.1
    
    async def _detect_liveness(self, audio_data: np.ndarray) -> float:
        """Detect voice liveness"""
        try:
            # Check for natural speech characteristics
            
            # Voice activity detection
            vad = webrtcvad.Vad(2)
            frame_duration = 30  # ms
            frame_size = int(16000 * frame_duration / 1000)
            
            # Convert to int16
            audio_int16 = (audio_data * 32767).astype(np.int16)
            
            voice_frames = 0
            total_frames = 0
            
            for i in range(0, len(audio_int16) - frame_size, frame_size):
                frame = audio_int16[i:i + frame_size].tobytes()
                if len(frame) == frame_size * 2:
                    if vad.is_speech(frame, 16000):
                        voice_frames += 1
                    total_frames += 1
            
            if total_frames == 0:
                return 0.0
            
            voice_ratio = voice_frames / total_frames
            
            # Natural speech should have pauses
            if 0.3 <= voice_ratio <= 0.8:
                liveness_score = 1.0
            else:
                liveness_score = max(0.0, 1.0 - abs(voice_ratio - 0.55) * 2)
            
            return liveness_score
            
        except Exception as e:
            logger.error(f"Liveness detection failed: {e}")
            return 0.5
    
    async def _detect_behavioral_anomalies(self, 
                                         audio_data: np.ndarray, 
                                         user_id: str,
                                         context: Dict[str, Any]) -> Dict[str, float]:
        """Detect behavioral anomalies"""
        anomalies = {}
        
        if user_id not in self.behavioral_baselines:
            return anomalies
        
        baseline = self.behavioral_baselines[user_id]
        
        try:
            # Speaking rate analysis
            current_rate = self._calculate_speaking_rate(audio_data)
            baseline_rate = baseline.get('speaking_rate', current_rate)
            
            rate_deviation = abs(current_rate - baseline_rate) / baseline_rate
            if rate_deviation > self.anomaly_thresholds['timing_anomaly']:
                anomalies['speaking_rate_anomaly'] = min(1.0, rate_deviation)
            
            # Voice characteristics deviation
            current_characteristics = self._extract_voice_characteristics(audio_data)
            
            for char_name, current_value in current_characteristics.items():
                baseline_value = baseline.get(char_name, current_value)
                if baseline_value > 0:
                    deviation = abs(current_value - baseline_value) / baseline_value
                    if deviation > self.anomaly_thresholds['voice_deviation']:
                        anomalies[f'{char_name}_anomaly'] = min(1.0, deviation)
            
        except Exception as e:
            logger.error(f"Behavioral analysis failed: {e}")
            anomalies['behavioral_analysis_error'] = 0.3
        
        return anomalies
    
    async def _detect_contextual_anomalies(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Detect contextual anomalies"""
        anomalies = {}
        
        # Time-based analysis
        current_hour = datetime.now().hour
        if 'typical_hours' in context:
            typical_hours = context['typical_hours']
            if current_hour not in typical_hours:
                anomalies['unusual_time'] = 0.3
        
        # Location-based analysis (if available)
        if 'location' in context:
            current_location = context['location']
            if 'typical_locations' in context:
                typical_locations = context['typical_locations']
                if current_location not in typical_locations:
                    anomalies['unusual_location'] = 0.4
        
        # Device fingerprint analysis
        if 'device_fingerprint' in context:
            current_device = context['device_fingerprint']
            if 'known_devices' in context:
                known_devices = context['known_devices']
                if current_device not in known_devices:
                    anomalies['unknown_device'] = 0.5
        
        # IP address analysis
        if 'ip_address' in context:
            current_ip = context['ip_address']
            if 'typical_ip_ranges' in context:
                typical_ranges = context['typical_ip_ranges']
                ip_suspicious = True
                for ip_range in typical_ranges:
                    if self._ip_in_range(current_ip, ip_range):
                        ip_suspicious = False
                        break
                
                if ip_suspicious:
                    anomalies['suspicious_ip'] = 0.6
        
        return anomalies
    
    def _calculate_speaking_rate(self, audio_data: np.ndarray) -> float:
        """Calculate speaking rate (words per minute estimate)"""
        try:
            # Estimate speaking rate based on syllable detection
            # This is a simplified approach
            
            # Find voice activity regions
            vad = webrtcvad.Vad(2)
            frame_duration = 30  # ms
            frame_size = int(16000 * frame_duration / 1000)
            
            audio_int16 = (audio_data * 32767).astype(np.int16)
            
            voice_segments = []
            current_segment = []
            
            for i in range(0, len(audio_int16) - frame_size, frame_size):
                frame = audio_int16[i:i + frame_size].tobytes()
                if len(frame) == frame_size * 2:
                    if vad.is_speech(frame, 16000):
                        current_segment.append(i)
                    else:
                        if current_segment:
                            voice_segments.append(current_segment)
                            current_segment = []
            
            if current_segment:
                voice_segments.append(current_segment)
            
            # Estimate syllables from voice segments
            total_syllables = 0
            for segment in voice_segments:
                segment_duration = len(segment) * frame_duration / 1000
                # Rough estimate: 2-4 syllables per second for normal speech
                estimated_syllables = segment_duration * 3
                total_syllables += estimated_syllables
            
            # Convert to words per minute (assuming ~1.5 syllables per word)
            total_duration = len(audio_data) / 16000  # seconds
            words_per_minute = (total_syllables / 1.5) * (60 / total_duration)
            
            return words_per_minute
            
        except Exception as e:
            logger.error(f"Speaking rate calculation failed: {e}")
            return 150.0  # Default rate
    
    def _extract_voice_characteristics(self, audio_data: np.ndarray) -> Dict[str, float]:
        """Extract voice characteristics for behavioral analysis"""
        characteristics = {}
        
        try:
            # Fundamental frequency statistics
            f0, voiced_flag, voiced_probs = librosa.pyin(audio_data, fmin=80, fmax=400)
            f0_clean = f0[voiced_flag]
            
            if len(f0_clean) > 0:
                characteristics['mean_f0'] = np.mean(f0_clean)
                characteristics['std_f0'] = np.std(f0_clean)
                characteristics['f0_range'] = np.max(f0_clean) - np.min(f0_clean)
            
            # Spectral characteristics
            stft = librosa.stft(audio_data)
            magnitude = np.abs(stft)
            
            spectral_centroids = librosa.feature.spectral_centroid(S=magnitude)[0]
            characteristics['mean_spectral_centroid'] = np.mean(spectral_centroids)
            
            spectral_bandwidth = librosa.feature.spectral_bandwidth(S=magnitude)[0]
            characteristics['mean_spectral_bandwidth'] = np.mean(spectral_bandwidth)
            
            # Energy characteristics
            rms_energy = librosa.feature.rms(y=audio_data)[0]
            characteristics['mean_energy'] = np.mean(rms_energy)
            characteristics['energy_variance'] = np.var(rms_energy)
            
            # Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(audio_data)[0]
            characteristics['mean_zcr'] = np.mean(zcr)
            
        except Exception as e:
            logger.error(f"Voice characteristics extraction failed: {e}")
        
        return characteristics
    
    def _ip_in_range(self, ip: str, ip_range: str) -> bool:
        """Check if IP is in given range"""
        try:
            import ipaddress
            return ipaddress.ip_address(ip) in ipaddress.ip_network(ip_range)
        except:
            return False
    
    def update_behavioral_baseline(self, user_id: str, audio_data: np.ndarray, context: Dict[str, Any]):
        """Update behavioral baseline for user"""
        if user_id not in self.behavioral_baselines:
            self.behavioral_baselines[user_id] = {}
        
        baseline = self.behavioral_baselines[user_id]
        
        # Update speaking rate
        current_rate = self._calculate_speaking_rate(audio_data)
        if 'speaking_rate' in baseline:
            baseline['speaking_rate'] = (baseline['speaking_rate'] * 0.9 + current_rate * 0.1)
        else:
            baseline['speaking_rate'] = current_rate
        
        # Update voice characteristics
        current_characteristics = self._extract_voice_characteristics(audio_data)
        for char_name, current_value in current_characteristics.items():
            if char_name in baseline:
                baseline[char_name] = (baseline[char_name] * 0.9 + current_value * 0.1)
            else:
                baseline[char_name] = current_value
        
        # Update contextual information
        current_hour = datetime.now().hour
        if 'typical_hours' not in baseline:
            baseline['typical_hours'] = set()
        baseline['typical_hours'].add(current_hour)
        
        if 'location' in context:
            if 'typical_locations' not in baseline:
                baseline['typical_locations'] = set()
            baseline['typical_locations'].add(context['location'])

class Shock2VoiceAuthEngine:
    """Complete voice authentication and security system"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Security configuration
        self.security_policy = SecurityPolicy(**config.get('security_policy', {}))
        
        # Models and engines
        self.biometric_model = None
        self.voice_encoder = None
        self.threat_detector = ThreatDetectionEngine(config.get('threat_detection', {}))
        
        # Data storage
        self.voice_prints: Dict[str, VoicePrint] = {}
        self.active_sessions: Dict[str, UserSession] = {}
        self.auth_attempts: List[AuthenticationAttempt] = []
        self.failed_attempts: Dict[str, List[datetime]] = {}
        self.locked_users: Dict[str, datetime] = {}
        
        # Encryption
        self.encryption_key = self._load_or_generate_encryption_key()
        self.cipher_suite = Fernet(self.encryption_key)
        
        # JWT configuration
        self.jwt_secret = config.get('jwt_secret', secrets.token_urlsafe(32))
        
        # Performance tracking
        self.auth_stats = {
            'total_attempts': 0,
            'successful_auths': 0,
            'failed_auths': 0,
            'threat_detections': 0,
            'avg_auth_time': 0.0,
            'false_acceptance_rate': 0.0,
            'false_rejection_rate': 0.0
        }
        
        # Initialize components
        self._initialize_models()
        self._load_voice_prints()
        self._start_session_cleanup()
    
    def _load_or_generate_encryption_key(self) -> bytes:
        """Load or generate encryption key"""
        key_path = self.config.get('encryption_key_path', 'data/security/encryption.key')
        
        if os.path.exists(key_path):
            with open(key_path, 'rb') as f:
                return f.read()
        else:
            # Generate new key
            key = Fernet.generate_key()
            os.makedirs(os.path.dirname(key_path), exist_ok=True)
            with open(key_path, 'wb') as f:
                f.write(key)
            return key
    
    def _initialize_models(self):
        """Initialize biometric models"""
        logger.info("🔐 Initializing voice authentication models...")
        
        # Initialize custom biometric model
        self.biometric_model = VoiceBiometricModel().to(self.device)
        
        # Load pre-trained weights if available
        model_path = self.config.get('biometric_model_path')
        if model_path and os.path.exists(model_path):
            try:
                checkpoint = torch.load(model_path, map_location=self.device)
                self.biometric_model.load_state_dict(checkpoint['model_state_dict'])
                logger.info("✅ Pre-trained biometric model loaded")
            except Exception as e:
                logger.warning(f"Failed to load pre-trained model: {e}")
        
        # Initialize Resemblyzer if available
        if RESEMBLYZER_AVAILABLE:
            try:
                self.voice_encoder = resemblyzer.VoiceEncoder()
                logger.info("✅ Resemblyzer voice encoder loaded")
            except Exception as e:
                logger.error(f"Failed to load Resemblyzer: {e}")
        
        logger.info("✅ Voice authentication models initialized")
    
    def _load_voice_prints(self):
        """Load existing voice prints"""
        voice_prints_dir = self.config.get('voice_prints_dir', 'data/security/voice_prints')
        
        if not os.path.exists(voice_prints_dir):
            os.makedirs(voice_prints_dir, exist_ok=True)
            return
        
        for print_file in os.listdir(voice_prints_dir):
            if print_file.endswith('.enc'):
                try:
                    print_path = os.path.join(voice_prints_dir, print_file)
                    
                    # Decrypt and load voice print
                    with open(print_path, 'rb') as f:
                        encrypted_data = f.read()
                    
                    decrypted_data = self.cipher_suite.decrypt(encrypted_data)
                    voice_print_data = pickle.loads(decrypted_data)
                    
                    voice_print = VoicePrint(**voice_print_data)
                    self.voice_prints[voice_print.user_id] = voice_print
                    
                    logger.info(f"🔐 Loaded voice print for user: {voice_print.user_id}")
                    
                except Exception as e:
                    logger.error(f"Failed to load voice print {print_file}: {e}")
        
        logger.info(f"✅ Loaded {len(self.voice_prints)} voice prints")
    
    def _start_session_cleanup(self):
        """Start background session cleanup"""
        async def cleanup_sessions():
            while True:
                try:
                    current_time = datetime.now()
                    expired_sessions = []
                    
                    for session_id, session in self.active_sessions.items():
                        session_age = (current_time - session.last_activity).total_seconds()
                        if session_age > self.security_policy.session_timeout:
                            expired_sessions.append(session_id)
                    
                    for session_id in expired_sessions:
                        del self.active_sessions[session_id]
                        logger.info(f"🕐 Session expired: {session_id}")
                    
                    # Cleanup failed attempts
                    cutoff_time = current_time - timedelta(seconds=self.security_policy.lockout_duration)
                    for user_id in list(self.failed_attempts.keys()):
                        self.failed_attempts[user_id] = [
                            attempt_time for attempt_time in self.failed_attempts[user_id]
                            if attempt_time > cutoff_time
                        ]
                        if not self.failed_attempts[user_id]:
                            del self.failed_attempts[user_id]
                    
                    # Cleanup locked users
                    for user_id in list(self.locked_users.keys()):
                        if (current_time - self.locked_users[user_id]).total_seconds() > self.security_policy.lockout_duration:
                            del self.locked_users[user_id]
                            logger.info(f"🔓 User unlocked: {user_id}")
                    
                except Exception as e:
                    logger.error(f"Session cleanup error: {e}")
                
                await asyncio.sleep(60)  # Cleanup every minute
        
        asyncio.create_task(cleanup_sessions())
    
    async def enroll_user(self, 
                         user_id: str, 
                         audio_samples: List[np.ndarray],
                         security_level: SecurityLevel = SecurityLevel.BASIC) -> bool:
        """Enroll user for voice authentication"""
        logger.info(f"👤 Enrolling user for voice authentication: {user_id}")
        
        try:
            if len(audio_samples) < 3:
                raise ValueError("Need at least 3 audio samples for enrollment")
            
            # Extract voice embeddings
            embeddings = []
            voice_characteristics_list = []
            
            for audio_data in audio_samples:
                # Generate embedding
                embedding = await self._generate_voice_embedding(audio_data)
                embeddings.append(embedding)
                
                # Extract characteristics
                characteristics = self.threat_detector._extract_voice_characteristics(audio_data)
                voice_characteristics_list.append(characteristics)
            
            # Average embeddings and characteristics
            avg_embedding = np.mean(embeddings, axis=0)
            avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)  # Normalize
            
            avg_characteristics = {}
            for key in voice_characteristics_list[0].keys():
                avg_characteristics[key] = np.mean([chars[key] for chars in voice_characteristics_list])
            
            # Calculate quality score
            quality_score = await self._calculate_enrollment_quality(embeddings, audio_samples)
            
            # Create voice print
            voice_print = VoicePrint(
                user_id=user_id,
                voice_embedding=avg_embedding,
                voice_characteristics=avg_characteristics,
                enrollment_samples=len(audio_samples),
                quality_score=quality_score,
                created_at=datetime.now(),
                last_updated=datetime.now()
            )
            
            # Save voice print
            await self._save_voice_print(voice_print)
            self.voice_prints[user_id] = voice_print
            
            # Initialize behavioral baseline
            for audio_data in audio_samples:
                self.threat_detector.update_behavioral_baseline(user_id, audio_data, {})
            
            logger.info(f"✅ User enrolled successfully: {user_id} (quality: {quality_score:.2f})")
            return True
            
        except Exception as e:
            logger.error(f"User enrollment failed: {e}")
            return False
    
    async def authenticate_user(self, 
                              audio_data: np.ndarray,
                              claimed_user_id: Optional[str] = None,
                              context: Optional[Dict[str, Any]] = None) -> Tuple[bool, Optional[str], float, Dict[str, Any]]:
        """Authenticate user via voice biometrics"""
        start_time = time.time()
        context = context or {}
        
        try:
            # Check if user is locked
            if claimed_user_id and claimed_user_id in self.locked_users:
                logger.warning(f"🔒 Authentication attempt for locked user: {claimed_user_id}")
                return False, None, 0.0, {'error': 'user_locked'}
            
            # Threat analysis
            threat_level, threat_indicators = await self.threat_detector.analyze_threat_level(
                audio_data, claimed_user_id or 'unknown', context
            )
            
            if threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]:
                logger.warning(f"🚨 High threat level detected: {threat_level}")
                await self._log_auth_attempt(
                    claimed_user_id, AuthenticationMethod.VOICE_BIOMETRIC, 
                    False, 0.0, threat_level, context
                )
                return False, None, 0.0, {'threat_level': threat_level.name, 'threats': threat_indicators}
            
            # Generate voice embedding
            test_embedding = await self._generate_voice_embedding(audio_data)
            
            # Authentication logic
            if claimed_user_id:
                # Verification mode
                success, confidence = await self._verify_user(claimed_user_id, test_embedding, audio_data)
                authenticated_user = claimed_user_id if success else None
            else:
                # Identification mode
                authenticated_user, confidence = await self._identify_user(test_embedding, audio_data)
                success = authenticated_user is not None
            
            # Apply security policy
            if success and confidence < self.security_policy.min_confidence_threshold:
                success = False
                authenticated_user = None
            
            # Handle failed authentication
            if not success:
                await self._handle_failed_authentication(claimed_user_id or 'unknown')
            else:
                # Update behavioral baseline
                if authenticated_user:
                    self.threat_detector.update_behavioral_baseline(authenticated_user, audio_data, context)
            
            # Log attempt
            auth_time = time.time() - start_time
            await self._log_auth_attempt(
                authenticated_user or claimed_user_id, 
                AuthenticationMethod.VOICE_BIOMETRIC,
                success, confidence, threat_level, context
            )
            
            # Update statistics
            self._update_auth_stats(success, auth_time, confidence)
            
            result_info = {
                'confidence': confidence,
                'threat_level': threat_level.name,
                'threats': threat_indicators,
                'auth_time': auth_time
            }
            
            logger.info(f"🔐 Authentication {'successful' if success else 'failed'}: {authenticated_user or 'unknown'} (confidence: {confidence:.2f})")
            
            return success, authenticated_user, confidence, result_info
            
        except Exception as e:
            auth_time = time.time() - start_time
            logger.error(f"Authentication error: {e}")
            await self._log_auth_attempt(
                claimed_user_id, AuthenticationMethod.VOICE_BIOMETRIC,
                False, 0.0, ThreatLevel.MEDIUM, context
            )
            return False, None, 0.0, {'error': str(e)}
    
    async def _generate_voice_embedding(self, audio_data: np.ndarray) -> np.ndarray:
        """Generate voice embedding from audio"""
        if self.voice_encoder:
            # Use Resemblyzer
            try:
                # Resample to 16kHz if needed
                if len(audio_data) > 16000:  # Assume sample rate > 16kHz
                    audio_resampled = librosa.resample(audio_data, orig_sr=22050, target_sr=16000)
                else:
                    audio_resampled = audio_data
                
                embedding = self.voice_encoder.embed_utterance(audio_resampled)
                return embedding
            except Exception as e:
                logger.error(f"Resemblyzer embedding failed: {e}")
        
        # Fallback to custom model
        return await self._generate_custom_embedding(audio_data)
    
    async def _generate_custom_embedding(self, audio_data: np.ndarray) -> np.ndarray:
        """Generate embedding using custom model"""
        self.biometric_model.eval()
        
        with torch.no_grad():
            # Convert to mel spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=audio_data,
                sr=22050,
                n_mels=80,
                hop_length=256,
                win_length=1024
            )
            
            mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
            mel_spec = (mel_spec - mel_spec.mean()) / mel_spec.std()
            
            # Convert to tensor
            mel_tensor = torch.FloatTensor(mel_spec.T).unsqueeze(0).to(self.device)
            
            # Generate embedding
            embedding = self.biometric_model(mel_tensor)
            
            return embedding.cpu().numpy().squeeze()
    
    async def _verify_user(self, user_id: str, test_embedding: np.ndarray, audio_data: np.ndarray) -> Tuple[bool, float]:
        """Verify claimed user identity"""
        if user_id not in self.voice_prints:
            return False, 0.0
        
        voice_print = self.voice_prints[user_id]
        
        # Calculate similarity
        similarity = 1 - cosine(voice_print.voice_embedding, test_embedding)
        
        # Additional verification using custom model if available
        if self.biometric_model:
            try:
                # Generate embeddings for both stored and test audio
                stored_embedding_tensor = torch.FloatTensor(voice_print.voice_embedding).unsqueeze(0).to(self.device)
                test_embedding_tensor = torch.FloatTensor(test_embedding).unsqueeze(0).to(self.device)
                
                # Use verification head
                with torch.no_grad():
                    verification_score = self.biometric_model.verification_head(
                        torch.cat([stored_embedding_tensor, test_embedding_tensor], dim=1)
                    ).item()
                
                # Combine scores
                confidence = (similarity * 0.6 + verification_score * 0.4)
            except Exception as e:
                logger.error(f"Custom verification failed: {e}")
                confidence = similarity
        else:
            confidence = similarity
        
        # Update voice print statistics
        voice_print.verification_count += 1
        if confidence >= self.security_policy.min_confidence_threshold:
            success = True
        else:
            success = False
            voice_print.false_rejection_count += 1
        
        return success, confidence
    
    async def _identify_user(self, test_embedding: np.ndarray, audio_data: np.ndarray) -> Tuple[Optional[str], float]:
        """Identify user from voice embedding"""
        best_user = None
        best_confidence = 0.0
        
        for user_id, voice_print in self.voice_prints.items():
            similarity = 1 - cosine(voice_print.voice_embedding, test_embedding)
            
            if similarity > best_confidence:
                best_confidence = similarity
                best_user = user_id
        
        # Only return identification if confidence is high enough
        if best_confidence >= self.security_policy.min_confidence_threshold:
            return best_user, best_confidence
        else:
            return None, best_confidence
    
    async def _calculate_enrollment_quality(self, embeddings: List[np.ndarray], audio_samples: List[np.ndarray]) -> float:
        """Calculate quality score for enrollment"""
        try:
            # Consistency of embeddings
            embedding_similarities = []
            for i in range(len(embeddings)):
                for j in range(i + 1, len(embeddings)):
                    similarity = 1 - cosine(embeddings[i], embeddings[j])
                    embedding_similarities.append(similarity)
            
            consistency_score = np.mean(embedding_similarities) if embedding_similarities else 0.5
            
            # Audio quality assessment
            audio_quality_scores = []
            for audio_data in audio_samples:
                # Signal-to-noise ratio estimation
                signal_power = np.mean(audio_data ** 2)
                noise_power = np.var(audio_data - np.mean(audio_data))
                snr = 10 * np.log10(signal_power / (noise_power + 1e-10))
                snr_score = min(1.0, max(0.0, (snr - 10) / 30))
                
                audio_quality_scores.append(snr_score)
            
            audio_quality = np.mean(audio_quality_scores)
            
            # Combined quality score
            quality_score = (consistency_score * 0.7 + audio_quality * 0.3)
            
            return max(0.0, min(1.0, quality_score))
            
        except Exception as e:
            logger.error(f"Quality calculation failed: {e}")
            return 0.5
    
    async def _save_voice_print(self, voice_print: VoicePrint):
        """Save encrypted voice print"""
        voice_prints_dir = self.config.get('voice_prints_dir', 'data/security/voice_prints')
        os.makedirs(voice_prints_dir, exist_ok=True)
        
        # Prepare data for encryption
        voice_print_data = {
            'user_id': voice_print.user_id,
            'voice_embedding': voice_print.voice_embedding.tolist(),
            'voice_characteristics': voice_print.voice_characteristics,
            'enrollment_samples': voice_print.enrollment_samples,
            'quality_score': voice_print.quality_score,
            'created_at': voice_print.created_at.isoformat(),
            'last_updated': voice_print.last_updated.isoformat(),
            'verification_count': voice_print.verification_count,
            'false_rejection_count': voice_print.false_rejection_count
        }
        
        # Encrypt and save
        serialized_data = pickle.dumps(voice_print_data)
        encrypted_data = self.cipher_suite.encrypt(serialized_data)
        
        print_path = os.path.join(voice_prints_dir, f"{voice_print.user_id}.enc")
        with open(print_path, 'wb') as f:
            f.write(encrypted_data)
        
        logger.info(f"💾 Voice print saved: {voice_print.user_id}")
    
    async def _handle_failed_authentication(self, user_id: str):
        """Handle failed authentication attempt"""
        current_time = datetime.now()
        
        if user_id not in self.failed_attempts:
            self.failed_attempts[user_id] = []
        
        self.failed_attempts[user_id].append(current_time)
        
        # Check if user should be locked
        recent_failures = [
            attempt_time for attempt_time in self.failed_attempts[user_id]
            if (current_time - attempt_time).total_seconds() < self.security_policy.lockout_duration
        ]
        
        if len(recent_failures) >= self.security_policy.max_failed_attempts:
            self.locked_users[user_id] = current_time
            logger.warning(f"🔒 User locked due to failed attempts: {user_id}")
    
    async def _log_auth_attempt(self, 
                              user_id: Optional[str],
                              method: AuthenticationMethod,
                              success: bool,
                              confidence: float,
                              threat_level: ThreatLevel,
                              context: Dict[str, Any]):
        """Log authentication attempt"""
        attempt = AuthenticationAttempt(
            attempt_id=secrets.token_urlsafe(16),
            user_id=user_id,
            method=method,
            timestamp=datetime.now(),
            success=success,
            confidence_score=confidence,
            threat_level=threat_level,
            audio_features={},  # Could store audio features for analysis
            ip_address=context.get('ip_address'),
            device_fingerprint=context.get('device_fingerprint'),
            behavioral_score=context.get('behavioral_score', 0.0)
        )
        
        self.auth_attempts.append(attempt)
        
        # Keep only recent attempts (last 1000)
        if len(self.auth_attempts) > 1000:
            self.auth_attempts = self.auth_attempts[-1000:]
        
        # Log to file if configured
        log_file = self.config.get('auth_log_file')
        if log_file:
            try:
                log_entry = {
                    'timestamp': attempt.timestamp.isoformat(),
                    'user_id': attempt.user_id,
                    'method': attempt.method.value,
                    'success': attempt.success,
                    'confidence': attempt.confidence_score,
                    'threat_level': attempt.threat_level.name,
                    'ip_address': attempt.ip_address
                }
                
                with open(log_file, 'a') as f:
                    f.write(json.dumps(log_entry) + '\n')
            except Exception as e:
                logger.error(f"Failed to write auth log: {e}")
    
    def _update_auth_stats(self, success: bool, auth_time: float, confidence: float):
        """Update authentication statistics"""
        self.auth_stats['total_attempts'] += 1
        
        if success:
            self.auth_stats['successful_auths'] += 1
        else:
            self.auth_stats['failed_auths'] += 1
        
        # Update average auth time
        total = self.auth_stats['total_attempts']
        current_avg = self.auth_stats['avg_auth_time']
        self.auth_stats['avg_auth_time'] = (current_avg * (total - 1) + auth_time) / total
    
    async def create_session(self, 
                           user_id: str, 
                           security_level: SecurityLevel,
                           authentication_method: AuthenticationMethod,
                           context: Optional[Dict[str, Any]] = None) -> str:
        """Create authenticated session"""
        session_id = secrets.token_urlsafe(32)
        
        session = UserSession(
            session_id=session_id,
            user_id=user_id,
            security_level=security_level,
            created_at=datetime.now(),
            last_activity=datetime.now(),
            authentication_method=authentication_method,
            device_fingerprint=context.get('device_fingerprint') if context else None
        )
        
        self.active_sessions[session_id] = session
        
        logger.info(f"🎫 Session created: {session_id} for user {user_id}")
        return session_id
    
    async def validate_session(self, session_id: str) -> Tuple[bool, Optional[UserSession]]:
        """Validate active session"""
        if session_id not in self.active_sessions:
            return False, None
        
        session = self.active_sessions[session_id]
        current_time = datetime.now()
        
        # Check session timeout
        session_age = (current_time - session.last_activity).total_seconds()
        if session_age > self.security_policy.session_timeout:
            del self.active_sessions[session_id]
            logger.info(f"🕐 Session expired: {session_id}")
            return False, None
        
        # Update last activity
        session.last_activity = current_time
        
        return True, session
    
    async def continuous_authentication(self, 
                                      session_id: str, 
                                      audio_data: np.ndarray,
                                      context: Optional[Dict[str, Any]] = None) -> Tuple[bool, float]:
        """Perform continuous authentication"""
        if not self.security_policy.require_continuous_auth:
            return True, 1.0
        
        valid, session = await self.validate_session(session_id)
        if not valid:
            return False, 0.0
        
        try:
            # Perform lightweight authentication
            success, _, confidence, _ = await self.authenticate_user(
                audio_data, session.user_id, context
            )
            
            # Update continuous auth score
            if success:
                session.continuous_auth_score = min(1.0, session.continuous_auth_score * 0.9 + confidence * 0.1)
            else:
                session.continuous_auth_score *= 0.8
            
            # Check if continuous auth score is acceptable
            if session.continuous_auth_score < 0.6:
                logger.warning(f"🔄 Continuous authentication failed for session: {session_id}")
                return False, session.continuous_auth_score
            
            return True, session.continuous_auth_score
            
        except Exception as e:
            logger.error(f"Continuous authentication error: {e}")
            return False, 0.0
    
    def revoke_session(self, session_id: str) -> bool:
        """Revoke active session"""
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]
            logger.info(f"🚫 Session revoked: {session_id}")
            return True
        return False
    
    def get_user_voice_print(self, user_id: str) -> Optional[VoicePrint]:
        """Get user voice print"""
        return self.voice_prints.get(user_id)
    
    def delete_user_voice_print(self, user_id: str) -> bool:
        """Delete user voice print"""
        if user_id not in self.voice_prints:
            return False
        
        try:
            # Delete encrypted file
            voice_prints_dir = self.config.get('voice_prints_dir', 'data/security/voice_prints')
            print_path = os.path.join(voice_prints_dir, f"{user_id}.enc")
            
            if os.path.exists(print_path):
                os.unlink(print_path)
            
            # Remove from memory
            del self.voice_prints[user_id]
            
            # Remove from behavioral baselines
            if user_id in self.threat_detector.behavioral_baselines:
                del self.threat_detector.behavioral_baselines[user_id]
            
            logger.info(f"🗑️ Voice print deleted: {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete voice print: {e}")
            return False
    
    def get_auth_stats(self) -> Dict[str, Any]:
        """Get authentication statistics"""
        stats = self.auth_stats.copy()
        
        # Calculate rates
        total = stats['total_attempts']
        if total > 0:
            stats['success_rate'] = stats['successful_auths'] / total
            stats['failure_rate'] = stats['failed_auths'] / total
        else:
            stats['success_rate'] = 0.0
            stats['failure_rate'] = 0.0
        
        # Add session information
        stats['active_sessions'] = len(self.active_sessions)
        stats['locked_users'] = len(self.locked_users)
        stats['enrolled_users'] = len(self.voice_prints)
        
        # Recent threat detections
        recent_threats = [
            attempt for attempt in self.auth_attempts[-100:]
            if attempt.threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]
        ]
        stats['recent_threats'] = len(recent_threats)
        
        return stats
    
    def generate_access_token(self, user_id: str, security_level: SecurityLevel, session_id: str) -> str:
        """Generate JWT access token"""
        payload = {
            'user_id': user_id,
            'security_level': security_level.value,
            'session_id': session_id,
            'iat': datetime.utcnow(),
            'exp': datetime.utcnow() + timedelta(seconds=self.security_policy.session_timeout)
        }
        
        token = jwt.encode(payload, self.jwt_secret, algorithm='HS256')
        return token
    
    def verify_access_token(self, token: str) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Verify JWT access token"""
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=['HS256'])
            
            # Check if session is still active
            session_id = payload.get('session_id')
            if session_id not in self.active_sessions:
                return False, None
            
            return True, payload
            
        except jwt.ExpiredSignatureError:
            logger.warning("🕐 Access token expired")
            return False, None
        except jwt.InvalidTokenError as e:
            logger.warning(f"🚫 Invalid access token: {e}")
            return False, None
    
    def cleanup(self):
        """Cleanup authentication engine"""
        logger.info("🧹 Cleaning up voice authentication engine...")
        
        # Clear sensitive data from memory
        self.voice_prints.clear()
        self.active_sessions.clear()
        self.auth_attempts.clear()
        
        # Clear models from GPU memory
        if self.biometric_model:
            del self.biometric_model
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("✅ Voice authentication engine cleanup complete")
