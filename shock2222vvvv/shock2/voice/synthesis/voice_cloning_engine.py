"""
Shock2 Voice Cloning Engine
Advanced voice synthesis with custom voice training and real-time cloning
"""

import asyncio
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import threading
import time
import os
import json
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import pickle
import librosa
from scipy.io import wavfile
import soundfile as sf

# Advanced TTS Models
from transformers import (
    SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan,
    VitsModel, VitsTokenizer, AutoProcessor
)

# Voice Conversion Models
try:
    import resemblyzer
    from resemblyzer import VoiceEncoder, preprocess_wav
    RESEMBLYZER_AVAILABLE = True
except ImportError:
    RESEMBLYZER_AVAILABLE = False
    logging.warning("Resemblyzer not available - voice cloning will be limited")

# Real-Time Voice Conversion
try:
    import so_vits_svc
    SO_VITS_AVAILABLE = True
except ImportError:
    SO_VITS_AVAILABLE = False
    logging.warning("SO-VITS-SVC not available")

# Neural Vocoder
try:
    import vocoder
    VOCODER_AVAILABLE = True
except ImportError:
    VOCODER_AVAILABLE = False

logger = logging.getLogger(__name__)

class VoiceCloneMethod(Enum):
    """Voice cloning methods"""
    SPEAKER_EMBEDDING = "speaker_embedding"
    FINE_TUNING = "fine_tuning"
    VOICE_CONVERSION = "voice_conversion"
    NEURAL_VOCODER = "neural_vocoder"
    REAL_TIME_VC = "real_time_vc"

@dataclass
class VoiceProfile:
    """Voice profile for cloning"""
    profile_id: str
    name: str
    description: str
    speaker_embedding: Optional[np.ndarray] = None
    model_path: Optional[str] = None
    sample_audio_paths: List[str] = field(default_factory=list)
    voice_characteristics: Dict[str, float] = field(default_factory=dict)
    training_data_hours: float = 0.0
    quality_score: float = 0.0
    created_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)

@dataclass
class CloneTrainingConfig:
    """Configuration for voice cloning training"""
    method: VoiceCloneMethod
    epochs: int = 100
    batch_size: int = 16
    learning_rate: float = 1e-4
    sample_rate: int = 22050
    hop_length: int = 256
    win_length: int = 1024
    n_mel: int = 80
    min_audio_length: float = 1.0  # seconds
    max_audio_length: float = 10.0  # seconds
    validation_split: float = 0.1
    early_stopping_patience: int = 10

@dataclass
class SynthesisResult:
    """Voice synthesis result"""
    audio_data: np.ndarray
    sample_rate: int
    text: str
    voice_profile_id: str
    method: VoiceCloneMethod
    quality_score: float
    synthesis_time: float
    speaker_similarity: float

class SpeakerEncoder(nn.Module):
    """Neural speaker encoder for voice embeddings"""
    
    def __init__(self, input_dim: int = 80, hidden_dim: int = 256, embedding_dim: int = 256):
        super().__init__()
        
        self.lstm_layers = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=3,
            batch_first=True,
            bidirectional=True,
            dropout=0.1
        )
        
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, embedding_dim),
            nn.Tanh()
        )
        
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
    
    def forward(self, mel_spectrograms):
        # mel_spectrograms: (batch_size, time_steps, mel_bins)
        
        # LSTM processing
        lstm_out, _ = self.lstm_layers(mel_spectrograms)
        
        # Self-attention
        attended, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Global average pooling
        pooled = torch.mean(attended, dim=1)
        
        # Project to embedding space
        embedding = self.projection(pooled)
        
        # L2 normalize
        embedding = F.normalize(embedding, p=2, dim=1)
        
        return embedding

class VoiceCloneModel(nn.Module):
    """Complete voice cloning model"""
    
    def __init__(self, config: CloneTrainingConfig):
        super().__init__()
        self.config = config
        
        # Speaker encoder
        self.speaker_encoder = SpeakerEncoder(
            input_dim=config.n_mel,
            embedding_dim=256
        )
        
        # Text encoder
        self.text_encoder = nn.Embedding(256, 512)  # Character-level encoding
        self.text_lstm = nn.LSTM(512, 256, batch_first=True, bidirectional=True)
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(512, 8, batch_first=True)
        
        # Decoder
        self.decoder_lstm = nn.LSTM(
            input_size=512 + 256,  # text + speaker embedding
            hidden_size=512,
            num_layers=2,
            batch_first=True
        )
        
        # Mel spectrogram predictor
        self.mel_predictor = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, config.n_mel)
        )
        
        # Stop token predictor
        self.stop_predictor = nn.Sequential(
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
    
    def forward(self, text_tokens, mel_spectrograms, speaker_embedding=None):
        batch_size = text_tokens.size(0)
        
        # Encode text
        text_embedded = self.text_encoder(text_tokens)
        text_encoded, _ = self.text_lstm(text_embedded)
        
        # If speaker embedding not provided, extract from mel spectrograms
        if speaker_embedding is None:
            speaker_embedding = self.speaker_encoder(mel_spectrograms)
        
        # Expand speaker embedding to match text sequence length
        seq_len = text_encoded.size(1)
        speaker_expanded = speaker_embedding.unsqueeze(1).expand(-1, seq_len, -1)
        
        # Combine text and speaker information
        combined = torch.cat([text_encoded, speaker_expanded], dim=-1)
        
        # Decode
        decoder_out, _ = self.decoder_lstm(combined)
        
        # Predict mel spectrograms
        mel_pred = self.mel_predictor(decoder_out)
        
        # Predict stop tokens
        stop_pred = self.stop_predictor(decoder_out)
        
        return mel_pred, stop_pred, speaker_embedding

class Shock2VoiceCloner:
    """Advanced voice cloning system for Shock2"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Training configuration
        self.training_config = CloneTrainingConfig(**config.get('training', {}))
        
        # Voice profiles storage
        self.voice_profiles: Dict[str, VoiceProfile] = {}
        self.current_profile_id = None
        
        # Models
        self.clone_model = None
        self.speaker_encoder = None
        self.vocoder = None
        self.resemblyzer_encoder = None
        
        # TTS Integration
        self.speecht5_processor = None
        self.speecht5_model = None
        self.speecht5_vocoder = None
        
        # Real-time processing
        self.real_time_buffer = []
        self.processing_thread = None
        self.is_processing = False
        
        # Performance tracking
        self.clone_stats = {
            'total_syntheses': 0,
            'successful_syntheses': 0,
            'avg_quality_score': 0.0,
            'avg_synthesis_time': 0.0,
            'voice_profile_usage': {}
        }
        
        # Initialize components
        self._initialize_models()
        self._load_voice_profiles()
    
    def _initialize_models(self):
        """Initialize voice cloning models"""
        logger.info("🎭 Initializing voice cloning models...")
        
        # Initialize custom clone model
        self.clone_model = VoiceCloneModel(self.training_config).to(self.device)
        
        # Load pre-trained weights if available
        model_path = self.config.get('pretrained_model_path')
        if model_path and os.path.exists(model_path):
            try:
                checkpoint = torch.load(model_path, map_location=self.device)
                self.clone_model.load_state_dict(checkpoint['model_state_dict'])
                logger.info("✅ Pre-trained clone model loaded")
            except Exception as e:
                logger.warning(f"Failed to load pre-trained model: {e}")
        
        # Initialize Resemblyzer for speaker embeddings
        if RESEMBLYZER_AVAILABLE:
            try:
                self.resemblyzer_encoder = resemblyzer.VoiceEncoder()
                logger.info("✅ Resemblyzer encoder loaded")
            except Exception as e:
                logger.error(f"Failed to load Resemblyzer: {e}")
        
        # Initialize SpeechT5 for high-quality synthesis
        try:
            self.speecht5_processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
            self.speecht5_model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
            self.speecht5_vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
            logger.info("✅ SpeechT5 models loaded")
        except Exception as e:
            logger.error(f"Failed to load SpeechT5: {e}")
        
        logger.info("✅ Voice cloning models initialized")
    
    def _load_voice_profiles(self):
        """Load existing voice profiles"""
        profiles_dir = self.config.get('profiles_dir', 'data/voice_profiles')
        
        if not os.path.exists(profiles_dir):
            os.makedirs(profiles_dir, exist_ok=True)
            return
        
        for profile_file in os.listdir(profiles_dir):
            if profile_file.endswith('.json'):
                try:
                    profile_path = os.path.join(profiles_dir, profile_file)
                    with open(profile_path, 'r') as f:
                        profile_data = json.load(f)
                    
                    profile = VoiceProfile(**profile_data)
                    
                    # Load speaker embedding if exists
                    embedding_path = profile_path.replace('.json', '_embedding.npy')
                    if os.path.exists(embedding_path):
                        profile.speaker_embedding = np.load(embedding_path)
                    
                    self.voice_profiles[profile.profile_id] = profile
                    logger.info(f"📁 Loaded voice profile: {profile.name}")
                    
                except Exception as e:
                    logger.error(f"Failed to load profile {profile_file}: {e}")
        
        logger.info(f"✅ Loaded {len(self.voice_profiles)} voice profiles")
    
    async def create_voice_profile(self, 
                                 profile_id: str, 
                                 name: str, 
                                 description: str,
                                 audio_samples: List[str],
                                 method: VoiceCloneMethod = VoiceCloneMethod.SPEAKER_EMBEDDING) -> VoiceProfile:
        """Create new voice profile from audio samples"""
        logger.info(f"🎤 Creating voice profile: {name}")
        
        try:
            # Validate audio samples
            valid_samples = []
            total_duration = 0.0
            
            for sample_path in audio_samples:
                if os.path.exists(sample_path):
                    try:
                        audio, sr = librosa.load(sample_path, sr=self.training_config.sample_rate)
                        duration = len(audio) / sr
                        
                        if self.training_config.min_audio_length <= duration <= self.training_config.max_audio_length:
                            valid_samples.append(sample_path)
                            total_duration += duration
                        else:
                            logger.warning(f"Audio sample {sample_path} duration {duration:.2f}s outside valid range")
                    except Exception as e:
                        logger.error(f"Failed to load audio sample {sample_path}: {e}")
                else:
                    logger.warning(f"Audio sample not found: {sample_path}")
            
            if len(valid_samples) < 3:
                raise ValueError("Need at least 3 valid audio samples for voice cloning")
            
            # Create voice profile
            profile = VoiceProfile(
                profile_id=profile_id,
                name=name,
                description=description,
                sample_audio_paths=valid_samples,
                training_data_hours=total_duration / 3600,
                voice_characteristics=await self._analyze_voice_characteristics(valid_samples)
            )
            
            # Generate speaker embedding
            if method == VoiceCloneMethod.SPEAKER_EMBEDDING:
                profile.speaker_embedding = await self._generate_speaker_embedding(valid_samples)
                profile.quality_score = await self._evaluate_embedding_quality(profile.speaker_embedding, valid_samples)
            
            elif method == VoiceCloneMethod.FINE_TUNING:
                model_path = await self._fine_tune_model(profile_id, valid_samples)
                profile.model_path = model_path
                profile.quality_score = await self._evaluate_model_quality(model_path, valid_samples)
            
            # Save profile
            await self._save_voice_profile(profile)
            self.voice_profiles[profile_id] = profile
            
            logger.info(f"✅ Voice profile created: {name} (quality: {profile.quality_score:.2f})")
            return profile
            
        except Exception as e:
            logger.error(f"Failed to create voice profile: {e}")
            raise
    
    async def _analyze_voice_characteristics(self, audio_samples: List[str]) -> Dict[str, float]:
        """Analyze voice characteristics from audio samples"""
        characteristics = {
            'fundamental_frequency': 0.0,
            'formant_f1': 0.0,
            'formant_f2': 0.0,
            'spectral_centroid': 0.0,
            'spectral_bandwidth': 0.0,
            'zero_crossing_rate': 0.0,
            'mfcc_variance': 0.0,
            'pitch_variance': 0.0,
            'energy_variance': 0.0
        }
        
        all_features = []
        
        for sample_path in audio_samples:
            try:
                audio, sr = librosa.load(sample_path, sr=self.training_config.sample_rate)
                
                # Fundamental frequency (pitch)
                f0, voiced_flag, voiced_probs = librosa.pyin(
                    audio, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7')
                )
                f0_clean = f0[voiced_flag]
                if len(f0_clean) > 0:
                    characteristics['fundamental_frequency'] += np.mean(f0_clean)
                    characteristics['pitch_variance'] += np.var(f0_clean)
                
                # Formants (simplified estimation)
                # This is a simplified approach - real formant analysis requires more sophisticated methods
                stft = librosa.stft(audio)
                magnitude = np.abs(stft)
                freqs = librosa.fft_frequencies(sr=sr)
                
                # Find peaks in average spectrum
                avg_spectrum = np.mean(magnitude, axis=1)
                peaks = librosa.util.peak_pick(avg_spectrum, pre_max=3, post_max=3, pre_avg=3, post_avg=5, delta=0.1, wait=10)
                
                if len(peaks) >= 2:
                    characteristics['formant_f1'] += freqs[peaks[0]]
                    characteristics['formant_f2'] += freqs[peaks[1]]
                
                # Spectral features
                spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
                characteristics['spectral_centroid'] += np.mean(spectral_centroids)
                
                spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0]
                characteristics['spectral_bandwidth'] += np.mean(spectral_bandwidth)
                
                # Zero crossing rate
                zcr = librosa.feature.zero_crossing_rate(audio)[0]
                characteristics['zero_crossing_rate'] += np.mean(zcr)
                
                # MFCC variance
                mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
                characteristics['mfcc_variance'] += np.mean(np.var(mfccs, axis=1))
                
                # Energy variance
                energy = librosa.feature.rms(y=audio)[0]
                characteristics['energy_variance'] += np.var(energy)
                
                all_features.append({
                    'f0': f0_clean,
                    'spectral_centroid': spectral_centroids,
                    'mfccs': mfccs
                })
                
            except Exception as e:
                logger.error(f"Failed to analyze {sample_path}: {e}")
        
        # Average characteristics
        num_samples = len(audio_samples)
        for key in characteristics:
            characteristics[key] /= num_samples
        
        return characteristics
    
    async def _generate_speaker_embedding(self, audio_samples: List[str]) -> np.ndarray:
        """Generate speaker embedding from audio samples"""
        if not self.resemblyzer_encoder:
            # Fallback to custom speaker encoder
            return await self._generate_custom_speaker_embedding(audio_samples)
        
        embeddings = []
        
        for sample_path in audio_samples:
            try:
                # Load and preprocess audio
                wav = preprocess_wav(sample_path)
                
                # Generate embedding
                embedding = self.resemblyzer_encoder.embed_utterance(wav)
                embeddings.append(embedding)
                
            except Exception as e:
                logger.error(f"Failed to generate embedding for {sample_path}: {e}")
        
        if not embeddings:
            raise ValueError("No valid embeddings generated")
        
        # Average embeddings
        speaker_embedding = np.mean(embeddings, axis=0)
        
        # Normalize
        speaker_embedding = speaker_embedding / np.linalg.norm(speaker_embedding)
        
        return speaker_embedding
    
    async def _generate_custom_speaker_embedding(self, audio_samples: List[str]) -> np.ndarray:
        """Generate speaker embedding using custom encoder"""
        self.clone_model.eval()
        embeddings = []
        
        with torch.no_grad():
            for sample_path in audio_samples:
                try:
                    # Load audio
                    audio, sr = librosa.load(sample_path, sr=self.training_config.sample_rate)
                    
                    # Convert to mel spectrogram
                    mel_spec = librosa.feature.melspectrogram(
                        y=audio,
                        sr=sr,
                        n_mels=self.training_config.n_mel,
                        hop_length=self.training_config.hop_length,
                        win_length=self.training_config.win_length
                    )
                    
                    # Convert to log scale
                    mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
                    
                    # Normalize
                    mel_spec = (mel_spec - mel_spec.mean()) / mel_spec.std()
                    
                    # Convert to tensor
                    mel_tensor = torch.FloatTensor(mel_spec.T).unsqueeze(0).to(self.device)
                    
                    # Generate embedding
                    embedding = self.clone_model.speaker_encoder(mel_tensor)
                    embeddings.append(embedding.cpu().numpy().squeeze())
                    
                except Exception as e:
                    logger.error(f"Failed to generate custom embedding for {sample_path}: {e}")
        
        if not embeddings:
            raise ValueError("No valid custom embeddings generated")
        
        # Average embeddings
        speaker_embedding = np.mean(embeddings, axis=0)
        
        return speaker_embedding
    
    async def _fine_tune_model(self, profile_id: str, audio_samples: List[str]) -> str:
        """Fine-tune model for specific voice"""
        logger.info(f"🎓 Fine-tuning model for profile: {profile_id}")
        
        # Prepare training data
        train_data = await self._prepare_training_data(audio_samples)
        
        # Create data loader
        dataset = VoiceCloneDataset(train_data, self.training_config)
        dataloader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=self.training_config.batch_size, 
            shuffle=True,
            num_workers=2
        )
        
        # Setup training
        optimizer = torch.optim.Adam(self.clone_model.parameters(), lr=self.training_config.learning_rate)
        criterion = nn.MSELoss()
        
        # Training loop
        self.clone_model.train()
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.training_config.epochs):
            total_loss = 0.0
            
            for batch in dataloader:
                text_tokens = batch['text_tokens'].to(self.device)
                mel_spectrograms = batch['mel_spectrograms'].to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass
                mel_pred, stop_pred, speaker_embedding = self.clone_model(text_tokens, mel_spectrograms)
                
                # Calculate loss
                mel_loss = criterion(mel_pred, mel_spectrograms)
                total_loss += mel_loss.item()
                
                # Backward pass
                mel_loss.backward()
                optimizer.step()
            
            avg_loss = total_loss / len(dataloader)
            
            # Early stopping
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
                
                # Save best model
                model_path = f"data/voice_profiles/{profile_id}_model.pth"
                torch.save({
                    'model_state_dict': self.clone_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'loss': avg_loss
                }, model_path)
            else:
                patience_counter += 1
                
                if patience_counter >= self.training_config.early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch + 1}/{self.training_config.epochs}, Loss: {avg_loss:.4f}")
        
        logger.info(f"✅ Model fine-tuning completed for {profile_id}")
        return model_path
    
    async def _prepare_training_data(self, audio_samples: List[str]) -> List[Dict[str, Any]]:
        """Prepare training data from audio samples"""
        training_data = []
        
        for sample_path in audio_samples:
            try:
                # Load audio
                audio, sr = librosa.load(sample_path, sr=self.training_config.sample_rate)
                
                # Convert to mel spectrogram
                mel_spec = librosa.feature.melspectrogram(
                    y=audio,
                    sr=sr,
                    n_mels=self.training_config.n_mel,
                    hop_length=self.training_config.hop_length,
                    win_length=self.training_config.win_length
                )
                
                mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
                mel_spec = (mel_spec - mel_spec.mean()) / mel_spec.std()
                
                # Generate dummy text tokens (in real implementation, you'd have transcriptions)
                text_tokens = np.random.randint(0, 256, size=(mel_spec.shape[1] // 4,))
                
                training_data.append({
                    'audio_path': sample_path,
                    'mel_spectrogram': mel_spec.T,  # Transpose for time-first format
                    'text_tokens': text_tokens
                })
                
            except Exception as e:
                logger.error(f"Failed to prepare training data for {sample_path}: {e}")
        
        return training_data
    
    async def _evaluate_embedding_quality(self, embedding: np.ndarray, audio_samples: List[str]) -> float:
        """Evaluate quality of speaker embedding"""
        if not self.resemblyzer_encoder:
            return 0.8  # Default score if no evaluation possible
        
        try:
            # Generate embeddings for each sample
            sample_embeddings = []
            
            for sample_path in audio_samples:
                wav = preprocess_wav(sample_path)
                sample_embedding = self.resemblyzer_encoder.embed_utterance(wav)
                sample_embeddings.append(sample_embedding)
            
            # Calculate similarity scores
            similarities = []
            for sample_embedding in sample_embeddings:
                similarity = np.dot(embedding, sample_embedding) / (
                    np.linalg.norm(embedding) * np.linalg.norm(sample_embedding)
                )
                similarities.append(similarity)
            
            # Quality score is average similarity
            quality_score = np.mean(similarities)
            
            return max(0.0, min(1.0, quality_score))
            
        except Exception as e:
            logger.error(f"Failed to evaluate embedding quality: {e}")
            return 0.5
    
    async def _evaluate_model_quality(self, model_path: str, audio_samples: List[str]) -> float:
        """Evaluate quality of fine-tuned model"""
        # This would involve generating test audio and comparing with originals
        # For now, return a placeholder score
        return 0.85
    
    async def _save_voice_profile(self, profile: VoiceProfile):
        """Save voice profile to disk"""
        profiles_dir = self.config.get('profiles_dir', 'data/voice_profiles')
        os.makedirs(profiles_dir, exist_ok=True)
        
        # Save profile metadata
        profile_path = os.path.join(profiles_dir, f"{profile.profile_id}.json")
        profile_dict = {
            'profile_id': profile.profile_id,
            'name': profile.name,
            'description': profile.description,
            'sample_audio_paths': profile.sample_audio_paths,
            'voice_characteristics': profile.voice_characteristics,
            'training_data_hours': profile.training_data_hours,
            'quality_score': profile.quality_score,
            'created_at': profile.created_at,
            'last_used': profile.last_used,
            'model_path': profile.model_path
        }
        
        with open(profile_path, 'w') as f:
            json.dump(profile_dict, f, indent=2)
        
        # Save speaker embedding
        if profile.speaker_embedding is not None:
            embedding_path = os.path.join(profiles_dir, f"{profile.profile_id}_embedding.npy")
            np.save(embedding_path, profile.speaker_embedding)
        
        logger.info(f"💾 Voice profile saved: {profile.name}")
    
    async def synthesize_speech(self, 
                              text: str, 
                              voice_profile_id: str,
                              method: Optional[VoiceCloneMethod] = None) -> SynthesisResult:
        """Synthesize speech using cloned voice"""
        start_time = time.time()
        
        if voice_profile_id not in self.voice_profiles:
            raise ValueError(f"Voice profile not found: {voice_profile_id}")
        
        profile = self.voice_profiles[voice_profile_id]
        
        # Update last used time
        profile.last_used = time.time()
        
        # Determine synthesis method
        if method is None:
            if profile.speaker_embedding is not None:
                method = VoiceCloneMethod.SPEAKER_EMBEDDING
            elif profile.model_path is not None:
                method = VoiceCloneMethod.FINE_TUNING
            else:
                raise ValueError("No synthesis method available for this profile")
        
        try:
            if method == VoiceCloneMethod.SPEAKER_EMBEDDING:
                audio_data, sample_rate = await self._synthesize_with_embedding(text, profile)
            elif method == VoiceCloneMethod.FINE_TUNING:
                audio_data, sample_rate = await self._synthesize_with_fine_tuned_model(text, profile)
            elif method == VoiceCloneMethod.VOICE_CONVERSION:
                audio_data, sample_rate = await self._synthesize_with_voice_conversion(text, profile)
            else:
                raise ValueError(f"Synthesis method not implemented: {method}")
            
            # Calculate quality metrics
            quality_score = await self._calculate_synthesis_quality(audio_data, profile)
            speaker_similarity = await self._calculate_speaker_similarity(audio_data, profile)
            
            synthesis_time = time.time() - start_time
            
            # Update statistics
            self._update_clone_stats(voice_profile_id, quality_score, synthesis_time, True)
            
            result = SynthesisResult(
                audio_data=audio_data,
                sample_rate=sample_rate,
                text=text,
                voice_profile_id=voice_profile_id,
                method=method,
                quality_score=quality_score,
                synthesis_time=synthesis_time,
                speaker_similarity=speaker_similarity
            )
            
            logger.info(f"🎤 Speech synthesized: '{text[:50]}...' using {profile.name} (quality: {quality_score:.2f})")
            return result
            
        except Exception as e:
            synthesis_time = time.time() - start_time
            self._update_clone_stats(voice_profile_id, 0.0, synthesis_time, False)
            logger.error(f"Speech synthesis failed: {e}")
            raise
    
    async def _synthesize_with_embedding(self, text: str, profile: VoiceProfile) -> Tuple[np.ndarray, int]:
        """Synthesize speech using speaker embedding"""
        if not self.speecht5_model:
            raise RuntimeError("SpeechT5 model not available")
        
        # Process text
        inputs = self.speecht5_processor(text=text, return_tensors="pt")
        
        # Convert speaker embedding to tensor
        speaker_embeddings = torch.FloatTensor(profile.speaker_embedding).unsqueeze(0)
        
        # Generate speech
        with torch.no_grad():
            speech = self.speecht5_model.generate_speech(
                inputs["input_ids"], 
                speaker_embeddings, 
                vocoder=self.speecht5_vocoder
            )
        
        # Convert to numpy
        audio_data = speech.numpy()
        sample_rate = 16000  # SpeechT5 default
        
        return audio_data, sample_rate
    
    async def _synthesize_with_fine_tuned_model(self, text: str, profile: VoiceProfile) -> Tuple[np.ndarray, int]:
        """Synthesize speech using fine-tuned model"""
        if not profile.model_path or not os.path.exists(profile.model_path):
            raise ValueError("Fine-tuned model not available")
        
        # Load fine-tuned model
        checkpoint = torch.load(profile.model_path, map_location=self.device)
        self.clone_model.load_state_dict(checkpoint['model_state_dict'])
        self.clone_model.eval()
        
        # Convert text to tokens (simplified)
        text_tokens = torch.LongTensor([ord(c) for c in text[:100]]).unsqueeze(0).to(self.device)
        
        # Generate mel spectrogram
        with torch.no_grad():
            # Create dummy mel input for autoregressive generation
            mel_input = torch.zeros(1, 1, self.training_config.n_mel).to(self.device)
            
            generated_mels = []
            hidden = None
            
            for _ in range(200):  # Max length
                mel_pred, stop_pred, _ = self.clone_model(text_tokens, mel_input)
                
                generated_mels.append(mel_pred[:, -1:, :])
                mel_input = torch.cat([mel_input, mel_pred[:, -1:, :]], dim=1)
                
                # Check stop condition
                if stop_pred[:, -1, 0] > 0.5:
                    break
            
            # Combine generated mels
            mel_spectrogram = torch.cat(generated_mels, dim=1)
        
        # Convert mel spectrogram to audio (simplified vocoder)
        mel_np = mel_spectrogram.cpu().numpy().squeeze().T
        audio_data = librosa.feature.inverse.mel_to_audio(
            mel_np, 
            sr=self.training_config.sample_rate,
            hop_length=self.training_config.hop_length,
            win_length=self.training_config.win_length
        )
        
        return audio_data, self.training_config.sample_rate
    
    async def _synthesize_with_voice_conversion(self, text: str, profile: VoiceProfile) -> Tuple[np.ndarray, int]:
        """Synthesize speech using voice conversion"""
        # This would require a base TTS system + voice conversion
        # For now, fallback to embedding method
        return await self._synthesize_with_embedding(text, profile)
    
    async def _calculate_synthesis_quality(self, audio_data: np.ndarray, profile: VoiceProfile) -> float:
        """Calculate quality score for synthesized audio"""
        try:
            # Basic quality metrics
            
            # Signal-to-noise ratio estimation
            signal_power = np.mean(audio_data ** 2)
            noise_power = np.var(audio_data - np.mean(audio_data))
            snr = 10 * np.log10(signal_power / (noise_power + 1e-10))
            snr_score = min(1.0, max(0.0, (snr - 10) / 30))  # Normalize 10-40 dB to 0-1
            
            # Spectral quality
            stft = librosa.stft(audio_data)
            magnitude = np.abs(stft)
            
            # Check for spectral artifacts
            spectral_flatness = librosa.feature.spectral_flatness(S=magnitude)[0]
            flatness_score = 1.0 - np.mean(spectral_flatness)  # Lower flatness = better quality
            
            # Dynamic range
            dynamic_range = np.max(audio_data) - np.min(audio_data)
            range_score = min(1.0, dynamic_range / 0.8)  # Normalize to 0-1
            
            # Combine scores
            quality_score = (snr_score * 0.4 + flatness_score * 0.3 + range_score * 0.3)
            
            return max(0.0, min(1.0, quality_score))
            
        except Exception as e:
            logger.error(f"Quality calculation failed: {e}")
            return 0.5
    
    async def _calculate_speaker_similarity(self, audio_data: np.ndarray, profile: VoiceProfile) -> float:
        """Calculate speaker similarity score"""
        if not self.resemblyzer_encoder or profile.speaker_embedding is None:
            return 0.8  # Default score
        
        try:
            # Generate embedding for synthesized audio
            # Save to temporary file for Resemblyzer
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_path = temp_file.name
            
            wavfile.write(temp_path, self.training_config.sample_rate, 
                         (audio_data * 32767).astype(np.int16))
            
            # Generate embedding
            wav = preprocess_wav(temp_path)
            synth_embedding = self.resemblyzer_encoder.embed_utterance(wav)
            
            # Calculate similarity
            similarity = np.dot(profile.speaker_embedding, synth_embedding) / (
                np.linalg.norm(profile.speaker_embedding) * np.linalg.norm(synth_embedding)
            )
            
            # Cleanup
            os.unlink(temp_path)
            
            return max(0.0, min(1.0, similarity))
            
        except Exception as e:
            logger.error(f"Speaker similarity calculation failed: {e}")
            return 0.5
    
    def _update_clone_stats(self, profile_id: str, quality_score: float, synthesis_time: float, success: bool):
        """Update cloning statistics"""
        self.clone_stats['total_syntheses'] += 1
        
        if success:
            self.clone_stats['successful_syntheses'] += 1
            
            # Update averages
            total = self.clone_stats['successful_syntheses']
            
            current_quality = self.clone_stats['avg_quality_score']
            self.clone_stats['avg_quality_score'] = (current_quality * (total - 1) + quality_score) / total
            
            current_time = self.clone_stats['avg_synthesis_time']
            self.clone_stats['avg_synthesis_time'] = (current_time * (total - 1) + synthesis_time) / total
        
        # Update profile usage
        if profile_id not in self.clone_stats['voice_profile_usage']:
            self.clone_stats['voice_profile_usage'][profile_id] = 0
        self.clone_stats['voice_profile_usage'][profile_id] += 1
    
    def get_voice_profiles(self) -> List[VoiceProfile]:
        """Get all available voice profiles"""
        return list(self.voice_profiles.values())
    
    def get_voice_profile(self, profile_id: str) -> Optional[VoiceProfile]:
        """Get specific voice profile"""
        return self.voice_profiles.get(profile_id)
    
    def delete_voice_profile(self, profile_id: str) -> bool:
        """Delete voice profile"""
        if profile_id not in self.voice_profiles:
            return False
        
        try:
            profile = self.voice_profiles[profile_id]
            
            # Delete files
            profiles_dir = self.config.get('profiles_dir', 'data/voice_profiles')
            
            profile_path = os.path.join(profiles_dir, f"{profile_id}.json")
            if os.path.exists(profile_path):
                os.unlink(profile_path)
            
            embedding_path = os.path.join(profiles_dir, f"{profile_id}_embedding.npy")
            if os.path.exists(embedding_path):
                os.unlink(embedding_path)
            
            if profile.model_path and os.path.exists(profile.model_path):
                os.unlink(profile.model_path)
            
            # Remove from memory
            del self.voice_profiles[profile_id]
            
            logger.info(f"🗑️ Voice profile deleted: {profile.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete voice profile: {e}")
            return False
    
    def get_clone_stats(self) -> Dict[str, Any]:
        """Get voice cloning statistics"""
        stats = self.clone_stats.copy()
        
        # Calculate success rate
        total = stats['total_syntheses']
        if total > 0:
            stats['success_rate'] = stats['successful_syntheses'] / total
        else:
            stats['success_rate'] = 0.0
        
        # Add profile information
        stats['total_profiles'] = len(self.voice_profiles)
        stats['profile_quality_scores'] = {
            pid: profile.quality_score 
            for pid, profile in self.voice_profiles.items()
        }
        
        return stats
    
    async def real_time_voice_conversion(self, audio_stream: np.ndarray, target_profile_id: str) -> np.ndarray:
        """Real-time voice conversion"""
        if target_profile_id not in self.voice_profiles:
            raise ValueError(f"Target profile not found: {target_profile_id}")
        
        # This would implement real-time voice conversion
        # For now, return the original audio
        logger.warning("Real-time voice conversion not yet implemented")
        return audio_stream
    
    def cleanup(self):
        """Cleanup voice cloning resources"""
        logger.info("🧹 Cleaning up voice cloning engine...")
        
        # Stop any running processes
        self.is_processing = False
        
        if self.processing_thread:
            self.processing_thread.join(timeout=2.0)
        
        # Clear models from GPU memory
        if self.clone_model:
            del self.clone_model
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("✅ Voice cloning engine cleanup complete")

class VoiceCloneDataset(torch.utils.data.Dataset):
    """Dataset for voice cloning training"""
    
    def __init__(self, training_data: List[Dict[str, Any]], config: CloneTrainingConfig):
        self.training_data = training_data
        self.config = config
    
    def __len__(self):
        return len(self.training_data)
    
    def __getitem__(self, idx):
        item = self.training_data[idx]
        
        return {
            'text_tokens': torch.LongTensor(item['text_tokens']),
            'mel_spectrograms': torch.FloatTensor(item['mel_spectrogram'])
        }
