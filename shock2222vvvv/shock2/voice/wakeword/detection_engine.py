"""
Shock2 Wake Word Detection Engine
Advanced wake word detection with custom training and multiple engine support
"""

import asyncio
import logging
import numpy as np
import threading
import time
import queue
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum
import json
import os

# Audio processing
import sounddevice as sd
import librosa
import webrtcvad
from scipy import signal

# Wake word detection engines
try:
    import pvporcupine
    PORCUPINE_AVAILABLE = True
except ImportError:
    PORCUPINE_AVAILABLE = False
    logging.warning("Porcupine not available")

try:
    import snowboy
    SNOWBOY_AVAILABLE = True
except ImportError:
    SNOWBOY_AVAILABLE = False
    logging.warning("Snowboy not available")

# Machine learning for custom wake words
import torch
import torch.nn as nn
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification

logger = logging.getLogger(__name__)

class WakeWordEngine(Enum):
    """Available wake word detection engines"""
    PORCUPINE = "porcupine"
    SNOWBOY = "snowboy"
    CUSTOM_ML = "custom_ml"
    SIMPLE_ENERGY = "simple_energy"

@dataclass
class WakeWordDetection:
    """Wake word detection result"""
    keyword: str
    confidence: float
    engine: WakeWordEngine
    timestamp: float
    audio_snippet: np.ndarray

class CustomWakeWordModel(nn.Module):
    """Custom neural network for wake word detection"""
    
    def __init__(self, input_size: int = 512, hidden_size: int = 256, num_classes: int = 2):
        super().__init__()
        
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(128, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size // 2, num_classes)
        )
    
    def forward(self, x):
        # x shape: (batch_size, sequence_length)
        x = x.unsqueeze(1)  # Add channel dimension
        features = self.feature_extractor(x)
        features = features.squeeze(-1)  # Remove last dimension
        output = self.classifier(features)
        return output

class Shock2WakeWordDetector:
    """Advanced wake word detection system"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.wake_words = config.get('wake_words', ['hey shock', 'shock two', 'shock'])
        self.primary_engine = WakeWordEngine(config.get('primary_engine', 'porcupine'))
        self.fallback_engines = [WakeWordEngine(e) for e in config.get('fallback_engines', ['custom_ml'])]
        
        # Audio configuration
        self.sample_rate = config.get('sample_rate', 16000)
        self.frame_length = config.get('frame_length', 512)
        self.hop_length = config.get('hop_length', 256)
        
        # Detection parameters
        self.detection_threshold = config.get('detection_threshold', 0.7)
        self.sensitivity = config.get('sensitivity', 0.5)
        self.cooldown_period = config.get('cooldown_period', 2.0)  # seconds
        
        # Audio processing
        self.audio_buffer = queue.Queue(maxsize=100)
        self.vad = webrtcvad.Vad(2)  # Moderate aggressiveness
        
        # Detection engines
        self.engines = {}
        self.custom_model = None
        self.wav2vec_processor = None
        self.wav2vec_model = None
        
        # State tracking
        self.is_listening = False
        self.last_detection_time = 0
        self.detection_callback = None
        
        # Performance tracking
        self.detection_stats = {
            'total_detections': 0,
            'false_positives': 0,
            'true_positives': 0,
            'avg_confidence': 0.0,
            'engine_performance': {}
        }
        
        # Initialize engines
        self._initialize_engines()
    
    def _initialize_engines(self):
        """Initialize all available wake word detection engines"""
        logger.info("🎯 Initializing wake word detection engines...")
        
        # Initialize Porcupine
        if PORCUPINE_AVAILABLE and self.primary_engine == WakeWordEngine.PORCUPINE:
            self._initialize_porcupine()
        
        # Initialize Snowboy
        if SNOWBOY_AVAILABLE and WakeWordEngine.SNOWBOY in [self.primary_engine] + self.fallback_engines:
            self._initialize_snowboy()
        
        # Initialize custom ML model
        if WakeWordEngine.CUSTOM_ML in [self.primary_engine] + self.fallback_engines:
            self._initialize_custom_ml()
        
        # Simple energy detector is always available
        self.engines[WakeWordEngine.SIMPLE_ENERGY] = True
        
        logger.info(f"✅ Initialized {len(self.engines)} wake word engines")
    
    def _initialize_porcupine(self):
        """Initialize Porcupine wake word engine"""
        try:
            access_key = self.config.get('porcupine_access_key')
            if not access_key:
                logger.error("Porcupine access key not provided")
                return
            
            # Built-in keywords
            keywords = []
            keyword_paths = []
            
            # Check for custom keyword files
            custom_keywords_dir = self.config.get('custom_keywords_dir', 'assets/keywords')
            
            for wake_word in self.wake_words:
                # Try to find custom keyword file
                keyword_file = os.path.join(custom_keywords_dir, f"{wake_word.replace(' ', '_')}.ppn")
                
                if os.path.exists(keyword_file):
                    keyword_paths.append(keyword_file)
                else:
                    # Use built-in keywords if available
                    if wake_word.lower() in ['hey google', 'alexa', 'computer']:
                        keywords.append(wake_word.lower())
            
            # Create Porcupine instance
            if keywords or keyword_paths:
                porcupine = pvporcupine.create(
                    access_key=access_key,
                    keywords=keywords if keywords else None,
                    keyword_paths=keyword_paths if keyword_paths else None,
                    sensitivities=[self.sensitivity] * (len(keywords) + len(keyword_paths))
                )
                
                self.engines[WakeWordEngine.PORCUPINE] = porcupine
                logger.info("✅ Porcupine initialized")
            else:
                logger.warning("No valid keywords found for Porcupine")
                
        except Exception as e:
            logger.error(f"Failed to initialize Porcupine: {e}")
    
    def _initialize_snowboy(self):
        """Initialize Snowboy wake word engine"""
        try:
            # Snowboy model files
            model_files = []
            
            for wake_word in self.wake_words:
                model_file = os.path.join('assets/snowboy_models', f"{wake_word.replace(' ', '_')}.pmdl")
                if os.path.exists(model_file):
                    model_files.append(model_file)
            
            if model_files:
                # Create Snowboy detector
                detector = snowboy.SnowboyDetect(
                    resource_filename='assets/snowboy_models/common.res',
                    model_str=','.join(model_files)
                )
                
                detector.SetSensitivity(','.join([str(self.sensitivity)] * len(model_files)))
                detector.SetAudioGain(1.0)
                
                self.engines[WakeWordEngine.SNOWBOY] = detector
                logger.info("✅ Snowboy initialized")
            else:
                logger.warning("No Snowboy model files found")
                
        except Exception as e:
            logger.error(f"Failed to initialize Snowboy: {e}")
    
    def _initialize_custom_ml(self):
        """Initialize custom ML wake word detection"""
        try:
            # Load pre-trained model if available
            model_path = self.config.get('custom_model_path', 'assets/models/wake_word_model.pth')
            
            if os.path.exists(model_path):
                self.custom_model = CustomWakeWordModel()
                self.custom_model.load_state_dict(torch.load(model_path, map_location='cpu'))
                self.custom_model.eval()
                logger.info("✅ Custom ML model loaded")
            else:
                # Create new model for training
                self.custom_model = CustomWakeWordModel()
                logger.info("✅ Custom ML model created (untrained)")
            
            # Initialize Wav2Vec2 for feature extraction
            try:
                self.wav2vec_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
                self.wav2vec_model = Wav2Vec2ForSequenceClassification.from_pretrained("facebook/wav2vec2-base")
                logger.info("✅ Wav2Vec2 feature extractor loaded")
            except Exception as e:
                logger.warning(f"Wav2Vec2 not available: {e}")
            
            self.engines[WakeWordEngine.CUSTOM_ML] = self.custom_model
            
        except Exception as e:
            logger.error(f"Failed to initialize custom ML: {e}")
    
    async def start_detection(self, callback: Callable[[WakeWordDetection], None]):
        """Start wake word detection"""
        self.detection_callback = callback
        self.is_listening = True
        
        logger.info("🎧 Starting wake word detection...")
        
        # Start audio capture
        await self._start_audio_capture()
    
    async def _start_audio_capture(self):
        """Start audio capture for wake word detection"""
        def audio_callback(indata, frames, time, status):
            """Audio input callback"""
            if status:
                logger.warning(f"Audio input status: {status}")
            
            # Add audio to buffer
            try:
                self.audio_buffer.put_nowait(indata[:, 0].copy())
            except queue.Full:
                # Remove oldest frame if buffer is full
                try:
                    self.audio_buffer.get_nowait()
                    self.audio_buffer.put_nowait(indata[:, 0].copy())
                except queue.Empty:
                    pass
        
        # Start audio stream
        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            callback=audio_callback,
            blocksize=self.frame_length,
            dtype=np.float32
        ):
            logger.info("🎤 Audio capture started for wake word detection")
            
            # Process audio continuously
            while self.is_listening:
                await self._process_audio_buffer()
                await asyncio.sleep(0.01)  # Small delay
    
    async def _process_audio_buffer(self):
        """Process audio buffer for wake word detection"""
        if self.audio_buffer.empty():
            return
        
        # Collect audio frames
        audio_frames = []
        while not self.audio_buffer.empty() and len(audio_frames) < 50:  # ~1 second of audio
            try:
                frame = self.audio_buffer.get_nowait()
                audio_frames.append(frame)
            except queue.Empty:
                break
        
        if len(audio_frames) < 10:  # Need minimum audio
            return
        
        # Combine frames
        audio_data = np.concatenate(audio_frames)
        
        # Check for voice activity
        if not self._has_voice_activity(audio_data):
            return
        
        # Check cooldown period
        current_time = time.time()
        if current_time - self.last_detection_time < self.cooldown_period:
            return
        
        # Run detection with primary engine
        detection = await self._detect_with_engine(audio_data, self.primary_engine)
        
        # Try fallback engines if primary fails
        if not detection or detection.confidence < self.detection_threshold:
            for engine in self.fallback_engines:
                if engine in self.engines:
                    fallback_detection = await self._detect_with_engine(audio_data, engine)
                    if fallback_detection and fallback_detection.confidence > detection.confidence if detection else 0:
                        detection = fallback_detection
                        break
        
        # Process detection result
        if detection and detection.confidence >= self.detection_threshold:
            self.last_detection_time = current_time
            self._update_detection_stats(detection, True)
            
            if self.detection_callback:
                self.detection_callback(detection)
            
            logger.info(f"🎯 Wake word detected: '{detection.keyword}' (confidence: {detection.confidence:.2f}, engine: {detection.engine.value})")
    
    def _has_voice_activity(self, audio_data: np.ndarray) -> bool:
        """Check if audio contains voice activity"""
        try:
            # Convert to bytes for VAD
            audio_int16 = (audio_data * 32767).astype(np.int16)
            
            # Check energy level
            energy = np.mean(audio_data ** 2)
            if energy < 0.001:  # Very quiet
                return False
            
            # Use WebRTC VAD
            frame_size = 320  # 20ms at 16kHz
            voice_frames = 0
            total_frames = 0
            
            for i in range(0, len(audio_int16) - frame_size, frame_size):
                frame = audio_int16[i:i + frame_size].tobytes()
                if len(frame) == frame_size * 2:  # Correct frame size
                    if self.vad.is_speech(frame, self.sample_rate):
                        voice_frames += 1
                    total_frames += 1
            
            # Require at least 30% voice activity
            return total_frames > 0 and (voice_frames / total_frames) > 0.3
            
        except Exception as e:
            logger.debug(f"VAD check failed: {e}")
            return True  # Default to processing if VAD fails
    
    async def _detect_with_engine(self, audio_data: np.ndarray, engine: WakeWordEngine) -> Optional[WakeWordDetection]:
        """Detect wake word with specific engine"""
        if engine not in self.engines:
            return None
        
        try:
            if engine == WakeWordEngine.PORCUPINE:
                return await self._detect_porcupine(audio_data)
            elif engine == WakeWordEngine.SNOWBOY:
                return await self._detect_snowboy(audio_data)
            elif engine == WakeWordEngine.CUSTOM_ML:
                return await self._detect_custom_ml(audio_data)
            elif engine == WakeWordEngine.SIMPLE_ENERGY:
                return await self._detect_simple_energy(audio_data)
            
        except Exception as e:
            logger.error(f"Detection failed with {engine.value}: {e}")
            return None
    
    async def _detect_porcupine(self, audio_data: np.ndarray) -> Optional[WakeWordDetection]:
        """Detect wake word using Porcupine"""
        porcupine = self.engines[WakeWordEngine.PORCUPINE]
        
        # Convert audio to int16
        audio_int16 = (audio_data * 32767).astype(np.int16)
        
        # Process in chunks
        frame_length = porcupine.frame_length
        
        for i in range(0, len(audio_int16) - frame_length, frame_length):
            frame = audio_int16[i:i + frame_length]
            keyword_index = porcupine.process(frame)
            
            if keyword_index >= 0:
                # Wake word detected
                keyword = self.wake_words[keyword_index] if keyword_index < len(self.wake_words) else "unknown"
                
                return WakeWordDetection(
                    keyword=keyword,
                    confidence=0.9,  # Porcupine doesn't provide confidence
                    engine=WakeWordEngine.PORCUPINE,
                    timestamp=time.time(),
                    audio_snippet=audio_data
                )
        
        return None
    
    async def _detect_snowboy(self, audio_data: np.ndarray) -> Optional[WakeWordDetection]:
        """Detect wake word using Snowboy"""
        detector = self.engines[WakeWordEngine.SNOWBOY]
        
        # Convert audio to int16
        audio_int16 = (audio_data * 32767).astype(np.int16)
        
        # Process audio
        result = detector.RunDetection(audio_int16.tobytes())
        
        if result > 0:
            # Wake word detected
            keyword_index = result - 1
            keyword = self.wake_words[keyword_index] if keyword_index < len(self.wake_words) else "unknown"
            
            return WakeWordDetection(
                keyword=keyword,
                confidence=0.8,  # Snowboy doesn't provide confidence
                engine=WakeWordEngine.SNOWBOY,
                timestamp=time.time(),
                audio_snippet=audio_data
            )
        
        return None
    
    async def _detect_custom_ml(self, audio_data: np.ndarray) -> Optional[WakeWordDetection]:
        """Detect wake word using custom ML model"""
        if not self.custom_model:
            return None
        
        try:
            # Extract features
            features = self._extract_audio_features(audio_data)
            
            if features is None:
                return None
            
            # Run inference
            with torch.no_grad():
                features_tensor = torch.FloatTensor(features).unsqueeze(0)
                outputs = self.custom_model(features_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidence = probabilities[0, 1].item()  # Probability of wake word class
            
            if confidence > self.detection_threshold:
                return WakeWordDetection(
                    keyword="shock2",  # Default keyword for custom model
                    confidence=confidence,
                    engine=WakeWordEngine.CUSTOM_ML,
                    timestamp=time.time(),
                    audio_snippet=audio_data
                )
            
        except Exception as e:
            logger.error(f"Custom ML detection failed: {e}")
        
        return None
    
    async def _detect_simple_energy(self, audio_data: np.ndarray) -> Optional[WakeWordDetection]:
        """Simple energy-based wake word detection (fallback)"""
        # Calculate energy and spectral features
        energy = np.mean(audio_data ** 2)
        
        # Simple threshold-based detection
        if energy > 0.01:  # Adjust threshold as needed
            # Calculate spectral centroid for voice-like characteristics
            stft = librosa.stft(audio_data, hop_length=self.hop_length)
            spectral_centroids = librosa.feature.spectral_centroid(S=np.abs(stft))[0]
            avg_centroid = np.mean(spectral_centroids)
            
            # Voice typically has centroid between 1000-3000 Hz
            if 1000 < avg_centroid < 3000:
                confidence = min(energy * 10, 1.0)  # Scale energy to confidence
                
                return WakeWordDetection(
                    keyword="voice_detected",
                    confidence=confidence,
                    engine=WakeWordEngine.SIMPLE_ENERGY,
                    timestamp=time.time(),
                    audio_snippet=audio_data
                )
        
        return None
    
    def _extract_audio_features(self, audio_data: np.ndarray) -> Optional[np.ndarray]:
        """Extract features for custom ML model"""
        try:
            # Use Wav2Vec2 if available
            if self.wav2vec_processor and self.wav2vec_model:
                inputs = self.wav2vec_processor(audio_data, sampling_rate=self.sample_rate, return_tensors="pt")
                with torch.no_grad():
                    features = self.wav2vec_model.wav2vec2.feature_extractor(inputs.input_values)
                    features = features.last_hidden_state.mean(dim=1).squeeze().numpy()
                return features
            
            # Fallback to traditional features
            # MFCC features
            mfccs = librosa.feature.mfcc(y=audio_data, sr=self.sample_rate, n_mfcc=13)
            
            # Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=audio_data, sr=self.sample_rate)
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=self.sample_rate)
            zero_crossing_rate = librosa.feature.zero_crossing_rate(audio_data)
            
            # Combine features
            features = np.concatenate([
                np.mean(mfccs, axis=1),
                np.std(mfccs, axis=1),
                np.mean(spectral_centroids),
                np.mean(spectral_rolloff),
                np.mean(zero_crossing_rate)
            ])
            
            return features
            
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            return None
    
    def train_custom_model(self, positive_samples: List[np.ndarray], negative_samples: List[np.ndarray]):
        """Train custom wake word model"""
        if not self.custom_model:
            logger.error("Custom model not initialized")
            return
        
        logger.info(f"🎓 Training custom model with {len(positive_samples)} positive and {len(negative_samples)} negative samples")
        
        try:
            # Prepare training data
            X_train = []
            y_train = []
            
            # Process positive samples
            for sample in positive_samples:
                features = self._extract_audio_features(sample)
                if features is not None:
                    X_train.append(features)
                    y_train.append(1)  # Wake word class
            
            # Process negative samples
            for sample in negative_samples:
                features = self._extract_audio_features(sample)
                if features is not None:
                    X_train.append(features)
                    y_train.append(0)  # Non-wake word class
            
            if len(X_train) == 0:
                logger.error("No valid training samples")
                return
            
            # Convert to tensors
            X_train = torch.FloatTensor(np.array(X_train))
            y_train = torch.LongTensor(y_train)
            
            # Create data loader
            dataset = torch.utils.data.TensorDataset(X_train, y_train)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
            
            # Training setup
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(self.custom_model.parameters(), lr=0.001)
            
            # Training loop
            self.custom_model.train()
            num_epochs = 50
            
            for epoch in range(num_epochs):
                total_loss = 0
                for batch_X, batch_y in dataloader:
                    optimizer.zero_grad()
                    outputs = self.custom_model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                
                if (epoch + 1) % 10 == 0:
                    logger.info(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(dataloader):.4f}")
            
            # Save trained model
            model_path = self.config.get('custom_model_path', 'assets/models/wake_word_model.pth')
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            torch.save(self.custom_model.state_dict(), model_path)
            
            self.custom_model.eval()
            logger.info("✅ Custom model training completed")
            
        except Exception as e:
            logger.error(f"Model training failed: {e}")
    
    def add_wake_word(self, wake_word: str):
        """Add new wake word"""
        if wake_word not in self.wake_words:
            self.wake_words.append(wake_word)
            logger.info(f"➕ Added wake word: '{wake_word}'")
    
    def remove_wake_word(self, wake_word: str):
        """Remove wake word"""
        if wake_word in self.wake_words:
            self.wake_words.remove(wake_word)
            logger.info(f"➖ Removed wake word: '{wake_word}'")
    
    def set_sensitivity(self, sensitivity: float):
        """Set detection sensitivity"""
        self.sensitivity = max(0.0, min(1.0, sensitivity))
        
        # Update engine sensitivities
        if WakeWordEngine.PORCUPINE in self.engines:
            # Porcupine sensitivity is set during initialization
            pass
        
        if WakeWordEngine.SNOWBOY in self.engines:
            detector = self.engines[WakeWordEngine.SNOWBOY]
            detector.SetSensitivity(','.join([str(self.sensitivity)] * len(self.wake_words)))
        
        logger.info(f"🎛️ Sensitivity set to: {self.sensitivity}")
    
    def stop_detection(self):
        """Stop wake word detection"""
        self.is_listening = False
        logger.info("🛑 Wake word detection stopped")
    
    def _update_detection_stats(self, detection: WakeWordDetection, is_true_positive: bool):
        """Update detection statistics"""
        self.detection_stats['total_detections'] += 1
        
        if is_true_positive:
            self.detection_stats['true_positives'] += 1
        else:
            self.detection_stats['false_positives'] += 1
        
        # Update average confidence
        total = self.detection_stats['total_detections']
        current_avg = self.detection_stats['avg_confidence']
        self.detection_stats['avg_confidence'] = (current_avg * (total - 1) + detection.confidence) / total
        
        # Update engine performance
        engine_name = detection.engine.value
        if engine_name not in self.detection_stats['engine_performance']:
            self.detection_stats['engine_performance'][engine_name] = {
                'detections': 0,
                'avg_confidence': 0.0
            }
        
        engine_stats = self.detection_stats['engine_performance'][engine_name]
        engine_stats['detections'] += 1
        engine_avg = engine_stats['avg_confidence']
        engine_stats['avg_confidence'] = (engine_avg * (engine_stats['detections'] - 1) + detection.confidence) / engine_stats['detections']
    
    def get_detection_stats(self) -> Dict[str, Any]:
        """Get wake word detection statistics"""
        stats = self.detection_stats.copy()
        
        # Calculate accuracy
        total = stats['total_detections']
        if total > 0:
            stats['accuracy'] = stats['true_positives'] / total
            stats['false_positive_rate'] = stats['false_positives'] / total
        else:
            stats['accuracy'] = 0.0
            stats['false_positive_rate'] = 0.0
        
        return stats
    
    def cleanup(self):
        """Cleanup wake word detection resources"""
        logger.info("🧹 Cleaning up wake word detection...")
        
        self.stop_detection()
        
        # Cleanup Porcupine
        if WakeWordEngine.PORCUPINE in self.engines:
            try:
                self.engines[WakeWordEngine.PORCUPINE].delete()
            except:
                pass
        
        # Cleanup Snowboy
        if WakeWordEngine.SNOWBOY in self.engines:
            try:
                del self.engines[WakeWordEngine.SNOWBOY]
            except:
                pass
        
        logger.info("✅ Wake word detection cleanup complete")
