"""
Shock2 Voice Integration Orchestrator
Central coordination system linking all voice interface components
"""

import asyncio
import logging
import threading
import time
import json
import os
import queue
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import uuid
import signal
import psutil
import gc

# Core voice components
from ..core.speech_engine import AdvancedSpeechEngine, SpeechResult
from ..core.nlu_engine import Shock2NLUEngine, Shock2Command
from ..core.command_dispatcher import CommandDispatcher, CommandResult
from ..core.tts_engine import Shock2TTSEngine, TTSResult
from ..animation.lipsync_engine import LipSyncEngine, LipSyncData
from ..animation.face_engine import Shock2FaceAnimationSystem, FaceExpression
from ..synthesis.voice_cloning_engine import Shock2VoiceCloner, SynthesisResult
from ..security.voice_auth_engine import Shock2VoiceAuthEngine, SecurityLevel, AuthenticationMethod
from ..personas.persona_manager import Shock2PersonaManager, PersonalityType, EmotionalState, InteractionContext
from ..wakeword.detection_engine import WakeWordDetectionEngine

# Shock2 core integration
from ...core.system_manager import Shock2SystemManager
from ...core.orchestrator import CoreOrchestrator
from ...core.autonomous_controller import AutonomousController
from ...config.settings import load_config

logger = logging.getLogger(__name__)

class VoiceSystemState(Enum):
    """Voice system operational states"""
    INITIALIZING = "initializing"
    READY = "ready"
    LISTENING = "listening"
    PROCESSING = "processing"
    SPEAKING = "speaking"
    AUTHENTICATING = "authenticating"
    ERROR = "error"
    SHUTDOWN = "shutdown"

class AudioStreamType(Enum):
    """Audio stream types"""
    INPUT = "input"
    OUTPUT = "output"
    PROCESSED = "processed"
    SYNTHESIZED = "synthesized"

@dataclass
class VoiceInteraction:
    """Complete voice interaction record"""
    interaction_id: str
    user_id: Optional[str]
    session_id: Optional[str]
    start_time: datetime
    end_time: Optional[datetime] = None
    
    # Audio data
    input_audio: Optional[np.ndarray] = None
    output_audio: Optional[np.ndarray] = None
    
    # Processing results
    speech_result: Optional[SpeechResult] = None
    nlu_result: Optional[Shock2Command] = None
    command_result: Optional[CommandResult] = None
    tts_result: Optional[TTSResult] = None
    
    # Authentication
    auth_success: bool = False
    security_level: SecurityLevel = SecurityLevel.PUBLIC
    threat_level: str = "NONE"
    
    # Personality
    personality_id: Optional[str] = None
    emotional_state: Optional[str] = None
    context: Optional[str] = None
    
    # Performance metrics
    total_latency: float = 0.0
    processing_stages: Dict[str, float] = field(default_factory=dict)
    
    # Status
    success: bool = False
    error_message: Optional[str] = None

@dataclass
class SystemPerformanceMetrics:
    """System-wide performance metrics"""
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    gpu_usage: float = 0.0
    gpu_memory: float = 0.0
    
    # Audio metrics
    audio_latency: float = 0.0
    audio_dropouts: int = 0
    sample_rate: int = 22050
    buffer_size: int = 1024
    
    # Component metrics
    speech_recognition_time: float = 0.0
    nlu_processing_time: float = 0.0
    command_execution_time: float = 0.0
    tts_synthesis_time: float = 0.0
    animation_render_time: float = 0.0
    
    # System metrics
    total_interactions: int = 0
    successful_interactions: int = 0
    failed_interactions: int = 0
    avg_response_time: float = 0.0
    
    # Real-time metrics
    frames_per_second: float = 0.0
    audio_quality_score: float = 0.0
    voice_clarity_score: float = 0.0

class VoiceStreamManager:
    """Manages real-time audio streaming and buffering"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.sample_rate = config.get('sample_rate', 22050)
        self.buffer_size = config.get('buffer_size', 1024)
        self.channels = config.get('channels', 1)
        
        # Audio streams
        self.input_stream = None
        self.output_stream = None
        
        # Buffers
        self.input_buffer = queue.Queue(maxsize=100)
        self.output_buffer = queue.Queue(maxsize=100)
        self.processing_buffer = queue.Queue(maxsize=50)
        
        # Stream state
        self.is_streaming = False
        self.stream_threads = []
        
        # Performance tracking
        self.stream_stats = {
            'input_frames': 0,
            'output_frames': 0,
            'buffer_overruns': 0,
            'buffer_underruns': 0,
            'latency_samples': []
        }
    
    async def start_streaming(self):
        """Start audio streaming"""
        if self.is_streaming:
            return
        
        try:
            import sounddevice as sd
            
            # Input stream callback
            def input_callback(indata, frames, time, status):
                if status:
                    logger.warning(f"Input stream status: {status}")
                    self.stream_stats['buffer_overruns'] += 1
                
                try:
                    audio_chunk = indata[:, 0].copy()
                    self.input_buffer.put_nowait((audio_chunk, time.inputBufferAdcTime))
                    self.stream_stats['input_frames'] += frames
                except queue.Full:
                    self.stream_stats['buffer_overruns'] += 1
            
            # Output stream callback
            def output_callback(outdata, frames, time, status):
                if status:
                    logger.warning(f"Output stream status: {status}")
                    self.stream_stats['buffer_underruns'] += 1
                
                try:
                    audio_chunk, timestamp = self.output_buffer.get_nowait()
                    
                    # Calculate latency
                    current_time = time.outputBufferDacTime
                    latency = current_time - timestamp
                    self.stream_stats['latency_samples'].append(latency)
                    
                    # Keep only recent latency samples
                    if len(self.stream_stats['latency_samples']) > 100:
                        self.stream_stats['latency_samples'] = self.stream_stats['latency_samples'][-100:]
                    
                    # Ensure correct shape
                    if len(audio_chunk) == frames:
                        outdata[:, 0] = audio_chunk
                    else:
                        # Pad or truncate as needed
                        if len(audio_chunk) < frames:
                            padded = np.zeros(frames)
                            padded[:len(audio_chunk)] = audio_chunk
                            outdata[:, 0] = padded
                        else:
                            outdata[:, 0] = audio_chunk[:frames]
                    
                    self.stream_stats['output_frames'] += frames
                    
                except queue.Empty:
                    # No audio to play, output silence
                    outdata.fill(0)
                    self.stream_stats['buffer_underruns'] += 1
            
            # Start streams
            self.input_stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                callback=input_callback,
                blocksize=self.buffer_size,
                dtype=np.float32
            )
            
            self.output_stream = sd.OutputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                callback=output_callback,
                blocksize=self.buffer_size,
                dtype=np.float32
            )
            
            self.input_stream.start()
            self.output_stream.start()
            
            self.is_streaming = True
            logger.info("🎵 Audio streaming started")
            
        except Exception as e:
            logger.error(f"Failed to start audio streaming: {e}")
            raise
    
    async def stop_streaming(self):
        """Stop audio streaming"""
        if not self.is_streaming:
            return
        
        try:
            if self.input_stream:
                self.input_stream.stop()
                self.input_stream.close()
            
            if self.output_stream:
                self.output_stream.stop()
                self.output_stream.close()
            
            self.is_streaming = False
            logger.info("🎵 Audio streaming stopped")
            
        except Exception as e:
            logger.error(f"Error stopping audio streaming: {e}")
    
    def get_input_audio(self, timeout: float = 0.1) -> Optional[Tuple[np.ndarray, float]]:
        """Get audio chunk from input buffer"""
        try:
            return self.input_buffer.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def put_output_audio(self, audio_data: np.ndarray, timestamp: Optional[float] = None):
        """Put audio chunk to output buffer"""
        if timestamp is None:
            timestamp = time.time()
        
        try:
            self.output_buffer.put_nowait((audio_data, timestamp))
        except queue.Full:
            # Remove oldest item and add new one
            try:
                self.output_buffer.get_nowait()
                self.output_buffer.put_nowait((audio_data, timestamp))
            except queue.Empty:
                pass
    
    def get_stream_stats(self) -> Dict[str, Any]:
        """Get streaming statistics"""
        stats = self.stream_stats.copy()
        
        # Calculate average latency
        if stats['latency_samples']:
            stats['avg_latency'] = np.mean(stats['latency_samples'])
            stats['max_latency'] = np.max(stats['latency_samples'])
            stats['min_latency'] = np.min(stats['latency_samples'])
        else:
            stats['avg_latency'] = 0.0
            stats['max_latency'] = 0.0
            stats['min_latency'] = 0.0
        
        # Remove raw samples from stats
        del stats['latency_samples']
        
        return stats

class Shock2VoiceOrchestrator:
    """Central orchestrator for all voice interface components"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.system_state = VoiceSystemState.INITIALIZING
        
        # Core Shock2 integration
        self.shock2_config = None
        self.system_manager = None
        self.core_orchestrator = None
        self.autonomous_controller = None
        
        # Voice components
        self.speech_engine = None
        self.nlu_engine = None
        self.command_dispatcher = None
        self.tts_engine = None
        self.lipsync_engine = None
        self.face_animation = None
        self.voice_cloner = None
        self.auth_engine = None
        self.persona_manager = None
        self.wakeword_detector = None
        
        # Stream management
        self.stream_manager = VoiceStreamManager(config.get('audio', {}))
        
        # Interaction management
        self.active_interactions: Dict[str, VoiceInteraction] = {}
        self.interaction_history: List[VoiceInteraction] = []
        self.current_session_id = None
        self.current_user_id = None
        
        # Performance monitoring
        self.performance_metrics = SystemPerformanceMetrics()
        self.performance_monitor_task = None
        
        # Processing pipeline
        self.processing_pipeline = asyncio.Queue(maxsize=10)
        self.pipeline_workers = []
        self.num_workers = config.get('pipeline_workers', 3)
        
        # Event system
        self.event_handlers: Dict[str, List[Callable]] = {}
        
        # System control
        self.is_running = False
        self.shutdown_event = asyncio.Event()
        
        # Error recovery
        self.error_recovery_enabled = config.get('error_recovery', True)
        self.max_consecutive_errors = config.get('max_consecutive_errors', 5)
        self.consecutive_errors = 0
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        asyncio.create_task(self.shutdown())
    
    async def initialize(self):
        """Initialize all voice interface components"""
        logger.info("🚀 Initializing Shock2 Voice Orchestrator...")
        
        try:
            self.system_state = VoiceSystemState.INITIALIZING
            
            # Load Shock2 configuration
            self.shock2_config = load_config()
            
            # Initialize core Shock2 systems
            await self._initialize_shock2_core()
            
            # Initialize voice components
            await self._initialize_voice_components()
            
            # Setup component integrations
            await self._setup_component_integrations()
            
            # Start processing pipeline
            await self._start_processing_pipeline()
            
            # Start performance monitoring
            await self._start_performance_monitoring()
            
            # Start audio streaming
            await self.stream_manager.start_streaming()
            
            # Start main processing loop
            await self._start_main_loop()
            
            self.system_state = VoiceSystemState.READY
            self.is_running = True
            
            logger.info("✅ Shock2 Voice Orchestrator fully initialized and operational")
            
            # Emit initialization complete event
            await self._emit_event('system_initialized', {
                'timestamp': datetime.now(),
                'components_loaded': self._get_component_status()
            })
            
        except Exception as e:
            self.system_state = VoiceSystemState.ERROR
            logger.error(f"❌ Failed to initialize Voice Orchestrator: {e}")
            raise
    
    async def _initialize_shock2_core(self):
        """Initialize core Shock2 systems"""
        logger.info("🧠 Initializing Shock2 core systems...")
        
        # Initialize system manager
        self.system_manager = Shock2SystemManager(self.shock2_config)
        await self.system_manager.initialize()
        
        # Initialize core orchestrator
        self.core_orchestrator = CoreOrchestrator(self.shock2_config)
        await self.core_orchestrator.initialize()
        
        # Initialize autonomous controller
        self.autonomous_controller = AutonomousController(self.shock2_config, self.core_orchestrator)
        await self.autonomous_controller.initialize()
        
        logger.info("✅ Shock2 core systems initialized")
    
    async def _initialize_voice_components(self):
        """Initialize all voice interface components"""
        logger.info("🎤 Initializing voice components...")
        
        # Initialize speech recognition
        speech_config = self.config.get('speech_recognition', {})
        self.speech_engine = AdvancedSpeechEngine(speech_config)
        
        # Initialize NLU
        nlu_config = self.config.get('nlu', {})
        self.nlu_engine = Shock2NLUEngine(nlu_config)
        await self.nlu_engine.initialize()
        
        # Initialize command dispatcher
        self.command_dispatcher = CommandDispatcher(
            self.system_manager,
            self.core_orchestrator,
            self.autonomous_controller
        )
        
        # Initialize TTS
        tts_config = self.config.get('tts', {})
        self.tts_engine = Shock2TTSEngine(tts_config)
        
        # Initialize lip sync
        lipsync_config = self.config.get('lipsync', {})
        self.lipsync_engine = LipSyncEngine(lipsync_config)
        
        # Initialize face animation
        face_config = self.config.get('face_animation', {})
        self.face_animation = Shock2FaceAnimationSystem(face_config)
        
        # Initialize voice cloning
        cloning_config = self.config.get('voice_cloning', {})
        self.voice_cloner = Shock2VoiceCloner(cloning_config)
        
        # Initialize authentication
        auth_config = self.config.get('authentication', {})
        self.auth_engine = Shock2VoiceAuthEngine(auth_config)
        
        # Initialize persona manager
        persona_config = self.config.get('persona_management', {})
        self.persona_manager = Shock2PersonaManager(persona_config)
        
        # Initialize wake word detection
        wakeword_config = self.config.get('wake_word', {})
        self.wakeword_detector = WakeWordDetectionEngine(wakeword_config)
        
        logger.info("✅ Voice components initialized")
    
    async def _setup_component_integrations(self):
        """Setup integrations between components"""
        logger.info("🔗 Setting up component integrations...")
        
        # Connect persona manager to voice systems
        self.persona_manager.set_voice_cloner(self.voice_cloner)
        self.persona_manager.set_tts_engine(self.tts_engine)
        self.persona_manager.set_face_animation(self.face_animation)
        
        # Setup event handlers
        self._setup_event_handlers()
        
        logger.info("✅ Component integrations configured")
    
    def _setup_event_handlers(self):
        """Setup event handlers for component communication"""
        
        # Authentication events
        self.register_event_handler('authentication_success', self._handle_auth_success)
        self.register_event_handler('authentication_failure', self._handle_auth_failure)
        self.register_event_handler('security_threat_detected', self._handle_security_threat)
        
        # Persona events
        self.register_event_handler('personality_switched', self._handle_personality_switch)
        self.register_event_handler('emotional_state_changed', self._handle_emotion_change)
        
        # System events
        self.register_event_handler('wake_word_detected', self._handle_wake_word)
        self.register_event_handler('voice_command_received', self._handle_voice_command)
        self.register_event_handler('system_error', self._handle_system_error)
    
    async def _start_processing_pipeline(self):
        """Start the voice processing pipeline"""
        logger.info("⚙️ Starting voice processing pipeline...")
        
        # Start pipeline workers
        for i in range(self.num_workers):
            worker = asyncio.create_task(self._pipeline_worker(f"worker_{i}"))
            self.pipeline_workers.append(worker)
        
        logger.info(f"✅ Started {self.num_workers} pipeline workers")
    
    async def _start_performance_monitoring(self):
        """Start performance monitoring"""
        logger.info("📊 Starting performance monitoring...")
        
        self.performance_monitor_task = asyncio.create_task(self._performance_monitor_loop())
        
        logger.info("✅ Performance monitoring started")
    
    async def _start_main_loop(self):
        """Start main processing loop"""
        logger.info("🔄 Starting main processing loop...")
        
        # Start wake word detection
        asyncio.create_task(self._wake_word_loop())
        
        # Start continuous audio processing
        asyncio.create_task(self._audio_processing_loop())
        
        logger.info("✅ Main processing loop started")
    
    async def _wake_word_loop(self):
        """Continuous wake word detection loop"""
        while self.is_running:
            try:
                # Get audio from stream
                audio_data = self.stream_manager.get_input_audio(timeout=0.1)
                if audio_data is None:
                    continue
                
                audio_chunk, timestamp = audio_data
                
                # Detect wake word
                if self.wakeword_detector:
                    wake_detected = await self.wakeword_detector.detect_wake_word(audio_chunk)
                    
                    if wake_detected:
                        await self._emit_event('wake_word_detected', {
                            'timestamp': timestamp,
                            'audio_chunk': audio_chunk
                        })
                
            except Exception as e:
                logger.error(f"Wake word detection error: {e}")
                await asyncio.sleep(0.1)
    
    async def _audio_processing_loop(self):
        """Continuous audio processing loop"""
        audio_buffer = []
        buffer_duration = 0.0
        max_buffer_duration = 3.0  # seconds
        
        while self.is_running:
            try:
                # Get audio from stream
                audio_data = self.stream_manager.get_input_audio(timeout=0.1)
                if audio_data is None:
                    continue
                
                audio_chunk, timestamp = audio_data
                
                # Add to buffer
                audio_buffer.append(audio_chunk)
                buffer_duration += len(audio_chunk) / self.stream_manager.sample_rate
                
                # Process buffer when it reaches max duration or on silence
                if buffer_duration >= max_buffer_duration:
                    if len(audio_buffer) > 0:
                        # Combine audio chunks
                        combined_audio = np.concatenate(audio_buffer)
                        
                        # Create interaction for processing
                        interaction = VoiceInteraction(
                            interaction_id=str(uuid.uuid4()),
                            user_id=self.current_user_id,
                            session_id=self.current_session_id,
                            start_time=datetime.now(),
                            input_audio=combined_audio
                        )
                        
                        # Add to processing pipeline
                        try:
                            await self.processing_pipeline.put(interaction)
                        except asyncio.QueueFull:
                            logger.warning("Processing pipeline full, dropping audio")
                    
                    # Reset buffer
                    audio_buffer = []
                    buffer_duration = 0.0
                
            except Exception as e:
                logger.error(f"Audio processing loop error: {e}")
                await asyncio.sleep(0.1)
    
    async def _pipeline_worker(self, worker_id: str):
        """Pipeline worker for processing voice interactions"""
        logger.info(f"👷 Pipeline worker {worker_id} started")
        
        while self.is_running:
            try:
                # Get interaction from pipeline
                interaction = await asyncio.wait_for(
                    self.processing_pipeline.get(),
                    timeout=1.0
                )
                
                # Process the interaction
                await self._process_voice_interaction(interaction)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Pipeline worker {worker_id} error: {e}")
                self.consecutive_errors += 1
                
                if self.consecutive_errors >= self.max_consecutive_errors:
                    logger.error("Too many consecutive errors, initiating error recovery")
                    await self._handle_error_recovery()
                
                await asyncio.sleep(1.0)
        
        logger.info(f"👷 Pipeline worker {worker_id} stopped")
    
    async def _process_voice_interaction(self, interaction: VoiceInteraction):
        """Process a complete voice interaction"""
        try:
            self.active_interactions[interaction.interaction_id] = interaction
            stage_start = time.time()
            
            # Stage 1: Speech Recognition
            if interaction.input_audio is not None:
                speech_result = await self.speech_engine.recognize_speech(interaction.input_audio)
                interaction.speech_result = speech_result
                interaction.processing_stages['speech_recognition'] = time.time() - stage_start
                
                if not speech_result or not speech_result.text.strip():
                    interaction.success = False
                    interaction.error_message = "No speech recognized"
                    return
                
                await self._emit_event('speech_recognized', {
                    'interaction_id': interaction.interaction_id,
                    'text': speech_result.text,
                    'confidence': speech_result.confidence
                })
            
            # Stage 2: Authentication (if required)
            stage_start = time.time()
            if self.auth_engine and interaction.input_audio is not None:
                auth_success, user_id, confidence, auth_info = await self.auth_engine.authenticate_user(
                    interaction.input_audio,
                    claimed_user_id=interaction.user_id
                )
                
                interaction.auth_success = auth_success
                interaction.user_id = user_id
                interaction.threat_level = auth_info.get('threat_level', 'NONE')
                interaction.processing_stages['authentication'] = time.time() - stage_start
                
                if not auth_success and self.config.get('require_authentication', False):
                    interaction.success = False
                    interaction.error_message = "Authentication failed"
                    return
                
                # Create session if authenticated
                if auth_success and user_id:
                    session_id = await self.auth_engine.create_session(
                        user_id, SecurityLevel.BASIC, AuthenticationMethod.VOICE_BIOMETRIC
                    )
                    interaction.session_id = session_id
                    self.current_session_id = session_id
                    self.current_user_id = user_id
            
            # Stage 3: NLU Processing
            stage_start = time.time()
            if interaction.speech_result:
                nlu_command = await self.nlu_engine.parse_command(interaction.speech_result.text)
                interaction.nlu_result = nlu_command
                interaction.processing_stages['nlu'] = time.time() - stage_start
                
                # Adapt persona based on command context
                if self.persona_manager and nlu_command.intent:
                    context_mapping = {
                        'system_status': InteractionContext.SYSTEM_CONTROL,
                        'generate_breaking': InteractionContext.CONTENT_GENERATION,
                        'intel_scan': InteractionContext.INTELLIGENCE_BRIEFING,
                        'emergency_response': InteractionContext.EMERGENCY_RESPONSE
                    }
                    
                    context = context_mapping.get(nlu_command.intent.value, InteractionContext.CASUAL_CONVERSATION)
                    await self.persona_manager.adapt_to_context(context)
                    
                    interaction.context = context.value
                    interaction.personality_id = self.persona_manager.get_current_personality().personality_id
                    interaction.emotional_state = self.persona_manager.get_current_state().emotional_state.value
            
            # Stage 4: Command Execution
            stage_start = time.time()
            if interaction.nlu_result:
                command_result = await self.command_dispatcher.dispatch_command(interaction.nlu_result)
                interaction.command_result = command_result
                interaction.processing_stages['command_execution'] = time.time() - stage_start
                
                if not command_result.success:
                    interaction.success = False
                    interaction.error_message = command_result.error
                    return
            
            # Stage 5: Response Generation
            stage_start = time.time()
            if interaction.command_result and self.persona_manager:
                # Generate personality-appropriate response
                response_text = await self.persona_manager.generate_response(
                    interaction.speech_result.text if interaction.speech_result else "",
                    response_type='task_completion'
                )
                
                # Add command result information
                if interaction.command_result.data:
                    result_summary = self._format_command_result(interaction.command_result.data)
                    response_text += f" {result_summary}"
                
                interaction.processing_stages['response_generation'] = time.time() - stage_start
            else:
                response_text = "Command processed successfully."
            
            # Stage 6: Speech Synthesis
            stage_start = time.time()
            if self.voice_cloner and interaction.personality_id:
                # Use voice cloning for personality-specific voice
                synthesis_result = await self.voice_cloner.synthesize_speech(
                    response_text,
                    interaction.personality_id
                )
                interaction.output_audio = synthesis_result.audio_data
            elif self.tts_engine:
                # Use standard TTS
                tts_result = await self.tts_engine.speak(response_text, play_immediately=False)
                interaction.tts_result = tts_result
                interaction.output_audio = tts_result.audio_data
            
            interaction.processing_stages['speech_synthesis'] = time.time() - stage_start
            
            # Stage 7: Lip Sync and Animation
            stage_start = time.time()
            if interaction.output_audio is not None and self.lipsync_engine and self.face_animation:
                # Generate lip sync data
                lipsync_data = await self.lipsync_engine.generate_lipsync_data(
                    interaction.output_audio,
                    self.stream_manager.sample_rate,
                    response_text
                )
                
                # Start face animation
                await self.face_animation.speak_with_animation(
                    response_text,
                    interaction.output_audio,
                    self.stream_manager.sample_rate,
                    self.persona_manager.get_current_personality().default_expression
                )
            
            interaction.processing_stages['animation'] = time.time() - stage_start
            
            # Stage 8: Audio Output
            if interaction.output_audio is not None:
                self.stream_manager.put_output_audio(interaction.output_audio)
            
            # Mark interaction as successful
            interaction.success = True
            interaction.end_time = datetime.now()
            interaction.total_latency = (interaction.end_time - interaction.start_time).total_seconds()
            
            # Reset consecutive errors on success
            self.consecutive_errors = 0
            
            await self._emit_event('interaction_completed', {
                'interaction_id': interaction.interaction_id,
                'success': interaction.success,
                'latency': interaction.total_latency
            })
            
        except Exception as e:
            interaction.success = False
            interaction.error_message = str(e)
            interaction.end_time = datetime.now()
            
            logger.error(f"Voice interaction processing failed: {e}")
            
            await self._emit_event('interaction_failed', {
                'interaction_id': interaction.interaction_id,
                'error': str(e)
            })
        
        finally:
            # Move to history and cleanup
            self.interaction_history.append(interaction)
            if interaction.interaction_id in self.active_interactions:
                del self.active_interactions[interaction.interaction_id]
            
            # Keep only recent history
            if len(self.interaction_history) > 1000:
                self.interaction_history = self.interaction_history[-1000:]
    
    def _format_command_result(self, result_data: Dict[str, Any]) -> str:
        """Format command result for speech output"""
        if 'articles_generated' in result_data:
            return f"Generated {result_data['articles_generated']} articles successfully."
        elif 'sources_scanned' in result_data:
            return f"Scanned {result_data['sources_scanned']} sources and found {result_data.get('articles_found', 0)} articles."
        elif 'system_status' in result_data:
            uptime = result_data['system_status'].get('uptime', 0)
            return f"System operational for {uptime/3600:.1f} hours."
        elif 'stealth_mode' in result_data:
            return f"Stealth mode {result_data['stealth_mode']}."
        else:
            return "Operation completed."
    
    async def _performance_monitor_loop(self):
        """Performance monitoring loop"""
        while self.is_running:
            try:
                # Update system metrics
                await self._update_performance_metrics()
                
                # Check for performance issues
                await self._check_performance_thresholds()
                
                # Cleanup memory if needed
                if self.performance_metrics.memory_usage > 0.8:
                    await self._cleanup_memory()
                
                await asyncio.sleep(5.0)  # Update every 5 seconds
                
            except Exception as e:
                logger.error(f"Performance monitoring error: {e}")
                await asyncio.sleep(10.0)
    
    async def _update_performance_metrics(self):
        """Update system performance metrics"""
        try:
            # CPU and memory usage
            self.performance_metrics.cpu_usage = psutil.cpu_percent()
            self.performance_metrics.memory_usage = psutil.virtual_memory().percent / 100.0
            
            # GPU metrics (if available)
            try:
                import GPUtil
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]
                    self.performance_metrics.gpu_usage = gpu.load
                    self.performance_metrics.gpu_memory = gpu.memoryUtil
            except ImportError:
                pass
            
            # Audio streaming metrics
            stream_stats = self.stream_manager.get_stream_stats()
            self.performance_metrics.audio_latency = stream_stats.get('avg_latency', 0.0)
            self.performance_metrics.audio_dropouts = stream_stats.get('buffer_overruns', 0) + stream_stats.get('buffer_underruns', 0)
            
            # Interaction metrics
            if self.interaction_history:
                recent_interactions = [i for i in self.interaction_history[-100:] if i.end_time]
                if recent_interactions:
                    successful = [i for i in recent_interactions if i.success]
                    
                    self.performance_metrics.successful_interactions = len(successful)
                    self.performance_metrics.failed_interactions = len(recent_interactions) - len(successful)
                    
                    if successful:
                        avg_latency = np.mean([i.total_latency for i in successful])
                        self.performance_metrics.avg_response_time = avg_latency
                        
                        # Component-specific metrics
                        if any('speech_recognition' in i.processing_stages for i in successful):
                            speech_times = [i.processing_stages['speech_recognition'] for i in successful if 'speech_recognition' in i.processing_stages]
                            self.performance_metrics.speech_recognition_time = np.mean(speech_times)
                        
                        if any('nlu' in i.processing_stages for i in successful):
                            nlu_times = [i.processing_stages['nlu'] for i in successful if 'nlu' in i.processing_stages]
                            self.performance_metrics.nlu_processing_time = np.mean(nlu_times)
                        
                        if any('command_execution' in i.processing_stages for i in successful):
                            cmd_times = [i.processing_stages['command_execution'] for i in successful if 'command_execution' in i.processing_stages]
                            self.performance_metrics.command_execution_time = np.mean(cmd_times)
                        
                        if any('speech_synthesis' in i.processing_stages for i in successful):
                            tts_times = [i.processing_stages['speech_synthesis'] for i in successful if 'speech_synthesis' in i.processing_stages]
                            self.performance_metrics.tts_synthesis_time = np.mean(tts_times)
            
        except Exception as e:
            logger.error(f"Failed to update performance metrics: {e}")
    
    async def _check_performance_thresholds(self):
        """Check performance thresholds and emit warnings"""
        metrics = self.performance_metrics
        
        # CPU usage warning
        if metrics.cpu_usage > 80:
            await self._emit_event('performance_warning', {
                'type': 'high_cpu_usage',
                'value': metrics.cpu_usage,
                'threshold': 80
            })
        
        # Memory usage warning
        if metrics.memory_usage > 0.85:
            await self._emit_event('performance_warning', {
                'type': 'high_memory_usage',
                'value': metrics.memory_usage,
                'threshold': 0.85
            })
        
        # Audio latency warning
        if metrics.audio_latency > 0.1:  # 100ms
            await self._emit_event('performance_warning', {
                'type': 'high_audio_latency',
                'value': metrics.audio_latency,
                'threshold': 0.1
            })
        
        # Response time warning
        if metrics.avg_response_time > 3.0:  # 3 seconds
            await self._emit_event('performance_warning', {
                'type': 'slow_response_time',
                'value': metrics.avg_response_time,
                'threshold': 3.0
            })
    
    async def _cleanup_memory(self):
        """Cleanup memory to improve performance"""
        logger.info("🧹 Performing memory cleanup...")
        
        try:
            # Cleanup old interactions
            if len(self.interaction_history) > 500:
                self.interaction_history = self.interaction_history[-500:]
            
            # Force garbage collection
            gc.collect()
            
            # Clear GPU cache if available
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
            
            logger.info("✅ Memory cleanup completed")
            
        except Exception as e:
            logger.error(f"Memory cleanup failed: {e}")
    
    async def _handle_error_recovery(self):
        """Handle error recovery procedures"""
        logger.info("🔧 Initiating error recovery...")
        
        try:
            # Reset consecutive error counter
            self.consecutive_errors = 0
            
            # Restart failed components
            if self.speech_engine:
                logger.info("Restarting speech engine...")
                # Reinitialize speech engine
                speech_config = self.config.get('speech_recognition', {})
                self.speech_engine = AdvancedSpeechEngine(speech_config)
            
            # Clear processing pipeline
            while not self.processing_pipeline.empty():
                try:
                    self.processing_pipeline.get_nowait()
                except asyncio.QueueEmpty:
                    break
            
            # Restart audio streaming if needed
            if not self.stream_manager.is_streaming:
                await self.stream_manager.start_streaming()
            
            logger.info("✅ Error recovery completed")
            
        except Exception as e:
            logger.error(f"Error recovery failed: {e}")
    
    # Event system methods
    def register_event_handler(self, event_type: str, handler: Callable):
        """Register event handler"""
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        self.event_handlers[event_type].append(handler)
    
    async def _emit_event(self, event_type: str, data: Dict[str, Any]):
        """Emit event to registered handlers"""
        if event_type in self.event_handlers:
            for handler in self.event_handlers[event_type]:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(data)
                    else:
                        handler(data)
                except Exception as e:
                    logger.error(f"Event handler error for {event_type}: {e}")
    
    # Event handlers
    async def _handle_auth_success(self, data: Dict[str, Any]):
        """Handle successful authentication"""
        logger.info(f"🔐 Authentication successful for user: {data.get('user_id')}")
    
    async def _handle_auth_failure(self, data: Dict[str, Any]):
        """Handle authentication failure"""
        logger.warning(f"🚫 Authentication failed: {data.get('reason')}")
    
    async def _handle_security_threat(self, data: Dict[str, Any]):
        """Handle security threat detection"""
        threat_level = data.get('threat_level', 'UNKNOWN')
        logger.warning(f"🚨 Security threat detected: {threat_level}")
        
        # Switch to security-focused personality
        if self.persona_manager and threat_level in ['HIGH', 'CRITICAL']:
            await self.persona_manager.switch_personality('menacing_calm')
            await self.persona_manager.set_emotional_state(EmotionalState.MENACING)
    
    async def _handle_personality_switch(self, data: Dict[str, Any]):
        """Handle personality switch"""
        personality_name = data.get('personality_name', 'Unknown')
        logger.info(f"🎭 Personality switched to: {personality_name}")
    
    async def _handle_emotion_change(self, data: Dict[str, Any]):
        """Handle emotional state change"""
        emotion = data.get('emotion', 'neutral')
        logger.info(f"😈 Emotional state changed to: {emotion}")
    
    async def _handle_wake_word(self, data: Dict[str, Any]):
        """Handle wake word detection"""
        logger.info("👂 Wake word detected - activating voice interface")
        self.system_state = VoiceSystemState.LISTENING
    
    async def _handle_voice_command(self, data: Dict[str, Any]):
        """Handle voice command received"""
        command = data.get('command', 'Unknown')
        logger.info(f"🗣️ Voice command received: {command}")
    
    async def _handle_system_error(self, data: Dict[str, Any]):
        """Handle system error"""
        error = data.get('error', 'Unknown error')
        logger.error(f"⚠️ System error: {error}")
        
        if self.error_recovery_enabled:
            await self._handle_error_recovery()
    
    # Public API methods
    async def process_voice_input(self, audio_data: np.ndarray, user_id: Optional[str] = None) -> VoiceInteraction:
        """Process voice input directly (for external integrations)"""
        interaction = VoiceInteraction(
            interaction_id=str(uuid.uuid4()),
            user_id=user_id,
            session_id=self.current_session_id,
            start_time=datetime.now(),
            input_audio=audio_data
        )
        
        await self._process_voice_interaction(interaction)
        return interaction
    
    async def synthesize_and_speak(self, text: str, personality_id: Optional[str] = None) -> bool:
        """Synthesize and speak text"""
        try:
            if personality_id and personality_id != self.persona_manager.get_current_personality().personality_id:
                await self.persona_manager.switch_personality(personality_id)
            
            # Generate response with current personality
            if self.persona_manager:
                response_text = await self.persona_manager.generate_response(text, 'general')
            else:
                response_text = text
            
            # Synthesize speech
            if self.voice_cloner and self.persona_manager:
                current_personality = self.persona_manager.get_current_personality()
                synthesis_result = await self.voice_cloner.synthesize_speech(
                    response_text,
                    current_personality.personality_id
                )
                audio_data = synthesis_result.audio_data
            elif self.tts_engine:
                tts_result = await self.tts_engine.speak(response_text, play_immediately=False)
                audio_data = tts_result.audio_data
            else:
                return False
            
            # Play audio
            self.stream_manager.put_output_audio(audio_data)
            
            # Animate face if available
            if self.face_animation:
                await self.face_animation.speak_with_animation(
                    response_text,
                    audio_data,
                    self.stream_manager.sample_rate
                )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to synthesize and speak: {e}")
            return False
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return {
            'system_state': self.system_state.value,
            'is_running': self.is_running,
            'active_interactions': len(self.active_interactions),
            'total_interactions': len(self.interaction_history),
            'current_user': self.current_user_id,
            'current_session': self.current_session_id,
            'performance_metrics': {
                'cpu_usage': self.performance_metrics.cpu_usage,
                'memory_usage': self.performance_metrics.memory_usage,
                'audio_latency': self.performance_metrics.audio_latency,
                'avg_response_time': self.performance_metrics.avg_response_time,
                'success_rate': self._calculate_success_rate()
            },
            'component_status': self._get_component_status(),
            'stream_stats': self.stream_manager.get_stream_stats()
        }
    
    def _calculate_success_rate(self) -> float:
        """Calculate interaction success rate"""
        if not self.interaction_history:
            return 0.0
        
        recent_interactions = self.interaction_history[-100:]  # Last 100 interactions
        successful = sum(1 for i in recent_interactions if i.success)
        
        return successful / len(recent_interactions)
    
    def _get_component_status(self) -> Dict[str, bool]:
        """Get status of all components"""
        return {
            'speech_engine': self.speech_engine is not None,
            'nlu_engine': self.nlu_engine is not None,
            'command_dispatcher': self.command_dispatcher is not None,
            'tts_engine': self.tts_engine is not None,
            'lipsync_engine': self.lipsync_engine is not None,
            'face_animation': self.face_animation is not None,
            'voice_cloner': self.voice_cloner is not None,
            'auth_engine': self.auth_engine is not None,
            'persona_manager': self.persona_manager is not None,
            'wakeword_detector': self.wakeword_detector is not None,
            'stream_manager': self.stream_manager.is_streaming,
            'shock2_core': self.system_manager is not None
        }
    
    async def shutdown(self):
        """Graceful shutdown of the voice orchestrator"""
        logger.info("🛑 Shutting down Shock2 Voice Orchestrator...")
        
        self.is_running = False
        self.system_state = VoiceSystemState.SHUTDOWN
        
        try:
            # Stop audio streaming
            await self.stream_manager.stop_streaming()
            
            # Stop pipeline workers
            for worker in self.pipeline_workers:
                worker.cancel()
            
            # Stop performance monitoring
            if self.performance_monitor_task:
                self.performance_monitor_task.cancel()
            
            # Shutdown voice components
            if self.face_animation:
                self.face_animation.cleanup()
            
            if self.voice_cloner:
                self.voice_cloner.cleanup()
            
            if self.auth_engine:
                self.auth_engine.cleanup()
            
            if self.persona_manager:
                self.persona_manager.cleanup()
            
            if self.tts_engine:
                self.tts_engine.shutdown()
            
            # Shutdown Shock2 core
            if self.autonomous_controller:
                await self.autonomous_controller.shutdown()
            
            if self.core_orchestrator:
                await self.core_orchestrator.shutdown()
            
            if self.system_manager:
                await self.system_manager._shutdown()
            
            # Set shutdown event
            self.shutdown_event.set()
            
            logger.info("✅ Shock2 Voice Orchestrator shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
    
    async def wait_for_shutdown(self):
        """Wait for shutdown to complete"""
        await self.shutdown_event.wait()

# Main entry point
async def main():
    """Main entry point for Shock2 Voice Orchestrator"""
    
    # Load configuration
    config_path = os.getenv('SHOCK2_VOICE_CONFIG', 'config/voice_config.yaml')
    
    if os.path.exists(config_path):
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        # Default configuration
        config = {
            'audio': {
                'sample_rate': 22050,
                'buffer_size': 1024,
                'channels': 1
            },
            'speech_recognition': {
                'whisper_model': 'base',
                'sample_rate': 16000
            },
            'nlu': {
                'spacy_model': 'en_core_web_sm'
            },
            'tts': {
                'sample_rate': 22050
            },
            'voice_cloning': {
                'profiles_dir': 'data/voice_profiles'
            },
            'authentication': {
                'voice_prints_dir': 'data/security/voice_prints',
                'require_authentication': False
            },
            'persona_management': {
                'personalities_dir': 'data/personalities',
                'default_personality': 'villainous_mastermind'
            },
            'pipeline_workers': 3,
            'error_recovery': True,
            'max_consecutive_errors': 5
        }
    
    # Create and initialize orchestrator
    orchestrator = Shock2VoiceOrchestrator(config)
    
    try:
        # Initialize system
        await orchestrator.initialize()
        
        # Print startup message
        print("🤖 Shock2 Voice Interface is now operational!")
        print("🎤 Listening for voice commands...")
        print("🎭 AI personalities ready for interaction")
        print("🔐 Security systems active")
        print("\nPress Ctrl+C to shutdown\n")
        
        # Wait for shutdown
        await orchestrator.wait_for_shutdown()
        
    except KeyboardInterrupt:
        logger.info("🛑 Shutdown signal received")
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
    finally:
        await orchestrator.shutdown()

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run the orchestrator
    asyncio.run(main())
