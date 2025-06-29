#!/usr/bin/env python3
"""
Shock2 Voice Interface Test Suite
Comprehensive testing for voice interface components
"""

import asyncio
import unittest
import numpy as np
import tempfile
import os
from unittest.mock import Mock, patch, AsyncMock

# Import voice interface components
from shock2_voice_interface import (
    AdvancedSpeechRecognition,
    AdvancedNLPProcessor,
    Shock2PersonalityEngine,
    AdvancedTextToSpeech,
    FaceAnimationEngine,
    Shock2VoiceInterface,
    VoiceCommand,
    VoiceInput
)

class TestAdvancedSpeechRecognition(unittest.TestCase):
    """Test speech recognition functionality"""
    
    def setUp(self):
        self.speech_recognition = AdvancedSpeechRecognition()
    
    def test_initialization(self):
        """Test speech recognition initialization"""
        self.assertIsNotNone(self.speech_recognition.recognizer)
        self.assertIsNotNone(self.speech_recognition.microphone)
        self.assertTrue(self.speech_recognition.calibrated)
    
    def test_vad_detection(self):
        """Test voice activity detection"""
        # Create test audio frame (silence)
        silence_frame = np.zeros(480, dtype=np.int16).tobytes()
        
        # Test VAD
        is_speech = self.speech_recognition._is_speech(silence_frame)
        self.assertFalse(is_speech)
    
    def test_noise_reduction(self):
        """Test noise reduction functionality"""
        # Create test audio with noise
        test_audio = np.random.randn(1000).astype(np.float32)
        
        # Apply noise reduction
        clean_audio = self.speech_recognition._reduce_noise(test_audio)
        
        self.assertEqual(len(clean_audio), len(test_audio))
        self.assertEqual(clean_audio.dtype, np.float32)

class TestAdvancedNLPProcessor(unittest.IsolatedAsyncioTestCase):
    """Test NLP processing functionality"""
    
    async def asyncSetUp(self):
        self.nlp_processor = AdvancedNLPProcessor()
        # Mock the initialization to avoid downloading models
        self.nlp_processor.nlp = Mock()
        self.nlp_processor.sentiment_analyzer = Mock()
        self.nlp_processor.intent_classifier = Mock()
    
    async def test_intent_classification(self):
        """Test intent classification"""
        # Test system control intent
        intent = await self.nlp_processor._classify_intent("show me system status")
        self.assertEqual(intent, VoiceCommand.SYSTEM_CONTROL)
        
        # Test news generation intent
        intent = await self.nlp_processor._classify_intent("generate news about technology")
        self.assertEqual(intent, VoiceCommand.NEWS_GENERATION)
        
        # Test stealth operations intent
        intent = await self.nlp_processor._classify_intent("activate stealth mode")
        self.assertEqual(intent, VoiceCommand.STEALTH_OPERATIONS)
    
    def test_command_parameter_extraction(self):
        """Test extraction of command parameters"""
        text = "generate 5 breaking news articles about artificial intelligence"
        params = self.nlp_processor._extract_command_parameters(text)
        
        self.assertEqual(params.get('topic'), 'artificial intelligence')
        self.assertEqual(params.get('article_type'), 'breaking')
        self.assertEqual(params.get('quantity'), 5)
    
    def test_urgency_calculation(self):
        """Test urgency level calculation"""
        # High urgency text
        high_urgency = self.nlp_processor._calculate_urgency(
            "emergency breaking news now", 
            VoiceCommand.EMERGENCY, 
            -0.5
        )
        self.assertGreater(high_urgency, 0.8)
        
        # Low urgency text
        low_urgency = self.nlp_processor._calculate_urgency(
            "show me some news", 
            VoiceCommand.NEWS_GENERATION, 
            0.0
        )
        self.assertLess(low_urgency, 0.7)

class TestShock2PersonalityEngine(unittest.TestCase):
    """Test personality engine functionality"""
    
    def setUp(self):
        self.personality_engine = Shock2PersonalityEngine()
    
    def test_response_generation(self):
        """Test response generation for different intents"""
        # Test system control response
        response = self.personality_engine.generate_response(
            VoiceCommand.SYSTEM_CONTROL,
            {'entities': {'keywords': ['status']}}
        )
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 10)
        
        # Test news generation response
        response = self.personality_engine.generate_response(
            VoiceCommand.NEWS_GENERATION,
            {'entities': {'topic': 'AI'}}
        )
        self.assertIsInstance(response, str)
        self.assertIn("AI", response)
    
    def test_contextual_modifiers(self):
        """Test contextual response modifiers"""
        # High urgency context
        response = self.personality_engine.generate_response(
            VoiceCommand.EMERGENCY,
            {'urgency': 0.9, 'entities': {}}
        )
        self.assertIn("Urgent", response)
        
        # Success context
        response = self.personality_engine.generate_response(
            VoiceCommand.SYSTEM_CONTROL,
            {'success': True, 'entities': {}}
        )
        self.assertIn("flawless", response)
    
    def test_personality_traits(self):
        """Test personality trait consistency"""
        traits = self.personality_engine.personality_traits
        
        # Check key personality traits
        self.assertGreater(traits['intelligence'], 0.9)
        self.assertGreater(traits['cunning'], 0.8)
        self.assertGreater(traits['efficiency'], 0.8)

class TestAdvancedTextToSpeech(unittest.TestCase):
    """Test text-to-speech functionality"""
    
    def setUp(self):
        self.tts_engine = AdvancedTextToSpeech()
    
    def test_initialization(self):
        """Test TTS initialization"""
        self.assertIsNotNone(self.tts_engine.engine)
        self.assertIsInstance(self.tts_engine.voice_config, dict)
        self.assertIsInstance(self.tts_engine.audio_effects, dict)
    
    def test_audio_effects(self):
        """Test audio effects processing"""
        # Create test audio
        test_audio = np.random.randn(1000).astype(np.float32)
        
        # Apply effects
        processed_audio = self.tts_engine._apply_audio_effects(test_audio)
        
        self.assertEqual(len(processed_audio), len(test_audio))
        self.assertEqual(processed_audio.dtype, np.float32)

class TestFaceAnimationEngine(unittest.TestCase):
    """Test face animation functionality"""
    
    def setUp(self):
        # Create temporary face image for testing
        self.temp_dir = tempfile.mkdtemp()
        self.face_image_path = os.path.join(self.temp_dir, "test_face.gif")
        
        # Create a simple test image
        import cv2
        test_frame = np.zeros((400, 400, 3), dtype=np.uint8)
        cv2.putText(test_frame, "TEST", (150, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        cv2.imwrite(self.face_image_path.replace('.gif', '.jpg'), test_frame)
        
        # Mock pygame to avoid display issues in testing
        with patch('pygame.init'), patch('pygame.display.set_mode'):
            self.face_animation = FaceAnimationEngine(self.face_image_path.replace('.gif', '.jpg'))
    
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """Test face animation initialization"""
        self.assertGreater(len(self.face_animation.face_frames), 0)
        self.assertIsInstance(self.face_animation.animation_config, dict)
    
    def test_phoneme_extraction(self):
        """Test phoneme extraction from text"""
        phonemes = self.face_animation._text_to_phonemes("Hello world")
        self.assertIsInstance(phonemes, list)
        self.assertGreater(len(phonemes), 0)
    
    def test_lip_sync_generation(self):
        """Test lip sync frame generation"""
        test_audio = np.random.randn(1000).astype(np.float32)
        test_text = "Hello Shock2"
        
        self.face_animation._generate_lip_sync_frames(test_audio, test_text)
        self.assertGreater(len(self.face_animation.lip_sync_frames), 0)

class TestShock2VoiceInterface(unittest.IsolatedAsyncioTestCase):
    """Test complete voice interface system"""
    
    async def asyncSetUp(self):
        # Mock all external dependencies
        with patch('shock2_voice_interface.load_config'), \
             patch('shock2_voice_interface.Shock2SystemManager'), \
             patch('shock2_voice_interface.CoreOrchestrator'), \
             patch('shock2_voice_interface.AutonomousController'), \
             patch('pygame.init'), \
             patch('pygame.display.set_mode'):
            
            self.voice_interface = Shock2VoiceInterface()
            
            # Mock the components
            self.voice_interface.nlp_processor = Mock()
            self.voice_interface.personality_engine = Mock()
            self.voice_interface.tts_engine = Mock()
            self.voice_interface.face_animation = Mock()
            
            # Mock system components
            self.voice_interface.system_manager = Mock()
            self.voice_interface.orchestrator = Mock()
            self.voice_interface.autonomous_controller = Mock()
    
    async def test_voice_input_handling(self):
        """Test voice input processing"""
        # Mock voice input
        mock_voice_input = VoiceInput(
            raw_audio=np.array([]),
            transcribed_text="show system status",
            confidence=0.9,
            intent=VoiceCommand.SYSTEM_CONTROL,
            entities={'keywords': ['status']},
            sentiment=0.0,
            urgency=0.5,
            timestamp=None
        )
        
        # Mock NLP processor
        self.voice_interface.nlp_processor.process_voice_input = AsyncMock(return_value=mock_voice_input)
        
        # Mock personality engine
        self.voice_interface.personality_engine.generate_response = Mock(return_value="System status: OPTIMAL")
        
        # Mock TTS
        self.voice_interface.tts_engine.speak = AsyncMock(return_value=np.array([]))
        
        # Mock face animation
        self.voice_interface.face_animation.speak_with_animation = AsyncMock()
        
        # Test voice input handling
        await self.voice_interface._handle_voice_input("show system status", 0.9, np.array([]))
        
        # Verify calls were made
        self.voice_interface.nlp_processor.process_voice_input.assert_called_once()
        self.voice_interface.personality_engine.generate_response.assert_called_once()
    
    async def test_command_execution(self):
        """Test command execution for different intents"""
        # Test system control command
        mock_voice_input = VoiceInput(
            raw_audio=np.array([]),
            transcribed_text="show system status",
            confidence=0.9,
            intent=VoiceCommand.SYSTEM_CONTROL,
            entities={'keywords': ['status']},
            sentiment=0.0,
            urgency=0.5,
            timestamp=None
        )
        
        # Mock system manager
        self.voice_interface.system_manager.get_system_status = Mock(return_value={'uptime': 3600, 'components': {}})
        self.voice_interface.orchestrator.get_orchestrator_status = Mock(return_value={'uptime': 3600})
        
        result = await self.voice_interface._execute_voice_command(mock_voice_input)
        
        self.assertTrue(result['success'])
        self.assertIn('data', result)
    
    def test_interface_statistics(self):
        """Test interface statistics tracking"""
        # Update stats
        self.voice_interface._update_interface_stats(True, 1.5, 0.9)
        self.voice_interface._update_interface_stats(False, 2.0, 0.7)
        
        stats = self.voice_interface.interface_stats
        
        self.assertEqual(stats['total_interactions'], 2)
        self.assertEqual(stats['successful_commands'], 1)
        self.assertEqual(stats['failed_commands'], 1)
        self.assertGreater(stats['avg_response_time'], 0)
        self.assertGreater(stats['voice_recognition_accuracy'], 0)

class TestIntegration(unittest.IsolatedAsyncioTestCase):
    """Integration tests for voice interface"""
    
    async def test_end_to_end_workflow(self):
        """Test complete end-to-end workflow"""
        # This would test the complete flow from voice input to response
        # In a real implementation, this would use actual audio files
        pass
    
    async def test_error_handling(self):
        """Test error handling in various scenarios"""
        # Test handling of speech recognition errors
        # Test handling of NLP processing errors
        # Test handling of system command errors
        pass
    
    async def test_performance_benchmarks(self):
        """Test performance benchmarks"""
        # Test response time requirements
        # Test memory usage
        # Test CPU usage
        pass

def run_tests():
    """Run all voice interface tests"""
    print("🧪 Running Shock2 Voice Interface Tests...")
    print("=" * 50)
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_classes = [
        TestAdvancedSpeechRecognition,
        TestAdvancedNLPProcessor,
        TestShock2PersonalityEngine,
        TestAdvancedTextToSpeech,
        TestFaceAnimationEngine,
        TestShock2VoiceInterface,
        TestIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print results
    print("\n" + "=" * 50)
    if result.wasSuccessful():
        print("✅ All tests passed!")
    else:
        print(f"❌ {len(result.failures)} test(s) failed")
        print(f"❌ {len(result.errors)} error(s) occurred")
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)
