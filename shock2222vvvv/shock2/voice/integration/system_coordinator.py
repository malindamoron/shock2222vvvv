"""
Shock2 System Coordinator
High-level coordination between voice interface and core Shock2 systems
"""

import asyncio
import logging
import json
import time
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta

from .voice_orchestrator import Shock2VoiceOrchestrator, VoiceInteraction, VoiceSystemState
from ...core.system_manager import Shock2SystemManager
from ...core.orchestrator import CoreOrchestrator, Task, TaskPriority
from ...core.autonomous_controller import AutonomousController, DecisionType
from ...config.settings import Shock2Config

logger = logging.getLogger(__name__)

class SystemMode(Enum):
    """System operational modes"""
    AUTONOMOUS = "autonomous"
    VOICE_CONTROLLED = "voice_controlled"
    HYBRID = "hybrid"
    MAINTENANCE = "maintenance"
    EMERGENCY = "emergency"

class CoordinationLevel(Enum):
    """Coordination levels between systems"""
    MINIMAL = "minimal"
    STANDARD = "standard"
    FULL_INTEGRATION = "full_integration"
    DEEP_SYNC = "deep_sync"

@dataclass
class SystemCommand:
    """System-level command"""
    command_id: str
    source: str  # 'voice', 'autonomous', 'external'
    target_system: str
    command_type: str
    parameters: Dict[str, Any]
    priority: TaskPriority
    timestamp: datetime
    requires_confirmation: bool = False
    estimated_duration: float = 0.0

@dataclass
class CoordinationMetrics:
    """Coordination performance metrics"""
    total_commands: int = 0
    voice_commands: int = 0
    autonomous_commands: int = 0
    successful_coordinations: int = 0
    failed_coordinations: int = 0
    avg_coordination_time: float = 0.0
    system_sync_score: float = 1.0
    conflict_resolutions: int = 0

class Shock2SystemCoordinator:
    """Coordinates voice interface with core Shock2 systems"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # System components
        self.voice_orchestrator: Optional[Shock2VoiceOrchestrator] = None
        self.system_manager: Optional[Shock2SystemManager] = None
        self.core_orchestrator: Optional[CoreOrchestrator] = None
        self.autonomous_controller: Optional[AutonomousController] = None
        
        # Coordination state
        self.system_mode = SystemMode.HYBRID
        self.coordination_level = CoordinationLevel.FULL_INTEGRATION
        self.is_coordinating = False
        
        # Command management
        self.pending_commands: Dict[str, SystemCommand] = {}
        self.command_history: List[SystemCommand] = []
        self.command_queue = asyncio.Queue(maxsize=100)
        
        # Conflict resolution
        self.conflict_resolver = ConflictResolver(config.get('conflict_resolution', {}))
        
        # Performance tracking
        self.coordination_metrics = CoordinationMetrics()
        
        # Event handlers
        self.event_handlers: Dict[str, List[Callable]] = {}
        
        # Coordination rules
        self.coordination_rules = self._load_coordination_rules()
        
        # System synchronization
        self.sync_interval = config.get('sync_interval', 5.0)  # seconds
        self.last_sync_time = time.time()
    
    def _load_coordination_rules(self) -> Dict[str, Any]:
        """Load coordination rules"""
        return {
            'voice_priority_commands': [
                'emergency_response',
                'system_shutdown',
                'security_alert'
            ],
            'autonomous_priority_commands': [
                'routine_maintenance',
                'performance_optimization',
                'background_processing'
            ],
            'require_confirmation': [
                'system_shutdown',
                'delete_data',
                'security_changes'
            ],
            'coordination_timeouts': {
                'voice_command': 30.0,
                'autonomous_command': 300.0,
                'system_sync': 10.0
            }
        }
    
    async def initialize(self, 
                        voice_orchestrator: Shock2VoiceOrchestrator,
                        system_manager: Shock2SystemManager,
                        core_orchestrator: CoreOrchestrator,
                        autonomous_controller: AutonomousController):
        """Initialize system coordinator"""
        logger.info("🔗 Initializing Shock2 System Coordinator...")
        
        # Store component references
        self.voice_orchestrator = voice_orchestrator
        self.system_manager = system_manager
        self.core_orchestrator = core_orchestrator
        self.autonomous_controller = autonomous_controller
        
        # Setup event handlers
        self._setup_event_handlers()
        
        # Start coordination processes
        await self._start_coordination_processes()
        
        self.is_coordinating = True
        
        logger.info("✅ System Coordinator initialized and active")
    
    def _setup_event_handlers(self):
        """Setup event handlers for system coordination"""
        
        # Voice orchestrator events
        if self.voice_orchestrator:
            self.voice_orchestrator.register_event_handler(
                'interaction_completed', self._handle_voice_interaction_completed
            )
            self.voice_orchestrator.register_event_handler(
                'system_command_received', self._handle_voice_system_command
            )
            self.voice_orchestrator.register_event_handler(
                'emergency_detected', self._handle_emergency_from_voice
            )
        
        # Core orchestrator events
        if self.core_orchestrator:
            self.core_orchestrator.event_bus.subscribe(
                'workflow.completed', self._handle_workflow_completed
            )
            self.core_orchestrator.event_bus.subscribe(
                'system.error', self._handle_system_error
            )
        
        # Autonomous controller events
        if self.autonomous_controller:
            # Setup autonomous decision handlers
            pass
    
    async def _start_coordination_processes(self):
        """Start coordination background processes"""
        
        # Start command processing
        asyncio.create_task(self._command_processor())
        
        # Start system synchronization
        asyncio.create_task(self._system_sync_loop())
        
        # Start performance monitoring
        asyncio.create_task(self._coordination_monitor())
        
        # Start conflict resolution
        asyncio.create_task(self._conflict_resolution_loop())
    
    async def _command_processor(self):
        """Process coordination commands"""
        while self.is_coordinating:
            try:
                # Get command from queue
                command = await asyncio.wait_for(self.command_queue.get(), timeout=1.0)
                
                # Process the command
                await self._process_system_command(command)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Command processor error: {e}")
                await asyncio.sleep(1.0)
    
    async def _process_system_command(self, command: SystemCommand):
        """Process a system command with coordination"""
        logger.info(f"🎯 Processing system command: {command.command_type} from {command.source}")
        
        start_time = time.time()
        
        try:
            # Check for conflicts
            conflicts = await self._check_command_conflicts(command)
            if conflicts:
                resolved_command = await self.conflict_resolver.resolve_conflicts(command, conflicts)
                if not resolved_command:
                    logger.warning(f"Command {command.command_id} could not be resolved")
                    return
                command = resolved_command
            
            # Add to pending commands
            self.pending_commands[command.command_id] = command
            
            # Route command to appropriate system
            success = await self._route_command(command)
            
            # Update metrics
            processing_time = time.time() - start_time
            self._update_coordination_metrics(command, success, processing_time)
            
            # Remove from pending
            if command.command_id in self.pending_commands:
                del self.pending_commands[command.command_id]
            
            # Add to history
            self.command_history.append(command)
            
            # Keep history manageable
            if len(self.command_history) > 1000:
                self.command_history = self.command_history[-1000:]
            
        except Exception as e:
            logger.error(f"Failed to process command {command.command_id}: {e}")
            if command.command_id in self.pending_commands:
                del self.pending_commands[command.command_id]
    
    async def _route_command(self, command: SystemCommand) -> bool:
        """Route command to appropriate system"""
        try:
            if command.target_system == 'voice':
                return await self._execute_voice_command(command)
            elif command.target_system == 'core':
                return await self._execute_core_command(command)
            elif command.target_system == 'autonomous':
                return await self._execute_autonomous_command(command)
            elif command.target_system == 'system':
                return await self._execute_system_command(command)
            else:
                logger.error(f"Unknown target system: {command.target_system}")
                return False
                
        except Exception as e:
            logger.error(f"Command routing failed: {e}")
            return False
    
    async def _execute_voice_command(self, command: SystemCommand) -> bool:
        """Execute voice system command"""
        if not self.voice_orchestrator:
            return False
        
        try:
            if command.command_type == 'speak':
                text = command.parameters.get('text', '')
                personality = command.parameters.get('personality')
                return await self.voice_orchestrator.synthesize_and_speak(text, personality)
            
            elif command.command_type == 'switch_personality':
                personality_id = command.parameters.get('personality_id')
                if self.voice_orchestrator.persona_manager:
                    return await self.voice_orchestrator.persona_manager.switch_personality(personality_id)
            
            elif command.command_type == 'set_emotion':
                emotion = command.parameters.get('emotion')
                if self.voice_orchestrator.persona_manager:
                    from ..personas.persona_manager import EmotionalState
                    emotion_state = EmotionalState(emotion)
                    return await self.voice_orchestrator.persona_manager.set_emotional_state(emotion_state)
            
            elif command.command_type == 'voice_status':
                status = self.voice_orchestrator.get_system_status()
                logger.info(f"Voice system status: {status}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Voice command execution failed: {e}")
            return False
    
    async def _execute_core_command(self, command: SystemCommand) -> bool:
        """Execute core orchestrator command"""
        if not self.core_orchestrator:
            return False
        
        try:
            if command.command_type ==  
            return False
        
        try:
            if command.command_type == 'execute_workflow':
                workflow_id = command.parameters.get('workflow_id')
                context = command.parameters.get('context', {})
                result = await self.core_orchestrator.execute_workflow(workflow_id, context)
                return result.get('status') == 'completed'
            
            elif command.command_type == 'schedule_task':
                task_data = command.parameters.get('task')
                task = Task(**task_data)
                task_id = await self.core_orchestrator.schedule_task(task)
                return task_id is not None
            
            elif command.command_type == 'get_status':
                status = self.core_orchestrator.get_orchestrator_status()
                logger.info(f"Core orchestrator status: {status}")
                return True
            
            elif command.command_type == 'publish_event':
                event_type = command.parameters.get('event_type')
                event_data = command.parameters.get('data', {})
                event_id = await self.core_orchestrator.publish_event(event_type, event_data)
                return event_id is not None
            
            return False
            
        except Exception as e:
            logger.error(f"Core command execution failed: {e}")
            return False
    
    async def _execute_autonomous_command(self, command: SystemCommand) -> bool:
        """Execute autonomous controller command"""
        if not self.autonomous_controller:
            return False
        
        try:
            if command.command_type == 'make_decision':
                decision_type = DecisionType(command.parameters.get('decision_type'))
                context = command.parameters.get('context', {})
                decision = await self.autonomous_controller.make_decision(decision_type, context)
                return decision is not None
            
            elif command.command_type == 'enable_autonomous_mode':
                mode = command.parameters.get('mode', 'standard')
                # Enable autonomous operations
                return True
            
            elif command.command_type == 'get_autonomous_status':
                status = self.autonomous_controller.get_controller_status()
                logger.info(f"Autonomous controller status: {status}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Autonomous command execution failed: {e}")
            return False
    
    async def _execute_system_command(self, command: SystemCommand) -> bool:
        """Execute system-level command"""
        if not self.system_manager:
            return False
        
        try:
            if command.command_type == 'get_system_status':
                status = self.system_manager.get_system_status()
                logger.info(f"System status: {status}")
                return True
            
            elif command.command_type == 'start_system':
                if not self.system_manager.is_running:
                    await self.system_manager.run()
                return True
            
            elif command.command_type == 'system_health_check':
                # Perform comprehensive health check
                health_status = await self._perform_health_check()
                return health_status['overall_health'] > 0.8
            
            elif command.command_type == 'emergency_shutdown':
                # Initiate emergency shutdown
                await self._emergency_shutdown()
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"System command execution failed: {e}")
            return False
    
    async def _check_command_conflicts(self, command: SystemCommand) -> List[SystemCommand]:
        """Check for command conflicts"""
        conflicts = []
        
        # Check against pending commands
        for pending_command in self.pending_commands.values():
            if self._commands_conflict(command, pending_command):
                conflicts.append(pending_command)
        
        # Check against system state
        if await self._conflicts_with_system_state(command):
            # Create a virtual conflict representing system state
            system_conflict = SystemCommand(
                command_id="system_state_conflict",
                source="system",
                target_system=command.target_system,
                command_type="system_busy",
                parameters={},
                priority=TaskPriority.HIGH,
                timestamp=datetime.now()
            )
            conflicts.append(system_conflict)
        
        return conflicts
    
    def _commands_conflict(self, command1: SystemCommand, command2: SystemCommand) -> bool:
        """Check if two commands conflict"""
        
        # Same target system conflicts
        if command1.target_system == command2.target_system:
            # Check for resource conflicts
            conflicting_pairs = [
                ('speak', 'speak'),  # Can't speak simultaneously
                ('switch_personality', 'switch_personality'),  # Can't switch personalities simultaneously
                ('system_shutdown', 'start_system'),  # Conflicting system operations
            ]
            
            for conflict_pair in conflicting_pairs:
                if (command1.command_type, command2.command_type) == conflict_pair or \
                   (command2.command_type, command1.command_type) == conflict_pair:
                    return True
        
        # Priority-based conflicts
        if command1.priority == TaskPriority.CRITICAL and command2.priority != TaskPriority.CRITICAL:
            return True
        
        return False
    
    async def _conflicts_with_system_state(self, command: SystemCommand) -> bool:
        """Check if command conflicts with current system state"""
        
        # Check voice system state
        if self.voice_orchestrator:
            voice_status = self.voice_orchestrator.get_system_status()
            
            # Can't speak if already speaking
            if command.command_type == 'speak' and voice_status['system_state'] == 'speaking':
                return True
            
            # Can't process voice commands if system is in error state
            if command.target_system == 'voice' and voice_status['system_state'] == 'error':
                return True
        
        # Check system manager state
        if self.system_manager:
            system_status = self.system_manager.get_system_status()
            
            # Can't start system if already running
            if command.command_type == 'start_system' and system_status['is_running']:
                return True
        
        return False
    
    async def _system_sync_loop(self):
        """System synchronization loop"""
        while self.is_coordinating:
            try:
                current_time = time.time()
                
                if current_time - self.last_sync_time >= self.sync_interval:
                    await self._synchronize_systems()
                    self.last_sync_time = current_time
                
                await asyncio.sleep(1.0)
                
            except Exception as e:
                logger.error(f"System sync loop error: {e}")
                await asyncio.sleep(5.0)
    
    async def _synchronize_systems(self):
        """Synchronize state between systems"""
        try:
            # Collect system states
            voice_status = self.voice_orchestrator.get_system_status() if self.voice_orchestrator else {}
            system_status = self.system_manager.get_system_status() if self.system_manager else {}
            orchestrator_status = self.core_orchestrator.get_orchestrator_status() if self.core_orchestrator else {}
            
            # Calculate sync score
            sync_score = await self._calculate_sync_score(voice_status, system_status, orchestrator_status)
            self.coordination_metrics.system_sync_score = sync_score
            
            # Perform synchronization actions if needed
            if sync_score < 0.8:
                await self._perform_sync_actions(voice_status, system_status, orchestrator_status)
            
            logger.debug(f"System sync completed - Score: {sync_score:.2f}")
            
        except Exception as e:
            logger.error(f"System synchronization failed: {e}")
    
    async def _calculate_sync_score(self, voice_status: Dict, system_status: Dict, orchestrator_status: Dict) -> float:
        """Calculate system synchronization score"""
        score = 1.0
        
        # Check if all systems are running
        systems_running = [
            voice_status.get('is_running', False),
            system_status.get('is_running', False),
            orchestrator_status.get('is_running', False)
        ]
        
        if not all(systems_running):
            score -= 0.3
        
        # Check performance alignment
        voice_response_time = voice_status.get('performance_metrics', {}).get('avg_response_time', 0)
        system_uptime = system_status.get('uptime', 0)
        
        if voice_response_time > 5.0:  # Slow voice responses
            score -= 0.2
        
        if system_uptime < 3600:  # System recently restarted
            score -= 0.1
        
        # Check error states
        if voice_status.get('system_state') == 'error':
            score -= 0.4
        
        return max(0.0, score)
    
    async def _perform_sync_actions(self, voice_status: Dict, system_status: Dict, orchestrator_status: Dict):
        """Perform synchronization actions"""
        
        # Restart failed components
        if voice_status.get('system_state') == 'error' and self.voice_orchestrator:
            logger.info("Attempting to recover voice system...")
            # Could implement voice system recovery here
        
        # Optimize performance if needed
        voice_response_time = voice_status.get('performance_metrics', {}).get('avg_response_time', 0)
        if voice_response_time > 5.0:
            logger.info("Optimizing voice system performance...")
            await self._optimize_voice_performance()
    
    async def _optimize_voice_performance(self):
        """Optimize voice system performance"""
        try:
            # Clear voice caches
            if self.voice_orchestrator and self.voice_orchestrator.tts_engine:
                # Could implement cache clearing
                pass
            
            # Reduce processing load
            if self.voice_orchestrator:
                # Could reduce number of pipeline workers temporarily
                pass
            
            logger.info("Voice performance optimization completed")
            
        except Exception as e:
            logger.error(f"Voice performance optimization failed: {e}")
    
    async def _coordination_monitor(self):
        """Monitor coordination performance"""
        while self.is_coordinating:
            try:
                # Update coordination metrics
                await self._update_coordination_health()
                
                # Check for performance issues
                if self.coordination_metrics.avg_coordination_time > 5.0:
                    logger.warning("High coordination latency detected")
                
                if self.coordination_metrics.system_sync_score < 0.7:
                    logger.warning("Poor system synchronization detected")
                
                await asyncio.sleep(10.0)  # Monitor every 10 seconds
                
            except Exception as e:
                logger.error(f"Coordination monitor error: {e}")
                await asyncio.sleep(30.0)
    
    async def _update_coordination_health(self):
        """Update coordination health metrics"""
        try:
            # Calculate success rate
            total_commands = self.coordination_metrics.total_commands
            if total_commands > 0:
                success_rate = self.coordination_metrics.successful_coordinations / total_commands
                
                # Update system health based on success rate
                if success_rate < 0.8:
                    logger.warning(f"Low coordination success rate: {success_rate:.2f}")
            
            # Check pending command queue
            if len(self.pending_commands) > 10:
                logger.warning(f"High number of pending commands: {len(self.pending_commands)}")
            
        except Exception as e:
            logger.error(f"Failed to update coordination health: {e}")
    
    async def _conflict_resolution_loop(self):
        """Conflict resolution background process"""
        while self.is_coordinating:
            try:
                # Check for long-running conflicts
                current_time = time.time()
                timeout = self.coordination_rules['coordination_timeouts']['voice_command']
                
                expired_commands = []
                for command_id, command in self.pending_commands.items():
                    command_age = current_time - command.timestamp.timestamp()
                    if command_age > timeout:
                        expired_commands.append(command_id)
                
                # Remove expired commands
                for command_id in expired_commands:
                    logger.warning(f"Command {command_id} expired, removing from pending")
                    del self.pending_commands[command_id]
                
                await asyncio.sleep(5.0)
                
            except Exception as e:
                logger.error(f"Conflict resolution loop error: {e}")
                await asyncio.sleep(10.0)
    
    async def _perform_health_check(self) -> Dict[str, Any]:
        """Perform comprehensive system health check"""
        health_status = {
            'overall_health': 0.0,
            'component_health': {},
            'issues': [],
            'recommendations': []
        }
        
        try:
            component_scores = []
            
            # Check voice system health
            if self.voice_orchestrator:
                voice_status = self.voice_orchestrator.get_system_status()
                voice_health = self._calculate_component_health(voice_status)
                health_status['component_health']['voice'] = voice_health
                component_scores.append(voice_health)
                
                if voice_health < 0.8:
                    health_status['issues'].append("Voice system performance degraded")
                    health_status['recommendations'].append("Restart voice components")
            
            # Check core system health
            if self.system_manager:
                system_status = self.system_manager.get_system_status()
                system_health = self._calculate_component_health(system_status)
                health_status['component_health']['core'] = system_health
                component_scores.append(system_health)
                
                if system_health < 0.8:
                    health_status['issues'].append("Core system performance degraded")
            
            # Check orchestrator health
            if self.core_orchestrator:
                orchestrator_status = self.core_orchestrator.get_orchestrator_status()
                orchestrator_health = self._calculate_component_health(orchestrator_status)
                health_status['component_health']['orchestrator'] = orchestrator_health
                component_scores.append(orchestrator_health)
            
            # Calculate overall health
            if component_scores:
                health_status['overall_health'] = sum(component_scores) / len(component_scores)
            
            return health_status
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            health_status['issues'].append(f"Health check error: {e}")
            return health_status
    
    def _calculate_component_health(self, status: Dict[str, Any]) -> float:
        """Calculate health score for a component"""
        health_score = 1.0
        
        # Check if component is running
        if not status.get('is_running', False):
            health_score -= 0.5
        
        # Check performance metrics
        performance = status.get('performance_metrics', {})
        
        # CPU usage
        cpu_usage = performance.get('cpu_usage', 0)
        if cpu_usage > 80:
            health_score -= 0.2
        
        # Memory usage
        memory_usage = performance.get('memory_usage', 0)
        if memory_usage > 0.85:
            health_score -= 0.2
        
        # Response time
        response_time = performance.get('avg_response_time', 0)
        if response_time > 3.0:
            health_score -= 0.1
        
        return max(0.0, health_score)
    
    async def _emergency_shutdown(self):
        """Perform emergency shutdown"""
        logger.critical("🚨 EMERGENCY SHUTDOWN INITIATED")
        
        try:
            # Stop voice processing immediately
            if self.voice_orchestrator:
                await self.voice_orchestrator.shutdown()
            
            # Stop core systems
            if self.autonomous_controller:
                await self.autonomous_controller.shutdown()
            
            if self.core_orchestrator:
                await self.core_orchestrator.shutdown()
            
            if self.system_manager:
                await self.system_manager._shutdown()
            
            logger.critical("🛑 Emergency shutdown completed")
            
        except Exception as e:
            logger.critical(f"Emergency shutdown failed: {e}")
    
    def _update_coordination_metrics(self, command: SystemCommand, success: bool, processing_time: float):
        """Update coordination metrics"""
        self.coordination_metrics.total_commands += 1
        
        if command.source == 'voice':
            self.coordination_metrics.voice_commands += 1
        elif command.source == 'autonomous':
            self.coordination_metrics.autonomous_commands += 1
        
        if success:
            self.coordination_metrics.successful_coordinations += 1
        else:
            self.coordination_metrics.failed_coordinations += 1
        
        # Update average coordination time
        total = self.coordination_metrics.total_commands
        current_avg = self.coordination_metrics.avg_coordination_time
        self.coordination_metrics.avg_coordination_time = (current_avg * (total - 1) + processing_time) / total
    
    # Event handlers
    async def _handle_voice_interaction_completed(self, data: Dict[str, Any]):
        """Handle completed voice interaction"""
        interaction_id = data.get('interaction_id')
        success = data.get('success', False)
        
        if success:
            logger.debug(f"Voice interaction {interaction_id} completed successfully")
        else:
            logger.warning(f"Voice interaction {interaction_id} failed")
    
    async def _handle_voice_system_command(self, data: Dict[str, Any]):
        """Handle system command from voice interface"""
        command_data = data.get('command', {})
        
        command = SystemCommand(
            command_id=f"voice_{int(time.time())}",
            source='voice',
            target_system=command_data.get('target_system', 'core'),
            command_type=command_data.get('command_type'),
            parameters=command_data.get('parameters', {}),
            priority=TaskPriority.HIGH,
            timestamp=datetime.now()
        )
        
        await self.command_queue.put(command)
    
    async def _handle_emergency_from_voice(self, data: Dict[str, Any]):
        """Handle emergency detected from voice interface"""
        emergency_type = data.get('emergency_type', 'unknown')
        
        logger.critical(f"🚨 Emergency detected from voice interface: {emergency_type}")
        
        # Switch to emergency mode
        self.system_mode = SystemMode.EMERGENCY
        
        # Create emergency command
        emergency_command = SystemCommand(
            command_id=f"emergency_{int(time.time())}",
            source='voice',
            target_system='system',
            command_type='emergency_response',
            parameters={'emergency_type': emergency_type},
            priority=TaskPriority.CRITICAL,
            timestamp=datetime.now()
        )
        
        await self.command_queue.put(emergency_command)
    
    async def _handle_workflow_completed(self, event: Dict[str, Any]):
        """Handle completed workflow from core orchestrator"""
        workflow_id = event.get('data', {}).get('workflow_id')
        result = event.get('data', {}).get('result', {})
        
        logger.info(f"Workflow {workflow_id} completed with status: {result.get('status')}")
    
    async def _handle_system_error(self, event: Dict[str, Any]):
        """Handle system error from core orchestrator"""
        error = event.get('data', {}).get('error', 'Unknown error')
        
        logger.error(f"System error detected: {error}")
        
        # Create error recovery command
        recovery_command = SystemCommand(
            command_id=f"recovery_{int(time.time())}",
            source='system',
            target_system='system',
            command_type='error_recovery',
            parameters={'error': error},
            priority=TaskPriority.HIGH,
            timestamp=datetime.now()
        )
        
        await self.command_queue.put(recovery_command)
    
    # Public API methods
    async def submit_command(self, command: SystemCommand) -> bool:
        """Submit command for coordination"""
        try:
            await self.command_queue.put(command)
            return True
        except asyncio.QueueFull:
            logger.error("Command queue full, cannot submit command")
            return False
    
    def get_coordination_status(self) -> Dict[str, Any]:
        """Get coordination system status"""
        return {
            'is_coordinating': self.is_coordinating,
            'system_mode': self.system_mode.value,
            'coordination_level': self.coordination_level.value,
            'pending_commands': len(self.pending_commands),
            'metrics': {
                'total_commands': self.coordination_metrics.total_commands,
                'success_rate': (self.coordination_metrics.successful_coordinations / 
                               max(1, self.coordination_metrics.total_commands)),
                'avg_coordination_time': self.coordination_metrics.avg_coordination_time,
                'system_sync_score': self.coordination_metrics.system_sync_score
            },
            'component_status': {
                'voice_orchestrator': self.voice_orchestrator is not None,
                'system_manager': self.system_manager is not None,
                'core_orchestrator': self.core_orchestrator is not None,
                'autonomous_controller': self.autonomous_controller is not None
            }
        }
    
    async def shutdown(self):
        """Shutdown system coordinator"""
        logger.info("🛑 Shutting down System Coordinator...")
        
        self.is_coordinating = False
        
        # Clear pending commands
        self.pending_commands.clear()
        
        logger.info("✅ System Coordinator shutdown complete")

class ConflictResolver:
    """Resolves conflicts between system commands"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.resolution_strategies = {
            'priority_based': self._resolve_by_priority,
            'temporal_based': self._resolve_by_time,
            'resource_based': self._resolve_by_resources,
            'user_confirmation': self._resolve_by_confirmation
        }
    
    async def resolve_conflicts(self, command: SystemCommand, conflicts: List[SystemCommand]) -> Optional[SystemCommand]:
        """Resolve command conflicts"""
        
        if not conflicts:
            return command
        
        # Try different resolution strategies
        for strategy_name, strategy_func in self.resolution_strategies.items():
            try:
                resolved_command = await strategy_func(command, conflicts)
                if resolved_command:
                    logger.info(f"Conflict resolved using {strategy_name} strategy")
                    return resolved_command
            except Exception as e:
                logger.error(f"Conflict resolution strategy {strategy_name} failed: {e}")
        
        # If no strategy worked, return None (command rejected)
        logger.warning(f"Could not resolve conflicts for command {command.command_id}")
        return None
    
    async def _resolve_by_priority(self, command: SystemCommand, conflicts: List[SystemCommand]) -> Optional[SystemCommand]:
        """Resolve conflicts based on priority"""
        
        # If command has higher priority than all conflicts, allow it
        command_priority_value = command.priority.value
        conflict_priorities = [c.priority.value for c in conflicts]
        
        if command_priority_value < min(conflict_priorities):  # Lower value = higher priority
            return command
        
        return None
    
    async def _resolve_by_time(self, command: SystemCommand, conflicts: List[SystemCommand]) -> Optional[SystemCommand]:
        """Resolve conflicts based on timing"""
        
        # If command is newer and urgent, it might override older commands
        current_time = datetime.now()
        
        for conflict in conflicts:
            conflict_age = (current_time - conflict.timestamp).total_seconds()
            
            # If conflict is old (>30 seconds) and command is urgent, override
            if conflict_age > 30 and command.priority in [TaskPriority.CRITICAL, TaskPriority.HIGH]:
                return command
        
        return None
    
    async def _resolve_by_resources(self, command: SystemCommand, conflicts: List[SystemCommand]) -> Optional[SystemCommand]:
        """Resolve conflicts based on resource availability"""
        
        # Check if command can be delayed or modified to avoid conflicts
        if command.command_type in ['speak', 'switch_personality']:
            # These commands can potentially be queued
            modified_command = command
            modified_command.parameters['delayed'] = True
            return modified_command
        
        return None
    
    async def _resolve_by_confirmation(self, command: SystemCommand, conflicts: List[SystemCommand]) -> Optional[SystemCommand]:
        """Resolve conflicts by requiring user confirmation"""
        
        # For critical commands, require confirmation
        if command.priority == TaskPriority.CRITICAL:
            command.requires_confirmation = True
            return command
        
        return None
