"""
Shock2 System Manager - Main orchestration and coordination
"""

import asyncio
import logging
import signal
from datetime import datetime
from typing import Dict, Any, Optional

from ..config.settings import Shock2Config
from ..neural.quantum_core.neural_mesh import QuantumNeuralMesh
from ..intelligence.data_collection.rss_scraper import RSSDataCollector
from ..generation.engines.news_generator import NewsGenerationEngine
from ..publishing.platforms.file_publisher import FilePublisher
from ..monitoring.metrics.system_metrics import SystemMetricsCollector
from ..utils.exceptions import Shock2Exception

logger = logging.getLogger(__name__)

class Shock2SystemManager:
    """
    Main system manager coordinating all Shock2 components
    Handles initialization, orchestration, and lifecycle management
    """
    
    def __init__(self, config: Shock2Config):
        self.config = config
        self.is_running = False
        self.start_time = None
        
        # Core components
        self.neural_mesh = None
        self.data_collector = None
        self.generation_engine = None
        self.publisher = None
        self.metrics_collector = None
        
        # System state
        self.system_stats = {
            "articles_processed": 0,
            "content_generated": 0,
            "articles_published": 0,
            "errors": 0,
            "uptime": 0
        }
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info("🤖 Shock2 System Manager initialized")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.is_running = False
    
    async def initialize(self):
        """Initialize all system components"""
        logger.info("🔧 Initializing Shock2 system components...")
        
        try:
            # Initialize neural mesh
            logger.info("🧠 Initializing Quantum Neural Mesh...")
            self.neural_mesh = QuantumNeuralMesh(self.config)
            await self.neural_mesh.initialize()
            
            # Initialize data collector
            logger.info("📡 Initializing Intelligence Gathering...")
            self.data_collector = RSSDataCollector(self.config)
            await self.data_collector.initialize()
            
            # Initialize generation engine
            logger.info("✍️ Initializing Content Generation Engine...")
            self.generation_engine = NewsGenerationEngine(self.config, self.neural_mesh)
            await self.generation_engine.initialize()
            
            # Initialize publisher
            logger.info("📢 Initializing Publishing System...")
            self.publisher = FilePublisher(self.config)
            await self.publisher.initialize()
            
            # Initialize metrics collector
            logger.info("📊 Initializing Monitoring System...")
            self.metrics_collector = SystemMetricsCollector(self.config)
            await self.metrics_collector.initialize()
            
            logger.info("✅ All system components initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize system components: {e}")
            raise Shock2Exception(f"System initialization failed: {e}")
    
    async def run(self):
        """Main system execution loop"""
        logger.info("🚀 Starting Shock2 autonomous operation...")
        
        try:
            # Initialize components
            await self.initialize()
            
            # Start system
            self.is_running = True
            self.start_time = datetime.now()
            
            logger.info("🎯 Shock2 system is now fully operational")
            print("✅ Shock2 AI News System is running!")
            print("📊 Monitoring console: http://localhost:8080")
            print("📁 Output directory: ./output/")
            print("🔍 Logs directory: ./logs/")
            print("\nPress Ctrl+C to stop the system\n")
            
            # Main processing loop
            while self.is_running:
                try:
                    await self._execute_cycle()
                    await asyncio.sleep(self.config.cycle_interval)
                    
                except Exception as e:
                    logger.error(f"Error in processing cycle: {e}")
                    self.system_stats["errors"] += 1
                    await asyncio.sleep(60)  # Wait before retry
                    
        except Exception as e:
            logger.error(f"Fatal error in main execution: {e}")
            raise
        finally:
            await self._shutdown()
    
    async def _execute_cycle(self):
        """Execute one complete processing cycle"""
        cycle_start = datetime.now()
        logger.info("🔄 Starting new processing cycle...")
        
        try:
            # Step 1: Collect intelligence data
            logger.info("📡 Gathering intelligence from sources...")
            raw_articles = await self.data_collector.collect_news_data()
            
            if not raw_articles:
                logger.info("No new articles found, skipping cycle")
                return
            
            self.system_stats["articles_processed"] += len(raw_articles)
            logger.info(f"📊 Processed {len(raw_articles)} source articles")
            
            # Step 2: Generate content using neural mesh
            logger.info("🧠 Generating content with neural networks...")
            generated_content = await self.generation_engine.generate_articles(raw_articles)
            
            if not generated_content:
                logger.info("No content generated, skipping publishing")
                return
            
            self.system_stats["content_generated"] += len(generated_content)
            logger.info(f"✍️ Generated {len(generated_content)} articles")
            
            # Step 3: Publish content
            logger.info("📢 Publishing generated content...")
            published_count = await self.publisher.publish_articles(generated_content)
            
            self.system_stats["articles_published"] += published_count
            logger.info(f"📰 Published {published_count} articles")
            
            # Step 4: Update metrics
            cycle_duration = (datetime.now() - cycle_start).total_seconds()
            await self.metrics_collector.record_cycle_metrics({
                "cycle_duration": cycle_duration,
                "articles_processed": len(raw_articles),
                "content_generated": len(generated_content),
                "articles_published": published_count
            })
            
            logger.info(f"✅ Cycle completed in {cycle_duration:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Error in processing cycle: {e}")
            raise
    
    async def _shutdown(self):
        """Graceful system shutdown"""
        logger.info("🛑 Initiating Shock2 system shutdown...")
        
        try:
            # Shutdown components in reverse order
            if self.metrics_collector:
                await self.metrics_collector.shutdown()
                
            if self.publisher:
                await self.publisher.shutdown()
                
            if self.generation_engine:
                await self.generation_engine.shutdown()
                
            if self.data_collector:
                await self.data_collector.shutdown()
                
            if self.neural_mesh:
                await self.neural_mesh.shutdown()
            
            # Log final statistics
            if self.start_time:
                uptime = (datetime.now() - self.start_time).total_seconds()
                self.system_stats["uptime"] = uptime
                
                logger.info("📊 Final System Statistics:")
                logger.info(f"   Uptime: {uptime:.2f} seconds")
                logger.info(f"   Articles Processed: {self.system_stats['articles_processed']}")
                logger.info(f"   Content Generated: {self.system_stats['content_generated']}")
                logger.info(f"   Articles Published: {self.system_stats['articles_published']}")
                logger.info(f"   Errors: {self.system_stats['errors']}")
            
            logger.info("✅ Shock2 system shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        uptime = 0
        if self.start_time:
            uptime = (datetime.now() - self.start_time).total_seconds()
            
        return {
            "is_running": self.is_running,
            "uptime": uptime,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "statistics": self.system_stats.copy(),
            "components": {
                "neural_mesh": self.neural_mesh is not None,
                "data_collector": self.data_collector is not None,
                "generation_engine": self.generation_engine is not None,
                "publisher": self.publisher is not None,
                "metrics_collector": self.metrics_collector is not None
            }
        }
