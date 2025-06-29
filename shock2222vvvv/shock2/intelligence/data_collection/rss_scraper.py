"""
Shock2 RSS Data Collector - Advanced Multi-Source Intelligence Gathering
Implements sophisticated RSS scraping with stealth, caching, and real-time monitoring
"""

import asyncio
import aiohttp
import feedparser
import logging
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import hashlib
import json
import re
from urllib.parse import urljoin, urlparse
import random
import time
from concurrent.futures import ThreadPoolExecutor
import sqlite3
import threading

from ...config.settings import Shock2Config
from ...utils.exceptions import DataCollectionError
from ...stealth.detection_evasion.signature_masker import SignatureMasker

logger = logging.getLogger(__name__)

@dataclass
class NewsArticle:
    """News article data structure"""
    title: str
    content: str
    url: str
    source: str
    published_date: datetime
    author: Optional[str] = None
    category: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    sentiment_score: Optional[float] = None
    importance_score: Optional[float] = None
    hash_id: str = field(init=False)
    
    def __post_init__(self):
        # Generate unique hash ID
        content_hash = hashlib.md5(f"{self.title}{self.url}".encode()).hexdigest()
        self.hash_id = content_hash

@dataclass
class SourceMetrics:
    """RSS source performance metrics"""
    url: str
    total_articles: int = 0
    successful_fetches: int = 0
    failed_fetches: int = 0
    avg_response_time: float = 0.0
    last_fetch_time: Optional[datetime] = None
    reliability_score: float = 1.0
    articles_per_hour: float = 0.0

class StealthHTTPClient:
    """Stealth HTTP client for RSS scraping"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.session = None
        self.user_agents = config.get('user_agents', [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        ])
        self.request_delays = {}
        
    async def initialize(self):
        """Initialize HTTP client"""
        connector = aiohttp.TCPConnector(
            limit=self.config.get('max_concurrent_requests', 10),
            limit_per_host=5,
            ttl_dns_cache=300,
            use_dns_cache=True
        )
        
        timeout = aiohttp.ClientTimeout(
            total=self.config.get('timeout', 30),
            connect=10,
            sock_read=20
        )
        
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={'Accept-Encoding': 'gzip, deflate'}
        )
        
        logger.info("🌐 Stealth HTTP Client initialized")
    
    async def fetch_with_stealth(self, url: str, source_name: str) -> Optional[str]:
        """Fetch URL with stealth techniques"""
        if not self.session:
            await self.initialize()
        
        # Apply request delay for this source
        await self._apply_request_delay(source_name)
        
        # Rotate user agent
        headers = {
            'User-Agent': random.choice(self.user_agents),
            'Accept': 'application/rss+xml, application/xml, text/xml, */*',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }
        
        # Add random headers to appear more human
        if random.random() < 0.3:
            headers['Cache-Control'] = 'no-cache'
        if random.random() < 0.2:
            headers['Pragma'] = 'no-cache'
        
        try:
            start_time = time.time()
            
            async with self.session.get(url, headers=headers) as response:
                if response.status == 200:
                    content = await response.text()
                    
                    # Update metrics
                    response_time = time.time() - start_time
                    self._update_source_metrics(source_name, True, response_time)
                    
                    return content
                else:
                    logger.warning(f"HTTP {response.status} for {url}")
                    self._update_source_metrics(source_name, False, 0)
                    return None
                    
        except Exception as e:
            logger.error(f"Failed to fetch {url}: {e}")
            self._update_source_metrics(source_name, False, 0)
            return None
    
    async def _apply_request_delay(self, source_name: str):
        """Apply intelligent request delay"""
        current_time = time.time()
        last_request = self.request_delays.get(source_name, 0)
        
        # Calculate delay based on source reliability and politeness
        base_delay = self.config.get('request_delay', 1.0)
        time_since_last = current_time - last_request
        
        if time_since_last < base_delay:
            delay = base_delay - time_since_last
            # Add random jitter
            delay += random.uniform(0, 0.5)
            await asyncio.sleep(delay)
        
        self.request_delays[source_name] = time.time()
    
    def _update_source_metrics(self, source_name: str, success: bool, response_time: float):
        """Update source performance metrics"""
        # This would update metrics in a shared metrics store
        pass
    
    async def shutdown(self):
        """Shutdown HTTP client"""
        if self.session:
            await self.session.close()
        logger.info("🌐 Stealth HTTP Client shutdown")

class IntelligentCache:
    """Intelligent caching system for RSS data"""
    
    def __init__(self, cache_dir: str = "data/cache"):
        self.cache_dir = cache_dir
        self.db_path = f"{cache_dir}/rss_cache.db"
        self.lock = threading.Lock()
        self._ensure_cache_dir()
        self._initialize_db()
    
    def _ensure_cache_dir(self):
        """Ensure cache directory exists"""
        import os
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def _initialize_db(self):
        """Initialize cache database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS rss_cache (
                    url TEXT PRIMARY KEY,
                    content TEXT,
                    timestamp REAL,
                    etag TEXT,
                    last_modified TEXT,
                    expires REAL
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS article_cache (
                    hash_id TEXT PRIMARY KEY,
                    title TEXT,
                    content TEXT,
                    url TEXT,
                    source TEXT,
                    published_date REAL,
                    cached_date REAL
                )
            ''')
            conn.commit()
    
    def get_cached_feed(self, url: str) -> Optional[Dict]:
        """Get cached RSS feed if still valid"""
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    'SELECT content, timestamp, expires FROM rss_cache WHERE url = ?',
                    (url,)
                )
                row = cursor.fetchone()
                
                if row:
                    content, timestamp, expires = row
                    current_time = time.time()
                    
                    # Check if cache is still valid
                    if expires > current_time:
                        return {
                            'content': content,
                            'timestamp': timestamp,
                            'from_cache': True
                        }
        
        return None
    
    def cache_feed(self, url: str, content: str, ttl: int = 300):
        """Cache RSS feed content"""
        current_time = time.time()
        expires = current_time + ttl
        
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT OR REPLACE INTO rss_cache 
                    (url, content, timestamp, expires) 
                    VALUES (?, ?, ?, ?)
                ''', (url, content, current_time, expires))
                conn.commit()
    
    def cache_article(self, article: NewsArticle):
        """Cache processed article"""
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT OR REPLACE INTO article_cache 
                    (hash_id, title, content, url, source, published_date, cached_date) 
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    article.hash_id,
                    article.title,
                    article.content,
                    article.url,
                    article.source,
                    article.published_date.timestamp(),
                    time.time()
                ))
                conn.commit()
    
    def is_article_cached(self, hash_id: str) -> bool:
        """Check if article is already cached"""
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    'SELECT 1 FROM article_cache WHERE hash_id = ?',
                    (hash_id,)
                )
                return cursor.fetchone() is not None
    
    def cleanup_expired(self):
        """Clean up expired cache entries"""
        current_time = time.time()
        
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                # Clean expired RSS cache
                conn.execute('DELETE FROM rss_cache WHERE expires < ?', (current_time,))
                
                # Clean old articles (older than 7 days)
                week_ago = current_time - (7 * 24 * 3600)
                conn.execute('DELETE FROM article_cache WHERE cached_date < ?', (week_ago,))
                
                conn.commit()

class ContentProcessor:
    """Advanced content processing for RSS articles"""
    
    def __init__(self):
        self.nlp = None
        self.importance_keywords = {
            'breaking': 3.0,
            'urgent': 2.5,
            'exclusive': 2.0,
            'developing': 2.0,
            'alert': 2.5,
            'crisis': 2.0,
            'emergency': 2.0,
            'major': 1.5,
            'significant': 1.3,
            'important': 1.2
        }
        
        self.category_keywords = {
            'politics': ['election', 'government', 'policy', 'congress', 'senate', 'president'],
            'technology': ['tech', 'ai', 'software', 'hardware', 'startup', 'innovation'],
            'business': ['market', 'stock', 'economy', 'finance', 'company', 'earnings'],
            'health': ['medical', 'health', 'disease', 'treatment', 'vaccine', 'hospital'],
            'science': ['research', 'study', 'discovery', 'experiment', 'scientist', 'university'],
            'sports': ['game', 'team', 'player', 'championship', 'league', 'tournament']
        }
    
    async def initialize(self):
        """Initialize content processor"""
        try:
            import spacy
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("📝 Content Processor initialized with spaCy")
        except:
            logger.warning("spaCy not available, using basic processing")
    
    def process_article(self, entry: Dict, source_name: str) -> Optional[NewsArticle]:
        """Process RSS entry into NewsArticle"""
        try:
            # Extract basic information
            title = entry.get('title', '').strip()
            content = self._extract_content(entry)
            url = entry.get('link', '')
            
            if not title or not url:
                return None
            
            # Parse published date
            published_date = self._parse_date(entry.get('published_parsed'))
            
            # Extract author
            author = entry.get('author', '').strip() or None
            
            # Create article
            article = NewsArticle(
                title=title,
                content=content,
                url=url,
                source=source_name,
                published_date=published_date,
                author=author
            )
            
            # Process content
            article.category = self._classify_category(title + ' ' + content)
            article.tags = self._extract_tags(title + ' ' + content)
            article.importance_score = self._calculate_importance(title + ' ' + content)
            article.sentiment_score = self._analyze_sentiment(title + ' ' + content)
            
            return article
            
        except Exception as e:
            logger.error(f"Error processing article: {e}")
            return None
    
    def _extract_content(self, entry: Dict) -> str:
        """Extract content from RSS entry"""
        # Try different content fields
        content_fields = ['content', 'summary', 'description']
        
        for field in content_fields:
            if field in entry:
                content = entry[field]
                if isinstance(content, list) and content:
                    content = content[0].get('value', '')
                elif isinstance(content, dict):
                    content = content.get('value', '')
                
                if content:
                    # Clean HTML tags
                    content = re.sub(r'<[^>]+>', '', str(content))
                    # Clean extra whitespace
                    content = re.sub(r'\s+', ' ', content).strip()
                    return content
        
        return ""
    
    def _parse_date(self, date_tuple) -> datetime:
        """Parse RSS date tuple"""
        if date_tuple:
            try:
                return datetime(*date_tuple[:6])
            except:
                pass
        
        return datetime.now()
    
    def _classify_category(self, text: str) -> Optional[str]:
        """Classify article category"""
        text_lower = text.lower()
        category_scores = {}
        
        for category, keywords in self.category_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                category_scores[category] = score
        
        if category_scores:
            return max(category_scores, key=category_scores.get)
        
        return None
    
    def _extract_tags(self, text: str) -> List[str]:
        """Extract relevant tags from text"""
        if not self.nlp:
            return []
        
        try:
            doc = self.nlp(text[:1000])  # Limit text length
            
            # Extract named entities
            entities = [ent.text.lower() for ent in doc.ents 
                       if ent.label_ in ['PERSON', 'ORG', 'GPE', 'EVENT']]
            
            # Extract important noun phrases
            noun_phrases = [chunk.text.lower() for chunk in doc.noun_chunks 
                           if len(chunk.text.split()) <= 3]
            
            # Combine and deduplicate
            tags = list(set(entities + noun_phrases))
            
            # Filter and limit
            tags = [tag for tag in tags if len(tag) > 2 and len(tag) < 50]
            return tags[:10]  # Limit to 10 tags
            
        except Exception as e:
            logger.warning(f"Tag extraction failed: {e}")
            return []
    
    def _calculate_importance(self, text: str) -> float:
        """Calculate article importance score"""
        text_lower = text.lower()
        importance_score = 1.0
        
        # Check for importance keywords
        for keyword, weight in self.importance_keywords.items():
            if keyword in text_lower:
                importance_score *= weight
        
        # Length factor (longer articles might be more important)
        word_count = len(text.split())
        if word_count > 500:
            importance_score *= 1.2
        elif word_count < 100:
            importance_score *= 0.8
        
        # Normalize to 0-10 scale
        return min(importance_score, 10.0)
    
    def _analyze_sentiment(self, text: str) -> float:
        """Analyze sentiment of text"""
        # Simple sentiment analysis based on keywords
        positive_words = ['good', 'great', 'excellent', 'positive', 'success', 'win', 'growth']
        negative_words = ['bad', 'terrible', 'negative', 'crisis', 'fail', 'loss', 'decline']
        
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        total_words = len(text.split())
        if total_words == 0:
            return 0.0
        
        sentiment = (positive_count - negative_count) / max(total_words / 100, 1)
        return max(-1.0, min(1.0, sentiment))  # Normalize to -1 to 1

class RSSDataCollector:
    """Main RSS data collection system"""
    
    def __init__(self, config: Shock2Config):
        self.config = config
        self.http_client = StealthHTTPClient(config.intelligence.__dict__)
        self.cache = IntelligentCache()
        self.processor = ContentProcessor()
        
        self.sources = config.data_sources
        self.source_metrics: Dict[str, SourceMetrics] = {}
        self.collected_articles: List[NewsArticle] = []
        self.executor = ThreadPoolExecutor(max_workers=5)
        
        self.is_running = False
        self.collection_stats = {
            'total_articles': 0,
            'successful_sources': 0,
            'failed_sources': 0,
            'cache_hits': 0,
            'processing_errors': 0
        }
    
    async def initialize(self):
        """Initialize RSS data collector"""
        logger.info("📡 Initializing RSS Data Collector...")
        
        await self.http_client.initialize()
        await self.processor.initialize()
        
        # Initialize source metrics
        for source_url in self.sources:
            source_name = self._get_source_name(source_url)
            self.source_metrics[source_name] = SourceMetrics(url=source_url)
        
        self.is_running = True
        logger.info(f"✅ RSS Data Collector initialized - {len(self.sources)} sources configured")
    
    async def collect_news_data(self) -> List[NewsArticle]:
        """Collect news data from all sources"""
        if not self.is_running:
            raise DataCollectionError("Collector not initialized")
        
        logger.info(f"📊 Collecting news from {len(self.sources)} sources...")
        
        # Clean up expired cache
        self.cache.cleanup_expired()
        
        # Collect from all sources concurrently
        collection_tasks = []
        for source_url in self.sources:
            task = asyncio.create_task(self._collect_from_source(source_url))
            collection_tasks.append(task)
        
        # Wait for all collections to complete
        source_results = await asyncio.gather(*collection_tasks, return_exceptions=True)
        
        # Process results
        all_articles = []
        successful_sources = 0
        
        for i, result in enumerate(source_results):
            if isinstance(result, Exception):
                logger.error(f"Source {self.sources[i]} failed: {result}")
                self.collection_stats['failed_sources'] += 1
            elif result:
                all_articles.extend(result)
                successful_sources += 1
        
        # Deduplicate articles
        unique_articles = self._deduplicate_articles(all_articles)
        
        # Update statistics
        self.collection_stats.update({
            'total_articles': len(unique_articles),
            'successful_sources': successful_sources,
            'last_collection': datetime.now()
        })
        
        self.collected_articles = unique_articles
        
        logger.info(f"✅ Collection complete - {len(unique_articles)} unique articles from {successful_sources} sources")
        
        return unique_articles
    
    async def _collect_from_source(self, source_url: str) -> List[NewsArticle]:
        """Collect articles from a single RSS source"""
        source_name = self._get_source_name(source_url)
        
        try:
            # Check cache first
            cached_feed = self.cache.get_cached_feed(source_url)
            
            if cached_feed:
                feed_content = cached_feed['content']
                self.collection_stats['cache_hits'] += 1
                logger.debug(f"Using cached feed for {source_name}")
            else:
                # Fetch fresh content
                feed_content = await self.http_client.fetch_with_stealth(source_url, source_name)
                
                if not feed_content:
                    return []
                
                # Cache the content
                self.cache.cache_feed(source_url, feed_content)
            
            # Parse RSS feed
            articles = await self._parse_rss_feed(feed_content, source_name)
            
            # Update source metrics
            metrics = self.source_metrics[source_name]
            metrics.total_articles += len(articles)
            metrics.successful_fetches += 1
            metrics.last_fetch_time = datetime.now()
            
            return articles
            
        except Exception as e:
            logger.error(f"Failed to collect from {source_name}: {e}")
            
            # Update failure metrics
            if source_name in self.source_metrics:
                self.source_metrics[source_name].failed_fetches += 1
            
            return []
    
    async def _parse_rss_feed(self, feed_content: str, source_name: str) -> List[NewsArticle]:
        """Parse RSS feed content"""
        try:
            # Parse feed using feedparser
            feed = await asyncio.get_event_loop().run_in_executor(
                self.executor, feedparser.parse, feed_content
            )
            
            if not feed.entries:
                logger.warning(f"No entries found in feed from {source_name}")
                return []
            
            # Process entries
            articles = []
            for entry in feed.entries:
                try:
                    article = self.processor.process_article(entry, source_name)
                    
                    if article:
                        # Check if already cached
                        if not self.cache.is_article_cached(article.hash_id):
                            articles.append(article)
                            self.cache.cache_article(article)
                        
                except Exception as e:
                    logger.warning(f"Failed to process article from {source_name}: {e}")
                    self.collection_stats['processing_errors'] += 1
            
            logger.debug(f"Parsed {len(articles)} new articles from {source_name}")
            return articles
            
        except Exception as e:
            logger.error(f"Failed to parse RSS feed from {source_name}: {e}")
            return []
    
    def _get_source_name(self, url: str) -> str:
        """Extract source name from URL"""
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            # Remove www. prefix
            if domain.startswith('www.'):
                domain = domain[4:]
            return domain
        except:
            return url
    
    def _deduplicate_articles(self, articles: List[NewsArticle]) -> List[NewsArticle]:
        """Remove duplicate articles"""
        seen_hashes = set()
        unique_articles = []
        
        for article in articles:
            if article.hash_id not in seen_hashes:
                seen_hashes.add(article.hash_id)
                unique_articles.append(article)
        
        return unique_articles
    
    def get_collection_metrics(self) -> Dict:
        """Get comprehensive collection metrics"""
        source_stats = {}
        for name, metrics in self.source_metrics.items():
            reliability = 0
            if metrics.successful_fetches + metrics.failed_fetches > 0:
                reliability = metrics.successful_fetches / (metrics.successful_fetches + metrics.failed_fetches)
            
            source_stats[name] = {
                'total_articles': metrics.total_articles,
                'successful_fetches': metrics.successful_fetches,
                'failed_fetches': metrics.failed_fetches,
                'reliability_score': reliability,
                'last_fetch': metrics.last_fetch_time.isoformat() if metrics.last_fetch_time else None
            }
        
        return {
            'collection_stats': self.collection_stats,
            'source_metrics': source_stats,
            'total_sources': len(self.sources),
            'active_sources': len([m for m in self.source_metrics.values() if m.successful_fetches > 0])
        }
    
    async def shutdown(self):
        """Shutdown RSS data collector"""
        logger.info("📡 Shutting down RSS Data Collector...")
        
        self.is_running = False
        await self.http_client.shutdown()
        self.executor.shutdown(wait=True)
        
        logger.info("✅ RSS Data Collector shutdown complete")
