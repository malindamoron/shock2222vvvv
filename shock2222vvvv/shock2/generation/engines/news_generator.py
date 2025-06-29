"""
Shock2 News Generation Engine - Advanced AI Content Creation
Implements sophisticated news generation with multiple article types and styles
"""

import asyncio
import logging
import random
import re
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
import numpy as np
import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    pipeline, GPT2LMHeadModel, GPT2Tokenizer
)

from ...config.settings import Shock2Config
from ...neural.quantum_core.neural_mesh import QuantumNeuralMesh
from ...intelligence.data_collection.rss_scraper import NewsArticle
from ...stealth.detection_evasion.signature_masker import SignatureMasker
from ...utils.exceptions import ContentGenerationError

logger = logging.getLogger(__name__)

@dataclass
class GeneratedArticle:
    """Generated article data structure"""
    title: str
    content: str
    article_type: str  # 'breaking', 'analysis', 'opinion', 'summary'
    source_articles: List[str]  # Source article IDs
    generation_time: datetime
    word_count: int
    quality_score: float
    seo_keywords: List[str]
    readability_score: float
    uniqueness_score: float

class ArticleTemplate:
    """Article template system for different content types"""
    
    def __init__(self):
        self.templates = {
            'breaking': {
                'structure': [
                    'headline_hook',
                    'lead_paragraph',
                    'key_details',
                    'background_context',
                    'expert_quotes',
                    'implications',
                    'conclusion'
                ],
                'tone': 'urgent',
                'length_range': (300, 800),
                'keywords_density': 0.02
            },
            'analysis': {
                'structure': [
                    'analytical_headline',
                    'thesis_statement',
                    'evidence_presentation',
                    'multiple_perspectives',
                    'data_analysis',
                    'expert_insights',
                    'future_implications',
                    'conclusion'
                ],
                'tone': 'analytical',
                'length_range': (800, 1500),
                'keywords_density': 0.015
            },
            'opinion': {
                'structure': [
                    'provocative_headline',
                    'personal_stance',
                    'supporting_arguments',
                    'counterargument_acknowledgment',
                    'evidence_backing',
                    'emotional_appeal',
                    'call_to_action'
                ],
                'tone': 'persuasive',
                'length_range': (600, 1200),
                'keywords_density': 0.018
            },
            'summary': {
                'structure': [
                    'clear_headline',
                    'key_points_overview',
                    'chronological_events',
                    'important_figures',
                    'current_status',
                    'next_steps'
                ],
                'tone': 'informative',
                'length_range': (200, 500),
                'keywords_density': 0.025
            }
        }
    
    def get_template(self, article_type: str) -> Dict:
        """Get template for article type"""
        return self.templates.get(article_type, self.templates['breaking'])

class AdvancedLanguageModel:
    """Advanced language model for content generation"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.tokenizer = None
        self.model = None
        self.generation_pipeline = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    async def initialize(self):
        """Initialize language model"""
        logger.info("🤖 Initializing Advanced Language Model...")
        
        model_name = self.config.get('model_name', 'microsoft/DialoGPT-medium')
        
        try:
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            
            # Add padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Move model to device
            self.model.to(self.device)
            self.model.eval()
            
            # Create generation pipeline
            self.generation_pipeline = pipeline(
                'text-generation',
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device.type == 'cuda' else -1
            )
            
            logger.info(f"✅ Language Model initialized - {model_name} on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to initialize language model: {e}")
            raise ContentGenerationError(f"Model initialization failed: {e}")
    
    async def generate_text(self, prompt: str, max_length: int = 500, temperature: float = 0.8) -> str:
        """Generate text from prompt"""
        try:
            # Prepare generation parameters
            generation_params = {
                'max_length': max_length,
                'temperature': temperature,
                'top_p': self.config.get('top_p', 0.9),
                'do_sample': True,
                'pad_token_id': self.tokenizer.eos_token_id,
                'num_return_sequences': 1,
                'repetition_penalty': 1.1,
                'length_penalty': 1.0
            }
            
            # Generate text
            with torch.no_grad():
                generated = self.generation_pipeline(
                    prompt,
                    **generation_params
                )
            
            # Extract generated text
            generated_text = generated[0]['generated_text']
            
            # Remove the original prompt
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):].strip()
            
            return generated_text
            
        except Exception as e:
            logger.error(f"Text generation failed: {e}")
            return ""
    
    async def generate_with_constraints(self, prompt: str, constraints: Dict) -> str:
        """Generate text with specific constraints"""
        max_length = constraints.get('max_length', 500)
        min_length = constraints.get('min_length', 100)
        keywords = constraints.get('keywords', [])
        tone = constraints.get('tone', 'neutral')
        
        # Adjust generation parameters based on constraints
        temperature = self._get_temperature_for_tone(tone)
        
        # Generate initial text
        generated_text = await self.generate_text(prompt, max_length, temperature)
        
        # Apply constraints
        if len(generated_text.split()) < min_length:
            # Extend text if too short
            extension_prompt = f"{prompt} {generated_text} Furthermore,"
            additional_text = await self.generate_text(extension_prompt, max_length - len(generated_text.split()))
            generated_text += " " + additional_text
        
        # Ensure keywords are present
        generated_text = self._ensure_keywords(generated_text, keywords)
        
        return generated_text
    
    def _get_temperature_for_tone(self, tone: str) -> float:
        """Get appropriate temperature for tone"""
        tone_temperatures = {
            'urgent': 0.7,
            'analytical': 0.6,
            'persuasive': 0.8,
            'informative': 0.5,
            'neutral': 0.7
        }
        return tone_temperatures.get(tone, 0.7)
    
    def _ensure_keywords(self, text: str, keywords: List[str]) -> str:
        """Ensure keywords are naturally integrated"""
        if not keywords:
            return text
        
        text_lower = text.lower()
        missing_keywords = [kw for kw in keywords if kw.lower() not in text_lower]
        
        if missing_keywords:
            # Add missing keywords naturally
            sentences = text.split('. ')
            for keyword in missing_keywords[:3]:  # Limit to 3 keywords
                if sentences:
                    # Insert keyword into a random sentence
                    insert_idx = random.randint(0, len(sentences) - 1)
                    sentence = sentences[insert_idx]
                    
                    # Find a good insertion point
                    words = sentence.split()
                    if len(words) > 5:
                        insert_pos = random.randint(2, len(words) - 2)
                        words.insert(insert_pos, f"regarding {keyword},")
                        sentences[insert_idx] = ' '.join(words)
            
            text = '. '.join(sentences)
        
        return text

class ContentOptimizer:
    """Content optimization for SEO and readability"""
    
    def __init__(self):
        self.seo_keywords_db = {}
        self.readability_metrics = {}
        
    async def initialize(self):
        """Initialize content optimizer"""
        logger.info("📈 Initializing Content Optimizer...")
        
        # Load SEO keyword database (simplified)
        self.seo_keywords_db = {
            'politics': ['election', 'government', 'policy', 'congress', 'senate'],
            'technology': ['AI', 'tech', 'innovation', 'startup', 'digital'],
            'business': ['market', 'economy', 'finance', 'investment', 'growth'],
            'health': ['medical', 'healthcare', 'treatment', 'research', 'wellness'],
            'science': ['research', 'study', 'discovery', 'experiment', 'breakthrough'],
            'sports': ['game', 'championship', 'team', 'player', 'tournament']
        }
        
        logger.info("✅ Content Optimizer initialized")
    
    def optimize_for_seo(self, content: str, category: str = None) -> Tuple[str, List[str]]:
        """Optimize content for SEO"""
        # Get relevant keywords
        keywords = self._get_seo_keywords(content, category)
        
        # Optimize content structure
        optimized_content = self._optimize_content_structure(content, keywords)
        
        # Add meta information
        optimized_content = self._add_seo_elements(optimized_content, keywords)
        
        return optimized_content, keywords
    
    def _get_seo_keywords(self, content: str, category: str = None) -> List[str]:
        """Extract and generate SEO keywords"""
        # Extract existing keywords from content
        words = re.findall(r'\b\w+\b', content.lower())
        word_freq = {}
        
        for word in words:
            if len(word) > 3:  # Ignore short words
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Get most frequent words
        frequent_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
        content_keywords = [word for word, freq in frequent_words]
        
        # Add category-specific keywords
        category_keywords = []
        if category and category in self.seo_keywords_db:
            category_keywords = self.seo_keywords_db[category][:5]
        
        # Combine and deduplicate
        all_keywords = list(set(content_keywords + category_keywords))
        
        return all_keywords[:8]  # Limit to 8 keywords
    
    def _optimize_content_structure(self, content: str, keywords: List[str]) -> str:
        """Optimize content structure for SEO"""
        sentences = content.split('. ')
        
        # Ensure first paragraph contains keywords
        if sentences and keywords:
            first_sentence = sentences[0]
            primary_keyword = keywords[0]
            
            if primary_keyword.lower() not in first_sentence.lower():
                # Add keyword to first sentence naturally
                first_sentence = f"In recent developments regarding {primary_keyword}, {first_sentence.lower()}"
                sentences[0] = first_sentence
        
        # Add keyword variations throughout
        for i, sentence in enumerate(sentences[1:], 1):
            if i < len(keywords) and len(sentences) > i:
                keyword = keywords[i % len(keywords)]
                if keyword.lower() not in sentence.lower() and random.random() < 0.3:
                    # Add keyword naturally
                    sentences[i] = sentence.replace(',', f', particularly in {keyword},', 1)
        
        return '. '.join(sentences)
    
    def _add_seo_elements(self, content: str, keywords: List[str]) -> str:
        """Add SEO elements to content"""
        # Add keyword-rich conclusion if content is long enough
        if len(content.split()) > 200:
            conclusion_keywords = keywords[:3]
            conclusion = f"\n\nThis development in {', '.join(conclusion_keywords)} represents a significant shift in the current landscape."
            content += conclusion
        
        return content
    
    def calculate_readability_score(self, content: str) -> float:
        """Calculate readability score (simplified Flesch Reading Ease)"""
        sentences = content.split('.')
        words = content.split()
        
        if not sentences or not words:
            return 0.0
        
        # Count syllables (simplified)
        syllables = sum(self._count_syllables(word) for word in words)
        
        # Flesch Reading Ease formula (simplified)
        avg_sentence_length = len(words) / len(sentences)
        avg_syllables_per_word = syllables / len(words)
        
        score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
        
        # Normalize to 0-1 scale
        return max(0, min(1, score / 100))
    
    def _count_syllables(self, word: str) -> int:
        """Count syllables in a word (simplified)"""
        word = word.lower()
        vowels = 'aeiouy'
        syllable_count = 0
        prev_was_vowel = False
        
        for char in word:
            if char in vowels:
                if not prev_was_vowel:
                    syllable_count += 1
                prev_was_vowel = True
            else:
                prev_was_vowel = False
        
        # Handle silent e
        if word.endswith('e'):
            syllable_count -= 1
        
        return max(1, syllable_count)

class NewsGenerationEngine:
    """Main news generation engine"""
    
    def __init__(self, config: Shock2Config, neural_mesh: QuantumNeuralMesh):
        self.config = config
        self.neural_mesh = neural_mesh
        
        # Initialize components
        self.language_model = AdvancedLanguageModel(config.neural.__dict__)
        self.template_system = ArticleTemplate()
        self.content_optimizer = ContentOptimizer()
        self.signature_masker = SignatureMasker(config.stealth.__dict__)
        
        # Generation statistics
        self.generation_stats = {
            'total_generated': 0,
            'by_type': {'breaking': 0, 'analysis': 0, 'opinion': 0, 'summary': 0},
            'avg_quality_score': 0.0,
            'avg_generation_time': 0.0
        }
        
        self.is_initialized = False
    
    async def initialize(self):
        """Initialize news generation engine"""
        logger.info("✍️ Initializing News Generation Engine...")
        
        # Initialize all components
        await self.language_model.initialize()
        await self.content_optimizer.initialize()
        await self.signature_masker.initialize()
        
        self.is_initialized = True
        logger.info("✅ News Generation Engine initialized")
    
    async def generate_articles(self, source_articles: List[NewsArticle]) -> List[GeneratedArticle]:
        """Generate articles from source material"""
        if not self.is_initialized:
            raise ContentGenerationError("Generation engine not initialized")
        
        if not source_articles:
            logger.warning("No source articles provided")
            return []
        
        logger.info(f"🎯 Generating articles from {len(source_articles)} source articles...")
        
        # Analyze source articles
        article_analysis = self._analyze_source_articles(source_articles)
        
        # Determine what to generate
        generation_plan = self._create_generation_plan(article_analysis)
        
        # Generate articles
        generated_articles = []
        generation_tasks = []
        
        for plan_item in generation_plan:
            task = asyncio.create_task(self._generate_single_article(plan_item, source_articles))
            generation_tasks.append(task)
        
        # Wait for all generations
        results = await asyncio.gather(*generation_tasks, return_exceptions=True)
        
        # Process results
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Article generation failed: {result}")
            elif result:
                generated_articles.append(result)
        
        # Update statistics
        self._update_generation_stats(generated_articles)
        
        logger.info(f"✅ Generated {len(generated_articles)} articles")
        
        return generated_articles
    
    def _analyze_source_articles(self, articles: List[NewsArticle]) -> Dict:
        """Analyze source articles to determine generation strategy"""
        analysis = {
            'total_articles': len(articles),
            'categories': {},
            'importance_levels': [],
            'sentiment_distribution': [],
            'trending_topics': [],
            'breaking_news_count': 0
        }
        
        # Analyze categories
        for article in articles:
            if article.category:
                analysis['categories'][article.category] = analysis['categories'].get(article.category, 0) + 1
            
            if article.importance_score:
                analysis['importance_levels'].append(article.importance_score)
            
            if article.sentiment_score is not None:
                analysis['sentiment_distribution'].append(article.sentiment_score)
            
            # Check for breaking news indicators
            if any(keyword in article.title.lower() for keyword in ['breaking', 'urgent', 'alert']):
                analysis['breaking_news_count'] += 1
        
        # Extract trending topics from tags
        all_tags = []
        for article in articles:
            all_tags.extend(article.tags)
        
        # Count tag frequency
        tag_freq = {}
        for tag in all_tags:
            tag_freq[tag] = tag_freq.get(tag, 0) + 1
        
        # Get top trending topics
        analysis['trending_topics'] = sorted(tag_freq.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return analysis
    
    def _create_generation_plan(self, analysis: Dict) -> List[Dict]:
        """Create generation plan based on analysis"""
        plan = []
        max_articles = self.config.generation.max_articles_per_cycle
        
        # Determine article types to generate
        if analysis['breaking_news_count'] > 0:
            # Generate breaking news articles
            plan.append({
                'type': 'breaking',
                'priority': 'high',
                'count': min(3, analysis['breaking_news_count'])
            })
        
        # Generate analysis pieces for trending topics
        if analysis['trending_topics']:
            plan.append({
                'type': 'analysis',
                'priority': 'medium',
                'count': min(2, len(analysis['trending_topics']))
            })
        
        # Generate opinion pieces for controversial topics
        controversial_indicators = ['crisis', 'debate', 'controversy', 'conflict']
        if any(topic[0] in controversial_indicators for topic in analysis['trending_topics']):
            plan.append({
                'type': 'opinion',
                'priority': 'medium',
                'count': 1
            })
        
        # Generate summaries for complex topics
        if analysis['total_articles'] > 5:
            plan.append({
                'type': 'summary',
                'priority': 'low',
                'count': min(2, analysis['total_articles'] // 5)
            })
        
        # Limit total articles
        total_planned = sum(item['count'] for item in plan)
        if total_planned > max_articles:
            # Reduce counts proportionally
            reduction_factor = max_articles / total_planned
            for item in plan:
                item['count'] = max(1, int(item['count'] * reduction_factor))
        
        return plan
    
    async def _generate_single_article(self, plan_item: Dict, source_articles: List[NewsArticle]) -> Optional[GeneratedArticle]:
        """Generate a single article"""
        start_time = datetime.now()
        article_type = plan_item['type']
        
        try:
            # Get template
            template = self.template_system.get_template(article_type)
            
            # Select relevant source articles
            relevant_sources = self._select_relevant_sources(source_articles, article_type, plan_item)
            
            if not relevant_sources:
                logger.warning(f"No relevant sources for {article_type} article")
                return None
            
            # Create generation prompt
            prompt = self._create_generation_prompt(relevant_sources, template)
            
            # Generate content using neural mesh if available
            if self.neural_mesh and self.neural_mesh.is_initialized:
                content = await self._generate_with_neural_mesh(prompt, template)
            else:
                content = await self._generate_with_language_model(prompt, template)
            
            if not content:
                logger.warning(f"Failed to generate content for {article_type}")
                return None
            
            # Generate title
            title = await self._generate_title(content, article_type)
            
            # Optimize content
            optimized_content, seo_keywords = self.content_optimizer.optimize_for_seo(
                content, 
                relevant_sources[0].category if relevant_sources else None
            )
            
            # Apply stealth masking
            masking_result = await self.signature_masker.mask_ai_signatures(optimized_content)
            final_content = masking_result['masked_text']
            
            # Calculate metrics
            quality_score = self._calculate_quality_score(final_content, relevant_sources)
            readability_score = self.content_optimizer.calculate_readability_score(final_content)
            uniqueness_score = self._calculate_uniqueness_score(final_content, source_articles)
            
            # Create generated article
            generated_article = GeneratedArticle(
                title=title,
                content=final_content,
                article_type=article_type,
                source_articles=[article.hash_id for article in relevant_sources],
                generation_time=start_time,
                word_count=len(final_content.split()),
                quality_score=quality_score,
                seo_keywords=seo_keywords,
                readability_score=readability_score,
                uniqueness_score=uniqueness_score
            )
            
            generation_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"✅ Generated {article_type} article in {generation_time:.2f}s - Quality: {quality_score:.2f}")
            
            return generated_article
            
        except Exception as e:
            logger.error(f"Failed to generate {article_type} article: {e}")
            return None
    
    def _select_relevant_sources(self, articles: List[NewsArticle], article_type: str, plan_item: Dict) -> List[NewsArticle]:
        """Select relevant source articles for generation"""
        if article_type == 'breaking':
            # Select highest importance articles
            return sorted(articles, key=lambda x: x.importance_score or 0, reverse=True)[:3]
        
        elif article_type == 'analysis':
            # Select articles from same category or related topics
            if articles:
                primary_category = max(set(a.category for a in articles if a.category), 
                                     key=lambda x: sum(1 for a in articles if a.category == x))
                return [a for a in articles if a.category == primary_category][:5]
        
        elif article_type == 'opinion':
            # Select controversial or high-sentiment articles
            return sorted(articles, key=lambda x: abs(x.sentiment_score or 0), reverse=True)[:3]
        
        elif article_type == 'summary':
            # Select diverse articles
            return articles[:7]  # Take a broader sample
        
        return articles[:3]  # Default selection
    
    def _create_generation_prompt(self, sources: List[NewsArticle], template: Dict) -> str:
        """Create generation prompt from sources and template"""
        # Extract key information from sources
        titles = [article.title for article in sources]
        key_points = []
        
        for article in sources:
            # Extract first sentence as key point
            sentences = article.content.split('.')
            if sentences:
                key_points.append(sentences[0].strip())
        
        # Create prompt based on article type
        article_type = template.get('tone', 'informative')
        
        if article_type == 'urgent':
            prompt = f"Write a breaking news article about the following developments:\n"
        elif article_type == 'analytical':
            prompt = f"Write an in-depth analysis of the following situation:\n"
        elif article_type == 'persuasive':
            prompt = f"Write an opinion piece discussing:\n"
        else:
            prompt = f"Write a comprehensive summary of:\n"
        
        # Add source information
        for i, (title, point) in enumerate(zip(titles[:3], key_points[:3]), 1):
            prompt += f"{i}. {title}\n   Key point: {point}\n"
        
        prompt += f"\nWrite a {template['length_range'][0]}-{template['length_range'][1]} word article with a {article_type} tone:\n\n"
        
        return prompt
    
    async def _generate_with_neural_mesh(self, prompt: str, template: Dict) -> str:
        """Generate content using quantum neural mesh"""
        try:
            # Convert prompt to tensor
            prompt_tensor = torch.tensor([[ord(c) for c in prompt[:512]]], dtype=torch.float32)
            
            # Process through neural mesh
            result = await self.neural_mesh.process_distributed(prompt_tensor, 'inference')
            
            if result['success'] and result['final_output'] is not None:
                # Convert output back to text (simplified)
                output_tensor = result['final_output']
                # This is a simplified conversion - in practice, you'd use proper decoding
                generated_text = "Neural mesh generated content based on the provided sources."
                return generated_text
            else:
                # Fallback to language model
                return await self._generate_with_language_model(prompt, template)
                
        except Exception as e:
            logger.warning(f"Neural mesh generation failed, using language model: {e}")
            return await self._generate_with_language_model(prompt, template)
    
    async def _generate_with_language_model(self, prompt: str, template: Dict) -> str:
        """Generate content using language model"""
        constraints = {
            'max_length': template['length_range'][1],
            'min_length': template['length_range'][0],
            'tone': template['tone']
        }
        
        return await self.language_model.generate_with_constraints(prompt, constraints)
    
    async def _generate_title(self, content: str, article_type: str) -> str:
        """Generate compelling title for article"""
        # Extract first sentence for context
        first_sentence = content.split('.')[0] if content else ""
        
        # Create title generation prompt
        if article_type == 'breaking':
            title_prompt = f"Write a breaking news headline for: {first_sentence}"
        elif article_type == 'analysis':
            title_prompt = f"Write an analytical headline for: {first_sentence}"
        elif article_type == 'opinion':
            title_prompt = f"Write a provocative opinion headline for: {first_sentence}"
        else:
            title_prompt = f"Write a clear, informative headline for: {first_sentence}"
        
        # Generate title
        generated_title = await self.language_model.generate_text(title_prompt, max_length=50, temperature=0.9)
        
        # Clean and format title
        title = generated_title.split('\n')[0].strip()
        title = re.sub(r'^(Headline:|Title:)\s*', '', title, flags=re.IGNORECASE)
        
        # Ensure title is not too long
        if len(title) > 100:
            title = title[:97] + "..."
        
        return title or "Breaking News Update"
    
    def _calculate_quality_score(self, content: str, sources: List[NewsArticle]) -> float:
        """Calculate content quality score"""
        score = 0.0
        
        # Length score (0.2 weight)
        word_count = len(content.split())
        if 200 <= word_count <= 1000:
            length_score = 1.0
        elif word_count < 200:
            length_score = word_count / 200
        else:
            length_score = max(0.5, 1000 / word_count)
        
        score += length_score * 0.2
        
        # Coherence score (0.3 weight)
        sentences = content.split('.')
        if len(sentences) > 1:
            avg_sentence_length = np.mean([len(s.split()) for s in sentences if s.strip()])
            coherence_score = min(1.0, avg_sentence_length / 20)  # Optimal ~20 words per sentence
        else:
            coherence_score = 0.5
        
        score += coherence_score * 0.3
        
        # Relevance score (0.3 weight)
        if sources:
            source_keywords = set()
            for source in sources:
                source_keywords.update(source.tags)
            
            content_words = set(re.findall(r'\b\w+\b', content.lower()))
            relevance = len(source_keywords.intersection(content_words)) / max(len(source_keywords), 1)
            score += relevance * 0.3
        
        # Uniqueness score (0.2 weight)
        # Simple uniqueness check - in practice, you'd compare against existing content
        unique_words = len(set(content.lower().split()))
        total_words = len(content.split())
        uniqueness = unique_words / max(total_words, 1)
        score += uniqueness * 0.2
        
        return min(1.0, score)
    
    def _calculate_uniqueness_score(self, content: str, all_sources: List[NewsArticle]) -> float:
        """Calculate content uniqueness score"""
        content_words = set(re.findall(r'\b\w+\b', content.lower()))
        
        # Compare against source articles
        max_overlap = 0.0
        for source in all_sources:
            source_words = set(re.findall(r'\b\w+\b', source.content.lower()))
            if source_words:
                overlap = len(content_words.intersection(source_words)) / len(content_words.union(source_words))
                max_overlap = max(max_overlap, overlap)
        
        # Uniqueness is inverse of maximum overlap
        return 1.0 - max_overlap
    
    def _update_generation_stats(self, articles: List[GeneratedArticle]):
        """Update generation statistics"""
        if not articles:
            return
        
        self.generation_stats['total_generated'] += len(articles)
        
        # Update by type
        for article in articles:
            self.generation_stats['by_type'][article.article_type] += 1
        
        # Update averages
        quality_scores = [a.quality_score for a in articles]
        self.generation_stats['avg_quality_score'] = np.mean(quality_scores)
        
        # Calculate average generation time (simplified)
        self.generation_stats['avg_generation_time'] = 2.5  # Placeholder
    
    def get_generation_metrics(self) -> Dict:
        """Get comprehensive generation metrics"""
        return {
            'generation_stats': self.generation_stats,
            'is_initialized': self.is_initialized,
            'neural_mesh_available': self.neural_mesh is not None and self.neural_mesh.is_initialized,
            'language_model_device': str(self.language_model.device) if self.language_model.model else 'not_loaded'
        }
    
    async def shutdown(self):
        """Shutdown news generation engine"""
        logger.info("✍️ Shutting down News Generation Engine...")
        
        # Shutdown signature masker
        await self.signature_masker.shutdown()
        
        self.is_initialized = False
        logger.info("✅ News Generation Engine shutdown complete")
