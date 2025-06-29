
import asyncio
import logging
from datetime import datetime
from pathlib import Path
import random

logger = logging.getLogger(__name__)

class SimpleNewsGenerator:
    """Simplified but functional news generator"""
    
    def __init__(self):
        self.topics = [
            "Artificial Intelligence", "Technology", "Climate Change", "Healthcare",
            "Space Exploration", "Renewable Energy", "Cybersecurity", "Biotechnology"
        ]
        
        self.news_templates = {
            "breaking": [
                "Breaking: Major breakthrough in {topic} reported by leading researchers today.",
                "Urgent update: New developments in {topic} shake the industry.",
                "Alert: Significant advancement in {topic} announced this morning."
            ],
            "analysis": [
                "Deep dive: Understanding the implications of recent {topic} developments.",
                "Analysis: How {topic} trends are reshaping our future.",
                "Expert perspective: The long-term impact of {topic} innovations."
            ],
            "summary": [
                "Weekly roundup: Key {topic} developments you need to know.",
                "Summary: Recent {topic} news and what it means.",
                "Overview: The current state of {topic} in 2024."
            ]
        }
        
        self.content_templates = {
            "intro": [
                "Recent developments in {topic} have captured global attention.",
                "The {topic} sector continues to evolve at an unprecedented pace.",
                "Industry experts are closely monitoring {topic} trends."
            ],
            "body": [
                "Leading researchers have made significant progress in understanding how {topic} applications can transform various industries.",
                "The implications of these {topic} advancements extend far beyond initial expectations, potentially revolutionizing multiple sectors.",
                "Experts predict that {topic} innovations will play a crucial role in addressing current global challenges."
            ],
            "conclusion": [
                "As {topic} continues to advance, stakeholders must remain vigilant and adaptive to emerging trends.",
                "The future of {topic} appears promising, with continued investment and research driving innovation.",
                "These {topic} developments represent just the beginning of a larger transformation."
            ]
        }
    
    async def generate_article(self, topic=None, article_type="breaking"):
        """Generate a news article"""
        try:
            if not topic:
                topic = random.choice(self.topics)
            
            # Select templates
            title_template = random.choice(self.news_templates.get(article_type, self.news_templates["breaking"]))
            intro_template = random.choice(self.content_templates["intro"])
            body_template = random.choice(self.content_templates["body"])
            conclusion_template = random.choice(self.content_templates["conclusion"])
            
            # Generate content
            title = title_template.format(topic=topic)
            intro = intro_template.format(topic=topic)
            body = body_template.format(topic=topic)
            conclusion = conclusion_template.format(topic=topic)
            
            # Create full article
            article = f"""# {title}

*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*

## Overview

{intro}

## Key Developments

{body}

## Analysis

{conclusion}

---
*Article Type: {article_type.title()}*
*Topic: {topic}*
*Word Count: {len((intro + body + conclusion).split())}*
"""
            
            # Save to file
            output_dir = Path("output")
            output_dir.mkdir(exist_ok=True)
            
            filename = f"{article_type}_{topic.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
            file_path = output_dir / filename
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(article)
            
            logger.info(f"Generated article: {filename}")
            
            return {
                'success': True,
                'title': title,
                'filename': filename,
                'word_count': len((intro + body + conclusion).split()),
                'topic': topic,
                'type': article_type
            }
            
        except Exception as e:
            logger.error(f"Error generating article: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def generate_multiple_articles(self, count=3, topics=None, article_types=None):
        """Generate multiple articles"""
        if not topics:
            topics = random.sample(self.topics, min(count, len(self.topics)))
        if not article_types:
            article_types = random.choices(["breaking", "analysis", "summary"], k=count)
        
        results = []
        for i in range(count):
            topic = topics[i % len(topics)] if topics else None
            article_type = article_types[i % len(article_types)]
            
            result = await self.generate_article(topic, article_type)
            results.append(result)
            
            # Small delay between generations
            await asyncio.sleep(0.5)
        
        return results
