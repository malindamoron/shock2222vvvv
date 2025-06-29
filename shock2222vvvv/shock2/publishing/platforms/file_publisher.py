"""
Shock2 File Publisher
Publishes generated content to local files in various formats
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path
import markdown
import yaml

logger = logging.getLogger(__name__)


class FilePublisher:
    """File-based content publisher"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.output_dir = config.get('output_directory', 'output')
        self.formats = config.get('formats', ['markdown', 'json', 'html'])
        self.organize_by_category = config.get('organize_by_category', True)
        self.include_metadata = config.get('include_metadata', True)
        self.backup_enabled = config.get('backup_enabled', True)

        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

        # Publishing statistics
        self.stats = {
            'total_published': 0,
            'by_format': {},
            'by_category': {},
            'last_published': None
        }

    async def publish_article(self, article: Dict[str, Any]) -> Dict[str, Any]:
        """Publish a single article"""
        try:
            published_files = []

            # Prepare article data
            article_data = self._prepare_article_data(article)

            # Generate filename
            filename = self._generate_filename(article_data)

            # Determine output path
            output_path = self._get_output_path(article_data, filename)

            # Publish in each format
            for format_type in self.formats:
                file_path = await self._publish_format(article_data, output_path, format_type)
                if file_path:
                    published_files.append(file_path)

            # Update statistics
            self._update_stats(article_data)

            logger.info(
                f"? Published article: {article_data.get('title', 'Untitled')} in {len(published_files)} formats")

            return {
                'success': True,
                'published_files': published_files,
                'article_id': article_data.get('id'),
                'formats': self.formats
            }

        except Exception as e:
            logger.error(f"Failed to publish article: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    async def publish_batch(self, articles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Publish multiple articles"""
        results = []
        successful = 0
        failed = 0

        for article in articles:
            result = await self.publish_article(article)
            results.append(result)

            if result['success']:
                successful += 1
            else:
                failed += 1

        # Create batch summary
        summary = {
            'total_articles': len(articles),
            'successful': successful,
            'failed': failed,
            'results': results,
            'batch_id': f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        }

        # Save batch summary
        await self._save_batch_summary(summary)

        logger.info(f"? Published batch: {successful}/{len(articles)} articles successful")

        return summary

    def _prepare_article_data(self, article: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare article data for publishing"""
        # Generate unique ID if not present
        if 'id' not in article:
            article['id'] = f"article_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(article.get('title', ''))}"

        # Add publishing metadata
        if self.include_metadata:
            article['metadata'] = {
                'published_at': datetime.now().isoformat(),
                'publisher': 'Shock2FilePublisher',
                'version': '1.0',
                'formats': self.formats
            }

        # Ensure required fields
        article.setdefault('title', 'Untitled Article')
        article.setdefault('content', '')
        article.setdefault('category', 'general')
        article.setdefault('tags', [])
        article.setdefault('author', 'Shock2 AI')

        return article

    def _generate_filename(self, article: Dict[str, Any]) -> str:
        """Generate filename for article"""
        # Clean title for filename
        title = article.get('title', 'untitled')
        clean_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_')).rstrip()
        clean_title = clean_title.replace(' ', '_').lower()[:50]  # Limit length

        # Add timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        return f"{timestamp}_{clean_title}"

    def _get_output_path(self, article: Dict[str, Any], filename: str) -> str:
        """Get output path for article"""
        if self.organize_by_category:
            category = article.get('category', 'general')
            category_dir = os.path.join(self.output_dir, category)
            os.makedirs(category_dir, exist_ok=True)
            return os.path.join(category_dir, filename)
        else:
            return os.path.join(self.output_dir, filename)

    async def _publish_format(self, article: Dict[str, Any], base_path: str, format_type: str) -> Optional[str]:
        """Publish article in specific format"""
        try:
            if format_type == 'markdown':
                return await self._publish_markdown(article, base_path)
            elif format_type == 'json':
                return await self._publish_json(article, base_path)
            elif format_type == 'html':
                return await self._publish_html(article, base_path)
            elif format_type == 'txt':
                return await self._publish_txt(article, base_path)
            else:
                logger.warning(f"Unknown format type: {format_type}")
                return None

        except Exception as e:
            logger.error(f"Failed to publish {format_type} format: {e}")
            return None

    async def _publish_markdown(self, article: Dict[str, Any], base_path: str) -> str:
        """Publish article as Markdown"""
        file_path = f"{base_path}.md"

        # Create Markdown content
        content = []

        # Title
        content.append(f"# {article['title']}\n")

        # Metadata
        if self.include_metadata and 'metadata' in article:
            content.append("---")
            content.append(yaml.dump(article['metadata'], default_flow_style=False))
            content.append("---\n")

        # Article info
        content.append(f"**Author:** {article.get('author', 'Unknown')}")
        content.append(f"**Category:** {article.get('category', 'General')}")

        if article.get('tags'):
            tags_str = ", ".join(article['tags'])
            content.append(f"**Tags:** {tags_str}")

        content.append("")  # Empty line

        # Content
        content.append(article['content'])

        # Write to file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(content))

        return file_path

    async def _publish_json(self, article: Dict[str, Any], base_path: str) -> str:
        """Publish article as JSON"""
        file_path = f"{base_path}.json"

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(article, f, indent=2, ensure_ascii=False, default=str)

        return file_path

    async def _publish_html(self, article: Dict[str, Any], base_path: str) -> str:
        """Publish article as HTML"""
        file_path = f"{base_path}.html"

        # Convert Markdown content to HTML
        html_content = markdown.markdown(article['content'])

        # Create full HTML document
        html_template = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{article['title']}</title>
    <style>
        body {{ font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }}
        .metadata {{ background: #f5f5f5; padding: 10px; margin-bottom: 20px; border-radius: 5px; }}
        .tags {{ margin-top: 10px; }}
        .tag {{ background: #007cba; color: white; padding: 2px 8px; border-radius: 3px; margin-right: 5px; }}
    </style>
</head>
<body>
    <h1>{article['title']}</h1>

    <div class="metadata">
        <p><strong>Author:</strong> {article.get('author', 'Unknown')}</p>
        <p><strong>Category:</strong> {article.get('category', 'General')}</p>
        {f'<div class="tags"><strong>Tags:</strong> {" ".join([f"<span class=\\"tag\\">{tag}</span>" for tag in article.get("tags", [])])}</div>' if article.get('tags') else ''}
    </div>

    <div class="content">
        {html_content}
    </div>

    {f'<div class="metadata"><small>Published: {article.get("metadata", {}).get("published_at", "Unknown")}</small></div>' if self.include_metadata else ''}
</body>
</html>
        """

        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(html_template.strip())

        return file_path

    async def _publish_txt(self, article: Dict[str, Any], base_path: str) -> str:
        """Publish article as plain text"""
        file_path = f"{base_path}.txt"

        content = []
        content.append(f"Title: {article['title']}")
        content.append(f"Author: {article.get('author', 'Unknown')}")
        content.append(f"Category: {article.get('category', 'General')}")

        if article.get('tags'):
            content.append(f"Tags: {', '.join(article['tags'])}")

        content.append("")  # Empty line
        content.append(article['content'])

        with open(file_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(content))

        return file_path

    async def _save_batch_summary(self, summary: Dict[str, Any]):
        """Save batch publishing summary"""
        summary_path = os.path.join(self.output_dir, 'batch_summaries')
        os.makedirs(summary_path, exist_ok=True)

        filename = f"{summary['batch_id']}.json"
        file_path = os.path.join(summary_path, filename)

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, default=str)

    def _update_stats(self, article: Dict[str, Any]):
        """Update publishing statistics"""
        self.stats['total_published'] += 1
        self.stats['last_published'] = datetime.now().isoformat()

        # Update format stats
        for format_type in self.formats:
            if format_type not in self.stats['by_format']:
                self.stats['by_format'][format_type] = 0
            self.stats['by_format'][format_type] += 1

        # Update category stats
        category = article.get('category', 'general')
        if category not in self.stats['by_category']:
            self.stats['by_category'][category] = 0
        self.stats['by_category'][category] += 1

    def get_stats(self) -> Dict[str, Any]:
        """Get publishing statistics"""
        return self.stats.copy()

    async def cleanup_old_files(self, days_old: int = 30):
        """Clean up old published files"""
        cutoff_time = datetime.now().timestamp() - (days_old * 24 * 3600)
        cleaned_count = 0

        for root, dirs, files in os.walk(self.output_dir):
            for file in files:
                file_path = os.path.join(root, file)
                if os.path.getmtime(file_path) < cutoff_time:
                    try:
                        os.remove(file_path)
                        cleaned_count += 1
                    except Exception as e:
                        logger.warning(f"Failed to remove old file {file_path}: {e}")

        logger.info(f"? Cleaned up {cleaned_count} old files")
        return cleaned_count
