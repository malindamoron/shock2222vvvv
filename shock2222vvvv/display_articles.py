
#!/usr/bin/env python3
"""
Article Display Utility - Show generated articles
"""

import sys
from pathlib import Path
from datetime import datetime

def display_articles():
    """Display all generated articles"""
    output_dir = Path("output")
    
    if not output_dir.exists():
        print("❌ No output directory found. No articles have been generated yet.")
        return
    
    article_files = list(output_dir.glob("*.md"))
    
    if not article_files:
        print("❌ No articles found in output directory.")
        return
    
    # Sort by modification time
    article_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    print(f"\n📰 Found {len(article_files)} Generated Articles")
    print("=" * 60)
    
    for i, file_path in enumerate(article_files, 1):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')
                title = lines[0].replace('#', '').strip() if lines else file_path.stem
                
                # Extract metadata
                word_count = len(content.split())
                mod_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                
                print(f"{i}. {title}")
                print(f"   📁 File: {file_path.name}")
                print(f"   📅 Created: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"   📊 Words: {word_count}")
                print()
                
        except Exception as e:
            print(f"   ❌ Error reading {file_path.name}: {e}")
    
    print("💡 Tip: You can also ask Shock2 to 'show me the articles' via voice command")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--latest":
        # Show content of latest article
        output_dir = Path("output")
        if output_dir.exists():
            article_files = list(output_dir.glob("*.md"))
            if article_files:
                latest = max(article_files, key=lambda x: x.stat().st_mtime)
                print(f"\n📰 Latest Article: {latest.name}")
                print("=" * 60)
                with open(latest, 'r', encoding='utf-8') as f:
                    print(f.read())
    else:
        display_articles()
