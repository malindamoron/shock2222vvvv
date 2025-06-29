"""
Shock2 Signature Masker - Advanced AI Detection Evasion
Implements sophisticated techniques to mask AI-generated content signatures
"""

import re
import random
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any
import asyncio
import logging
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModel
import spacy
from collections import defaultdict
import hashlib
import time

logger = logging.getLogger(__name__)

@dataclass
class SignaturePattern:
    """AI signature pattern definition"""
    pattern_type: str  # 'repetition', 'structure', 'vocabulary', 'style'
    pattern_regex: str
    confidence_threshold: float
    masking_strategy: str

class LinguisticAnalyzer:
    """Advanced linguistic analysis for AI detection"""
    
    def __init__(self):
        self.nlp = None
        self.tokenizer = None
        self.model = None
        self.ai_patterns = []
        self.human_patterns = []
        
    async def initialize(self):
        """Initialize linguistic analyzer"""
        logger.info("🔍 Initializing Linguistic Analyzer...")
        
        # Load spaCy model
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy model not found, using basic analysis")
            self.nlp = None
        
        # Load transformer model for embeddings
        try:
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            self.model = AutoModel.from_pretrained("bert-base-uncased")
        except Exception as e:
            logger.warning(f"Could not load BERT model: {e}")
        
        # Define AI signature patterns
        self.ai_patterns = [
            SignaturePattern(
                pattern_type="repetition",
                pattern_regex=r'\b(\w+)\s+\1\b',  # Word repetition
                confidence_threshold=0.7,
                masking_strategy="synonym_replacement"
            ),
            SignaturePattern(
                pattern_type="structure",
                pattern_regex=r'^(Furthermore|Moreover|Additionally|In conclusion),',
                confidence_threshold=0.8,
                masking_strategy="transition_variation"
            ),
            SignaturePattern(
                pattern_type="vocabulary",
                pattern_regex=r'\b(utilize|facilitate|implement|comprehensive|significant)\b',
                confidence_threshold=0.6,
                masking_strategy="vocabulary_simplification"
            ),
            SignaturePattern(
                pattern_type="style",
                pattern_regex=r'\. [A-Z][^.]*\. [A-Z][^.]*\. [A-Z][^.]*\.',  # Repetitive sentence structure
                confidence_threshold=0.75,
                masking_strategy="sentence_restructuring"
            )
        ]
        
        logger.info("✅ Linguistic Analyzer initialized")
    
    def analyze_text_patterns(self, text: str) -> Dict[str, float]:
        """Analyze text for AI signature patterns"""
        pattern_scores = {}
        
        for pattern in self.ai_patterns:
            matches = re.findall(pattern.pattern_regex, text, re.IGNORECASE)
            match_count = len(matches)
            
            # Calculate pattern strength
            text_length = len(text.split())
            pattern_density = match_count / max(text_length / 100, 1)  # Per 100 words
            
            pattern_scores[pattern.pattern_type] = min(pattern_density, 1.0)
        
        return pattern_scores
    
    def calculate_perplexity(self, text: str) -> float:
        """Calculate text perplexity (AI-generated text often has lower perplexity)"""
        if not self.tokenizer or not self.model:
            return 0.5  # Default neutral score
        
        try:
            # Tokenize text
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Simplified perplexity calculation
                logits = outputs.last_hidden_state
                perplexity = torch.exp(torch.mean(torch.log(torch.softmax(logits, dim=-1) + 1e-10)))
                
            return float(perplexity)
        except Exception as e:
            logger.warning(f"Perplexity calculation failed: {e}")
            return 0.5
    
    def analyze_sentence_complexity(self, text: str) -> Dict[str, float]:
        """Analyze sentence complexity patterns"""
        if not self.nlp:
            return {"complexity_score": 0.5}
        
        doc = self.nlp(text)
        sentences = list(doc.sents)
        
        if not sentences:
            return {"complexity_score": 0.5}
        
        # Calculate various complexity metrics
        avg_sentence_length = np.mean([len(sent.text.split()) for sent in sentences])
        avg_dependency_depth = np.mean([self._calculate_dependency_depth(sent) for sent in sentences])
        
        # AI-generated text often has more uniform complexity
        sentence_lengths = [len(sent.text.split()) for sent in sentences]
        length_variance = np.var(sentence_lengths) if len(sentence_lengths) > 1 else 0
        
        complexity_score = {
            "avg_sentence_length": avg_sentence_length,
            "avg_dependency_depth": avg_dependency_depth,
            "length_variance": length_variance,
            "complexity_score": (avg_dependency_depth * length_variance) / max(avg_sentence_length, 1)
        }
        
        return complexity_score
    
    def _calculate_dependency_depth(self, sentence) -> int:
        """Calculate maximum dependency depth in sentence"""
        def get_depth(token, current_depth=0):
            if not list(token.children):
                return current_depth
            return max(get_depth(child, current_depth + 1) for child in token.children)
        
        return max(get_depth(token) for token in sentence if token.dep_ == "ROOT")

class StyleMimicry:
    """Advanced style mimicry for human-like writing"""
    
    def __init__(self):
        self.human_writing_patterns = {}
        self.style_templates = {}
        self.vocabulary_replacements = {}
        
    async def initialize(self):
        """Initialize style mimicry system"""
        logger.info("🎭 Initializing Style Mimicry...")
        
        # Load human writing patterns
        self.human_writing_patterns = {
            "contractions": ["don't", "won't", "can't", "shouldn't", "wouldn't", "couldn't"],
            "informal_transitions": ["So", "Well", "Now", "But", "And", "Plus"],
            "hedging_words": ["maybe", "perhaps", "probably", "likely", "seems", "appears"],
            "personal_pronouns": ["I", "we", "you", "my", "our", "your"],
            "colloquialisms": ["pretty much", "kind of", "sort of", "a bit", "quite a few"]
        }
        
        # Vocabulary replacements (AI -> Human)
        self.vocabulary_replacements = {
            "utilize": ["use", "employ", "apply"],
            "facilitate": ["help", "enable", "make easier"],
            "implement": ["put in place", "carry out", "execute"],
            "comprehensive": ["complete", "thorough", "full"],
            "significant": ["important", "major", "big"],
            "furthermore": ["also", "plus", "what's more"],
            "moreover": ["besides", "on top of that", "additionally"],
            "subsequently": ["then", "after that", "next"],
            "consequently": ["so", "as a result", "therefore"]
        }
        
        logger.info("✅ Style Mimicry initialized")
    
    def apply_human_patterns(self, text: str) -> str:
        """Apply human writing patterns to text"""
        modified_text = text
        
        # Add contractions
        modified_text = self._add_contractions(modified_text)
        
        # Replace formal vocabulary
        modified_text = self._replace_formal_vocabulary(modified_text)
        
        # Add informal transitions
        modified_text = self._add_informal_transitions(modified_text)
        
        # Add hedging language
        modified_text = self._add_hedging(modified_text)
        
        # Add personal touches
        modified_text = self._add_personal_touches(modified_text)
        
        return modified_text
    
    def _add_contractions(self, text: str) -> str:
        """Add contractions to make text more human-like"""
        contractions_map = {
            "do not": "don't",
            "will not": "won't",
            "cannot": "can't",
            "should not": "shouldn't",
            "would not": "wouldn't",
            "could not": "couldn't",
            "is not": "isn't",
            "are not": "aren't",
            "was not": "wasn't",
            "were not": "weren't",
            "have not": "haven't",
            "has not": "hasn't",
            "had not": "hadn't"
        }
        
        for formal, contraction in contractions_map.items():
            # Apply contractions probabilistically
            if random.random() < 0.7:  # 70% chance to apply contraction
                text = re.sub(r'\b' + formal + r'\b', contraction, text, flags=re.IGNORECASE)
        
        return text
    
    def _replace_formal_vocabulary(self, text: str) -> str:
        """Replace formal vocabulary with more casual alternatives"""
        for formal_word, alternatives in self.vocabulary_replacements.items():
            pattern = r'\b' + formal_word + r'\b'
            matches = re.finditer(pattern, text, re.IGNORECASE)
            
            for match in matches:
                if random.random() < 0.6:  # 60% chance to replace
                    replacement = random.choice(alternatives)
                    # Preserve original case
                    if match.group().isupper():
                        replacement = replacement.upper()
                    elif match.group().istitle():
                        replacement = replacement.capitalize()
                    
                    text = text[:match.start()] + replacement + text[match.end():]
        
        return text
    
    def _add_informal_transitions(self, text: str) -> str:
        """Add informal transitions between sentences"""
        sentences = text.split('. ')
        modified_sentences = []
        
        for i, sentence in enumerate(sentences):
            if i > 0 and random.random() < 0.3:  # 30% chance to add informal transition
                transition = random.choice(self.human_writing_patterns["informal_transitions"])
                sentence = f"{transition}, {sentence.lower()}"
            
            modified_sentences.append(sentence)
        
        return '. '.join(modified_sentences)
    
    def _add_hedging(self, text: str) -> str:
        """Add hedging language to make statements less absolute"""
        # Find absolute statements
        absolute_patterns = [
            r'\bis\b',
            r'\bare\b',
            r'\bwill\b',
            r'\bmust\b',
            r'\balways\b',
            r'\bnever\b'
        ]
        
        for pattern in absolute_patterns:
            matches = list(re.finditer(pattern, text, re.IGNORECASE))
            for match in matches:
                if random.random() < 0.2:  # 20% chance to add hedging
                    hedge = random.choice(self.human_writing_patterns["hedging_words"])
                    text = text[:match.start()] + f"{hedge} " + text[match.start():]
        
        return text
    
    def _add_personal_touches(self, text: str) -> str:
        """Add personal touches to make text more human"""
        # Occasionally add personal perspective
        if random.random() < 0.1:  # 10% chance
            personal_intros = [
                "In my opinion, ",
                "I think ",
                "From what I've seen, ",
                "It seems to me that "
            ]
            intro = random.choice(personal_intros)
            text = intro + text[0].lower() + text[1:]
        
        return text

class SignatureMasker:
    """Main signature masking system"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.linguistic_analyzer = LinguisticAnalyzer()
        self.style_mimicry = StyleMimicry()
        self.masking_history = []
        self.detection_threshold = config.get('detection_threshold', 0.3)
        
    async def initialize(self):
        """Initialize signature masker"""
        logger.info("🕶️ Initializing Signature Masker...")
        
        await self.linguistic_analyzer.initialize()
        await self.style_mimicry.initialize()
        
        logger.info("✅ Signature Masker initialized - AI signatures masked")
    
    async def mask_ai_signatures(self, text: str, aggressiveness: float = 0.7) -> Dict[str, Any]:
        """Mask AI signatures in text"""
        logger.info("🎭 Masking AI signatures...")
        
        # Analyze original text
        original_analysis = await self._analyze_text(text)
        
        # Apply masking techniques
        masked_text = text
        masking_steps = []
        
        # Step 1: Style mimicry
        if original_analysis['ai_probability'] > self.detection_threshold:
            masked_text = self.style_mimicry.apply_human_patterns(masked_text)
            masking_steps.append("style_mimicry")
        
        # Step 2: Pattern disruption
        if original_analysis['pattern_scores']['repetition'] > 0.5:
            masked_text = self._disrupt_repetition_patterns(masked_text)
            masking_steps.append("repetition_disruption")
        
        # Step 3: Sentence restructuring
        if original_analysis['pattern_scores']['structure'] > 0.6:
            masked_text = self._restructure_sentences(masked_text)
            masking_steps.append("sentence_restructuring")
        
        # Step 4: Vocabulary diversification
        if original_analysis['pattern_scores']['vocabulary'] > 0.4:
            masked_text = self._diversify_vocabulary(masked_text)
            masking_steps.append("vocabulary_diversification")
        
        # Step 5: Perplexity adjustment
        if original_analysis['perplexity'] < 20:  # Low perplexity indicates AI
            masked_text = self._adjust_perplexity(masked_text)
            masking_steps.append("perplexity_adjustment")
        
        # Analyze masked text
        masked_analysis = await self._analyze_text(masked_text)
        
        # Calculate masking effectiveness
        effectiveness = self._calculate_masking_effectiveness(original_analysis, masked_analysis)
        
        result = {
            'original_text': text,
            'masked_text': masked_text,
            'original_analysis': original_analysis,
            'masked_analysis': masked_analysis,
            'masking_steps': masking_steps,
            'effectiveness': effectiveness,
            'detection_probability_reduction': original_analysis['ai_probability'] - masked_analysis['ai_probability']
        }
        
        # Store in history
        self.masking_history.append({
            'timestamp': time.time(),
            'effectiveness': effectiveness,
            'steps_used': masking_steps
        })
        
        logger.info(f"✅ AI signatures masked - Detection probability reduced by {result['detection_probability_reduction']:.2%}")
        
        return result
    
    async def _analyze_text(self, text: str) -> Dict[str, Any]:
        """Comprehensive text analysis"""
        # Pattern analysis
        pattern_scores = self.linguistic_analyzer.analyze_text_patterns(text)
        
        # Perplexity analysis
        perplexity = self.linguistic_analyzer.calculate_perplexity(text)
        
        # Complexity analysis
        complexity = self.linguistic_analyzer.analyze_sentence_complexity(text)
        
        # Calculate overall AI probability
        ai_probability = self._calculate_ai_probability(pattern_scores, perplexity, complexity)
        
        return {
            'pattern_scores': pattern_scores,
            'perplexity': perplexity,
            'complexity': complexity,
            'ai_probability': ai_probability,
            'text_length': len(text),
            'word_count': len(text.split())
        }
    
    def _calculate_ai_probability(self, patterns: Dict, perplexity: float, complexity: Dict) -> float:
        """Calculate probability that text is AI-generated"""
        # Weighted scoring
        pattern_weight = 0.4
        perplexity_weight = 0.3
        complexity_weight = 0.3
        
        # Pattern score (higher = more AI-like)
        pattern_score = np.mean(list(patterns.values()))
        
        # Perplexity score (lower perplexity = more AI-like)
        perplexity_score = max(0, (50 - perplexity) / 50)  # Normalize to 0-1
        
        # Complexity score (lower variance = more AI-like)
        complexity_score = max(0, (10 - complexity.get('length_variance', 5)) / 10)
        
        ai_probability = (
            pattern_score * pattern_weight +
            perplexity_score * perplexity_weight +
            complexity_score * complexity_weight
        )
        
        return min(ai_probability, 1.0)
    
    def _disrupt_repetition_patterns(self, text: str) -> str:
        """Disrupt repetitive patterns in text"""
        # Find and replace repetitive phrases
        words = text.split()
        modified_words = []
        
        i = 0
        while i < len(words):
            # Check for word repetition
            if i < len(words) - 1 and words[i].lower() == words[i + 1].lower():
                # Replace second occurrence with synonym or remove
                if random.random() < 0.7:
                    modified_words.append(words[i])
                    i += 2  # Skip the repeated word
                else:
                    modified_words.append(words[i])
                    i += 1
            else:
                modified_words.append(words[i])
                i += 1
        
        return ' '.join(modified_words)
    
    def _restructure_sentences(self, text: str) -> str:
        """Restructure sentences to break AI patterns"""
        sentences = text.split('. ')
        restructured = []
        
        for sentence in sentences:
            if len(sentence.strip()) == 0:
                continue
                
            # Randomly restructure some sentences
            if random.random() < 0.3:  # 30% chance
                restructured_sentence = self._restructure_single_sentence(sentence)
                restructured.append(restructured_sentence)
            else:
                restructured.append(sentence)
        
        return '. '.join(restructured)
    
    def _restructure_single_sentence(self, sentence: str) -> str:
        """Restructure a single sentence"""
        # Simple restructuring strategies
        strategies = [
            self._move_clause_to_beginning,
            self._split_compound_sentence,
            self._add_parenthetical_remark
        ]
        
        strategy = random.choice(strategies)
        return strategy(sentence)
    
    def _move_clause_to_beginning(self, sentence: str) -> str:
        """Move a clause to the beginning of the sentence"""
        # Look for clauses starting with common words
        clause_starters = ['because', 'since', 'although', 'while', 'when', 'if']
        
        for starter in clause_starters:
            pattern = f', {starter} ([^,]+)'
            match = re.search(pattern, sentence, re.IGNORECASE)
            if match:
                clause = match.group(1)
                main_part = sentence.replace(match.group(0), '')
                return f"{starter.capitalize()} {clause}, {main_part.strip()}"
        
        return sentence
    
    def _split_compound_sentence(self, sentence: str) -> str:
        """Split compound sentences"""
        # Look for coordinating conjunctions
        conjunctions = [', and ', ', but ', ', or ', ', so ']
        
        for conj in conjunctions:
            if conj in sentence:
                parts = sentence.split(conj, 1)
                if len(parts) == 2 and len(parts[1]) > 10:
                    return f"{parts[0].strip()}. {parts[1].strip().capitalize()}"
        
        return sentence
    
    def _add_parenthetical_remark(self, sentence: str) -> str:
        """Add parenthetical remarks to break flow"""
        remarks = [
            "of course",
            "naturally",
            "obviously",
            "clearly",
            "interestingly",
            "surprisingly"
        ]
        
        if random.random() < 0.5:
            remark = random.choice(remarks)
            # Insert after first clause
            words = sentence.split()
            if len(words) > 5:
                insert_pos = len(words) // 2
                words.insert(insert_pos, f"({remark})")
                return ' '.join(words)
        
        return sentence
    
    def _diversify_vocabulary(self, text: str) -> str:
        """Diversify vocabulary to reduce AI patterns"""
        # This would integrate with the style mimicry vocabulary replacement
        return self.style_mimicry._replace_formal_vocabulary(text)
    
    def _adjust_perplexity(self, text: str) -> str:
        """Adjust text to increase perplexity"""
        # Add slight variations and imperfections
        sentences = text.split('. ')
        adjusted = []
        
        for sentence in sentences:
            if random.random() < 0.2:  # 20% chance to adjust
                # Add filler words or slight redundancy
                fillers = ["well", "you know", "I mean", "basically", "actually"]
                filler = random.choice(fillers)
                
                words = sentence.split()
                if len(words) > 3:
                    insert_pos = random.randint(1, len(words) - 1)
                    words.insert(insert_pos, f"{filler},")
                    sentence = ' '.join(words)
            
            adjusted.append(sentence)
        
        return '. '.join(adjusted)
    
    def _calculate_masking_effectiveness(self, original: Dict, masked: Dict) -> float:
        """Calculate effectiveness of masking"""
        # Compare AI probabilities
        probability_reduction = original['ai_probability'] - masked['ai_probability']
        
        # Compare pattern scores
        pattern_improvement = 0
        for pattern_type in original['pattern_scores']:
            original_score = original['pattern_scores'][pattern_type]
            masked_score = masked['pattern_scores'].get(pattern_type, original_score)
            pattern_improvement += max(0, original_score - masked_score)
        
        pattern_improvement /= len(original['pattern_scores'])
        
        # Overall effectiveness
        effectiveness = (probability_reduction * 0.7 + pattern_improvement * 0.3)
        
        return max(0, min(1, effectiveness))
    
    def get_masking_statistics(self) -> Dict:
        """Get masking performance statistics"""
        if not self.masking_history:
            return {"status": "no_data"}
        
        recent_history = self.masking_history[-100:]  # Last 100 operations
        
        avg_effectiveness = np.mean([h['effectiveness'] for h in recent_history])
        most_common_steps = defaultdict(int)
        
        for history in recent_history:
            for step in history['steps_used']:
                most_common_steps[step] += 1
        
        return {
            "total_operations": len(self.masking_history),
            "recent_operations": len(recent_history),
            "average_effectiveness": avg_effectiveness,
            "most_common_steps": dict(most_common_steps),
            "detection_threshold": self.detection_threshold
        }
    
    async def shutdown(self):
        """Shutdown signature masker"""
        logger.info("🕶️ Signature Masker shutdown complete")
