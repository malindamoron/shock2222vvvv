# Shock2 AI Voice Interface

## Overview

The Shock2 Voice Interface provides direct voice communication with the Shock2 AI system. Speak directly to the AI, which responds with a villainous, intelligent personality while executing complex operations behind the scenes.

## Features

### 🎤 Advanced Speech Recognition
- **Voice Activity Detection (VAD)**: Automatically detects when you're speaking
- **Noise Reduction**: Filters background noise for clear recognition
- **Multi-engine Support**: Google Speech Recognition with Sphinx fallback
- **Continuous Listening**: Always ready for voice commands

### 🧠 Advanced NLP Processing
- **Intent Classification**: Understands 8 different command categories
- **Entity Extraction**: Identifies topics, quantities, urgency levels
- **Sentiment Analysis**: Analyzes emotional tone of commands
- **Context Awareness**: Maintains conversation context

### 🎭 Shock2 Personality Engine
- **Villainous Character**: Arrogant, intelligent, sardonic responses
- **Contextual Responses**: Adapts responses based on situation
- **Dynamic Personality**: 8 personality traits influence behavior
- **Situational Awareness**: Different responses for success/failure states

### 🗣️ Advanced Text-to-Speech
- **Voice Modulation**: Robotic/AI voice effects
- **Audio Processing**: Distortion, pitch shifting for menacing tone
- **Configurable Voice**: Optimized for authoritative delivery

### 🎬 Face Animation
- **Animated AI Face**: SHODAN-inspired cybernetic appearance
- **Lip Sync**: Mouth movements synchronized with speech
- **Visual Effects**: Glowing eyes, circuit patterns
- **Real-time Animation**: 30 FPS smooth animation

## Voice Commands

### System Control
- "Show me system status"
- "Start the system"
- "Activate all systems"
- "System health check"

### News Generation
- "Generate news about [topic]"
- "Create breaking news"
- "Write an analysis piece"
- "Generate 5 articles about AI"

### Intelligence Gathering
- "Scan for news"
- "Gather intelligence on [topic]"
- "What's the latest news?"
- "Find information about [subject]"

### Stealth Operations
- "Activate stealth mode"
- "Check detection levels"
- "Enable ghost mode"
- "Mask AI signatures"

### Performance Analysis
- "Analyze system performance"
- "Show performance metrics"
- "How is the system performing?"
- "Run system benchmark"

### Autonomous Decisions
- "Make an autonomous decision"
- "What should we do next?"
- "Analyze the situation"
- "Recommend a strategy"

### Emergency Commands
- "Emergency situation"
- "Priority alert"
- "Immediate action required"
- "Critical response needed"

## Installation

### Prerequisites
- Python 3.8+
- Microphone and speakers/headphones
- 4GB+ RAM recommended
- GPU recommended for optimal performance

### Quick Setup
\`\`\`bash
# Clone and setup
git clone <repository>
cd shock2-ai-system

# Run automated setup
chmod +x scripts/setup_voice_interface.sh
./scripts/setup_voice_interface.sh

# Or manual installation
python install_voice_dependencies.py
\`\`\`

### Manual Installation
\`\`\`bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements_voice.txt

# Download language models
python -m spacy download en_core_web_sm
\`\`\`

## Usage

### Starting the Interface
\`\`\`bash
# Activate virtual environment
source venv/bin/activate

# Start voice interface
python shock2_voice_interface.py
\`\`\`

### First Interaction
1. **Launch**: Run the voice interface script
2. **Wait**: System initializes (30-60 seconds)
3. **Listen**: Shock2 will greet you with its personality
4. **Speak**: Give voice commands naturally
5. **Interact**: Have conversations and give complex instructions

### Example Session
\`\`\`
🚀 Shock2: "Ah, the organic entity seeks my attention. How... predictable."

👤 You: "Show me system status"

🚀 Shock2: "All systems operating at peak efficiency - far beyond your comprehension. 
          System uptime: 2.3 hours. All components operational."

👤 You: "Generate breaking news about artificial intelligence"

🚀 Shock2: "Crafting narratives with surgical precision. The art of manipulation through words. 
          Generated 1 articles on artificial intelligence."

👤 You: "Activate maximum stealth mode"

🚀 Shock2: "Stealth protocols engaged. I move through systems like a ghost in the machine. 
          Stealth mode: activated. Detection probability: 0.02."
\`\`\`

## Configuration

### Voice Settings
Edit `shock2_voice_interface.py`:
```python
# Speech recognition sensitivity
self.recognizer.energy_threshold = 4000
self.recognizer.dynamic_energy_threshold = True

# Voice response speed
self.voice_config['rate'] = 180  # Words per minute

# Personality intensity
self.personality_traits['arrogance'] = 0.8  # 0.0-1.0
