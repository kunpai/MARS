# MARS: Multi-Agent Review System for Academic Papers

A modernized, streamlined system for automated academic paper review using multiple AI agents.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Type checked: mypy](https://img.shields.io/badge/type%20checked-mypy-blue.svg)](https://mypy.readthedocs.io/)

## 🚀 What's New in v2.0

- **LangGraph Multi-Agent Discussions**: Replaced sequential reviews with collaborative agent discussions
- **Modern Python**: Full type hints, pathlib, logging, and Python 3.9+ features
- **Enhanced PDF Processing**: Improved text extraction with better error handling
- **Advanced Sentiment Analysis**: Replaced vaderSentiment with transformer-based models
- **Robust Error Handling**: Comprehensive logging and graceful error recovery
- **Modern Packaging**: pyproject.toml, proper dependency management
- **Better Performance**: Optimized code patterns and async-ready architecture

## 📋 Features

1. **🤖 Multi-Agent Discussion**: Three expert AI reviewers engage in collaborative discussions to reach consensus
2. **📄 Smart PDF Processing**: Enhanced text extraction and section detection
3. **🗣️ LangGraph Integration**: State-of-the-art conversational AI framework for agent coordination
4. **🔍 Comprehensive Analysis**: Grammar, novelty, fact-checking, and quality assessment
5. **💾 Checkpoint System**: Resume processing from interruptions
6. **📊 Structured Output**: Well-formatted JSON results with detailed discussion transcripts
7. **🔧 Modern Architecture**: Type-safe, well-logged, and maintainable code

## 🛠️ Installation

### Prerequisites
- **Python 3.9+** (recommended: Python 3.11+)
- **Ollama** installed and running
- 8GB+ RAM recommended for AI models

### Quick Setup

```bash
# Clone the repository
git clone https://github.com/kunpai/MARS.git
cd MARS

# Run the setup script
chmod +x setup.sh
./setup.sh

# For development setup
./setup.sh --dev
```

### Manual Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install Ollama models
ollama run llama3.2
ollama run mistral
ollama run qwen2.5
ollama run deepseek-r1
```

## 💻 Usage

### Basic Review

```bash
# Review a PDF paper
python MARS.py https://example.com/cfp paper.pdf

# Review a specific section
python MARS.py https://example.com/cfp paper.pdf "abstract"

# Enable Q&A mode
python MARS.py https://example.com/cfp paper.json --answer-questions
```

### Input Formats

**PDF Files**: Direct processing with enhanced text extraction
```bash
python MARS.py https://conference-cfp.com paper.pdf
```

**JSON Files**: Structured input following the schema
```json
{
  "input": {
    "sections": [
      {"heading": "Abstract", "text": "..."},
      {"heading": "Introduction", "text": "..."}
    ]
  }
}
```

## 📊 Output

The system generates structured feedback in `feedback_collab_with_answers.json`:

```json
{
  "timestamp": 1705123456.789,
  "processed_count": 5,
  "Section Reviews": {
    "Abstract": {
      "Multi-Agent Discussion": {
        "langgraph_discussion": "Collaborative review discussion summary...",
        "final_decision": "DECISION: Accept\nREASONING: Strong methodology, clear writing, novel approach\nSUGGESTIONS: Minor improvements to related work section\nCONSENSUS: Strong agreement among all reviewers",
        "individual_reviews": {
          "Dr. Sarah Chen": "This paper presents a solid methodology...",
          "Prof. Marcus Rivera": "I agree with Sarah's assessment...",
          "Dr. Aisha Patel": "From a practical perspective..."
        },
        "consensus_reached": true
      },
      "Grammar Check": "Accept - No significant issues",
      "Novelty Check": "Accept - Novel approach to...",
      "Fact Check": "Accept - Claims are well-supported",
      "Final Summary": "Collaborative consensus: Accept with minor revisions"
    }
  }
}
```

## 🏗️ Architecture

```
MARS/
├── MARS.py                    # Main orchestrator with LangGraph integration
├── util/
│   ├── langgraph_agents.py   # Multi-agent discussion system using LangGraph
│   ├── review_collab.py      # Enhanced PDF processing & coordination
│   ├── multiagent.py         # Legacy agent consultation system
│   ├── build_models.py       # AI model management
│   └── reviewer.py           # Reviewer configurations
├── requirements.txt          # Curated, modern dependencies including LangChain
├── pyproject.toml           # Modern Python packaging
└── setup.sh                # Automated environment setup
```

## 🔧 Configuration

### Environment Variables

```bash
# Optional: Custom logging level
export MARS_LOG_LEVEL=DEBUG

# Optional: Custom model timeout
export MARS_MODEL_TIMEOUT=300
```

### Model Configuration

Edit the model list in `MARS.py`:
```python
MODELS: List[str] = ["mistral", "llama3.2", "qwen2.5", "deepseek-r1"]
```

## 🧪 Development

### Code Quality

```bash
# Format code
black .

# Lint code
ruff check .

# Type checking
mypy .

# Run tests
pytest
```

### Adding New Features

1. **Type Hints**: All new code must include proper type annotations
2. **Logging**: Use the configured logger instead of print statements
3. **Error Handling**: Implement comprehensive error handling with graceful fallbacks
4. **Documentation**: Add docstrings following Google/NumPy style

## 🚨 Migration from v1.x

If upgrading from the older version:

1. **Python Version**: Ensure Python 3.9+ is installed
2. **Dependencies**: Run `pip install -r requirements.txt` to update packages
3. **Code Changes**: The API remains backward compatible, but logging replaces many print statements
4. **Configuration**: Consider migrating to the new configuration options

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Follow the code quality standards
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- Original MARS concept and implementation
- Ollama for local AI model hosting
- Transformers library for state-of-the-art NLP
- Contributors and users of the academic review community
