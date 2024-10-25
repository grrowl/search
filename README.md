# Smart Chat Assistant with Memory

A Streamlit-based chat application that uses Claude AI, with memory management and web search capabilities.

## Features

- Conversational AI powered by Claude 3 Opus
- Persistent memory system with importance scoring
- Web search integration via DuckDuckGo
- Voting system for memory importance
- Chat history persistence
- Clean, intuitive UI with Streamlit

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
- Create a `.env` file with your API keys:
```
ANTHROPIC_API_KEY=your_api_key_here
```

## Project Structure

```
.
├── src/
│   ├── __init__.py
│   └── app.py          # Main application code
├── tests/
│   └── __init__.py     # Test files
├── data/               # Data storage directory
├── requirements.txt    # Project dependencies
├── README.md          # This file
└── .gitignore         # Git ignore rules
```

## Running the Application

```bash
streamlit run src/app.py
```

## Dependencies

Key dependencies include:
- streamlit
- anthropic
- pandas
- duckduckgo-search
- tiktoken

## Features

### Memory Management
- Stores conversation history with importance scoring
- Voting system to adjust memory importance
- Automatic memory pruning based on token limits

### Web Search Integration
- Real-time web search via DuckDuckGo
- Incorporates search results into AI responses
- Configurable search depth

### Chat Interface
- Clean, responsive UI
- Persistent chat history
- Real-time response streaming
- Memory visualization and management

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## License

MIT License
