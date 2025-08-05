# LLM Service with LangChain - Developer Guidelines

This document provides essential information for developers working on this LLM Service with LangChain project.
It covers build/configuration instructions, testing procedures, and additional development information.

## Build/Configuration Instructions

### Environment Setup

1. **Python Version**: This project requires Python 3.10 or higher.

2. **Dependencies Installation**:
   
   The project uses modern Python packaging with `pyproject.toml`. You can install dependencies using:
   
   ```bash
   # Using pip
   pip install -e .
   
   # Using uv (recommended for faster installation)
   uv pip install -e .
   ```

3. **API Keys Setup**:
   
   The project requires an OpenAI API key for most functionality. Set it as an environment variable:
   
   ```bash
   # Windows (PowerShell)
   $env:OPENAI_API_KEY = "your-api-key"
   
   # Windows (Command Prompt)
   set OPENAI_API_KEY=your-api-key
   
   # Linux/macOS
   export OPENAI_API_KEY="your-api-key"
   ```
   
   For persistent configuration, add this to your environment variables or use a `.env` file with a package like `python-dotenv`.

4. **Vector Database Setup**:
   
   The project uses ChromaDB for vector storage. No additional setup is required as it's included in the dependencies and creates local storage by default. For production deployments, consider configuring a persistent storage location.

## Testing Information

### Running Tests

The project uses standalone Python scripts for testing rather than a formal testing framework. Tests are located in the `LLM_practice` directory.

To run a test:

```bash
# Navigate to the project root
cd /path/to/LLM_Service_with_LangChain

# Run a specific test
python LLM_practice/simple_llm_test.py
```

### Creating New Tests

When creating new tests, follow these guidelines:

1. **Test Structure**:
   
   ```python
   import os
   # Import required LangChain components
   from langchain_openai import ChatOpenAI
   
   def test_your_functionality():
       """
       Docstring explaining the test purpose and requirements
       """
       try:
           # Test implementation
           # ...
           
           # Return True if test passes
           return True
       except Exception as e:
           print(f"Test failed with error: {e}")
           return False
   
   if __name__ == "__main__":
       result = test_your_functionality()
       print(f"Test {'passed' if result else 'failed'}.")
   ```

2. **Test Naming**:
   
   Name your test files descriptively, following the pattern `feature_test.py` (e.g., `llm_chain_test.py`, `embedding_test.py`).

3. **Test Data**:
   
   Place test data files in the `Data` directory. For small test data, you can include it directly in the test file.

### Example Test

Here's a simple test that demonstrates using LangChain with OpenAI:

```python
import os
from langchain_openai import ChatOpenAI

def test_simple_llm_response():
    """
    A simple test to demonstrate how to use LangChain with OpenAI.
    This test creates a ChatOpenAI instance and generates a response to a simple prompt.
    
    Requirements:
    - OpenAI API key set as an environment variable (OPENAI_API_KEY)
    
    Returns:
    - True if the test passes (response is generated successfully)
    - False if the test fails (error occurs)
    """
    try:
        # Get OpenAI API key from environment variable
        open_api_key = os.environ.get("OPENAI_API_KEY")
        
        if not open_api_key:
            print("Error: OPENAI_API_KEY environment variable not set.")
            return False
        
        # Initialize ChatOpenAI model
        llm = ChatOpenAI(
            model="gpt-3.5-turbo",  # Using a smaller model for testing
            api_key=open_api_key,
        )
        
        # Generate a response
        response = llm.invoke("What is LangChain?")
        
        # Print the response
        print("Test successful! Response:")
        print(response.content)
        
        return True
    
    except Exception as e:
        print(f"Test failed with error: {e}")
        return False

if __name__ == "__main__":
    print("Running simple LLM test...")
    result = test_simple_llm_response()
    print(f"Test {'passed' if result else 'failed'}.")
```

## Additional Development Information

### Project Structure

- **LLM_practice/**: Contains example scripts demonstrating various LLM functionalities
- **Data/**: Contains data files used by the examples
- **docs/**: Contains documentation, including a comprehensive memo on LLM concepts
- **실습/**: Contains practice exercises (Korean for "practice")

### Key Components

1. **LangChain Components**:
   
   The project uses several key LangChain components:
   
   - **Model I/O**: `ChatOpenAI`, `OpenAIEmbeddings`
   - **Data Connection**: `TextLoader`, `RecursiveCharacterTextSplitter`, `Chroma`
   - **Chains**: `load_qa_chain`
   - **Memory**: Used in conversational chatbots
   - **Agents/Tools**: Used in more advanced examples

2. **RAG Implementation**:
   
   The project implements RAG (Retrieval-Augmented Generation) using:
   
   - Document loading and splitting
   - Embedding generation
   - Vector storage in ChromaDB
   - Similarity search
   - Question answering with retrieved context

### Best Practices

1. **API Key Management**:
   
   Never hardcode API keys in your code. Always use environment variables or a secure configuration management system.

2. **Model Selection**:
   
   - Use smaller models (e.g., `gpt-3.5-turbo`) for testing and development
   - Use more capable models (e.g., `gpt-4o`) for production or when higher quality is needed
   - Consider using local models via HuggingFace for sensitive data or to reduce costs

3. **Chunking Strategy**:
   
   When implementing RAG, the chunking strategy significantly impacts performance:
   
   - Smaller chunks (e.g., 500-1000 characters) work better for precise information retrieval
   - Larger chunks (e.g., 1500-2000 characters) provide more context but may reduce precision
   - Experiment with chunk overlap (typically 10-20% of chunk size)

4. **Error Handling**:
   
   Implement robust error handling, especially for API calls that may fail due to rate limits, network issues, or invalid inputs.

5. **Testing Large Models**:
   
   When testing with large language models:
   
   - Use deterministic settings (temperature=0) for consistent results
   - Mock API responses for unit tests to avoid costs and dependencies
   - Create integration tests that verify the entire pipeline

### Common Issues and Solutions

1. **ImportError with LangChain**:
   
   LangChain has undergone significant changes in its package structure. If you encounter import errors:
   
   - Check the LangChain version (`pip show langchain`)
   - Update import statements according to the current version
   - For older code, you may need to use `langchain_community` instead of `langchain` for certain modules

2. **ChromaDB Persistence**:
   
   By default, ChromaDB creates an in-memory database. For persistence:
   
   ```python
   db = Chroma.from_documents(
       documents=docs,
       embedding=embeddings,
       persist_directory="./chroma_db"
   )
   ```

3. **Token Limits**:
   
   Be aware of token limits when working with large documents:
   
   - GPT-3.5-Turbo: ~16K tokens
   - GPT-4: ~8K tokens (older versions) to ~128K tokens (newer versions)
   - Implement proper chunking and summarization for large documents