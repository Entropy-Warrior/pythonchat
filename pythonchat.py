import requests
import json
import os
from datetime import datetime, timedelta
from rich.console import Console
from rich.markdown import Markdown
from rich.prompt import Prompt
from rich.table import Table
from rich.text import Text
from rich.progress import Progress
from rich.panel import Panel
from rich.live import Live
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.embeddings import Embeddings
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from typing import Any, Dict, List, Optional
import tiktoken
from pydantic import BaseModel, Field
import base64
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from getpass import getpass

# Initialize Rich console for better formatting
console = Console()

# Debug control
DEBUG_MODE = False

def set_debug(enabled: bool):
    """Enable or disable debug output."""
    global DEBUG_MODE
    DEBUG_MODE = enabled
    console.print(f"[yellow]Debug mode {'enabled' if enabled else 'disabled'}[/yellow]")

def debug_print(message: str):
    """Print debug message if debug mode is enabled."""
    if DEBUG_MODE:
        console.print(f"[yellow]Debug: {message}[/yellow]")

# Encrypted API key (will be replaced with actual encrypted key)
ENCRYPTED_API_KEY = "gAAAAABnmEo__06SYg3CsTS3uNbJt_5OvUS7Tt4qjInT_fwWE88b8ihOzrkVUP5GKbxZpyHmGpYSn4halGuvUik2ypooRfjMWtZhylglkUpnGXu9HFmmuyrzj9go4Ils2c0cg5GNNI5ZrCucYqfiZRixisQCT2CmLPsv8RKv8YEms5N3zkbgevA="
SALT = b'\xf8S\x92\xfa\xb6\x16\xe1$\xd2\x9a*`\xe2_~D'  # Generated salt value

def get_encryption_key(password: str) -> bytes:
    """Derive encryption key from password."""
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=SALT,
        iterations=480000,
    )
    key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
    return key

def decrypt_api_key(password: str) -> str:
    """Decrypt the API key using the provided password."""
    try:
        key = get_encryption_key(password)
        f = Fernet(key)
        decrypted_key = f.decrypt(ENCRYPTED_API_KEY.encode()).decode()
        return decrypted_key
    except Exception as e:
        console.print("[red]Failed to decrypt API key. Wrong password?[/red]")
        raise Exception("Decryption failed")

class HybridMemory:
    """Hybrid memory system using vector store and summarization."""
    
    def __init__(self, embedding_model=None, max_tokens=4000):
        self.max_tokens = max_tokens
        self.messages = []
        
        # Set up model path
        models_dir = os.getenv('MODELS_DIR', 'storage/models')
        default_model_path = os.path.join(models_dir, 'all-MiniLM-L6-v2')
        
        debug_print(f"Initializing HybridMemory with models_dir: {models_dir}")
        debug_print(f"Default model path: {default_model_path}")
        
        # Try to use local model first
        if embedding_model is None:
            if os.path.exists(default_model_path):
                embedding_model = default_model_path
                debug_print(f"Using local model from: {default_model_path}")
            else:
                embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
                debug_print("Local model not found, using remote model")
                # Create the models directory if it doesn't exist
                os.makedirs(models_dir, exist_ok=True)
        
        # Initialize embeddings with more error handling
        try:
            debug_print(f"Attempting to initialize HuggingFaceEmbeddings with model: {embedding_model}")
            self.embeddings = HuggingFaceEmbeddings(
                model_name=embedding_model,
                cache_folder=models_dir,
                encode_kwargs={'normalize_embeddings': True}  # Add this for better stability
            )
            debug_print("HuggingFaceEmbeddings initialized successfully")
        except Exception as e:
            console.print(f"[red]Error loading embeddings model: {str(e)}[/red]")
            debug_print(f"Embeddings initialization error: {str(e)}")
            raise
            
        # Set up vector store with more error handling
        try:
            history_dir = os.getenv('HISTORY_DIR', 'storage/history')
            persist_dir = os.path.join(history_dir, '.chat_memory')
            os.makedirs(persist_dir, exist_ok=True)
            debug_print(f"Setting up Chroma with persist_directory: {persist_dir}")
            
            self.vector_store = Chroma(
                collection_name="chat_memory",
                embedding_function=self.embeddings,
                persist_directory=persist_dir
            )
            debug_print("Chroma vector store initialized successfully")
        except Exception as e:
            console.print(f"[red]Error initializing vector store: {str(e)}[/red]")
            debug_print(f"Vector store initialization error: {str(e)}")
            raise
        
        self.encoding = tiktoken.get_encoding("cl100k_base")
        debug_print("HybridMemory initialization complete")
        
    def add_message(self, message):
        """Add a message to both vector store and message list."""
        self.messages.append(message)
        
        # Add to vector store
        if isinstance(message, (HumanMessage, AIMessage)):
            self.vector_store.add_texts(
                texts=[message.content],
                metadatas=[{
                    "type": "human" if isinstance(message, HumanMessage) else "ai",
                    "timestamp": datetime.now().isoformat()
                }]
            )
            
    def get_relevant_history(self, query: str, k: int = 5) -> List[str]:
        """Get most relevant messages from vector store."""
        try:
            # Get total number of documents in the collection
            collection = self.vector_store._collection
            doc_count = collection.count()
            
            # Adjust k to not exceed document count
            k = min(k, doc_count) if doc_count > 0 else 1
            
            debug_print(f"Searching for {k} relevant messages from {doc_count} total messages")
            
            if doc_count == 0:
                debug_print("No messages in vector store")
                return []
                
            results = self.vector_store.similarity_search(query, k=k)
            return [doc.page_content for doc in results]
        except Exception as e:
            debug_print(f"Error retrieving history: {str(e)}")
            return []
        
    def get_token_count(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.encoding.encode(text))
        
    def get_messages(self, query: str = None) -> List[Any]:
        """Get messages for context, using hybrid approach."""
        # Always include system message if present
        final_messages = []
        if self.messages and isinstance(self.messages[0], SystemMessage):
            final_messages.append(self.messages[0])
            
        if query:
            # Get relevant historical context
            relevant_history = self.get_relevant_history(query)
            
            # Convert relevant history to messages
            for text in relevant_history:
                # Add as AI message since we don't store the role in vector store
                final_messages.append(AIMessage(content=text))
        
        # Add recent messages up to token limit
        token_count = sum(self.get_token_count(msg.content) for msg in final_messages)
        
        for message in reversed(self.messages):
            if not isinstance(message, SystemMessage):  # Skip system message as it's already added
                msg_tokens = self.get_token_count(message.content)
                if token_count + msg_tokens <= self.max_tokens:
                    final_messages.append(message)
                    token_count += msg_tokens
                else:
                    break
                    
        return final_messages
        
    def clear(self):
        """Clear all memory."""
        self.messages = []
        self.vector_store.delete_collection()
        self.vector_store = Chroma(
            collection_name="chat_memory",
            embedding_function=self.embeddings,
            persist_directory=os.path.join(os.getenv('HISTORY_DIR', 'storage/history'), '.chat_memory')
        )
        
    def save(self, filename="chat_memory.json"):
        """Save memory state."""
        data = {
            "messages": [
                {
                    "role": "system" if isinstance(msg, SystemMessage)
                    else "user" if isinstance(msg, HumanMessage)
                    else "assistant",
                    "content": msg.content
                }
                for msg in self.messages
            ],
            "timestamp": datetime.now().isoformat()
        }
        try:
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
            self.vector_store.persist()
            console.print(f"[green]Memory saved to {filename}[/green]")
        except Exception as e:
            console.print(f"[red]Failed to save memory: {e}[/red]")
            
    def load(self, filename="chat_memory.json"):
        """Load memory state."""
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
            
            self.clear()
            for msg in data["messages"]:
                if msg["role"] == "system":
                    self.add_message(SystemMessage(content=msg["content"]))
                elif msg["role"] == "user":
                    self.add_message(HumanMessage(content=msg["content"]))
                else:
                    self.add_message(AIMessage(content=msg["content"]))
                    
            console.print(f"[green]Memory loaded from {filename}[/green]")
            return True
        except Exception as e:
            console.print(f"[red]Failed to load memory: {e}[/red]")
            return False

class StreamingCallbackHandler(BaseCallbackHandler):
    """Callback handler for streaming responses."""
    
    def __init__(self):
        self.live = None
        self.content = []
        self.current_content = ""
        
    def on_llm_new_token(self, token: str, **kwargs):
        """Handle new tokens as they stream in."""
        self.content.append(token)
        self.current_content += token
        if self.live:
            self.live.update(Markdown(self.current_content))

class OpenRouterChatModel(BaseChatModel):
    """Custom LangChain chat model for OpenRouter."""
    
    class Config:
        arbitrary_types_allowed = True
        
    api_key: str = Field(description="OpenRouter API key")
    model_name: str = Field(description="Name of the model to use")
    headers: Dict[str, str] = Field(description="Headers for API requests")
    
    def __init__(self, **data):
        super().__init__(**data)
        
    @property
    def _llm_type(self) -> str:
        return "openrouter"
        
    def _generate(
        self,
        messages: List[Any],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> Any:
        url = "https://openrouter.ai/api/v1/chat/completions"
        
        # Properly format messages for the API
        formatted_messages = []
        for msg_list in messages:
            for msg in msg_list:
                if isinstance(msg, SystemMessage):
                    formatted_messages.append({"role": "system", "content": msg.content})
                elif isinstance(msg, HumanMessage):
                    formatted_messages.append({"role": "user", "content": msg.content})
                elif isinstance(msg, AIMessage):
                    formatted_messages.append({"role": "assistant", "content": msg.content})
        
        debug_print(f"Sending request to {url} with model {self.model_name}")
        
        data = {
            "model": self.model_name,
            "messages": formatted_messages,
            "stream": True,
            "temperature": 0.7,
        }
        
        response = requests.post(url, headers=self.headers, json=data, stream=True)
        response.raise_for_status()
        
        handler = StreamingCallbackHandler()
        with Live(Markdown(""), refresh_per_second=4) as handler.live:
            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith('data: '):
                        try:
                            if line.strip() == 'data: [DONE]':
                                debug_print("Received DONE signal")
                                continue
                            data = json.loads(line[6:])
                            if data.get("choices"):
                                chunk = data["choices"][0].get("delta", {}).get("content", "")
                                if chunk:
                                    if run_manager:
                                        run_manager.on_llm_new_token(chunk)
                                    handler.on_llm_new_token(chunk)
                        except json.JSONDecodeError:
                            debug_print("Failed to decode JSON from chunk")
                            continue
                        except Exception as e:
                            debug_print(f"Error processing chunk: {str(e)}")
                            debug_print(f"Raw line: {line}")
                            continue
        
        content = "".join(handler.content)
        if not content:
            debug_print("No content generated from model")
            raise Exception("No response generated from the model")
            
        debug_print("Successfully generated response")
        return {"generations": [{"text": content}]}

class ChatBot:
    """
    A terminal-based chatbot using OpenRouter API with LangChain integration.
    Supports multiple AI models, caches available models,
    and tracks token usage and costs.
    """
    
    API_URL = "https://openrouter.ai/api/v1"
    CACHE_DURATION = timedelta(hours=24)
    DEFAULT_SYSTEM_PROMPT = "You are a helpful AI assistant."
    
    def __init__(self, api_key):
        self.api_key = api_key
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "HTTP-Referer": "https://github.com/OpenRouterTeam/openrouter-python",
            "X-Title": "Terminal Chat Bot",
            "Content-Type": "application/json"
        }
        self.model = None
        self.model_context_length = None
        self.tested_models = {}
        self.stats = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "cost": 0,
            "pricing": {"prompt": 0, "completion": 0}
        }
        
        # Initialize paths
        self.config_dir = os.getenv('CONFIG_DIR', 'storage/config')
        self.models_dir = os.getenv('MODELS_DIR', 'storage/models')
        self.history_dir = os.getenv('HISTORY_DIR', 'storage/history')
        
        # Create directories if they don't exist
        for d in [self.config_dir, self.models_dir, self.history_dir]:
            os.makedirs(d, exist_ok=True)
        
        # Initialize hybrid memory system
        self.memory = HybridMemory(max_tokens=self.model_context_length or 4000)
        
        # Initialize chat model (will be set when model is selected)
        self.chat_model = None
        
    def set_system_prompt(self, prompt):
        """Set a new system prompt and reset memory."""
        self.memory.clear()
        self.memory.add_message(SystemMessage(content=prompt))
        
    def reset_context(self):
        """Reset conversation memory to just the system prompt."""
        self.memory.clear()
        self.memory.add_message(SystemMessage(content=self.DEFAULT_SYSTEM_PROMPT))
        
    def save_context(self, filename=None):
        """Save current conversation context to file."""
        if filename is None:
            filename = os.path.join(self.history_dir, "chat_context.json")
        self.memory.save(filename)
            
    def load_context(self, filename=None):
        """Load conversation context from file."""
        if filename is None:
            filename = os.path.join(self.history_dir, "chat_context.json")
        return self.memory.load(filename)
            
    def _make_request(self, endpoint, method="get", **kwargs):
        """
        Make an HTTP request to OpenRouter API.
        
        Args:
            endpoint (str): API endpoint to call
            method (str): HTTP method (get/post)
            **kwargs: Additional arguments for requests
            
        Returns:
            dict: JSON response from API
            
        Raises:
            Exception: If API request fails
        """
        url = f"{self.API_URL}/{endpoint}"
        response = requests.request(method, url, headers=self.headers, **kwargs)
        response.raise_for_status()
        return response.json()
    
    def _load_cache(self, cache_file):
        """
        Load cached models if available and not expired.
        
        Args:
            cache_file (str): Path to cache file
            
        Returns:
            list: Cached models if valid, None otherwise
        """
        if not os.path.exists(cache_file):
            return None
            
        try:
            with open(cache_file, 'r') as f:
                cache = json.load(f)
            cache_time = datetime.fromisoformat(cache['timestamp'])
            
            if datetime.now() - cache_time > self.CACHE_DURATION:
                return None
            return cache['models']
        except Exception:
            return None
            
    def _save_cache(self, cache_file, models):
        """
        Save models list to cache file.
        
        Args:
            cache_file (str): Path to cache file
            models (list): List of models to cache
        """
        try:
            cache = {
                'timestamp': datetime.now().isoformat(),
                'models': models
            }
            with open(cache_file, 'w') as f:
                json.dump(cache, f)
        except Exception as e:
            console.print(f"[yellow]Warning: Could not save cache: {e}[/yellow]")
    
    def _format_price(self, price_per_token):
        """
        Format price with appropriate decimal precision based on amount.
        
        Args:
            price_per_token (float): Price per token
            
        Returns:
            str: Formatted price string with appropriate precision
        """
        if price_per_token is None:
            return "N/A"
            
        try:
            price_per_k = float(price_per_token) * 1000
            if price_per_k < 0:
                return "N/A"
            
            if price_per_k == 0:
                return "$0.000000"
            elif price_per_k < 0.000001:
                return f"${price_per_k:.8f}"  # Ultra small prices
            elif price_per_k < 0.0001:
                return f"${price_per_k:.7f}"  # Very small prices
            elif price_per_k < 0.01:
                return f"${price_per_k:.6f}"  # Small prices
            elif price_per_k < 1:
                return f"${price_per_k:.5f}"  # Medium prices
            return f"${price_per_k:.4f}"      # Large prices
        except (ValueError, TypeError):
            return "N/A"
    
    def test_model(self, model_id):
        """
        Test if a model is actually available by sending a test message.
        
        Args:
            model_id (str): ID of model to test
            
        Returns:
            bool: True if model works, False otherwise
        """
        if model_id in self.tested_models:
            return self.tested_models[model_id]
            
        try:
            debug_print(f"Testing model {model_id}")
            data = self._make_request("chat/completions", 
                method="post",
                json={
                    "model": model_id,
                    "messages": [{"role": "user", "content": "Hi"}]
                },
                timeout=5
            )
            works = bool(data.get("choices"))
            self.tested_models[model_id] = works
            return works
        except Exception as e:
            debug_print(f"Model test failed: {str(e)}")
            self.tested_models[model_id] = False
            return False
    
    def get_models(self, cache_file=None, force_refresh=False):
        """Get list of available models, using cache if possible."""
        if cache_file is None:
            cache_file = os.path.join(self.config_dir, "models_cache.json")
            
        debug_print(f"Using cache file: {cache_file}")
            
        if not force_refresh:
            cached = self._load_cache(cache_file)
            if cached:
                console.print("[green]Using cached models list[/green]")
                debug_print(f"Loaded {len(cached)} models from cache")
                return cached
        
        console.print("[yellow]Fetching models from OpenRouter...[/yellow]")
        try:
            response = self._make_request("models")
            models = response["data"]
            debug_print(f"Fetched {len(models)} models from API")
        except Exception as e:
            console.print(f"[red]Error fetching models: {str(e)}[/red]")
            raise
        
        # Mark all models as available by default
        for model in models:
            model["actually_available"] = True
        
        # Only test first few free models to save time
        free_models = [m for m in models if ":free" in m["id"].lower()][:5]  # Test only first 5
        debug_print(f"Testing {len(free_models)} free models")
        
        if free_models:
            console.print("\n[yellow]Testing free models availability...[/yellow]")
            for model in free_models:
                model["actually_available"] = self.test_model(model["id"])
                
        # Filter and cache available models
        available_models = [m for m in models if m.get("actually_available", True)]
        debug_print(f"{len(available_models)} models available in total")
        
        try:
            self._save_cache(cache_file, available_models)
            console.print("[green]Models cache saved successfully[/green]")
        except Exception as e:
            console.print(f"[red]Error saving cache: {str(e)}[/red]")
        
        return available_models
    
    def display_models(self):
        """
        Display available models in a formatted table.
        Models are sorted with free models first, then by price.
        
        Returns:
            list: Sorted list of available models
        """
        force_refresh = True
        if os.path.exists(os.path.join(self.config_dir, "models_cache.json")):
            refresh = Prompt.ask(
                "\n[yellow]Refresh models list?[/yellow]",
                choices=["y", "n"],
                default="n"
            )
            force_refresh = refresh.lower() == "y"
        
        models = self.get_models(force_refresh=force_refresh)
        models.sort(key=lambda m: (
            not ":free" in m["id"].lower(),
            float(m.get("pricing", {}).get("prompt", 0) or 0) + 
            float(m.get("pricing", {}).get("completion", 0) or 0),
            m["id"]
        ))
        
        table = Table(title="Available Models")
        table.add_column("ID", justify="right", style="cyan")
        table.add_column("Model", style="green")
        table.add_column("Context Length", justify="right", style="magenta")
        table.add_column("Cost per 1K tokens (Input/Output)", style="yellow")
        
        for idx, model in enumerate(models, 1):
            name_style = "magenta" if ":free" in model["id"].lower() else "green"
            pricing = model.get("pricing", {})
            price_display = (
                "Auto-selected based on usage" if "auto" in model["id"].lower()
                else f"{self._format_price(pricing.get('prompt'))} / {self._format_price(pricing.get('completion'))}"
            )
            
            table.add_row(
                str(idx),
                Text(model["id"], style=name_style),
                str(model.get("context_length", "N/A")),
                price_display
            )
        
        console.print(table)
        return models
    
    def update_stats(self, prompt_tokens, completion_tokens):
        """
        Update and display token usage and cost statistics.
        
        Args:
            prompt_tokens (int): Number of input tokens used
            completion_tokens (int): Number of output tokens used
        """
        # Calculate costs
        prompt_cost = prompt_tokens * self.stats["pricing"]["prompt"]
        completion_cost = completion_tokens * self.stats["pricing"]["completion"]
        exchange_cost = prompt_cost + completion_cost
        
        # Update totals
        self.stats["prompt_tokens"] += prompt_tokens
        self.stats["completion_tokens"] += completion_tokens
        self.stats["cost"] += exchange_cost
        
        # Display stats
        current = f"Current: {prompt_tokens}↑ {completion_tokens}↓ (${exchange_cost:.6f})"
        total = f"Total: {self.stats['prompt_tokens']}↑ {self.stats['completion_tokens']}↓ (${self.stats['cost']:.6f})"
        console.print(Panel(
            f"[cyan]{current}[/cyan] • [yellow]{total}[/yellow]",
            title="[bold yellow]Usage[/bold yellow]",
            border_style="yellow"
        ))
    
    def select_model(self):
        """Let user select a model and initialize LangChain chat model."""
        try:
            models = self.display_models()
            debug_print(f"Found {len(models)} models")
            
            while True:
                try:
                    choice = Prompt.ask(
                        "\n[yellow]Enter model ID number[/yellow]",
                        default="1"
                    ).strip()
                    
                    debug_print(f"User input: '{choice}'")
                    
                    # Check if input is a valid number
                    if not choice.isdigit():
                        console.print("[red]Please enter a valid number.[/red]")
                        continue
                    
                    idx = int(choice) - 1
                    debug_print(f"Converted to index: {idx}")
                    
                    if 0 <= idx < len(models):
                        model = models[idx]
                        debug_print(f"Selected model: {model['id']}")
                        
                        if not model.get("actually_available", True):
                            console.print("[red]Model unavailable. Please select another.[/red]")
                            continue
                        
                        self.model = model["id"]
                        self.model_context_length = model.get("context_length", None)
                        debug_print(f"Context length: {self.model_context_length}")
                        
                        # Initialize memory before chat model
                        try:
                            debug_print("Initializing new HybridMemory")
                            self.memory = HybridMemory(max_tokens=self.model_context_length or 4000)
                            debug_print("HybridMemory initialized successfully")
                        except Exception as e:
                            console.print(f"[red]Error initializing memory system: {str(e)}[/red]")
                            debug_print(f"Memory initialization error: {str(e)}")
                            raise
                        
                        # Initialize LangChain chat model
                        try:
                            debug_print("Initializing OpenRouterChatModel")
                            self.chat_model = OpenRouterChatModel(
                                api_key=self.api_key,
                                model_name=self.model,
                                headers=self.headers
                            )
                            debug_print("OpenRouterChatModel initialized successfully")
                            
                            # Initialize stats after successful model selection
                            self.stats.update({
                                "prompt_tokens": 0,
                                "completion_tokens": 0,
                                "cost": 0,
                                "pricing": {
                                    "prompt": float(model.get("pricing", {}).get("prompt", 0) or 0),
                                    "completion": float(model.get("pricing", {}).get("completion", 0) or 0)
                                }
                            })
                            
                        except Exception as e:
                            console.print(f"[red]Error initializing chat model: {str(e)}[/red]")
                            debug_print(f"Chat model initialization error: {str(e)}")
                            raise
                        
                        name_style = "magenta" if ":free" in model["id"].lower() else "green"
                        console.print(f"\n[green]Selected model:[/green] {Text(model['id'], style=name_style)}")
                        console.print(f"[green]Context length:[/green] {self.model_context_length or 'Unknown'}\n")
                        return
                    else:
                        console.print(f"[red]Please enter a number between 1 and {len(models)}.[/red]")
                        debug_print(f"Index {idx} out of range [0, {len(models)-1}]")
                except (ValueError, KeyboardInterrupt, EOFError) as e:
                    debug_print(f"Exception caught: {type(e).__name__}: {str(e)}")
                    console.print("\n[red]Invalid input. Please enter a valid number.[/red]")
                    continue
        except Exception as e:
            debug_print(f"Fatal error in select_model: {str(e)}")
            raise
    
    def chat(self, message):
        """Send a chat message using LangChain memory and chat model."""
        try:
            # Add user message to memory
            self.memory.add_message(HumanMessage(content=message))
            
            # Get relevant context for this message
            messages = self.memory.get_messages(query=message)
            
            # Generate response
            console.print("\n[bold purple]Assistant[/bold purple]")
            response = self.chat_model._generate([messages])
            content = response["generations"][0]["text"]
            
            # Add assistant response to memory
            self.memory.add_message(AIMessage(content=content))
            
            # Update usage stats (need to make a non-streaming request for this)
            url = f"{self.API_URL}/chat/completions"
            
            # Format messages for stats request
            formatted_messages = []
            for msg in messages:
                if isinstance(msg, SystemMessage):
                    formatted_messages.append({"role": "system", "content": msg.content})
                elif isinstance(msg, HumanMessage):
                    formatted_messages.append({"role": "user", "content": msg.content})
                elif isinstance(msg, AIMessage):
                    formatted_messages.append({"role": "assistant", "content": msg.content})
            
            stats_response = requests.post(
                url,
                headers=self.headers,
                json={
                    "model": self.model,
                    "messages": formatted_messages,
                    "stream": False
                }
            )
            stats_response.raise_for_status()
            stats_data = stats_response.json()
            
            # Update usage stats
            usage = stats_data.get("usage", {})
            self.update_stats(
                usage.get("prompt_tokens", 0),
                usage.get("completion_tokens", 0)
            )
            
            return content
            
        except Exception as e:
            error = str(e)
            if "Provider returned error" in error:
                raise Exception("Model unavailable or rate limited. Try another model or wait.")
            raise Exception(f"API Error: {error}")

def main():
    """
    Main function to run the chatbot.
    Handles model selection, chat loop, and command processing.
    """
    try:
        # Get password and decrypt API key
        password = getpass("Enter password to decrypt API key: ")
        api_key = decrypt_api_key(password)
        
        chatbot = ChatBot(api_key)
        
        console.print("[bold green]Welcome to OpenRouter Terminal Chat![/bold green]")
        console.print("Commands: 'exit', 'quit', 'switch model', 'save', 'load', 'clear', 'prompt <new prompt>', 'debug on/off'")
        console.print()
        
        chatbot.select_model()
        chatbot.reset_context()
        
        while True:
            try:
                user_input = Prompt.ask("[bold blue]You[/bold blue]")
            except (KeyboardInterrupt, EOFError):
                console.print("\n[bold green]Goodbye![/bold green]")
                break
                
            if user_input.lower() in ['exit', 'quit']:
                console.print("\n[bold green]Goodbye![/bold green]")
                break
                
            if user_input.lower() == 'switch model':
                chatbot.select_model()
                continue
                
            if user_input.lower() == 'save':
                chatbot.save_context()
                continue
                
            if user_input.lower() == 'load':
                chatbot.load_context()
                continue
                
            if user_input.lower() == 'clear':
                chatbot.reset_context()
                console.print("[green]Context cleared[/green]")
                continue
                
            if user_input.lower().startswith('prompt '):
                new_prompt = user_input[7:].strip()
                chatbot.set_system_prompt(new_prompt)
                console.print(f"[green]System prompt updated: {new_prompt}[/green]")
                continue

            if user_input.lower() in ['debug on', 'debug off']:
                set_debug(user_input.lower() == 'debug on')
                continue
            
            try:
                response = chatbot.chat(user_input)
                console.print()
                
            except Exception as e:
                console.print(f"[bold red]{str(e)}[/bold red]")
                
    except Exception as e:
        console.print(f"[bold red]Fatal error: {str(e)}[/bold red]")
        return 1
        
    return 0

if __name__ == "__main__":
    exit(main())
