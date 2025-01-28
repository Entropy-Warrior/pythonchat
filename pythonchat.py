# Standard library imports
import os
import json
import base64
from datetime import datetime, timedelta
from getpass import getpass
from typing import Any, Dict, List, Optional

# Third-party imports - UI
from rich.console import Console
from rich.markdown import Markdown
from rich.prompt import Prompt
from rich.table import Table
from rich.text import Text
from rich.progress import Progress
from rich.panel import Panel
from rich.live import Live

# Third-party imports - LangChain
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.embeddings import Embeddings
from langchain_chroma import Chroma
from sentence_transformers import SentenceTransformer

# Third-party imports - Other
import requests
import tiktoken
from pydantic import BaseModel, Field
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

# Initialize console and debug settings
console = Console()
DEBUG_MODE = False

# =====================================
# Configuration and Settings
# =====================================
class Config:
    """Global configuration settings for the application."""
    # API Configuration
    ENCRYPTED_API_KEY = "gAAAAABnmEo__06SYg3CsTS3uNbJt_5OvUS7Tt4qjInT_fwWE88b8ihOzrkVUP5GKbxZpyHmGpYSn4halGuvUik2ypooRfjMWtZhylglkUpnGXu9HFmmuyrzj9go4Ils2c0cg5GNNI5ZrCucYqfiZRixisQCT2CmLPsv8RKv8YEms5N3zkbgevA="
    SALT = b'\xf8S\x92\xfa\xb6\x16\xe1$\xd2\x9a*`\xe2_~D'
    
    # API Endpoints
    ENDPOINTS = {
        "openrouter": "https://openrouter.ai/api/v1",
        "wangscience": "https://wangscience.com/api/v1"
    }
    API_ENDPOINT = None  # Will be set at runtime
    
    # Storage Paths
    STORAGE = {
        "models": os.getenv('MODELS_DIR', 'storage/models'),
        "config": os.getenv('CONFIG_DIR', 'storage/config'),
        "history": os.getenv('HISTORY_DIR', 'storage/history')
    }
    
    # Model Settings
    MODEL_SETTINGS = {
        "name": "all-MiniLM-L6-v2",
        "system_prompt": "You are a helpful AI assistant.",
        "max_tokens": 4000,
        "temperature": 0.7,
        "timeout": 10
    }
    
    @classmethod
    def select_endpoint(cls):
        """Select which API endpoint to use."""
        choice = Prompt.ask(
            "\nSelect API endpoint [1/2]",
            choices=["1", "2"],
            default="1"
        )
        cls.API_ENDPOINT = (
            cls.ENDPOINTS["openrouter"] if choice == "1" 
            else cls.ENDPOINTS["wangscience"]
        )
        endpoint_name = "OpenRouter" if choice == "1" else "WangScience"
        console.print(f"Using {endpoint_name} API")

# =====================================
# Debug Utilities
# =====================================
def set_debug(enabled: bool):
    """Enable or disable debug output."""
    global DEBUG_MODE
    DEBUG_MODE = enabled
    console.print(f"[yellow]Debug mode {'enabled' if enabled else 'disabled'}[/yellow]")

def debug_print(message: str):
    """Print debug message if debug mode is enabled."""
    if DEBUG_MODE:
        console.print(f"[yellow]Debug: {message}[/yellow]")

# =====================================
# Encryption Utilities
# =====================================
class Encryption:
    """Handle API key encryption and decryption."""
    
    @staticmethod
    def get_key(password: str) -> bytes:
        """Generate encryption key from password."""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=Config.SALT,
            iterations=480000,
        )
        return base64.urlsafe_b64encode(kdf.derive(password.encode()))
    
    @staticmethod
    def decrypt_api_key(password: str) -> str:
        """Decrypt API key using password."""
        try:
            f = Fernet(Encryption.get_key(password))
            return f.decrypt(Config.ENCRYPTED_API_KEY.encode()).decode()
        except Exception:
            console.print("[red]Failed to decrypt API key. Wrong password?[/red]")
            raise ValueError("Decryption failed")

# =====================================
# Streaming Callback Handler
# =====================================
class StreamingCallback(BaseCallbackHandler):
    """Handle streaming responses from the model."""
    
    def __init__(self):
        self.live = None
        self.content = []
        self.current_content = ""
    
    def on_llm_new_token(self, token: str, **kwargs):
        """Process incoming tokens."""
        self.content.append(token)
        self.current_content += token
        if self.live:
            self.live.update(Markdown(self.current_content))

# =====================================
# OpenRouter Chat Model
# =====================================
class ChatModel(BaseChatModel):
    """OpenRouter chat model implementation."""
    
    api_key: str = Field(description="API key")
    model_name: str = Field(description="Model name")
    headers: Dict[str, str] = Field(description="Request headers")
    temperature: float = Field(default=Config.MODEL_SETTINGS["temperature"])
    
    class Config:
        arbitrary_types_allowed = True
    
    @property
    def _llm_type(self) -> str:
        return "openrouter"
    
    def _generate(
        self,
        messages: List[Any],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Generate response from the model."""
        url = f"{Config.API_ENDPOINT}/chat/completions"
        
        # Format messages
        formatted_messages = [
            {
                "role": {
                    SystemMessage: "system",
                    HumanMessage: "user",
                    AIMessage: "assistant"
                }[type(msg)],
                "content": msg.content
            }
            for msg_list in messages
            for msg in msg_list
        ]
        
        # Prepare request
        data = {
            "model": self.model_name,
            "messages": formatted_messages,
            "stream": True,
            "temperature": self.temperature,
        }
        
        # Make request
        response = requests.post(
            url, 
            headers=self.headers, 
            json=data, 
            stream=True,
            timeout=Config.MODEL_SETTINGS["timeout"]
        )
        response.raise_for_status()
        
        # Handle streaming response
        handler = StreamingCallback()
        with Live(Markdown(""), refresh_per_second=4) as handler.live:
            for line in response.iter_lines():
                if not line or not line.startswith(b'data: '):
                    continue
                    
                line = line.decode('utf-8')[6:]  # Remove 'data: ' prefix
                if line.strip() == '[DONE]':
                    continue
                    
                try:
                    data = json.loads(line)
                    chunk = data["choices"][0].get("delta", {}).get("content", "")
                    if chunk:
                        if run_manager:
                            run_manager.on_llm_new_token(chunk)
                        handler.on_llm_new_token(chunk)
                except (json.JSONDecodeError, KeyError, IndexError) as e:
                    console.print(f"[red]Error processing response: {e}[/red]")
                    continue
        
        content = "".join(handler.content)
        if not content:
            raise ValueError("No response generated from the model")
            
        return {"generations": [{"text": content}]}

# =====================================
# Hybrid Memory System
# =====================================
class HybridMemory:
    """Memory system using vector store and summarization."""
    
    # =====================================
    # Initialization
    # =====================================
    def __init__(self, embedding_model=None, max_tokens=Config.MODEL_SETTINGS["max_tokens"]):
        """Initialize the hybrid memory system."""
        self.max_tokens = max_tokens
        self.messages = []
        self.encoding = tiktoken.get_encoding("cl100k_base")
        
        self._init_embeddings(embedding_model)
        self._init_vector_store()
        
        debug_print("HybridMemory initialization complete")
    
    # =====================================
    # Setup Methods
    # =====================================
    def _init_embeddings(self, embedding_model):
        """Initialize the embeddings model."""
        try:
            if embedding_model is None:
                embedding_model = os.path.join(Config.STORAGE["models"], Config.MODEL_SETTINGS["name"])
            
            if not os.path.exists(embedding_model):
                raise ValueError(
                    f"Model not found at {embedding_model}. "
                    "The model should be included in the Docker image. "
                    "Please ensure you're using the correct image from the registry."
                )
            
            debug_print(f"Initializing embeddings with local model: {embedding_model}")
            self.embeddings = SentenceTransformer(embedding_model, device="cpu")
            debug_print("Embeddings initialized successfully")
            
        except Exception as e:
            console.print(f"[red]Error loading embeddings model: {str(e)}[/red]")
            debug_print(f"Embeddings initialization error: {str(e)}")
            raise
    
    def _init_vector_store(self):
        """Initialize the vector store."""
        try:
            persist_dir = os.path.join(Config.STORAGE["history"], '.chat_memory')
            os.makedirs(persist_dir, exist_ok=True)
            
            debug_print(f"Initializing vector store in: {persist_dir}")
            self.vector_store = Chroma(
                collection_name="chat_memory",
                embedding_function=self.embeddings,
                persist_directory=persist_dir
            )
            debug_print("Vector store initialized successfully")
            
        except Exception as e:
            console.print(f"[red]Error initializing vector store: {str(e)}[/red]")
            debug_print(f"Vector store initialization error: {str(e)}")
            raise
    
    # =====================================
    # Memory Operations
    # =====================================
    def add_message(self, message):
        """Add a message to both vector store and message list."""
        self.messages.append(message)
        
        if isinstance(message, (HumanMessage, AIMessage)):
            self.vector_store.add_texts(
                texts=[message.content],
                metadatas=[{
                    "type": "human" if isinstance(message, HumanMessage) else "ai",
                    "timestamp": datetime.now().isoformat()
                }]
            )
    
    def get_token_count(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.encoding.encode(text))
    
    def get_relevant_history(self, query: str, k: int = 5) -> List[str]:
        """Get most relevant messages from vector store."""
        try:
            collection = self.vector_store._collection
            doc_count = collection.count()
            
            if doc_count == 0:
                debug_print("No messages in vector store")
                return []
            
            k = min(k, doc_count)
            debug_print(f"Searching for {k} relevant messages from {doc_count} total messages")
            
            results = self.vector_store.similarity_search(query, k=k)
            return [doc.page_content for doc in results]
            
        except Exception as e:
            debug_print(f"Error retrieving history: {str(e)}")
            return []
    
    def get_messages(self, query: str = None) -> List[Any]:
        """Get messages for context, using hybrid approach."""
        final_messages = []
        
        if self.messages and isinstance(self.messages[0], SystemMessage):
            final_messages.append(self.messages[0])
        
        if query:
            relevant_history = self.get_relevant_history(query)
            final_messages.extend(AIMessage(content=text) for text in relevant_history)
        
        token_count = sum(self.get_token_count(msg.content) for msg in final_messages)
        
        for message in reversed(self.messages):
            if not isinstance(message, SystemMessage):
                msg_tokens = self.get_token_count(message.content)
                if token_count + msg_tokens <= self.max_tokens:
                    final_messages.append(message)
                    token_count += msg_tokens
                else:
                    break
        
        return final_messages
    
    # =====================================
    # State Management
    # =====================================
    def clear(self):
        """Clear all memory."""
        self.messages = []
        self.vector_store.delete_collection()
        self._init_vector_store()
    
    def save(self, filename="chat_memory.json"):
        """Save memory state to file."""
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
        """Load memory state from file."""
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

# =====================================
# Main Chat Bot
# =====================================
class ChatBot:
    """Terminal-based chatbot using OpenRouter API with LangChain integration."""
    
    # =====================================
    # Constants
    # =====================================
    CACHE_DURATION = timedelta(hours=24)
    DEFAULT_SYSTEM_PROMPT = "You are a helpful AI assistant."
    
    # =====================================
    # Initialization
    # =====================================
    def __init__(self, api_key):
        """Initialize the chatbot with API key."""
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
        self.config_dir = Config.STORAGE["config"]
        self.models_dir = Config.STORAGE["models"]
        self.history_dir = Config.STORAGE["history"]
        
        # Create directories if they don't exist
        for d in [self.config_dir, self.models_dir, self.history_dir]:
            os.makedirs(d, exist_ok=True)
        
        # Initialize hybrid memory system with local model path
        model_path = os.path.join(Config.STORAGE["models"], Config.MODEL_SETTINGS["name"])
        self.memory = HybridMemory(embedding_model=model_path, max_tokens=self.model_context_length or Config.MODEL_SETTINGS["max_tokens"])
        
        # Initialize chat model (will be set when model is selected)
        self.chat_model = None
    
    # =====================================
    # System Prompt Management
    # =====================================
    def set_system_prompt(self, prompt):
        """Set a new system prompt and reset memory."""
        self.memory.clear()
        self.memory.add_message(SystemMessage(content=prompt))
    
    def reset_context(self):
        """Reset conversation memory to just the system prompt."""
        self.memory.clear()
        self.memory.add_message(SystemMessage(content=self.DEFAULT_SYSTEM_PROMPT))
    
    # =====================================
    # Context Management
    # =====================================
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
    
    # =====================================
    # Model Management
    # =====================================
    def _make_request(self, endpoint, method="get", **kwargs):
        """Make an HTTP request to OpenRouter API."""
        url = f"{Config.API_ENDPOINT}/{endpoint}"
        response = requests.request(method, url, headers=self.headers, **kwargs)
        response.raise_for_status()
        return response.json()
    
    def _load_cache(self, cache_file):
        """Load cached models if available and not expired."""
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
        """Save models list to cache file."""
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
        """Format price with appropriate decimal precision."""
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
        """Test if a model is actually available."""
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
        free_models = [m for m in models if ":free" in m["id"].lower()][:5]
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
        """Display available models in a formatted table."""
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
    
    def select_model(self):
        """Let user select a model and initialize chat model."""
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
                            self.memory = HybridMemory(max_tokens=self.model_context_length or Config.MODEL_SETTINGS["max_tokens"])
                            debug_print("HybridMemory initialized successfully")
                        except Exception as e:
                            console.print(f"[red]Error initializing memory system: {str(e)}[/red]")
                            debug_print(f"Memory initialization error: {str(e)}")
                            raise
                        
                        # Initialize LangChain chat model
                        try:
                            debug_print("Initializing OpenRouterChatModel")
                            self.chat_model = ChatModel(
                                api_key=self.api_key,
                                model_name=self.model,
                                temperature=Config.MODEL_SETTINGS["temperature"],
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
    
    # =====================================
    # Usage Statistics
    # =====================================
    def update_stats(self, prompt_tokens, completion_tokens):
        """Update and display token usage and cost statistics."""
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
    
    # =====================================
    # Chat Interaction
    # =====================================
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
            url = f"{Config.API_ENDPOINT}/chat/completions"
            
            # Format messages for stats request
            formatted_messages = []
            for msg in messages:
                if isinstance(msg, SystemMessage):
                    formatted_messages.append({"role": "system", "content": msg.content})
                elif isinstance(msg, HumanMessage):
                    formatted_messages.append({"role": "user", "content": msg.content})
                elif isinstance(msg, AIMessage):
                    formatted_messages.append({"role": "assistant", "content": msg.content})
            
            try:
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
            except Exception as e:
                debug_print(f"Failed to get usage stats: {str(e)}")
                console.print("[yellow]Warning: Could not retrieve usage statistics[/yellow]")
            
            return content
            
        except Exception as e:
            error = str(e)
            if "Provider returned error" in error:
                raise Exception("Model unavailable or rate limited. Try another model or wait.")
            raise Exception(f"API Error: {error}")

# =====================================
# Main Entry Point
# =====================================
def main():
    """Main function to run the chatbot."""
    try:
        # Get password and decrypt API key
        password = getpass("Enter password to decrypt API key: ")
        api_key = Encryption.decrypt_api_key(password)
        
        # Select API endpoint
        Config.select_endpoint()
        
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
