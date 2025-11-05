"""
Model Manager for Real LLM Operations
Handles downloading, status checking, and management of LLM models.
"""

import os
import json
import logging
import subprocess
import requests
from typing import Dict, List, Any, Optional
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelManager:
    """Manages LLM model downloading and status checking."""
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        
        # Model configurations with real download info
        self.models_config = {
            "llama3": {
                "hf_model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
                "size_gb": 16,
                "description": "Meta's Llama 3.1 8B Instruct",
                "requires_auth": True
            },
            "qwen3": {
                "hf_model": "Qwen/Qwen2.5-7B-Instruct", 
                "size_gb": 14,
                "description": "Alibaba's Qwen 2.5 7B Instruct",
                "requires_auth": False
            },
            "deepseek": {
                "hf_model": "deepseek-ai/deepseek-llm-7b-chat",
                "size_gb": 14,
                "description": "DeepSeek 7B Chat",
                "requires_auth": False
            },
            "gemma3": {
                "hf_model": "google/gemma-2-9b-it",
                "size_gb": 18,
                "description": "Google Gemma 2 9B IT",
                "requires_auth": True
            },
            "gpt_oss": {
                "ollama_model": "gpt-oss:20b",
                "size_gb": 40,
                "description": "GPT-OSS 20B via Ollama",
                "requires_auth": False
            }
        }
    
    def get_directory_size(self, path: Path) -> float:
        """
        Calculate the total size of a directory in GB.
        
        Args:
            path: Path to the directory
            
        Returns:
            float: Size in GB
        """
        total_size = 0
        try:
            if path.exists() and path.is_dir():
                for dirpath, dirnames, filenames in os.walk(path):
                    for filename in filenames:
                        filepath = os.path.join(dirpath, filename)
                        try:
                            total_size += os.path.getsize(filepath)
                        except (OSError, FileNotFoundError):
                            # Skip files that can't be accessed
                            continue
            return total_size / (1024**3)  # Convert bytes to GB
        except Exception as e:
            logger.warning(f"Error calculating directory size for {path}: {e}")
            return 0.0

    def get_model_disk_size(self, model_name: str) -> dict:
        """
        Get the actual disk size of a downloaded model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            dict: Size information including actual disk usage
        """
        model_dir = self.models_dir / model_name
        
        size_info = {
            "model_name": model_name,
            "estimated_gb": self.models_config.get(model_name, {}).get("size_gb", 0),
            "actual_gb": 0.0,
            "disk_usage": "Not downloaded"
        }
        
        if model_dir.exists():
            actual_size = self.get_directory_size(model_dir)
            size_info["actual_gb"] = round(actual_size, 2)
            
            if actual_size > 0:
                size_info["disk_usage"] = f"{actual_size:.2f} GB"
            else:
                size_info["disk_usage"] = "< 0.01 GB"
        
        # For Ollama models, check Ollama's storage
        config = self.models_config.get(model_name, {})
        if "ollama_model" in config and self.check_ollama_status():
            try:
                # Try to get Ollama model size
                result = subprocess.run(['ollama', 'list'], 
                                      capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    ollama_model = config["ollama_model"]
                    lines = result.stdout.split('\n')
                    for line in lines:
                        if ollama_model in line:
                            # Parse size from Ollama list output
                            parts = line.split()
                            if len(parts) >= 3:
                                size_str = parts[2]  # Usually the size column
                                if 'GB' in size_str:
                                    try:
                                        ollama_size = float(size_str.replace('GB', ''))
                                        size_info["actual_gb"] = ollama_size
                                        size_info["disk_usage"] = f"{ollama_size:.2f} GB (Ollama)"
                                    except ValueError:
                                        pass
                            break
            except Exception as e:
                logger.warning(f"Error getting Ollama model size: {e}")
        
        return size_info

    def authenticate_huggingface(self) -> bool:
        """
        Authenticate with HuggingFace using CLI login.
        
        Returns:
            bool: True if authentication successful, False otherwise
        """
        try:
            import subprocess
            
            logger.info("Starting HuggingFace authentication...")
            print("🔐 HuggingFace Authentication")
            print("-" * 40)
            print("This will open the HuggingFace login process.")
            print("You will need your HuggingFace token from: https://huggingface.co/settings/tokens")
            print()
            
            # Check if huggingface-cli is available
            try:
                result = subprocess.run(['huggingface-cli', '--version'], 
                                      capture_output=True, text=True, timeout=10)
                if result.returncode != 0:
                    print("❌ huggingface-cli not found. Installing...")
                    # Install huggingface-hub if not available
                    subprocess.run(['pip', 'install', 'huggingface-hub[cli]'], check=True)
            except (subprocess.TimeoutExpired, FileNotFoundError):
                print("❌ Installing huggingface-hub CLI...")
                subprocess.run(['pip', 'install', 'huggingface-hub[cli]'], check=True)
            
            print("Starting authentication process...")
            print("Please follow the instructions in the terminal.")
            print()
            
            # Run huggingface-cli login interactively
            result = subprocess.run(['huggingface-cli', 'login'], 
                                  stdin=None, stdout=None, stderr=None)
            
            if result.returncode == 0:
                print("✅ HuggingFace authentication successful!")
                logger.info("HuggingFace authentication completed successfully")
                
                # Verify authentication
                try:
                    result = subprocess.run(['huggingface-cli', 'whoami'], 
                                          capture_output=True, text=True, timeout=10)
                    if result.returncode == 0:
                        username = result.stdout.strip()
                        print(f"✅ Logged in as: {username}")
                        return True
                except:
                    pass
                
                return True
            else:
                print("❌ Authentication failed or cancelled")
                logger.warning("HuggingFace authentication failed")
                return False
                
        except subprocess.CalledProcessError as e:
            print(f"❌ Error during authentication: {e}")
            logger.error(f"HuggingFace authentication error: {e}")
            return False
        except Exception as e:
            print(f"❌ Unexpected error during authentication: {e}")
            logger.error(f"Unexpected authentication error: {e}")
            return False

    def check_huggingface_auth(self) -> bool:
        """Check if HuggingFace authentication is available."""
        try:
            from huggingface_hub import HfApi
            api = HfApi()
            # Try to get user info to check auth
            user_info = api.whoami()
            logger.info(f"HuggingFace authenticated as: {user_info['name']}")
            return True
        except Exception as e:
            logger.warning(f"HuggingFace authentication not available: {e}")
            return False
    
    def check_ollama_status(self) -> bool:
        """Check if Ollama is installed and running."""
        try:
            result = subprocess.run(['ollama', 'list'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                logger.info("Ollama is installed and running")
                return True
            else:
                logger.warning("Ollama installed but not running")
                return False
        except (subprocess.TimeoutExpired, FileNotFoundError):
            logger.warning("Ollama not found or not responding")
            return False
    
    def install_ollama(self) -> bool:
        """Install Ollama if not present."""
        try:
            logger.info("Installing Ollama...")
            # Download and install Ollama
            install_script = requests.get("https://ollama.ai/install.sh")
            if install_script.status_code == 200:
                result = subprocess.run(['bash', '-c', install_script.text], 
                                      capture_output=True, text=True, timeout=300)
                if result.returncode == 0:
                    logger.info("Ollama installed successfully")
                    return True
                else:
                    logger.error(f"Ollama installation failed: {result.stderr}")
                    return False
            else:
                logger.error("Failed to download Ollama installer")
                return False
        except Exception as e:
            logger.error(f"Error installing Ollama: {e}")
            return False
    
    def download_huggingface_model(self, model_name: str) -> bool:
        """Download a model from HuggingFace."""
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            
            config = self.models_config[model_name]
            hf_model = config["hf_model"]
            
            logger.info(f"Downloading {model_name} from HuggingFace: {hf_model}")
            
            # Check if auth is required
            if config.get("requires_auth", False):
                if not self.check_huggingface_auth():
                    logger.error(f"Model {model_name} requires HuggingFace authentication")
                    logger.info("Please run: huggingface-cli login")
                    return False
            
            # Create model directory
            model_dir = self.models_dir / model_name
            model_dir.mkdir(exist_ok=True)
            
            # Download tokenizer
            logger.info(f"Downloading tokenizer for {model_name}...")
            tokenizer = AutoTokenizer.from_pretrained(
                hf_model,
                cache_dir=str(model_dir / "tokenizer"),
                trust_remote_code=True
            )
            
            # Download model (this will cache it)
            logger.info(f"Downloading model weights for {model_name}...")
            model = AutoModelForCausalLM.from_pretrained(
                hf_model,
                cache_dir=str(model_dir / "model"),
                torch_dtype="auto",
                device_map="auto" if self._has_gpu() else "cpu",
                trust_remote_code=True
            )
            
            # Save model info
            model_info = {
                "model_name": model_name,
                "hf_model": hf_model,
                "download_date": str(Path().cwd()),
                "status": "downloaded",
                "size_gb": config["size_gb"],
                "local_path": str(model_dir)
            }
            
            with open(model_dir / "model_info.json", "w") as f:
                json.dump(model_info, f, indent=2)
            
            logger.info(f"Successfully downloaded {model_name}")
            return True
            
        except ImportError:
            logger.error("transformers library not available. Install with: pip install transformers torch")
            return False
        except Exception as e:
            logger.error(f"Error downloading {model_name}: {e}")
            return False
    
    def download_ollama_model(self, model_name: str) -> bool:
        """Download a model via Ollama."""
        try:
            config = self.models_config[model_name]
            ollama_model = config["ollama_model"]
            
            # Check if Ollama is available
            if not self.check_ollama_status():
                logger.info("Ollama not available, attempting to install...")
                if not self.install_ollama():
                    logger.error("Failed to install Ollama")
                    return False
            
            logger.info(f"Downloading {model_name} via Ollama: {ollama_model}")
            
            # Pull the model
            result = subprocess.run(['ollama', 'pull', ollama_model], 
                                  capture_output=True, text=True, timeout=1800)  # 30 min timeout
            
            if result.returncode == 0:
                # Save model info
                model_dir = self.models_dir / model_name
                model_dir.mkdir(exist_ok=True)
                
                model_info = {
                    "model_name": model_name,
                    "ollama_model": ollama_model,
                    "download_date": str(Path().cwd()),
                    "status": "downloaded",
                    "size_gb": config["size_gb"],
                    "type": "ollama"
                }
                
                with open(model_dir / "model_info.json", "w") as f:
                    json.dump(model_info, f, indent=2)
                
                logger.info(f"Successfully downloaded {model_name} via Ollama")
                return True
            else:
                logger.error(f"Failed to download {model_name}: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Error downloading {model_name} via Ollama: {e}")
            return False
    
    def download_model(self, model_name: str) -> bool:
        """Download a specific model."""
        if model_name not in self.models_config:
            logger.error(f"Unknown model: {model_name}")
            return False
        
        config = self.models_config[model_name]
        
        if "ollama_model" in config:
            return self.download_ollama_model(model_name)
        else:
            return self.download_huggingface_model(model_name)
    
    def download_all_models(self) -> Dict[str, bool]:
        """Download all configured models."""
        results = {}
        
        logger.info("Starting download of all models...")
        
        for model_name in self.models_config.keys():
            logger.info(f"Downloading {model_name}...")
            results[model_name] = self.download_model(model_name)
            
            if results[model_name]:
                logger.info(f"✅ {model_name} downloaded successfully")
            else:
                logger.error(f"❌ {model_name} download failed")
        
        return results
    
    def check_model_status(self, model_name: str) -> Dict[str, Any]:
        """Check the status of a specific model."""
        model_dir = self.models_dir / model_name
        model_info_file = model_dir / "model_info.json"
        
        status = {
            "model_name": model_name,
            "downloaded": False,
            "size_gb": self.models_config.get(model_name, {}).get("size_gb", 0),
            "description": self.models_config.get(model_name, {}).get("description", ""),
            "local_path": str(model_dir) if model_dir.exists() else None,
            "type": "ollama" if "ollama_model" in self.models_config.get(model_name, {}) else "huggingface"
        }
        
        if model_info_file.exists():
            try:
                with open(model_info_file, "r") as f:
                    model_info = json.load(f)
                status.update(model_info)
                status["downloaded"] = True
            except Exception as e:
                logger.warning(f"Error reading model info for {model_name}: {e}")
        
        # Additional checks for Ollama models
        if status["type"] == "ollama" and status["downloaded"]:
            if self.check_ollama_status():
                try:
                    result = subprocess.run(['ollama', 'list'], 
                                          capture_output=True, text=True, timeout=10)
                    if result.returncode == 0:
                        ollama_model = self.models_config[model_name].get("ollama_model", "")
                        if ollama_model in result.stdout:
                            status["ollama_available"] = True
                        else:
                            status["ollama_available"] = False
                            status["downloaded"] = False
                except Exception:
                    status["ollama_available"] = False
        
        return status
    
    def check_all_models_status(self) -> Dict[str, Dict[str, Any]]:
        """Check status of all models."""
        status_report = {}
        
        for model_name in self.models_config.keys():
            status_report[model_name] = self.check_model_status(model_name)
        
        return status_report
    
    def get_available_models(self) -> List[str]:
        """Get list of models that are downloaded and available."""
        available = []
        
        for model_name in self.models_config.keys():
            status = self.check_model_status(model_name)
            if status["downloaded"]:
                available.append(model_name)
        
        return available
    
    def _has_gpu(self) -> bool:
        """Check if GPU is available."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get system information for model compatibility."""
        info = {
            "gpu_available": self._has_gpu(),
            "ollama_available": self.check_ollama_status(),
            "hf_auth": self.check_huggingface_auth(),
            "total_models": len(self.models_config),
            "downloaded_models": len(self.get_available_models())
        }
        
        if info["gpu_available"]:
            try:
                import torch
                info["gpu_count"] = torch.cuda.device_count()
                info["gpu_memory"] = [torch.cuda.get_device_properties(i).total_memory // (1024**3) 
                                    for i in range(torch.cuda.device_count())]
            except Exception:
                pass
        
        return info


def main():
    """Test the model manager."""
    manager = ModelManager()
    
    # Check system info
    print("System Information:")
    print("=" * 50)
    system_info = manager.get_system_info()
    for key, value in system_info.items():
        print(f"{key}: {value}")
    
    # Check model status
    print("\nModel Status:")
    print("=" * 50)
    status_report = manager.check_all_models_status()
    
    for model_name, status in status_report.items():
        downloaded = "✅" if status["downloaded"] else "❌"
        print(f"{downloaded} {model_name}: {status['description']}")
        print(f"   Size: {status['size_gb']}GB, Type: {status['type']}")
        if status["downloaded"]:
            print(f"   Path: {status['local_path']}")
        print()


if __name__ == "__main__":
    main()
