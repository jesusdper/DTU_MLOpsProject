import os
import requests
import logging
import yaml
from dotenv import load_dotenv
import time

# Setup logging
logging.basicConfig(level=logging.INFO)

# Load configuration from config.yaml
def load_config(config_path=r'C:\Users\jdiaz\Desktop\DTU_MLOpsProject\configs\model_registry_config.yaml'):
    """Loads configuration from a YAML file."""
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
            logging.info("Configuration loaded successfully.")
            return config
    except Exception as e:
        logging.error(f"Error loading configuration: {e}")
        raise

# Load model registry settings
config = load_config()

# Load environment variables from .env file
load_dotenv()

# Define your model registry URL and API key from environment variables
MODEL_REGISTRY_URL = os.getenv('MODEL_REGISTRY_URL')
API_KEY = os.getenv('API_KEY')

def get_new_version(model_name):
    """Generate a version string for the model using timestamp."""
    current_time = time.strftime("%Y%m%d%H%M%S")  # Using timestamp as version
    return f"{model_name}_v{current_time}"

def upload_model(model_archive_path, model_name, api_key=None,model_registry_url=None):
    """Uploads a model archive to the model registry."""
    logging.info(f"Uploading model {model_name} to the registry...")

    # Ensure the archive file exists
    if not os.path.exists(model_archive_path):
        logging.error(f"Model archive {model_archive_path} does not exist.")
        return

    # Use the provided API key or fallback to the global API_KEY
    api_key = api_key or API_KEY
    model_registry_url = model_registry_url or MODEL_REGISTRY_URL

    # Check if API key is available
    if not api_key:
        logging.error("API key is not provided and not found in the environment.")
        return

    # Generate a version for the model
    model_version = get_new_version(model_name)
    logging.info(f"Uploading model version: {model_version}")

    # Open the model archive and upload
    try:
        with open(model_archive_path, 'rb') as model_file:
            files = {'file': (model_version, model_file, 'application/octet-stream')}
            headers = {'Authorization': f'Bearer {api_key}'}

            response = requests.post(f"{model_registry_url}/upload", files=files, headers=headers)

            if response.status_code == 200:
                logging.info(f"Model {model_version} uploaded successfully!")
            else:
                logging.error(f"Failed to upload model {model_version}. Status code: {response.status_code}")
                logging.error(response.text)
    except Exception as e:
        logging.error(f"Error uploading model: {e}")

def get_model_status(model_name):
    """Fetches the status of the model from the registry."""
    logging.info(f"Checking status of model {model_name}...")

    headers = {'Authorization': f'Bearer {API_KEY}'}
    response = requests.get(f"{MODEL_REGISTRY_URL}/status/{model_name}", headers=headers)

    if response.status_code == 200:
        status = response.json()
        logging.info(f"Model {model_name} status: {status}")
        return status
    else:
        logging.error(f"Failed to fetch status for model {model_name}. Status code: {response.status_code}")
        logging.error(response.text)

def upload_latest_model(model_path, model_name):
    """Automate uploading the latest model."""
    model_version = get_new_version(model_name)
    zip_model_path = f"{model_path}.zip"  # Ensure the model is compressed before upload
    upload_model(zip_model_path, model_version)

if __name__ == "__main__":
    # Example usage
    model_path = r'C:\Users\jdiaz\Desktop\DTU_MLOpsProject\models\yolov8_voc_test'  # Path to model directory (before zipping)
    model_name = 'yolov8_voc_test'  # Replace with actual model name

    # Compress and upload the latest model
    upload_latest_model(model_path, model_name)

    # Get model status
    get_model_status(model_name)