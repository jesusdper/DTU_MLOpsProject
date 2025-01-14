import os
import requests
import logging
import yaml
from dotenv import load_dotenv

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

def upload_model(model_archive_path, model_name):
    """Uploads a model archive to the model registry."""
    logging.info(f"Uploading model {model_name} to the registry...")

    # Ensure the archive file exists
    if not os.path.exists(model_archive_path):
        logging.error(f"Model archive {model_archive_path} does not exist.")
        return

    # Open the model archive and upload
    try:
        with open(model_archive_path, 'rb') as model_file:
            files = {'file': (model_name, model_file, 'application/octet-stream')}
            headers = {'Authorization': f'Bearer {API_KEY}'}

            response = requests.post(f"{MODEL_REGISTRY_URL}/upload", files=files, headers=headers)

            if response.status_code == 200:
                logging.info(f"Model {model_name} uploaded successfully!")
            else:
                logging.error(f"Failed to upload model {model_name}. Status code: {response.status_code}")
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

if __name__ == "__main__":
    # Example usage
    model_path = r'C:\Users\jdiaz\Desktop\DTU_MLOpsProject\models\yolov8_voc5.zip'  # Ensure the path is correct
    model_name = 'yolov8_voc5'  # Replace with actual model name

    # Upload model to registry
    upload_model(model_path, model_name)

    # Get model status
    get_model_status(model_name)
