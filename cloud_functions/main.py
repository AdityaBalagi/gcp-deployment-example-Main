import joblib
import numpy as np
import pandas as pd # Good practice if you might do more complex data handling
import os
from flask import request, jsonify # Import jsonify for proper JSON responses
from google.cloud import storage
import logging # For better logging

# Configure logging to output to Cloud Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Global Variables and Model Loading ---
# Initialize storage client globally to avoid re-creation on every invocation
storage_client = storage.Client()

# Define your bucket name and model path (consider using environment variables for these)
# For this example, we'll hardcode them based on your previous setup
BUCKET_NAME = "gcp-deployment-example-iris-models"
GCS_MODEL_PATH = "logistic_regression_v1.pkl"
LOCAL_MODEL_PATH = "/tmp/logistic_regression_v1.pkl" # ephemeral storage for function

# Global variable for the model
# Loading it outside the function ensures it's only loaded once per instance (after cold start)
model = None

# --- Model Download and Load Function ---
def _load_model():
    """
    Downloads the model from GCS if not already present in /tmp,
    then loads it into the global 'model' variable.
    This function is designed to be called once per function instance.
    """
    global model

    if model is None: # Only download and load if not already in memory
        try:
            # Ensure /tmp directory exists
            temp_dir = os.path.dirname(LOCAL_MODEL_PATH)
            if not os.path.exists(temp_dir):
                os.makedirs(temp_dir)

            # Check if model already downloaded to /tmp (useful for warm starts)
            if not os.path.exists(LOCAL_MODEL_PATH):
                logger.info(f"Downloading model from gs://{BUCKET_NAME}/{GCS_MODEL_PATH} to {LOCAL_MODEL_PATH}")
                bucket = storage_client.get_bucket(BUCKET_NAME)
                blob = bucket.blob(GCS_MODEL_PATH)
                blob.download_to_filename(LOCAL_MODEL_PATH)
                logger.info("Model downloaded successfully.")
            else:
                logger.info(f"Model already exists at {LOCAL_MODEL_PATH}, skipping download.")

            # Load the model into memory
            model = joblib.load(LOCAL_MODEL_PATH)
            logger.info("Model loaded into memory successfully.")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            # Re-raise the exception to indicate a critical error
            raise RuntimeError(f"Failed to initialize model: {e}")

# Call _load_model() outside the predict function to handle cold starts efficiently
# The first invocation will trigger this, and subsequent ones will use the loaded model.
try:
    _load_model()
except RuntimeError:
    # If model loading fails at startup, the function will not be healthy.
    # Cloud Functions will eventually show it as unhealthy or fail deployments.
    pass # Let Cloud Functions handle startup errors


# --- Main Prediction Function ---
def predict(request):
    """
    Responds to an HTTP request with a prediction from the Iris model.
    Handles input validation, error logging, and CORS.

    Args:
        request (flask.Request): The request object.
                                 <https://flask.palletsprojects.com/en/1.1.x/api/#incoming-request-data>
    Returns:
        A Flask response object with the prediction or an error message.
    """
    
    # --- CORS Preflight Handling ---
    # This is crucial for web applications calling your API from a different domain.
    if request.method == 'OPTIONS':
        headers = {
            'Access-Control-Allow-Origin': '*', # Adjust this to specific origins in production if needed
            'Access-Control-Allow-Methods': 'POST',
            'Access-Control-Allow-Headers': 'Content-Type',
            'Access-Control-Max-Age': '3600' # Cache preflight response for 1 hour
        }
        return ('', 204, headers) # 204 No Content for successful preflight

    # Set standard CORS headers for actual requests
    headers = {
        'Access-Control-Allow-Origin': '*' # Adjust this to specific origins in production if needed
    }

    # --- Model Availability Check ---
    # Ensure the model is loaded. This check is redundant if _load_model()
    # is called at the global scope, but acts as a safeguard.
    if model is None:
        logger.error("Model is not loaded. This indicates a critical startup issue.")
        return jsonify({"error": "Model not ready. Please try again shortly or contact support."}), 500, headers

    # --- Input Validation ---
    try:
        data_json = request.get_json(silent=True) # silent=True avoids exceptions for bad JSON parsing
        if not data_json:
            logger.warning("Received request with no JSON payload or invalid JSON.")
            return jsonify({"error": "Invalid JSON payload. Please provide a JSON object in the request body."}), 400, headers

        required_features = ["sepal_length_cm", "sepal_width_cm", "petal_length_cm", "petal_width_cm"]
        input_values = []
        for feature in required_features:
            if feature not in data_json or not isinstance(data_json[feature], (int, float)):
                logger.warning(f"Missing or invalid data for feature: {feature}")
                return jsonify({"error": f"Missing or invalid data for feature '{feature}'. All features must be numeric."}), 400, headers
            input_values.append(data_json[feature])

    except Exception as e:
        logger.error(f"Error parsing input data: {e}", exc_info=True) # exc_info for full traceback
        return jsonify({"error": f"Failed to parse input data: {str(e)}"}), 400, headers

    # --- Prediction ---
    try:
        # Convert input_values to a NumPy array for prediction
        input_array = np.array([input_values])
        
        # You can optionally convert to DataFrame if your model expects it,
        # but for simple scikit-learn models, a 2D numpy array is usually fine.
        # input_df = pd.DataFrame(input_array, columns=required_features)

        predictions = model.predict(input_array)
        
        # --- Output Formatting ---
        # Assuming Iris model returns a single integer for class (0, 1, 2)
        # You might want to map these to human-readable labels:
        iris_species_labels = {
            0: "Iris-setosa",
            1: "Iris-versicolor",
            2: "Iris-virginica"
        }
        predicted_label_id = int(predictions[0]) # Ensure it's an integer
        predicted_species = iris_species_labels.get(predicted_label_id, "Unknown Species")
        
        logger.info(f"Prediction made: Input {input_values} -> Predicted {predicted_species}")

        return jsonify({"prediction_id": predicted_label_id, "prediction_label": predicted_species, "input_data": data_json}), 200, headers

    except Exception as e:
        logger.error(f"Error during model prediction: {e}", exc_info=True)
        return jsonify({"error": f"An error occurred during prediction: {str(e)}"}), 500, headers