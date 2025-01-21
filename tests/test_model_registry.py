import unittest
from src.object_detection.model_registry_helper import upload_model, get_model_status


class TestModelRegistryHelper(unittest.TestCase):

    def test_upload_model(self):
        # Test uploading a model
        model_path = 'path/to/mock_model.pt'  # Replace with an actual mock model file
        model_name = 'test_model'
        response = upload_model(model_path, model_name)
        self.assertEqual(response.status_code, 200)

    def test_get_model_status(self):
        # Test getting model status
        model_name = 'test_model'
        status = get_model_status(model_name)
        self.assertIn('status', status)


if __name__ == '__main__':
    unittest.main()
