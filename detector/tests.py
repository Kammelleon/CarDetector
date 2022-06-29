from django.test import TestCase
from django.test import Client


class YoloTestCase(TestCase):

    def test_redirect_to_file_uploader_on_wrong_model_type(self):
        c = Client()
        response = c.post('/model/', {'model': ['yolov5_model_that_doesnt_exist']})
        self.assertEqual(response.status_code, 302)

    def test_redirect_to_preview_on_correct_model_type(self):
        c = Client()
        with open('./detector/test_files/car.jpg', 'rb') as fp:
            c.post('', {'file': fp})
        response = c.post('/model/', {'model': ['yolov5n']})
        self.assertRedirects(response, '/preview/')
