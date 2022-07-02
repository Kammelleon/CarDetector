from django.test import TestCase
from django.test import Client


class YoloTestCase(TestCase):

    def test_redirect_to_file_uploader_on_wrong_model_type(self):
        c = Client()
        response = c.post('/model/', {'model': ['yolov5_model_that_doesnt_exist']})
        self.assertRedirects(response, '/')

    def test_redirect_to_preview_on_correct_yolo_model_type(self):
        c = Client()
        with open('./tests/test_files/car.jpg', 'rb') as car_image:
            c.post('/', {'image': car_image})
        response = c.post('/model/', {'model': ['yolov5n']})
        self.assertRedirects(response, '/preview/')

    def test_redirect_to_preview_on_correct_torch_pretrained_model_type(self):
        c = Client()
        with open('./tests/test_files/car.jpg', 'rb') as car_image:
            c.post('/', {'image': car_image})
        response = c.post('/model/', {'model': ['ssd300']})
        self.assertRedirects(response, '/preview/')
