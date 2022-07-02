from django.test import TestCase
from django.test import Client


class UploaderTestCase(TestCase):
    def test_upload_damaged_image_file(self):
        c = Client()
        with open('./tests/test_files/image_damaged.png', 'rb') as image:
            response = c.post('/', {'image': image})
        self.assertInHTML('{"redirect_url": "/"}', response.content.decode())


    def test_upload_correct_image_file(self):
        c = Client()
        with open('./tests/test_files/car.jpg', 'rb') as image:
            response = c.post('/', {'image': image})
        self.assertInHTML('{"redirect_url": "/model/"}', response.content.decode())

    def test_upload_non_image_file(self):
        c = Client()
        with open('./tests/test_files/90_class_coco_dataset.pickle', 'rb') as image:
            response = c.post('/', {'image': image})
        self.assertInHTML('{"redirect_url": "/"}', response.content.decode())

class ModelChooserTestCase(TestCase):
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

