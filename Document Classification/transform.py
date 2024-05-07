import cv2
import albumentations as A
from albumentations.core.transforms_interface import ImageOnlyTransform
from albumentations.pytorch import ToTensorV2

class NonLocalMeansDenoising(ImageOnlyTransform):
    def __init__(self, h=10, templateWindowSize=7, searchWindowSize=21, always_apply=False, p=0.5):
        super(NonLocalMeansDenoising, self).__init__(always_apply, p)
        self.h = h
        self.templateWindowSize = templateWindowSize
        self.searchWindowSize = searchWindowSize

    def apply(self, image, **params):
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        denoised_image = cv2.fastNlMeansDenoisingColored(image, None, self.h, self.h, self.templateWindowSize, self.searchWindowSize)
        denoised_image = cv2.cvtColor(denoised_image, cv2.COLOR_BGR2RGB)
        return denoised_image

def get_transforms():
    totensor_transform = A.Compose([A.Resize(380, 380), ToTensorV2()])
    test_transform = A.Compose([
        A.Resize(380, 380),
        NonLocalMeansDenoising(h=10, templateWindowSize=7, searchWindowSize=21, p=1.0), 
        ToTensorV2()
    ])
    return totensor_transform, test_transform
