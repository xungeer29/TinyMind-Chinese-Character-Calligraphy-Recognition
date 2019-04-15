import sys
sys.path.append('/home/gfx/Projects/Tinymind')
from PIL import Image, ImageFilter

class MyGaussianBlur(ImageFilter.Filter):
    name = "GaussianBlur"

    def __init__(self, radius=2):
        self.radius = radius

    def filter(self, image):
        return image.gaussian_blur(self.radius)

if __name__ == '__main__':
    # test MyGaussianBlur
    path = '/media/gfx/data1/DATA/Tinymind/test1/5f8a3a2a9d45dd5ce09bc6c9859d1dc88662590f.jpg'
    im = Image.open(path)
    im = im.filter(MyGaussianBlur(radius=5))
    im.save('./figs/GaussianBlur.jpg')
