import io
import base64

from PIL import Image


def img_to_base64(img):
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue())
    img_str = "data:image/png;base64," + img_str.decode()
    return img_str


def base64_to_img(img_string):
    img = Image.open(io.BytesIO(base64.b64decode(img_string.split(',')[1])))
    return img
