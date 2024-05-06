import os
from PIL import Image
from tqdm import tqdm


def center_crop_box(w: int, h: int, crop_w: int, crop_h: int) -> tuple[int, int, int, int]:
    """
    Calculate the box coordinates to crop an image at its center.

    :param w: Width of the original image.
    :param h: Height of the original image.
    :param crop_w: Desired width of the cropped area.
    :param crop_h: Desired height of the cropped area.
    :return: A tuple of (left, top, right, bottom) which are the coordinates of the cropped box.
    """

    left = (w - crop_w) // 2
    top = (h - crop_h) // 2
    right = (w + crop_w) // 2
    bottom = (h + crop_h) // 2

    return left, top, right, bottom


def center_crop(folder: str, out_w: int, out_h: int):
    """
    Crop all images in a folder to a specified size, centering the crop area.

    :param folder: Directory containing images to be cropped.
    :param out_w: Output width of the cropped images.
    :param out_h: Output height of the cropped images.
    :return: None, but saves the cropped images in a subfolder.
    """

    if not os.path.exists(folder):
        return
    filenames = os.listdir(folder)

    target_folder = os.path.join(folder, f'cropped_{out_w}x{out_h}')
    os.mkdir(target_folder)

    for filename in tqdm(filenames):
        image = Image.open(os.path.join(folder, filename))
        box = center_crop_box(w=image.width, h=image.height, crop_w=out_w, crop_h=out_h)
        image.crop(box).save(os.path.join(target_folder, filename.split('.')[0] + '.png'))


def resize(folder: str, scale: float, mode = Image.BICUBIC):
    """
    Resize all images in a folder by a specified scale factor using a specified interpolation method.

    :param folder: Directory containing images to be resized.
    :param scale: Scale factor for resizing, e.g., 0.5 for half size.
    :param mode: Interpolation method to use for resizing.
    :return: None, but saves the resized images in a subfolder.
    """

    if not os.path.exists(folder):
        return
    filenames = os.listdir(folder)

    target_folder = os.path.join(folder, str(scale))
    os.mkdir(target_folder)

    for filename in tqdm(filenames):
        image = Image.open(os.path.join(folder, filename))
        out_w, out_h = round(image.width * scale), round(image.height * scale)
        image.resize((out_w, out_h), mode) \
            .save(os.path.join(target_folder, filename.split('.')[0] + '.png'))


def continuous_resize(folder: str, end_scale: float, steps: int = 1, mode=Image.BICUBIC):
    """
    Continuously resize images in a folder across multiple scale steps and crop to maintain a specific dimension.

    :param folder: Directory containing the images.
    :param end_scale: The final scale to reach after all steps.
    :param steps: Number of resizing steps.
    :param mode: Interpolation method to use for resizing.
    :return: None, but saves each step's resized image in a dedicated subfolder.
    """

    if not os.path.exists(folder):
        return

    filenames = os.listdir(folder)
    start_scale = 1
    step = (end_scale / start_scale) ** (1/steps)

    for filename in tqdm(filenames):
        target_folder = os.path.join(folder, filename.split(".")[0])
        os.mkdir(target_folder)

        scale = start_scale
        original_image = Image.open(os.path.join(folder, filename))
        in_w, in_h = original_image.width, original_image.height
        crop_w, crop_h = (in_w, in_h) if end_scale >= 1 else (in_w * end_scale / start_scale, in_h * end_scale / start_scale)

        for _ in range(steps):
            scale *= step

            out_w, out_h = round(in_w * scale), round(in_h * scale)
            box = center_crop_box(w=out_w, h=out_h, crop_w=crop_w, crop_h=crop_h)

            original_image.resize((out_w, out_h), mode)\
                .crop(box)\
                .save(os.path.join(target_folder, f'{filename.split(".")[0]}_x{scale:.3f}.png'))


if __name__ == '__main__':
    resize(folder="./inputs", scale=4, mode=Image.BICUBIC)
