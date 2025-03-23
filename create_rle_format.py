import numpy as np
import cv2
import os

def main():
    src_path = "preds"
    output_fp = 'submission.csv'
    with open(output_fp, 'w') as f:
        f.write(construct_submission(src_path))

# From: https://www.kaggle.com/competitions/aaltoes-2025-computer-vision-v-1/data
def mask2rle(img):
    """
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formatted
    """
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

# From: https://www.kaggle.com/competitions/aaltoes-2025-computer-vision-v-1/data
def rle2mask(mask_rle: str, label=1, shape=(256, 256)):
    """
    mask_rle: run-length as string formatted (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background

    """
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = label
    return img.reshape(shape) 

def construct_submission(src_path):
    lines = []
    for img_fn in os.listdir(src_path):
        img_fp = os.path.join(src_path, img_fn)

        mask_from_file = cv2.imread(img_fp, cv2.IMREAD_GRAYSCALE)
        mask_from_file = (mask_from_file > 0).astype(np.uint8)

        mask_rle = mask2rle(mask_from_file)
        lines.append(','.join([img_fn.split('.')[0], mask_rle]))
    return '\n'.join(['ImageId,EncodedPixels', *lines])

if __name__ == '__main__':
    main()
