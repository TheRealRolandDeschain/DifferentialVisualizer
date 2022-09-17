import cv2
import glob
import numpy as np
import os
import math
import json

source_folder = r"C:\Users\Johannes\source\repos\ProbelessLightDetection\assets\RenderScreenshots\s1"

def calulate_psnr_mse(i, k):

    nr_active_pixels = np.where(np.sum(i, 2) > 1)[0].shape[0]

    # copy to be able to work with bigger values
    I = np.array(i, copy=True).astype('int64')
    K = np.array(k, copy=True).astype('int64')

    diff_r = I[:, :, 0] - K[:, :, 0]
    sum_r = np.sum(diff_r * diff_r)
    diff_g = I[:, :, 1] - K[:, :, 1]
    sum_g = np.sum(diff_g * diff_g)
    diff_b = I[:, :, 2] - K[:, :, 2]
    sum_b = np.sum(diff_b * diff_b)

    mse_sh_r = sum_r / nr_active_pixels
    mse_sh_g = sum_g / nr_active_pixels
    mse_sh_b = sum_b / nr_active_pixels

    mse = (mse_sh_r + mse_sh_g + mse_sh_b) / 3.0
    psnr = 10 * math.log10(255 * 255 / mse)

    return mse, psnr

def process_images(gt_image, ar_image, sh_image):

    mse_ar, psnr_ar = calulate_psnr_mse(gt_image, ar_image)
    mse_sh, psnr_sh = calulate_psnr_mse(gt_image, sh_image)


    result = {
        "ar_mse": mse_ar,
        "ar_psnr": psnr_ar,
        "sh_mse": mse_sh,
        "sh_psnr": psnr_sh
    }

    # print(": {}; psnr: {}".format(mse_ar, psnr_ar))
    # print("sh: mse: {}; psnr: {}".format(mse_sh, psnr_sh))

    # print("test_ar: {}".format(cv2.PSNR(gt_image, ar_image)))
    # print("test_sh: {}".format(cv2.PSNR(gt_image, sh_image)))

    return result

def render_diff_image(i, k):

    diff = cv2.absdiff(i, k)
    diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    return cv2.applyColorMap(diff, cv2.COLORMAP_JET)


def main():
    img_paths = glob.glob('{}/*.png'.format(source_folder))
    gt_paths = [x for x in img_paths if "gt.png" in os.path.basename(x)]

    all_errors = {}


    for gt_path in gt_paths:

        key = os.path.basename(gt_path.replace("_gt.png", ""))

        ar_path = gt_path.replace("gt.png", "ar.png")
        sh_path = gt_path.replace("gt.png", "sh.png")

        gt_image = cv2.imread(gt_path)
        ar_image = cv2.imread(ar_path)
        sh_image = cv2.imread(sh_path)

        errors = process_images(gt_image, ar_image, sh_image)
        all_errors[key] = errors

        diff_ar = render_diff_image(gt_image, ar_image)
        diff_sh = render_diff_image(gt_image, sh_image)

        cv2.imwrite(ar_path.replace("ar.png", "ar_diff.png"), diff_ar)
        cv2.imwrite(sh_path.replace("sh.png", "sh_diff.png"), diff_sh)

    with open(source_folder + r"\errors.json", "w") as f:
        json.dump(all_errors, f, indent=4)





if __name__ == '__main__':
    main()
