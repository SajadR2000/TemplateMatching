import numpy as np
import cv2
import matplotlib.pyplot as plt


def normalized_cross_correlation(image, template):
    w, h = template.shape[1], template.shape[0]
    if len(image.shape) == 3:
        channels = image.shape[2]
    else:
        channels = 1
    normalized_cross_correlation_total = np.zeros(image.shape)
    for i in range(channels):
        # Working on i-th channel
        img_i = image[:, :, i]
        template_i = template[:, :, i]
        # Subtracting mean of i-th channel
        template_mean_i = np.mean(template_i)
        template_reduced_mean_i = template_i - template_mean_i
        img_window_mean_i = cv2.filter2D(img_i, cv2.CV_64F, (1. / (w * h)) * np.ones((h, w)))
        img_reduced_mean_i = img_i - img_window_mean_i
        # Computing NCC numerator which is CC of image and template
        normalized_cross_correlation_numerator_i = cv2.filter2D(img_reduced_mean_i, cv2.CV_64F, template_reduced_mean_i)
        # Norm of template when its mean is subtracted
        normalized_cross_correlation_denominator_first_term_i = np.sqrt(np.sum(np.square(template_reduced_mean_i)))
        # Windowed norm of image when its windowed mean is subtracted
        normalized_cross_correlation_denominator_second_term_i = np.sqrt(
            cv2.filter2D(np.square(img_reduced_mean_i), cv2.CV_64F, np.ones((h, w))))
        # Computing NCC denominator which is multiplication of above norms
        normalized_cross_correlation_denominator_i = normalized_cross_correlation_denominator_first_term_i * \
            normalized_cross_correlation_denominator_second_term_i
        # Finally calculating NCC
        normalized_cross_correlation_i = normalized_cross_correlation_numerator_i / \
            normalized_cross_correlation_denominator_i
        normalized_cross_correlation_total[:, :, i] = normalized_cross_correlation_i

    return normalized_cross_correlation_total


def template_matching(ncc, threshold):
    # Apply thresholding on all channels
    # Doing logical AND on the result of separate channels
    matches_ = np.ones(ncc.shape[:2])
    for i in range(3):
        matches_ = (ncc[:, :, i] > threshold[i]) * matches_

    return matches_


img = cv2.imread('Greek-ship.jpg', cv2.IMREAD_COLOR)
if img is None:
    raise Exception("Couldn't load the image")

my_template = cv2.imread('patch.png', cv2.IMREAD_COLOR)
if my_template is None:
    raise Exception("Couldn't load the my_template")

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float64)
template_rgb = cv2.cvtColor(my_template, cv2.COLOR_BGR2RGB).astype(np.float64)
# Cropping the template seems to improve the results for the patches above the ship
template_rgb_cropped = template_rgb[75:330, 70:130, :]

# I consider three types:
# First: normal image , cropped image
# Second: reduced mean image, cropped image
# Third: reduced mean image, scaled full image

THRESHOLD = [0.2, 0.2, 0.36]
THRESHOLD_2 = [0.2, 0.3, 0.3]
THRESHOLD_3 = [0.32, 0.45, 0.48]

ncc_mat = normalized_cross_correlation(img_rgb, template_rgb_cropped)
out = template_matching(ncc_mat, THRESHOLD)

ncc_mat = normalized_cross_correlation(img_rgb - np.mean(img_rgb), template_rgb_cropped)
out = np.logical_or(out.astype(bool), template_matching(ncc_mat, THRESHOLD_2).astype(bool)).astype(int)

scaled_template = cv2.resize(template_rgb, (int(template_rgb.shape[1]*10/26), int(template_rgb.shape[0]*180/305)))
ncc_mat = normalized_cross_correlation(img_rgb - img_rgb.mean(), scaled_template)
out2 = template_matching(ncc_mat, THRESHOLD_3)

# Dilation along vertical axis. This is to connect the found parts along vertical axis. And this can be used to remove
# the false positives.
kernel = np.ones((8, 1))
out = cv2.filter2D(out.astype(np.uint8), -1, kernel)
out[out >= 1] = 1

# This part draws the rectangles.
# Components shorter than some threshold are ignored.
matches = np.where(out == 1)
centers_x_pos = []
for loc in zip(*matches[::]):
    if (loc[0] + 80, loc[1]) not in zip(*matches[::]):
        continue
    flag = False
    for cx in centers_x_pos:
        if cx - 15 < loc[1] < cx + 15:
            flag = True
            break
    if flag:
        continue
    centers_x_pos.append(loc[1])
    top_left = (loc[0] - int(template_rgb.shape[0] / 2), loc[1] - int(template_rgb.shape[1] / 2))
    bottom_right = (loc[0] + int(template_rgb.shape[0] / 2), loc[1] + int(template_rgb.shape[1] / 2))
    cv2.rectangle(img_rgb,
                  (top_left[1], top_left[0]),
                  (bottom_right[1], bottom_right[0]),
                  (255, 0, 0),
                  5)

matches = np.where(out2 == 1)
for loc in zip(*matches[::]):
    if (loc[0] + 21, loc[1]) not in zip(*matches[::]):
        continue
    flag = False
    for cx in centers_x_pos:
        if cx - 10 < loc[1] < cx + 10:
            flag = True
            break
    if flag:
        continue
    # print(loc[1])
    centers_x_pos.append(loc[1])
    top_left = (loc[0] - int(template_rgb.shape[0] / 2), loc[1] - int(template_rgb.shape[1] / 2))
    bottom_right = (loc[0] + int(template_rgb.shape[0] / 2), loc[1] + int(template_rgb.shape[1] / 2))
    cv2.rectangle(img_rgb,
                  (top_left[1], top_left[0]),
                  (bottom_right[1], bottom_right[0]),
                  (255, 0, 0),
                  5)


plt.imsave('res15.jpg', img_rgb.astype(np.uint8))
