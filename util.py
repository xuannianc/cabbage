import logging
import sys
import cv2
import pandas as pd
import numpy as np
import pytesseract

BUFFER_I = 5
BUFFER_II = 10
BUFFER_III = 20

logger = logging.getLogger('okra')
logger.setLevel(logging.DEBUG)  # default log level
format = logging.Formatter("%(asctime)s %(name)-8s %(levelname)-8s %(lineno)-4d %(message)s")  # output format
sh = logging.StreamHandler(stream=sys.stdout)  # output to standard output
sh.setFormatter(format)
logger.addHandler(sh)


def get_vertical_lines(image):
    """
    获取合并后所有竖线的 x 坐标
    :param image:要处理的已经二值化取反后的图像
    :return: 所有竖线的 x 坐标
    """
    all_combined_x = []
    H, W = image.shape
    # 获取所有竖线
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, H // BUFFER_II))
    veroded = cv2.erode(image, kernel, iterations=1)
    vdilated = cv2.dilate(veroded, kernel, iterations=1)
    # cv2.imshow('vdilated', vdilated)
    # cv2.waitKey(0)
    # 所有白点的 x 坐标
    all_x = np.where(vdilated == 255)[1]
    all_x_s = pd.Series(all_x)
    # 所有白点的 x 坐标统计，按 x 坐标排序
    x_statistics = all_x_s.value_counts().sort_index()
    indexes = x_statistics.index.tolist()
    if not indexes:
        logger.error('未找到竖线')
        return all_combined_x
    seq_begin = indexes[0]
    seq_end = indexes[0]
    for idx, i in enumerate(indexes[:-1]):
        # 不在 BUFFER 范围的 index
        if indexes[idx + 1] - i > BUFFER_I:
            seq_end = i
            seq_middle = (seq_begin + seq_end) // 2
            all_combined_x.append(seq_middle)
            seq_begin = indexes[idx + 1]
    seq_end = indexes[-1]
    seq_middle = (seq_begin + seq_end) // 2
    all_combined_x.append(seq_middle)
    logger.debug('all_combined_x={}'.format(all_combined_x))
    logger.debug('len_of_all_combined_x={}'.format(len(all_combined_x)))
    return all_combined_x


def get_horizontal_lines(image):
    """
    获取合并后所有横线的 y 坐标
    :param image: 要处理的已经二值化取反后的图像
    :return:
    """
    all_combined_y = []
    H, W = image.shape
    # 获取所有横线
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (W // 5, 1))
    heroded = cv2.erode(image, kernel, iterations=1)
    hdilated = cv2.dilate(heroded, kernel, iterations=1)
    # cv2.namedWindow('get_horizontal_lines:hdilated',cv2.WINDOW_NORMAL)
    # cv2.imshow('get_horizontal_lines:hdilated', hdilated)
    # cv2.waitKey(0)
    # 处理横线
    # 所有白点的 y 坐标
    all_y = np.where(hdilated == 255)[0]
    all_y_s = pd.Series(all_y)
    # 所有白点的 y 坐标的统计
    y_statistics = all_y_s.value_counts().sort_index()
    # print('y_statistics={}'.format(y_statistics))
    # 合并紧连的横线
    indexes = y_statistics.index.tolist()
    if not indexes:
        logger.error('未找到横线')
        return all_combined_y
    # print('y_statistics_indexes={}'.format(indexes))
    seq_begin = indexes[0]
    seq_end = indexes[0]
    for idx, i in enumerate(indexes[:-1]):
        # 不相连 index
        if indexes[idx + 1] - i > BUFFER_I:
            seq_end = i
            seq_middle = (seq_begin + seq_end) // 2
            all_combined_y.append(seq_middle)
            seq_begin = indexes[idx + 1]
    seq_end = indexes[-1]
    seq_middle = (seq_begin + seq_end) // 2
    all_combined_y.append(seq_middle)
    logger.debug('all_combined_y={}'.format(all_combined_y))
    logger.debug('len_of_all_combined_y={}'.format(len(all_combined_y)))
    return all_combined_y


def draw_lines(image, indexes, axis=0, width=2):
    H, W = image.shape[:2]
    if axis == 0:
        for idx in indexes:
            cv2.line(image, (idx, 0), (idx, H), (0, 0, 255), width)
    elif axis == 1:
        for idx in indexes:
            cv2.line(image, (0, idx), (W, idx), (0, 0, 255), width)
    cv2.namedWindow('lines', cv2.WINDOW_NORMAL)
    cv2.imshow('lines', image)
    cv2.waitKey(0)


def extract_text(image, lang='chi_sim', psm=6, oem=1, is_digit=False):
    if is_digit:
        text = pytesseract.image_to_string(image,
                                           config='--psm {} --oem {} -c tessedit_char_whitelist=0123456789'.format(psm,
                                                                                                                   0))
    else:
        text = pytesseract.image_to_string(image, lang=lang, config='--psm {} --oem {}'.format(psm, oem))
    return text


def get_concatenated_image(col_value_images):
    sumed_height = sum(col_value_image.shape[0] for col_value_image in col_value_images)
    max_width = max(col_value_image.shape[1] for col_value_image in col_value_images)
    concatenated_image = np.full((sumed_height, max_width, 3), 255, dtype=np.uint8)
    y = 0
    for col_value_image in col_value_images:
        h, w, d = col_value_image.shape
        concatenated_image[y:y + h, 0:w] = col_value_image
        y += h
    return concatenated_image
