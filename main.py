from util import *
import cv2


def split(image):
    H, W = image.shape[:2]
    image = image[BUFFER_III:H - BUFFER_III, BUFFER_III:W - BUFFER_III]
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    bin_inv_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    # cv2.namedWindow('bin_inv_image', cv2.WINDOW_NORMAL)
    # cv2.imshow('bin_inv_image', bin_inv_image)
    # cv2.waitKey(0)
    h_lines_y = get_horizontal_lines(bin_inv_image)
    if len(h_lines_y) == 5:
        first_h_line_y = h_lines_y[0]
    elif len(h_lines_y) == 7:
        first_h_line_y = h_lines_y[2]
    else:
        logger.error('找到的横线数量和期望不匹配：找到 {} 期望 5 或者 7'.format(len(h_lines_y)))
    # draw_lines(image, h_lines_y, axis=1)
    v_lines_x = get_vertical_lines(bin_inv_image)
    # draw_lines(image, v_lines_x, axis=0)

    invoice_code_image = image[first_h_line_y - 160:first_h_line_y - 100, v_lines_x[0] + 180:v_lines_x[0] + 515]
    invoice_number_image = image[first_h_line_y - 160:first_h_line_y - 90, v_lines_x[-1] - 415: v_lines_x[-1] - 150]
    invoice_date_image = image[first_h_line_y - 55:first_h_line_y - 5, v_lines_x[-1] - 280:]
    cv2.namedWindow('invoice_code_image', cv2.WINDOW_NORMAL)
    cv2.imshow('invoice_code_image', invoice_code_image)
    cv2.imshow('invoice_date_image', invoice_date_image)
    cv2.imshow('invoice_number_image', invoice_number_image)
    concatenated_image = get_concatenated_image([invoice_code_image,invoice_number_image])
    cv2.imshow('concatenated_image', concatenated_image)
    text = extract_text(concatenated_image,is_digit=True)
    lines = text.split('\n')
    if len(lines) == 2:
        logger.info('invoice_code={}'.format(lines[0]))
        logger.info('invoice_number={}'.format(lines[1]))
    else:
        logger.error('scanning invoice_code and invoice_number failed')
    invoice_date = extract_text(invoice_date_image)
    logger.info('invoice_date={}'.format(invoice_date))
    cv2.waitKey(0)


image = cv2.imread("tickets/201709261041213.jpg")
split(image)
