from util import *
import cv2
import csv
from pprint import pprint
import json


def scan(image):
    H, W = image.shape[:2]
    image = image[BUFFER_III:H - BUFFER_III, BUFFER_III:W - BUFFER_III]
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    bin_inv_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    # cv2.namedWindow('bin_inv_image', cv2.WINDOW_NORMAL)
    # cv2.imshow('bin_inv_image', bin_inv_image)
    # cv2.waitKey(0)
    h_lines_y = get_horizontal_lines(bin_inv_image)
    # draw_lines(image, h_lines_y, axis=1)
    if len(h_lines_y) == 5:
        logger.debug('找到 5 条横线')
        first_h_line_y = h_lines_y[0]
    elif len(h_lines_y) == 6:
        logger.debug('找到 6 条横线')
        first_h_line_y = h_lines_y[1]
    elif len(h_lines_y) == 7:
        logger.debug('找到 7 条横线')
        first_h_line_y = h_lines_y[2]
    else:
        logger.error('找到的横线数量和期望不匹配：找到 {} 期望 5 或者 7'.format(len(h_lines_y)))
        return '', '', ''
    v_lines_x = get_vertical_lines(bin_inv_image)
    # draw_lines(image, v_lines_x, axis=0)

    invoice_code_image = image[first_h_line_y - 160:first_h_line_y - 100, v_lines_x[0] + 180:v_lines_x[0] + 515]
    invoice_number_image = image[first_h_line_y - 160:first_h_line_y - 90, v_lines_x[-1] - 415: v_lines_x[-1] - 150]
    invoice_date_image = image[first_h_line_y - 55:first_h_line_y - 5, v_lines_x[-1] - 280:]
    # cv2.namedWindow('invoice_code_image', cv2.WINDOW_NORMAL)
    # cv2.imshow('invoice_code_image', invoice_code_image)
    # cv2.imshow('invoice_date_image', invoice_date_image)
    # cv2.imshow('invoice_number_image', invoice_number_image)
    # cv2.waitKey(0)
    concatenated_image = get_concatenated_image([invoice_code_image, invoice_number_image])
    # cv2.imshow('concatenated_image', concatenated_image)
    text = extract_text(concatenated_image, is_digit=True)
    lines = text.split('\n')
    if len(lines) == 2:
        invoice_code = lines[0]
        logger.info('invoice_code={}'.format(invoice_code))
        invoice_number = lines[1]
        logger.info('invoice_number={}'.format(invoice_number))
    else:
        invoice_code = ''
        invoice_number = ''
        logger.error('scanning invoice_code and invoice_number failed')
    invoice_date = extract_text(invoice_date_image)
    logger.info('invoice_date={}'.format(invoice_date))
    return invoice_code, invoice_number, invoice_date


def bulk_scan():
    scan_result = {'success': [], 'failure': [], 'error': []}
    for line in csv.reader(open('statistics.csv')):
        try:
            file_name = line[0]
            real_invoice_code = line[1]
            real_invoice_number = line[2]
            real_invoice_date = line[3]
            image = cv2.imread("tickets/" + file_name)
            invoice_code, invoice_number, invoice_date = scan(image)
            result = {}
            if invoice_code != real_invoice_code:
                logger.debug('Scanning invoice code of {} failed'.format(file_name))
                result[real_invoice_code] = invoice_code
            if invoice_number != real_invoice_number:
                logger.debug('Scanning invoice number of {} failed'.format(file_name))
                result[real_invoice_number] = invoice_number
            if invoice_date != real_invoice_date:
                logger.debug('Scanning invoice date of {} failed'.format(file_name))
                result[real_invoice_date] = invoice_date
            if not result:
                scan_result['success'].append(file_name)
            else:
                scan_result['failure'].append({file_name: result})
            # logger.debug(scan_result)
            logger.debug('success_num={}'.format(len(scan_result['success'])))
            logger.debug('failure_num={}'.format(len(scan_result['failure'])))
        except Exception as e:
            scan_result['error'].append(file_name)
            logger.exception('{} failed'.format(file_name))
    json.dump(scan_result, open('scan_result.txt', 'w'))
    return scan_result


# image = cv2.imread("tickets/2017092210555597.jpg")  # date
# image = cv2.imread('tickets/2017092509304674.jpg')
# scan(image)
bulk_scan()
