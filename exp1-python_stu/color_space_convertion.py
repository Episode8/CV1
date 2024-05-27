import numpy as np
import cv2
import math
import os

#def函数
def RGB2YUV_enhance(img, lightness_en=3.5):
    temp_YUV = np.zeros((img.shape[0], img.shape[1], 3), np.float32)
    res_RGB  = np.zeros((img.shape[0], img.shape[1], 3), np.float32)
    timg = img.astype(np.float32)
    for i in range(timg.shape[0]):
        for j in range(timg.shape[1]):
            ##############################################################
            # Note that, should be careful about the RGB or BGR order
            # Hint: check the transformation matrix to convert RGB to YUV
            ##############################################################
            ## write your code here
            # OpenCV uses BGR by default, so we need to adjust accordingly
            B = timg[i, j, 0]
            G = timg[i, j, 1]
            R = timg[i, j, 2]

            # Convert RGB to YUV
            Y = 0.299 * R + 0.587 * G + 0.114 * B
            U = -0.14713 * R - 0.28886 * G + 0.436 * B
            V = 0.615 * R - 0.51498 * G - 0.10001 * B

            ## 1. save temp_YUV for visualization
            temp_YUV[i, j, 0] = Y
            temp_YUV[i, j, 1] = U
            temp_YUV[i, j, 2] = V

            ## 2. enhance Y and convert YUV back to the RGB
            Y_enhanced = Y * lightness_en

            R_enhanced = Y_enhanced + 1.13983 * V
            G_enhanced = Y_enhanced - 0.39465 * U - 0.58060 * V
            B_enhanced = Y_enhanced + 2.03211 * U

            ## 3. store the enhanced RGB
            res_RGB[i, j, 0] = np.clip(B_enhanced, 0, 255)
            res_RGB[i, j, 1] = np.clip(G_enhanced, 0, 255)
            res_RGB[i, j, 2] = np.clip(R_enhanced, 0, 255)

    temp_YUV = temp_YUV.astype(np.uint8)
    res_RGB = res_RGB.astype(np.uint8)
    return temp_YUV, res_RGB

#主函数
if __name__ == '__main__':
    img = cv2.imread("Lena.jpg")
    imgyuv, res_rgb = RGB2YUV_enhance(img)
    cv2.imshow('Original Image', img)
    cv2.imshow('Y', imgyuv[:,:,0])
    cv2.imshow('U', imgyuv[:,:,1])
    cv2.imshow('V', imgyuv[:,:,2])
    cv2.imshow('Enhanced Light', res_rgb)
    cv2.imwrite("enhanced_image.jpg", res_rgb)  # Save the enhanced image
    cv2.waitKey(0)
    cv2.destroyAllWindows()
