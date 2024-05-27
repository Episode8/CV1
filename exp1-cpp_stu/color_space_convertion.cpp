#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

void RGB2YUV_enhance(Mat img, Mat& temp_YUV, Mat& res_rgb, float lightness_en = 3.5) {
    temp_YUV = Mat::zeros(img.size(), CV_8UC3);
    res_rgb = Mat::zeros(img.size(), CV_8UC3);
    Mat timg;
    img.convertTo(timg, CV_32FC3);

    for (int i = 0; i < timg.rows; i++) {
        for (int j = 0; j < timg.cols; j++) {
            /*##############################################################
                # Note that, should be careful about the RGB or BGR order
                # Hint: check the transformation matrix to convert RGB to YUV
                ##############################################################
                ## write your code here
                # Y =
                # U =
                # V =

                ## 1. save temp_YUV for visualization
#
#
#
#
#

                ## 2. enhance Y and convert YUV back to the RGB
#
#
#
#
#

                ## 3. store the enhanced RGB
# 
#
#
#
#

                #############################################################
                # end of your code
                #############################################################
                pass
                #############################################################
# (Optional) consider more efficent way to implement such a conversion
                #############################################################*/
        }
    }
    

}

int main(int argc, char** argv) {
    Mat img = imread("test/Lena.jpg");
    if (img.empty()) {
        cerr << "Image not found" << endl;
        return -1;
    }


    Mat imgyuv, res_rgb;
    RGB2YUV_enhance(img, imgyuv, res_rgb);
    vector<Mat> channels;
    split(imgyuv, channels);
    imshow("orginal image", img);
    imshow("Y", channels[0]);
    imshow("U", channels[1]);
    imshow("V", channels[2]);
    imshow("Enhance Light", res_rgb);
    waitKey(0);
    destroyAllWindows();
    return 0;
}
