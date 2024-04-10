#include <omp.h>
#include <iostream>
#include <opencv2/objdetect.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;

int main()
{
    CascadeClassifier face_cascade;
    if (!face_cascade.load(samples::findFile("C:/Users/zayka/Source/Repos/CV_NGTU/cascad/haarcascade_frontalface_default.xml"))) {
        return -1;
    }
    CascadeClassifier eye_cascade;
    if (!eye_cascade.load(samples::findFile("C:/Users/zayka/Source/Repos/CV_NGTU/cascad/haarcascade_eye_tree_eyeglasses.xml"))) {
        return -1;
    }
    CascadeClassifier smile_cascade;
    if (!smile_cascade.load(samples::findFile("C:/Users/zayka/Source/Repos/CV_NGTU/cascad/haarcascade_smile.xml"))) {
        return -1;
    }
    VideoCapture cap("C:/Users/zayka/Source/Repos/CV_NGTU/zua.mp4");
    if (!cap.isOpened()) {
        cout << "Video has problem" << endl;
        return -1;
    }

    VideoWriter out("C:/Users/zayka/Source/Repos/CV_NGTU/zua_out_omp.mp4", cap.get(CAP_PROP_FOURCC), cap.get(CAP_PROP_FPS), Size(cap.get(CAP_PROP_FRAME_WIDTH), cap.get(CAP_PROP_FRAME_HEIGHT)));

    Mat frame, new_image, gray_image;
#pragma omp parallel
    {
        while (true) {
            cap >> frame;
            if (frame.empty()) break;

            GaussianBlur(frame, new_image, Size(0, 0), 3);
            cvtColor(new_image, gray_image, COLOR_BGR2GRAY);

            vector<Rect> faces, eyes, smiles;

#pragma omp sections
            {
#pragma omp section
                {
                    face_cascade.detectMultiScale(gray_image, faces, 1.1, 5);
                }

#pragma omp section
                {
                    eye_cascade.detectMultiScale(gray_image, eyes, 1.1, 5);
                }

#pragma omp section
                {
                    smile_cascade.detectMultiScale(gray_image, smiles, 1.9, 25);
                }
            }

            new_image = frame.clone();
#pragma omp for
            for (int i = 0; i < faces.size(); i++) {
                rectangle(new_image, faces[i], Scalar(0, 255, 0), 2);
            }

#pragma omp for
            for (int i = 0; i < eyes.size(); i++) {
                Point eye_center(eyes[i].x + eyes[i].width / 2, eyes[i].y + eyes[i].height / 2);
                int radius = cvRound((eyes[i].width + eyes[i].height) * 0.25);
                circle(new_image, eye_center, radius, Scalar(255, 0, 0), 2);
            }

#pragma omp for
            for (int i = 0; i < smiles.size(); i++) {
                rectangle(new_image, smiles[i], Scalar(0, 0, 255), 2);
            }

#pragma omp critical
            {
                imshow("Work space", new_image);
                out.write(new_image);
            }

            char c = (char)waitKey(30);
            if (c == 27) break;

            if (c == 32) {
                while (true) {
                    char c = (char)waitKey(30);
                    if (c == 32) break;
                }
            }
        }
    }

    cap.release();
    out.release();
    destroyAllWindows();

    return 0;
}