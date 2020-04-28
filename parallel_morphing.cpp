/*
This is the parallel version for morphing.cpp
Use the parallel_for_ form opnecv to do parallel computing
*/
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

class Line{
public:
    Point2f start_point;
    Point2f end_point;
    Mat vector;
    Mat perpendicular_vector;
    float length;
    float square_length;
    Line(Point2f start, Point2f end)
        : start_point(start), end_point(end)
        {
            vector = Mat(end_point - start_point).t();
            perpendicular_vector = Mat(1, 2, CV_32FC1);
            perpendicular_vector.at<float>(0) = vector.at<float>(1);
            perpendicular_vector.at<float>(1) = -vector.at<float>(0);
            square_length = pow(vector.at<float>(0), 2) + pow(vector.at<float>(1), 2);
            length = sqrt(square_length);
        }
    void print(){
        cout << "in class " << start_point << " " << end_point << " " << endl;
        cout << vector << " " << perpendicular_vector << " " << square_length << " " << length << endl;
    }
};

struct MouseParams{
    vector<Point2f> *points;
    vector<Line> *line_list;
    Mat *img;
};

void GetLineVector(int event, int x, int y, int flags, void* data){
    MouseParams* params = (MouseParams *)data;
    vector<Point2f> *point_list = params->points;
    vector<Line> *line_list = params->line_list;
    Mat *img = params->img;

    if(event == EVENT_LBUTTONDOWN){
        point_list->push_back(Point2f(y, x));
        circle(*img, Point2f(x, y), 3, Scalar(0, 0, 255), -1);
    
        int len = (int)point_list->size();
        if(!(len & 1) && len > 0){
            line(*img, Point2f(x, y), Point2f(point_list->at(len-2).y, point_list->at(len-2).x), Scalar(0, 0, 255), 2);
            Line line{point_list->at(len-2), Point2f(y, x)};
            line_list->push_back(line);
        }
    }    
}

vector<Line> LineInterpolator(vector<Line> src_lines, vector<Line> dst_lines, float t){
    vector<Line> inter_lines;
    int len = (int)src_lines.size();
    for(int i = 0; i < len; ++i){
        Point2f start = (1 - t) * src_lines[i].start_point + t * dst_lines[i].start_point;
        Point2f end = (1 - t) * src_lines[i].end_point + t * dst_lines[i].end_point;
        Line line(start, end);
        inter_lines.push_back(line);
    }

    return inter_lines;
}

Mat Mapping(Point2f current, Line src_vector, Line inter_vector, int p, int a, int b, float *weight){
    // cout << current << " " << src_vector.vector << " " << inter_vector.vector << endl;
    Mat P_Q_perpen = src_vector.perpendicular_vector.clone();
    Mat PQ_perpen = inter_vector.perpendicular_vector.clone();
    Point2f PQ_start = inter_vector.start_point;
    Mat P_Q_start = Mat(src_vector.start_point).t();
    // cout << P_Q_perpen << " " << PQ_perpen << " " << PQ_start << P_Q_start << endl;

    Mat PX = Mat(current - PQ_start).t();
    Mat PQ = inter_vector.vector.clone();
    float inter_len = inter_vector.square_length;
    // cout << PX << " " << PQ << " " << inter_len << endl;

    float u = PX.dot(PQ) / inter_len;
    float v = PX.dot(PQ_perpen) / inter_vector.length;
    // cout << u << " " << v << endl;

    Mat P_Q_ = src_vector.vector.clone();
    float src_len = src_vector.length;
    Mat Xt = Mat(src_vector.start_point).t() + u * P_Q_ + v * P_Q_perpen / src_len;
    // cout << P_Q_ << " " << src_len << " " << Xt << endl;

    float dist = 0;    
    if(u < 0){
        Mat temp = Xt - P_Q_start;
        dist = sqrt(temp.dot(temp));
    }
    else if(u > 0){
        Mat temp = Xt - Mat(src_vector.end_point).t();
        dist = sqrt(temp.dot(temp));
    }
    else    dist = abs(v);

    *weight = 0;
    float length = pow(inter_vector.length, p);
    *weight = pow((length / (a + dist)), b);

    Xt = Xt * (*weight);
    // cout << weight << Xt << endl;
    return Xt;
}

Vec3b bilinear(Mat img, Mat point, int h, int w){
    float x = point.at<float>(0), y = point.at<float>(1);
    float x1 = floor(x), x2 = ceil(x);
    float y1 = floor(y), y2 = ceil(y);
    
    if(x2 >= h) x2 = h - 1;
    if(y2 >= w) y2 = w - 1;

    float a = x - x1, b = y - y1;
    Vec3b pixel_val = (1 - a) * (1 - b) * img.at<Vec3b>(x1, y1)
                    + a * (1 - b) * img.at<Vec3b>(x2, y1)
                    + (1 - a) * b * img.at<Vec3b>(x1, y2)
                    + a * b * img.at<Vec3b>(x2, y2);

    return pixel_val;
}

/* class for parallel computing, modify from official document */
class ParallelWarp : public ParallelLoopBody{
public:
    ParallelWarp(Mat img, vector<Line> src, vector<Line> inter, int p_val, int a_val, int b_val, Mat &result)
    : src_img(img), src_lines(src), inter_lines(inter), result_img(result), p(p_val), a(a_val), b(b_val)
    {
        h = img.size().height;
        w = img.size().width;
        result_img = Mat::zeros(img.size(), img.type());
    }

    virtual void operator ()(const Range& range) const
    {
        for(int r = range.start; r < range.end; r++){
            int i = r / src_img.cols;
            int j = r % src_img.cols;

            Mat psum = Mat::zeros(1, 2, CV_32FC1);
            float wsum = 0;
            
            for(int k = 0; k < inter_lines.size(); ++k){
                float weight = 0;
                Mat Xt_weighted = Mapping(Point2f(i, j), src_lines[k], inter_lines[k], p, a, b, &weight);
                
                psum = psum + Xt_weighted;
                wsum = wsum + weight;
            }
            
            Mat point = psum / wsum;
            float x = point.at<float>(0);
            float y = point.at<float>(1);

            if(x < 0)   point.at<float>(0) = 0;
            else if(x >= h) point.at<float>(0) = h - 1;
            if(y < 0)   point.at<float>(1) = 0;
            else if(y >= w) point.at<float>(1) = w - 1;

            /* be careful, cannot directly assign the result from bilinear() to result_img because of const */
            Vec3b color = bilinear(src_img, point, h, w);
            result_img.ptr<Vec3b>(i)[j] = color;
        }
    }
    ParallelWarp& operator=(const ParallelWarp &){
        return *this;
    }

private:
    int h, w;
    int p, a, b;
    Mat src_img;
    Mat &result_img;
    vector<Line> src_lines;
    vector<Line> inter_lines;
};

Mat WarpImages(Mat img, vector<Line> src_lines, vector<Line> inter_lines, int p=0, int a=1, int b=2){
    int h = img.size().height;
    int w = img.size().width;
    Mat result = Mat::zeros(img.size(), img.type());

    // setNumThreads(8);    // set number of threads for parallel computing, if not set(default -1), then all the threads will be used
    /* the content is the computing of each pixel, pixels will be compute parallel by threads */
    #ifdef CV_CXX11
    parallel_for_(Range(0, h * w), [&](const Range& range){
        for(int r = range.start; r < range.end; r++){
            int i = r / img.cols;
            int j = r % img.cols;

            Mat psum = Mat::zeros(1, 2, CV_32FC1);
            float wsum = 0;
            
            for(int k = 0; k < inter_lines.size(); ++k){
                float weight = 0;
                Mat Xt_weighted = Mapping(Point2f(i, j), src_lines[k], inter_lines[k], p, a, b, &weight);
                
                psum = psum + Xt_weighted;
                wsum = wsum + weight;
            }
            
            Mat point = psum / wsum;
            float x = point.at<float>(0);
            float y = point.at<float>(1);

            if(x < 0)   point.at<float>(0) = 0;
            else if(x >= h) point.at<float>(0) = h - 1;
            if(y < 0)   point.at<float>(1) = 0;
            else if(y >= w) point.at<float>(1) = w - 1;

            result.at<Vec3b>(i, j) = bilinear(img, point, h, w);
        }
    });
    #else
    ParallelWarp parallelWarp(img, src_lines, inter_lines, p, a, b, result);
    parallel_for_(Range(0, h * w), parallelWarp);
    #endif

    return result;
}

int main(int argc, char** argv){
    const String keys = 
        "{help h usage ?|                   | show this message                     }"
        "{@img1         |./img/women.jpg    | path of src img                       }"
        "{@img2         |./img/cheetah.jpg  | path of dst img                       }"
        "{f             |101                | number of frames to generate animation}"
        "{p             |0                  | p value for computing weight          }"
        "{a             |1                  | a value for computing weight          }"
        "{b             |2                  | b value for computing weight          }"
    ;
    CommandLineParser parser(argc, argv, keys);
    if(parser.has("help")){
        parser.printMessage();
        return 0;
    }
    String image1 = parser.get<String>(0);
    String image2 = parser.get<String>(1);
    int frames = parser.get<int>("f");
    int p = parser.get<int>("p");
    int a = parser.get<int>("a");
    int b = parser.get<int>("b");
    vector<Point2f> src_point_list, dst_point_list;
    vector<Line> src_line_list, dst_line_list, inter_line_list;

    Mat src_img = imread(image1, IMREAD_COLOR);
    Mat src_origin = src_img.clone();

    Mat dst_img = imread(image2, IMREAD_COLOR);
    Mat dst_origin = dst_img.clone();
    int h = src_img.size().height;
    // cout << h << endl;
    int w = src_img.size().width;
    
    MouseParams params_src, params_dst;
    params_src.points = &src_point_list;
    params_src.line_list = &src_line_list;
    params_src.img = &src_img;

    params_dst.points = &dst_point_list;
    params_dst.line_list = &dst_line_list;
    params_dst.img = &dst_img;

    namedWindow("src img", WINDOW_AUTOSIZE);
    setMouseCallback("src img", GetLineVector, &params_src);
    namedWindow("dst img", WINDOW_AUTOSIZE);
    setMouseCallback("dst img", GetLineVector, &params_dst);
    while(true){
        imshow("src img", src_img);
        imshow("dst img", dst_img);
        if(waitKey(1) == 113) break;
    }
    cout << src_line_list.size() << " pairs of feature vectors" << endl;
    CV_Assert(src_line_list.size() == dst_line_list.size() && src_line_list.size() > 0 && dst_line_list.size() > 0);
    
    vector<Mat> animation;
    for(int i = 0; i < frames; ++i){
        Mat src_warp = Mat::zeros(src_img.size(), src_img.type());
        Mat dst_warp = Mat::zeros(src_img.size(), src_img.type());
        Mat result = Mat::zeros(src_img.size(), src_img.type());

        float t = i / (float)(frames - 1);
        if(i % 10 == 0)  cout << (i+1) << "/" << frames << " frame, t = " << t << endl;

        inter_line_list = LineInterpolator(src_line_list, dst_line_list, t);
        src_warp = WarpImages(src_origin, src_line_list, inter_line_list, p, a, b);
        dst_warp = WarpImages(dst_origin, dst_line_list, inter_line_list, p, a, b);

        /* parallel computing for blending each pixel */
        #ifdef CV_CXX11
        parallel_for_(Range(0, h * w), [&](const Range& range){
            for(int r = range.start; r < range.end; r++){
                int j = r / result.cols;
                int k = r % result.cols;
                result.at<Vec3b>(j, k) = (1 - t) * src_warp.at<Vec3b>(j, k) + t * dst_warp.at<Vec3b>(j, k);
            } 
        });
        #else
        for(int j = 0; j < h; j++){
            for(int k = 0; k < w; ++k){
                result.at<Vec3b>(j, k) = (1 - t) * src_warp.at<Vec3b>(j, k) + t * dst_warp.at<Vec3b>(j, k);
            }
        }
        #endif
        animation.push_back(result);
    }

    vector<Mat>::iterator begin = animation.begin();
    vector<Mat>::iterator end = animation.end();
    vector<Mat>::iterator it;
    for(it=begin; it != end; it++){
        imshow("Animation", *it);
        waitKey(100);
    }   

    destroyAllWindows();
    return 0;
}