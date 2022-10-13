#include <iostream>
#include <dlib/dnn.h>
#include <dlib/data_io.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include <dlib/opencv/cv_image.h>
#include <unistd.h>
#include <opencv2/opencv.hpp>


using namespace std;
using namespace cv;
using namespace dlib;



template <long num_filters, typename SUBNET> using con5d = con<num_filters,5,5,2,2,SUBNET>;
template <long num_filters, typename SUBNET> using con5  = con<num_filters,5,5,1,1,SUBNET>;
template <long num_filters, typename SUBNET> using con3  = con<num_filters,3,3,1,1,SUBNET>;

template <typename SUBNET> using downsampler  = relu<affine<con5d<32, relu<affine<con5d<32, relu<affine<con5d<16,SUBNET>>>>>>>>>;

template <typename SUBNET> using downsampler_bn  = relu<bn_con<con5d<32, relu<bn_con<con5d<32, relu<bn_con<con5d<32,SUBNET>>>>>>>>>;

template <typename SUBNET> using rcon5  = relu<affine<con5<55,SUBNET>>>;

template <typename SUBNET> using rcon3  = relu<affine<con3<32,SUBNET>>>;

template <typename SUBNET> using rcon3_bn  = relu<bn_con<con3<32,SUBNET>>>;

//using net_type = loss_mmod<con<1,6,6,1,1,rcon3<rcon3<rcon3<downsampler<input_rgb_image_pyramid<pyramid_down<20>>>>>>>>;

using net_type  = loss_mmod<con<1,6,6,1,1,rcon3_bn<rcon3_bn<rcon3_bn<downsampler_bn<input_rgb_image_pyramid<pyramid_down<6>>>>>>>>;

// ----------------------------------------------------------------------------------------

VideoCapture *cap = new VideoCapture();

int main(int argc, char** argv) try
{
    cout << "Starting" << endl;

    if (argc != 3)
    {
        
        cout << "Test the gun detection network.\n " << endl;
        cout << "gun_tester <neural network model> <video file or rtsp stream>" << endl;
        cout << endl;
        return 0;
    }
    //const std::string faces_directory = argv[1];
    const std::string netFile = argv[1];
    const std::string fileName = argv[2];
   
    //std::vector<matrix<rgb_pixel>> images_test;

    //std::vector<std::vector<mmod_rect>> face_boxes_test;
     
    //load_image_dataset(images_test, face_boxes_test, faces_directory+"/testing.xml");



    net_type net;



    deserialize(netFile.c_str()) >> net;



    //cout << "testing results:  " << test_object_detection_function(net, images_test, face_boxes_test) << endl;
 
    image_window win;

    //cout << "Debug 1" << endl;

    cap->open(fileName);

    //cout << "Debug 2" << endl;

    int counter = 0;

    int max_frames = 4;
 

    while(1)
    {
      Mat frame;
        Mat frame_resize;
      // Capture frame-by-frame
        cap->operator>>(frame);

        cv::resize(frame, frame_resize, cv::Size(320,180));

      if ( counter >= max_frames)
      {

      

        cv_image<bgr_pixel> image(frame_resize);
        
        matrix<rgb_pixel> dlibImage;

        // dlib::array2d<rgb_pixel> dlibImage;
        assign_image(dlibImage, image);

        auto&& img = dlibImage;

        

        pyramid_up(img);

        auto dets = net(img);

        win.clear_overlay();

        win.set_image(img);

        for (auto&& d : dets)
        {
	   double c = d.detection_confidence;
            win.add_overlay(d.rect);
            cout << "GUN DETECTED "  << c << endl;


        }

        
        counter = 0;
      }
      else 
      {

          counter++;
      }

      usleep(15000);
    }

/*
    for (auto&& img : images_test)
    {
        pyramid_up(img);
        auto dets = net(img);
        win.clear_overlay();
        win.set_image(img);
        for (auto&& d : dets)
            win.add_overlay(d);
        cin.get();
    }
    */
    return 0;


}
catch(std::exception& e)
{
    cout << e.what() << endl;
}
