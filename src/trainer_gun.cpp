#include <iostream>
#include <dlib/dnn.h>
#include <dlib/data_io.h>
#include <dlib/gui_widgets.h>
#include <iostream>
#include <fstream>

using namespace std;
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

void write_loss(double loss, long no_prog)
{
  ofstream myfile;
  myfile.open ("/var/www/html/status.json");
  myfile << "{\n\"loss\":" << loss << ",\n\"no_prog_steps\":" << no_prog  << "\n}";
  myfile.close();
}

int main(int argc, char** argv) try
{
    if (argc != 5)
    {
        cout << "Give the path to the examples/faces directory as the argument to this" << endl;
        cout << "program.  For example, if you are in the examples folder then execute " << endl;
        cout << "this program by running: " << endl;
        cout << "   ./trainer faces_directory training_xml_file gpu_frames_max max_epocs_timeout" << endl;
        cout << endl;
        return 0;
    }
    const std::string training_file = argv[1];

    const int num_chips = atoi(argv[2]);

    const int max_epocs = atoi(argv[3]);

    const int show_crops = atoi(argv[4]);
  
    std::vector<matrix<rgb_pixel>> image_data;
    
    std::vector<std::vector<mmod_rect>> image_boxes;
 
    load_image_dataset(image_data, image_boxes, training_file);

    cout << "num chips: " << num_chips << endl;

    cout << "num training images: " << image_data.size() << endl;

    mmod_options options(image_boxes, 20,20);

    cout << "num detector windows: "<< options.detector_windows.size() << endl;

    for (auto& w : options.detector_windows)
        cout << "detector window width by height: " << w.width << " x " << w.height << endl;

    cout << "overlap NMS IOU thresh:             " << options.overlaps_nms.get_iou_thresh() << endl;

    cout << "overlap NMS percent covered thresh: " << options.overlaps_nms.get_percent_covered_thresh() << endl;

    // Now we are ready to create our network and trainer.  
    net_type net(options);
    // The MMOD loss requires that the number of filters in the final network layer equal
    // options.detector_windows.size().  So we set that here as well.
    net.subnet().layer_details().set_num_filters(options.detector_windows.size());

    dnn_trainer<net_type> trainer(net);

    trainer.set_iterations_without_progress_threshold(max_epocs);

    trainer.set_learning_rate_shrink_factor(0.1);

    trainer.set_learning_rate(0.05);

    trainer.be_verbose();

    trainer.set_synchronization_file("ravenwatch_neural_network_sync", std::chrono::seconds(120));



    //#################Random Batch Samples populated with random_cropper#########################

    std::vector<matrix<rgb_pixel>> batch_data;

    std::vector<std::vector<mmod_rect>> batch_boxes; 
    
   
    //###############Random Cropper##################################### 
    
    random_cropper cropper;

    cropper.set_seed(time(0));

    cropper.set_chip_dims(180, 320);

    cropper.set_randomly_flip(true);  //true as default

    cropper.set_max_rotation_degrees(15); //.35 default

    cropper.set_min_object_size(20,20);

    cropper.set_background_crops_fraction(.70); //.50 default

    dlib::rand rnd;
    
    if (show_crops > 0 )
    {
        cropper(num_chips, image_data, image_boxes, batch_data, batch_boxes);
        
        image_window win;
        
        for (size_t i = 0; i < image_data.size(); ++i)
        {
            win.clear_overlay();
            win.set_image(batch_data[i]);
            for (auto b : batch_boxes[i])
            {
		//#############Let use know if any of the boxes are ignored
                if (b.ignore)
                    win.add_overlay(b.rect, rgb_pixel(255,255,0)); // draw ignored boxes as orange 
                else
                    win.add_overlay(b.rect, rgb_pixel(0,0,255));   //blue boxes are good 
            }

            cout << "Hit enter to view the next random crop.";

            cin.get();
        }

        
        return 0;
        
    }
    
    
    
    while(trainer.get_learning_rate() >= 1e-4)
    {
        cropper(num_chips, image_data, image_boxes, batch_data, batch_boxes);
        // We can also randomly jitter the colors and that often helps a detector
        // generalize better to new images. 
        for (auto&& img : batch_data)
            disturb_colors(img, rnd);

        trainer.train_one_step(batch_data, batch_boxes);

	write_loss(trainer.get_average_loss(),trainer.get_steps_without_progress());
	//printf("loss: %f\n",trainer.get_average_loss());
    }

    trainer.get_net();

    cout << "done training" << endl;

    net.clean();

    serialize("gun_detection_network.dat") << net;

}
catch(std::exception& e)
{
    cout << e.what() << endl;
}
