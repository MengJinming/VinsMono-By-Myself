//; 2022/02/25 全部阅读完毕 

#include "parameters.h"

std::string IMAGE_TOPIC;
std::string IMU_TOPIC;
std::vector<std::string> CAM_NAMES;     //; 相机的名称，里面存的就是config.yaml
std::string FISHEYE_MASK;
int MAX_CNT;
int MIN_DIST;
int WINDOW_SIZE;
int FREQ;
double F_THRESHOLD;
int SHOW_TRACK;
int STEREO_TRACK;
int EQUALIZE;
int ROW;
int COL;
int FOCAL_LENGTH;
int FISHEYE;
bool PUB_THIS_FRAME;

template <typename T>
T readParam(ros::NodeHandle &n, std::string name)
{
    T ans;
    if (n.getParam(name, ans))  //; n.getParam是ros的内置函数，用来读入ROS的参数服务器中的参数
    {
        ROS_INFO_STREAM("Loaded " << name << ": " << ans);
    }
    else
    {
        ROS_ERROR_STREAM("Failed to load " << name);
        n.shutdown();       //; 没有注意到这里，shutdown之后是整个节点就关闭了吗？
    }
    return ans;
}

// 读配置参数，通过roslaunch文件的参数服务器获得
void readParameters(ros::NodeHandle &n)
{
    std::string config_file;
    // 首先获得配置文件的路径
    //; 这里的config_file就是ros参数服务器中的参数变量名称，最后得到的就是config.yaml文件的带路径的文件名
    config_file = readParam<std::string>(n, "config_file");
    // 使用opencv的yaml文件接口来读取文件
    cv::FileStorage fsSettings(config_file, cv::FileStorage::READ);
    if(!fsSettings.isOpened())
    {
        std::cerr << "ERROR: Wrong path to settings" << std::endl;
    }
    //; 这个命名全用大写，但是实际就是一个局部变量，所以感觉这里命名不是很规范
    std::string VINS_FOLDER_PATH = readParam<std::string>(n, "vins_folder");

    //; 以下这些变量全部用全局变量来存储。另外用了>>和=两种赋值写法，感觉也是不太规范
    //; 另外注意这里仅仅读取了和相机有关的参数，IMU有关的参数都没有读取
    fsSettings["image_topic"] >> IMAGE_TOPIC;
    fsSettings["imu_topic"] >> IMU_TOPIC;
    MAX_CNT = fsSettings["max_cnt"];
    MIN_DIST = fsSettings["min_dist"];
    ROW = fsSettings["image_height"];
    COL = fsSettings["image_width"];
    FREQ = fsSettings["freq"];  //; 这里的freq是发给后端的频率
    F_THRESHOLD = fsSettings["F_threshold"];
    SHOW_TRACK = fsSettings["show_track"];
    EQUALIZE = fsSettings["equalize"];  //; 是否做均衡化处理
    FISHEYE = fsSettings["fisheye"];
    if (FISHEYE == 1)
        FISHEYE_MASK = VINS_FOLDER_PATH + "config/fisheye_mask.jpg";
    CAM_NAMES.push_back(config_file);

    WINDOW_SIZE = 20;       //; 这个是滑窗中图像帧的数量？
    STEREO_TRACK = false;
    FOCAL_LENGTH = 460;
    PUB_THIS_FRAME = false;     //; 默认这一帧图像不发布

    if (FREQ == 0)
        FREQ = 100;

    fsSettings.release();


}
