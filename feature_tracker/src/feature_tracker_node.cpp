//; 2022/02/25 全部阅读完毕 

#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/Imu.h>
#include <std_msgs/Bool.h>
#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>

#include "feature_tracker.h"

#define SHOW_UNDISTORTION 0     //; 是否显示去畸变后的图像，这个主要用于验证递归去畸变算法的效果

//; 以下三个变量好像没有使用
vector<uchar> r_status;
vector<float> r_err;
queue<sensor_msgs::ImageConstPtr> img_buf;

//; 发布者对象定义成全局变量，因为会在接收回调函数中进行消息发布，会调用这些对象
ros::Publisher pub_img,pub_match;
ros::Publisher pub_restart;

FeatureTracker trackerData[NUM_OF_CAM];     //; 一个存储FeatureTracker对象的数组
bool init_pub = 0;      //; 是否是第一次向后端发送数据的标志，0表示还没有向后端发送过数据
int pub_count = 1;      //; 已经发送给后端的图片的张数
bool first_image_flag = true;   //; 是否是第一张图片的标志
double first_image_time;        //; 第一张图片的时间戳
double last_image_time = 0;     //; 上一张图片的时间戳


// 图片的回调函数
void img_callback(const sensor_msgs::ImageConstPtr &img_msg)
{
    if(first_image_flag) // 对第一帧图像的基本操作
    {
        first_image_flag = false;
        first_image_time = img_msg->header.stamp.toSec();
        last_image_time = img_msg->header.stamp.toSec();
        return;
    }
    // detect unstable camera stream
    // 检查时间戳是否正常，这里认为超过一秒或者错乱就异常
    // 图像时间差太多光流追踪就会失败，这里没有描述子匹配，因此对时间戳要求就高
    if (img_msg->header.stamp.toSec() - last_image_time > 1.0 || img_msg->header.stamp.toSec() < last_image_time)
    {
        // 一些常规的reset操作
        ROS_WARN("image discontinue! reset the feature tracker!");
        first_image_flag = true; 
        last_image_time = 0;
        pub_count = 1;
        std_msgs::Bool restart_flag;
        restart_flag.data = true;
        pub_restart.publish(restart_flag);  // 告诉其他模块要重启了
        return;
    }
    last_image_time = img_msg->header.stamp.toSec();    //; 更新最新一帧图像的时间戳
    // frequency control
    // 控制一下发给后端的频率
    //; round是四舍五入取整
    if (round(1.0 * pub_count / (img_msg->header.stamp.toSec() - first_image_time)) <= FREQ)    // 保证发给后端的不超过这个频率
    {
        PUB_THIS_FRAME = true;
        // reset the frequency control
        // 这段时间的频率和预设频率十分接近，就认为这段时间很棒，重启一下，避免delta t太大
        //; 主要就是当t累计的比较长的时候，变化就不敏感了，也就是此时即使一下发多张图片，计算出来的频率
        //; 也不会突然变大。但是实际上此时一下给后端发送了多张图片，后端是处理不过来的
        if (abs(1.0 * pub_count / (img_msg->header.stamp.toSec() - first_image_time) - FREQ) < 0.01 * FREQ)
        {
            first_image_time = img_msg->header.stamp.toSec();
            pub_count = 0;
        }
    }
    
    
    else
        PUB_THIS_FRAME = false;

    // 即使不发布也是正常做光流追踪的！光流对图像的变化要求尽可能小

    cv_bridge::CvImageConstPtr ptr;
    // 把ros message转成cv::Mat
    if (img_msg->encoding == "8UC1")
    {
        sensor_msgs::Image img;
        img.header = img_msg->header;
        img.height = img_msg->height;
        img.width = img_msg->width;
        img.is_bigendian = img_msg->is_bigendian;
        img.step = img_msg->step;
        img.data = img_msg->data;
        img.encoding = "mono8";
        ptr = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::MONO8);
    }
    else
        ptr = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::MONO8);

    //; 将ros的img转换为opencv中的mat
    cv::Mat show_img = ptr->image;
    TicToc t_r;
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        ROS_DEBUG("processing camera %d", i);
        if (i != 1 || !STEREO_TRACK)   //; 单目相机，i=0，因此就执行这个语句
            //; readImage()函数就是真正进行光流追踪算法的函数
            trackerData[i].readImage(ptr->image.rowRange(ROW * i, ROW * (i + 1)), img_msg->header.stamp.toSec());
        else
        {
            if (EQUALIZE)
            {
                cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
                clahe->apply(ptr->image.rowRange(ROW * i, ROW * (i + 1)), trackerData[i].cur_img);
            }
            else
                trackerData[i].cur_img = ptr->image.rowRange(ROW * i, ROW * (i + 1));
        }

#if SHOW_UNDISTORTION
        trackerData[i].showUndistortion("undistrotion_" + std::to_string(i));
#endif
    }

    //; i是遍历所有的特征点，把其中id=-1的特征点，也就是后来提取的特征点的id更新
    for (unsigned int i = 0;; i++)
    {
        bool completed = false;
        for (int j = 0; j < NUM_OF_CAM; j++)
            if (j != 1 || !STEREO_TRACK)
                completed |= trackerData[j].updateID(i);    // 单目的情况下可以直接用=号
        if (!completed)
            break;
    }
    // 给后端喂数据
   if (PUB_THIS_FRAME)
   {
        pub_count++;    // 计数器更新
        //; 点云消息，具体消息类型的定义还需要看ros的文档
        sensor_msgs::PointCloudPtr feature_points(new sensor_msgs::PointCloud);
        sensor_msgs::ChannelFloat32 id_of_point;
        sensor_msgs::ChannelFloat32 u_of_point;
        sensor_msgs::ChannelFloat32 v_of_point;
        sensor_msgs::ChannelFloat32 velocity_x_of_point;
        sensor_msgs::ChannelFloat32 velocity_y_of_point;

        feature_points->header = img_msg->header;
        feature_points->header.frame_id = "world";   //; 点云的参考坐标系是世界坐标系

        vector<set<int>> hash_ids(NUM_OF_CAM);      //; 干什么的？
        for (int i = 0; i < NUM_OF_CAM; i++)
        {
            //; 下面这些引用得到的都是数组
            auto &un_pts = trackerData[i].cur_un_pts;   // 去畸变的归一化相机坐标系
            auto &cur_pts = trackerData[i].cur_pts; // 像素坐标
            auto &ids = trackerData[i].ids; // id
            auto &pts_velocity = trackerData[i].pts_velocity;   // 归一化坐标下的速度
            for (unsigned int j = 0; j < ids.size(); j++)
            {
                // 只发布追踪大于1的，因为等于1没法构成重投影约束，也没法三角化
                if (trackerData[i].track_cnt[j] > 1)
                {
                    int p_id = ids[j];
                    hash_ids[i].insert(p_id);   // 这个并没有用到
                    geometry_msgs::Point32 p;
                    p.x = un_pts[j].x;
                    p.y = un_pts[j].y;
                    p.z = 1;
                    // 利用这个ros消息的格式进行信息存储
                    feature_points->points.push_back(p);
                    id_of_point.values.push_back(p_id * NUM_OF_CAM + i);    //; 这里就是特征点的id
                    u_of_point.values.push_back(cur_pts[j].x);
                    v_of_point.values.push_back(cur_pts[j].y);
                    velocity_x_of_point.values.push_back(pts_velocity[j].x);
                    velocity_y_of_point.values.push_back(pts_velocity[j].y);
                }
            }
        }
        feature_points->channels.push_back(id_of_point);
        feature_points->channels.push_back(u_of_point);
        feature_points->channels.push_back(v_of_point);
        feature_points->channels.push_back(velocity_x_of_point);
        feature_points->channels.push_back(velocity_y_of_point);
        ROS_DEBUG("publish %f, at %f", feature_points->header.stamp.toSec(), ros::Time::now().toSec());
        // skip the first image; since no optical speed on frist image
        //; 第一帧图片特征点没有速度，因此不发布
        if (!init_pub)
        {
            init_pub = 1;
        }
        else
            pub_img.publish(feature_points);    // 前端得到的信息通过这个publisher发布出去

        // 可视化相关操作
        if (SHOW_TRACK)
        {
            //; 需要把转化后的cv::Mat的图片再转成ros的图片
            ptr = cv_bridge::cvtColor(ptr, sensor_msgs::image_encodings::BGR8);
            //cv::Mat stereo_img(ROW * NUM_OF_CAM, COL, CV_8UC3);
            cv::Mat stereo_img = ptr->image;

            for (int i = 0; i < NUM_OF_CAM; i++)
            {
                cv::Mat tmp_img = stereo_img.rowRange(i * ROW, (i + 1) * ROW);
                cv::cvtColor(show_img, tmp_img, CV_GRAY2RGB);

                for (unsigned int j = 0; j < trackerData[i].cur_pts.size(); j++)
                {
                    double len = std::min(1.0, 1.0 * trackerData[i].track_cnt[j] / WINDOW_SIZE);
                    cv::circle(tmp_img, trackerData[i].cur_pts[j], 2, cv::Scalar(255 * (1 - len), 0, 255 * len), 2);
                    //draw speed line
                    /*
                    Vector2d tmp_cur_un_pts (trackerData[i].cur_un_pts[j].x, trackerData[i].cur_un_pts[j].y);
                    Vector2d tmp_pts_velocity (trackerData[i].pts_velocity[j].x, trackerData[i].pts_velocity[j].y);
                    Vector3d tmp_prev_un_pts;
                    tmp_prev_un_pts.head(2) = tmp_cur_un_pts - 0.10 * tmp_pts_velocity;
                    tmp_prev_un_pts.z() = 1;
                    Vector2d tmp_prev_uv;
                    trackerData[i].m_camera->spaceToPlane(tmp_prev_un_pts, tmp_prev_uv);
                    cv::line(tmp_img, trackerData[i].cur_pts[j], cv::Point2f(tmp_prev_uv.x(), tmp_prev_uv.y()), cv::Scalar(255 , 0, 0), 1 , 8, 0);
                    */
                    //char name[10];
                    //sprintf(name, "%d", trackerData[i].ids[j]);
                    //cv::putText(tmp_img, name, trackerData[i].cur_pts[j], cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
                }
            }
            //cv::imshow("vis", stereo_img);
            //cv::waitKey(5);
            pub_match.publish(ptr->toImageMsg());   //; 向后端发布带有特征点的图像，主要用于显示用
        }
    }
    ROS_INFO("whole feature tracker processing costs: %f", t_r.toc());
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "feature_tracker");   // ros节点初始化
    ros::NodeHandle n("~"); // 声明一个句柄，～代表这个节点的命名空间
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);    // 设置ros log级别
    readParameters(n); //; 读取config.yaml配置文件，这里仅仅读取和相机有关的参数

    for (int i = 0; i < NUM_OF_CAM; i++)
        //; CAM_NAMES[i]存的就是config.yaml, 这个简单理解就是生成了一个相机模型的指针，并读取了相机内参
        trackerData[i].readIntrinsicParameter(CAM_NAMES[i]);    // 获得每个相机的内参

    //; 如果是鱼眼相机才执行下面语句，一般不是鱼眼相机，所以不执行
    if(FISHEYE)
    {
        for (int i = 0; i < NUM_OF_CAM; i++)
        {
            trackerData[i].fisheye_mask = cv::imread(FISHEYE_MASK, 0);
            if(!trackerData[i].fisheye_mask.data)
            {
                ROS_INFO("load mask fail");
                ROS_BREAK();
            }
            else
                ROS_INFO("load mask success");
        }
    }
    
    // 这个向roscore注册订阅这个topic，收到一次message就执行一次回调函数
    ros::Subscriber sub_img = n.subscribe(IMAGE_TOPIC, 100, img_callback);
    // 注册一些publisher
    //; 发布特征点信息，是后端真正可以使用到的数据
    pub_img = n.advertise<sensor_msgs::PointCloud>("feature", 1000); // 实际发出去的是 /feature_tracker/feature
    pub_match = n.advertise<sensor_msgs::Image>("feature_img",1000);  //; 发送带有特征点的图像，用于显示用
    pub_restart = n.advertise<std_msgs::Bool>("restart",1000);
    /*
    if (SHOW_TRACK)
        cv::namedWindow("vis", cv::WINDOW_NORMAL);
    */
    ros::spin();    // spin代表这个节点开始循环查询topic是否接收
    return 0;
}


// new points velocity is 0, pub or not?
// track cnt > 1 pub?