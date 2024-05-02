#include <stdio.h>
#include <queue>
#include <map>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

#include "estimator.h"
#include "parameters.h"
#include "utility/visualization.h"


Estimator estimator;    // 全局变量，状态估计器对象

std::condition_variable con;
double current_time = -1;   //; 全局变量，current_time存储的是上一帧imu数据的时间戳
queue<sensor_msgs::ImuConstPtr> imu_buf;
queue<sensor_msgs::PointCloudConstPtr> feature_buf;
queue<sensor_msgs::PointCloudConstPtr> relo_buf;
int sum_of_wait = 0;

std::mutex m_buf;
std::mutex m_state;
std::mutex i_buf;
std::mutex m_estimator;

double latest_time;

//; 以下五个是全局变量，存储暂时的状态变量
Eigen::Vector3d tmp_P;
Eigen::Quaterniond tmp_Q;
Eigen::Vector3d tmp_V;
Eigen::Vector3d tmp_Ba;
Eigen::Vector3d tmp_Bg;
//; 以下两个是全局变量，存储上一次的加速度和角速度值
Eigen::Vector3d acc_0;
Eigen::Vector3d gyr_0;

bool init_feature = 0;  //; 指示是否是第一帧的前端特征点数据，默认0是第一帧    
bool init_imu = 1;      //; 指示是否是第一帧imu数据，默认1是第一帧imu数据
double last_imu_t = 0;  //; 全局变量，存储上一帧imu的时间戳

/**
 * @brief 根据当前imu数据预测当前位姿
 * 
 * @param[in] imu_msg 
 */
//; 仅仅使用imu的值进行积分，得到的状态变量都存储在上面定义的全局变量中
void predict(const sensor_msgs::ImuConstPtr &imu_msg)
{
    double t = imu_msg->header.stamp.toSec();
    if (init_imu)
    {
        latest_time = t;
        init_imu = 0;
        return;
    }
    double dt = t - latest_time;
    latest_time = t;

    // 得到加速度
    double dx = imu_msg->linear_acceleration.x;
    double dy = imu_msg->linear_acceleration.y;
    double dz = imu_msg->linear_acceleration.z;
    Eigen::Vector3d linear_acceleration{dx, dy, dz};
    // 得到角速度
    double rx = imu_msg->angular_velocity.x;
    double ry = imu_msg->angular_velocity.y;
    double rz = imu_msg->angular_velocity.z;
    Eigen::Vector3d angular_velocity{rx, ry, rz};
    // 上一时刻世界坐标系下加速度值
    Eigen::Vector3d un_acc_0 = tmp_Q * (acc_0 - tmp_Ba) - estimator.g;  //; 转到世界坐标系下，减去重力加速度

    // 中值陀螺仪的结果
    Eigen::Vector3d un_gyr = 0.5 * (gyr_0 + angular_velocity) - tmp_Bg;
    // 更新姿态
    tmp_Q = tmp_Q * Utility::deltaQ(un_gyr * dt);
    // 当前时刻世界坐标系下的加速度值
    Eigen::Vector3d un_acc_1 = tmp_Q * (linear_acceleration - tmp_Ba) - estimator.g;
    // 加速度中值积分的值
    Eigen::Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
    // 经典物理中位置，速度更新方程
    tmp_P = tmp_P + dt * tmp_V + 0.5 * dt * dt * un_acc;
    tmp_V = tmp_V + dt * un_acc;

    acc_0 = linear_acceleration;
    gyr_0 = angular_velocity;
}

// 用最新VIO结果更新最新imu对应的位姿
void update()
{
    TicToc t_predict;
    latest_time = current_time;
    tmp_P = estimator.Ps[WINDOW_SIZE];
    tmp_Q = estimator.Rs[WINDOW_SIZE];
    tmp_V = estimator.Vs[WINDOW_SIZE];
    tmp_Ba = estimator.Bas[WINDOW_SIZE];
    tmp_Bg = estimator.Bgs[WINDOW_SIZE];
    acc_0 = estimator.acc_0;
    gyr_0 = estimator.gyr_0;
    
    //; 这里tmp_imu_buf就是当前帧图像后面的哪些imu数据，因为下一次的图像帧数据还没有来,
    //; 因此这些IMU数据就没有进行预积分操作，这里就单纯利用这些imu数据进行一个当前最新位姿的推算
    queue<sensor_msgs::ImuConstPtr> tmp_imu_buf = imu_buf;  // 遗留的imu的buffer，因为下面需要pop，所以copy了一份
    for (sensor_msgs::ImuConstPtr tmp_imu_msg; !tmp_imu_buf.empty(); tmp_imu_buf.pop())
        // 得到最新imu时刻的可靠的位姿
        predict(tmp_imu_buf.front());
}

// 获得匹配好的图像imu组
std::vector<std::pair<std::vector<sensor_msgs::ImuConstPtr>, sensor_msgs::PointCloudConstPtr>>
getMeasurements()
{
    std::vector<std::pair<std::vector<sensor_msgs::ImuConstPtr>, sensor_msgs::PointCloudConstPtr>> measurements;

    while (true)
    {
        //; 判断1
        if (imu_buf.empty() || feature_buf.empty())
            return measurements;
        // imu   *******
        // image          *****
        // 这就是imu还没来
        //; 判断2，注意这里是！，也就是满足条件是imu <= feature
        if (!(imu_buf.back()->header.stamp.toSec() > feature_buf.front()->header.stamp.toSec() + estimator.td))
        {
            //ROS_WARN("wait for imu, only should happen at the beginning");
            sum_of_wait++;  //; 这个变量有什么用？
            return measurements;
        }
        // imu        ****
        // image    ******
        // 这种只能扔掉一些image帧
        //; 判断3, 也就是imu第一帧时间 > 图像第一帧时间，那么就要把图像第一帧丢掉
        if (!(imu_buf.front()->header.stamp.toSec() < feature_buf.front()->header.stamp.toSec() + estimator.td))
        {
            ROS_WARN("throw img, only should happen at the beginning");
            feature_buf.pop();
            continue;
        }
        // 此时就保证了图像前一定有imu数据
        //; 上面种种判断，就是为了保证第一帧图像前面有imu数据，这样才能按照图像帧来截取数据包
        sensor_msgs::PointCloudConstPtr img_msg = feature_buf.front();
        feature_buf.pop();

        //; 读取buf中的数据
        // 一般第一帧不会严格对齐，但是后面就都会对齐，当然第一帧也不会用到
        std::vector<sensor_msgs::ImuConstPtr> IMUs;
        //; 存储在图像帧数据之前的所有imu数据
        while (imu_buf.front()->header.stamp.toSec() < img_msg->header.stamp.toSec() + estimator.td)
        {
            //; emplace_back和push_back有什么区别？
            IMUs.emplace_back(imu_buf.front());
            imu_buf.pop();
        }
        // 保留图像时间戳后一个imu数据，但不会从buffer中扔掉
        // imu    *   *
        // image    *
        IMUs.emplace_back(imu_buf.front());
        if (IMUs.empty())
            ROS_WARN("no imu between two image");
        measurements.emplace_back(IMUs, img_msg);
    }
    return measurements;
}


/**
 * @brief imu消息存进buffer，同时按照imu频率预测位姿并发送，这样就可以提高里程计频率
 * 
 * @param[in] imu_msg 
 */
void imu_callback(const sensor_msgs::ImuConstPtr &imu_msg)
{
    //; last_imu_t 全局变量，上一次imu的时间戳
    if (imu_msg->header.stamp.toSec() <= last_imu_t)    
    {
        ROS_WARN("imu message in disorder!");
        return;
    }
    last_imu_t = imu_msg->header.stamp.toSec();
    // 讲一下线程锁 条件变量用法
    //; 注意这里是手动加锁，即在处理数据前std::mutex::lock,处理完数据之后在unlock
    m_buf.lock();   //; std::mutex m_buf;
    imu_buf.push(imu_msg);  //; queue<sensor_msgs::ImuConstPtr> imu_buf;
    m_buf.unlock();
    con.notify_one();   //; std::condition_variable con;

    last_imu_t = imu_msg->header.stamp.toSec(); //; 这里重复设置时间了

    {
        std::lock_guard<std::mutex> lg(m_state);  //; std::mutex m_state;
        predict(imu_msg);   //; 仅仅使用IMU进行积分得到状态变量，存储在全局变量中
        std_msgs::Header header = imu_msg->header;
        header.frame_id = "world";
        // 只有初始化完成后才发送当前结果
        if (estimator.solver_flag == Estimator::SolverFlag::NON_LINEAR)
            //; 发布仅仅使用imu积分的结果，不知道这个结果有什么用？
            pubLatestOdometry(tmp_P, tmp_Q, tmp_V, header);
    }
}

/**
 * @brief 将前端信息送进buffer
 * 
 * @param[in] feature_msg 
 */
void feature_callback(const sensor_msgs::PointCloudConstPtr &feature_msg)
{
    if (!init_feature)
    {
        //skip the first detected feature, which doesn't contain optical flow speed
        //; 感觉这里跳过是有点问题的，因为前端发送的时候就已经有过类似的判断了，如果是第一帧的数据由于特征点速度
        //; 是0就不发布。这里还判断第一帧就不接受，实际上这个第一帧已经是第二帧了，已经有速度了
        init_feature = 1;
        return;
    }
    m_buf.lock();
    feature_buf.push(feature_msg);  //; queue<sensor_msgs::PointCloudConstPtr> feature_buf;
    m_buf.unlock();
    con.notify_one();
}

/**
 * @brief 将vins估计器复位
 * 
 * @param[in] restart_msg 
 */

void restart_callback(const std_msgs::BoolConstPtr &restart_msg)
{
    if (restart_msg->data == true)
    {
        ROS_WARN("restart the estimator!");
        m_buf.lock();
        while(!feature_buf.empty())
            feature_buf.pop();
        while(!imu_buf.empty())
            imu_buf.pop();
        m_buf.unlock();
        m_estimator.lock();
        estimator.clearState();
        estimator.setParameter();
        m_estimator.unlock();
        current_time = -1;
        last_imu_t = 0;
    }
    return;
}

void relocalization_callback(const sensor_msgs::PointCloudConstPtr &points_msg)
{
    //printf("relocalization callback! \n");
    m_buf.lock();
    relo_buf.push(points_msg);
    m_buf.unlock();
}

/**
 * @brief 主循环线程，完成所有处理逻辑
 */
void process()
{
    while (true)  // 这个线程是会一直循环下去
    {
        //; 以图像帧作为索引，每帧图像匹配很多帧的IMU数据
        std::vector<std::pair<std::vector<sensor_msgs::ImuConstPtr>, sensor_msgs::PointCloudConstPtr>> measurements;
        //; 下面的线程锁没有看懂
        std::unique_lock<std::mutex> lk(m_buf);
        //; 应该是条件锁的意思，符合条件就唤醒process线程中的getMeasurements()函数
        con.wait(lk, [&]
                 {
                    return (measurements = getMeasurements()).size() != 0;
                 });
        //; 能够执行到下面这一句，说明getMeasurements已经拿到数据了，所以本线程继续持有线程锁。
        //; 但是下面会先处理拿到的数据，所以需要手动释放线程锁，给imu/feature回调线程继续塞数据
        lk.unlock();    // 数据buffer的锁解锁，回调可以继续塞数据了
        m_estimator.lock(); // 进行后端求解，不能和复位重启冲突

        // 给予范围的for循环，这里就是遍历每组image imu组合
        for (auto &measurement : measurements)
        {
            auto img_msg = measurement.second;  //; 图像数据，实际上是前端处理得到的点云数据
            double dx = 0, dy = 0, dz = 0, rx = 0, ry = 0, rz = 0;
            // 遍历imu
            // Step 1 首先处理所有的IMU数据，对IMU数据进行预积分
            for (auto &imu_msg : measurement.first)
            {
                double t = imu_msg->header.stamp.toSec();   //; IMU数据的时间戳 
                double img_t = img_msg->header.stamp.toSec() + estimator.td;    //; 图像数据的时间戳
                if (t <= img_t)
                { 
                    if (current_time < 0)
                        current_time = t;   //; 当前的imu时间
                    double dt = t - current_time;
                    ROS_ASSERT(dt >= 0);
                    current_time = t;       //; current_time存储的是上一帧imu数据的时间戳
                    dx = imu_msg->linear_acceleration.x;
                    dy = imu_msg->linear_acceleration.y;
                    dz = imu_msg->linear_acceleration.z;
                    rx = imu_msg->angular_velocity.x;
                    ry = imu_msg->angular_velocity.y;
                    rz = imu_msg->angular_velocity.z;
                    // 时间差和imu数据送进去
                    //; 传入的三个变量分别是当前imu和上一帧imu的时间差，当前帧IMU的加速度，角速度
                    estimator.processIMU(dt, Vector3d(dx, dy, dz), Vector3d(rx, ry, rz));
                    //printf("imu: dt:%f a: %f %f %f w: %f %f %f\n",dt, dx, dy, dz, rx, ry, rz);
                }
                else    // 这就是针对最后一个imu数据，需要做一个简单的线性插值
                {
                    double dt_1 = img_t - current_time;
                    double dt_2 = t - img_t;
                    current_time = img_t;   //; 上一帧imu时间戳就赋值成图像时间戳，因为这里把imu数据对齐到图像时间戳了
                    ROS_ASSERT(dt_1 >= 0);
                    ROS_ASSERT(dt_2 >= 0);
                    ROS_ASSERT(dt_1 + dt_2 > 0);
                    double w1 = dt_2 / (dt_1 + dt_2);
                    double w2 = dt_1 / (dt_1 + dt_2);
                    //; 对imu数据进行线性插值
                    dx = w1 * dx + w2 * imu_msg->linear_acceleration.x;
                    dy = w1 * dy + w2 * imu_msg->linear_acceleration.y;
                    dz = w1 * dz + w2 * imu_msg->linear_acceleration.z;
                    rx = w1 * rx + w2 * imu_msg->angular_velocity.x;
                    ry = w1 * ry + w2 * imu_msg->angular_velocity.y;
                    rz = w1 * rz + w2 * imu_msg->angular_velocity.z;
                    estimator.processIMU(dt_1, Vector3d(dx, dy, dz), Vector3d(rx, ry, rz));
                    //printf("dimu: dt:%f a: %f %f %f w: %f %f %f\n",dt_1, dx, dy, dz, rx, ry, rz);
                }
            }

            // Step 2 再检测是否有回环产生
            // set relocalization frame
            // 回环相关部分
            sensor_msgs::PointCloudConstPtr relo_msg = NULL;
            while (!relo_buf.empty())   // 取出最新的回环帧
            {
                relo_msg = relo_buf.front();
                relo_buf.pop();
            }
            if (relo_msg != NULL)   // 有效回环信息
            {
                vector<Vector3d> match_points;
                double frame_stamp = relo_msg->header.stamp.toSec();    // 回环的当前帧时间戳
                for (unsigned int i = 0; i < relo_msg->points.size(); i++)
                {
                    Vector3d u_v_id;
                    u_v_id.x() = relo_msg->points[i].x; // 回环帧的归一化坐标和地图点idx
                    u_v_id.y() = relo_msg->points[i].y;
                    u_v_id.z() = relo_msg->points[i].z;
                    match_points.push_back(u_v_id);
                }
                // 回环帧的位姿
                Vector3d relo_t(relo_msg->channels[0].values[0], relo_msg->channels[0].values[1], relo_msg->channels[0].values[2]);
                Quaterniond relo_q(relo_msg->channels[0].values[3], relo_msg->channels[0].values[4], relo_msg->channels[0].values[5], relo_msg->channels[0].values[6]);
                Matrix3d relo_r = relo_q.toRotationMatrix();
                int frame_index;
                frame_index = relo_msg->channels[0].values[7];
                estimator.setReloFrame(frame_stamp, frame_index, match_points, relo_t, relo_r);
            }

            ROS_DEBUG("processing vision data with stamp %f \n", img_msg->header.stamp.toSec());

            // Step 3 开始处理图像数据
            TicToc t_s;
            //; image的格式是：特征点id，[相机id,特征点信息（归一化相机坐标，像素坐标，速度）]。实际上单目相机id始终是0
            //; 这里map的中的值为什么要用vector呢？这里image是局部变量，每次图像发来的所有特征点id不可能有重复的啊？
            //; 所以说map中的键不会有重复的，这样每个键只可能对应一个值，那么值就没必要用vector啊？
            map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> image;
            //; img_msg并不是图像， 而是前端处理后得到的点云数据
            for (unsigned int i = 0; i < img_msg->points.size(); i++)
            {
                //; channels[0]存储的是特征点的id。但是这里没明白为什么+0.5，本来id就是int数啊？
                //; 前端处理id的时候，公式是id = p_id * NUM_OF_CAM + i, pid是真正的特征点id, i是轮询相机的序号
                int v = img_msg->channels[0].values[i] + 0.5;   
                //; 所以下面取整得到特征点id, 取余得到相机的id
                int feature_id = v / NUM_OF_CAM;
                int camera_id = v % NUM_OF_CAM;
                double x = img_msg->points[i].x;    // 去畸变后归一化像素坐标
                double y = img_msg->points[i].y;
                double z = img_msg->points[i].z;
                double p_u = img_msg->channels[1].values[i];    //; 特征点像素坐标, 感觉这个没用？后面看看有没有用
                double p_v = img_msg->channels[2].values[i];
                double velocity_x = img_msg->channels[3].values[i]; // 特征点速度
                double velocity_y = img_msg->channels[4].values[i];
                ROS_ASSERT(z == 1); // 检查是不是归一化
                Eigen::Matrix<double, 7, 1> xyz_uv_velocity;
                xyz_uv_velocity << x, y, z, p_u, p_v, velocity_x, velocity_y;
                image[feature_id].emplace_back(camera_id,  xyz_uv_velocity);
            }
            // Step 处理图像数据的主函数，里面的步骤很多，非常重要！
            estimator.processImage(image, img_msg->header);

            // Step 4 主要工作基本完成，进行一些其他细节工作，主要是发布ros topic，用于ros的可视化
            // 一些打印以及topic的发送
            double whole_t = t_s.toc();
            printStatistics(estimator, whole_t);
            std_msgs::Header header = img_msg->header;
            header.frame_id = "world";

            pubOdometry(estimator, header);
            pubKeyPoses(estimator, header);
            pubCameraPose(estimator, header);
            pubPointCloud(estimator, header);
            pubTF(estimator, header);
            pubKeyframe(estimator);
            if (relo_msg != NULL)
                pubRelocalization(estimator);
            //ROS_ERROR("end: %f, at %f", img_msg->header.stamp.toSec(), ros::Time::now().toSec());
        }
        m_estimator.unlock();
        m_buf.lock();
        m_state.lock();
        //; 如果当前后端求解器已经进入了非线性优化阶段（即初始化已经完成），那么还要更新一下本文件中的全局位姿变量
        //; 主要是用于基于滑出中最新帧的位姿完全依靠IMU的当前最新位姿推算，就是为了得到一个比较高频的位姿信息
        if (estimator.solver_flag == Estimator::SolverFlag::NON_LINEAR)
            update();
        m_state.unlock();
        m_buf.unlock();
    }
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "vins_estimator");
    ros::NodeHandle n("~");
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);
    
    // Step 1 读取配置文件中的参数，并赋值到estimator对象中
    readParameters(n);      
    estimator.setParameter();  
#ifdef EIGEN_DONT_PARALLELIZE
    ROS_DEBUG("EIGEN_DONT_PARALLELIZE");
#endif
    ROS_WARN("waiting for image and imu...");

    // Step 2 注册一些要发布的消息
    registerPub(n);

    // Step 3 订阅消息
    // 接受imu消息
    ros::Subscriber sub_imu = n.subscribe(IMU_TOPIC, 2000, 
        imu_callback, ros::TransportHints().tcpNoDelay());
    // 接受前端视觉光流结果, 这里的feature只包含追踪到的特征点，不包含图像
    ros::Subscriber sub_image = n.subscribe("/feature_tracker/feature", 2000, 
        feature_callback);
    // 接受前端重启命令
    ros::Subscriber sub_restart = n.subscribe("/feature_tracker/restart", 2000, 
        restart_callback);
    // 回环检测的fast relocalization响应
    ros::Subscriber sub_relo_points = n.subscribe("/pose_graph/match_points", 2000, 
        relocalization_callback);

    // Step 4 核心处理线程
    std::thread measurement_process{process};
    ros::spin();

    return 0;
}
