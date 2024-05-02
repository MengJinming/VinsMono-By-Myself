#include "estimator.h"

Estimator::Estimator(): f_manager{Rs}
{
    ROS_INFO("init begins");
    clearState();
}

/**
 * @brief 外参，重投影置信度，延时设置
 * 
 */
void Estimator::setParameter()
{
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        tic[i] = TIC[i];    //; tic ric是类成员变量
        ric[i] = RIC[i];
    }
    f_manager.setRic(ric);  //; FeatureManager这个类中也有相机和IMU外参的类成员变量
    // 这里可以看到虚拟相机的用法
    //; 下面这些变量和残差的信息矩阵有关，所以factor这个文件夹下的文件暂时可以理解成系数
    //! 问题：怎么理解这个信息矩阵的设置？
    //! 不确定的回答：认为视觉提取特征点的不确定度是1.5个像素
    ProjectionFactor::sqrt_info = FOCAL_LENGTH / 1.5 * Matrix2d::Identity();
    ProjectionTdFactor::sqrt_info = FOCAL_LENGTH / 1.5 * Matrix2d::Identity();
    td = TD;
}

// 所有状态全部重置
void Estimator::clearState()
{
    for (int i = 0; i < WINDOW_SIZE + 1; i++)
    {
        Rs[i].setIdentity();
        Ps[i].setZero();
        Vs[i].setZero();
        Bas[i].setZero();
        Bgs[i].setZero();
        dt_buf[i].clear();
        linear_acceleration_buf[i].clear();
        angular_velocity_buf[i].clear();

        if (pre_integrations[i] != nullptr)
            delete pre_integrations[i];
        pre_integrations[i] = nullptr;
    }

    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        tic[i] = Vector3d::Zero();
        ric[i] = Matrix3d::Identity();
    }

    for (auto &it : all_image_frame)
    {
        if (it.second.pre_integration != nullptr)
        {
            delete it.second.pre_integration;
            it.second.pre_integration = nullptr;
        }
    }

    solver_flag = INITIAL;
    first_imu = false,
    sum_of_back = 0;
    sum_of_front = 0;
    frame_count = 0;
    solver_flag = INITIAL;
    initial_timestamp = 0;
    all_image_frame.clear();
    td = TD;


    if (tmp_pre_integration != nullptr)
        delete tmp_pre_integration;
    if (last_marginalization_info != nullptr)
        delete last_marginalization_info;

    tmp_pre_integration = nullptr;
    last_marginalization_info = nullptr;
    last_marginalization_parameter_blocks.clear();

    f_manager.clearState();

    failure_occur = 0;
    relocalization_info = 0;

    drift_correct_r = Matrix3d::Identity();
    drift_correct_t = Vector3d::Zero();
}

/**
 * @brief 对imu数据进行处理，包括更新预积分量，和提供优化状态量的初始值
 * 
 * @param[in] dt 
 * @param[in] linear_acceleration 
 * @param[in] angular_velocity 
 */
void Estimator::processIMU(double dt, const Vector3d &linear_acceleration, const Vector3d &angular_velocity)
{
    if (!first_imu)    //; 成员变量，是否是第一帧imu数据
    {
        first_imu = true;
        acc_0 = linear_acceleration;   //; 上一帧imu数据值，这里第一次imu数据就把上一帧imu数据赋值成当前帧
        gyr_0 = angular_velocity;
    }
    // 滑窗中保留11帧，frame_count表示现在处理第几帧，一般处理到第11帧时就保持不变了
    // 由于预积分是帧间约束，因此第1个预积分量实际上是用不到的
    //; frame_count是类成员变量，指示滑窗中存在多少帧图像
    if (!pre_integrations[frame_count])
    {
        //; 存储预积分类指针的数组
        //; 传入的四个参数是上一帧的加速度、角速度、加速度零偏、角速度零偏
        pre_integrations[frame_count] = new IntegrationBase{acc_0, gyr_0, Bas[frame_count], Bgs[frame_count]};
    }
    // 所以只有大于0才处理
    if (frame_count != 0)
    {
        //; 1.存储当前帧传入的加速度、角速度数据 2.进行预积分传播
        //; 但是注意这个函数中最后把acc_0和gyro_0更新成了当前帧的加速度和角速度值，这是bug吗？
        //; 解答：不是！因为这个函数中更新的是预积分类IntegrationBase中的acc_0和gyro_0，而estimator类中也有一个acc_0和gyro_0，是在本程序后面更新的
        pre_integrations[frame_count]->push_back(dt, linear_acceleration, angular_velocity);
        //if(solver_flag != NON_LINEAR)
            // 这个量用来做初始化用的
            //; tmp_pre_integration是一个预积分类指针，并不是一个vector，但是这里调用的也不是vector的push_back，而是预积分类中自定义的push_back函数
            //; 注意：pre_integrations代表了滑窗中的关键帧，而tmp_pre_integration代表每一帧，包括关键帧和非关键帧
            tmp_pre_integration->push_back(dt, linear_acceleration, angular_velocity);
        // 保存传感器数据
        //; 注意这里三个变量是estimator类成员变量，在预积分类中同样有这些成员变量
        dt_buf[frame_count].push_back(dt);
        linear_acceleration_buf[frame_count].push_back(linear_acceleration);
        angular_velocity_buf[frame_count].push_back(angular_velocity);

        // 又是一个中值积分，更新滑窗中状态量，本质是给非线性优化提供可信的初始值
        //; 注意这里计算的是公式中的PVQ, 而不是预积分。这里积分的作用上面也说了，就是给后端优化提供较好的初始值
        int j = frame_count;         
        //; 问题：去看Rs[j]是什么时候被赋初值的？
        Vector3d un_acc_0 = Rs[j] * (acc_0 - Bas[j]) - g;  //; 这里Rs[j]就是公式中的Rw_bk
        Vector3d un_gyr = 0.5 * (gyr_0 + angular_velocity) - Bgs[j];
        Rs[j] *= Utility::deltaQ(un_gyr * dt).toRotationMatrix();   //; 这里更新了Rs, 也就是公式中的Rw_bk+1
        Vector3d un_acc_1 = Rs[j] * (linear_acceleration - Bas[j]) - g;
        Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
        Ps[j] += dt * Vs[j] + 0.5 * dt * dt * un_acc;   //; 注意这里用的就是Vs[j]，而不是平均值，这是位移公式决定的
        Vs[j] += dt * un_acc;
    }
    acc_0 = linear_acceleration;
    gyr_0 = angular_velocity;
}

void Estimator::processImage(const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &image, const std_msgs::Header &header)
{
    ROS_DEBUG("new image coming ------------------------------------------");
    ROS_DEBUG("Adding feature points %lu", image.size());
    // Step 1 将特征点信息加到f_manager这个特征点管理器中，同时进行是否关键帧的检查
    if (f_manager.addFeatureCheckParallax(frame_count, image, td))
        // 如果上一帧是关键帧，则滑窗中最老的帧就要被移出滑窗
        //; marginalization_flag是类成员变量
        marginalization_flag = MARGIN_OLD;    //; 0
    else
        // 否则移除上一帧
        marginalization_flag = MARGIN_SECOND_NEW;  //; 1

    ROS_DEBUG("this frame is--------------------%s", marginalization_flag ? "reject" : "accept");
    ROS_DEBUG("%s", marginalization_flag ? "Non-keyframe" : "Keyframe");
    ROS_DEBUG("Solving %d", frame_count);
    ROS_DEBUG("number of feature: %d", f_manager.getFeatureCount());
    Headers[frame_count] = header;

    //! 下面的操作没有太看懂
    // all_image_frame用来做初始化相关操作，他保留滑窗起始到当前的所有帧
    // 有一些帧会因为不是KF，被MARGIN_SECOND_NEW，但是即使较新的帧被margin，他也会保留在这个容器中，
    //    因为初始化要求使用所有的帧，而非只要KF
    ImageFrame imageframe(image, header.stamp.toSec());
    imageframe.pre_integration = tmp_pre_integration;   //; tmp_pre_integration是类成员变量
    // 这里就是简单的把图像和预积分绑定在一起，这里预积分就是两帧之间的，滑窗中实际上是两个KF之间的
    // 实际上是准备用来初始化的相关数据
    all_image_frame.insert(make_pair(header.stamp.toSec(), imageframe));
    //; 这里为什么又重新计算了预积分？
    tmp_pre_integration = new IntegrationBase{acc_0, gyr_0, Bas[frame_count], Bgs[frame_count]};

    // 没有外参初值
    // Step 2： 外参初始化，也就是进行外参标定
    if(ESTIMATE_EXTRINSIC == 2)
    {
        ROS_INFO("calibrating extrinsic param, rotation movement is needed");
        if (frame_count != 0)
        {
            // 这里标定imu和相机的旋转外参的初值
            // 因为预积分是相邻帧的约束，因为这里得到的图像关联也是相邻的
            //; 下面就是利用这一帧和上一帧之间的特征点匹配进行外参标定
            vector<pair<Vector3d, Vector3d>> corres = f_manager.getCorresponding(frame_count - 1, frame_count);
            Matrix3d calib_ric;
            if (initial_ex_rotation.CalibrationExRotation(corres, pre_integrations[frame_count]->delta_q, calib_ric))
            {
                ROS_WARN("initial extrinsic rotation calib success");
                ROS_WARN_STREAM("initial extrinsic rotation: " << endl << calib_ric);
                ric[0] = calib_ric;     //; 这个ric是estimator的类成员变量，存储旋转外参
                RIC[0] = calib_ric;     //; 这个RIC是parameter中的全局变量，存储旋转外参
                // 标志位设置成可信的外参初值
                ESTIMATE_EXTRINSIC = 1; //; 如果外参标定成功，会修改外参标志位，可见实际上只会标定一次外参（如果成功的话）
            }
        }
    }

    //; 下面根据solver_flag来判断是进行初始化还是进行非线性优化
    if (solver_flag == INITIAL)
    {
        if (frame_count == WINDOW_SIZE) // 有足够的帧数
        {
            bool result = false;
            // 要有可信的外参值，同时距离上次初始化不成功至少相邻0.1s
            // Step 3： VIO初始化，视觉-惯导联合初始化
            if( ESTIMATE_EXTRINSIC != 2 && (header.stamp.toSec() - initial_timestamp) > 0.1)
            {
               result = initialStructure();     //; 视觉-惯导联合初始化
               initial_timestamp = header.stamp.toSec();    //; initial_timestamp类成员变量，是上一次初始化的时间戳
            }
            //; 如果result=true，那么说明Step3中的VIO初始化成功，那么直接修改solver_flag = NON_LINEAR;
            //; 这样下次最外面的if都不会进入了
            if(result)
            {
                solver_flag = NON_LINEAR;
                // Step 4： 后端非线性优化，边缘化
                solveOdometry();

                // Step 5： 滑动窗口，移除边缘化的帧
                slideWindow();

                // Step 6： 移除无效地图点，就是被三角化失败的点
                f_manager.removeFailures(); // 移除无效地图点
                ROS_INFO("Initialization finish!");
                last_R = Rs[WINDOW_SIZE];   // 滑窗里最新的位姿
                last_P = Ps[WINDOW_SIZE];
                last_R0 = Rs[0];    // 滑窗里最老的位姿
                last_P0 = Ps[0];
                
            }

            // 如果上面根据最开始的10帧图像初始化失败了，那么就直接滑窗，根据关键帧的情况把图像滑出去
            else
                slideWindow();
        }
        
        //; 这里可以看到，前端发来的前10帧，不管是不是关键帧都会增加图像关键帧的帧数，知道增加到填满滑窗，然后就会执行上面的if语句
        else
            frame_count++;
    }

    //; 进行初始化之后，后面都会进入else这个分支
    else
    {
        TicToc t_solve;
        // Step 1 后端非线性优化，边缘化
        solveOdometry();
        ROS_DEBUG("solver costs: %fms", t_solve.toc());
        
        // Step 1.5 检测VIO是否正常
        if (failureDetection())
        {
            ROS_WARN("failure detection!");
            failure_occur = 1;
            // 如果异常，重启VIO
            clearState();
            setParameter();
            ROS_WARN("system reboot!");
            return;
        }

        TicToc t_margin;
        // Step 2 滑窗，移除最老帧或者倒数第二帧
        slideWindow();

        // Step 3 移除无效地图点，就是被三角化失败的点
        f_manager.removeFailures();
        ROS_DEBUG("marginalization costs: %fms", t_margin.toc());

        // prepare output of VINS
        // 给可视化用的
        key_poses.clear();
        for (int i = 0; i <= WINDOW_SIZE; i++)
            key_poses.push_back(Ps[i]);

        last_R = Rs[WINDOW_SIZE];
        last_P = Ps[WINDOW_SIZE];
        last_R0 = Rs[0];
        last_P0 = Ps[0];
    }
}

/**
 * @brief VIO初始化，将滑窗中的P V Q恢复到第0帧并且和重力对齐
 * 
 * @return true 
 * @return false 
 */
bool Estimator::initialStructure()
{
    TicToc t_sfm;
    // Step 1 check imu observibility
    //; 计算all_image_frame中所有图像的IMU加速度的标准差。但是实际上最后return false注释掉了，也就是这个判断并没有使用
    {
        map<double, ImageFrame>::iterator frame_it;
        Vector3d sum_g;
        // 从第二帧开始检查imu
        for (frame_it = all_image_frame.begin(), frame_it++; frame_it != all_image_frame.end(); frame_it++)
        {
            double dt = frame_it->second.pre_integration->sum_dt;
            Vector3d tmp_g = frame_it->second.pre_integration->delta_v / dt;
            // 累加每一帧带重力加速度的deltav
            sum_g += tmp_g;
        }
        Vector3d aver_g;
        aver_g = sum_g * 1.0 / ((int)all_image_frame.size() - 1);
        double var = 0;
        for (frame_it = all_image_frame.begin(), frame_it++; frame_it != all_image_frame.end(); frame_it++)
        {
            double dt = frame_it->second.pre_integration->sum_dt;
            Vector3d tmp_g = frame_it->second.pre_integration->delta_v / dt;
            // 求方差
            var += (tmp_g - aver_g).transpose() * (tmp_g - aver_g);
            //cout << "frame g " << tmp_g.transpose() << endl;
        }
        // 得到的标准差
        var = sqrt(var / ((int)all_image_frame.size() - 1));
        //ROS_WARN("IMU variation %f!", var);
        // 实际上检查结果并没有用
        if(var < 0.25)
        {
            ROS_INFO("IMU excitation not enouth!");
            //return false;
        }
    }
    // Step 2 global sfm
    //; 做一个纯视觉slam，所以这里的位置和姿态都是相机的，而不是IMU的，并且是相机相对世界坐标系的位姿，即Rwc
    Quaterniond Q[frame_count + 1];
    Vector3d T[frame_count + 1];
    map<int, Vector3d> sfm_tracked_points;  //; sfm三角化出来的特征点
    vector<SFMFeature> sfm_f;   // 保存每个特征点的信息
    //; 遍历所有的特征点，记录它的id和它在各帧图像中的观测信息
    for (auto &it_per_id : f_manager.feature)
    {
        //; 这里第一次观测到特征点的图像帧索引-1，是因为后面存储的时候先+1了
        int imu_j = it_per_id.start_frame - 1;  // 这个跟imu无关，就是存储观测特征点的帧的索引
        SFMFeature tmp_feature; // 用来后续做sfm
        tmp_feature.state = false;
        tmp_feature.id = it_per_id.feature_id;
        //; 遍历能观测到这个特征点的所有帧
        for (auto &it_per_frame : it_per_id.feature_per_frame)
        {
            imu_j++;
            Vector3d pts_j = it_per_frame.point;
            //; 观测到这个特征点的图像帧的索引 以及 在各帧相机归一化坐标系下的坐标
            tmp_feature.observation.push_back(make_pair(imu_j, Eigen::Vector2d{pts_j.x(), pts_j.y()}));
        }
        sfm_f.push_back(tmp_feature);
    } 
    // Step 2.2 在滑窗中寻找枢纽帧
    Matrix3d relative_R;
    Vector3d relative_T;
    int l;
    //; relativePose就是在寻找枢纽帧，找到则可以继续进行三角化，返回true。否则无法继续进行三角化，返回false
    if (!relativePose(relative_R, relative_T, l))
    {
        ROS_INFO("Not enough features or parallax; Move device around");
        return false;
    }
    // Step 2.3 求解sfm
    GlobalSFM sfm;
    // 进行sfm的求解
    //; 内部进行global SfM的求解，并且最后做一个全局BA提高精度
    if(!sfm.construct(frame_count + 1, Q, T, l,
              relative_R, relative_T,
              sfm_f, sfm_tracked_points))
    {
        ROS_DEBUG("global SFM failed!");
        marginalization_flag = MARGIN_OLD;
        return false;
    }

    // Step 3 solve pnp for all frame
    // step2只是针对KF进行sfm，初始化需要all_image_frame中的所有元素，因此下面通过KF来求解其他的非KF的位姿
    map<double, ImageFrame>::iterator frame_it;
    map<int, Vector3d>::iterator it;
    frame_it = all_image_frame.begin( );
    // i代表跟这个帧最近的KF的索引
    //; 遍历滑窗中的所有帧，包括关键帧和非关键帧
    //; map<double, ImageFrame> all_image_frame;  第一个double是时间戳
    for (int i = 0; frame_it != all_image_frame.end( ); frame_it++)
    {
        // provide initial guess
        cv::Mat r, rvec, t, D, tmp_r;
        // 这一帧本身就是KF，因此可以直接得到位姿
        if((frame_it->first) == Headers[i].stamp.toSec())
        {
            frame_it->second.is_key_frame = true;
            //; 这里把相机的位姿转换成了这一帧图像对应的IMU的位姿
            frame_it->second.R = Q[i].toRotationMatrix() * RIC[0].transpose();  // 得到Rwi
            frame_it->second.T = T[i];  // 初始化不估计平移外参
            i++;
            continue;
        }
        //; 这个普通帧在当前关键帧之后，那么就利用下一个关键帧的位姿作为这个普通帧的位姿初始估计
        //! 问题：这样并不一定是最接近这个普通帧的啊？
        if((frame_it->first) > Headers[i].stamp.toSec())
        {
            i++;
        }
        // 最近的KF提供一个初始值，Twc -> Tcw, 之所以这么转换是因为视觉slam中存储的都是Tcw
        Matrix3d R_inital = (Q[i].inverse()).toRotationMatrix();
        Vector3d P_inital = - R_inital * T[i];
        // eigen -> cv
        cv::eigen2cv(R_inital, tmp_r);
        cv::Rodrigues(tmp_r, rvec);
        cv::eigen2cv(P_inital, t);

        //; frame_it->seconds是ImageFrame
        frame_it->second.is_key_frame = false;  
        vector<cv::Point3f> pts_3_vector;
        vector<cv::Point2f> pts_2_vector;
        // 遍历这一帧对应的特征点
        //; frame_it->second.points 是 map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>> > > points;
        //; 这里就是遍历map中的每一个键值对
        for (auto &id_pts : frame_it->second.points)
        {
            int feature_id = id_pts.first;     //; 特征点id
            // 由于是单目，这里id_pts.second大小就是1，也就是每个vector中只有一个变量
            for (auto &i_p : id_pts.second)
            {
                it = sfm_tracked_points.find(feature_id);
                if(it != sfm_tracked_points.end())  // 有对应的三角化出来的3d点
                {
                    Vector3d world_pts = it->second;    // 地图点的世界坐标
                    cv::Point3f pts_3(world_pts(0), world_pts(1), world_pts(2));
                    pts_3_vector.push_back(pts_3);
                    Vector2d img_pts = i_p.second.head<2>();    //; 前2维，就是归一化平面上的（x,y）
                    cv::Point2f pts_2(img_pts(0), img_pts(1));
                    pts_2_vector.push_back(pts_2);
                }
            }
        }
        //; 2d观测是归一化相机平面上的，所以这里内参仍然是I
        cv::Mat K = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);     
        if(pts_3_vector.size() < 6)
        {
            cout << "pts_3_vector size " << pts_3_vector.size() << endl;
            ROS_DEBUG("Not enough points for solve pnp !");
            return false;
        }
        // 依然是调用opencv求解pnp接口
        if (! cv::solvePnP(pts_3_vector, pts_2_vector, K, D, rvec, t, 1))
        {
            ROS_DEBUG("solve pnp fail!");
            return false;
        }
        // cv -> eigen,同时Tcw -> Twc
        cv::Rodrigues(rvec, r);
        MatrixXd R_pnp,tmp_R_pnp;
        cv::cv2eigen(r, tmp_R_pnp);
        R_pnp = tmp_R_pnp.transpose();
        MatrixXd T_pnp;
        cv::cv2eigen(t, T_pnp);
        T_pnp = R_pnp * (-T_pnp);
        // Twc -> Twi
        // 由于尺度未恢复，因此平移暂时不转到imu系
        frame_it->second.R = R_pnp * RIC[0].transpose();
        frame_it->second.T = T_pnp;
    }

    // 到此就求解出用来做视觉惯性对齐的所有视觉帧的位姿
    // Step 4 视觉惯性对齐
    if (visualInitialAlign())
        return true;
    else
    {
        ROS_INFO("misalign visual structure with IMU");
        return false;
    }
}

//; 视觉惯性对齐
bool Estimator::visualInitialAlign()
{
    TicToc t_g;
    VectorXd x;
    //solve scale
    // Step 1 状态变量的初始估计
    //; 下面的函数进行了两大步：
    //;   1.估计陀螺仪零偏，得到零偏后对滑窗中的所有帧all_image_frame重新计算预积分
    //;   2.估计所有帧的速度，重力在枢纽帧的表示g_c0，尺度s
    bool result = VisualIMUAlignment(all_image_frame, Bgs, g, x);
    if(!result)
    {
        ROS_DEBUG("solve g failed!");
        return false;
    }

    // Step 2 把所有帧对齐到世界坐标系
    // change state
    // 首先把对齐后KF的位姿附给滑窗中的值，Rwi twc
    for (int i = 0; i <= frame_count; i++)
    {
        Matrix3d Ri = all_image_frame[Headers[i].stamp.toSec()].R;
        Vector3d Pi = all_image_frame[Headers[i].stamp.toSec()].T;
        Ps[i] = Pi;   //; 但是平移没有转到IMU系下，所以仍然是相机系之间的平移
        Rs[i] = Ri;   //; 关键帧的imu系到枢纽帧的旋转
        all_image_frame[Headers[i].stamp.toSec()].is_key_frame = true;
    }

    VectorXd dep = f_manager.getDepthVector();  // 根据有效特征点数初始化这个动态向量
    for (int i = 0; i < dep.size(); i++)
        dep[i] = -1;    // 深度预设都是-1，不过其实getDepthVector()函数返回的预设值也是-1
    f_manager.clearDepth(dep);  // 特征管理器把所有的特征点逆深度也设置为-1

    //triangulat on cam pose , no tic
    Vector3d TIC_TMP[NUM_OF_CAM];
    for(int i = 0; i < NUM_OF_CAM; i++)
        TIC_TMP[i].setZero();
    ric[0] = RIC[0];    //; 进行外参估计后RIC就更新了
    f_manager.setRic(ric);  //; FeatureManager这个类中也有相机和IMU的外参
    // 多约束三角化所有的特征点，注意，仍带是尺度模糊的
    //! 问题：为什么要再次三角化所有的特征点呢？利用多约束再次三角化得到的结果更加准确？
    f_manager.triangulate(Ps, &(TIC_TMP[0]), &(RIC[0]));

    //; 取出前面估计得到的尺度
    double s = (x.tail<1>())(0);
    // 将滑窗中的预积分重新计算
    for (int i = 0; i <= WINDOW_SIZE; i++)
    {
        //; 注意这里是对滑窗中的关键帧重新计算预积分，不是滑窗中的所有帧
        pre_integrations[i]->repropagate(Vector3d::Zero(), Bgs[i]);
    }
    // 下面开始把所有的状态对齐到第0帧的imu坐标系
    for (int i = frame_count; i >= 0; i--)
        // twi - tw0 = t0i,就是把所有的平移对齐到滑窗中的第0帧
        //; 注意：t0i指的是i系到0系的平移，在0系下的表示
        //; 1.所有关键帧的imu平移对齐到滑窗中第0帧的imu系下
        //;   w是枢纽帧相机系，i是某个关键帧的imu系，0是第0帧的imu系
        //;                    ------
        //;                  / imu_i /
        //;                   ------
        //;                  ^      ^
        //;                 /        \
        //; t_0i=t_wi-t_w0 /          \ t_wi
        //;               /            \
        //;              /              \
        //;        ------       t_w0     -------
        //;      / imu_0 /   <--------  | cam_w |
        //;       ------                 -------
        //; 2.注意下面的w是指枢纽帧，c和i分别指某一帧（可以是第0帧）的相机系和imu系
        //;   T_w_i = T_w_c * T_ic^-1
        //;  （1）展开计算公式：
        //;     [R_w_i   P_w_i] = [R_w_c   P_w_c] * [R_c_i  -R_c_i*P_i_c]
        //;     [  0       1  ] = [  0       1  ] * [  0         1     ]
        //;  （2）只看平移部分：
        //;      P_w_i = -R_w_c * R_c_i * P_ic + P_w_c = P_w_c - R_w_i * P_i_c
        //;      注意其中的P_w_c是相机的平移，是带有尺度的。而P_i_c是平移外参，是没有尺度的
        //;      所以最后不带尺度（对齐到米单位的）公式：P_w_i = s*P_w_c - R_w_i * P_i_c
        //! 注意：这里得到的t_0i，即从第0帧imu系指向第i帧imu系的向量，仍然是在 枢纽帧相机坐标系 下的表示！
        Ps[i] = s * Ps[i] - Rs[i] * TIC[0] - (s * Ps[0] - Rs[0] * TIC[0]);
    int kv = -1;
    map<double, ImageFrame>::iterator frame_i;
    // 把求解出来KF的速度赋给滑窗中
    //; 最后只需要对滑窗中的关键帧进行对齐。之前估计状态变量的时候对滑窗中的所有帧都进行处理是因为处理所有帧的话
    //; 时间差小更精确，而且这样约束更多，进行状态变量的估计结果更可靠
    for (frame_i = all_image_frame.begin(); frame_i != all_image_frame.end(); frame_i++)
    {
        if(frame_i->second.is_key_frame)
        {
            kv++;
            // 当时求得速度是imu系（相对自身坐标系），现在转到world系，注意这里的word系还是指枢纽帧相机坐标系
            Vs[kv] = frame_i->second.R * x.segment<3>(kv * 3);
        }
    }
    // 把尺度模糊的3d点恢复到真实尺度下
    for (auto &it_per_id : f_manager.feature)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;
        //; 这里处理很简单，就是×s，没有坐标系的变换，因为深度一直都是相对于第一次观测到这个3d点的相机坐标系的
        it_per_id.estimated_depth *= s;
    }

    // 所有的P V Q全部对齐到第0帧的，同时和对齐到重力方向
    //; 1.R0是枢纽帧c0到真正的世界坐标系w之间的旋转，即R_w_c0，但是注意补偿掉了yaw角的旋转
    //;   这里补偿掉yaw的旋转可以认为是把w系的Z轴先对齐到c0系上
    Matrix3d R0 = Utility::g2R(g);  // g是枢纽帧下的重力方向，得到R_w_j
    //; 2.Rs[0]是第0帧的imu系到枢纽帧c0之间的旋转，则R0*Rs[0]是 第0帧的imu系 到 真正的世界坐标系w 之间的旋转
    //;   R2ypr返回一个Eigen的Vecotr3d，.x()应该是取第0维？也就是yaw角度
    double yaw = Utility::R2ypr(R0 * Rs[0]).x();    // Rs[0]实际上是R_j_0
    //; 3.又补偿掉了第0帧的yaw角，这样相当于又把w系的Z轴对齐到了第0帧的imu系上。
    //;   所以其实最后yaw角补偿只是把w系的Z轴对齐到第0帧的imu系上，先对齐到c0系上只是一个中间步骤。
    //;   这样最终得到的这个R0就是c0系相对真正的世界坐标系（Z轴对齐到第0帧imu系）的旋转R_w_c0
    R0 = Utility::ypr2R(Eigen::Vector3d{-yaw, 0, 0}) * R0;  // 第一帧yaw赋0
    g = R0 * g;  //; 此时的g就是在真正的世界坐标系（Z轴对齐到第0帧imu系）的表示
    //Matrix3d rot_diff = R0 * Rs[0].transpose();
    Matrix3d rot_diff = R0;
    for (int i = 0; i <= frame_count; i++)
    {
        //; 下面就把PVQ从枢纽帧c0下的表示全部转换到真正的世界坐标系（Z轴对齐到第0帧imu系）下表示
        Ps[i] = rot_diff * Ps[i];
        Rs[i] = rot_diff * Rs[i];   // 全部对齐到重力下，同时yaw角对齐到第一帧
        Vs[i] = rot_diff * Vs[i];
    }
    //! 问题：这里输出了g0，不知道结果是不是(0,0,9.8)? 实际跑一下看看！
    ROS_DEBUG_STREAM("g0     " << g.transpose());
    ROS_DEBUG_STREAM("my R0  " << Utility::R2ypr(Rs[0]).transpose()); 
    return true;
}

/**
 * @brief 寻找滑窗内一个帧作为枢纽帧，要求和最后一帧既有足够的共视也要有足够的视差
 *        这样其他帧都对齐到这个枢纽帧上
 *        得到T_l_last
 * @param[in] relative_R 
 * @param[in] relative_T 
 * @param[in] l 
 * @return true 
 * @return false 
 */
//; 在滑窗中寻找枢纽帧，要求枢纽帧和滑窗中的最后一帧有足够的共视关系（求解相对位姿准），
//; 并且要有足够的视差（三角化地图点准）。找到枢纽帧的同时，计算得到最后一帧相对枢纽帧的旋转和平移（带尺度）
bool Estimator::relativePose(Matrix3d &relative_R, Vector3d &relative_T, int &l)
{
    // find previous frame which contians enough correspondance and parallex with newest frame
    // 优先从最前面开始
    for (int i = 0; i < WINDOW_SIZE; i++)
    {
        vector<pair<Vector3d, Vector3d>> corres;
        corres = f_manager.getCorresponding(i, WINDOW_SIZE);
        // 要求共视的特征点足够多
        if (corres.size() > 20)
        {
            double sum_parallax = 0;
            double average_parallax;
            for (int j = 0; j < int(corres.size()); j++)
            {
                Vector2d pts_0(corres[j].first(0), corres[j].first(1));
                Vector2d pts_1(corres[j].second(0), corres[j].second(1));
                double parallax = (pts_0 - pts_1).norm();   // 计算了视差
                sum_parallax = sum_parallax + parallax;

            }
            // 计算每个特征点的平均视差
            average_parallax = 1.0 * sum_parallax / int(corres.size());
            // 有足够的视差在通过本质矩阵恢复第i帧和最后一帧之间的 R t T_i_last
            //; 这里又用到了虚拟焦距，上面计算的视差是在归一化相机平面上的，这里乘以焦距转化到了一个虚拟相机平面上
            if(average_parallax * 460 > 30 && m_estimator.solveRelativeRT(corres, relative_R, relative_T))
            {
                l = i;  //; 找到滑窗中的枢纽帧索引
                ROS_DEBUG("average_parallax %f choose l %d and newest frame to triangulate the whole structure", average_parallax * 460, l);
                return true;
            }
        }
    }
    return false;
}

void Estimator::solveOdometry()
{
    // 保证滑窗中帧数满了
    //; 这里的判断其实也是多余的，因为前面滑窗不填满的话就无法进行初始化，初始化不完成就不会调用这个函数
    if (frame_count < WINDOW_SIZE)
        return;
    // 其次要求初始化完成
    //; 其实根据调用solveOdometry()这个函数的逻辑来看，下面这个if判断是一定满足的 
    if (solver_flag == NON_LINEAR)
    {
        TicToc t_tri;
        // 先把应该三角化但是没有三角化的特征点三角化
        //! 问题：在初始化部分不是调用过这个函数进行初始化了吗？为什么这里又调用一次？
        f_manager.triangulate(Ps, tic, ric);
        ROS_DEBUG("triangulation costs %f", t_tri.toc());
        optimization();
    }
}

/**
 * @brief 由于ceres的参数块都是double数组，因此这里把参数块从eigen的表示转成double数组
 * 
 */
void Estimator::vector2double()
{
    // KF的位姿
    for (int i = 0; i <= WINDOW_SIZE; i++)
    {
        para_Pose[i][0] = Ps[i].x();
        para_Pose[i][1] = Ps[i].y();
        para_Pose[i][2] = Ps[i].z();
        Quaterniond q{Rs[i]};
        para_Pose[i][3] = q.x();
        para_Pose[i][4] = q.y();
        para_Pose[i][5] = q.z();
        para_Pose[i][6] = q.w();

        para_SpeedBias[i][0] = Vs[i].x();
        para_SpeedBias[i][1] = Vs[i].y();
        para_SpeedBias[i][2] = Vs[i].z();

        para_SpeedBias[i][3] = Bas[i].x();
        para_SpeedBias[i][4] = Bas[i].y();
        para_SpeedBias[i][5] = Bas[i].z();

        para_SpeedBias[i][6] = Bgs[i].x();
        para_SpeedBias[i][7] = Bgs[i].y();
        para_SpeedBias[i][8] = Bgs[i].z();
    }
    // 外参
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        para_Ex_Pose[i][0] = tic[i].x();
        para_Ex_Pose[i][1] = tic[i].y();
        para_Ex_Pose[i][2] = tic[i].z();
        Quaterniond q{ric[i]};
        para_Ex_Pose[i][3] = q.x();
        para_Ex_Pose[i][4] = q.y();
        para_Ex_Pose[i][5] = q.z();
        para_Ex_Pose[i][6] = q.w();
    }
    // 特征点逆深度
    VectorXd dep = f_manager.getDepthVector();
    for (int i = 0; i < f_manager.getFeatureCount(); i++)
        para_Feature[i][0] = dep(i);  //; 这里数组设置了上限1000，如果getFeatureCount>1000，岂不是数组越界了？
    // 传感器时间同步
    if (ESTIMATE_TD)
        para_Td[0][0] = td;
}

/**
 * @brief double -> eigen 同时fix第一帧的yaw和平移，固定了四自由度的零空间
 * 
 */
void Estimator::double2vector()
{
    // 取出优化前的第一帧的位姿
    Vector3d origin_R0 = Utility::R2ypr(Rs[0]);
    Vector3d origin_P0 = Ps[0];

    if (failure_occur)
    {
        origin_R0 = Utility::R2ypr(last_R0);
        origin_P0 = last_P0;
        failure_occur = 0;
    }
    // 优化后的第一帧的位姿
    Vector3d origin_R00 = Utility::R2ypr(Quaterniond(para_Pose[0][6],
                                                      para_Pose[0][3],
                                                      para_Pose[0][4],
                                                      para_Pose[0][5]).toRotationMatrix());
    // yaw角差，即优化前第1帧的yaw - 优化后第1帧的yaw
    double y_diff = origin_R0.x() - origin_R00.x();

    //; 这里就是得到由于yaw角度的变化构成的旋转矩阵，用于把优化后的第0帧姿态的yaw角重新对齐到0
    Matrix3d rot_diff = Utility::ypr2R(Vector3d(y_diff, 0, 0));

    // 接近万象节死锁的问题 https://blog.csdn.net/AndrewFan/article/details/60981437
    if (abs(abs(origin_R0.y()) - 90) < 1.0 || abs(abs(origin_R00.y()) - 90) < 1.0)
    {
        ROS_DEBUG("euler singular point!");
        rot_diff = Rs[0] * Quaterniond(para_Pose[0][6],
                                       para_Pose[0][3],
                                       para_Pose[0][4],
                                       para_Pose[0][5]).toRotationMatrix().transpose();
    }

    for (int i = 0; i <= WINDOW_SIZE; i++)
    {
        // 保持第1帧的yaw不变
        Rs[i] = rot_diff * Quaterniond(para_Pose[i][6], para_Pose[i][3], para_Pose[i][4], para_Pose[i][5]).normalized().toRotationMatrix();
        
        // 保持第1帧的位移不变，里面的减法是其他帧相对第1帧的相对位移
        Ps[i] = rot_diff * Vector3d(para_Pose[i][0] - para_Pose[0][0],
                                para_Pose[i][1] - para_Pose[0][1],
                                para_Pose[i][2] - para_Pose[0][2]) + origin_P0;

        Vs[i] = rot_diff * Vector3d(para_SpeedBias[i][0],
                                    para_SpeedBias[i][1],
                                    para_SpeedBias[i][2]);

        //; bias自然是不受第一帧的位姿的影响的
        Bas[i] = Vector3d(para_SpeedBias[i][3],
                          para_SpeedBias[i][4],
                          para_SpeedBias[i][5]);

        Bgs[i] = Vector3d(para_SpeedBias[i][6],
                          para_SpeedBias[i][7],
                          para_SpeedBias[i][8]);
    }

    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        tic[i] = Vector3d(para_Ex_Pose[i][0],
                          para_Ex_Pose[i][1],
                          para_Ex_Pose[i][2]);
        ric[i] = Quaterniond(para_Ex_Pose[i][6],
                             para_Ex_Pose[i][3],
                             para_Ex_Pose[i][4],
                             para_Ex_Pose[i][5]).toRotationMatrix();
    }
    // 重新设置各个特征点的逆深度
    VectorXd dep = f_manager.getDepthVector();
    for (int i = 0; i < f_manager.getFeatureCount(); i++)
        dep(i) = para_Feature[i][0];
    f_manager.setDepth(dep);
    if (ESTIMATE_TD)
        td = para_Td[0][0];

    // relative info between two loop frame
    if(relocalization_info) // 类似进行一个调整
    { 
        Matrix3d relo_r;
        Vector3d relo_t;
        relo_r = rot_diff * Quaterniond(relo_Pose[6], relo_Pose[3], relo_Pose[4], relo_Pose[5]).normalized().toRotationMatrix();
        relo_t = rot_diff * Vector3d(relo_Pose[0] - para_Pose[0][0],
                                     relo_Pose[1] - para_Pose[0][1],
                                     relo_Pose[2] - para_Pose[0][2]) + origin_P0;
        double drift_correct_yaw;
        drift_correct_yaw = Utility::R2ypr(prev_relo_r).x() - Utility::R2ypr(relo_r).x();
        drift_correct_r = Utility::ypr2R(Vector3d(drift_correct_yaw, 0, 0));
        drift_correct_t = prev_relo_t - drift_correct_r * relo_t;   
        // T_loop_w * T_w_cur = T_loop_cur
        relo_relative_t = relo_r.transpose() * (Ps[relo_frame_local_index] - relo_t);
        relo_relative_q = relo_r.transpose() * Rs[relo_frame_local_index];
        relo_relative_yaw = Utility::normalizeAngle(Utility::R2ypr(Rs[relo_frame_local_index]).x() - Utility::R2ypr(relo_r).x());
        //cout << "vins relo " << endl;
        //cout << "vins relative_t " << relo_relative_t.transpose() << endl;
        //cout << "vins relative_yaw " <<relo_relative_yaw << endl;
        relocalization_info = 0;    

    }
}

bool Estimator::failureDetection()
{
    //; 追踪的地图点数量太少，视觉失效，整个VIO也就失效了
    if (f_manager.last_track_num < 2)   // 地图点数目是否足够
    {
        ROS_INFO(" little feature %d", f_manager.last_track_num);
        //return true;
    }
    //; 加速度零偏太大，2.5 m/s^2
    if (Bas[WINDOW_SIZE].norm() > 2.5)  // 零偏是否正常
    {
        ROS_INFO(" big IMU acc bias estimation %f", Bas[WINDOW_SIZE].norm());
        return true;
    }
    //; 陀螺仪零偏太大，1.0 rad/s = 57.3 °/s
    if (Bgs[WINDOW_SIZE].norm() > 1.0)
    {
        ROS_INFO(" big IMU gyr bias estimation %f", Bgs[WINDOW_SIZE].norm());
        return true;
    }
    /*
    if (tic(0) > 1)
    {
        ROS_INFO(" big extri param estimation %d", tic(0) > 1);
        return true;
    }
    */
    Vector3d tmp_P = Ps[WINDOW_SIZE];
    //; last_P是滑窗中上一帧的位姿，tmp_P是滑窗中最新帧的位姿，两个图像帧之间时间大约0.1s
    if ((tmp_P - last_P).norm() > 5)    // 两帧之间运动是否过大
    {
        ROS_INFO(" big translation");
        return true;
    }
    //; 沿着重力方向运动过大，比如自由落体或者克服重力做功，一般不常见，所以也不正常
    if (abs(tmp_P.z() - last_P.z()) > 1)    // 重力方向运动是否过大
    {
        ROS_INFO(" big z translation");
        return true; 
    }
    Matrix3d tmp_R = Rs[WINDOW_SIZE];
    Matrix3d delta_R = tmp_R.transpose() * last_R;
    Quaterniond delta_Q(delta_R);
    double delta_angle;
    //; 当前帧和上一帧的旋转角度过大
    delta_angle = acos(delta_Q.w()) * 2.0 / 3.14 * 180.0;
    if (delta_angle > 50)   // 两帧姿态变化是否过大
    {
        ROS_INFO(" big delta_angle ");
        //return true;
    }
    return false;
}

/**
 * @brief 进行非线性优化
 * 
 */
void Estimator::optimization()
{
    // 借助ceres进行非线性优化
    ceres::Problem problem;
    ceres::LossFunction *loss_function;  //; 注意这个是鲁邦核函数，只给视觉重投影误差使用
    //loss_function = new ceres::HuberLoss(1.0);
    loss_function = new ceres::CauchyLoss(1.0);
    // Step 1 定义待优化的参数块，类似g2o的顶点
    // 参数块 1： 滑窗中位姿包括位置和姿态，共11帧
    for (int i = 0; i < WINDOW_SIZE + 1; i++)
    {
        // 由于姿态不满足正常的加法，也就是李群上没有加法，因此需要自己定义他的加法
        //; 如果像初始化的SfM最后进行的全局BA那样，优化的是四元数，那么可以直接使用ceres官方定义的四元数局部参数块
        //; 来实现四元数独特的加法。但是这里维护的参数块是位置+四元数，一个7维的参数块，其中位置是普通加，四元数
        //; 是独特的加，所以需要自己定义局部参数块PoseLocalParameterization来实现这种加法
        ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
        //; 添加参数块，形参：这个参数的指针，参数的维度（数组维度），自定义加法的类指针
        problem.AddParameterBlock(para_Pose[i], SIZE_POSE, local_parameterization);
        problem.AddParameterBlock(para_SpeedBias[i], SIZE_SPEEDBIAS);
    }
    // 参数块 2： 相机imu间的外参
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
        problem.AddParameterBlock(para_Ex_Pose[i], SIZE_POSE, local_parameterization);
        //; 如果外参固定（给定的非常精确），那么就不需要优化
        if (!ESTIMATE_EXTRINSIC)
        {
            ROS_DEBUG("fix extinsic param");
            // 如果不需要优化外参就设置为fix
            problem.SetParameterBlockConstant(para_Ex_Pose[i]);
        }
        else
            ROS_DEBUG("estimate extinsic param");
    }
    // 传感器的时间同步
    if (ESTIMATE_TD)
    {
        problem.AddParameterBlock(para_Td[0], 1);
        //problem.SetParameterBlockConstant(para_Td[0]);
    }
    //! 注意：实际上还有地图点，其实平凡的参数块不需要调用AddParameterBlock，增加残差块接口时会自动绑定
    //!      也就是说对于地图点来说非常多，并且在增加残差块的时候一定会输入地图点，此时会自动绑定
    //; 扩展：实际上，只要是不需要定义独特加法的参数块，都可以不显性地调用AddParameterBlock添加参数块。
    //;      因为后面定义残差块的时候，会自动把要优化的参数块作为形参传入的残差块的函数中，这样就会实现参数块的自动绑定
    TicToc t_whole, t_prepare;

    //; 因为ceres优化的都是double数组，上面把这些数组添加进参数块了，但是数组中的数据和实际的位姿、bais等参数还没有
    //; 对应关系，所以这里要把Eigen维护的vector数据更新到ceres优化用的double数组中
    // eigen -> double
    vector2double();

    // Step 2 通过残差约束来添加残差块，类似g2o的边
    // Step 2.1 边缘化约束
    // 上一次的边缘化结果作为这一次的先验
    if (last_marginalization_info)
    {
        // construct new marginlization_factor
        MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
        problem.AddResidualBlock(marginalization_factor, NULL,
                                 last_marginalization_parameter_blocks);
    }
    // Step 2.2 imu预积分的约束
    for (int i = 0; i < WINDOW_SIZE; i++)
    {
        int j = i + 1;  //; j最后的值就是WINDOW_SIZE，也就是滑窗中的最后一帧
        // 滑窗中的两帧直接时间过长，这个IMU预积分约束就不可信了
        if (pre_integrations[j]->sum_dt > 10.0)
            continue;
        //; 注意这里用预积分类初始化IMUFactor类中的预积分成员变量，这样IMUFactor就有了预积分量
        IMUFactor* imu_factor = new IMUFactor(pre_integrations[j]);
        //; 参数：CostFunction, 核函数，要优化的参数块
        problem.AddResidualBlock(imu_factor, NULL, para_Pose[i], para_SpeedBias[i], para_Pose[j], para_SpeedBias[j]);
    }
    int f_m_cnt = 0;
    int feature_index = -1;

    // Step 2.3 视觉重投影的约束
    // 遍历每一个特征点
    for (auto &it_per_id : f_manager.feature)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        // 进行特征点有效性的检查
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;
 
        ++feature_index;
        // 第一个观测到这个特征点的帧idx
        int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;
        // 特征点在第一个帧下的归一化相机系坐标
        Vector3d pts_i = it_per_id.feature_per_frame[0].point;
        // 遍历看到这个特征点的所有KF
        for (auto &it_per_frame : it_per_id.feature_per_frame)
        {
            imu_j++;
            if (imu_i == imu_j) // 自己跟自己不能形成重投影
            {
                continue;
            }
            // 取出另一帧的归一化相机坐标
            Vector3d pts_j = it_per_frame.point;
            // 带有时间延时的是另一种形式
            if (ESTIMATE_TD)
            {
                    ProjectionTdFactor *f_td = new ProjectionTdFactor(pts_i, pts_j, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocity,
                                                                     it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td,
                                                                     it_per_id.feature_per_frame[0].uv.y(), it_per_frame.uv.y());
                    problem.AddResidualBlock(f_td, loss_function, para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Feature[feature_index], para_Td[0]);
                    /*
                    double **para = new double *[5];
                    para[0] = para_Pose[imu_i];
                    para[1] = para_Pose[imu_j];
                    para[2] = para_Ex_Pose[0];
                    para[3] = para_Feature[feature_index];
                    para[4] = para_Td[0];
                    f_td->check(para);
                    */
            }
            else
            {
                ProjectionFactor *f = new ProjectionFactor(pts_i, pts_j);   // 构造函数就是同一个特征点在不同帧的观测
                // 约束的变量是该特征点的第一个观测帧以及其他一个观测帧，加上外参和特征点逆深度
                problem.AddResidualBlock(f, loss_function, para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Feature[feature_index]);
            }
            f_m_cnt++;
        }
    }

    ROS_DEBUG("visual measurement count: %d", f_m_cnt);
    ROS_DEBUG("prepare for ceres: %f", t_prepare.toc());
    // 回环检测相关的约束
    if(relocalization_info)
    {
        //printf("set relocalization factor! \n");
        ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
        problem.AddParameterBlock(relo_Pose, SIZE_POSE, local_parameterization);    // 需要优化的回环帧位姿
        int retrive_feature_index = 0;
        int feature_index = -1;
        // 遍历现有地图点
        for (auto &it_per_id : f_manager.feature)
        {
            it_per_id.used_num = it_per_id.feature_per_frame.size();
            if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
                continue;
            ++feature_index;
            int start = it_per_id.start_frame;
            if(start <= relo_frame_local_index)   // 这个地图点能被对应的当前帧看到
            {   
                // 寻找回环帧能看到的地图点
                while((int)match_points[retrive_feature_index].z() < it_per_id.feature_id)
                {
                    retrive_feature_index++;
                }
                // 这个地图点也能被回环帧看到
                if((int)match_points[retrive_feature_index].z() == it_per_id.feature_id)
                {
                    // 构建一个重投影约束，这个地图点的起始帧和该回环帧之间
                    Vector3d pts_j = Vector3d(match_points[retrive_feature_index].x(), match_points[retrive_feature_index].y(), 1.0);
                    Vector3d pts_i = it_per_id.feature_per_frame[0].point;
                    
                    ProjectionFactor *f = new ProjectionFactor(pts_i, pts_j);
                    problem.AddResidualBlock(f, loss_function, para_Pose[start], relo_Pose, para_Ex_Pose[0], para_Feature[feature_index]);
                    retrive_feature_index++;
                }     
            }
        }

    }
    // Step 3 ceres优化求解
    ceres::Solver::Options options;

    options.linear_solver_type = ceres::DENSE_SCHUR;
    //options.num_threads = 2;
    options.trust_region_strategy_type = ceres::DOGLEG;
    options.max_num_iterations = NUM_ITERATIONS;
    //options.use_explicit_schur_complement = true;
    //options.minimizer_progress_to_stdout = true;
    //options.use_nonmonotonic_steps = true;
    if (marginalization_flag == MARGIN_OLD)
        // 下面的边缘化老的操作比较多，因此给他优化时间就少一些
        options.max_solver_time_in_seconds = SOLVER_TIME * 4.0 / 5.0;
    else
        options.max_solver_time_in_seconds = SOLVER_TIME;
    TicToc t_solver;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);  // ceres优化求解
    //cout << summary.BriefReport() << endl;
    ROS_DEBUG("Iterations : %d", static_cast<int>(summary.iterations.size()));
    ROS_DEBUG("solver costs: %f", t_solver.toc());
    // 把优化后double -> eigen
    //; 因为优化的是double数组，因此要把优化后的double数组的数值更新到滑窗中的Eigen变量中
    double2vector();

    // Step 4 边缘化
    // 科普一下舒尔补
    TicToc t_whole_marginalization;
    //; 如果倒数第2帧是关键帧的话，那么就需要把滑窗中最老的那一帧边缘化掉
    if (marginalization_flag == MARGIN_OLD)
    {
        // 一个用来边缘化操作的对象
        MarginalizationInfo *marginalization_info = new MarginalizationInfo();
        // 这里类似手写高斯牛顿，因此也需要都转成double数组
        //; 这里类似手写高斯牛顿，也会用到ceres之前定义的一些雅克比，因此这里还是要转化成double
        vector2double();
        // 关于边缘化有几点注意的地方
        // 1、找到需要边缘化的参数块，这里是地图点，第0帧位姿，第0帧速度零偏   
        // 2、找到构造高斯牛顿下降时跟这些待边缘化相关的参数块有关的残差约束，那就是预积分约束，重投影约束，以及上一次边缘化约束
        // 3、这些约束连接的参数块中，不需要被边缘化的参数块，就是被提供先验约束的部分，也就是滑窗中剩下的位姿和速度零偏

        // 上一次的边缘化结果
        if (last_marginalization_info)
        {
            vector<int> drop_set;
            // last_marginalization_parameter_blocks是上一次边缘化对哪些当前参数块有约束
            for (int i = 0; i < static_cast<int>(last_marginalization_parameter_blocks.size()); i++)
            {
                // 涉及到的待边缘化的上一次边缘化留下来的当前参数块只有位姿和速度零偏
                if (last_marginalization_parameter_blocks[i] == para_Pose[0] ||
                    last_marginalization_parameter_blocks[i] == para_SpeedBias[0])
                    drop_set.push_back(i);
            }
            // 处理方式和其他残差块相同
            // construct new marginlization_factor
            MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
            ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(marginalization_factor, NULL,
                                                                           last_marginalization_parameter_blocks,
                                                                           drop_set);

            marginalization_info->addResidualBlockInfo(residual_block_info);
        }
        // 只有第1个预积分和待边缘化参数块相连
        {
            //; 同理这里需要判断预积分的约束时间要<10,否则时间太久不准，那就不能边缘化形成先验约束了，直接丢掉即可
            if (pre_integrations[1]->sum_dt < 10.0)
            {
                // 跟构建ceres约束问题一样，这里也需要得到残差和雅克比
                IMUFactor* imu_factor = new IMUFactor(pre_integrations[1]);
                //; ResidualBlockInfo 残差块信息类
                ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(
                    imu_factor, NULL, //; CostFunction， 鲁邦核函数
                    //; 这个IMU预积分和哪些参数块有关
                    //; 这里vector<double *>就是取这些参数块的首地址
                    vector<double *>{para_Pose[0], para_SpeedBias[0], para_Pose[1], para_SpeedBias[1]},
                    //; 这里就是第0和1个参数块是需要被边缘化的, 注意要被边缘化的参数放在前面方便进行舒尔补
                    vector<int>{0, 1});  
                marginalization_info->addResidualBlockInfo(residual_block_info);
            }
        }
        // 遍历视觉重投影的约束
        {
            int feature_index = -1;
            for (auto &it_per_id : f_manager.feature)
            {
                it_per_id.used_num = it_per_id.feature_per_frame.size();
                if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
                    continue;

                ++feature_index;

                int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;
                // 只找能被第0帧看到的特征点
                if (imu_i != 0)
                    continue;

                Vector3d pts_i = it_per_id.feature_per_frame[0].point;
                // 遍历看到这个特征点的所有KF，通过这个特征点，建立和第0帧的约束
                for (auto &it_per_frame : it_per_id.feature_per_frame)
                {
                    imu_j++;
                    if (imu_i == imu_j)
                        continue;

                    Vector3d pts_j = it_per_frame.point;
                    // 根据是否约束延时确定残差阵
                    if (ESTIMATE_TD)
                    {
                        ProjectionTdFactor *f_td = new ProjectionTdFactor(pts_i, pts_j, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocity,
                                                                          it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td,
                                                                          it_per_id.feature_per_frame[0].uv.y(), it_per_frame.uv.y());
                        ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(f_td, loss_function,
                                                                                        vector<double *>{para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Feature[feature_index], para_Td[0]},
                                                                                        vector<int>{0, 3});
                        marginalization_info->addResidualBlockInfo(residual_block_info);
                    }
                    else
                    {
                        ProjectionFactor *f = new ProjectionFactor(pts_i, pts_j);
                        ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(
                            f, loss_function,  //; CostFunction， 鲁邦核函数
                            //; 与视觉重投影有关的参数块：i帧位姿，j帧位姿，外参，3d点的逆深度       
                            vector<double *>{para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Feature[feature_index]},
                            //; 要边缘化掉的参数块：第0帧位姿，地图点（降低fill-in现象的影响）
                            vector<int>{0, 3});  // 这里第0帧和地图点被margin
                        marginalization_info->addResidualBlockInfo(residual_block_info);
                    }
                }
            }
        }

        // 所有的残差块都收集好了
        TicToc t_pre_margin;
        // 进行预处理
        marginalization_info->preMarginalize();
        ROS_DEBUG("pre marginalization %f ms", t_pre_margin.toc());
        
        TicToc t_margin;
        // Step 边缘化操作
        marginalization_info->marginalize();
        ROS_DEBUG("marginalization %f ms", t_margin.toc());

        // 即将滑窗，因此记录新地址对应的老地址
        std::unordered_map<long, double *> addr_shift;
        for (int i = 1; i <= WINDOW_SIZE; i++)
        {
            // 位姿和速度都要滑窗移动，键是后一帧的地址，值是它即将存储到的新地址
            addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i - 1];
            addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i - 1];
        }
        // 外参和时间延时不变，不需要进行滑窗移动的操作，所以存储地址不变
        for (int i = 0; i < NUM_OF_CAM; i++)
            addr_shift[reinterpret_cast<long>(para_Ex_Pose[i])] = para_Ex_Pose[i];
        if (ESTIMATE_TD)
        {
            addr_shift[reinterpret_cast<long>(para_Td[0])] = para_Td[0];
        }
        // parameter_blocks实际上就是addr_shift的索引的集合及搬进去的新地址
        vector<double *> parameter_blocks = marginalization_info->getParameterBlocks(addr_shift);

        if (last_marginalization_info)
            delete last_marginalization_info;
        //; 保留本次边缘化的所有信息到last_marginalization_info
        last_marginalization_info = marginalization_info;   
        //; parameter_blocks代表该次边缘化对某些参数块形成约束，这些参数块在滑窗之后的存储的内存地址
        //; 同时这个值也进行保存，用于下次非线性优化之前对这些参数块添加边缘化的先验约束
        last_marginalization_parameter_blocks = parameter_blocks;   
        
    }
    //; 1.下面这种情况是边缘化倒数第二帧，是因为之前经过一些判断决定倒数第二帧不是关键帧，那么
    //;   倒数第二帧其实就不存在预积分约束和视觉重投影的约束
    //; 2.但是如果存在一种情况会对倒数第二帧存在约束，就是之前边缘化掉的老关键帧看到的3d点也可以被
    //;   倒数第二帧看到，这样的话老关键帧被边缘化掉之后也会对倒数第二帧形成约束
    else    // 边缘化倒数第二帧
    {
        // 要求有上一次边缘化的结果同时，即将被margin掉的在上一次边缘化后的约束中
        // 预积分结果合并，因此只有位姿margin掉
        if (last_marginalization_info &&
            //; 1.last_marginalization_parameter_blocks存储的是上一次边缘化操作的时候对留下的那些参数块
            //;   存在约束，然后在这个变量中存储了这些参数块进行滑窗之后存储的新的位置。
            //; 2.这里从last_marginalization_parameter_blocks中寻找是否有倒数第二帧的位姿这个参数块
            //;   如果找到了说明上次边缘化确实对倒数第二帧形成了约束，那么就进入if进行处理
            std::count(std::begin(last_marginalization_parameter_blocks), 
                       std::end(last_marginalization_parameter_blocks), para_Pose[WINDOW_SIZE - 1]))
        {

            MarginalizationInfo *marginalization_info = new MarginalizationInfo();
            vector2double();
            //; 这里的判断是多余的，一定满足
            if (last_marginalization_info)
            {
                vector<int> drop_set;
                for (int i = 0; i < static_cast<int>(last_marginalization_parameter_blocks.size()); i++)
                {
                    // 速度零偏只会margin第1个，不可能出现倒数第二个
                    //; 1.这是因为速度和零偏只在IMU约束中才有，而上次边缘化是边缘化掉滑窗中的第0帧，IMU约束
                    //;   只会在第0帧和第1帧之间建立，不可能在第0帧和倒数第二帧之间建立。
                    //; 2.但是第0帧可以通过共视的3d点和倒数第二帧之间形成约束，但此时仅能约束位姿（没有速度零偏约束），
                    //;   所以最外面的if判断成立，但下面的Assert不成立，就说明这个约束肯定是共视3d点产生的视觉约束
                    ROS_ASSERT(last_marginalization_parameter_blocks[i] != para_SpeedBias[WINDOW_SIZE - 1]);
                    // 这种case只会margin掉倒数第二个位姿
                    //; last_marginalization_parameter_blocks[i]是一个double类型的指针，而para_Pose是一个二维数组，
                    //; para_Pose[WINDOW_SIZE - 1]这种访问方式就得到二维数组中的第WINDOW_SIZE - 1个一维数组，其实也是一个指针，
                    //; 指向这个一维数组的首地址
                    if (last_marginalization_parameter_blocks[i] == para_Pose[WINDOW_SIZE - 1])
                        drop_set.push_back(i);
                }
                // construct new marginlization_factor
                // 这里只会更新一下margin factor
                MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
                ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(marginalization_factor, NULL,
                                                                               last_marginalization_parameter_blocks,
                                                                               drop_set);

                marginalization_info->addResidualBlockInfo(residual_block_info);
            }
            // 这里的操作如出一辙，先预处理，然后边缘化
            TicToc t_pre_margin;
            ROS_DEBUG("begin marginalization");
            marginalization_info->preMarginalize();
            ROS_DEBUG("end pre marginalization, %f ms", t_pre_margin.toc());

            TicToc t_margin;
            ROS_DEBUG("begin marginalization");
            marginalization_info->marginalize();
            ROS_DEBUG("end marginalization, %f ms", t_margin.toc());
            
            //; 进行滑窗中关键帧的内存地址的变化
            std::unordered_map<long, double *> addr_shift;
            for (int i = 0; i <= WINDOW_SIZE; i++)
            {
                //; 如果是倒数第二帧，因为它直接被边缘化掉了，所以直接丢掉，内存中没有它的地址了
                if (i == WINDOW_SIZE - 1)
                    continue;
                //; 如果是最后一帧，那么它的新内存地址就变成前一帧的内存地址
                else if (i == WINDOW_SIZE)  // 滑窗，最新帧成为次新帧
                {
                    addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i - 1];
                    addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i - 1];
                }
                //; 倒数第二帧之前的其他帧不受影响
                else    // 其他不变
                {
                    addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i];
                    addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i];
                }
            }
            //; 同理，外参和td不受滑窗影响，内存地址不变
            for (int i = 0; i < NUM_OF_CAM; i++)
                addr_shift[reinterpret_cast<long>(para_Ex_Pose[i])] = para_Ex_Pose[i];
            if (ESTIMATE_TD)
            {
                addr_shift[reinterpret_cast<long>(para_Td[0])] = para_Td[0];
            }
            
            //; 下面的操作也和边缘化老帧的时候一样
            vector<double *> parameter_blocks = marginalization_info->getParameterBlocks(addr_shift);
            if (last_marginalization_info)
                delete last_marginalization_info;
            last_marginalization_info = marginalization_info;
            last_marginalization_parameter_blocks = parameter_blocks;
        }
    }
    ROS_DEBUG("whole marginalization costs: %f", t_whole_marginalization.toc());
    
    ROS_DEBUG("whole time for ceres: %f", t_whole.toc());
}

// 滑动窗口 
void Estimator::slideWindow()
{
    TicToc t_margin;
    // 根据边缘化种类的不同，进行滑窗的方式也不同
    if (marginalization_flag == MARGIN_OLD)
    {
        double t_0 = Headers[0].stamp.toSec();
        //; 先备份一下将要被滑出窗口的老帧的位姿，这个主要是在移交第0帧看到的地图的管辖权的时候使用，
        //; 因为需要将第0帧看到的地图点坐标通过第0帧的位姿转移到第1帧中
        back_R0 = Rs[0];
        back_P0 = Ps[0];
        // 必须是填满了滑窗才可以, 实际上这里是一定满足的，因为如果不填满滑窗，那么连初始化都无法完成
        if (frame_count == WINDOW_SIZE)
        {
            // 一帧一帧交换过去
            for (int i = 0; i < WINDOW_SIZE; i++)
            {
                Rs[i].swap(Rs[i + 1]);

                std::swap(pre_integrations[i], pre_integrations[i + 1]);

                dt_buf[i].swap(dt_buf[i + 1]);
                linear_acceleration_buf[i].swap(linear_acceleration_buf[i + 1]);
                angular_velocity_buf[i].swap(angular_velocity_buf[i + 1]);

                Headers[i] = Headers[i + 1];
                //; i最大是WINDOW_SIZE - 1，所以交换后，WINDOW_SIZE的地方变成滑窗中第0帧的数据
                Ps[i].swap(Ps[i + 1]);  
                Vs[i].swap(Vs[i + 1]);
                Bas[i].swap(Bas[i + 1]);
                Bgs[i].swap(Bgs[i + 1]);
            }
            // 最后一帧的状态量赋上当前值，最为初始值，因为下一次仅仅使用
            // IMU进行状态变量的估计的时候，需要在最新的一帧的位姿的基础上进行推算
            Headers[WINDOW_SIZE] = Headers[WINDOW_SIZE - 1];
            Ps[WINDOW_SIZE] = Ps[WINDOW_SIZE - 1];
            Vs[WINDOW_SIZE] = Vs[WINDOW_SIZE - 1];
            Rs[WINDOW_SIZE] = Rs[WINDOW_SIZE - 1];
            Bas[WINDOW_SIZE] = Bas[WINDOW_SIZE - 1];
            Bgs[WINDOW_SIZE] = Bgs[WINDOW_SIZE - 1];
            // 预积分量就得置零，这里就是直接删除掉预积分类这个指针
            delete pre_integrations[WINDOW_SIZE];
            //; 用最近的一些变量来新生成一个预积分类，然后等待最新的IMU数据到来，进行下一次的预积分
            pre_integrations[WINDOW_SIZE] = 
                new IntegrationBase{acc_0, gyr_0, Bas[WINDOW_SIZE], Bgs[WINDOW_SIZE]};
           
           
            // buffer清空，等待新的数据来填，这里主要是等待新的IMU数据到来
            dt_buf[WINDOW_SIZE].clear();
            linear_acceleration_buf[WINDOW_SIZE].clear();
            angular_velocity_buf[WINDOW_SIZE].clear(); 

            // 清空all_image_frame最老帧之前的状态
            //; 1.这个肯定会进来啊？判断solver_flag还有什么意义
            //; 2.下面这个操作其实和初始化有关系，因为处理的是all_image_frame
            if (true || solver_flag == INITIAL)
            {
                // 预积分量是堆上的空间，因此需要手动释放
                map<double, ImageFrame>::iterator it_0;
                //; 找到滑出滑窗的最老帧在all_image_frame中的位置
                it_0 = all_image_frame.find(t_0);  
                //; 释放堆上面的空间，下面两步缺一不可（所以推荐使用智能指针）
                delete it_0->second.pre_integration;    //; 1.删除指针，否则会导致内存泄漏
                it_0->second.pre_integration = nullptr; //; 2.赋值空指针，否则会导致野指针出现，程序运行可能会带来不可预知的错误
                //; 把all_image_frame中滑出滑窗的帧之前的所有帧都删掉
                for (map<double, ImageFrame>::iterator it = all_image_frame.begin(); it != it_0; ++it)
                {
                    if (it->second.pre_integration)
                        delete it->second.pre_integration;
                    it->second.pre_integration = NULL;
                }
                // 释放完空间之后再erase，不能直接erase，否则会导致内存泄漏现象
                all_image_frame.erase(all_image_frame.begin(), it_0);
                all_image_frame.erase(t_0);
            }
            //; 之前一直没有对地图点进行操作，这里对地图点进行操作
            slideWindowOld();
        }
    }
    else
    {
        if (frame_count == WINDOW_SIZE)
        {
            // 将最后两个预积分观测合并成一个
            //; dt_buf[frame_count]是最后一帧图像对应的所有的IMU值，
            //; dt_buf[frame_count-1]是倒数第二帧（被边缘化掉的）图像对应的所有的IMU值，
            //; 所以这里把两个预积分合成一个，就是把最后一帧图像的IMU值加入到倒数第二帧的IMU值中，
            //; 然后重新进行传播，计算预积分、协方差等
            for (unsigned int i = 0; i < dt_buf[frame_count].size(); i++)
            {
                double tmp_dt = dt_buf[frame_count][i];
                Vector3d tmp_linear_acceleration = linear_acceleration_buf[frame_count][i];
                Vector3d tmp_angular_velocity = angular_velocity_buf[frame_count][i];

                pre_integrations[frame_count - 1]->push_back(tmp_dt, tmp_linear_acceleration, tmp_angular_velocity);

                dt_buf[frame_count - 1].push_back(tmp_dt);
                linear_acceleration_buf[frame_count - 1].push_back(tmp_linear_acceleration);
                angular_velocity_buf[frame_count - 1].push_back(tmp_angular_velocity);
            }
            // 简单的滑窗交换
            Headers[frame_count - 1] = Headers[frame_count];
            Ps[frame_count - 1] = Ps[frame_count];
            Vs[frame_count - 1] = Vs[frame_count];
            Rs[frame_count - 1] = Rs[frame_count];
            Bas[frame_count - 1] = Bas[frame_count];
            Bgs[frame_count - 1] = Bgs[frame_count];

            // reset最新预积分量
            delete pre_integrations[WINDOW_SIZE];
            //; 这里不new的话就要赋值成nullptr，否则会产生也野指针的问题
            pre_integrations[WINDOW_SIZE] = new IntegrationBase{acc_0, gyr_0, Bas[WINDOW_SIZE], Bgs[WINDOW_SIZE]};
            // clear相关buffer
            dt_buf[WINDOW_SIZE].clear();
            linear_acceleration_buf[WINDOW_SIZE].clear();
            angular_velocity_buf[WINDOW_SIZE].clear();

            slideWindowNew();
        }
    }
}

// real marginalization is removed in solve_ceres()
// 对被移除的倒数第二帧的地图点进行处理
void Estimator::slideWindowNew()
{
    sum_of_front++;
    f_manager.removeFront(frame_count);
}


// real marginalization is removed in solve_ceres()
// 由于地图点是绑定在第一个看见它的位姿上的，因此需要对被移除的帧看见的地图点进行解绑，以及每个地图点的首个观测帧id减1
//; 这里的解绑就是把地图点绑定到看到这个地图点的第1帧上（假设上一次被滑出滑窗的是看到这个地图点的第0帧）
void Estimator::slideWindowOld()
{
    sum_of_back++;

    bool shift_depth = solver_flag == NON_LINEAR ? true : false;
    if (shift_depth)    // 如果初始化过了
    {
        Matrix3d R0, R1;
        Vector3d P0, P1;
        // back_R0 back_P0 是被移除的帧的位姿
        //; 下面的操作就是计算 移除滑出的帧 和 当前滑窗中的第0帧 的相机系位姿，
        //; 因为滑窗中维护的都是IMU系的位姿，而和3d点有关的是在相机系下的位姿
        R0 = back_R0 * ric[0];  // 上一次被移除的相机的姿态
        R1 = Rs[0] * ric[0];    // 当前滑窗中最老的相机姿态
        P0 = back_P0 + back_R0 * tic[0];    // 被移除的相机的位置
        P1 = Ps[0] + Rs[0] * tic[0];    // 当前最老的相机位置

        // 下面要做的事情把被移除帧看见地图点的管理权交给当前的最老帧
        f_manager.removeBackShiftDepth(R0, P0, R1, P1);
    }
    //; 还没有初始化结束
    else
        f_manager.removeBack();
}

/**
 * @brief 接受回环帧的消息
 * 
 * @param[in] _frame_stamp 
 * @param[in] _frame_index 
 * @param[in] _match_points 
 * @param[in] _relo_t 
 * @param[in] _relo_r 
 */
void Estimator::setReloFrame(double _frame_stamp, int _frame_index, vector<Vector3d> &_match_points, Vector3d _relo_t, Matrix3d _relo_r)
{
    relo_frame_stamp = _frame_stamp;
    relo_frame_index = _frame_index;
    match_points.clear();
    match_points = _match_points;
    prev_relo_t = _relo_t;
    prev_relo_r = _relo_r;
    // 在滑窗中寻找当前帧，因为VIO送给回环结点的是倒数第三帧，因此，很有可能这个当前帧还在滑窗里
    for(int i = 0; i < WINDOW_SIZE; i++)
    {
        if(relo_frame_stamp == Headers[i].stamp.toSec())
        {
            relo_frame_local_index = i; // 对应滑窗中的第i帧
            relocalization_info = 1;    // 这是一个有效的回环信息
            for (int j = 0; j < SIZE_POSE; j++)
                relo_Pose[j] = para_Pose[i][j]; // 借助VIO优化回环帧位姿，初值先设为当前帧位姿
        }
    }
}

