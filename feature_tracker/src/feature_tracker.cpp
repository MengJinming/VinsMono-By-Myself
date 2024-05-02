//; 2022/02/25 全部阅读完毕 

#include "feature_tracker.h"

int FeatureTracker::n_id = 0;

bool inBorder(const cv::Point2f &pt)
{
    const int BORDER_SIZE = 1;
    int img_x = cvRound(pt.x);
    int img_y = cvRound(pt.y);
    return BORDER_SIZE <= img_x && img_x < COL - BORDER_SIZE && BORDER_SIZE <= img_y && img_y < ROW - BORDER_SIZE;
}

// 根据状态位，进行“瘦身”
void reduceVector(vector<cv::Point2f> &v, vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}

void reduceVector(vector<int> &v, vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}


FeatureTracker::FeatureTracker()
{
}

// 给现有的特征点设置mask，目的为了特征点的均匀化
void FeatureTracker::setMask()
{
    if(FISHEYE)
        mask = fisheye_mask.clone();
    else
        mask = cv::Mat(ROW, COL, CV_8UC1, cv::Scalar(255));
    

    // prefer to keep features that are tracked for long time
    vector<pair<int, pair<cv::Point2f, int>>> cnt_pts_id;

    for (unsigned int i = 0; i < forw_pts.size(); i++)
        //; 特征点被追踪的次数， <特征点坐标，特征点id>
        cnt_pts_id.push_back(make_pair(track_cnt[i], make_pair(forw_pts[i], ids[i])));
    // 利用光流特点，追踪多的稳定性好，排前面
    //; 注意写法，第二个形参是lambda表达式
    sort(cnt_pts_id.begin(), cnt_pts_id.end(), [](const pair<int, pair<cv::Point2f, int>> &a, const pair<int, pair<cv::Point2f, int>> &b)
         {
            return a.first > b.first;
         });

    forw_pts.clear();
    ids.clear();
    track_cnt.clear();

    for (auto &it : cnt_pts_id)
    {
        //; 排序后的特征点位置，如果还没有被画圈，那么就把这个特征点挑选存储起来，
        //; 并且画圈表示这个特征点周围不能再提点
        if (mask.at<uchar>(it.second.first) == 255)
        {
            // 把挑选剩下的特征点重新放进容器
            forw_pts.push_back(it.second.first);
            ids.push_back(it.second.second);
            track_cnt.push_back(it.first);
            // opencv函数，把周围一个圆内全部置0,这个区域不允许别的特征点存在，避免特征点过于集中
            //; 0是填充的值，-1表示这个半径内的圆全部进行填充，而不是只算圆周上的点
            cv::circle(mask, it.second.first, MIN_DIST, 0, -1);
        }
    }
}

// 把新的点加入容器，id给-1作为区分
void FeatureTracker::addPoints()
{
    for (auto &p : n_pts)
    {
        forw_pts.push_back(p);
        ids.push_back(-1);
        track_cnt.push_back(1);
    }
}

/**
 * @brief 
 * 
 * @param[in] _img 输入图像
 * @param[in] _cur_time 图像的时间戳
 * 1、图像均衡化预处理
 * 2、光流追踪
 * 3、提取新的特征点（如果发布）
 * 4、所有特征点去畸变，计算速度
 */
void FeatureTracker::readImage(const cv::Mat &_img, double _cur_time)
{
    cv::Mat img;
    TicToc t_r;
    cur_time = _cur_time;   //; 成员变量，时间戳

    //; 是否要进行图像均衡化，是EQUALIZE的值是从配置文件中读出来的
    if (EQUALIZE)
    {
        // 图像太暗或者太亮，提特征点比较难，所以均衡化一下
        // ! opencv 函数看一下
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
        TicToc t_c;
        clahe->apply(_img, img);
        ROS_DEBUG("CLAHE costs: %fms", t_c.toc());
    }
    else
        img = _img;

    // 这里forw（forward）表示当前，cur表示上一帧
    //; forw_img是类成员变量，是当前输入的图像，也就是当前帧
    if (forw_img.empty())   // 第一次输入图像，prev_img这个没用
    {
        prev_img = cur_img = forw_img = img;
    }
    else
    {
        forw_img = img;
    }

    //; vector<cv::Point2f> forw_pts
    forw_pts.clear();   //; 当前帧光流追踪到的特征点，也是类成员变量

    if (cur_pts.size() > 0) // 上一帧有特征点，就可以进行光流追踪了
    {
        TicToc t_o;
        vector<uchar> status;
        vector<float> err;
        // 调用opencv函数进行光流追踪
        // Step 1 通过opencv光流追踪给的状态位剔除outlier
        //; 形参分别是：上一帧图像，当前帧图像，上一帧光流追踪到的特征点，当前帧追踪后得到的特征点
        //; status:上一帧的特征点在当前帧是否被成功追踪； err:错误vector?； 
        //; cv::Size(21, 21) 是光流追踪的窗口大小，因为光流是匹配特征点周围的一个patch的光度一致
        //; 3是金字塔层数，注意不算第0层，也就是实际上是3+1=4层
        cv::calcOpticalFlowPyrLK(cur_img, forw_img, cur_pts, forw_pts, status, err, cv::Size(21, 21), 3);

        for (int i = 0; i < int(forw_pts.size()); i++)
            // Step 2 通过图像边界剔除outlier
            if (status[i] && !inBorder(forw_pts[i]))    // 追踪状态好检查在不在图像范围
                status[i] = 0;
        reduceVector(prev_pts, status); // 没用到
        //; 本次追踪结果把上一帧和上上帧的追踪结果也瘦身了，为什么？
        reduceVector(cur_pts, status);
        reduceVector(forw_pts, status);
        //; 下面这些变量在本次光流追踪后并未赋值，查看在哪里赋值？
        //; 注意：实际上这些变量都是上一帧追踪的特征点的属性，所以上一帧追踪后已经赋值了，这里只是把本次没有
        //; 追踪到的哪些特征点的属性删除掉
        reduceVector(ids, status);  // 特征点的id
        reduceVector(cur_un_pts, status);   // 去畸变后的坐标
        reduceVector(track_cnt, status);    // 追踪次数
        ROS_DEBUG("temporal optical flow costs: %fms", t_o.toc());
    }
    // 被追踪到的是上一帧就存在的，因此追踪数+1
    for (auto &n : track_cnt)
        n++;

    //;如果当前帧需要给后端发送，那么还需要进一步剔除外点 
    if (PUB_THIS_FRAME)
    {
        // Step 3 通过对级约束来剔除outlier
        rejectWithF();
        ROS_DEBUG("set mask begins");

        TicToc t_m;
        //; 设置mask的同时，把追踪次数较多且没有被mask覆盖的点保存起来了
        setMask();
        ROS_DEBUG("set mask costs %fms", t_m.toc());

        ROS_DEBUG("detect feature begins");
        TicToc t_t;
        //; 经过setMask对特征点进行均匀化之后，得到的forw_pts是追踪次数最多的特征点
        //; 此时可能提取的特征点少于要求的特征点数量，所以需要再提取一些
        int n_max_cnt = MAX_CNT - static_cast<int>(forw_pts.size());    //; 这里为什么要static_cast处理？
        if (n_max_cnt > 0)
        {
            if(mask.empty())
                cout << "mask is empty " << endl;
            if (mask.type() != CV_8UC1)
                cout << "mask type wrong " << endl;
            if (mask.size() != forw_img.size())
                cout << "wrong size " << endl;
            // 只有发布才可以提取更多特征点，同时避免提的点进mask
            // 会不会这些点集中？会，不过没关系，他们下一次作为老将就得接受均匀化的洗礼
            //; 这个函数是opencv中的一个角点检测函数，实际上光流法检测出来的也是角点。
            //; 最后的mask的作用是如果mask为0的点，那么就会放弃这个位置检测出来的角点。
            //; 0.01是质量系数，越小提点的要求越宽松
            cv::goodFeaturesToTrack(forw_img, n_pts, MAX_CNT - forw_pts.size(), 0.01, MIN_DIST, mask);
        }
        else
            n_pts.clear();
        ROS_DEBUG("detect feature costs: %fms", t_t.toc());

        ROS_DEBUG("add feature begins");
        TicToc t_a;
        addPoints();    //; 把上一步用goodFeaturesToTrack提取不够的特征点再加入
        ROS_DEBUG("selectFeature costs: %fms", t_a.toc());
    }
    //; 更新变量
    prev_img = cur_img;
    prev_pts = cur_pts;
    prev_un_pts = cur_un_pts;   // 以上三个量无用
    cur_img = forw_img; // 实际上是上一帧的图像
    cur_pts = forw_pts; // 上一帧的特征点
    //; 这里把所有的点再统一去一次畸变，还会计算光流追踪到的特征点的速度
    //; 因为其实自始至终这些点都没有进行去畸变的操作，在rejectWithF()函数中是对局部变量进行去畸变，目的是为了计算F排除外点
    undistortedPoints();
    prev_time = cur_time;     //; 更新时间戳的时间变量
}

/**
 * @brief 
 * 
 */
void FeatureTracker::rejectWithF()
{
    // 当前被追踪到的光流至少8个点
    if (forw_pts.size() >= 8)
    {
        ROS_DEBUG("FM ransac begins");
        TicToc t_f;
        //; 存储去畸变后的特征点？注意写法，括号里面的值应该是vector的长度 
        //; 而且这里只是一个局部变量，用于对极几何判断外点来使用，也就是得到status
        vector<cv::Point2f> un_cur_pts(cur_pts.size()), un_forw_pts(forw_pts.size());
        
        //; 遍历上一帧的特征点
        for (unsigned int i = 0; i < cur_pts.size(); i++)
        {
            Eigen::Vector3d tmp_p;
            //; 对上一帧的图像进行处理，得到去畸变后的在相机归一化坐标系下的坐标
            m_camera->liftProjective(Eigen::Vector2d(cur_pts[i].x, cur_pts[i].y), tmp_p);
            // 这里用一个虚拟相机，原因同样参考https://github.com/HKUST-Aerial-Robotics/VINS-Mono/issues/48
            // 这里有个好处就是对F_THRESHOLD和相机无关
            //; 注意这里的解释很好，就是因为这里是进行判断外点，所以需要有一个像素阈值，因此投影到一个虚拟
            //; 焦距的相机上之后，这个像素阈值就和相机焦距无关了，也就是和实际使用的相机无关了
            // 投影到虚拟相机的像素坐标系
            //; 1.注意下面访问Eigen的Vector3d的写法，是.x()
            //; 2.这里的操作就是先把像素坐标转到归一化相机平面，然后在归一化相机平面对特征点进行去畸变，
            //;   然后又投影到像素坐标，得到像素平面上的去畸变后的像素坐标。
            //; 为什么要按2.中的步骤这么去畸变呢？不能直接在像素坐标上进行去畸变处理吗？
            tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + COL / 2.0;
            tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + ROW / 2.0;
            un_cur_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());

            m_camera->liftProjective(Eigen::Vector2d(forw_pts[i].x, forw_pts[i].y), tmp_p);
            tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + COL / 2.0;
            tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + ROW / 2.0;
            un_forw_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());
        }

        vector<uchar> status;
        // opencv接口计算本质矩阵，某种意义也是一种对级约束的outlier剔除
        //; F_THRESHOLD 是剔除外点的阈值，也就是匹配点离对极约束算出来的极线距离不能超过这个阈值
        //; 另外这里就可以看出来为什么上一步光流追踪后要把上一帧的特征点也删除了，因为这里计算F矩阵的特征点
        //; 必须是对应的
        cv::findFundamentalMat(un_cur_pts, un_forw_pts, cv::FM_RANSAC, F_THRESHOLD, 0.99, status);
        int size_a = cur_pts.size();
        reduceVector(prev_pts, status);
        reduceVector(cur_pts, status);
        reduceVector(forw_pts, status);
        reduceVector(cur_un_pts, status);
        reduceVector(ids, status);
        reduceVector(track_cnt, status);
        ROS_DEBUG("FM ransac: %d -> %lu: %f", size_a, forw_pts.size(), 1.0 * forw_pts.size() / size_a);
        ROS_DEBUG("FM ransac costs: %fms", t_f.toc());
    }
}

/**
 * @brief 
 * 
 * @param[in] i 
 * @return true 
 * @return false 
 *  给新的特征点赋上id,越界就返回false
 */
bool FeatureTracker::updateID(unsigned int i)
{
    //; 输入的i是特征点在数组中的索引
    if (i < ids.size())
    {
        if (ids[i] == -1)
            ids[i] = n_id++;   //; n_id是类的静态成员变量，需要在类外赋值，可以看成是一个全局变量
        return true;
    }
    else
        return false;
}

void FeatureTracker::readIntrinsicParameter(const string &calib_file)
{
    ROS_INFO("reading paramerter of camera %s", calib_file.c_str());
    //; 读到的相机内参赋给m_camera, 里面的写法跳来跳去比较繁琐，暂时先不管
    m_camera = CameraFactory::instance()->generateCameraFromYamlFile(calib_file);
}

//; 显示去畸变的图像，应该是实验中验证递归去畸变算法的效果
void FeatureTracker::showUndistortion(const string &name)
{
    cv::Mat undistortedImg(ROW + 600, COL + 600, CV_8UC1, cv::Scalar(0));
    vector<Eigen::Vector2d> distortedp, undistortedp;
    for (int i = 0; i < COL; i++)
        for (int j = 0; j < ROW; j++)
        {
            Eigen::Vector2d a(i, j);
            Eigen::Vector3d b;
            m_camera->liftProjective(a, b);
            distortedp.push_back(a);
            undistortedp.push_back(Eigen::Vector2d(b.x() / b.z(), b.y() / b.z()));
            //printf("%f,%f->%f,%f,%f\n)\n", a.x(), a.y(), b.x(), b.y(), b.z());
        }
    for (int i = 0; i < int(undistortedp.size()); i++)
    {
        cv::Mat pp(3, 1, CV_32FC1);
        pp.at<float>(0, 0) = undistortedp[i].x() * FOCAL_LENGTH + COL / 2;
        pp.at<float>(1, 0) = undistortedp[i].y() * FOCAL_LENGTH + ROW / 2;
        pp.at<float>(2, 0) = 1.0;
        //cout << trackerData[0].K << endl;
        //printf("%lf %lf\n", p.at<float>(1, 0), p.at<float>(0, 0));
        //printf("%lf %lf\n", pp.at<float>(1, 0), pp.at<float>(0, 0));
        if (pp.at<float>(1, 0) + 300 >= 0 && pp.at<float>(1, 0) + 300 < ROW + 600 && pp.at<float>(0, 0) + 300 >= 0 && pp.at<float>(0, 0) + 300 < COL + 600)
        {
            undistortedImg.at<uchar>(pp.at<float>(1, 0) + 300, pp.at<float>(0, 0) + 300) = cur_img.at<uchar>(distortedp[i].y(), distortedp[i].x());
        }
        else
        {
            //ROS_ERROR("(%f %f) -> (%f %f)", distortedp[i].y, distortedp[i].x, pp.at<float>(1, 0), pp.at<float>(0, 0));
        }
    }
    cv::imshow(name, undistortedImg);
    cv::waitKey(0);
}

// 当前帧所有点统一去畸变，同时计算特征点速度，用来后续时间戳标定
void FeatureTracker::undistortedPoints()
{
    cur_un_pts.clear();
    cur_un_pts_map.clear();     //; map<int, cv::Point2f> cur_un_pts_map;
    //cv::undistortPoints(cur_pts, un_pts, K, cv::Mat());
    for (unsigned int i = 0; i < cur_pts.size(); i++)
    {
        // 有的之前去过畸变了，这里连同新人重新做一次
        Eigen::Vector2d a(cur_pts[i].x, cur_pts[i].y);
        Eigen::Vector3d b;
        //; 对上一帧的特征点再去一次畸变，得到b是相机归一化平面下的坐标
        m_camera->liftProjective(a, b);
        cur_un_pts.push_back(cv::Point2f(b.x() / b.z(), b.y() / b.z()));
        // id->坐标 组成的map
        cur_un_pts_map.insert(make_pair(ids[i], cv::Point2f(b.x() / b.z(), b.y() / b.z())));
        //printf("cur pts id %d %f %f", ids[i], cur_un_pts[i].x, cur_un_pts[i].y);
    }
    // caculate points velocity
    if (!prev_un_pts_map.empty())
    {
        double dt = cur_time - prev_time;
        pts_velocity.clear();
        for (unsigned int i = 0; i < cur_un_pts.size(); i++)
        {
            //; 不是新提取的点
            if (ids[i] != -1)
            {
                std::map<int, cv::Point2f>::iterator it;
                it = prev_un_pts_map.find(ids[i]);
                // 找到同一个特征点
                if (it != prev_un_pts_map.end())  //; 寻找上帧的特征点在上上帧中是否存在
                {
                    double v_x = (cur_un_pts[i].x - it->second.x) / dt;
                    double v_y = (cur_un_pts[i].y - it->second.y) / dt;
                    // 得到在归一化平面的速度
                    pts_velocity.push_back(cv::Point2f(v_x, v_y));
                }
                else
                    pts_velocity.push_back(cv::Point2f(0, 0));
            }
            //; 如果这个是新提取的点，那么就把速度设置成0
            else  
            {
                pts_velocity.push_back(cv::Point2f(0, 0));
            }
        }
    }
    else
    {
        // 第一帧的情况
        for (unsigned int i = 0; i < cur_pts.size(); i++)
        {
            pts_velocity.push_back(cv::Point2f(0, 0));
        }
    }
    prev_un_pts_map = cur_un_pts_map;
}
