#include "initial_ex_rotation.h"

InitialEXRotation::InitialEXRotation(){
    frame_count = 0;
    Rc.push_back(Matrix3d::Identity());
    Rc_g.push_back(Matrix3d::Identity());
    Rimu.push_back(Matrix3d::Identity());
    ric = Matrix3d::Identity();
}

// 标定imu和相机之间的旋转外参，通过imu和图像计算的旋转使用手眼标定计算获得
bool InitialEXRotation::CalibrationExRotation(vector<pair<Vector3d, Vector3d>> corres, Quaterniond delta_q_imu, Matrix3d &calib_ric_result)
{
    frame_count++;     //; 注意这个是InitialEXRotation类的成员变量，不要搞混了
    // 根据特征关联求解两个连续帧相机的旋转R12
    Rc.push_back(solveRelativeR(corres));       //; 注意是camera2 到 camera1 的旋转
    Rimu.push_back(delta_q_imu.toRotationMatrix()); //; 注意是从imu2 到 imu1 的旋转
    // 通过外参把imu的旋转转移到相机坐标系
    //; 这里利用上一次求解的外参，把本次imu的预积分值，转化成本次相机到上一次相机的旋转，即camera2 到 camera1 的旋转
    //; 这部分是用于鲁棒核函数的使用的
    Rc_g.push_back(ric.inverse() * delta_q_imu * ric);  // ric是上一次求解得到的外参

    //; 这里是构造一个超定方程的系数矩阵
    Eigen::MatrixXd A(frame_count * 4, 4);
    A.setZero();
    int sum_ok = 0;
    //; 问题1：为什么i从1开始？那么第0帧的旋转什么时候使用？
    //; 问题2：另外从这里来看，是每插入一个关键帧，都利用从头到尾的所有关键帧组成A，计算一次外参？
    for (int i = 1; i <= frame_count; i++)
    {
        //; 使用下面两个旋转就是为了实现核函数，也就是判断本次相机匹配计算的旋转如果太离谱，那么就降低它的权重
        Quaterniond r1(Rc[i]);      //; 相机匹配计算的旋转
        Quaterniond r2(Rc_g[i]);    //; 根据上一次计算的外参，和本次的IMU预积分旋转，计算相机之间的旋转

        //; angularDistance是四元数的库函数，计算两个旋转之间的角度差（转成轴角，单位是弧度）
        double angular_distance = 180 / M_PI * r1.angularDistance(r2);
        ROS_DEBUG(
            "%d %f", i, angular_distance);
        //; 一个简单的核函数，
        double huber = angular_distance > 5.0 ? 5.0 / angular_distance : 1.0;
        ++sum_ok;
        Matrix4d L, R;

        double w = Quaterniond(Rc[i]).w();  //; w是实部
        Vector3d q = Quaterniond(Rc[i]).vec();
        //; 四元数转矩阵左右乘，注意这里使用的四元数是虚部在前，实部在后
        L.block<3, 3>(0, 0) = w * Matrix3d::Identity() + Utility::skewSymmetric(q);
        L.block<3, 1>(0, 3) = q;
        L.block<1, 3>(3, 0) = -q.transpose();
        L(3, 3) = w;

        Quaterniond R_ij(Rimu[i]);
        w = R_ij.w();
        q = R_ij.vec();
        R.block<3, 3>(0, 0) = w * Matrix3d::Identity() - Utility::skewSymmetric(q);
        R.block<3, 1>(0, 3) = q;
        R.block<1, 3>(3, 0) = -q.transpose();
        R(3, 3) = w;

        //; 问题：这个huber的使用作用在超定方程上如何起作用不是特别明白，如果相差太大，那么huber参数就很小
        //;      这样在矩阵中这部分就会变小，会如何影响最后的结果呢？
        //; 解答：是否可以把一个大矩阵的问题想成很多小矩阵的最小二乘问题？这样huber参数作用的地方就相当于在
        //;      最小二乘的这一项前面×了一个系数，这样就减少了它在损失函数中的占比
        A.block<4, 4>((i - 1) * 4, 0) = huber * (L - R);    // 作用在残差上面
    }

    //; JacobiSVD是Eigen库中的内容
    JacobiSVD<MatrixXd> svd(A, ComputeFullU | ComputeFullV);
    Matrix<double, 4, 1> x = svd.matrixV().col(3);      //; svd分解的最后一列
    Quaterniond estimated_R(x);     //; 上面的公式分解得到的是Rcb? 和推导中不太一样
    ric = estimated_R.toRotationMatrix().inverse(); //; 这里转成Rbc
    //cout << svd.singularValues().transpose() << endl;
    //cout << ric << endl;
    Vector3d ric_cov;
    ric_cov = svd.singularValues().tail<3>();   //; 得到奇异值的后面3个
    // 倒数第二个奇异值，因为旋转是3个自由度，因此检查一下第三小的奇异值是否足够大，通常需要足够的运动激励才能保证得到没有奇异的解
    //; 在尾部3个奇异值中的索引是1，对应原来的奇异值就是第3个
    //! 问题：下面这个判断的依据没有很明白？
    //! 解答：在手写VIO的课程中贺一佳博士给出了简单的解释：
    //;      其实判断倒数第二维的奇异值是否>0.25，就是在判断A这个系数矩阵是不是一个好矩阵，即是否有比较好的数值稳定性
    //;      比如在手写VIO的课程中有一个曲线拟合的问题，给定的数据噪声非常大，并且给定的数据段区间又比较小，这样
    //;      进行曲线拟合的话得到的结果就非常不准确。解决方法要么降低噪声，要么增加数据拟合的区间段。
    //;      这里是同样的道理，判断A的奇异值是否满足要求，就相当于判断A这个系数矩阵代表的运动是否是有效的
    //;     （按照上面的解释，是否有足够的运动，运动足够大的时候，相当于降低了噪声的影响）。
    //! 新的问题：阈值0.25可能是经验值不追究，为什么是判断倒数第2个奇异值呢？
    if (frame_count >= WINDOW_SIZE && ric_cov(1) > 0.25)   
    {
        calib_ric_result = ric;
        return true;
    }
    else
        return false;
}

//; 通过两帧图像匹配的特征点计算本质矩阵，并从本质矩阵中分解得到R12，也就是后一帧到前一帧的旋转变化
Matrix3d InitialEXRotation::solveRelativeR(const vector<pair<Vector3d, Vector3d>> &corres)
{
    //; 去看一下十四讲，这里为什么要>=9个点？
    if (corres.size() >= 9)
    {
        vector<cv::Point2f> ll, rr;
        for (int i = 0; i < int(corres.size()); i++)
        {
            ll.push_back(cv::Point2f(corres[i].first(0), corres[i].first(1)));
            rr.push_back(cv::Point2f(corres[i].second(0), corres[i].second(1)));
        }
        // 这里用的是相机坐标系，因此这个函数得到的也就是E矩阵
        cv::Mat E = cv::findFundamentalMat(ll, rr);
        cv::Mat_<double> R1, R2, t1, t2;
        decomposeE(E, R1, R2, t1, t2);

        // 旋转矩阵的行列式应该是1,这里如果是-1就取一下反
        if (determinant(R1) + 1.0 < 1e-09)
        {
            E = -E;
            decomposeE(E, R1, R2, t1, t2);
        }
        double ratio1 = max(testTriangulation(ll, rr, R1, t1), testTriangulation(ll, rr, R1, t2));
        double ratio2 = max(testTriangulation(ll, rr, R2, t1), testTriangulation(ll, rr, R2, t2));
        cv::Mat_<double> ans_R_cv = ratio1 > ratio2 ? R1 : R2;

        // 解出来的是R21

        Matrix3d ans_R_eigen;
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                ans_R_eigen(j, i) = ans_R_cv(i, j); // 这里转换成R12
        return ans_R_eigen;
    }
    return Matrix3d::Identity();
}

/**
 * @brief 通过三角化来检查R t是否合理
 * 
 * @param[in] l l相机的观测
 * @param[in] r r相机的观测
 * @param[in] R 旋转矩阵
 * @param[in] t 位移
 * @return double 
 */
double InitialEXRotation::testTriangulation(const vector<cv::Point2f> &l,
                                          const vector<cv::Point2f> &r,
                                          cv::Mat_<double> R, cv::Mat_<double> t)
{
    cv::Mat pointcloud;
    // 其中一帧设置为单位阵
    cv::Matx34f P = cv::Matx34f(1, 0, 0, 0,
                                0, 1, 0, 0,
                                0, 0, 1, 0);
    // 第二帧就设置为R t对应的位姿
    cv::Matx34f P1 = cv::Matx34f(R(0, 0), R(0, 1), R(0, 2), t(0),
                                 R(1, 0), R(1, 1), R(1, 2), t(1),
                                 R(2, 0), R(2, 1), R(2, 2), t(2));
    // 看一下opencv的定义
    cv::triangulatePoints(P, P1, l, r, pointcloud);
    int front_count = 0;
    for (int i = 0; i < pointcloud.cols; i++)
    {
        // 因为是齐次的，所以要求最后一维等于1
        double normal_factor = pointcloud.col(i).at<float>(3);
        // 得到在各自相机坐标系下的3d坐标
        cv::Mat_<double> p_3d_l = cv::Mat(P) * (pointcloud.col(i) / normal_factor);
        cv::Mat_<double> p_3d_r = cv::Mat(P1) * (pointcloud.col(i) / normal_factor);
        // 通过深度是否大于0来判断是否合理
        if (p_3d_l(2) > 0 && p_3d_r(2) > 0)
            front_count++;
    }
    ROS_DEBUG("MotionEstimator: %f", 1.0 * front_count / pointcloud.cols);
    return 1.0 * front_count / pointcloud.cols;
}

// 具体解法参考多视角几何
void InitialEXRotation::decomposeE(cv::Mat E,
                                 cv::Mat_<double> &R1, cv::Mat_<double> &R2,
                                 cv::Mat_<double> &t1, cv::Mat_<double> &t2)
{
    cv::SVD svd(E, cv::SVD::MODIFY_A);
    cv::Matx33d W(0, -1, 0,
                  1, 0, 0,
                  0, 0, 1);
    cv::Matx33d Wt(0, 1, 0,
                   -1, 0, 0,
                   0, 0, 1);
    R1 = svd.u * cv::Mat(W) * svd.vt;
    R2 = svd.u * cv::Mat(Wt) * svd.vt;
    t1 = svd.u.col(2);
    t2 = -svd.u.col(2);
}
