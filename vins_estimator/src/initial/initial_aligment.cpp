#include "initial_alignment.h"

/**
 * @brief 求解陀螺仪零偏，同时利用求出来的零偏重新进行预积分
 * 
 * @param[in] all_image_frame 
 * @param[in] Bgs 
 */
void solveGyroscopeBias(map<double, ImageFrame> &all_image_frame, Vector3d* Bgs)
{
    Matrix3d A;
    Vector3d b;
    Vector3d delta_bg;
    A.setZero();
    b.setZero();
    map<double, ImageFrame>::iterator frame_i;
    map<double, ImageFrame>::iterator frame_j;
    //; 遍历滑窗中的所有普通帧
    for (frame_i = all_image_frame.begin(); next(frame_i) != all_image_frame.end(); frame_i++)
    {
        frame_j = next(frame_i);
        MatrixXd tmp_A(3, 3);
        tmp_A.setZero();
        VectorXd tmp_b(3);
        tmp_b.setZero();
        //; 首先注意ImageFrame中的R存储的是q_wc * q_cb = q_wb，也就是相机的位姿乘以外参变成了IMU的位姿
        //; q_bi_w * q_w_bj = q_bi_bj，即后一帧IMU到前一帧IMU的旋转
        Eigen::Quaterniond q_ij(frame_i->second.R.transpose() * frame_j->second.R);
        //; tmp_A 就是q的预积分关于bg的雅克比J
        tmp_A = frame_j->second.pre_integration->jacobian.template block<3, 3>(O_R, O_BG);
        //; tmp_b则是右侧的，这里和公式也可以对应上
        tmp_b = 2 * (frame_j->second.pre_integration->delta_q.inverse() * q_ij).vec();
        //; 这里用累加的方法，和摞起来变成一个大矩阵的效果是一样的。因为最后求最小二乘解是（A^T*A)^-1 * A^T * b
        A += tmp_A.transpose() * tmp_A;
        b += tmp_A.transpose() * tmp_b;

    }
    //; 求解（A^T*A) * x = A^T * b
    delta_bg = A.ldlt().solve(b);
    ROS_WARN_STREAM("gyroscope bias initial calibration " << delta_bg.transpose());
    // 滑窗中的零偏设置为求解出来的零偏
    for (int i = 0; i <= WINDOW_SIZE; i++)
        Bgs[i] += delta_bg;     //; 滑窗中的零偏都设置成同一个值，因为是利用滑窗中的所有数据求出来的平均值
    // 对all_image_frame中预积分量根据当前零偏重新积分
    for (frame_i = all_image_frame.begin(); next(frame_i) != all_image_frame.end( ); frame_i++)
    {
        frame_j = next(frame_i);
        //; 重新计算预积分，就是按照公式严格重新计算一遍，不是使用一阶泰勒展开近似更新
        //; 因为估计出来的bias可能比较大，可能不符合一阶展开的条件了
        frame_j->second.pre_integration->repropagate(Vector3d::Zero(), Bgs[0]);
    }
}


MatrixXd TangentBasis(Vector3d &g0)
{
    Vector3d b, c;
    Vector3d a = g0.normalized();
    Vector3d tmp(0, 0, 1);
    if(a == tmp)
        tmp << 1, 0, 0;
    b = (tmp - a * (a.transpose() * tmp)).normalized();
    c = a.cross(b);
    MatrixXd bc(3, 2);
    bc.block<3, 1>(0, 0) = b;
    bc.block<3, 1>(0, 1) = c;
    return bc;
}

/**
 * @brief 得到了一个初始的重力向量，引入重力大小作为先验，再进行几次迭代优化，求解最终的变量
 * 
 * @param[in] all_image_frame 
 * @param[in] g 
 * @param[in] x 
 */
void RefineGravity(map<double, ImageFrame> &all_image_frame, Vector3d &g, VectorXd &x)
{
    // 参考论文
    Vector3d g0 = g.normalized() * G.norm();    //; 以初始计算得到的g为方向，模长强制设为9.8
    Vector3d lx, ly;    //; 其实这两个量没用到，后面直接把b1和b2拼成一个3x2的矩阵lxly了
    //VectorXd x;
    int all_frame_count = all_image_frame.size();
    int n_state = all_frame_count * 3 + 2 + 1;  //; 重力变成2维，因此维度相比原来减小1

    MatrixXd A{n_state, n_state};
    A.setZero();
    VectorXd b{n_state};
    b.setZero();

    map<double, ImageFrame>::iterator frame_i;
    map<double, ImageFrame>::iterator frame_j;
    //; 优化4次重力，即迭代求解4次
    for(int k = 0; k < 4; k++)
    {
        MatrixXd lxly(3, 2);
        lxly = TangentBasis(g0);
        int i = 0;
        for (frame_i = all_image_frame.begin(); next(frame_i) != all_image_frame.end(); frame_i++, i++)
        {
            frame_j = next(frame_i);
            
            //; 重力减小1维，因此从6x10变成6x9
            MatrixXd tmp_A(6, 9);
            tmp_A.setZero();
            VectorXd tmp_b(6);
            tmp_b.setZero();

            double dt = frame_j->second.pre_integration->sum_dt;


            tmp_A.block<3, 3>(0, 0) = -dt * Matrix3d::Identity();
            tmp_A.block<3, 2>(0, 6) = frame_i->second.R.transpose() * dt * dt / 2 * Matrix3d::Identity() * lxly;
            tmp_A.block<3, 1>(0, 8) = frame_i->second.R.transpose() * (frame_j->second.T - frame_i->second.T) / 100.0;     
            tmp_b.block<3, 1>(0, 0) = frame_j->second.pre_integration->delta_p + frame_i->second.R.transpose() * frame_j->second.R * TIC[0] - TIC[0] - frame_i->second.R.transpose() * dt * dt / 2 * g0;

            tmp_A.block<3, 3>(3, 0) = -Matrix3d::Identity();
            tmp_A.block<3, 3>(3, 3) = frame_i->second.R.transpose() * frame_j->second.R;
            tmp_A.block<3, 2>(3, 6) = frame_i->second.R.transpose() * dt * Matrix3d::Identity() * lxly;
            tmp_b.block<3, 1>(3, 0) = frame_j->second.pre_integration->delta_v - frame_i->second.R.transpose() * dt * Matrix3d::Identity() * g0;


            Matrix<double, 6, 6> cov_inv = Matrix<double, 6, 6>::Zero();
            //cov.block<6, 6>(0, 0) = IMU_cov[i + 1];
            //MatrixXd cov_inv = cov.inverse();
            cov_inv.setIdentity();

            MatrixXd r_A = tmp_A.transpose() * cov_inv * tmp_A;
            VectorXd r_b = tmp_A.transpose() * cov_inv * tmp_b;

            A.block<6, 6>(i * 3, i * 3) += r_A.topLeftCorner<6, 6>();
            b.segment<6>(i * 3) += r_b.head<6>();

            A.bottomRightCorner<3, 3>() += r_A.bottomRightCorner<3, 3>();
            b.tail<3>() += r_b.tail<3>();
            
            //; 同理这里从6x4和4x6变成6x3和3x6了，对应的索引位置也从变成n_state-4变成n_state-3了
            A.block<6, 3>(i * 3, n_state - 3) += r_A.topRightCorner<6, 3>();
            A.block<3, 6>(n_state - 3, i * 3) += r_A.bottomLeftCorner<3, 6>();
        }
            A = A * 1000.0;
            b = b * 1000.0;
            x = A.ldlt().solve(b);
            //; 求出来的dg就是重力沿着切平面的变化
            VectorXd dg = x.segment<2>(n_state - 3);
            g0 = (g0 + lxly * dg).normalized() * G.norm();
            //double s = x(n_state - 1);
    }   
    g = g0;
}

/**
 * @brief 求解各帧的速度，枢纽帧的重力方向，以及尺度
 * 
 * @param[in] all_image_frame 
 * @param[in] g 
 * @param[in] x 
 * @return true 
 * @return false 
 */
bool LinearAlignment(map<double, ImageFrame> &all_image_frame, Vector3d &g, VectorXd &x)
{
    // 这一部分内容对照论文进行理解
    int all_frame_count = all_image_frame.size();
    //; all_frame_count * 3是滑窗中所有帧的速度的维度，3是g_c0的维度，1是s的维度
    int n_state = all_frame_count * 3 + 3 + 1;

    //; 最终拼成的大矩阵A
    MatrixXd A{n_state, n_state};
    A.setZero();
    VectorXd b{n_state};
    b.setZero();

    map<double, ImageFrame>::iterator frame_i;
    map<double, ImageFrame>::iterator frame_j;
    int i = 0;
    for (frame_i = all_image_frame.begin(); next(frame_i) != all_image_frame.end(); frame_i++, i++)
    {
        frame_j = next(frame_i);

        //; 对相邻的两帧构造小的A和b,然后累加摞起来得到大的A和b
        MatrixXd tmp_A(6, 10);
        tmp_A.setZero();
        VectorXd tmp_b(6);
        tmp_b.setZero();

        double dt = frame_j->second.pre_integration->sum_dt;

        tmp_A.block<3, 3>(0, 0) = -dt * Matrix3d::Identity();
        //; 从这里可以看出来，确实c0是枢纽帧，而不是初始的第0帧
        tmp_A.block<3, 3>(0, 6) = frame_i->second.R.transpose() * dt * dt / 2 * Matrix3d::Identity();
        //; 注意这里/100是为了放大这部分的系数，因为平移部分可能很小，这样求解的时候误差会很大
        tmp_A.block<3, 1>(0, 9) = frame_i->second.R.transpose() * (frame_j->second.T - frame_i->second.T) / 100.0;     
        tmp_b.block<3, 1>(0, 0) = frame_j->second.pre_integration->delta_p + frame_i->second.R.transpose() * frame_j->second.R * TIC[0] - TIC[0];
        //cout << "delta_p   " << frame_j->second.pre_integration->delta_p.transpose() << endl;
        tmp_A.block<3, 3>(3, 0) = -Matrix3d::Identity();
        tmp_A.block<3, 3>(3, 3) = frame_i->second.R.transpose() * frame_j->second.R;
        tmp_A.block<3, 3>(3, 6) = frame_i->second.R.transpose() * dt * Matrix3d::Identity();
        tmp_b.block<3, 1>(3, 0) = frame_j->second.pre_integration->delta_v;
        //cout << "delta_v   " << frame_j->second.pre_integration->delta_v.transpose() << endl;

        Matrix<double, 6, 6> cov_inv = Matrix<double, 6, 6>::Zero();
        //cov.block<6, 6>(0, 0) = IMU_cov[i + 1];
        //MatrixXd cov_inv = cov.inverse();
        //! 问题： 这里还有协方差矩阵，但是设置成I就没有用了？为什么？
        cov_inv.setIdentity();  

        MatrixXd r_A = tmp_A.transpose() * cov_inv * tmp_A;
        VectorXd r_b = tmp_A.transpose() * cov_inv * tmp_b;

        //; 把小矩阵累加起来得到大的矩阵，这样可以求解所有的待估计状态变量
        //; 注意这里位置索引是*3而不是*6，因为每次构建的小A、小b都是相邻k和k+1两帧的，
        //; 计算下一次的时候，变成k+1和k+2这两帧，也就是中间的k+1帧是会重复累加的。
        //; 所以v_k和v_k+1的系数矩阵部分会有一半的重叠
        A.block<6, 6>(i * 3, i * 3) += r_A.topLeftCorner<6, 6>();
        b.segment<6>(i * 3) += r_b.head<6>();

        A.bottomRightCorner<4, 4>() += r_A.bottomRightCorner<4, 4>();
        b.tail<4>() += r_b.tail<4>();

        //; 而6x4大小的部分，列在最后4列； 4x6大小的部分，行在最后4行
        A.block<6, 4>(i * 3, n_state - 4) += r_A.topRightCorner<6, 4>();
        A.block<4, 6>(n_state - 4, i * 3) += r_A.bottomLeftCorner<4, 6>();
    }
    //; 增强数值稳定性：为什么？因为如果A和b都很小的话，那么光计算的舍去误差就会导致结果的精度降低很多了
    //; 所以放大系数矩阵可以通过降低舍去误差来提高计算精度
    A = A * 1000.0;
    b = b * 1000.0;
    x = A.ldlt().solve(b);
    //; n_state是x的维度，由于数组索引从0开始，所以最后一维的索引是n_state-1
    //; 提问：另外这里还/100, 为什么不是*100?
    //; 解答：假设系数矩阵A和s有关的部分是A(s)，b向量和s有关的部分是b(s)，则原式是 A(s) * s = b(s)
    //;      现在把A(s)/100，，则原来的公式变成 A(s)/100 * s' = b(s)；
    //;      对比上下两个式子可以发现， s' / 100 = s，即缩放系数矩阵后求出来的尺度s'再/100之后才是真正的s
    double s = x(n_state - 1) / 100.0;
    ROS_DEBUG("estimated scale: %f", s);
    g = x.segment<3>(n_state - 4);  //; 注意这个g是g_c0，即g_w在枢纽帧坐标系下的表示
    ROS_DEBUG_STREAM(" result g     " << g.norm() << " " << g.transpose());
    // 做一些检查
    if(fabs(g.norm() - G.norm()) > 1.0 || s < 0)
    {
        return false;
    }
    //; 重力修复，利用重力只有两个自由度重新进行状态变量的估计
    //; 上面的状态估计相当于只是给重力提供了一个初始的方向
    RefineGravity(all_image_frame, g, x);  //; 注意这里重力再优化后x维度减小了1
    // 得到真实尺度
    s = (x.tail<1>())(0) / 100.0;
    (x.tail<1>())(0) = s;   //; 因为最终程序返回的是x，所以要把尺度s更新到x中
    ROS_DEBUG_STREAM(" refine     " << g.norm() << " " << g.transpose());
    if(s < 0.0 )
        return false;   
    else
        return true;
}

/**
 * @brief 
 * 
 * @param[in] all_image_frame 每帧的位姿和对应的预积分量
 * @param[out] Bgs 陀螺仪零偏
 * @param[out] g 重力向量在枢纽帧下的表示g_c0
 * @param[out] x 各帧的速度在各自坐标系下的表示，重力向量再优化的更新增量，尺度s
 * @return true 
 * @return false 
 */

bool VisualIMUAlignment(map<double, ImageFrame> &all_image_frame, Vector3d* Bgs, Vector3d &g, VectorXd &x)
{
    //; 1. 进行陀螺仪的零偏估计，估计结果输出到estimator的成员变量Bgs中。计算得到Bgs后会重新计算所有帧的预积分
    //; 2. 注意这一步没有判断是否成功，因为一般来说都会成功，IMU的陀螺仪是比较准确的
    //; 3. 这里可以看到滑窗中的普通帧的应用，就是在这里求陀螺仪零偏的时候，每两个相邻的普通帧都计算图像的旋转和IMU的旋转，从而估计零偏
    solveGyroscopeBias(all_image_frame, Bgs);

    if(LinearAlignment(all_image_frame, g, x))
        return true;
    else 
        return false;
}
