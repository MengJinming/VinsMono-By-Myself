#include "utility.h"

Eigen::Matrix3d Utility::g2R(const Eigen::Vector3d &g)
{
    Eigen::Matrix3d R0;
    Eigen::Vector3d ng1 = g.normalized();   //; 枢纽帧下的重力归一化，得到重力方向
    Eigen::Vector3d ng2{0, 0, 1.0};   //; 世界坐标下的重力方向
    //; 从两个向量计算它们之间的旋转矩阵，原理是角轴。这个得到的应该是枢纽帧重力 到 世界坐标重力的旋转？
    //; 也就是枢纽帧到世界坐标系之间的旋转 R_w_c0
    R0 = Eigen::Quaterniond::FromTwoVectors(ng1, ng2).toRotationMatrix();
    double yaw = Utility::R2ypr(R0).x();   //; 得到这个旋转的yaw角度
    //; R(-yaw) * [R(yaw)R(pitch)R(roll)] = R(pitch)R(roll), 即补偿掉了枢纽帧到世界坐标的yaw角旋转
    R0 = Utility::ypr2R(Eigen::Vector3d{-yaw, 0, 0}) * R0;
    // R0 = Utility::ypr2R(Eigen::Vector3d{-90, 0, 0}) * R0;
    return R0;
}
