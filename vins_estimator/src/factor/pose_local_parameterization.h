#pragma once

#include <eigen3/Eigen/Dense>
#include <ceres/ceres.h>
#include "../utility/utility.h"

//; 要想定义自己的局部参数块，则需要继承ceres::LocalParameterization类，并且重载基类中的下面四个成员函数
class PoseLocalParameterization : public ceres::LocalParameterization
{
    //; Plus就定义了加法，x是原来的参数，delta是参数变化量，x_plus_delta是更新后的结果
    virtual bool Plus(const double *x, const double *delta, double *x_plus_delta) const;
    //; 由于后端优化使用解析求导，所以这里计算雅克比的函数不重要，后面会有专门的地方计算雅克比
    //! 问题：所以说这里的计算雅克比的方式是给自动求导使用的？
    virtual bool ComputeJacobian(const double *x, double *jacobian) const;
    virtual int GlobalSize() const { return 7; };//; 局部参数块的维度，注意就是数组长度，所以这里是7
    virtual int LocalSize() const { return 6; }; //; 局部参数块的自由度，注意这个和物理意义相关，平移+旋转一共6自由度
};
