#include "pose_local_parameterization.h"

bool PoseLocalParameterization::Plus(const double *x, const double *delta, double *x_plus_delta) const
{
    //; 注意Eigen::Map的作用就是实现内存的映射，把原来内存中的数组映射成Eigen的数据类型，这样就能利用Eigen的数据
    //; 类型进行数据运算了，运算的结果仍然是存在原来的内存里。这样的好处就是避免重新定义Eigen数据类型的局部变量进行中间计算
    Eigen::Map<const Eigen::Vector3d> _p(x);  //; 原先的位置，取x参数的前3维
    Eigen::Map<const Eigen::Quaterniond> _q(x + 3); //; 原先的旋转，取x参数的后4维

    Eigen::Map<const Eigen::Vector3d> dp(delta);  //; 位置更新量，前3维

    //; 注意这里！参数的更新量中旋转部分是在李代数上进行求导，得到的是一个旋转向量的增量，是3维的
    //; 所以下面就是把旋转向量的增量转成四元数，然后更新到原来的参数上
    //! 问题：为什么旋转增量部分是旋转向量？  解答：应该是其他表示方式都是过参数化的，而在李代数上求导不是过参数化的，
    //!      即得到旋转增量是旋转向量的方式不会过参数化
    Eigen::Quaterniond dq = Utility::deltaQ(Eigen::Map<const Eigen::Vector3d>(delta + 3));

    Eigen::Map<Eigen::Vector3d> p(x_plus_delta);
    Eigen::Map<Eigen::Quaterniond> q(x_plus_delta + 3);

    //; 参数更新
    p = _p + dp;
    q = (_q * dq).normalized();

    return true;
}
bool PoseLocalParameterization::ComputeJacobian(const double *x, double *jacobian) const
{
    Eigen::Map<Eigen::Matrix<double, 7, 6, Eigen::RowMajor>> j(jacobian);
    j.topRows<6>().setIdentity();
    j.bottomRows<1>().setZero();

    return true;
}
