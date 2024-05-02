#pragma once 
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <cstdlib>
#include <deque>
#include <map>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>
using namespace Eigen;
using namespace std;



struct SFMFeature
{
    bool state;		//; 这个3D点的状态，true表示已经被三角化过
    int id;			//; 这个3D点的id
    vector<pair<int,Vector2d>> observation;	//; 这个3D点被哪些关键帧看到，并且在这些关键帧中的观测坐标
    double position[3];	//; 这个3D点的位置
    double depth;		//; 这个3D点的深度
};

//; 实现自动求导，必须自己定义的一个结构体
struct ReprojectionError3D
{
	ReprojectionError3D(double observed_u, double observed_v)
		:observed_u(observed_u), observed_v(observed_v)
		{}

	template <typename T>
	//; 自动求导必须重新定义()运算符，并且这个()运算符必须是模板函数，在其中定义残差的计算方式
	//; 形参：相机旋转、相机平移、3d点、视觉观测2d点，其中前面的几个是待优化的变量，最后一个是残差
	bool operator()(const T* const camera_R, const T* const camera_T, const T* point, T* residuals) const
	{
		T p[3];
		ceres::QuaternionRotatePoint(camera_R, point, p);	// 旋转这个点
		p[0] += camera_T[0]; p[1] += camera_T[1]; p[2] += camera_T[2];	// 这其实就是Rcw * pw + tcw
		// 得到该相机坐标系下的3d坐标
		T xp = p[0] / p[2];
    	T yp = p[1] / p[2];	// 归一化处理
			// 跟现有观测形成残差
    	residuals[0] = xp - T(observed_u);
    	residuals[1] = yp - T(observed_v);
    	return true;
	}

	static ceres::CostFunction* Create(const double observed_x,
	                                   const double observed_y) 
	{
     //; 自动求导的CostFunction必须继承于AutoDiffCostFunction
	  return (new ceres::AutoDiffCostFunction<
	          //; 注意这里参数的维度2, 4, 3, 3和上面()运算符中的形参并不是一一对应的，而是把残差的维度放到了最前面
	          ReprojectionError3D, 2, 4, 3, 3>(
				//; 注意这里new了一个定义的对象
	          	new ReprojectionError3D(observed_x,observed_y)));
	}

	double observed_u;
	double observed_v;
};

class GlobalSFM
{
public:
	GlobalSFM();
	bool construct(int frame_num, Quaterniond* q, Vector3d* T, int l,
			  const Matrix3d relative_R, const Vector3d relative_T,
			  vector<SFMFeature> &sfm_f, map<int, Vector3d> &sfm_tracked_points);

private:
	bool solveFrameByPnP(Matrix3d &R_initial, Vector3d &P_initial, int i, vector<SFMFeature> &sfm_f);

	void triangulatePoint(Eigen::Matrix<double, 3, 4> &Pose0, Eigen::Matrix<double, 3, 4> &Pose1,
							Vector2d &point0, Vector2d &point1, Vector3d &point_3d);
	void triangulateTwoFrames(int frame0, Eigen::Matrix<double, 3, 4> &Pose0, 
							  int frame1, Eigen::Matrix<double, 3, 4> &Pose1,
							  vector<SFMFeature> &sfm_f);

	int feature_num;
};