#include "taichi.h" //only used for gui
#include <iostream>
#include <cmath>
#include <vector>
#include <array>

#include <Eigen/Sparse>
#include <Eigen/Dense>

// Math Type
using real = double;
using Vec2f = Eigen::Vector2f;
using Vec2d = Eigen::Vector2d;
using Vec2 = Eigen::Matrix<real, 2, 1>;
using Vec2i = Eigen::Vector2i;
using Mat2f = Eigen::Matrix2f;
using Mat2d = Eigen::Matrix2d;
using Mat2 = Eigen::Matrix<real, 2, 2>;
using Mat2i = Eigen::Matrix2i;
using Vec3f = Eigen::Vector3f;
using Vec3d = Eigen::Vector3d;
using Vec3 = Eigen::Matrix<real, 3, 1>;
using Vec3i = Eigen::Vector3i;
using Mat3f = Eigen::Matrix3f;
using Mat3d = Eigen::Matrix3d;
using Mat3 = Eigen::Matrix<real, 3, 3>;
using Mat3i = Eigen::Matrix3i;
using Vec4f = Eigen::Vector4f;
using Vec4d = Eigen::Vector4d;
using Vec4 = Eigen::Matrix<real, 4, 1>;
using Vec4i = Eigen::Vector4i;
using Mat4f = Eigen::Matrix4f;
using Mat4d = Eigen::Matrix4d;
using Mat4 = Eigen::Matrix<real, 4, 4>;
using Mat4i = Eigen::Matrix4i;
using VecXf = Eigen::VectorXf;
using VecXd = Eigen::VectorXd;
using VecX = Eigen::Matrix<real, -1, 1>;
using VecXi = Eigen::VectorXi;
using MatXf = Eigen::MatrixXf;
using MatXd = Eigen::MatrixXd;
using MatX = Eigen::Matrix<real, -1, -1>;
using MatXi = Eigen::MatrixXi;
using SpVecf = Eigen::SparseVector<float>;
using SpVecd = Eigen::SparseVector<double>;
using SpVeci = Eigen::SparseVector<int>;
using SpMatf = Eigen::SparseMatrix<float>;
using SpMatd = Eigen::SparseMatrix<double>;
using SpMati = Eigen::SparseMatrix<int>;
using Tripletf = Eigen::Triplet<float>;
using Tripletd = Eigen::Triplet<double>;
using Tripleti = Eigen::Triplet<int>;
using Eigen::Quaterniond;
using Eigen::Quaternionf;
using Vec6f = Eigen::Matrix<float, 6, 1>;
using Vec6d = Eigen::Matrix<double, 6, 1>;
using Vec6i = Eigen::Matrix<int, 6, 1>;
using Vec6 = Eigen::Matrix<real, 6, 1>;
using Mat6f = Eigen::Matrix<float, 6, 6>;
using Mat6d = Eigen::Matrix<double, 6, 6>;
using Mat6 = Eigen::Matrix<real, 6, 6>;
using Mat6i = Eigen::Matrix<int, 6, 6>;
using Mat3x6f = Eigen::Matrix<float, 3, 6>;
using Mat3x6d = Eigen::Matrix<double, 3, 6>;
using Mat3x6 = Eigen::Matrix<real, 3, 6>;
using Mat3x6i = Eigen::Matrix<int, 3, 6>;
using Mat6x3f = Eigen::Matrix<float, 6, 3>;
using Mat6x3d = Eigen::Matrix<double, 6, 3>;
using Mat6x3 = Eigen::Matrix<real, 6, 3>;
using Mat6x3i = Eigen::Matrix<int, 6, 3>;
template <int d>
using Vec = Eigen::Matrix<real, d, 1>;
template <int d>
using Mat = Eigen::Matrix<real, d, d>;
#define PI 3.141592653589793238463
// Constants
const int N = 1000;
const real R = 1.0;
const real l0 = sqrt(PI * R * R / N);
const real re = 3.1 * l0;
const real rs = re / 2;
// Data
std::array<Vec2, N> X, V;
Vec2 domain_min = Vec2(0.0, 0.0);
Vec2 domain_max = Vec2(1.0, 1.0);

inline real W(real r)
{
    real ret = 0.0;
    if (r < re)
        ret = pow(1 - r / re, 2);
    return ret;
}

int main()
{
    taichi::GUI gui("Real-time 2D MLS-MPM", 800, 800);
    auto &canvas = gui.get_canvas();
    int f = 0;
    for (int i = 0;; i++)
    {                                                                   
            canvas.clear(0x112F41);                                              
            canvas.rect(taichi::Vector2(0.04), taichi::Vector2(0.96)).radius(2).color(0x4FB99F).close();
            gui.update();
    }
}