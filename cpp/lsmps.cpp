#include "taichi.h" //only used for gui
#include <iostream>
#include <cmath>
#include <vector>
#include <array>

#include <Eigen/Sparse>
#include <Eigen/Dense>

// Math Type
using real = double;
using Vec2 = Eigen::Matrix<real, 2, 1>;
using Vec2i = Eigen::Vector2i;
using Mat2 = Eigen::Matrix<real, 2, 2>;
using VecX = Eigen::Matrix<real, -1, 1>;
using MatX = Eigen::Matrix<real, -1, -1>;
using SpVecd = Eigen::SparseVector<double>;
using SpMatd = Eigen::SparseMatrix<double>;
using Tripletd = Eigen::Triplet<double>;
template <int d>
using Vec = Eigen::Matrix<real, d, 1>;
template <int d>
using Mat = Eigen::Matrix<real, d, d>;
#define PI 3.141592653589793238463
// Constants
const int Nx = 20, N = Nx * Nx; // free particles
const int Nw = 30, Nb = 4 * Nw; // boundary particles
const int grid_res = 20, window_size = 800;
const int bucket_size = 64;
const real dx = 1.0 / grid_res, dx_inv = (real)grid_res;
const real R = 0.5;
const real l0 = sqrt(R * R / N);
const real re = 3.1 * l0;
const real rs = re / 2;
// Data
std::array<Vec2, N> X, V;
Vec2 domain_min = Vec2(0.0, 0.0), domain_max = Vec2(1.0, 1.0);
std::array<Mat<5>, N> M_inv;
using Bucket = std::array<int, bucket_size>;
std::array<Bucket, grid_res * grid_res> hashing;
std::array<Vec2, Nb> X_b, N_b;

inline real W(real r)
{
    real ret = 0.0;
    if (r < re)
        ret = pow(1 - r / re, 2);
    return ret;
}

inline Vec<5> P(const Vec2 &r)
{
    Vec<5> ret;
    ret << r(0) / rs, r(1) / rs, r(0) * r(0) / rs, r(0) * r(1) / rs, r(1) * r(1) / rs;
    return ret;
}

void reset_hashing()
{
    std::fill(hashing.begin(), hashing.end(), Bucket{});
    for (int i = 0; i < N; ++i)
    {
        Vec2i coord = (X[i] * dx_inv).cast<int>();
        if (coord[0] < 0 || coord[0] >= grid_res || coord[1] < 0 || coord[1] >= grid_res)
        {
            std::cerr << "[Error]out of box" << std::endl;
            exit(1);
        }
        Bucket &bucket = hashing[coord[0] * grid_res + coord[1]];
        if (bucket[0] >= bucket_size - 1)
        {
            std::cerr << "[Error]bucket full" << std::endl;
            exit(1);
        }
        ++bucket[0];
        bucket[bucket[0]] = i;
    }
}

void init()
{
    std::fill(V.begin(), V.end(), Vec2::Zero());
    real tdx = R / Nx;
    for (int i = 0; i < Nx; ++i)
    {
        for (int j = 0; j < Nx; ++j)
        {
            X[i * Nx + j] = Vec2(0.5 - R / 2 + tdx * i, 0.5 - R / 2 + tdx * j);
        }
    }
    real tdw = 0.9 / Nw;
    Vec2 norms[4] = {Vec2(0.0, 1.0), Vec2(-1.0, 0.0), Vec2(0.0, -1.0), Vec2(1.0, 0.0)};
    Vec2 dirs[4] = {Vec2(1.0, 0.0), Vec2(0.0, 1.0), Vec2(-1.0, 0.0), Vec2(0.0, -1.0)};
    Vec2 offsets[4] = {Vec2(0.05 + tdw / 2, 0.05), Vec2(0.95, 0.05 + tdw / 2), Vec2(0.95 - tdw / 2, 0.95), Vec2(0.05, 0.95 - tdw / 2)};
    for (int i = 0; i < Nw; ++i)
    {
        for (int j = 0; j < 4; ++j)
        {
            N_b[j * Nw + i] = norms[j];
            X_b[j * Nw + i] = offsets[j] + i * tdw * dirs[j];
        }
    }
}

int main()
{
    init();
    reset_hashing();
    taichi::GUI gui("LSMPS", window_size, window_size);
    auto &canvas = gui.get_canvas();
    while (true)
    {
        canvas.clear(0x112F41);
        //canvas.rect(taichi::Vector2(0.04), taichi::Vector2(0.96)).radius(2).color(0x4FB99F).close();
        for (int i = 0; i < N; ++i)
            canvas.circle(X[i][0], X[i][1]).radius(2).color(0xED553B);
        for (int i = 0; i < Nb; ++i)
            canvas.circle(X_b[i][0], X_b[i][1]).radius(2).color(0x068587);
        gui.update();
    }
}