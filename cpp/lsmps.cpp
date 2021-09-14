#include "taichi.h" //only used for gui
#include <iostream>
#include <cmath>
#include <vector>
#include <array>
#include <algorithm>

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
//template <int d>
//using Mat = Eigen::Matrix<real, d, d>;
template <int m, int n=m>
using Mat = Eigen::Matrix<real, m, n>;
#define PI 3.141592653589793238463

// Constants
const int Nx = 20, N = Nx * Nx; // free particles
const int Nw = 30, Nb = 4 * Nw; // boundary particles
const int grid_res = 10, window_size = 800;
const int bucket_size = 64;
const real dx = 1.0 / grid_res, dx_inv = (real)grid_res;
const real R = 0.5; //init rect side length
const Vec2i nb_dirs[9] = {{0, 0}, {1, 0}, {1, 1}, {0, 1}, {-1, 1}, {-1, 0}, {-1, -1}, {0, -1}, {1, -1}};

//parameters
real l0 = sqrt(R * R / N);
real dt = 1e-2;
Vec2 grav{0, -9.8};
real re = 3.1 * l0;
real rs = re / 2;
real alpha = 1e-2; // particle shifting
real eps = 1e-3;
real n0; //reference particle density
Eigen::DiagonalMatrix<real, 5> Hrs;
Eigen::DiagonalMatrix<real, 5> H1_inv;

// Data
std::array<Vec2, N> X, V, V_a, V_star;
std::array<real, N> Pr;
std::array<char, N> Lable; // 0=free, 1=near boundary
Vec2 domain_min = Vec2(0.0, 0.0), domain_max = Vec2(1.0, 1.0);
std::array<Mat<5>, N> M;
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

inline real Wa(real r)
{
    real ret = 0.0;
    if (r < re)
        ret = re / r - 1;
    return ret;
};

inline real Wup(real r, real costheta)
{
    return W(r) * std::max(2*costheta*costheta-1, eps);
}

inline Vec<5> P(const Vec2 &r, real _rs)
{
    Vec<5> ret;
    ret << r(0) / _rs, r(1) / _rs, r(0) * r(0) / _rs, r(0) * r(1) / _rs, r(1) * r(1) / _rs;
    return ret;
}

inline bool valid_cell(const Vec2i &c)
{
    return (c[0] >= 0) && (c[0] < grid_res) && (c[1] >= 0) && (c[1] < grid_res);
}

void reset_hashing()
{
    std::fill(hashing.begin(), hashing.end(), Bucket{});
    for (int i = 0; i < N; ++i)
    {
        Vec2i coord = (X[i] * dx_inv).cast<int>();
        if (!valid_cell(coord))
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

void compute_M()
{
    for (int i = 0; i < N; ++i)
    {
        M[i] = Mat<5>::Zero();
        Vec2i coord = (X[i] * dx_inv).cast<int>();
        for (int d = 0; d < 9; ++d)
        {
            Vec2i nb = coord + nb_dirs[d];
            if (!valid_cell(nb))
                continue;
            Bucket &bucket = hashing[nb[0] * grid_res + nb[1]];
            for (int p = 1; p <= bucket[0]; ++p)
            {
                int j = bucket[p];
                if(i==j) continue;
                Vec<5> pij = P(X[j] - X[i], rs);
                M[i] += W((X[j] - X[i]).norm()) * pij * pij.transpose();
            }
        }
    }
}

void init()
{
    Hrs.diagonal() << 1 / rs, 1 / rs, 2 / (rs * rs), 1 / (rs * rs), 2 / (rs * rs);
    H1_inv.diagonal() << 1, 1, 0.5, 1, 0.5;
    //free particles
    std::fill(V.begin(), V.end(), Vec2::Zero());
    real tdx = R / Nx;
    n0 = 0.0;
    Vec2 ref_X = Vec2(0.5 - R / 2 + tdx * (Nx / 2), 0.5 - R / 2 + tdx * (Nx / 2));
    for (int i = 0; i < Nx; ++i)
    {
        for (int j = 0; j < Nx; ++j)
        {
            X[i * Nx + j] = Vec2(0.5 - R / 2 + tdx * i, 0.5 - R / 2 + tdx * j);
            if ((i != Nx / 2) && (j != Nx / 2))
                n0 += Wa((X[i * Nx + j] - ref_X).norm());
        }
    }

    //wall boundary
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

void particle_shifting()
{
    for (int i = 0; i < N; ++i)
    {
        Vec2 delta = Vec2::Zero();
        Vec2i coord = (X[i] * dx_inv).cast<int>();
        for (int d = 0; d < 9; ++d)
        {
            Vec2i nb = coord + nb_dirs[d];
            if (!valid_cell(nb))
                continue;
            Bucket &bucket = hashing[nb[0] * grid_res + nb[1]];
            for (int p = 1; p <= bucket[0]; ++p)
            {
                int j = bucket[p];
                real rij = (X[j] - X[i]).norm();
                if (j != i)
                    delta -= alpha * l0 * l0 * 2 / n0 * Wa(rij) * (X[j] - X[i]) / (rij * rij);
            }
        }
        V_a[i] = V[i] + delta / dt;
    }
}

void advection_and_force()
{
    for (int i = 0; i < N; ++i)
    {
        V_star[i] = V[i] /*+ dt * grav*/;
        Mat<5> m_inv = M[i].inverse();
        Mat<2, 5> v_grad = Mat<2, 5>::Zero();
        Vec2i coord = (X[i] * dx_inv).cast<int>();
        for (int d = 0; d < 9; ++d)
        {
            Vec2i nb = coord + nb_dirs[d];
            if (!valid_cell(nb))
                continue;
            Bucket &bucket = hashing[nb[0] * grid_res + nb[1]];
            for (int p = 1; p <= bucket[0]; ++p)
            {
                int j = bucket[p];
                if(i==j) continue;
                Vec2 rij = X[j]-X[i];
                real costheta = rij.normalized().dot((V_a[i]-V[i]).normalized());
                v_grad += rij * (Hrs * m_inv * Wup(rij.norm(), costheta) * P(rij, rs)).transpose();
            }
        }
        V_star[i] += v_grad * H1_inv * P(dt*(V_a[i]-V[i]), 1);
    }
}

void update_position()
{
    for (int i = 0; i < N; ++i)
    {
        X[i] += V_a[i] * dt;
    }
}

void solve_pressure()
{
    //TODO: solve pressure
}

void projection()
{
    for (int i = 0; i < N; ++i)
    {
        V[i] = V_star[i];
        //TODO: pressure gradient
    }
}

void advance()
{
    reset_hashing();
    particle_shifting();
    compute_M();
    advection_and_force();
    update_position();
    solve_pressure();
    projection();
}

int main()
{
    init();
    std::cout << "re: " << re << " dx: " << dx << std::endl;
    taichi::GUI gui("LSMPS", window_size, window_size);
    auto &canvas = gui.get_canvas();
    while (true)
    {
        advance();
        canvas.clear(0x112F41);
        //canvas.rect(taichi::Vector2(0.04), taichi::Vector2(0.96)).radius(2).color(0x4FB99F).close();
        for (int i = 0; i < N; ++i)
            canvas.circle(X[i][0], X[i][1]).radius(2).color(0xED553B);
        for (int i = 0; i < Nb; ++i)
            canvas.circle(X_b[i][0], X_b[i][1]).radius(2).color(0x068587);
        gui.update();
    }
}