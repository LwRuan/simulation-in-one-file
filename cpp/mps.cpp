#include "taichi.h" //only used for gui
#include <iostream>
#include <cmath>
#include <vector>
#include <array>
#include <algorithm>

#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <unsupported/Eigen/SparseExtra>
#include <unsupported/Eigen/IterativeSolvers>

// Math Type
using real = double;
using Vec2 = Eigen::Matrix<real, 2, 1>;
using Vec2i = Eigen::Vector2i;
using Mat2 = Eigen::Matrix<real, 2, 2>;
using SpMat = Eigen::SparseMatrix<real>;
using Triplet = Eigen::Triplet<real>;
template <int d>
using Vec = Eigen::Matrix<real, d, 1>;
template <int m, int n = m>
using Mat = Eigen::Matrix<real, m, n>;
#define PI 3.141592653589793238463

// Constants
// const int Nx = 20, N = Nx * Nx; // free particles
// const int Nw = 30, Nb = 4 * Nw; // boundary particles
// const int grid_res = 10, window_size = 800;
const int Nx = 40, N = Nx * Nx; // free particles
const int Nw = 60, Nb = 4 * Nw; // boundary particles
const int grid_res = 20, window_size = 800;
const int bucket_size = 64;
const real dx = 1.0 / grid_res, dx_inv = (real)grid_res;
const real R = 0.5; //init rect side length
const Vec2i nb_dirs[9] = {{0, 0}, {1, 0}, {1, 1}, {0, 1}, {-1, 1}, {-1, 0}, {-1, -1}, {0, -1}, {1, -1}};
const int colors[4] = {0xED553B, 0xF2B134, 0x068587, 0x858706};
const int N_screen = 360;

//parameters
real l0 = sqrt(R * R / N);
real dt = 1e-2;
Vec2 grav{0, -1.0};
real re = 3.1 * l0;
real n0; //reference particle density
real lambda0;
real gamma = 0.2;
real rho = 1.0;
Vec2 bound_min = Vec2(0.05, 0.05), bound_max = Vec2(0.95, 0.95);

// Data
int current_frame;
std::array<Vec2, N> X, V;
std::array<real, N> N_d;   //number density
std::array<char, N> Label; // 0=free, 1=near wall boundary, 2=A, 3=B
using Bucket = std::array<int, bucket_size>;
std::array<Bucket, grid_res * grid_res> hashing;
std::array<Vec2, Nb> X_b, N_b;
std::vector<Triplet> triplets;
SpMat Laplacian;
Vec<N> rhs, Pdt;

inline real Wa(real r)
{
    // real ret = 0.0;
    // if (r < re)
    //     ret = re / r - 1;
    // return ret;
    real ret = 0.0;
    if (r < re)
        ret = pow(1 - r / re, 2);
    return ret;
};

inline bool valid_cell(const Vec2i &c)
{
    return (c[0] >= 0) && (c[0] < grid_res) && (c[1] >= 0) && (c[1] < grid_res);
}

inline bool near_boundary(const Vec2 &r)
{
    return (r[0] < bound_min[0] + re) || (r[0] > bound_max[0] - re) ||
           (r[1] < bound_min[1] + re) || (r[1] > bound_max[1] - re);
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

void update_label()
{
    std::fill(Label.begin(), Label.end(), 0);
    for (int i = 0; i < N; ++i)
    {
        bool screen[N_screen];
        std::fill(screen, screen + N_screen, false);
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
                if (i == j)
                    continue;
                Vec2 dX = X[j] - X[i];
                if (dX.norm() > re)
                    continue;
                real theta_ij = 180.0 / PI * (std::atan2(dX[1], dX[0]) + PI);
                real dtheta_ij = 180.0 / PI * std::atan2(0.5 * l0, sqrt(dX.dot(dX) - 0.25 * l0 * l0));
                int lowb = static_cast<int>(theta_ij - dtheta_ij) % N_screen;
                int rangeb = static_cast<int>(2 * dtheta_ij);
                for (int k = 0; k < rangeb; ++k){
                    int idx = (lowb + k) % N_screen;
                    if(idx < 0) idx += N_screen;
                    screen[idx] = true;
                }
            }
        }
        int n_block = 0;
        for (int k = 0; k < N_screen; ++k)
        {
            n_block += static_cast<int>(screen[k]);
        }
        if (static_cast<double>(n_block) / N_screen < 5.0 / 6.0)
            Label[i] = 2;
    }
    for (int i = 0; i < N; ++i)
    {
        if (Label[i] == 2)
            continue;
        real ni = 0.0;
        bool has_A = false;
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
                if (i == j)
                    continue;
                Vec2 dX = X[j] - X[i];
                if (dX.norm() > re)
                    continue;
                ni += Wa(dX.norm());
                if (Label[j] == 2)
                    has_A = true;
            }
        }
        if (has_A && ni <= n0)
            Label[i] = 3;
    }
}

void compute_N_d()
{
    std::fill(N_d.begin(), N_d.end(), 0.0);
    for (int i = 0; i < N; ++i)
    {
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
                if (i == j)
                    continue;
                Vec2 dX = X[j] - X[i];
                N_d[i] += Wa(dX.norm());
            }
        }
    }
}

void solve_pressure()
{
    Laplacian.setZero();
    Laplacian.resize(N, N);
    rhs.setZero();
    triplets.clear();
    for (int i = 0; i < N; ++i)
    {
        real diag_coef = 0.0;
        real n_star = 0.0;
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
                if (i == j)
                    continue;
                Vec2 dX = X[j] - X[i];
                if (dX.norm() > re)
                    continue;
                real coef = 4 / lambda0 / n0 * Wa(dX.norm());
                n_star += Wa(dX.norm());
                triplets.push_back(Triplet(i, j, -coef));
                diag_coef += coef;
            }
        }
        rhs[i] = gamma * rho / dt * (n_star - std::min(N_d[i], n0)) / n0;
        if ((Label[i] == 2) || (Label[i] == 3))
        {
            diag_coef += 4 / lambda0 / n0 * std::max(n0 - n_star, 0.0);
        }
        triplets.push_back(Triplet(i, i, diag_coef));
    }
    Laplacian.setFromTriplets(triplets.begin(), triplets.end());
    Eigen::GMRES<SpMat> solver(Laplacian);
    Pdt = solver.solve(rhs);
    std::cout << "[iters: " << solver.iterations() << " error: " << solver.error() << "]" << std::endl;
}

void pre_update()
{
    for (int i = 0; i < N; ++i)
    {
        V[i] += dt * 1e-1 * (Vec2(0.5, 0.5) - X[i]);
        X[i] += dt * V[i];
    }
}

void post_update()
{
    for (int i = 0; i < N; ++i)
    {
        Vec2 grad_pdt = Vec2::Zero();
        Vec2i coord = (X[i] * dx_inv).cast<int>();
        real minp = std::min(Pdt[i], 0.0);
        for (int d = 0; d < 9; ++d)
        {
            Vec2i nb = coord + nb_dirs[d];
            if (!valid_cell(nb))
                continue;
            Bucket &bucket = hashing[nb[0] * grid_res + nb[1]];
            for (int p = 1; p <= bucket[0]; ++p)
            {
                int j = bucket[p];
                if (i == j)
                    continue;
                if ((X[j] - X[i]).norm() <= re)
                    minp = std::min(minp, Pdt[j]);
            }
        }
        for (int d = 0; d < 9; ++d)
        {
            Vec2i nb = coord + nb_dirs[d];
            if (!valid_cell(nb))
                continue;
            Bucket &bucket = hashing[nb[0] * grid_res + nb[1]];
            for (int p = 1; p <= bucket[0]; ++p)
            {
                int j = bucket[p];
                if (i == j)
                    continue;
                Vec2 dX = X[j] - X[i];
                grad_pdt += 2 / n0 * (Pdt[j] - minp) / dX.dot(dX) * dX * Wa(dX.norm());
            }
        }
        V[i] -= 1 / rho * grad_pdt;
        X[i] -= dt / rho * grad_pdt;
    }
}

void advance()
{
    pre_update();
    update_label();
    compute_N_d();
    solve_pressure();
    post_update();
}

void init()
{
    //free particles
    std::fill(V.begin(), V.end(), Vec2::Zero());
    real tdx = R / Nx;
    n0 = 0.0;
    lambda0 = 0.0;
    Vec2 ref_X = Vec2(0.5 - R / 2 + tdx * (Nx / 2), 0.5 - R / 2 + tdx * (Nx / 2));
    for (int i = 0; i < Nx; ++i)
    {
        for (int j = 0; j < Nx; ++j)
        {
            X[i * Nx + j] = Vec2(0.5 - R / 2 + tdx * i, 0.5 - R / 2 + tdx * j);
            if ((i != Nx / 2) || (j != Nx / 2))
                n0 += Wa((X[i * Nx + j] - ref_X).norm());
        }
    }
    for (int i = 0; i < Nx; ++i)
    {
        for (int j = 0; j < Nx; ++j)
        {
            X[i * Nx + j] = Vec2(0.5 - R / 2 + tdx * i, 0.5 - R / 2 + tdx * j);
            if ((i != Nx / 2) || (j != Nx / 2))
            {
                Vec2 dX = X[i * Nx + j] - ref_X;
                lambda0 += dX.dot(dX) * Wa(dX.norm());
            }
        }
    }
    lambda0 /= n0;

    //wall boundary
    real tdw = (bound_max[0] - bound_min[0]) / Nw;
    Vec2 norms[4] = {Vec2(0.0, 1.0), Vec2(-1.0, 0.0), Vec2(0.0, -1.0), Vec2(1.0, 0.0)};
    Vec2 dirs[4] = {Vec2(1.0, 0.0), Vec2(0.0, 1.0), Vec2(-1.0, 0.0), Vec2(0.0, -1.0)};
    Vec2 offsets[4] = {Vec2(bound_min[0] + tdw / 2, bound_min[1]), Vec2(bound_max[0], bound_min[1] + tdw / 2),
                       Vec2(bound_max[0] - tdw / 2, bound_max[1]), Vec2(bound_min[0], bound_max[1] - tdw / 2)};
    for (int i = 0; i < Nw; ++i)
    {
        for (int j = 0; j < 4; ++j)
        {
            N_b[j * Nw + i] = norms[j];
            X_b[j * Nw + i] = offsets[j] + i * tdw * dirs[j];
        }
    }
    reset_hashing();
    update_label();
}

int main()
{
    init();
    taichi::GUI gui("LSMPS", window_size, window_size);
    auto &canvas = gui.get_canvas();
    for (current_frame = 0;; ++current_frame)
    {
        advance();
        canvas.clear(0x112F41);
        //canvas.rect(taichi::Vector2(0.04), taichi::Vector2(0.96)).radius(2).color(0x4FB99F).close();
        for (int i = 0; i < N; ++i)
            canvas.circle(X[i][0], X[i][1]).radius(2).color(colors[Label[i]]);
        for (int i = 0; i < Nb; ++i)
            canvas.circle(X_b[i][0], X_b[i][1]).radius(2).color(0x99CDCD);
        gui.update();
    }
}
