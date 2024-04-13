using Parameters: @with_kw, @unpack
using LinearAlgebra: eigvals
using ForwardDiff
using QuadGK: quadgk
using NLsolve
using DifferentialEquations
using PyPlot

function adapt_pref(u, p, t)
    return p.ω * u[2] / (p.ω * u[2] + (1 - p.ω) * u[4])
end

function fixed_pref(u, p, t)
    return p.Ω
end

@with_kw mutable struct ModelPar
    a_R1C1 = 0.9
    h_R1C1 = 0.6
    e_R1C1 = 0.7
    a_R2C2 = 0.9
    h_R2C2 = 0.6
    e_R2C2 = 0.7
    a_PC1 = 1.0
    h_PC1 = 0.6
    e_PC1 = 0.8
    a_PC2 = 0.9
    h_PC2 = 0.6
    e_PC2 = 0.8
    m_P = 0.3
    r1 = 1.5
    K1 = 1.2
    r2 = 1.5
    K2 = 1.
    m_C1 = 0.3
    m_C2 = 0.3
    pref::Function = fixed_pref
    Ω = 0.6
    ω = 0.6
end

function f_PC1(u, p, t)
    R1, C1, R2, C2, P = u
    Ω = p.pref(u, p, t)
    return Ω * p.a_PC1 * C1 * P / (1 + Ω * p.a_PC1 * p.h_PC1 * C1 + (1 - Ω) * p.a_PC2 * p.h_PC2 * C2)
end

function f_PC2(u, p, t)
    R1, C1, R2, C2, P = u
    Ω = p.pref(u, p, t)
    return Ω * p.a_PC2 * C2 * P / (1 + Ω * p.a_PC2 * p.h_PC2 * C2 + (1 - Ω) * p.a_PC1 * p.h_PC1 * C1)
end

degree_coupling(u, p) = f_PC1(u, p, 0.0) / (f_PC1(u, p, 0.0) + f_PC2(u, p, 0.0))

function model!(du, u, p, t)
    @unpack r1, K1, r2, K2 = p
    @unpack a_R1C1, h_R1C1, e_R1C1, a_R2C2, h_R2C2, e_R2C2, m_C1, m_C2 = p
    @unpack a_PC1, h_PC1, e_PC1, a_PC2, h_PC2, e_PC2, m_P, Ω, ω  = p
    R1, C1, R2, C2, P = u

    Ω = p.pref(u, p, t)

    int_R1C1 = a_R1C1 * R1 * C1 / (1 + a_R1C1 * h_R1C1 * R1)
    int_R2C2 = a_R2C2 * R2 * C2 / (1 + a_R2C2 * h_R2C2 * R2)
    denom_PC1C2 = 1 + Ω * a_PC1 * h_PC1 * C1 + (1 - Ω) * a_PC2 * h_PC2 * C2
    num_PC2 = (1 - Ω) * a_PC2 * C2 * P
    num_PC1 = Ω * a_PC1 * C1 * P

    du[1] = r1 * R1 * (1 - R1 / K1) - int_R1C1
    du[2] = e_R1C1 * int_R1C1 - (num_PC1/denom_PC1C2) - m_C1 * C1
    du[3] = r2 * R2 * (1 - R2 / K2) - int_R2C2
    du[4] = e_R2C2 * int_R2C2 - (num_PC2/denom_PC1C2) - m_C2 * C2
    du[5] = (e_PC1 * num_PC1 + e_PC2 * num_PC2) / denom_PC1C2 - m_P * P

    return du
end

#time series

let
    u0 = [0.8, 0.8, 0.4, 0.4, 0.3]
    t_span = (0, 2000.0)
    p = ModelPar()
    prob = ODEProblem(model!, u0, t_span, p)
    sol = solve(prob)
    model_ts = figure()
    for i in 1:5
        PyPlot.plot(sol.t, getindex.(sol.u, i), label = ["R1", "C1", "R2", "C2", "P"][i])
    end
    xlabel("Time")
    ylabel("Density")
    legend()
    return model_ts
end


function rhs(u, p)
    du = similar(u)
    model!(du, u, p, zero(u))
    return du
end

find_eq(u, p) = nlsolve((du, u) -> model!(du, u, p, zero(u)), u).zero
cmat(u, p) = ForwardDiff.jacobian(x -> rhs(x, p), u)
λ1_real(M) = maximum(real.(eigvals(M)))
λ1_imaginary(M) = maximum(abs.(imag.(eigvals(M))))  # Function to calculate maximum absolute imaginary part of eigenvalues

# Function to calculate the Jacobian matrix at the initial condition
function jacobian_at_u0(K2_val, u0, p)
    p.K2 = K2_val  # Update the K2 value
    return cmat(u0, p)  # Compute the Jacobian matrix at u0
end

# Function to plot the maximum real eigenvalues across a gradient of K2
function plot_max_eigenvalues(K2_range, u0, p)
    max_real_eig = Float64[]
    max_imaginary_eig = Float64[]

    for K2_val in K2_range
        jac = jacobian_at_u0(K2_val, u0, p)
        push!(max_real_eig, λ1_real(jac))
        push!(max_imaginary_eig, λ1_imaginary(jac))
    end

    stability_plot = figure()
    PyPlot.plot(K2_range, max_real_eig, label="Max Real Eigenvalue")
    PyPlot.plot(K2_range, max_imaginary_eig, label="Max Imaginary Eigenvalue", linestyle="--")
    xlabel("K2")
    ylabel("Eigenvalues")
    legend()
    return stability_plot
end

K2_range = 0.9:0.0001:1.2  # Define the range of K2 values
K1_base = 1.5  # Define the base value for K1
u0 = [0.8, 0.4, 0.8, 0.4, 0.3]
p = ModelPar()  # Assuming the parameter structure is defined
plot_max_eigenvalues(K2_range, u0, p)


#### 2nd and 3rd greatest eigs 
# Function to calculate the 2nd largest real part of eigenvalues
λ2_real(M) = sort(real.(eigvals(M)), rev=true)[2]

# Function to calculate the 2nd largest imaginary part of eigenvalues
λ2_imaginary(M) = sort(abs.(imag.(eigvals(M))), rev=true)[2]

# Function to calculate the 3rd largest real part of eigenvalues
λ3_real(M) = sort(real.(eigvals(M)), rev=true)[3]

# Function to calculate the 3rd largest imaginary part of eigenvalues
λ3_imaginary(M) = sort(abs.(imag.(eigvals(M))), rev=true)[3]


# Function to plot the 2nd largest real and imaginary parts of eigenvalues
function plot_2nd_eigenvalues(K2_range, u0, p)
    second_real_eig = Float64[]
    second_imaginary_eig = Float64[]

    for K2_val in K2_range
        jac = jacobian_at_u0(K2_val, u0, p)
        push!(second_real_eig, λ2_real(jac))
        push!(second_imaginary_eig, λ2_imaginary(jac))
    end

    plot = figure()
    PyPlot.plot(K2_range, second_real_eig, label="2nd Largest Real Eigenvalue")
    PyPlot.plot(K2_range, second_imaginary_eig, label="2nd Largest Imaginary Eigenvalue", linestyle="--")
    xlabel("K2")
    ylabel("Eigenvalues")
    legend()
    return plot
end

# Function to plot the 3rd largest real and imaginary parts of eigenvalues
function plot_3rd_eigenvalues(K2_range, u0, p)
    third_real_eig = Float64[]
    third_imaginary_eig = Float64[]

    for K2_val in K2_range
        jac = jacobian_at_u0(K2_val, u0, p)
        push!(third_real_eig, λ3_real(jac))
        push!(third_imaginary_eig, λ3_imaginary(jac))
    end

    plot = figure()
    PyPlot.plot(K2_range, third_real_eig, label="3rd Largest Real Eigenvalue")
    PyPlot.plot(K2_range, third_imaginary_eig, label="3rd Largest Imaginary Eigenvalue", linestyle="--")
    xlabel("K2")
    ylabel("Eigenvalues")
    legend()
    return plot
end

plot_2nd_eigenvalues(K2_range, u0, p)
plot_3rd_eigenvalues(K2_range, u0, p)


