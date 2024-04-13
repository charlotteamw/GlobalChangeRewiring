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
    K1 = 1.0
    r2 = 1.5
    K2 = 1.22
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

λ1_stability(M) = maximum(real.(eigvals(M)))


# Function to calculate the Jacobian matrix at the initial condition
function jacobian_at_u0(K2_val, K1_base, u0, p)
    p.K2 = K2_val  # Update the K2 value
    p.K1 = K1_base + (K1_base - K2_val)  # Update K1 value in proportion to the change in K2
    return cmat(u0, p)  # Compute the Jacobian matrix at u0
end

function plot_max_real_eigenvalues(K2_range, K1_base, u0, p)
    max_real_eigenvalues = Float64[]

    for K2_val in K2_range
        jac = jacobian_at_u0(K2_val, K1_base, u0, p)
        push!(max_real_eigenvalues, λ1_stability(jac))
    end

    stability_plot = figure()
    PyPlot.plot(K2_range, max_real_eigenvalues, color="black", linewidth=3)
    xlabel("K1 decreasing/K2 increasing")
    ylabel("Re(λ) max")

    # Invert the y-axis to have more negative values increase along the y-axis
    PyPlot.gca().invert_yaxis()

    return stability_plot
end
# Example usage
K2_range = 1.15:0.0001:1.67  # Define the range of K2 values
K1_base = 1.2  # Define the base value for K1
u0 = [0.8, 0.4, 0.8, 0.4, 0.3]
p = ModelPar()  # Assuming the parameter structure is defined
plot_max_real_eigenvalues(K2_range, K1_base, u0, p)


function plot_equilibrium_densities(K2_range, K1_base, u0, p)
    equilibrium_P = Float64[]
    equilibrium_C1 = Float64[]
    equilibrium_C2 = Float64[]
    equilibrium_R1 = Float64[]
    equilibrium_R2 = Float64[]

    for K2_val in K2_range
        p.K2 = K2_val
        p.K1 = K1_base + (K1_base - K2_val)
        eq = find_eq(u0, p)

        push!(equilibrium_R1, eq[1])
        push!(equilibrium_C1, eq[2])
        push!(equilibrium_R2, eq[3])
        push!(equilibrium_C2, eq[4])
        push!(equilibrium_P, eq[5])
    end

    densities_plot = figure()
    #PyPlot.plot(K2_range, equilibrium_R1, label="Resource 1", color="lightblue", linewidth = 4)
    #PyPlot.plot(K2_range, equilibrium_R2, label="Resource 2", color="yellow", linewidth = 4)
    PyPlot.plot(K2_range, equilibrium_C1, label="Consumer 1", color="#00ABFF", linewidth = 4)
    #PyPlot.plot(K2_range, equilibrium_C2, label="Consumer 2", color="#FFCD00", linewidth = 4)
    PyPlot.plot(K2_range, equilibrium_P, label="Predator", color="black", linewidth = 4)


    xlabel("K1 decreasing, K2 increasing")
    ylabel("Equilibrium Densities")

    return densities_plot
end



function plot_consumption(K2_range, K1_base, u0, p)
    consumption_C1 = Float64[]
    consumption_C2 = Float64[]

    for K2_val in K2_range
        p.K2 = K2_val
        p.K1 = K1_base + (K1_base - K2_val)
        eq = find_eq(u0, p)

        Ω = p.pref(eq, p, 0.0)
        consumption_C1_val = (Ω * p.a_PC1 * eq[2] * eq[5] / (1 + Ω * p.a_PC1 * p.h_PC1 * eq[2] + (1 - Ω) * p.a_PC2 * p.h_PC2 * eq[4]))
        consumption_C2_val = ((1 - Ω) * p.a_PC2 * eq[4] * eq[5] / (1 + Ω * p.a_PC1 * p.h_PC1 * eq[2] + (1 - Ω) * p.a_PC2 * p.h_PC2 * eq[4]))

        push!(consumption_C1, consumption_C1_val)
        push!(consumption_C2, consumption_C2_val)
    end

    consumption_plot = figure()
    PyPlot.plot(K2_range, consumption_C1, label="Consumption of C1", color="#00ABFF", linewidth = 4)
    PyPlot.plot(K2_range, consumption_C2, label="Consumption of C2", color="#FFCD00", linewidth = 4)

    xlabel("K1 decreasing, K2 icreasing")
    ylabel("Consumption")

    return consumption_plot
end


function plot_biomass_ratio(K2_range, K1_base, u0, p)
    biomass_r_PC1 = Float64[]

    for K2_val in K2_range
        p.K2 = K2_val
        p.K1 = K1_base + (K1_base - K2_val)
        eq = find_eq(u0, p)

        biomass_r_PC1_val = eq[5] / eq[2]

        push!(biomass_r_PC1, biomass_r_PC1_val)
    end

    biomass_r_plot = figure()
    PyPlot.plot(K2_range, biomass_r_PC1, label="Biomass Ratio (P:C1)", color="black", linewidth = 4)

    xlabel("K1 decreasing, K2 icreasing")
    ylabel("Biomass Ratio (P:C1)")

    return biomass_r_plot
    
end

function plot_productivity(K2_range, K1_base, u0, p)
    prod_S1 = Float64[]
    prod_S2 = Float64[]

    for K2_val in K2_range
        p.K2 = K2_val
        p.K1 = K1_base + (K1_base - K2_val)

        prod_S2_val = K2_val
        prod_S1_val = K1_base + (K1_base - K2_val)

        push!(prod_S1, prod_S1_val)
        push!(prod_S2, prod_S2_val)
   
    end

    prod_plot = figure()
    PyPlot.plot(K2_range, prod_S1, label="Pelagic Productivity", color="#00ABFF", linewidth = 4)
    PyPlot.plot(K2_range, prod_S2, label="Littoral Productivity", color="#FFCD00", linewidth = 4)

    xlabel("K1 decreasing, K2 increasing")
    ylabel("Habitat Productivity")

    return prod_plot
end


# Plot equilibrium densities
plot_equilibrium_densities(K2_range, K1_base, u0, p)

# Plot consumption of C1 and C2 by P
plot_consumption(K2_range, K1_base, u0, p)

# Plot biomass ratio of P:C1
plot_biomass_ratio(K2_range, K1_base, u0, p)

# Plot biomass ratio of P:C1
plot_productivity(K2_range, K1_base, u0, p)

