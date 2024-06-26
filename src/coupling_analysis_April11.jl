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
    a_PC1 = 1.2
    h_PC1 = 0.6
    e_PC1 = 0.7
    a_PC2 = 1.0
    h_PC2 = 0.6
    e_PC2 = 0.7
    m_P = 0.3
    r1 = 2.0
    K1 = 0.853
    r2 = 2.0
    K2 = 1.4
    m_C1 = 0.3
    m_C2 = 0.3
    pref::Function = fixed_pref
    Ω = 0.36
    ω = 0.6
end

function f_PC1(u, p, t)
    R1, C1, R2, C2, P = u
    Ω = p.pref(u, p, t)
    return (Ω * p.a_PC1 * C1 * P) / (1 + ((Ω * p.a_PC1 * p.h_PC1 * C1) + ((1 - Ω) * p.a_PC2 * p.h_PC2 * C2)))
end

function f_PC2(u, p, t)
    R1, C1, R2, C2, P = u
    Ω = p.pref(u, p, t)
    return ((1-Ω) * p.a_PC2 * C2 * P) / (1 + ((Ω * p.a_PC1 * p.h_PC1 * C1) + ((1 - Ω) * p.a_PC2 * p.h_PC2 * C2)))
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

function model(u, ModelPar, t)
    du = similar(u)
    model!(du, u, ModelPar, t)
    return du
end

let
    u0 = [0.8, 0.4, 0.8, 0.4, 0.3]
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

## calculating the jacobian 
function jac(u, model, p)
        ForwardDiff.jacobian(u -> model(u, p, NaN), u)
end 
    
## calculate equilibrium densities - using grid instead of sol to remove transient 
K1_vals = 0.9:0.001:1.2
K1_hold = fill(0.0,length(K1_vals),6)

u0 = [0.8, 0.4, 0.8, 0.4, 0.3]
p = ModelPar()
t_span = (0, 10000.0)
ts = range(1000, 1500, length = 500)

for i=1:length(K1_vals)
    p = ModelPar(K1 = K1_vals[i])
    u0 = [0.8, 0.4, 0.8, 0.4, 0.3]
    prob = ODEProblem(model!, u0, t_span, p)
    sol = solve(prob, reltol = 1e-8, abstol = 1e-8)
    grid = sol(ts)
    eq = nlsolve((du, u) -> model!(du, u, p, 0.0), grid.u[end]).zero
    K1_hold[i,1] = K1_vals[i]
    K1_hold[i,2:end] = eq
    println(K1_hold[i,:])
end

## plot equilibrium densities 
using Plots

eq_R1 = Plots.plot(K1_hold[:,1], K1_hold[:,2], legend = false, lw= 2.0, colour = "black", xlabel = " K1 ", ylabel = " R1 Equilibrium Density " , xflip = true)

eq_C1 = Plots.plot(K1_hold[:,1], K1_hold[:,3], legend = false, lw= 2.0, colour = "black", xlabel = " K1 ", ylabel = " C1 Equilibrium Density " , xflip = true)

eq_R2 = Plots.plot(K1_hold[:,1], K1_hold[:,4], legend = false, lw= 2.0, linecolour = "darkorange", xlabel = " K1 ", ylabel = " R2 Equilibrium Density " , xflip = true)

eq_C2 = Plots.plot(K1_hold[:,1], K1_hold[:,5], legend = false, lw= 2.0, linecolour = "green", xlabel = " K1 ", ylabel = " C2 Equilibrium Density " , xflip = true)

eq_P = Plots.plot(K1_hold[:,1], K1_hold[:,6], legend = false, lw= 2.0, colour = "black", xlabel = " K1 ", ylabel = " P Equilibrium Density " , xflip = true)

## plot P:C biomass ratio
PC_biomass = Plots.plot(K1_hold[:,1], K1_hold[:,6] ./ K1_hold[:,3], legend = false, lw = 5.0, color = "black", xlabel = "K2", ylabel = "P:C1 Biomass Ratio", xflip = true, grid = false,
xguidefontsize = 18, yguidefontsize = 18)


## calculate all five real eigs 

eig_hold = fill(0.0,length(K1_vals),6)

for i=1:length(K1_vals)
    p = ModelPar(K1 = K1_vals[i])
    u0 = [0.8, 0.4, 0.8, 0.4, 0.3]
    prob = ODEProblem(model!, u0, t_span, p)
    sol = solve(prob, reltol = 1e-8, abstol = 1e-8)
    grid = sol(ts)
    eq = nlsolve((du, u) -> model!(du, u, p, 0.0), grid.u[end]).zero
    K1_hold[i,1] = K1_vals[i]
    K1_hold[i,2:end] = eq
    jac_K1 = jac(eq, model, p)
    all_eig = real.(eigvals(jac_K1))
    eig_hold[i,1] = K1_vals[i]
    eig_hold[i,2:end] = all_eig
    println(K1_hold[i,:])
end

## plot all real eigs
eig_1 = Plots.plot(eig_hold[:,1], eig_hold[:,2], legend = false, lw= 2.0, colour = "black", xlabel = " K1 ", ylabel = " Eig 1 ", xflip = true )

eig_2 = Plots.plot(eig_hold[:,1], eig_hold[:,3], legend = false, lw= 2.0, colour = "black", xlabel = " K1 ", ylabel = " Eig 2 ", xflip = true )

eig_3 = Plots.plot(eig_hold[:,1], eig_hold[:,4], legend = false, lw= 2.0, colour = "black", xlabel = " K1 ", ylabel = " Eig 3 ", xflip = true )

eig_4 = Plots.plot(eig_hold[:,1], eig_hold[:,5], legend = false, lw= 2.0, colour = "black", xlabel = " K1 ", ylabel = " Eig 4 ", xflip = true )

eig_5 = Plots.plot(eig_hold[:,1], eig_hold[:,6], legend = false, lw= 2.0, colour = "black", xlabel = " K1 ", ylabel = " Eig 5 ", xflip = true )


## calculate max real eigs 
maxeig_hold = fill(0.0,length(K1_vals),2)

for i=1:length(K1_vals)
    p = ModelPar(K1 = K1_vals[i])
    u0 = [0.8, 0.4, 0.8, 0.4, 0.3]
    prob = ODEProblem(model!, u0, t_span, p)
    sol = solve(prob, reltol = 1e-8, abstol = 1e-8)
    grid = sol(ts)
    eq = nlsolve((du, u) -> model!(du, u, p, 0.0), grid.u[end]).zero
    coup_jac = jac(eq, model, p)
    max_eig = maximum(real.(eigvals(coup_jac)))
    maxeig_hold[i,1] = K1_vals[i]
    maxeig_hold[i,2] = max_eig
    println(maxeig_hold[i,:])
end


## plot max real eig 
max_eig = Plots.plot(maxeig_hold[:,1], maxeig_hold[:,2], legend = false, lw= 5.0, colour = "black", xlabel = " K1 ", ylabel = " Real Max Eig " , xflip = true,
yflip = true, grid = false,  xguidefontsize = 18, yguidefontsize = 18)


## calculate all five imaginary eigs 

eig_imag_hold = fill(0.0,length(K1_vals),6)

for i=1:length(K1_vals)
    p = ModelPar(K1 = K1_vals[i])
    u0 = [0.8, 0.4, 0.8, 0.4, 0.3]
    prob = ODEProblem(model!, u0, t_span, p)
    sol = solve(prob, reltol = 1e-8, abstol = 1e-8)
    grid = sol(ts)
    eq = nlsolve((du, u) -> model!(du, u, p, 0.0), grid.u[end]).zero
    K1_hold[i,1] = K1_vals[i]
    K1_hold[i,2:end] = eq
    jac_K2 = jac(eq, model, p)
    all_imag_eig = abs.(imag.(eigvals(jac_K2)))
    eig_imag_hold[i,1] = K1_vals[i]
    eig_imag_hold[i,2:end] = all_imag_eig
    println(K1_hold[i,:])
end

## plot all real eigs
eig_1 = Plots.plot(eig_imag_hold[:,1], eig_imag_hold[:,2], legend = false, lw= 2.0, colour = "black", xlabel = " K1 ", ylabel = " Complex Eig 1 ", xflip = true )

eig_2 = Plots.plot(eig_imag_hold[:,1], eig_imag_hold[:,3], legend = false, lw= 2.0, colour = "black", xlabel = " K1 ", ylabel = " Complex Eig 2 ", xflip = true )

eig_3 = Plots.plot(eig_imag_hold[:,1], eig_imag_hold[:,4], legend = false, lw= 2.0, colour = "black", xlabel = " K1 ", ylabel = " Complex Eig 3 ", xflip = true )

eig_4 = Plots.plot(eig_imag_hold[:,1], eig_imag_hold[:,5], legend = false, lw= 2.0, colour = "black", xlabel = " K1 ", ylabel = " Complex Eig 4 ", xflip = true )

eig_5 = Plots.plot(eig_imag_hold[:,1], eig_imag_hold[:,6], legend = false, lw= 2.0, colour = "black", xlabel = " K1 ", ylabel = " Complex Eig 5 ", xflip = true )


## calculate max imaginary eigs 
maxeig_imag_hold = fill(0.0,length(K1_vals),2)

for i=1:length(K1_vals)
    p = ModelPar(K1 = K1_vals[i])
    u0 = [0.8, 0.4, 0.8, 0.4, 0.3]
    prob = ODEProblem(model!, u0, t_span, p)
    sol = solve(prob, reltol = 1e-8, abstol = 1e-8)
    grid = sol(ts)
    eq = nlsolve((du, u) -> model!(du, u, p, 0.0), grid.u[end]).zero
    coup_jac = jac(eq, model, p)
    max_eig_imag = maximum(abs.(imag.(eigvals(coup_jac))))
    maxeig_imag_hold[i,1] = K1_vals[i]
    maxeig_imag_hold[i,2] = max_eig_imag
    println(maxeig_imag_hold[i,:])
end


## plot max imaginary eig 
max_eig_imag = Plots.plot(maxeig_imag_hold[:,1], maxeig_imag_hold[:,2], legend = false, lw= 2.0, colour = "black", xlabel = " K1 ", ylabel = " Imaginary Max Eig " , xflip = true)
 
### Degree coupling

function degree_coupling_K1(eq, p)
    return degree_coupling(eq, p)
end


coupling_hold= zeros(length(K1_vals), 2)

for i=1:length(K1_vals)
    p = ModelPar(K1 = K1_vals[i])
    u0 = [0.8, 0.4, 0.8, 0.4, 0.3]
    prob = ODEProblem(model!, u0, t_span, p)
    sol = solve(prob, reltol = 1e-8, abstol = 1e-8)
    grid = sol(ts)
    eq = nlsolve((du, u) -> model!(du, u, p, 0.0), grid.u[end]).zero
    coupling_val = degree_coupling_K1(eq, p)
    coupling_hold[i,1] = K1_vals[i]
    coupling_hold[i,2] = coupling_val
    println(coupling_hold[i,:])

end

degree_coupling_plot = Plots.plot(coupling_hold[:, 1], coupling_hold[:, 2], legend = false, grid = false, lw = 5.0, color = "black", xlabel = "K1", ylabel = "Degree of Coupling", xflip = true)


