using Parameters: @with_kw, @unpack
using LinearAlgebra: eigvals
using ForwardDiff
using QuadGK: quadgk
using NLsolve
using DifferentialEquations
using Plots
using PyPlot
using Statistics

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
    a_PC2 = 1.2
    h_PC2 = 0.6
    e_PC2 = 0.7
    m_P = 0.3
    r1 = 2.0
    K1 = 1.1
    r2 = 2.0
    K2 = 2.0
    m_C1 = 0.3
    m_C2 = 0.3
    pref::Function = adapt_pref
    Ω = 0.36
    ω = 0.
    noise = 0.001
end

## functions for later

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

function numres_PC1(u, p, t)
    R1, C1, R2, C2, P = u
    Ω = p.pref(u, p, t)
    return Ω * p.a_PC1 * C1 * P * p.e_PC1
end

function numres_PC2(u, p, t)
    R1, C1, R2, C2, P = u
    Ω = p.pref(u, p, t)
    return (1 - Ω) * p.a_PC2 * C2 * P * p.e_PC2
end

function log_growth_R1(u, p, t)
    R1 = u[1]  # Assuming R1 is the first component of the vector u
    return p.r1 * R1 * (1 - R1 / p.K1)
end

function log_growth_R2(u, p, t)
    R2 = u[3]  # Assuming R2 is the third component of the vector u
    return p.r2 * R2 * (1 - R2 / p.K2)
end

function jac(u, model, p)
    ForwardDiff.jacobian(u -> model(u, p, NaN), u)
end 

degree_coupling(u, p) = f_PC1(u, p, 0.0) / (f_PC1(u, p, 0.0) + f_PC2(u, p, 0.0))

sec_prod(u, p) = numres_PC1(u, p, 0.0) + numres_PC1(u, p, 0.0)

primary_prod(u, p) = log_growth_R1(u, p, 0.0) + log_growth_R2(u, p, 0.0)

### Food Web Model

function model!(du, u, p, t)
    @unpack r1, K1, r2, K2 = p
    @unpack a_R1C1, h_R1C1, e_R1C1, a_R2C2, h_R2C2, e_R2C2, m_C1, m_C2 = p
    @unpack a_PC1, h_PC1, e_PC1, a_PC2, h_PC2, e_PC2, m_P, Ω, ω  = p
    R1, C1, R2, C2, P = u

    Ω = p.pref(u, p, t)

    int_R1C1 = a_R1C1 * R1 * C1 / (1 + a_R1C1 * h_R1C1 * R1)
    int_R2C2 = a_R2C2 * R2 * C2 / (1 + a_R2C2 * h_R2C2 * R2)
    denom_PC1C2 = 1 + Ω * a_PC1 * h_PC1 * C1 + (1 - Ω) * a_PC2 * h_PC2 * C2
    num_PC1 = Ω * a_PC1 * C1 * P
    num_PC2 = (1 - Ω) * a_PC2 * C2 * P

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

## time series for food web model with ModelPar
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

    
### Details for all the calculations that follow
K1_vals = 0.705:0.005:1.4
u0 = [0.8, 0.4, 0.8, 0.4, 0.3]
p = ModelPar()
ts = range(0, 2000, length = 2000)  # Time steps
t_span = (0.0, 2000.0)  # Time span


## eq data frame
eq_hold = fill(0.0,length(K1_vals),6)
### Equilibrium densities
for i=1:length(K1_vals)
    p = ModelPar(K1 = K1_vals[i])
    u0 = [0.8, 0.4, 0.8, 0.4, 0.3]
    prob = ODEProblem(model!, u0, t_span, p)
    sol = solve(prob, reltol = 1e-8, abstol = 1e-8)
    grid = sol(ts)
    eq = nlsolve((du, u) -> model!(du, u, p, 0.0), grid.u[end]).zero
    eq_hold[i,1] = K1_vals[i]
    eq_hold[i,2:end] = eq
    println(eq_hold[i,:])
end

## plot equilibrium densities 
using Plots

eq_R1 = Plots.plot(eq_hold[:,1], eq_hold[:,2], legend = false, lw= 2.0, colour = "black", xlabel = " K1 ", ylabel = " R1 Equilibrium Density " , xflip = true)

eq_C1 = Plots.plot(eq_hold[:,1], eq_hold[:,3], legend = false, lw= 2.0, colour = "black", xlabel = " K1 ", ylabel = " C1 Equilibrium Density " , xflip = true)

eq_R2 = Plots.plot(eq_hold[:,1], eq_hold[:,4], legend = false, lw= 2.0, linecolour = "darkorange", xlabel = " K1 ", ylabel = " R2 Equilibrium Density " , xflip = true)

eq_C2 = Plots.plot(eq_hold[:,1], eq_hold[:,5], legend = false, lw= 2.0, linecolour = "green", xlabel = " K1 ", ylabel = " C2 Equilibrium Density " , xflip = true)

eq_P = Plots.plot(eq_hold[:,1], eq_hold[:,6], legend = false, lw= 2.0, colour = "black", xlabel = " K1 ", ylabel = " P Equilibrium Density " , xflip = true)

## plot P:C biomass ratio
PC_biomass = Plots.plot(eq_hold[:,1], eq_hold[:,6] ./ eq_hold[:,3], legend = false, lw = 5.0, color = "black", xlabel = "K1", ylabel = "P:C1 Biomass Ratio", xflip = true, grid = false,
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
    jac_K1 = jac(eq, model, p)
    all_eig = real.(eigvals(jac_K1))
    eig_hold[i,1] = K1_vals[i]
    eig_hold[i,2:end] = all_eig
    println(eig_hold[i,:])
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

## plot all imaginary eigs
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


### Secondary Production

function sec_prod_K1(eq, p)
    return sec_prod(eq, p)
end

sec_prod_hold= zeros(length(K1_vals), 2)

for i=1:length(K1_vals)
    p = ModelPar(K1 = K1_vals[i])
    u0 = [0.8, 0.4, 0.8, 0.4, 0.3]
    prob = ODEProblem(model!, u0, t_span, p)
    sol = solve(prob, reltol = 1e-8, abstol = 1e-8)
    grid = sol(ts)
    eq = nlsolve((du, u) -> model!(du, u, p, 0.0), grid.u[end]).zero
    sec_prod_val = sec_prod_K1(eq, p)
    sec_prod_hold[i,1] = K1_vals[i]
    sec_prod_hold[i,2] = sec_prod_val
    println(sec_prod_hold[i,:])

end

sec_prod_plot = Plots.plot(sec_prod_hold[:, 1], sec_prod_hold[:, 2], legend = false, grid = false, lw = 5.0, color = "black", xlabel = "K1", ylabel = "Predator Production", xflip = true)



### Primary production 

function primary_prod_K1(eq, p)
    return primary_prod(eq, p)
end

primary_prod_hold= zeros(length(K1_vals), 2)

for i=1:length(K1_vals)
    p = ModelPar(K1 = K1_vals[i])
    u0 = [0.8, 0.4, 0.8, 0.4, 0.3]
    prob = ODEProblem(model!, u0, t_span, p)
    sol = solve(prob, reltol = 1e-8, abstol = 1e-8)
    grid = sol(ts)
    eq = nlsolve((du, u) -> model!(du, u, p, 0.0), grid.u[end]).zero
    primary_prod_val = primary_prod_K1(eq, p)
    primary_prod_hold[i,1] = K1_vals[i]
    primary_prod_hold[i,2] = primary_prod_val
    println(primary_prod_hold[i,:])

end

primary_prod_plot = Plots.plot(primary_prod_hold[:, 1], primary_prod_hold[:, 2], legend = false, grid = false, lw = 5.0, color = "black", xlabel = "K1", ylabel = "Primary Production", xflip = true)


###### Stochastic model for CV

## Adding stochasticity to model using gaussian white noise (SDEproblem)
function stochmodel!(du, u, p2, t)
    @unpack  noise = p2

    du[1] = noise * u[1]
    du[2] = noise * u[2]
    du[3] = noise * u[3]
    du[4] = noise * u[4]
    du[5] = noise * u[5]

    return du 
end

## time series for stochastic food web model with ModelPar
let
    u0 = [0.8, 0.4, 0.8, 0.4, 0.3]
    t_span = (0, 2000.0)
    p = ModelPar(K1 = 0.705)
    prob = SDEProblem(model!, stochmodel!, u0, t_span, p)
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

##cv data frame
cv_hold = zeros(length(K1_vals), 2)
stdhold = fill(0.0, length(K1_vals), 2)
meanhold = fill(0.0, length(K1_vals), 2)

#### CV of predator population
for i=1:length(K1_vals)
    p = ModelPar(K1 = K1_vals[i])
    u0 = [0.8, 0.4, 0.8, 0.4, 0.3]
    prob_stoch = SDEProblem(model!, stochmodel!, u0, t_span, p)
    sol_stoch = solve(prob_stoch, reltol = 1e-15)
    grid_sol = sol_stoch(ts)

    # Recording the statistics
    cv_hold[i, 1] = K1_vals[i]
    stdhold[i, 1] = std(grid_sol[5, 1000:2000])
    meanhold[i, 1] = mean(grid_sol[5, 1000:2000])
    cv_hold[i, 2] = stdhold[i, 1] / meanhold[i, 1]

    println(cv_hold[i,:])
end

cv_plot = Plots.plot(cv_hold[:, 1], cv_hold[:, 2], legend = false, grid = false, lw = 5.0, color = "black", xlabel = "K1", ylabel = "Predator CV", xflip = true)



