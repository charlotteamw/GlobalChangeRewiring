using Parameters: @with_kw, @unpack
using DifferentialEquations
using PyPlot
using Plots
using Statistics

@with_kw mutable struct Params
    a_R1C = 0.9
    h_R1C = 0.6
    e_R1C = 0.4
    a_R2C = 0.9
    h_R2C = 0.6
    e_R2C = 0.4
    m_C = 0.2
    r1 = 1.0
    K1 = 1.5
    r2 = 1.0
    K2 = 1.5
    amp_K1 = 0.3
    amp_K2 = -0.3
    freq = 0.1
    Ω = 0.5  # Preference for R1 over R2
end


function sine_model!(du, u, p, t)
    R1, R2, C = u
    @unpack r1, K1, r2, K2, a_R1C, h_R1C, e_R1C, a_R2C, h_R2C, e_R2C, m_C, Ω = p
    @unpack freq, amp_K1, amp_K2 = p
    # Applying forcing to modulate carrying capacity or other parameters

    K1_modulated = K1 + amp_K1 * sin(2 * pi * freq * t)
    K2_modulated = K2 + amp_K2 * sin(2 * pi * freq * t)

    # Functional responses using actual resource populations
    intake_R1 = Ω * a_R1C * R1 * C / (1 + a_R1C * h_R1C * R1)
    intake_R2 = (1 - Ω) * a_R2C * R2 * C / (1 + a_R2C * h_R2C * R2)
    
    # Logistic growth using modulated carrying capacities
    du[1] = r1 * R1 * (1 - R1 / K1) - intake_R1
    du[2] = r2 * R2 * (1 - R2 / K2) - intake_R2
    du[3] = (e_R1C * intake_R1 + e_R2C * intake_R2) - m_C * C

end

function sine_model(u, Params, t)
    du = similar(u)
    sine_model!(du, u, Params, t)
    return du
end



let
    u0 = [0.8, 0.8, 0.5]
    t_span = (0, 100.0)
    p = Params()
    prob = ODEProblem(sine_model!, u0, t_span, p)
    sol = solve(prob, reltol=1e-8, abstol=1e-8) 
    model_ts = figure()
    colors = ["#199bf7", "#e8d03c", "#1a1913"]  # You can also use hex codes like "#1f77b4", "#2ca02c", "#d62728"
    labels = ["R1", "R2", "C"]
    for i in 1:3
        PyPlot.plot(sol.t, getindex.(sol.u, i), label=labels[i], color=colors[i], linewidth = 4.0)
    end
    xlabel("Time")
    ylabel("Density")
    return model_ts
end


## calculate equilibrium densities - using grid instead of sol to remove transient 
K2_amp_vals = -0.3:0.0001:0.3
K2_amp_hold = fill(0.0,length(K2_amp_vals),2)

u0 = [0.8, 0.8, 0.5]
p = Params()
t_span = (0, 10000.0)
ts = range(1000, 1500, length = 500)

for i = 1:length(K2_amp_vals)
    p = Params(amp_K2 = K2_amp_vals[i])  # Assume Params has been updated correctly to use amp_K2
    prob = ODEProblem(sine_model!, [0.8, 0.8, 0.5], (0.0, 1000.0), p)
    sol = solve(prob, reltol = 1e-8, abstol = 1e-8, saveat=1.0)  # Ensure you capture each step to get precise indexing

    # Extracting data from time steps 500 to 1000
    C_values = [sol.u[j][3] for j in 500:1000]
    cv_C = std(C_values) / mean(C_values)

    K2_amp_hold[i, 1] = K2_amp_vals[i]
    K2_amp_hold[i, 2] = cv_C
    println(K2_amp_hold[i,:])
end

# Plotting
cv_Consumer = Plots.plot(K2_amp_hold[:,1], K2_amp_hold[:,2], legend=false, lw=2.0, color="black", xlabel="K2 Amplitude", ylabel="Consumer CV", xflip = true,
                   grid=false,  # Disable grid lines
                   xguidefontsize=18,  # Set font size for x-axis label
                   yguidefontsize=18,  # Set font size for y-axis label
                   titlefontsize=18)  # Set font size for the plot title if needed

##Eigenvalues 
## calculating the jacobian 
function jac(u, sine_model, p)
    ForwardDiff.jacobian(u -> sine_model(u, p, NaN), u)
end 

## calculate equilibrium densities - using grid instead of sol to remove transient 
K2_vals = 0.741:0.01:1.5
K2_hold = fill(0.0,length(K2_vals),4)

u0 = [0.8, 0.8, 0.5]
p = Params()
t_span = (0, 10000.0)
ts = range(1000, 1500, length = 500)

for i=1:length(K2_vals)
p = Params(K2 = K2_vals[i])
u0 = [0.8, 0.8, 0.5]
prob = ODEProblem(sine_model!, u0, t_span, p)
sol = solve(prob, reltol = 1e-8, abstol = 1e-8)
grid = sol(ts)
eq = nlsolve((du, u) -> sine_model!(du, u, p, 0.0), grid.u[end]).zero
K2_hold[i,1] = K2_vals[i]
K2_hold[i,2:end] = eq
println(K2_hold[i,:])
end

## plot equilibrium densities 
using Plots

eq_R1 = Plots.plot(K2_hold[:,1], K2_hold[:,2], legend = false, lw= 2.0, colour = "black", xlabel = " K2 ", ylabel = " R1 Equilibrium Density " , xflip = true)

eq_R2 = Plots.plot(K2_hold[:,1], K2_hold[:,3], legend = false, lw= 2.0, colour = "black", xlabel = " K2 ", ylabel = " R2 Equilibrium Density " , xflip = true)

eq_C = Plots.plot(K2_hold[:,1], K2_hold[:,4], legend = false, lw= 2.0, linecolour = "darkorange", xlabel = " K2 ", ylabel = " C Equilibrium Density " , xflip = true)


## calculate all five real eigs 

eig_hold = fill(0.0,length(K2_vals),4)

for i=1:length(K2_vals)
p = Params(K2 = K2_vals[i])
u0 = [0.8, 0.8, 0.5]
prob = ODEProblem(sine_model!, u0, t_span, p)
sol = solve(prob, reltol = 1e-8, abstol = 1e-8)
grid = sol(ts)
eq = nlsolve((du, u) -> sine_model!(du, u, p, 0.0), grid.u[end]).zero
K2_hold[i,1] = K2_vals[i]
K2_hold[i,2:end] = eq
jac_K2 = jac(eq, sine_model, p)
all_eig = real.(eigvals(jac_K2))
eig_hold[i,1] = K2_vals[i]
eig_hold[i,2:end] = all_eig
println(K2_hold[i,:])
end

## plot all real eigs
eig_1 = Plots.plot(eig_hold[:,1], eig_hold[:,2], legend = false, lw= 2.0, colour = "black", xlabel = " K2 ", ylabel = " Eig 1 ", xflip = true )

eig_2 = Plots.plot(eig_hold[:,1], eig_hold[:,3], legend = false, lw= 2.0, colour = "black", xlabel = " K2 ", ylabel = " Eig 2 ", xflip = true )

eig_3 = Plots.plot(eig_hold[:,1], eig_hold[:,4], legend = false, lw= 2.0, colour = "black", xlabel = " K2 ", ylabel = " Eig 3 ", xflip = true )


## calculate max real eigs 
maxeig_hold = fill(0.0,length(K2_vals),2)

for i=1:length(K2_vals)
p = Params(K2 = K2_vals[i])
u0 = [0.8, 0.8, 0.5]
prob = ODEProblem(sine_model!, u0, t_span, p)
sol = solve(prob, reltol = 1e-8, abstol = 1e-8)
grid = sol(ts)
eq = nlsolve((du, u) -> sine_model!(du, u, p, 0.0), grid.u[end]).zero
coup_jac = jac(eq, sine_model, p)
max_eig = maximum(real.(eigvals(coup_jac)))
maxeig_hold[i,1] = K2_vals[i]
maxeig_hold[i,2] = max_eig
println(maxeig_hold[i,:])
end

## plot max real eig 
max_eig = Plots.plot(maxeig_hold[:,1], maxeig_hold[:,2], legend = false, lw= 2.0, colour = "black", xlabel = " K2 ", ylabel = " Real Max Eig " , 
yflip = true, grid = false,  xguidefontsize = 18, yguidefontsize = 18)
