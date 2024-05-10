using Parameters
using ForwardDiff
using LinearAlgebra
using PyPlot
using DifferentialEquations
using NLsolve
using Statistics
using RecursiveArrayTools
using Noise
using Distributed
using StatsBase
using Random
using CSV
using DataFrames

## Here we will illustrate the case of a generalist predator that feeds on fast and slow channel.
## Parameters are categorized by macrohabitat but coud be named fast slow more generally  -> parameters with "_2" or slow indicate 2oral macrohabitat values and those with "_1" or fast indicate 1agic macrohabitat values  

@with_kw mutable struct GenMod_Par
    α_1 = 0.0   ##competitive influence of R1 on R2 
    α_2 = 0.0   ##competitve influence of R2 on R1
    k_2 = 1.0
    k_1 = 1.0
    e_CR = 0.8
    e_PC = 0.6
    e_PR = 0.8
    m_P = 0.2

    a_PC_2 = 0.1
    a_PR_1 = 0.00 
    a_PC_1 = 0.3
    a_PR_2 = 0.00
    h_PC = 0.60
    h_PR = 1.0
  
#slow CR
    r_2 = 1.0
    a_CR_2 = 0.5
    m_Cl = 0.10
    h_CRl = 1.0

#fast CR
    r_1 = 1.0
    a_CR_1 = 0.9
    m_Cp = 0.20
    h_CRp = 1.0

# noise
    noise = 0.01
end


## Omnivory Module with Temp Dependent Attack Rates (a_PC_2 => aPC in 2oral zone; a_PC_1 => aPC in 1agic zone)

function GenMod_model!(du, u, p, t)
    @unpack r_2, r_1, k_2, k_1, α_1, α_2, e_CR, e_PC, e_PR, a_CR_2, a_CR_1, a_PR_2, a_PR_1, h_CRl, h_CRp,h_PC, h_PR, m_Cl, m_Cp,m_P, a_PC_2,a_PC_1 = p 
    
    
    R_2, R_1, C_2, C_1, P = u
    
    du[1]= r_2 * R_2 * (1 - (α_1 * R_1 + R_2)/k_2) - (a_CR_2 * R_2 * C_2)/(1 + a_CR_2 * h_CRl * R_2) - (a_PR_2 * R_2 * P)/(1 + a_PR_2 * h_PR * R_2 + a_PR_1 * h_PR * R_1 + a_PC_2 * h_PC * C_2 + a_PC_1 * h_PC * C_1)
    
    du[2] = r_1 * R_1 * (1 - (α_2 * R_2 + R_1)/k_1) - (a_CR_1 * R_1 * C_1)/(1 + a_CR_1 * h_CRp * R_1) - (a_PR_1 * R_1 * P)/(1 + a_PR_2 * h_PR * R_2 + a_PR_1 * h_PR * R_1 + a_PC_2 * h_PC * C_2 + a_PC_1 * h_PC * C_1)

    du[3] = (e_CR * a_CR_2 * R_2 * C_2)/(1 + a_CR_2 * h_CRl * R_2) - (a_PC_2 * C_2 * P)/(1 + a_PR_2 * h_PR * R_2 + a_PR_1 * h_PR * R_1 + a_PC_2 * h_PC * C_2 + a_PC_1 * h_PC * C_1) - m_Cl * C_2
    
    du[4] = (e_CR * a_CR_1 * R_1 * C_1)/(1 + a_PC_1 * h_CRp * R_1) - (a_PC_1 * C_1 * P)/(1 + a_PR_2 * h_PR * R_2 + a_PR_1 * h_PR * R_1 + a_PC_2 * h_PC * C_2 + a_PC_1 * h_PC * C_1) - m_Cp * C_1

    du[5] = (e_PR * a_PR_2 * R_2 * P + e_PR * a_PR_1 * R_1 * P + e_PC * a_PC_2 * C_2 * P + e_PC * a_PC_1 * C_1 * P)/(1 + a_PR_2 * h_PR * R_2 + a_PR_1 * h_PR * R_1 + a_PC_2 * h_PC * C_2 + a_PC_1 * h_PC * C_1) - m_P * P

    return 
end

function GenMod_model(u, p)
    du = similar(u)
    GenMod_model!(du, u, p, 0.0)
    return du
end


## Adding stochasticity to model using gaussian white noise (SDEproblem)
function GenMod_stochmodel!(du, u, p2, t)
    @unpack  noise = p2

    du[1] = noise * u[1]
    du[2] = noise * u[2]
    du[3] = noise * u[3]
    du[4] = noise * u[4]
    du[5] = noise * u[5]

    return du 
end


# λ_stability is from FoodWebs.jl -- it is a trivial function being:
# maximum(real.(eigvals(M))), where eigvals is from the standard library LinearAlgebra
λ_stability(M) = maximum(real.(eigvals(M)))
imag_eig(M) = maximum(imag(eigvals(M)))

calc_λ1(eq, par) = λ_stability(ForwardDiff.jacobian(eq -> GenMod_model(eq, par), eq))

calc_λe(eq, par) = imag_eig(ForwardDiff.jacobian(eq -> GenMod_model(eq, par), eq))

par = GenMod_Par()
tspan = (0.0, 1000.0)
u0 = [1.0, 1.0, 0.5, 0.5, 0.1]

prob = ODEProblem(GenMod_model!, u0, tspan, par)

sol = solve(prob)

eq = nlsolve((du, u) -> GenMod_model!(du, u, par, 0.0), sol.u[end]).zero

# stochastic C-R time series -- just to look at timeseries specifically whenever we want 
let
    u0 = [1.0, 1.0, 0.5, 0.5, 0.25]
    t_span = (0.0, 2000.0)
    rmax_add = 0.3
    par = GenMod_Par(noise = 0.01, α_1 = 0.0, α_2 = 0.0)

    par.a_CR_1 = (par.a_CR_1 + rmax_add)
    par.a_CR_2 = (par.a_CR_2 + rmax_add)
    par.a_PC_1 = (par.a_PC_1 + rmax_add)
    par.a_PC_2 = (par.a_PC_2 + rmax_add)

    prob_stoch = SDEProblem(GenMod_model!, GenMod_stochmodel!, u0, t_span, par)
    sol_stoch = solve(prob_stoch, reltol = 1e-15)
    ts_genmod = figure()
    plot(sol_stoch.t[1:end], sol_stoch[1, 1:end], label = "R1")
    plot(sol_stoch.t[1:end], sol_stoch[2, 1:end], label = "R2")
    plot(sol_stoch.t[1:end], sol_stoch[3, 1:end], label = "C2")
    plot(sol_stoch.t[1:end], sol_stoch[4, 1:end], label = "C1")
    plot(sol_stoch.t[1:end], sol_stoch[5, 1:end], label = "P")
    xlabel("Time")
    ylabel("Abundance")
    legend()
    return ts_genmod
end

/


#Set colour palette
myPal = ["#FEE391","#F7834D", "#DA464C", "#9E0142"]

####################################################################
##### Investigate all over changing predator and consumer attack rates ######
####################################################################

##### Deterministic Model #####
let
    rmax_increases = 0.01:0.01:0.60
    stab = fill(NaN, 2, length(rmax_increases))

    # Initialize your parameters
    par = GenMod_Par(noise = 0.00)  # Make sure to initialize this with your base parameters

    for (i, rmax_increase) in enumerate(rmax_increases)
        # Modify all attack rates by the percentage increase
        par.a_CR_1 = (par.a_CR_1 + rmax_increase)
        par.a_CR_2 = (par.a_CR_2 + rmax_increase)
        par.a_PC_1 = (par.a_PC_1 + rmax_increase)
        par.a_PC_2 = (par.a_PC_2 + rmax_increase)

        # Solve the ODE problem
        prob = SDEProblem(GenMod_model!, GenMod_stochmodel!, u0, tspan, par)
        sol = solve(prob, reltol = 1e-15)

        # Find the equilibrium state
        eq = nlsolve((du, u) -> GenMod_model!(du, u, par, 0.0), sol.u[end]).zero

        # Calculate and store the eigenvalues
        stab[1, i] = calc_λ1(eq, par)
        stab[2, i] = calc_λe(eq, par)

        # Reset the parameters to base values for next iteration
        par = GenMod_Par(noise = 0.01)  # Reset to base parameters
    end

    GenMod_eq = figure(figsize = (6,4))
    #subplot(211)
    plot(rmax_increases, stab[1, :], color = "black")
    axhline(0, color = "black", linestyle = "--")
    ylabel(L"\lambda_max", fontsize=16, fontweight=:bold)
    ylim(-0.175, 0.025)
    #axvspan(-0.40, 0.11, facecolor="gray", alpha=0.5)

    #subplot(212)
    #plot(rmax_increases, stab[2, :])
    #xlabel("Attack Rate (a)",fontsize=16,fontweight=:bold)
    #ylabel(L"\lambda_ imag",fontsize=16,fontweight=:bold)
    #axvspan(-0.40, 0.11, facecolor="gray", alpha=0.5)
    
    tight_layout()

    savefig("/Users/reillyoconnor/Desktop/Julia Projects/Slow-Fast/Figures/Eigenvalues_Deterministic_Module.pdf", dpi = 300)

    return GenMod_eq
end



##### Stochastic Model #####
let
    rmax_increases = 0.01:0.01:0.60
    stab = fill(NaN, 2, length(rmax_increases))

    # Initialize your parameters
    par = GenMod_Par(noise = 0.01)  # Make sure to initialize this with your base parameters

    for (i, rmax_increase) in enumerate(rmax_increases)
        # Modify all attack rates by the percentage increase
        par.a_CR_1 = (par.a_CR_1 + rmax_increase)
        par.a_CR_2 = (par.a_CR_2 + rmax_increase)
        par.a_PC_1 = (par.a_PC_1 + rmax_increase)
        par.a_PC_2 = (par.a_PC_2 + rmax_increase)

        # Solve the ODE problem
        prob = SDEProblem(GenMod_model!, GenMod_stochmodel!, u0, tspan, par)
        sol = solve(prob, reltol = 1e-15)

        # Find the equilibrium state
        eq = nlsolve((du, u) -> GenMod_model!(du, u, par, 0.0), sol.u[end]).zero

        # Calculate and store the eigenvalues
        stab[1, i] = calc_λ1(eq, par)
        stab[2, i] = calc_λe(eq, par)

        # Reset the parameters to base values for next iteration
        par = GenMod_Par(noise = 0.01)  # Reset to base parameters
    end

    GenMod_eq = figure()
    #subplot(211)
    plot(rmax_increases, stab[1, :])
    axhline(0, color = "black", linestyle = "--")
    ylabel(L"\lambda_max",fontsize=16,fontweight=:bold)
    #axvspan(-0.40, 0.11, facecolor="gray", alpha=0.5)

    #subplot(212)
    #plot(rmax_increases, stab[2, :])
    #xlabel("Attack Rate (a)",fontsize=16,fontweight=:bold)
    #ylabel(L"\lambda_ imag",fontsize=16,fontweight=:bold)
    #axvspan(-0.40, 0.11, facecolor="gray", alpha=0.5)
    
    tight_layout()

    #savefig("/Users/reillyoconnor/Desktop/Julia Projects/Slow-Fast/Figures/Eigenvalues_D0.pdf", dpi = 300)

    return GenMod_eq
end

/

let
    percent_increases = 0.0:0.001:0.05
    rmax_high = 0.11
    rmax_low = -0.40
    n_seeds = 100
    maxhold = fill(0.0, length(percent_increases), 1)
    stdhold = fill(0.0, length(percent_increases), 2)
    meanhold = fill(0.0, length(percent_increases), 2)
    cv_high_all = fill(0.0, length(percent_increases), n_seeds)
    cv_low_all = fill(0.0, length(percent_increases), n_seeds)
    
    
    ts = range(0, 2000, length = 2000)  # Time steps
    t_span = (0.0, 2000.0)  # Time span

    # Initialize your parameters
    par = GenMod_Par()  # Make sure to initialize this with your base parameters

  for seed in 1:n_seeds
    Random.seed!(seed)
    print(seed)

    for (i, perc_increase) in enumerate(percent_increases)
        # Modify all attack rates by the percentage increase
        par = GenMod_Par()
        

        par.a_CR_1 = (par.a_CR_1 + rmax_high)
        par.a_CR_2 = (par.a_CR_2 + rmax_high)
        par.a_PC_1 = (par.a_PC_1 + rmax_high)
        par.a_PC_2 = (par.a_PC_2 + rmax_high)

        par.a_CR_1 *= (1 + perc_increase)
        par.a_CR_2 *= (1 + perc_increase)
        par.a_PC_1 *= (1 + perc_increase)
        par.a_PC_2 *= (1 + perc_increase)

        u0 = [1.0, 1.0, 0.5, 0.5, 0.25]
        prob_stoch = SDEProblem(GenMod_model!, GenMod_stochmodel!, u0, t_span, par)
        sol_stoch = solve(prob_stoch, reltol = 1e-15)
        grid_sol = sol_stoch(ts)

        # Recording the statistics
        maxhold[i, 1] = percent_increases[i]
        stdhold[i, 1] = std(grid_sol[5, 1500:2000])
        meanhold[i, 1] = mean(grid_sol[5, 1500:2000])
        cv_high_all[i, seed] = stdhold[i, 1] / meanhold[i, 1]

        #Reset parameters to run rmax low
        par = GenMod_Par()

        par.a_CR_1 = (par.a_CR_1 + rmax_low)
        par.a_CR_2 = (par.a_CR_2 + rmax_low)
        par.a_PC_1 = (par.a_PC_1 + rmax_low)
        par.a_PC_2 = (par.a_PC_2 + rmax_low)

        par.a_CR_1 *= (1 + perc_increase)
        par.a_CR_2 *= (1 + perc_increase)
        par.a_PC_1 *= (1 + perc_increase)
        par.a_PC_2 *= (1 + perc_increase)

        u0 = [1.0, 1.0, 0.5, 0.5, 0.25]
        prob_stoch = SDEProblem(GenMod_model!, GenMod_stochmodel!, u0, t_span, par)
        sol_stoch = solve(prob_stoch, reltol = 1e-15)
        grid_sol = sol_stoch(ts)

        # Recording the statistics
        maxhold[i, 1] = percent_increases[i]
        stdhold[i, 2] = std(grid_sol[5, 1800:2000])
        meanhold[i, 2] = mean(grid_sol[5, 1800:2000])
        cv_low_all[i, seed] = stdhold[i, 2] / meanhold[i, 2]
        
    end
end  
    cv_high_avg = mean(cv_high_all, dims=2)
    cv_low_avg = mean(cv_low_all, dims=2)
    
    cv_rmax_high = cv_high_avg[1]
    percent_change_cv_high = [(cv - cv_rmax_high) / cv_rmax_high * 100 for cv in cv_high_avg]

    cv_rmax_low = cv_low_avg[1]
    percent_change_cv_low = [(cv - cv_rmax_low) / cv_rmax_low * 100 for cv in cv_low_avg]

    # Create a DataFrame
    df = DataFrame(
        Percent_Increase = vec(maxhold[:, 1]),
        Percent_Change_CV_High = vec(percent_change_cv_high[:, 1]), 
        Percent_Change_CV_Low = vec(percent_change_cv_low[:, 1]), 
    )

    # Write to CSV file
    CSV.write("/Users/reillyoconnor/Desktop/Julia Projects/Slow-Fast/Theory GC Percent.csv", df)

    # Plotting
    CV_figure = figure(figsize = (8, 6))

    #subplot(311)
    #plot(maxhold[:, 1], stdhold, color="black", linewidth=2)
    #ylabel("SD (P)", fontsize=12, fontweight=:bold)
    #ylim(0, 0.6)

    #subplot(312)
    #plot(maxhold[:, 1], meanhold, color="black", linewidth=3)
    #ylabel("Mean (P)", fontsize=12, fontweight=:bold)
    #ylim(0, 1.4)

    #subplot(313)
    plot(maxhold[:, 1], percent_change_cv_high, color="black", linewidth=3)
    #plot(maxhold[:, 1], percent_change_cv_low, color="gray", linewidth=3)
    xlabel("Attack Rate (a)", fontsize=12, fontweight=:bold)
    ylabel("Percent Increase in CV", fontsize=12, fontweight=:bold)
    axvspan(0, 0.02047822, facecolor=myPal[1], alpha=0.5)
    axvspan(0.02047822, 0.03122348, facecolor=myPal[2], alpha=0.5)
    axvspan(0.03122348, 0.05, facecolor=myPal[3], alpha=0.5)

    #ylim(0, 1.75)

    tight_layout()

    # Save the figure
    #savefig("/Users/reillyoconnor/Desktop/Julia Projects/Slow-Fast/Figures/Figure 5 - CV vs rmax Global Change.pdf", dpi = 300)

    return CV_figure
end


/

let
    percent_increases = 0.0:0.01:0.10
    rmax_add = 0.11
    stab = fill(NaN, 2, length(percent_increases))

    # Initialize your parameters
    par = GenMod_Par(noise = 0.01)  # Make sure to initialize this with your base parameters

    for (i, perc_increase) in enumerate(percent_increases)
        #Set Seed
        Random.seed!(16)

        # Modify all attack rates by the percentage increase
        par.a_CR_1 = (par.a_CR_1 + rmax_add)
        par.a_CR_2 = (par.a_CR_2 + rmax_add)
        par.a_PC_1 = (par.a_PC_1 + rmax_add)
        par.a_PC_2 = (par.a_PC_2 + rmax_add)

        par.a_CR_1 *= (1 + perc_increase)
        par.a_CR_2 *= (1 + perc_increase)
        par.a_PC_1 *= (1 + perc_increase)
        par.a_PC_2 *= (1 + perc_increase)

        # Solve the ODE problem
        prob = SDEProblem(GenMod_model!, GenMod_stochmodel!, u0, tspan, par)
        sol = solve(prob)

        # Find the equilibrium state
        eq = nlsolve((du, u) -> GenMod_model!(du, u, par, 0.0), sol.u[end]).zero

        # Calculate and store the eigenvalues
        stab[1, i] = calc_λ1(eq, par)
        stab[2, i] = calc_λe(eq, par)

        # Reset the parameters to base values for next iteration
        par = GenMod_Par(noise = 0.01)  # Reset to base parameters
    end



# Plotting the results
GenMod_eq = figure()
subplot(211)
plot(percent_increases, stab[1, :])
axhline(0, color = "black", linestyle = "--")
ylabel(L"\lambda_1",fontsize=16,fontweight=:bold)

subplot(212)
plot(percent_increases, stab[2, :])
xlabel("Percentage Increase",fontsize=16,fontweight=:bold)
ylabel(L"\lambda_ imag",fontsize=16,fontweight=:bold)

tight_layout()

# Uncomment to save the figure
# savefig("/path/to/save/Eigenvalues.pdf", dpi = 300)

return GenMod_eq
end

/
/
/
/
/
/
/
/
#Calculate Synchrony between C1 & C2 after transient
    percent_increases = 0.0:0.01:0.10
    rmax_add = 0.11
    maxhold = fill(0.0, length(percent_increases), 1)
    synchold = fill(0.0, length(percent_increases), 1)
    varhold_C1 = fill(0.0, length(percent_increases), 1)
    varhold_C2 = fill(0.0, length(percent_increases), 1)
    varhold_tot = fill(0.0, length(percent_increases), 1)

    ts = range(0, 2000, length = 2000)  # Time steps
    t_span = (0.0, 2000.0)  # Time span

    # Initialize your parameters
    par = GenMod_Par()  # Make sure to initialize this with your base parameters

    for (i, perc_increase) in enumerate(percent_increases)
        #Set Seed

        Random.seed!(16)

        # Modify all attack rates by the percentage increase
        par.a_CR_1 = (par.a_CR_1 + rmax_add)
        par.a_CR_2 = (par.a_CR_2 + rmax_add)
        par.a_PC_1 = (par.a_PC_1 + rmax_add)
        par.a_PC_2 = (par.a_PC_2 + rmax_add)

        par.a_CR_1 *= (1 + perc_increase)
        par.a_CR_2 *= (1 + perc_increase)
        par.a_PC_1 *= (1 + perc_increase)
        par.a_PC_2 *= (1 + perc_increase)

        u0 = [1.0, 1.0, 0.5, 0.5, 0.25]
        prob_stoch = SDEProblem(GenMod_model!, GenMod_stochmodel!, u0, t_span, par)
        sol_stoch = solve(prob_stoch, reltol = 1e-15)
        grid_sol = sol_stoch(ts)

        # Recording the statistics
        maxhold[i] = percent_increases[i]
        varhold_C1[i] = var(grid_sol[4, 500:2000])
        varhold_C2[i] = var(grid_sol[3, 500:2000])
        varhold_tot[i] = varhold_C1[i] + varhold_C2[i]

        synchold[i] = (sqrt(varhold_C1[i]) + sqrt(varhold_C2[i]))/(sqrt(varhold_tot[i]))

        par = GenMod_Par() 
    end

    # Plotting
    sync_figure = figure(figsize = (8, 8))
    
    plot(maxhold[:, 1], synchold, color="black", linewidth=2)
    ylabel("Synchrony(C1:C2)", fontsize=12, fontweight=:bold)
    xlabel("Attack Rate (a)", fontsize=12, fontweight=:bold)
    #ylim(0, 0.6)

    return sync_figure



/
/
/
/
/
/
/
/
/









/

let
    percent_increases = 0.0:0.01:0.10
    rmax_add = 0.0
    maxhold = fill(0.0, length(percent_increases), 2)
    stdhold = fill(0.0, length(percent_increases), 1)
    meanhold = fill(0.0, length(percent_increases), 1)
    cvhold = fill(0.0, length(percent_increases), 1)
    taylorhold = fill(0.0, length(percent_increases), 1)
    
    ts = range(0, 2000, length = 2000)  # Time steps
    t_span = (0.0, 2000.0)  # Time span

    # Initialize your parameters
    par = GenMod_Par()  # Make sure to initialize this with your base parameters

    for (i, perc_increase) in enumerate(percent_increases)
        # Modify all attack rates by the percentage increase
       
        par.a_CR_1 = (par.a_CR_1 + rmax_add)
        par.a_CR_2 = (par.a_CR_2 + rmax_add)
        par.a_PC_1 = (par.a_PC_1 + rmax_add)
        par.a_PC_2 = (par.a_PC_2 + rmax_add)

        par.a_CR_1 *= (1 + perc_increase)
        par.a_CR_2 *= (1 + perc_increase)
        par.a_PC_1 *= (1 + perc_increase)
        par.a_PC_2 *= (1 + perc_increase)

        u0 = [1.0, 1.0, 0.5, 0.5, 0.25]
        prob_stoch = SDEProblem(GenMod_model!, GenMod_stochmodel!, u0, t_span, par)
        sol_stoch = solve(prob_stoch, reltol = 1e-15)
        grid_sol = sol_stoch(ts)

        # Recording the statistics
        maxhold[i, 1] = percent_increases[i]
        maxhold[i, 2] = maximum(grid_sol[4, 1900:2000])
        stdhold[i] = std(grid_sol[4, 1900:2000])
        meanhold[i] = mean(grid_sol[4, 1900:2000])
        cvhold[i] = stdhold[i] / meanhold[i]
        taylorhold[i] = (stdhold[i]^2) / meanhold[i]

        par = GenMod_Par() 
    end

    # Plotting
    CV_figure = figure(figsize = (8, 10))

    subplot(311)
    plot(maxhold[:, 1], stdhold, color="black", linewidth=2)
    ylabel("SD (C)", fontsize=12, fontweight=:bold)
    #ylim(0, 0.6)

    subplot(312)
    plot(maxhold[:, 1], meanhold, color="black", linewidth=3)
    ylabel("Mean (C)", fontsize=12, fontweight=:bold)
    #ylim(0, 1.4)

    subplot(313)
    plot(maxhold[:, 1], cvhold, color="black", linewidth=3)
    xlabel("Attack Rate (a)", fontsize=12, fontweight=:bold)
    ylabel("CV (C)", fontsize=12, fontweight=:bold)
    #ylim(0, 1.75)

    tight_layout()

    # Save the figure
    #savefig("/path/to/save/CR_CV_D00.pdf", dpi = 300)

    return CV_figure
end


# Bifurcation Diagram
    function minmaxbifurc(tend)
    a_range = 0.1:0.001:3.0
    minC = zeros(length(a_range))
    maxC = zeros(length(a_range))

    u0 = [0.5, 0.5]
    t_span = (0.0, tend)
    trans = tend - 100.0
    remove_transient = trans:1.0:tend

    for (ai, a_vals) in enumerate(a_range)
        p = CRPar(a = a_vals, D = 0.05)
        print(p)
        prob = ODEProblem(CR_mod!, u0, t_span, p)
        sol = DifferentialEquations.solve(prob, reltol = 1e-8)
        solend = sol(remove_transient)
        solend[2, 1:end]

        minC[ai] = minimum(solend[2, :])
        maxC[ai] = maximum(solend[2, :])
    end

    return hcat(a_range, minC, maxC)
end

let
    data = minmaxbifurc(1000.0)
    minmaxplot = figure()
    scatter(data[:,1], data[:,2], label="C1 min", marker="o", s=1)
    scatter(data[:,1], data[:,3], label="C1 min", marker="o", s=1)
    xlabel("Attack Rate (a)")
    ylabel("Consumer Max/Min")

    savefig("/Users/reillyoconnor/Desktop/Julia Projects/Slow-Fast/Figures/bifurcations_D005.pdf", dpi = 300)

    return minmaxplot
end


## Lets say we want the solutions at only certain time steps (just make sure it is inside of `t_span`! extrapolation is the road to sadness)

# now loop over increasing ae/m -- by increasing a 
# 
# loop over i but really subbing in new a parmaeters as amaxs
let 
    amaxs = 0.80:0.01:3.0
    maxhold = fill(0.0,length(amaxs),2)
    stdhold = fill(0.0,length(amaxs),1)
    meanhold = fill(0.0,length(amaxs),1)
    cvhold = fill(0.0,length(amaxs),1)
    taylorhold = fill(0.0,length(amaxs),1)
    
    ## Lets say we want the solutions at only certain time steps (just make sure it is inside of `t_span`! extrapolation is the road to sadness)
    ts = range(0, 1000, length = 1000)
    t_span = (0.0, 1000.0)

    for i = 1:length(amaxs)
        p = CRPar(a = amaxs[i], noise = 0.01, D = 0.0)
        print(p)
        u0 = [1.0, 0.5]
        prob_stoch = SDEProblem(CR_mod!, stoch_CR_Mod!, u0, t_span, p)
        sol_stoch = solve(prob_stoch, reltol = 1e-15)
        grid_sol = sol_stoch(ts)
        grid_sol.t
        grid_sol.u
        maxhold[i,1] = amaxs[i]
        maxhold[i,2] = maximum(grid_sol[2,900:1000])
        stdhold[i] = std(grid_sol[2,900:1000])
        meanhold[i] = mean(grid_sol[2,900:1000])
        cvhold[i] = stdhold[i]/meanhold[i]
        taylorhold[i]= (stdhold[i]^2)/meanhold[i]
    end

    CV_figure = figure(figsize = (8,10))

    subplot(311)
    plot(maxhold[1:length(amaxs),1],stdhold[1:length(amaxs)], color="black", linewidth=2)
    ylabel("SD (C)",fontsize=12,fontweight=:bold)
    ylim(0,0.6)


    subplot(312)
    plot(maxhold[1:length(amaxs),1],meanhold[1:length(amaxs)], color="black", linewidth=3)
    ylabel("Mean (C)",fontsize=12,fontweight=:bold)
    ylim(0, 1.4)

    subplot(313)
    plot(maxhold[1:length(amaxs),1],cvhold[1:length(amaxs)], color="black", linewidth=3)
    xlabel("Attack Rate (a)",fontsize=12,fontweight=:bold)
    ylabel("CV (C)",fontsize=12,fontweight=:bold)
    ylim(0, 1.75)

    #subplot(414)
    #plot(maxhold[1:length(amaxs),1],taylorhold[1:length(amaxs)])
    #xlabel("Attack Rate (a)",fontsize=16,fontweight=:bold)
    #ylabel("Taylors Power Law (C)",fontsize=16,fontweight=:bold)

    tight_layout()

    savefig("/Users/reillyoconnor/Desktop/Julia Projects/Slow-Fast/Figures/Attack Rate (a)/CR_CV_D00.pdf", dpi = 300)

    return CV_figure
end


























## Plotting time series with noise 


    u0 = [0.5, 0.50, 0.30, 0.30, 0.150]
    t_span = (0.0, 10000.0)
    p = AdaptPar(noise =0.0)

    prob_stoch = SDEProblem(adapt_model!, stoch_adapt!, u0, t_span, p)
    sol_stoch = solve(prob_stoch, reltol = 1e-15)
    
    ## Lets say we want the solutions at only certain time steps (just make sure it is inside of `t_span`! extrapolation is the road to sadness)
    ts2 = range(0, 10000, length = 10000)
    ts = range(0, 10000, length = 10000)
    ## Then we just need to *call* the solution object
  
    grid_sol = sol_stoch(ts)
    grid_sol.t
    grid_sol.u
    plot(grid_sol.u)
    
    plot(grid_sol[5,1:1000])

    xlabel("time")
    ylabel("Density")
    legend(["P"])
    xlim(9000,10000) 

    println(grid_sol.u) 
 
    # for check of ode in more complex, do lower dimensional checks

# amax=1/h, Ro=1/(ah)
# slow CR
r= 1.0
a= 3.50
m= 0.60
h=0.90

# fast CR
r = 1.0
a = 3.250
m =0.70
h =.6
e=0.80
K=1.0


    Rhump = K - K/(a * h)
    Rhold = m/(e * a - m * a * h)
    Chold = r/a*(1-Rhold/K)*(1 + a * h * Rhold)
     

# having seen the time series and checked that the ode works at cases with clear answers we now proceed to 
# do CV calculations over a range of weak/slow 2 to symetrically fast/strong both channels 
# We are increasing all the parms of the slow channel until the symmetry point
Fast_mult = 0.0:0.01:1.0


stdhold = fill(0.0,length(Fast_mult),1)
meanhold = fill(0.0,length(Fast_mult),1)
cvhold = fill(0.0,length(Fast_mult),1)

stdhold34 = fill(0.0,length(Fast_mult),1)
meanhold34 = fill(0.0,length(Fast_mult),1)
cvhold34 = fill(0.0,length(Fast_mult),1)

print(Fast_mult[1])

# initial slow parm values
a_CRl_i =  3.25 
a_PCl_i = 2.025
h_CRl_i = .60

for i=1:length(Fast_mult)
u0 = [0.5, 0.50, 0.3, 0.30, 0.30]
t_span = (0.0, 10000.0)

p = AdaptPar(a_CR_2=a_CRl_i+Fast_mult[i]*(7.50-a_CRl_i), a_PC_2=a_PCl_i+Fast_mult[i]*(6.5-a_PCl_i), h_CRl=h_CRl_i-Fast_mult[i]*(h_CRl_i-0.250), noise=0.02)
#p = AdaptPar(noise =0.01)

prob_stoch = SDEProblem(adapt_model!, stoch_adapt!, u0, t_span, p)
sol_stoch = solve(prob_stoch, reltol = 1e-15)
print(i)

## Lets say we want the solutions at only certain time steps (just make sure it is inside of `t_span`! extrapolation is the road to sadness)
ts2 = range(0, 10000, length = 10000)
ts = range(0, 10000, length = 10000)
## Then we just need to *call* the solution object at time steps of 1 for time series work
grid_sol = sol_stoch(ts)

# now do CV calcs
    stdhold[i]=std(grid_sol[5,5000:10000])
    meanhold[i]=mean(grid_sol[5,5000:10000])
    cvhold[i] = stdhold[i]/meanhold[i]

    stdhold34[i]=std(grid_sol[4,5000:10000])
    meanhold34[i]=mean(grid_sol[4,5000:10000])
    cvhold34[i] = stdhold34[i]/meanhold34[i]


end

plot(stdhold,linewidth=4)
xlabel("Slow to Fast",fontsize=16,fontweight=:bold)
ylabel("Std Dev",fontsize=16,fontweight=:bold)

plot(meanhold,linewidth=4)
xlabel("Slow to Fast",fontsize=16,fontweight=:bold)
ylabel("Mean",fontsize=16,fontweight=:bold)
legend(["P"],fontsize=16,fontweight=:bold)

plot(cvhold,linewidth=4)
xlabel("Slow to Fast",fontsize=16,fontweight=:bold)
ylabel("CV",fontsize=16,fontweight=:bold)
legend(["P"],fontsize=16,fontweight=:bold)

plot(cvhold34)
plot(stdhold34)


