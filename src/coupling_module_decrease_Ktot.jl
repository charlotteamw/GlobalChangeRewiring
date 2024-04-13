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

function λ2_stability(M)
    evals = eigvals(M)  # Get all eigenvalues
    real_evals = filter(e -> imag(e) == 0, evals)  # Filter out purely real eigenvalues
    if length(real_evals) < 2
        return NaN  # Return NaN if there aren't at least two real eigenvalues
    else
        sorted_real_evals = sort(real.(real_evals), rev=true)
        return sorted_real_evals[2]  # Return the second greatest real eigenvalue
    end
end

function λ3_stability(M)
    evals = eigvals(M)  # Get all eigenvalues
    real_evals = filter(e -> imag(e) == 0, evals)  # Filter out purely real eigenvalues
    if length(real_evals) < 3
        return NaN  # Return NaN if there aren't at least three real eigenvalues
    else
        sorted_real_evals = sort(real.(real_evals), rev=true)
        return sorted_real_evals[3]  # Return the third greatest real eigenvalue
    end
end

function jacobian_at_u0(K2_val, K2_start, K2_end, K1_base, u0, p)
    # Calculate the fraction of the range covered by the current K2_val
    fraction_of_range = (K2_val - K2_start) / (K2_end - K2_start)
    
    # Calculate the current decrease to be applied based on the fraction of the total decrease 
    current_decrease = 1.15 * fraction_of_range
    
    # Adjust K1 based on the current decrease, ensuring total K decreases gradually across the K2 range
    p.K1 = K1_base - current_decrease  # Subtract only the current decrease from K1_base
    p.K2 = K2_val  # Update K2 value to current
    
    # Compute and return the Jacobian matrix at u0 with the updated parameters
    return cmat(u0, p)
end

#plot max real eigenvalues

function plot_max_real_eigenvalues(K2_range, K1_base, u0, p)
    K2_start = first(K2_range)  # Extract the start of K2 range
    K2_end = last(K2_range)  # Extract the end of K2 range
    max_real_eigenvalues = Float64[]

    for K2_val in K2_range
        # Pass the start and end of K2 range to jacobian_at_u0
        jac = jacobian_at_u0(K2_val, K2_start, K2_end, K1_base, u0, p)
        push!(max_real_eigenvalues, λ1_stability(jac))
    end

    stability_plot = figure()
    PyPlot.plot(K2_range, max_real_eigenvalues, color="black", linewidth=3)
    xlabel("K2 values")  # Reflect the focus on K2 values
    ylabel("Re(λ) max")

    # Invert the y-axis to have more negative values increase along the y-axis
    PyPlot.gca().invert_yaxis()

    return stability_plot
end


K2_range = 0.6:0.0001:1.4  # Define the range of K2 values
K1_base = 1.2  # Define the base value for K1
u0 = [0.8, 0.4, 0.8, 0.4, 0.3]
p = ModelPar()  # Assuming the parameter structure is defined
plot_max_real_eigenvalues(K2_range, K1_base, u0, p)

function plot_complex_part_eigenvalues(K2_range, K1_base, u0, p)
    K2_start = first(K2_range)  # Extract the start of K2 range
    K2_end = last(K2_range)  # Extract the end of K2 range
    complex_part_values = Float64[]

    for K2_val in K2_range
        # Pass the start and end of K2 range to jacobian_at_u0
        jac = jacobian_at_u0(K2_val, K2_start, K2_end, K1_base, u0, p)
        eigenvalues = eigvals(jac)
        # Extract the imaginary parts, take their absolute values (to get 'b'), and find the max
        max_imag_part = maximum(abs.(imag.(eigenvalues)))
        push!(complex_part_values, max_imag_part)
    end

    complex_plot = figure()
    PyPlot.plot(K2_range, complex_part_values, color="red", linewidth=3)
    xlabel("K2 values")
    ylabel("|Im(λ)| max")

    # No need to invert the y-axis here, but adjust based on your data's characteristics
    return complex_plot
end

# Then, you can call this function similar to how you've called the others
plot_complex_part_eigenvalues(K2_range, K1_base, u0, p)


function plot_second_greatest_real_eigenvalues(K2_range, K1_base, u0, p)
    K2_start = first(K2_range)  # Extract the start of K2 range
    K2_end = last(K2_range)  # Extract the end of K2 range
    second_greatest_real_eigenvalues = Float64[]

    for K2_val in K2_range
        # Pass the start and end of K2 range to jacobian_at_u0
        jac = jacobian_at_u0(K2_val, K2_start, K2_end, K1_base, u0, p)
        push!(second_greatest_real_eigenvalues, λ2_stability(jac))
    end

    stability_plot = figure()
    PyPlot.plot(K2_range, second_greatest_real_eigenvalues, color="blue", linewidth=3)
    xlabel("K2 values")
    ylabel("2nd Re(λ) max")

    # Invert the y-axis to have more negative values increase along the y-axis, if needed
    PyPlot.gca().invert_yaxis()

    return stability_plot
end



plot_second_greatest_real_eigenvalues(K2_range, K1_base, u0, p)



function plot_third_greatest_real_eigenvalues(K2_range, K1_base, u0, p)
    K2_start = first(K2_range)  # Extract the start of K2 range
    K2_end = last(K2_range)  # Extract the end of K2 range
    third_greatest_real_eigenvalues = Float64[]

    for K2_val in K2_range
        # Pass the start and end of K2 range to jacobian_at_u0
        jac = jacobian_at_u0(K2_val, K2_start, K2_end, K1_base, u0, p)
        push!(third_greatest_real_eigenvalues, λ3_stability(jac))
    end

    stability_plot = figure()
    PyPlot.plot(K2_range, third_greatest_real_eigenvalues, color="green", linewidth=3)
    xlabel("K2 values")
    ylabel("3rd Re(λ) max")

    # Invert the y-axis to have more negative values increase along the y-axis, if needed
    PyPlot.gca().invert_yaxis()

    return stability_plot
end

plot_third_greatest_real_eigenvalues(K2_range, K1_base, u0, p)


