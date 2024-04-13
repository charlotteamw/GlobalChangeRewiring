using Parameters: @with_kw, @unpack
using DifferentialEquations
using PyPlot
using Plots

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
    amp_K1 = 0.1
    amp_K2 = 0.1
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
    du[1] = r1 * R1 * (1 - R1 / K1_modulated) - intake_R1
    du[2] = r2 * R2 * (1 - R2 / K2_modulated) - intake_R2
    du[3] = (e_R1C * intake_R1 + e_R2C * intake_R2) - m_C * C

end

function sine_model(u, Params, t)
    du = similar(u)
    sine_model!(du, u, Params, t)
    return du
end

let
    u0 = [0.8, 0.8, 0.5]  # Initial conditions for R1, R2, C
    t_span = (0.0, 100.0)
    p = Params()
    prob = ODEProblem(sine_model!, u0, t_span, p)
    sol = solve(prob)
    Plots.plot(sol, xlabel="Time", ylabel="Density", title="Three Species Model Dynamics with Forcing")
end


