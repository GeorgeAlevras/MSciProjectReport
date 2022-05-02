import numpy as np


def sirhd_odes(t, x, b, g, h, d):
    # Compartments
    S = x[0]  # Susceptible
    I = x[1]  # Infected
    R = x[2]  # Recovered
    H = x[3]  # Hospitalised
    D = x[4]  # Deceased
    
    N = np.sum([S, I, R, H])  # Dynamic population (excludes deceased people)
    
    # Model: system of DEs
    dSdt = -(b/N)*I*S
    dIdt = (b/N)*I*S - h*I - g*I
    dRdt = g*I
    dHdt = h*I - d*H
    dDdt = d*H

    return [dSdt, dIdt, dRdt, dHdt, dDdt]


def sirhd_odes_noise(t, x, b, g, h, d, stochastic_noise_f=0):
    # Compartments
    S = x[0]  # Susceptible
    I = x[1]  # Infected
    R = x[2]  # Recovered
    H = x[3]  # Hospitalised
    D = x[4]  # Deceased
    
    N = np.sum([S, I, R, H])  # Dynamic population (excludes deceased people)
    
    # Introducing stochastic noise to the system of DEs by adding random noise to parameters in each time-step
    b = abs(np.random.normal(b, stochastic_noise_f*b))
    g = abs(np.random.normal(g, stochastic_noise_f*g))
    h = abs(np.random.normal(h, stochastic_noise_f*h))
    d = abs(np.random.normal(d, stochastic_noise_f*d))

    # Model: system of DEs
    dSdt = -(b/N)*I*S
    dIdt = (b/N)*I*S - h*I - g*I
    dRdt = g*I
    dHdt = h*I - d*H
    dDdt = d*H

    return [dSdt, dIdt, dRdt, dHdt, dDdt]


models = {
    'sirhd':sirhd_odes,
    'sirhd_noise':sirhd_odes_noise
}
