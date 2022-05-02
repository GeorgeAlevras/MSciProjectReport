import numpy as np


def sirhd_odes(t, x, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, g, h, d):
    # Compartments
    S = x[0]  # Susceptible
    I = x[1]  # Infected
    R = x[2]  # Recovered
    H = x[3]  # Hospitalised
    D = x[4]  # Deceased
    
    N = np.sum([S, I, R, H])  # Dynamic population (excludes deceased people)
    
    bs = 14*[b1] + 14*[b2] + 14*[b3] + 14*[b4] + 14*[b5] + 14*[b6] + 14*[b7] + 14*[b8] + 14*[b9] + [b10, b10]
    
    b = bs[int(t)]

    # Model: system of DEs
    dSdt = -(b/N)*I*S
    dIdt = (b/N)*I*S - h*I - g*I
    dRdt = g*I
    dHdt = h*I - d*H
    dDdt = d*H

    return [dSdt, dIdt, dRdt, dHdt, dDdt]


def sirhd_odes_noise(t, x, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, g, h, d, stochastic_noise_f=0):
    # Compartments
    S = x[0]  # Susceptible
    I = x[1]  # Infected
    R = x[2]  # Recovered
    H = x[3]  # Hospitalised
    D = x[4]  # Deceased
    
    N = np.sum([S, I, R, H])  # Dynamic population (excludes deceased people)
    
    # Introducing stochastic noise to the system of DEs by adding random noise to parameters in each time-step
    g = abs(np.random.normal(g, stochastic_noise_f*g))
    h = abs(np.random.normal(h, stochastic_noise_f*h))
    d = abs(np.random.normal(d, stochastic_noise_f*d))

    bs = 14*[b1] + 14*[b2] + 14*[b3] + 14*[b4] + 14*[b5] + 14*[b6] + 14*[b7] + 14*[b8] + 14*[b9] + [b10]
    bs = abs(np.random.normal(bs, stochastic_noise_f*bs))

    b = bs[int(t)]

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
