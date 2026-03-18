def update_pitch(theta_pitch, thetaI_old, omeg, omega_ref, KK, KP, KI, dt, theta_min, theta_max):
    GK = 1/(1+theta_pitch/KK)
    thetaP = GK*KP*(omeg-omega_ref)
    thetaI = thetaI_old + GK*KI*(omeg-omega_ref)*dt
    thetaSP = thetaP + thetaI

    thetaI = max(thetaI, theta_min)
    thetaI = min(thetaI, theta_max)
    thetaSP = max(thetaSP, theta_min)
    thetaSP = min(thetaSP, theta_max)

    return thetaI, thetaSP

def calculate_MG(omega, P_rated, K, omega_rated):
    if omega<omega_rated:
        MG = K*omega**2
    else:
        MG = P_rated/omega_rated

    return MG

def update_omega(omega, MG, dt, M_aero, I_rotor):
    
    omega_next = omega + (M_aero-MG)/I_rotor*dt

    return omega_next