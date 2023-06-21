import numpy as np

# units, choose to work with fermis in fitting
fm = 1.0
barn = 100.0 * fm**2
mbarn = 10.0**(-3) * barn
inv_GeV = 0.19733 * fm
GeV = 1.0 / inv_GeV
keV = 10**(-6) * GeV
MeV = 10**(-3) * GeV

# convert fm^2 to barn for cross section conversion
fm_sq = 1.0 / 100.0 # barn

# np parameters
inv_m_p_sq = 0.44232 * mbarn    # inverse proton mass squared
m_p = np.sqrt(1.0 / inv_m_p_sq) # proton mass
m_n = m_p                       # neutron mass
m_N = m_p                       # nucleon mass

# cm momentum squared
def qcm_sq(Ecm_sq):
    return 0.25 * ( Ecm_sq - 4.0 * m_N**2 )

# cm energy squared
def Ecm_sq(Tn):
    return 2.0 * m_N**2 + 2.0 * m_N * (m_N + Tn)

# qcot(delta) effective range expansion
def qcotd(q_sq, a, r):
    return -1.0 / a + 0.5 * r * q_sq

# cross section scalar scattering
def cross_section(q_sq, a, r):
    return 4.0 * np.pi / (qcotd(q_sq, a, r)**2 + q_sq)

# np cross section including spin degeneracy
def np_cross_section(q_sq, a_t, a_s, r_t, r_s):
    sig_s = cross_section(q_sq, a_s, r_s)
    sig_t = cross_section(q_sq, a_t, r_t)
    return 0.75 * sig_t + 0.25 * sig_s

# binding momentum given scattering length and effetive range
def binding_momentum(a, r):
    return ( 1.0 - np.sqrt(1.0 - 2.0 * r / a) ) / r

# binding energy given scattering length and effetive range
def binding_energy(a, r):
    kappa = binding_momentum(a, r)
    be = 2.0 * m_N -2.0 * np.sqrt(m_N**2 - kappa**2)
    return be / MeV # convert to MeV

def deriv_kappa_a(a, r):
    tmp = np.sqrt(1.0 - 2.0 * r / a)
    return -1.0 / (a**2 * tmp)

def deriv_kappa_r(a, r):
    tmp = np.sqrt(1.0 - 2.0 * r / a)
    num = a - r - a * tmp
    den = a * r**2 * tmp
    return num / den

def error_binding_momentum(a, da, r, dr, cov_ar):
    dkda = deriv_kappa_a(a, r)
    dkdr = deriv_kappa_r(a, r)
    kappa_err_sq = dkda**2 * da**2 + dkdr**2 * dr**2 + 2.0 * dkda * dkdr * cov_ar
    return np.sqrt(kappa_err_sq)

def error_binding_energy(a, da, r, dr, cov_ar):
    kappa     = binding_momentum(a, r)
    kappa_err = error_binding_momentum(a, da, r, dr, cov_ar)
    return 2.0 * kappa * kappa_err / np.sqrt(m_N**2 - kappa**2)