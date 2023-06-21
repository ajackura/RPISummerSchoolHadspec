import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt

from np_amplitudes import *
from np_fit_util import *

# reads data and stores in arrays
def read_np_data():    
    # files to import
    files = [
        "np_data/np_zero_energy_cross_section.dat",
        "np_data/np_daub_2013.dat", 
        "np_data/np_koester_1990.dat", 
        "np_data/np_houk_1971.dat", 
        "np_data/np_kirilyuk_1987.dat", 
        "np_data/np_larson_1980.dat"]

    # load data
    Ene_data  = []  # neutron lab frame kinetic energy (keV)
    Sig_data  = []  # total cross section (b)
    dSig_data = []  # error in total cross section (b)

    for file in files:
        print('reading: ', file)
        data = np.loadtxt(open(file,"r"))
        Ene_data.append(data[:,0])
        Sig_data.append(data[:,1])
        dSig_data.append(data[:,2])
    
    Ene_data = np.concatenate( Ene_data, axis=0 )
    Sig_data = np.concatenate( Sig_data, axis=0 )
    dSig_data = np.concatenate( dSig_data, axis=0 )

    return Ene_data, Sig_data, dSig_data

# fit model, Tn is neutron kinetic energy
def fit_model(Tn, a_t, a_s, r_t, r_s):
    E_sq = Ecm_sq(Tn * keV) # convert kinetic energies to Ecm^2, unit converts to fm implicitly
    q_sq = qcm_sq(E_sq)     # convert to qcm^2
    return np_cross_section(q_sq, a_t, a_s, r_t, r_s) * fm_sq # unit converts fm^2 to barns for fit

# chi^2 function
# x is neutron kinetic energy data
# y is cross section data
# yerr is error in cross section data
def chi_sq_func(param, x, y, yerr):
    # param is parameter array
    a_t = param[0]  # triplet scattering length
    a_s = param[1]  # singlet scattering length
    r_t = param[2]  # triplet effective range
    r_s = param[3]  # singlet effective range

    # set fit model array
    model = fit_model(x, a_t, a_s, r_t, r_s)

    # sum chi^2 functions
    chi_sq = np.sum( ( (y - model) / yerr )**2 )
    return chi_sq

# run fitting routine
# x0 is initial parameter guess
# x is neutron kinetic energy data
# y is cross section data
# yerr is error in cross section data
def run_fit(x0, x, y, yerr):

    # minimize chi^2
    result =  opt.minimize(chi_sq_func, x0, args = (x, y, yerr), tol=1e-6, options={'disp': True})
    covariance = result.hess_inv

    # extract results
    a_t, a_s, r_t, r_s = result.x
    p_err = param_errors(covariance)
    da_t = p_err[0]
    da_s = p_err[1]
    dr_t = p_err[2]
    dr_s = p_err[3]
    cov_ar_t = covariance[0][2] # covariance between a_t and r_t
    chi_sq_per_dof = result.fun / (len(y) - len(x0))
    print("\nfit results")
    print("\nn_data = ", x.size, "\n")
    print("chi^2 / dof = ", num_to_str(chi_sq_per_dof))
    print("a_t = ", num_to_str(a_t), " +/- ", num_to_str(da_t))
    print("a_s = ", num_to_str(a_s), " +/- ", num_to_str(da_s))
    print("r_t = ", num_to_str(r_t), " +/- ", num_to_str(dr_t))
    print("r_s = ", num_to_str(r_s), " +/- ", num_to_str(dr_s))

    print("\ncorrelation matrix")
    pretty_print_matrix(correlation_from_covariance(covariance))
    
    be = binding_energy(a_t, r_t)
    be_err = error_binding_energy(a_t, da_t, r_t, dr_t, cov_ar_t)
    print("\ndeuteron binding energy: ", num_to_str(be), " +/- ", num_to_str(be_err))

    # plot result of fit to data
    Tn_arr = np.logspace(-2, 5, 500)
    sig_arr = fit_model(Tn_arr, a_t, a_s, r_t, r_s) 
    plt.xscale("log")
    plt.plot(Tn_arr, sig_arr)
    plt.errorbar(x, y, yerr, ls='none', color = "r", elinewidth = 5)
    plt.show()

    return

# main function
def main():
    # read data arrays 
    Ene_data, Sig_data, dSig_data = read_np_data()

    # initial guess of parameters
    x0 = np.array([5.411, -23.71, 1.7, 2.7])

    # run the fit
    run_fit(x0, Ene_data, Sig_data, dSig_data)
    return

# main program
if __name__ == '__main__':
    main()