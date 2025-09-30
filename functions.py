#!/usr/bin/env python

import numpy as np
import math
import pickle
from scipy.optimize import minimize


import feo_thermodynamics
import feo_thermodynamics as feot

def mole2mass(xO):
    mFe = 56. * (1. - xO)
    mO  = 16. * xO
    return mO / (mFe+mO)

def mass(ri):
    rho_ic_poly  = np.array([13088.5, 0,              -8838.1/(6371e3)**2])
    rho_oc_poly = np.array([12581.5, -1263.8/6371e3, -3642.6/(6371e3)**2, -5528.1/(6371e3)**3])
    
    r_cmb = 3480e3
    rs = r_cmb
    #Integrate density polynomials to get mass polynomials
    poly_ic = np.zeros(rho_ic_poly.size + 3)
    poly_oc = np.zeros(rho_oc_poly.size + 3)

    for i in range(3, rho_ic_poly.size+3):
        poly_ic[i] = rho_ic_poly[i-3]/i

    for i in range(3, rho_oc_poly.size+3):
        poly_oc[i] = rho_oc_poly[i-3]/i


    M_ic   = 4*np.pi*np.polyval(poly_ic[::-1],ri)
    M_conv = 4*np.pi*(np.polyval(poly_oc[::-1],rs) - np.polyval(poly_oc[::-1],ri))
    Ms     = 4*np.pi*(np.polyval(poly_oc[::-1],r_cmb) - np.polyval(poly_oc[::-1],rs))

    M = M_ic + M_conv + Ms

    return M_ic, M_conv, Ms, M
core_mass = mass(1221e3)[3]

def delta_mu(x_l,P,T):
    g_mixture_t, g_fe_t, g_feo_t = feot.solid_free_energies(x_l, P, T)
    return feot.liquid_free_energy(x_l, P, T) - g_mixture_t

def dr_dt(k0,x_l,P,T):
    R = 8.314
    return k0 * (1. - np.exp(-delta_mu(x_l,P,T) / (R*T)))

def density(r):
    core_solid_density_params  = [13088.5, 0,              -8838.1/(6371e3)**2]                       # Inner core density polynomials (radial). List(float)
    #core_liquid_density_params = [12581.5, -1263.8/6371e3, -3642.6/(6371e3)**2, -5528.1/(6371e3)**3]  # Outer core density polynomials (radial). List(float)
    rho_s = np.polyval(core_solid_density_params[::-1], r)

    return rho_s

def Latent(r_existing,r_next):
    L = 750. * 1e3 # 750 kJ/kg
    Ql = 0.
    
    r = np.linspace(r_existing,r_next,100)
    
    for i in range(len(r)-1):
        dV = 4./3.*np.pi*r[i+1]**3. - 4./3.*np.pi*r[i]**3.
        rho = (density(r[i]) + density(r[i+1])) / 2.
        Ql += dV * rho * L # I only want J, not J/s or J/s/m^2
    return Ql

def O_mol(ri,O_kg):
    ic_mass = mass(ri)[0]
    oc_mass = core_mass - ic_mass
    Fe_kg = oc_mass - O_kg 
    n_Fe = Fe_kg / (56.*1.66054e-27)
    n_O = O_kg / (16.*1.66054e-27)
    return n_O / (n_Fe+n_O)

def P_interp(rt):
    return np.interp(rt,rc,pc)


def deltaG(xx, PP, TT):
    gl = feo_thermodynamics.liquid_free_energy(xx, PP, TT)
    gs = feo_thermodynamics.solid_free_energies(xx, PP, TT)
    return gs[0] - gl

def growth_rate(x, P, T, m, c):
    R = 8.314
    gsl = deltaG(x,P,T)
    return (m*T+c) * (1. - np.exp((gsl)/(R*T)))
    
def CoE_melting_temp(x):
    return P_ref_melting_temp(x,360.)

def P_ref_melting_temp(x,P):
    T = np.linspace(4000.,6500.,100)
    
    g_sl = []
    for TT in T:
        g_sl.append(deltaG(np.array([x]),P,TT))
    g_sl = np.array(g_sl)

    c = np.where(g_sl>0.)[0]
    g_sel, t_sel = g_sl[c], T[c]
    c = np.where(g_sel==min(g_sel))[0][0]
    x0, y0 = t_sel[c], g_sel[c] 

    c = np.where(g_sl<0.)[0]
    g_sel, t_sel = g_sl[c], T[c]
    c = np.where(g_sel==max(g_sel))[0][0]
    x1, y1 = t_sel[c], g_sel[c] 
    
    out = x0 - (((x1-x0) / (y1-y0)) * y0)
    return out[0]

def shell(r0,r1):
    return 4./3.*np.pi*r1**3. - 4./3.*np.pi*r0**3.

def generate_radial_points(inner_core_radius):
    inc = 0.1
    inc_mult = 1.5

    x = inc
    intervals = []
    while x < 1221e3:
        intervals.append(x)
        x += inc
        inc = inc * inc_mult

    intervals = np.array(sorted(intervals + [0.,-1221e3,1221e3] + list(-np.array(intervals))))

    bins = ri+intervals
    sel = np.where((bins>=0.)&(bins<=1221e3))
    bins = bins[sel]
    
    return bins


def get_Tm(PP,xx):
    def obj(T_arg):
        return abs(deltaG(xx,PP,T_arg))
    initial_guess = [5500.]  
    result = minimize(obj, initial_guess)
    return result.x[0]



def pre_stage(x_l,dT):
    data = pickle.load(open('test_profiles.pkl', 'rb')) 
    adiabat_params    = [1,  0, -2.52083167e-14,  7.75340296e-22]  

    i_r, i_Ta, i_P, i_rho, i_g, i_Tm = 0,1,2,3,4,5

    sTm, pc = np.array(data[0][1][i_Tm]), np.array(data[0][1][i_P])
    rc, rhoc = np.array(data[0][1][i_r]), np.array(data[0][1][i_rho])

    rf  = np.arange(0.,3480e3,1.)
    P   = np.interp(rf,rc,pc)
    rho = np.interp(rf,rc,rhoc)
    r   = rf

    Pi = P[0]/1e9
    Ti = 5000.
    interval = 500.
    direction = -1.

    switches = 0
    condition = 1e-5

    while switches < 1000:
        rslt = delta_mu(x_l,Pi,Ti)

        old_direction = direction

        direction = 1.
        if rslt < 0.:
            direction = -1.

        if direction != old_direction:
            switches += 1
            interval = interval / 2.

        Ti = Ti + direction * interval
        if abs(rslt) < condition:
            break
    
    T_ad = Ti*np.polyval(adiabat_params[::-1], r)
    T_ad = T_ad - dT

    
    return [r,T_ad,P,rho]


def itterate(inputs):

    t_in,ri_in,r_in,P_in,T_in,x_in,T_adiabat_in,m,c,dt_growth,dr,alphaT_l,alphaT_s,alphaC_l,alphaC_s,L,Cp,n_comp_points,TmFe_in,rho_in = inputs
    
    t, ri, new_ri = t_in, ri_in, ri_in
    r,P,T,x,rho = r_in.copy(), P_in.copy(), T_in.copy(), x_in.copy(), rho_in.copy()

    sx, sP, sT = np.interp(ri,r,x), np.interp(ri,r,P)/1e9, np.interp(ri,r,T)
    grow = growth_rate(sx, sP, sT, m, c) * dt_growth

    if grow >= dr:
        new_ri = ri + dr*float(int(grow/dr))
        # Latent heat
        nir = [np.where(abs(r-ri)==min(abs(r-ri)))[0][0],np.where(abs(r-new_ri)==min(abs(r-new_ri)))[0][0]]
        freeze_mass = 0.
        for ii in np.arange(nir[0],nir[1],1):
            freeze_mass += rho[ii] * 4./3.*np.pi*(r[ii+1]**3.-r[ii]**3.) 
        Ql = L*freeze_mass

        diffusion_length_l = (alphaT_l*dt_growth)**0.5
        diffusion_length_s = (alphaT_s*dt_growth)**0.5

        heated_radius = [ri-diffusion_length_s,new_ri+diffusion_length_l]
        if heated_radius[0] < 0.:
            heated_radius[0] = 0.
        nir = [np.where(abs(r-heated_radius[0])==min(abs(r-heated_radius[0])))[0][0],
               np.where(abs(r-heated_radius[1])==min(abs(r-heated_radius[1])))[0][0]]
        heated_mass = 0.
        for ii in np.arange(nir[0],nir[1],1):
                heated_mass += rho[ii] * 4./3.*np.pi*(r[ii+1]**3.-r[ii]**3.) 

        additional_heat = Ql / (heated_mass * Cp)

        T[nir[0]:nir[1]] += additional_heat

        # Oxygen partitioning
        chem_limit = new_ri + (alphaC_l*dt_growth)**0.5

        # Define compositional space as solute points
        xO = 1.-x
        x_solute = xO * n_comp_points
        
        # Remove all solute points from recently frozen volume
        nir = [np.where(abs(r-ri)==min(abs(r-ri)))[0][0],np.where(abs(r-new_ri)==min(abs(r-new_ri)))[0][0]]
        O_to_partition = sum(x_solute[nir[0]:nir[1]])

        x_solute[nir[0]:nir[1]] = 0.
        
        # Distribute all displaced solute points across chemical diffusion layer   
        nir[0] = nir[1]
        nir[1] = np.where(abs(r-chem_limit)==min(abs(r-chem_limit)))[0][0]
        
        x_solute[nir[0]:nir[1]] += O_to_partition / float(len(x_solute[nir[0]:nir[1]]))
        
        # Convert back to fractional comp (Fe)    
        x_new = 1. - (x_solute / n_comp_points)
        x = x_new.copy()
        
        # Update ri, time, record
        ri = new_ri
        t += dt_growth

    else:
        portions = np.zeros(3)
        dt_diffuse = min(np.gradient(r))**2./(2.*max([alphaT_l,alphaC_l]))/4.
        
        T_ = T - T_adiabat_in
        T_ref = T_.copy()
        
        # Define compositional space as solute points
        xO = 1.-x
        x_solute = xO * n_comp_points
        
        n_it = 1

        for it in range(n_it):
            T_new = T_.copy()
            xs_new = x_solute.copy()
            for iir in range(len(r)-2):
                ir = iir + 1
                
                # Heat diffusion
                portions[0] = alphaT_l * dt_diffuse / (r[ir-1]-r[ir])**2. * T_[ir-1]
                portions[1] = alphaT_l * dt_diffuse / ((r[ir-1]-r[ir+1])/2.)**2. * -2.*T_[ir]
                portions[2] = alphaT_l * dt_diffuse / (r[ir+1]-r[ir])**2. *  T_[ir+1]

                T_new[ir-1] = T_new[ir-1] - portions[0] - 0.5 * portions[1]
                T_new[ir]   = T_new[ir]   + portions[0] + portions[1] + portions[2]
                T_new[ir+1] = T_new[ir+1] - portions[2] - 0.5 * portions[1]
                
                # Oxygen diffusion
                if r[ir] > new_ri:
                    portions[0] = alphaC_l * dt_diffuse / (r[ir-1]-r[ir])**2. * x_solute[ir-1]
                    portions[1] = alphaC_l * dt_diffuse / ((r[ir-1]-r[ir+1])/2.)**2. * -2.*x_solute[ir]
                    portions[2] = alphaC_l * dt_diffuse / (r[ir+1]-r[ir])**2. *  x_solute[ir+1]

                    criteria = np.array([xs_new[ir-1] - portions[0] - 0.5 * portions[1],
                                         xs_new[ir]   + portions[0] + portions[1] + portions[2],
                                         xs_new[ir+1] - portions[2] - 0.5 * portions[1]])

                    if len(np.where(criteria<0)[0])==0 and len(np.where(criteria>n_comp_points)[0])==0:
                        xs_new[ir-1] = criteria[0]
                        xs_new[ir]   = criteria[1]
                        xs_new[ir+1] = criteria[2]
                    
            T_ = T_new.copy()
            x_solute = xs_new.copy()
            
        sel = np.where(T>TmFe)[0]
        T[sel] = TmFe[sel]

        T = T_ + T_adiabat
        x = 1.-(x_solute / n_comp_points)
        t += dt_diffuse

    return [t,ri,r,T,x]

def initialize(xO,dT,r_lim,n_comp_points,r_resolution):
    r, T, P, rho = pre_stage(1.-xO,dT)
    
    x_solute = (xO*n_comp_points)*np.ones(len(r))
    x = 1.-x_solute/n_comp_points
    s = np.where(r<r_lim)[0]
    T, P, r, x = T[s], P[s], r[s], x[s]

    TmFe = np.zeros(len(r))
    for ii in range(len(r)):
        TmFe[ii] = get_Tm(P[ii]/1e6,1.)

    r_fine = np.arange(0,r[-1],r_resolution)
    T_fine, x_fine, P_fine, TmFe_fine = np.interp(r_fine,r,T), np.interp(r_fine,r,x), np.interp(r_fine,r,P), np.interp(r_fine,r,TmFe)
    r,T,TmFe,P,x = r_fine.copy(), T_fine.copy(), TmFe_fine.copy(), P_fine.copy(), x_fine.copy()

    T_adiabat = T.copy()
    dr = r[1] - r[0]
    
    return [r,dr,P,T,x,T_adiabat,TmFe,rho]



def pre_stage(x_l,dT):
    data = pickle.load(open('test_profiles.pkl', 'rb')) 
    adiabat_params    = [1,  0, -2.52083167e-14,  7.75340296e-22]  

    i_r, i_Ta, i_P, i_rho, i_g, i_Tm = 0,1,2,3,4,5

    sTm, pc = np.array(data[0][1][i_Tm]), np.array(data[0][1][i_P])
    rc, rhoc = np.array(data[0][1][i_r]), np.array(data[0][1][i_rho])

    rf  = np.arange(0.,3480e3,1.)
    P   = np.interp(rf,rc,pc)
    rho = np.interp(rf,rc,rhoc)
    r   = rf

    Pi = P[0]/1e9
    Ti = 5000.
    interval = 500.
    direction = -1.

    switches = 0
    condition = 1e-5

    while switches < 1000:
        rslt = delta_mu(x_l,Pi,Ti)

        old_direction = direction

        direction = 1.
        if rslt < 0.:
            direction = -1.

        if direction != old_direction:
            switches += 1
            interval = interval / 2.

        Ti = Ti + direction * interval
        if abs(rslt) < condition:
            break
    
    T_ad = Ti*np.polyval(adiabat_params[::-1], r)
    T_ad = T_ad - dT

    
    return [r,T_ad,P,rho]


def itterate(inputs):

    t_in,ri_in,r_in,P_in,T_in,x_in,T_adiabat_in,m,c,dt_growth,dr,alphaT_l,alphaT_s,alphaC_l,alphaC_s,alpha_advect,L,Cp,n_comp_points,TmFe_in,rho_in,upper_boundary = inputs
    
    t, ri, new_ri = t_in, ri_in, ri_in
    r,P,T,x,rho,TmFe = r_in.copy(), P_in.copy(), T_in.copy(), x_in.copy(), rho_in.copy(), TmFe_in.copy()

    sx, sP, sT = np.interp(ri,r,x), np.interp(ri,r,P)/1e9, np.interp(ri,r,T)
    grow = growth_rate(sx, sP, sT, m, c) * dt_growth

    if grow >= dr:
        new_ri = ri + dr*float(int(grow/dr))
        
        if new_ri > ri:
            # Latent heat
            nir = [np.where(abs(r-ri)==min(abs(r-ri)))[0][0],np.where(abs(r-new_ri)==min(abs(r-new_ri)))[0][0]]
            freeze_mass = 0.
            for ii in np.arange(nir[0],nir[1],1):
                freeze_mass += rho[ii] * 4./3.*np.pi*(r[ii+1]**3.-r[ii]**3.) 
            Ql = L*freeze_mass

            diffusion_length_l = (alphaT_l*dt_growth)**0.5
            diffusion_length_s = (alphaT_s*dt_growth)**0.5

            heated_radius = [ri-diffusion_length_s,new_ri+diffusion_length_l]
            if heated_radius[0] < 0.:
                heated_radius[0] = 0.
            nir = [np.where(abs(r-heated_radius[0])==min(abs(r-heated_radius[0])))[0][0],
                   np.where(abs(r-heated_radius[1])==min(abs(r-heated_radius[1])))[0][0]]
            heated_mass = 0.
            for ii in np.arange(nir[0],nir[1],1):
                    heated_mass += rho[ii] * 4./3.*np.pi*(r[ii+1]**3.-r[ii]**3.) 
                    
            additional_heat = Ql / (heated_mass * Cp)

            T[nir[0]:nir[1]] += additional_heat

            # Oxygen partitioning
            chem_limit = new_ri + (alphaC_l*dt_growth)**0.5

            # Define compositional space as solute points
            xO = 1.-x
            x_solute = xO * n_comp_points

            # Remove all solute points from recently frozen volume
            nir = [np.where(abs(r-ri)==min(abs(r-ri)))[0][0],np.where(abs(r-new_ri)==min(abs(r-new_ri)))[0][0]]
            O_to_partition = sum(x_solute[:nir[1]])

            x_solute[nir[0]:nir[1]] = 0.

            # Distribute all displaced solute points across chemical diffusion layer   
            nir[0] = nir[1]
            nir[1] = np.where(abs(r-chem_limit)==min(abs(r-chem_limit)))[0][0]

            x_solute[nir[0]:nir[1]] += O_to_partition / float(len(x_solute[nir[0]:nir[1]]))

            # Convert back to fractional comp (Fe)    
            x_new = 1. - (x_solute / n_comp_points)
            x = x_new.copy()

            # Update ri, time, record
            ri = new_ri
            t += dt_growth

    else:
        portions = np.zeros(3)
        dt_diffuse = min(np.gradient(r))**2./(2.*max([alphaT_l,alphaC_l]))/4.
        dt_advect = min(np.gradient(r))**2./(2.*max([alpha_advect]))/4.
        T_ = T - T_adiabat_in
        T_ref = T_.copy()
        
        # Define compositional space as solute points
        xO = 1.-x
        x_solute = xO * n_comp_points
        
        n_it = 1

        for it in range(n_it):
            T_new = T_.copy()
            xs_new = x_solute.copy()
            
            skip_counter = 0
            for ir in np.arange(1,len(r)-1,1):
                
                if skip_counter >= 50:# if diffusive changes are not being made, they wont be at
                    break             # at larger r, so we can finish up here.  

                no_diff = [False, False]
                
                # Heat diffusion
                portions[0] = alphaT_l * dt_diffuse / (r[ir-1]-r[ir])**2. * T_[ir-1]
                portions[1] = alphaT_l * dt_diffuse / ((r[ir-1]-r[ir+1])/2.)**2. * -2.*T_[ir]
                portions[2] = alphaT_l * dt_diffuse / (r[ir+1]-r[ir])**2. *  T_[ir+1]
                
                shifts = np.array([-portions[0]-0.5*portions[1],
                                  portions[0] + portions[1] + portions[2],
                                   -portions[2]-0.5*portions[1]])
                
                if len(np.where(shifts<=1e-2)[0]) == 3:
                    no_diff[0] = True
                
                T_new[ir-1] += shifts[0]
                T_new[ir]   += shifts[1]
                T_new[ir+1] += shifts[2]
                

                
                # Oxygen diffusion
                if r[ir] > new_ri:
                    portions[0] = alphaC_l * dt_diffuse / (r[ir-1]-r[ir])**2. * x_solute[ir-1]
                    portions[1] = alphaC_l * dt_diffuse / ((r[ir-1]-r[ir+1])/2.)**2. * -2.*x_solute[ir]
                    portions[2] = alphaC_l * dt_diffuse / (r[ir+1]-r[ir])**2. *  x_solute[ir+1]
                    
                    shifts = np.array([-portions[0]-0.5*portions[1],
                                       portions[0] + portions[1] + portions[2],
                                       -portions[2]-0.5*portions[1]])
                    if len(np.where(shifts<=1e-2)[0]) == 3:
                        no_diff[1] = True

                    criteria = np.array([xs_new[ir-1] + shifts[0],
                                         xs_new[ir]   + shifts[1],
                                         xs_new[ir+1] + shifts[2]])

                    if len(np.where(criteria<0)[0])==0 and len(np.where(criteria>n_comp_points)[0])==0:
                        xs_new[ir-1] = criteria[0]
                        xs_new[ir]   = criteria[1]
                        xs_new[ir+1] = criteria[2]
                        
                if no_diff[0]==True and no_diff[1]==True:
                    skip_counter += 1
                    
            T_ = T_new.copy()
            x_solute = xs_new.copy()
            
        if dt_diffuse > dt_advect:
            n_it = int(dt_diffuse/dt_advect) # need a small timestep for the fast advection
        else:
            n_it = 1
            
        sel = np.where(r>new_ri)[0] # only look at the liquid core
        
        for it in range(n_it):
            T_new = T_.copy()
            xs_new = x_solute.copy()
            
            skip_counter = 0
            for ir in np.arange(sel[0],sel[-1],1):
                
                if skip_counter >= 50:# if diffusive changes are not being made, they wont be at
                    break             # at larger r, so we can finish up here.  

                no_diff = [False, False]
                
                # Heat diffusion
                portions[0] = alpha_advect * dt_advect / (r[ir-1]-r[ir])**2. * T_[ir-1]
                portions[1] = alpha_advect * dt_advect / ((r[ir-1]-r[ir+1])/2.)**2. * -2.*T_[ir]
                portions[2] = alpha_advect * dt_advect / (r[ir+1]-r[ir])**2. *  T_[ir+1]
                
                shifts = np.array([-portions[0]-0.5*portions[1],
                                   portions[0] + portions[1] + portions[2],
                                   -portions[2]-0.5*portions[1]])
                if len(np.where(shifts<=1e-2)[0]) == 3:
                    no_diff[0] = True

                T_new[ir-1] += shifts[0]
                T_new[ir]   += shifts[1]
                T_new[ir+1] += shifts[2]
                
                # Oxygen diffusion
                portions[0] = alpha_advect * dt_advect / (r[ir-1]-r[ir])**2. * x_solute[ir-1]
                portions[1] = alpha_advect * dt_advect / ((r[ir-1]-r[ir+1])/2.)**2. * -2.*x_solute[ir]
                portions[2] = alpha_advect * dt_advect / (r[ir+1]-r[ir])**2. *  x_solute[ir+1]
                
                shifts = np.array([-portions[0]-0.5*portions[1],
                                   portions[0] + portions[1] + portions[2],
                                   -portions[2]-0.5*portions[1]])
                if len(np.where(shifts<=1e-2)[0]) == 3:
                    no_diff[1] = True

                criteria = np.array([xs_new[ir-1] + shifts[0],
                                     xs_new[ir]   + shifts[1],
                                     xs_new[ir+1] + shifts[2]])

                if len(np.where(criteria<0)[0])==0 and len(np.where(criteria>n_comp_points)[0])==0:
                    xs_new[ir-1] = criteria[0]
                    xs_new[ir]   = criteria[1]
                    xs_new[ir+1] = criteria[2]
                    
                    
                if no_diff[0]==True and no_diff[1]==True:
                    skip_counter += 1
                    
            T_ = T_new.copy()
            x_solute = xs_new.copy()

        T = T_ + T_adiabat_in
        x = 1.-(x_solute / n_comp_points)
        
        sel = np.where(T>TmFe)[0]
        T[sel] = TmFe[sel]
        
        if upper_boundary[0][0] == True:
            T[-1] = upper_boundary[0][1]
        if upper_boundary[1][0] == True:
            x[-1] = upper_boundary[1][1]
        
        t += dt_diffuse

    return [t,ri,r,T,x]

def initialize(xO,dT,r_lim,n_comp_points,r_res):
    r, T, P, rho = pre_stage(1.-xO,dT)
    
    x_solute = (xO*n_comp_points)*np.ones(len(r))
    x = 1.-x_solute/n_comp_points
    s = np.where(r<=r_lim)[0]
    T, P, r, x = T[s], P[s], r[s], x[s]

    TmFe = np.zeros(len(r))
    for ii in range(len(r)):
        TmFe[ii] = get_Tm(P[ii]/1e6,1.)

    r_fine = np.arange(0,r[-1],r_res)
    T_fine, x_fine, P_fine, TmFe_fine = np.interp(r_fine,r,T), np.interp(r_fine,r,x), np.interp(r_fine,r,P), np.interp(r_fine,r,TmFe)
    r,T,TmFe,P,x = r_fine.copy(), T_fine.copy(), TmFe_fine.copy(), P_fine.copy(), x_fine.copy()

    T_adiabat = T.copy()
    dr = r[1] - r[0]
    
    return [r,dr,P,T,x,T_adiabat,TmFe,rho]

def print_progress(i,itterations,t,ri,r_lim):
    statement = 'Step: '+str(i).ljust(math.ceil(np.log10(itterations)))+' out of '+str(itterations)+'. IC = '+str(round(ri,5)).ljust(6)+' m. Cutoff at '+str(r_lim)
    statement += ' m. Total time: '+str(round(t,2))+' s.'
    print(statement)
    return