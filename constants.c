#include<stdio.h>
#include<math.h>
#include<gsl/gsl_const_mksa.h>
#include<gsl/gsl_const_cgsm.h>
#include<gsl/gsl_const_num.h>
#include"constants.h"

/*! \file constants.c
 *  \brief This file contains the member definitions for the Constants class. */


/*! \fn Constants::Constants(void)
 *  \brief The constructor for the Constants class.
 *
 *   The constructor uses SetConstants() to set the
 *   values of the constants.
 */
Constants::Constants(void)
{
	//set the values of the constants
	SetConstants();
}
/*! \fn void Constants::SetConstants(void)
 *  \brief Sets the values of the constants in the Constants class.
 *
 *  Uses the GSL library constant definitions where possible.
 */
void Constants::SetConstants(void)
{
	//geometrical constants
	Constants::pi	   	= 3.141592654;	//pi
	Constants::e        	= 2.718281828;  //base of natural log

	//physical constants

	Constants::c        	= GSL_CONST_MKSA_SPEED_OF_LIGHT; //speed of light in m/s
	Constants::G        	= GSL_CONST_MKSA_GRAVITATIONAL_CONSTANT; //G in m^3 kg^-1 s^-2
	Constants::h_bar    	= GSL_CONST_MKSA_PLANCKS_CONSTANT_HBAR;//planck's constant in Js
	Constants::h_planck 	= GSL_CONST_MKSA_PLANCKS_CONSTANT_H;//planck's constant in Js
	Constants::k_b      	= GSL_CONST_MKSA_BOLTZMANN;	//boltzmann constant in JK^-1
	Constants::m_proton 	= GSL_CONST_MKSA_MASS_PROTON; //proton mass in kg
	Constants::m_helium     = m_proton*(4.002602/1.00794); //helium mass in kg
	Constants::sigma_t  	= 6.65246e-29;	//Thomson crossection in m^2
	Constants::sigma_sb 	= (2./15.)*pow(pi,5)*pow(k_b,4)*pow(c,-2)*pow(h_planck,-3);	//Stefan-Boltzmann constant in J K^-4 m^-2 s^-1
	Constants::a_sb 	= sigma_sb/c;	//Stefan-Boltzmann radiative constant in J K^-4 m^-3

	//natural constants

	Constants::mpc	   	= GSL_CONST_MKSA_PARSEC*1.0e6; //megaparsec in m
	Constants::kpc	   	= GSL_CONST_MKSA_PARSEC*1.0e3; //kiloparsec in m
	Constants::pc	   	= GSL_CONST_MKSA_PARSEC;       //parsec in m
	Constants::year_in_sec 	= 3.155815e7;	//year in seconds
	Constants::msun        	= GSL_CONST_MKSA_SOLAR_MASS; //solar mass in kg
	Constants::rsun        	= 6.96e8;	//solar radius in m
	Constants::lsun        	= 3.90e26;	//solar luminosity in W
	Constants::angstrom 	= GSL_CONST_MKSA_ANGSTROM;	//angstrom in m
	Constants::radians_per_arcsec = 4.848136958e-6;
	Constants::arcsecs_per_radian = 206264.8;
	Constants::square_arcsecs_per_steradian = pow(206264.8,2);
	Constants::square_degrees_per_steradian = pow(180.0/pi,2);
	Constants::arcsecs_per_arcmin = 60.0;
	Constants::arcmins_per_degree = 60.0;
	Constants::arcsecs_per_degree = 60.0*60;
	Constants::square_arcmins_per_square_degree = 60.0*60.0;

	//other units
	Constants::h_planck_eVs = 4.13566733e-15;	//eV*s

	//physical constants in cgs
	Constants::c_cgs    = GSL_CONST_CGSM_SPEED_OF_LIGHT; //speed of light in cm/s
	Constants::G_cgs    = GSL_CONST_CGSM_GRAVITATIONAL_CONSTANT; //G in cm^3 g^-1 s^-2
	Constants::h_bar_cgs= GSL_CONST_CGSM_PLANCKS_CONSTANT_HBAR;//planck's constant in erg s
	Constants::h_planck_cgs = GSL_CONST_CGSM_PLANCKS_CONSTANT_H;//planck's constant in erg s
	Constants::k_b_cgs  = GSL_CONST_CGSM_BOLTZMANN;	//boltzmann constant in erg K^-1
	Constants::m_proton_cgs= GSL_CONST_CGSM_MASS_PROTON; //proton mass in g
	Constants::m_electron_cgs= GSL_CONST_CGSM_MASS_ELECTRON; //proton mass in g
	Constants::m_alpha_cgs= 6.64465620e-24; //alpha particle mass in g
	Constants::m_helium_cgs= m_proton_cgs*(4.002602/1.00794); //helium mass in g
	Constants::e_electron_cgs = 4.8032068e-10;	//electrostatic units
	Constants::sigma_t_cgs = 6.65246e-25;	//Thomson crossection in cm^2
	Constants::sigma_sb_cgs	= (2./15.)*pow(pi,5)*pow(k_b_cgs,4)*pow(c_cgs,-2)*pow(h_planck_cgs,-3); //Stefan-Boltzmann constant in erg cm^-2 K^-4 s^-1	
	Constants::a_sb_cgs 	= sigma_sb_cgs/c_cgs;	//Stefan-Boltzmann radiative constant in erg cm^-3 K^-4
	Constants::rydberg_in_hz = 3.2899e15;	//rydberg in Hz
	Constants::rydberg_cgs   = 1.0974e5;	//rydberg in cm^-1
	Constants::rydberg_in_ergs = 2.1798741e-11;	//rydberg in ergs
	Constants::nu_HI_ionization   = (c*1.0e10)/912.0; 	//912Ang in Hz 
	Constants::nu_HeI_ionization  = 24.6/h_planck_eVs;	//HeI ionization in freq
	Constants::nu_HeII_ionization = 54.4/h_planck_eVs; 	//HeII ionization in freq
	Constants::amu_to_cgs = 1.66053886e-24;//amu in g

	//geometrical and natural constants in cgs

	Constants::mpc_cgs = GSL_CONST_CGSM_PARSEC*1.0e6; //megaparsec in m
	Constants::kpc_cgs = GSL_CONST_CGSM_PARSEC*1.0e3; //kiloparsec in m
	Constants::pc_cgs  = GSL_CONST_CGSM_PARSEC;       //parsec in m
	Constants::msun_cgs= GSL_CONST_CGSM_SOLAR_MASS; //solar mass in g
	Constants::rsun_cgs= 6.96e10;	//solar radius in cm
	Constants::lsun_cgs= (3.90e26)*GSL_CONST_CGSM_JOULE;//solar luminosity in W
	Constants::angstrom_cgs = GSL_CONST_CGSM_ANGSTROM;	//angstrom in cm
	Constants::G_cosmo  = G*msun/(1.0e6*kpc); //G in (km/s)^2 kpc msun^-1
	Constants::G_gal    = G*(msun/pow(kpc,3.0))*pow(year_in_sec*1.0e9,2);
	Constants::a_bohr_cgs = GSL_CONST_CGSM_BOHR_RADIUS;	//bohr radius in cm
	Constants::alpha_fs = GSL_CONST_NUM_FINE_STRUCTURE;	//fine structure constant;
}



/*! \fn void Constants::ShowConstants(void)
 *  \brief Prints the values of important constants to stdout. */
void Constants::ShowConstants(void)
{
	//physical constants

	printf("c               =\t%e m/s\n",c);
	printf("G               =\t%e m^3 kg^-1 s^-1\n",G);
	printf("h_bar           =\t%e kg m s^-1\n",h_bar);
	printf("h_planck        =\t%e kg m s^-1\n",h_planck);
	printf("k_b             =\t%e\n",k_b);
	printf("m_proton        =\t%e\n",m_proton);
	printf("sigma_t         =\t%e\n",sigma_t);
	printf("sigma_sb        =\t%e\n",sigma_sb);

	printf("pi              =\t%e\n",pi);
	printf("mpc             =\t%e\n",mpc);
	printf("kpc             =\t%e\n",kpc);
	printf("pc              =\t%e\n",pc);
	printf("year_in_sec     =\t%e\n",year_in_sec);
	printf("msun            =\t%e\n",msun);
	printf("rsun            =\t%e\n",rsun);
	printf("lsun            =\t%e\n",lsun);
	printf("angstom         =\t%e\n",angstrom);

	printf("c_cgs           =\t%e\n",c_cgs);
	printf("G_cgs           =\t%e\n",G_cgs);
	printf("h_bar_cgs       =\t%e\n",h_bar_cgs);
	printf("h_planck_cgs    =\t%e\n",h_planck_cgs);
	printf("k_b_cgs         =\t%e\n",k_b_cgs);
	printf("m_proton_cgs    =\t%e\n",m_proton_cgs);
	printf("sigma_t_cgs     =\t%e\n",sigma_t_cgs);
	printf("sigma_sb_cgs    =\t%e\n",sigma_sb_cgs);

	printf("mpc_cgs         =\t%e\n",mpc_cgs);
	printf("kpc_cgs         =\t%e\n",kpc_cgs);
	printf("pc_cgs          =\t%e\n",pc_cgs);
	printf("msun_cgs        =\t%e\n",msun_cgs);
	printf("rsun_cgs        =\t%e\n",rsun_cgs);
	printf("lsun_cgs        =\t%e\n",lsun_cgs);
	printf("angstom_cgs     =\t%e\n",angstrom_cgs);
	printf("G_cosmo         =\t%e (km/s)^2 Msun^-1 kpc \n",G_cosmo);
	printf("G_gal           =\t%e kpc^3 Msun^-1 Gyr^-2\n",G_gal);

	printf("Radians per arcsec = %e\n",radians_per_arcsec);
	printf("Arcsecs per radian = %e\n",arcsecs_per_radian);
	printf("Square Arcsecs per sterradian = %e\n",square_arcsecs_per_steradian);
	printf("Square Degrees per sterradian = %e\n",square_degrees_per_steradian);
}
