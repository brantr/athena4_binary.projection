/*! \file constants.h
 *  \brief Declaration for the Constants class that contains
 *         many useful physical and geometrical constants.
 */
#ifndef BRANT_CONSTANTS
#define BRANT_CONSTANTS

/*! \class Constants
 *  \brief Class containing useful physical and geometrical constants. */
class Constants
{
	public:

	//physical constants

		/*! \var double c
                 *  \brief The speed of light in km/s. */
		double c;       

		/*! \var double G
                 *  \brief Newton's Gravitational constant in m^3  kg^-1 s^-2 */
		double G;

		/*! \var double G_cosmo
                 *  \brief Newton's Gravitational constant in (km/s)^2 kpc msun^-1 */
		double G_cosmo;

		/*! \var double G_gal
		 *  \brief Newton's Gravitational constant in kpc^3 Msun^-1 Gyr^-2 */
		double G_gal;

		/*! \var double h_bar
		 *  \brief planck's constant / 2pi in J * s */
		double h_bar;

		/*! \var double h_planck
		 *  \brief planck's constant in J * s */
		double h_planck;

		/*! \var double k_b
		 *  \brief Boltzmann's constant in J K^-1 */
		double k_b;

		/*! \var double m_proton
		 *  \brief Proton mass in kg. */
		double m_proton;

		/*! \var double m_helium
		 *  \brief Helium mass in kg. */
		double m_helium;

		/*! \var double sigma_t
		 *  \brief Thomson cross section in m^2. */
		double sigma_t; //thomson cross-section

		/*! \var double sigma_sb
		 *  \brief Stefan-Boltzmann constant in J m^-2 K^-4 s^-1 */
		double sigma_sb;//stefan-boltzmann

		/*! \var double a_sb
		 *  \brief Stefan-Boltzmann radiative constant in J m^-3 K^-4 */
		double a_sb;//stefan-boltzmann

	//geometrical and natural constants

		/*! \var double pi
                 *  \brief Pi, 3.14159... */
		double pi;

		/*! \var double e
		 *  \brief e, the base of the natural logarithm */
		double e;

		/*! \var double mpc
		 *  \brief mpc, a megaparsec in m */
		double mpc;

		/*! \var double kpc
		 *  \brief kpc, a kiloparsec in m */
		double kpc;

		/*! \var double pc
		 *  \brief pc, a parsec in m */
		double pc;

		/*! \var double year_in_sec
		 *  \brief A year in seconds */
		double year_in_sec; 

		/*! \var double msun
		 *  \brief A solar mass in kg. */
		double msun;        

		/*! \var double rsun
		 *  \brief A solar radius in m. */
		double rsun;        

		/*! \var double lsun
		 *  \brief A solar luminosity in Watts. */
		double lsun;

		/*! \var double angstrom
		 *  \brief An angstrom in m */
		double angstrom;

		/*! \var double radians_per_arcsec 
		 *  \brief Number of radians in an arcsec. */
		double radians_per_arcsec;

		/*! \var double arcsecs_per_radian
		 *  \brief Number of arcseconds in a radian. */
		double arcsecs_per_radian;

		/*! \var double square_arcsecs_per_steradian
		 *  \brief Number of square arcseconds in a steradian. */
		double square_arcsecs_per_steradian;

		/*! \var double square_degrees_per_steradian
		 *  \brief Number of square degrees in a steradian. */
		double square_degrees_per_steradian;

		/*! \var double arcmins_per_degree
		 *  \brief Number of arcminutes per degree. */
		double arcmins_per_degree;

		/*! \var double arcsecs_per_arcmin
		 *  \brief Number of arcseconds per arcminute. */
		double arcsecs_per_arcmin;

		/*! \var double arcsecs_per_degree
		 *  \brief Number of arcseconds per degree. */
		double arcsecs_per_degree;

		/*! \var double square_arcmins_per_square_degree
		 *  \brief Number of square arcseconds per square degree. */
		double square_arcmins_per_square_degree;
	
	//other units

		/*! \var double h_planck_eVs
		 *  \brief Planck's constant in electron volt * seconds */
		double h_planck_eVs;

	//physical constants in cgs

		/*! \var double c_cgs
                 *  \brief The speed of light in cm/s. */
		double c_cgs;       		

		/*! \var double G_cgs
                 *  \brief Newton's Gravitational constant in cm^3 g^-1 s^-2 */
		double G_cgs;

		/*! \var double h_bar_cgs
		 *  \brief planck's constant / 2pi in erg * s */
		double h_bar_cgs;

		/*! \var double h_planck_cgs
		 *  \brief planck's constant in ergs * s */
		double h_planck_cgs;		

		/*! \var double k_b_cgs
		 *  \brief Boltzmann's constant in ergs K^-1 */
		double k_b_cgs;

		/*! \var double m_proton_cgs
		 *  \brief Proton mass in g */
		double m_proton_cgs;

		/*! \var double m_electron_cgs
		 *  \brief Electron mass in g */
		double m_electron_cgs;

		/*! \var double m_alpha_cgs
		 *  \brief Alpha particle mass in g */
		double m_alpha_cgs;

		/*! \var double m_helium_cgs
		 *  \brief Helium mass in g */
		double m_helium_cgs;

		/*! \var double e_electron_cgs
		 *  \brief Electron charge in electrostatic units */
		double e_electron_cgs; //electron charge in electrostatic units

		/*! \var double sigma_t_cgs
		 *  \brief Thomson cross-section in cm^2 */
		double sigma_t_cgs;

		/*! \var double sigma_sb_cgs
		 *  \brief Stefan-Boltzmann constant in erg cm^-2 K^-4 s^-1 */
		double sigma_sb_cgs;//stefan-boltzmann

		/*! \var double a_sb_cgs
		 *  \brief Stefan-Boltzmann radiative constant in erg cm^-3 K^-4 */
		double a_sb_cgs;//stefan-boltzmann

		/*! \var double rydberg_in_hz
 		 *  \brief rydberg (13.6 eV) in Hz */
		double rydberg_in_hz;//rydberg in hz 

		/*! \var double rydberg_cgs
 		 *  \brief rydberg (13.6 eV) in cgs units (cm^-1) */
		double rydberg_cgs;//rydberg in cgs units (cm^-1)

		/*! \var double rydberg_in_ergs
 		 *  \brief rydberg (13.6 eV) in ergs units */
		double rydberg_in_ergs;//rydberg in ergs

		/*! \var double nu_HI_ionization
 		 *  \brief HI ionization potential in Hz */
		double nu_HI_ionization;	//HI ionization in Hz

		/*! \var double nu_HeI_ionization
 		 *  \brief HeI ionization potential in Hz */
		double nu_HeI_ionization;	//HeI ionization in Hz

		/*! \var double nu_HeII_ionization
 		 *  \brief HeII ionization potential in Hz */
		double nu_HeII_ionization;	//HeII ionization in Hz

		/*! \var double a_bohr_cgs
 		 *  \brief Bohr radius in cm */
		double a_bohr_cgs;		//bohr radius in cm

		/*! \var double alpha_fs
 		 *  \brief Fine structure constant */
		double alpha_fs;		//fine structure constant

		/*! \var double amu_to_cgs
 		 *  \brief Atomic mass unit in g */
		double amu_to_cgs;		//amu in g


	//geometrical and natural constants in cgs

		/*! \var double mpc_cgs
 		 *  \brief A megaparsec in cm. */
		double mpc_cgs;

		/*! \var double kpc_cgs
 		 *  \brief A kiloparsec in cm. */
		double kpc_cgs;

		/*! \var double pc_cgs
 		 *  \brief A parsec in cm. */
		double pc_cgs;

		/*! \var double msun_cgs
 		 *  \brief A solar mass in g. */
		double msun_cgs;        

		/*! \var double rsun_cgs
 		 *  \brief A solar radius in cm. */
		double rsun_cgs;        

		/*! \var double lsun_cgs
 		 *  \brief A solar luminosity in ergs s^-1. */
		double lsun_cgs;

		/*! \var double angstrom_cgs
 		 *  \brief An angstrom in cm */
		double angstrom_cgs;

		/*! \fn Constants(void)
	 	 *  \brief Constructor for the Constants class */
		Constants(void);

		/*! \fn void ShowConstants(void)
 		*  \brief Prints the values of important constants to stdout. */
		void SetConstants(void);

		/*! \fn void Constants::SetConstants(void)
 		*  \brief Sets the values of the constants in the Constants class.
 		*
 		*  Uses the GSL library constant definitions where possible.
 		*/
		void ShowConstants(void);
};
#endif
