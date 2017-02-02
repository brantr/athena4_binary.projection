/*! \file read_athena_binary.h
 *  \brief Functions for reading athena binaries.
 *
 *  The individual athena binary fields (density, 
 *  velocity, etc.) are stored as grid_fft grids
 *  or fields, which eases use with Fourier 
 *  transform operations in k-space. This method
 *  is naturally parallelized using mpi.*/
#ifndef READ_ATHENA_BINARY
#define READ_ATHENA_BINARY

#include<mpi.h>
#ifndef NO_FFTW
#include<fftw3-mpi.h>
#include<fftw3.h>
#include"grid_fft.h"
#else  //NO_FFTW

#include<stddef.h>

/*! \struct FFTW_Grid_Info
 *  \brief  Structure to contain
 *          information for use with
 *          3-d dimensional FFTW grids
 *	    parallelized with MPI.
 */
struct FFTW_Grid_Info
{
	/*! \var ptrdiff_t nx
	 *  \brief Number of grid points in x-direction. */
	ptrdiff_t nx;
	/*! \var ptrdiff_t ny
	 *  \brief Number of grid points in y-direction. */
	ptrdiff_t ny;

	/*! \var ptrdiff_t nz
	 *  \brief Number of grid points in z-direction. */
	ptrdiff_t nz;

	/*! \var ptrdiff_t nz_complex
	 *  \brief Number of grid points in z-direction for complex data. */
	ptrdiff_t nz_complex;

	/*! \var ptrdiff_t nx_local
	 *  \brief Local number of grid points in x-direction */
	ptrdiff_t nx_local;

	/*! \var ptrdiff_t nx_local_start
	 *  \brief First grid point in x-direction */
	ptrdiff_t nx_local_start;

	/*! \var int *n_local_real_size
	 *  \brief Total size of real grids on local process.*/
	ptrdiff_t n_local_real_size;

	/*! \var int *n_local_complex_size
	 *  \brief Total size of complex grids on local process.*/
	ptrdiff_t n_local_complex_size;

	/*! \var double BoxSize
	 *  \brief 1-d length of the grid. */
	double BoxSize;

	/*! \var double dx 
	 *  \brief Length of one grid cell in x direction. */
	double dx;

	/*! \var double dy 
	 *  \brief Length of one grid cell in y direction. */
	double dy;

	/*! \var double dz 
	 *  \brief Length of one grid cell in z direction. */
	double dz;

	/*! \var double dV 
	 *  \brief Volume of one grid cell. */
	double dV;

	/*! \var double dVk
	 *  \brief Volume of one grid cell in k-space. */
	double dVk;

	/*! \var int ndim
	 *  \brief Number of dimensions. */
	int ndim;
};


/*! \fn int grid_ijk(int i, int j, int k, FFTW_Grid_Info grid_info)
 *  \brief Array index for fftw grid based on coordinates i,j,k. */
int grid_ijk(int i, int j, int k, FFTW_Grid_Info grid_info);

/*! \fn int grid_index(double x, double y, double z, FFTW_Grid_Info grid_info)
 *  \brief Given a position, return the grid index. */
int grid_index(double x, double y, double z, FFTW_Grid_Info grid_info);

/*! \fn void initialize_mpi_local_sizes(FFTW_Grid_Info *grid_info, int myid, int numprocs, MPI_Comm world);
 *  \brief Function to determine local grid sizes for parallel FFT. */
void athena_initialize_mpi_local_sizes(FFTW_Grid_Info *grid_info, int myid, int numprocs, MPI_Comm world);

/*! \fn double *allocate_real_fftw_grid_sized(int n_size);
 *  \brief Allocates a pre-sized 3-d real grid for use with fftw.*/
double *allocate_real_fftw_grid_sized(int n_size);

/*! \fn double *allocate_real_fftw_grid(FFTW_Grid_Info grid_info)
 *  \brief Allocates a 3-d real grid for use with fftw.*/
double *allocate_real_fftw_grid(FFTW_Grid_Info grid_info);

/*! \fn double **allocate_field_fftw_grid(int nd, FFTW_Grid_Info grid_info);
 *  \brief Allocates a field[ndim][n_local_real_size] (of dimension ndim) of  3-d real grids for use with fftw.*/
double **allocate_field_fftw_grid(int nd, FFTW_Grid_Info grid_info);

/*! \fn void deallocate_field_fftw_grid(double **field, int nd, FFTW_Grid_Info grid_info)
 *  \brief De-allocates a field[ndim][n_local_size] (of dimension ndim) of  3-d real grids for use with fftw.*/
void deallocate_field_fftw_grid(double **field, int nd, FFTW_Grid_Info grid_info);

#endif //NO_FFTW




class AthenaBinarySlice
{
	public:
		//; nvar=4 means isothermal hydro.  nvar=5 means adiabatic hydro
		//; nvar=7 means isothermal MHD.    nvar=8 means adiabatic mhd
		int nvar;
		int nscalars;
		int ngrav;

		int flag_gravity;
		int flag_scalars;
		int flag_mhd;
		int flag_isothermal;
		int flag_memory;
		int flag_pressure;
	
		int ipressure;

		//(gamma-1) and isothermal sound speed

		double gamma_minus_1;	
		double c_s_iso;	

		int ny;
		int nz;

		//data
		double **density;
		double **etot;
		double ***velocity;
		double ***b_field;
		double ***scalars;
		double **phi;
		double **mask;

		//constructor and destructor
		AthenaBinarySlice(void);	
		~AthenaBinarySlice(void);	

		//routine to allocate data
		void AllocateData(void);

		//routine to de-allocate data
		void DestroyData(void);

		//derived quantities
		double pressure(int j, int k);
		double energy(int j, int k);

		//copy slice
		void CopySlice(AthenaBinarySlice *copy);

};


class AthenaBinary
{
	public:
		//MPI information
		int myid;
		int numprocs;
		MPI_Comm world;

		//grid information
		FFTW_Grid_Info grid_info;

		//number of files
		int nfiles;

#ifdef ATHENA4
		int CoordinateSystem;
#endif /*ATHENA4*/

		//grid size information

		int nx;
		int ny;
		int nz;

		//local size on each processor
		int *nx_local_array;
		int *nx_local_start_array;
		int *n_local_real_size_array;

		//local file sizes

		int *nx_file;
		int *ny_file;
		int *nz_file;
		int *nx_start;
		int *ny_start;
		int *nz_start;

		//number of dimensions

		int ndim;


		//; nvar=4 means isothermal hydro.  nvar=5 means adiabatic hydro
		//; nvar=7 means isothermal MHD.    nvar=8 means adiabatic mhd
		int nvar;
		int nscalars;
		int ngrav;

		int flag_gravity;
		int flag_scalars;
		int flag_mhd;
		int flag_isothermal;
		int flag_memory;
		int flag_pressure;
#ifdef ATHENA4
		int flag_tracers;
#endif /*ATHENA4*/
	
		int ipressure;


		//(gamma-1) and isothermal sound speed

		double gamma_minus_1;	
		double c_s_iso;	

		//snapshot time and timestep
		double t;
		double dt;

		//grid coordinates
		double *x;
		double *y;
		double *z;

		//double BoxSize;
	
		//data
		double  *density;
		double  *etot;
		double **velocity;
		double **b_field;
		double **scalars;
		double  *phi;
		double  *mask;

		//constructor and destructor
		AthenaBinary(void);	
		~AthenaBinary(void);	

		//initialize class
		void Initialize(void);

		//read single athena binary
		void ReadAthenaHeader(char *fname);

		void ReadAthenaBinary(char *fname);

		void WriteAthenaBinary(char *fname);
  void WriteAthenaBinaryLimits(char *fname, int nx_min, int nx_max, int ny_min, int ny_max, int nz_min, int nz_max);


		//read binary split across multiple files
		void ReadAthenaBinarySplit(char *fdir, char *fbase, int isnap);
		void ReadAthenaInfoBinarySplit(char *fdir, char *fbase, int isnap);

		void ReadAthenaBinaryField(FILE *fp, int nfield);

		void ReadAthenaBinaryFieldSplit(char *fdir, char *fbase, int isnap, fpos_t *fpos, int nfield);

		void WriteAthenaBinaryField(FILE *fp, int nfield);
		void WriteAthenaBinaryFieldLimits(FILE *fp, int nfield, int nx_min, int nx_max, int ny_min, int ny_max, int nz_min, int nz_max);


		void StoreAthenaBinaryField(double *frbuf, int ip, int nfield);
		void RecoverAthenaBinaryField(double *frbuf, int ip, int nfield);

		//routine to allocate data
		void AllocateData(void);

		//routine to de-allocate data
		void DestroyData(void);

		//derived quantities
		double pressure(int i, int j, int k);
		double energy(int i, int j, int k);


		//show information
		void ShowAthenaHeader(void);
		void ShowAthenaBinary(void);


		//share slices
		AthenaBinarySlice *ExchangeAthenaBinarySlices(int dir);

		//ghost slice
		AthenaBinarySlice slice;
};


#endif  //READ_ATHENA_BINARY
