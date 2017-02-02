/*! \file grid_operations.c
 *  \brief Function definitions for fft grid operations. */
#include <mpi.h>
#include <math.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_interp.h>
#include "grid_operations.h"
#include "routines.hpp"

//random number generator
gsl_rng *r;
const gsl_rng_type *T;

/*! \fn void grid_set_value(double value, double *A, FFTW_Grid_Info grid_info)
 *  \brief Set the grid to a single value. */
void grid_set_value(double value, double *A, FFTW_Grid_Info grid_info)
{

	int i, j, k;
	int ijk;

	int nx_local = grid_info.nx_local;
	int nx       = grid_info.nx;
	int ny       = grid_info.ny;
	int nz       = grid_info.nz;

	//set the grid to value
	for(i=0;i<nx_local;++i)
		for(j=0;j<ny;++j)
			for(k=0;k<nz;++k)
			{	
				//get grid index
				ijk = grid_ijk(i,j,k,grid_info);

				//set grid elements to value
				A[ijk] = value;
			}
}

/*! \fn void grid_field_set_value(double value, double **A, FFTW_Grid_Info grid_info)
 *  \brief Set the field to a single value. */
void grid_field_set_value(double value, double **A, FFTW_Grid_Info grid_info)
{

	int i, j, k;
	int ijk;

	int nx_local = grid_info.nx_local;
	int nx       = grid_info.nx;
	int ny       = grid_info.ny;
	int nz       = grid_info.nz;

	//set the grid to value
	for(i=0;i<nx_local;++i)
		for(j=0;j<ny;++j)
			for(k=0;k<nz;++k)
			{	
				//get grid index
				ijk = grid_ijk(i,j,k,grid_info);

				//set grid elements to value
				for(int m=0;m<grid_info.ndim;m++)
					A[m][ijk] = value;
			}
}

/*! \fn double *grid_copy(double *A, FFTW_Grid_Info grid_info)
 *  \brief Produces a copy of a double grid. */
double *grid_copy(double *A, FFTW_Grid_Info grid_info)
{
	int i, j, k, ijk;

	int nx_local = grid_info.nx_local;
	int nx       = grid_info.nx;
	int ny       = grid_info.ny;
	int nz       = grid_info.nz;

	//the copy to return
	double *copy;

	//allocate copy
	copy = allocate_real_fftw_grid(grid_info);

	//Copy A into copy.
	for(i=0;i<nx_local;++i)
		for(j=0;j<ny;++j)
			for(k=0;k<nz;++k)
			{	
				//grid index
				ijk = grid_ijk(i,j,k,grid_info);

				//copy the grid
				copy[ijk] = A[ijk];
			}

	//return the copy
	return copy;
}


/*! \fn double **grid_field_copy(double **source, FFTW_Grid_Info grid_info)
 *  \brief Produces a copy of a double grid field. */
double **grid_field_copy(double **source, FFTW_Grid_Info grid_info)
{
	int i, j, k;
	int ijk;

	int nx_local = grid_info.nx_local;
	int nx       = grid_info.nx;
	int ny       = grid_info.ny;
	int nz       = grid_info.nz;

	int ndim = grid_info.ndim;

	//the copy
	double **copy ;

	//allocate copy
	copy = allocate_field_fftw_grid(ndim,grid_info);

	//copy the field
	grid_field_copy_in_place(source,copy,grid_info);

	//return the copy
	return copy;
}

/*! \fn double *grid_make_gaussian_kernel(double r_cells, FFTW_Grid_Info grid_info)
 *  \brief make a gaussian kernel with stddev r_cells */
double *grid_make_gaussian_kernel(double r_cells, FFTW_Grid_Info grid_info)
{
  double *kernel = allocate_real_fftw_grid(grid_info);
  int i,j,k,ijk;
  int nx_start = grid_info.nx_local_start;
  int nx_local = grid_info.nx_local;
  int nx = grid_info.nx;
  int ny = grid_info.ny;
  int nz = grid_info.nz;
  double r;
  double x;
  double y;
  double z;

  /*populate gaussian kernel*/
  for(i=0;i<nx_local;++i)
    for(j=0;j<ny;++j)
      for(k=0;k<nz;++k)
      {
	//grid index
	ijk = grid_ijk(i,j,k,grid_info);

	if(i+nx_start>nx/2)
	{
	  //radius from corner in cells
	  x = nx - (i + nx_start);
	}else{
	  x = (i + nx_start);
	}

	if(j>ny/2)
	{
	  //radius from corner in cells
	  y = ny - j;
	}else{
	  y = j;
	}

	if(k>nz/2)
	{
	  //radius from corner in cells
	  z = nz - k;
	}else{
	  z = k;
	}

	//radius
	r = sqrt(x*x + y*y + z*z);

	//3-d gaussian
	kernel[ijk] = 1./sqrt(pow(2*M_PI*r_cells*r_cells,3.)) *exp( -0.5 * pow( r/r_cells, 2) );
      }

  /*return the answer*/
  return kernel;
}

/*! \fn double *grid_convolve(double *A, double *B, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world)
 *  \brief convolve grid A with grid B */
/*
double *grid_convolve(double *A, double *B, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world)
{
  //smoothed grid
  double *C; 
	
  //complex data field
  fftw_complex *Ck;

  //fourier components of C
  fftw_complex *Ak;

  //fourier components of B
  fftw_complex *Bk;

  //indices
  int kf;
  int i, j, k;

  int ijk;

  int nx_local = grid_info.nx_local;
  int nx_local_start = grid_info.nx_local_start;
  int nx       = grid_info.nx;
  int ny       = grid_info.ny;
  int nz       = grid_info.nz;
  int nz_complex = grid_info.nz_complex;
  int nzl;

  double kx, ky, kz;
  double kk;
  double L = grid_info.BoxSize;


  int ndim = grid_info.ndim;

  //normalization
  double scale = 1./( ((double) grid_info.nx)*((double) grid_info.ny)*((double) grid_info.nz) );

  //forward and reverse FFTW plans
  fftw_plan plan_Ak;
  fftw_plan plan_Bk;
  fftw_plan iplan_C;


  //allocate smoothed grid array
  C = allocate_real_fftw_grid(grid_info);

  //allocate work and transofrm
  Ck    = allocate_complex_fftw_grid(grid_info);
  Ak    = allocate_complex_fftw_grid(grid_info);
  Bk    = allocate_complex_fftw_grid(grid_info);


  //create the fftw plans
  plan_Ak = fftw_mpi_plan_dft_3d(grid_info.nx, grid_info.ny, grid_info.nz, Ak, Ak, world, FFTW_FORWARD,  FFTW_ESTIMATE);
  plan_Bk = fftw_mpi_plan_dft_3d(grid_info.nx, grid_info.ny, grid_info.nz, Bk, Bk, world, FFTW_FORWARD,  FFTW_ESTIMATE);
  iplan_C = fftw_mpi_plan_dft_3d(grid_info.nx, grid_info.ny, grid_info.nz, Ck, Ck, world, FFTW_BACKWARD, FFTW_ESTIMATE);

  //get complex version of A
  grid_copy_real_to_complex_in_place(A, Ak, grid_info);

  //get complex version of B
  grid_copy_real_to_complex_in_place(B, Bk, grid_info);

  //perform the forward transform on A
  fftw_execute(plan_Ak);

  //perform the forward transform on B
  fftw_execute(plan_Bk);


  //at this stage, uk contains the Fourier transform of u

  for(int i=0;i<nx_local;++i)
    for(int j=0;j<ny;++j)
      for(int k=0;k<nz;++k)
      {

	//index of uk corresponding to k  

	ijk = grid_complex_ijk(i,j,k,grid_info);
	
	//perform convolution
	//appears to be correct, check 10/31/2013
	Ck[ijk][0] = (Ak[ijk][0]*Bk[ijk][0] - Ak[ijk][1]*Bk[ijk][1]) * scale;
	Ck[ijk][1] = (Ak[ijk][0]*Bk[ijk][1] + Ak[ijk][1]*Bk[ijk][0]) * scale;
      }

  //perform the inverse transform of the derivative
  fftw_execute(iplan_C);
  
  //copy inverse transform into convolved field
  grid_copy_complex_to_real_in_place(Ck, C, grid_info);

  //free the buffer memory
  fftw_free(Ak);
  fftw_free(Bk);
  fftw_free(Ck);

  //destroy the plans
  fftw_destroy_plan(plan_Ak);
  fftw_destroy_plan(plan_Bk);
  fftw_destroy_plan(iplan_C);

  //return the answer
  return C;
}
*/
/*! \fn double **grid_field_smooth(double **u, double *kernel, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world)
 *  \brief smooth a field using a kernel */
/*
double **grid_field_smooth(double **uin, double *kernel, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world)
{
  //smoothed grid
  double **u_smooth; 
	
  //complex data field
  fftw_complex *Ck;

  //fourier components of C
  fftw_complex *Ak;

  //fourier components of B
  fftw_complex *Bk;

  //indices
  int l; //dimension
  int i, j, k;

  int ijk;

  int nx_local = grid_info.nx_local;
  int nx_local_start = grid_info.nx_local_start;
  int nx       = grid_info.nx;
  int ny       = grid_info.ny;
  int nz       = grid_info.nz;
  int nz_complex = grid_info.nz_complex;
  int nzl;

  double kx, ky, kz;
  double kk;
  double L = grid_info.BoxSize;


  int ndim = grid_info.ndim;

  //normalization
  double scale = 1./( ((double) grid_info.nx)*((double) grid_info.ny)*((double) grid_info.nz) );

  //forward and reverse FFTW plans
  fftw_plan plan_Ak;
  fftw_plan plan_Bk;
  fftw_plan iplan_C;


  //allocate smoothed grid array
  u_smooth = allocate_field_fftw_grid(grid_info.ndim, grid_info);

  //allocate work and transofrm
  Ck    = allocate_complex_fftw_grid(grid_info);
  Ak    = allocate_complex_fftw_grid(grid_info);
  Bk    = allocate_complex_fftw_grid(grid_info);


  //create the fftw plans
  plan_Ak = fftw_mpi_plan_dft_3d(grid_info.nx, grid_info.ny, grid_info.nz, Ak, Ak, world, FFTW_FORWARD,  FFTW_ESTIMATE);
  plan_Bk = fftw_mpi_plan_dft_3d(grid_info.nx, grid_info.ny, grid_info.nz, Bk, Bk, world, FFTW_FORWARD,  FFTW_ESTIMATE);
  iplan_C = fftw_mpi_plan_dft_3d(grid_info.nx, grid_info.ny, grid_info.nz, Ck, Ck, world, FFTW_BACKWARD, FFTW_ESTIMATE);


  //get complex version of B
  grid_copy_real_to_complex_in_place(kernel, Bk, grid_info);


  //perform the forward transform on B
  fftw_execute(plan_Bk);


  //at this stage, uk contains the Fourier transform of u
  for(l=0;l<grid_info.ndim;l++)
  {
    //get complex version of A
    grid_copy_real_to_complex_in_place(uin[l], Ak, grid_info);

    //perform the forward transform on A
    fftw_execute(plan_Ak);

    for(int i=0;i<nx_local;++i)
      for(int j=0;j<ny;++j)
	for(int k=0;k<nz;++k)
	{

	  //index of uk corresponding to k  

	  ijk = grid_complex_ijk(i,j,k,grid_info);
	
	  //perform convolution
	  //appears to be correct, check 10/31/2013
	  Ck[ijk][0] = (Ak[ijk][0]*Bk[ijk][0] - Ak[ijk][1]*Bk[ijk][1]) * scale;
	  Ck[ijk][1] = (Ak[ijk][0]*Bk[ijk][1] + Ak[ijk][1]*Bk[ijk][0]) * scale;
	}

    //perform the inverse transform of the derivative
    fftw_execute(iplan_C);
  
    //copy inverse transform into convolved field
    grid_copy_complex_to_real_in_place(Ck, u_smooth[l], grid_info);
  }

  //free the buffer memory
  fftw_free(Ak);
  fftw_free(Bk);
  fftw_free(Ck);

  //destroy the plans
  fftw_destroy_plan(plan_Ak);
  fftw_destroy_plan(plan_Bk);
  fftw_destroy_plan(iplan_C);

  //return the answer
  return u_smooth;
}
*/
/*! \fn void grid_copy_in_place(double *source, double *copy, FFTW_Grid_Info grid_info)
 *  \brief Produces a copy of a double grid in place. */
void grid_copy_in_place(double *source, double *copy, FFTW_Grid_Info grid_info)
{
	int i, j, k;

	int nx_local = grid_info.nx_local;
	int nx       = grid_info.nx;
	int ny       = grid_info.ny;
	int nz       = grid_info.nz;

	int ijk;

	//Copy source into copy.
	for(i=0;i<nx_local;++i)
		for(j=0;j<ny;++j)
			for(k=0;k<nz;++k)
			{
				//grid index
				ijk = grid_ijk(i,j,k,grid_info);

				//copy
				copy[ijk] = source[ijk];
			}
}

/*! \fn void grid_field_copy_in_place(double **source, double **copy, FFTW_Grid_Info grid_info)
 *  \brief Produces a copy of a double grid field in place. */
void grid_field_copy_in_place(double **source, double **copy, FFTW_Grid_Info grid_info)
{
	int i, j, k, n;

	int nx_local = grid_info.nx_local;
	int nx       = grid_info.nx;
	int ny       = grid_info.ny;
	int nz       = grid_info.nz;

	int ijk;
	int ndim = grid_info.ndim;

	//Copy source into copy.
	for(i=0;i<nx_local;++i)
		for(j=0;j<ny;++j)
			for(k=0;k<nz;++k)
			{
				//grid index
				ijk = grid_ijk(i,j,k,grid_info);

				//copy grid
				for(n=0;n<ndim;n++)
					copy[n][ijk] = source[n][ijk];
			}
}


/*! \fn void grid_scaled_copy_in_place(double *source, double scale, double *copy, FFTW_Grid_Info grid_info, MPI_Comm world) 
 *  \brief Produces a scaled copy of a double grid in place. */
void grid_scaled_copy_in_place(double *source, double scale, double *copy, FFTW_Grid_Info grid_info, MPI_Comm world) 
{
	int i, j, k;

	int nx_local = grid_info.nx_local;
	int nx       = grid_info.nx;
	int ny       = grid_info.ny;
	int nz       = grid_info.nz;

	int ijk;

	//Copy source into copy.
	for(i=0;i<nx_local;++i)
		for(j=0;j<ny;++j)
			for(k=0;k<nz;++k)
			{
				//grid index
				ijk = grid_ijk(i,j,k,grid_info);

				//grid copy
				copy[ijk] = scale*source[ijk];
			}
}

/*! \fn double *grid_copy_real_to_complex_in_place(double *source, fftw_complex *copy, FFTW_Grid_Info grid_info)
 *  \brief Produces a copy of a double grid in place into the real elements of a complex grid. */
void grid_copy_real_to_complex_in_place(double *source, fftw_complex *copy, FFTW_Grid_Info grid_info)
{
	int i, j, k;

	int nx_local = grid_info.nx_local;
	int nx       = grid_info.nx;
	int ny       = grid_info.ny;
	int nz       = grid_info.nz;

	int ijk;
	int ijkc;

	//Copy source into copy.
	for(i=0;i<nx_local;++i)
		for(j=0;j<ny;++j)
			for(k=0;k<nz;++k)
			{
				//real grid index
				ijk = grid_ijk(i,j,k,grid_info);

				//complex grid index
				ijkc = grid_complex_ijk(i,j,k,grid_info);

				//copy
				copy[ijkc][0] = source[ijk];
				copy[ijkc][1] = 0;
			}
}

/*! \fn double *grid_copy_complex_to_real_in_place(fftw_complex *source, double *copy, FFTW_Grid_Info grid_info)
 *  \brief Produces a copy of a double grid in place into the real elements of a complex grid. */
void grid_copy_complex_to_real_in_place(fftw_complex *source, double *copy, FFTW_Grid_Info grid_info)
{
	int i, j, k;

	int nx_local = grid_info.nx_local;
	int nx       = grid_info.nx;
	int ny       = grid_info.ny;
	int nz       = grid_info.nz;

	int ijk;
	int ijkc;

	//Copy source into copy.
	for(i=0;i<nx_local;++i)
		for(j=0;j<ny;++j)
			for(k=0;k<nz;++k)
			{
				//real grid index
				ijk = grid_ijk(i,j,k,grid_info);

				//complex grid index
				ijkc = grid_complex_ijk(i,j,k,grid_info);

				//copy
				copy[ijk] = source[ijkc][0];
			}
}

/*! \fn void grid_set_function(double (*setting_function)(double, double, double, void *), void *params, double *A, FFTW_Grid_Info grid_info)
 *  \brief Set the grid to the value of the supplied function at the cell centers. */
void grid_set_function(double (*setting_function)(double,double,double,void *), void *params, double *A, FFTW_Grid_Info grid_info)
{

	double x=0;
	double y=0;
	double z=0;

	int ijk;
	int i, j, k;

	int nx_local = grid_info.nx_local;
	int nx       = grid_info.nx;
	int ny       = grid_info.ny;
	int nz       = grid_info.nz;

	if(grid_info.ndim==2)
		nz=1;

	//Set the grid to setting_function
	for(i=0;i<nx_local;++i)
		for(j=0;j<ny;++j)
			for(k=0;k<nz;++k)
			{
				ijk = grid_ijk(i,j,k,grid_info);


				//cell centered positions
				x = ((double) grid_info.nx_local_start + i + 0.5)*grid_info.dx;
				y = ((double) j + 0.5)*grid_info.dy;
				if(grid_info.ndim==3)
					z = ((double) k + 0.5)*grid_info.dz;

				//face centered
				//x = ((double) grid_info.nx_local_start + i)*grid_info.dx;
				//y = ((double) j)*grid_info.dx;
				//z = ((double) k)*grid_info.dx;

				//set grid elements to value
				A[ijk] = setting_function(x,y,z,params);
			}
}


/*! \fn void grid_field_set_function(double (*setting_function)(double, double, double, int, void *), void *params, double **A, FFTW_Grid_Info grid_info)
 *  \brief Set the grid field to the value of the supplied function at the cell centers. */
void grid_field_set_function(double (*setting_function)(double,double,double,int,void *), void *params, double **A, FFTW_Grid_Info grid_info)
{
	double x=0;
	double y=0;
	double z=0;

	int ijk;
	int i, j, k;

	int nx_local = grid_info.nx_local;
	int nx       = grid_info.nx;
	int ny       = grid_info.ny;
	int nz       = grid_info.nz;

	int ndim = grid_info.ndim;

	for(int m=0;m<ndim;m++)
	{
		//Set the grid to setting_function
		for(i=0;i<nx_local;++i)
			for(j=0;j<ny;++j)
				for(k=0;k<nz;++k)
				{
					ijk = grid_ijk(i,j,k,grid_info);


					//cell centered positions
					x = ((double) grid_info.nx_local_start + i + 0.5)*grid_info.dx;
					y = ((double) j + 0.5)*grid_info.dx;
					z = ((double) k + 0.5)*grid_info.dx;

					//face centered
					//x = ((double) grid_info.nx_local_start + i)*grid_info.dx;
					//y = ((double) j)*grid_info.dx;
					//z = ((double) k)*grid_info.dx;

					//set grid elements to value
					A[m][ijk] = setting_function(x,y,z,m,params);
				}
	}
}

/*! \fn void grid_tensor_set_function(double (*setting_function)(double, double, double, int, int, void *), void *params, double ***A, FFTW_Grid_Info grid_info);
 *  \brief Set the grid tensor to the value of the supplied function at the cell centers. */
void grid_tensor_set_function(double (*setting_function)(double,double,double,int,int,void *), void *params, double ***A, FFTW_Grid_Info grid_info)
{
	double x=0;
	double y=0;
	double z=0;

	int ijk;
	int i, j, k;

	int nx_local = grid_info.nx_local;
	int nx       = grid_info.nx;
	int ny       = grid_info.ny;
	int nz       = grid_info.nz;

	int ndim = grid_info.ndim;

	for(int n=0;n<ndim;n++)
	{
		for(int m=0;m<ndim;m++)
		{
			//Set the grid to setting_function
			for(i=0;i<nx_local;++i)
				for(j=0;j<ny;++j)
					for(k=0;k<nz;++k)
					{
						ijk = grid_ijk(i,j,k,grid_info);


						//cell centered positions
						x = ((double) grid_info.nx_local_start + i + 0.5)*grid_info.dx;
						y = ((double) j + 0.5)*grid_info.dx;
						z = ((double) k + 0.5)*grid_info.dx;

						//face centered
						//x = ((double) grid_info.nx_local_start + i)*grid_info.dx;
						//y = ((double) j)*grid_info.dx;
						//z = ((double) k)*grid_info.dx;

						//set grid elements to value
						A[n][m][ijk] = setting_function(x,y,z,n,m,params);
					}
		}	
	}
}

/*! \fn void grid_rescale(double scale, double *A, FFTW_Grid_Info grid_info)
 *  \brief Rescale the grid by a factor of scale. */
void grid_rescale(double scale, double *A, FFTW_Grid_Info grid_info)
{

	int i, j, k;
	int ijk;

	int nx_local = grid_info.nx_local;
	int nx       = grid_info.nx;
	int ny       = grid_info.ny;
	int nz       = grid_info.nz;


	if(grid_info.ndim==2)
		nz = 1;

	//set the grid to value
	for(i=0;i<nx_local;++i)
		for(j=0;j<ny;++j)
			for(k=0;k<nz;++k)
			{
				//grid index
				ijk = grid_ijk(i,j,k,grid_info);

				//rescale grid elements by a factor scale
				A[ijk] *= scale;
			}
}

/*! \fn int grid_is_peak(int i, int j, int k, double *u, FFTW_Grid_Info grid_info)
 *  \brief Is this location a peak in u ? */
int grid_is_peak(int ii, int jj, int kk, double *u, FFTW_Grid_Info grid_info)
{
    double u_max;
    int ijk;
    int ijkt;

    ijk = grid_ijk(ii,jj,kk,grid_info);
    u_max = u[ijk];

    if(ii!=grid_info.nx_local)
    {
	for(int i=-1;i<=1;i++)
	    for(int j=-1;j<=1;j++)
		for(int k=-1;k<=1;k++)
		{
		    ijkt = grid_ijk(ii+i,jj+j,kk+k,grid_info);
		    if(u[ijk]>u_max)
			u_max = u[ijkt]; 
		}
    }else{
	return 0;
    }

    if(u_max > u[ijk])
    {
	return 0;
    }else{
	return 1;
    }
}

/*! \fn double *grid_mask_threshold(double threshold, double *u, FFTW_Grid_Info grid_info)
 *  \brief Creates a mask based on a threshold */
double *grid_mask_threshold(double threshold, double *u, FFTW_Grid_Info grid_info)
{

	//grid mask 
	double *mask;

	int i, j, k;
	int ijk;

	int nx_local = grid_info.nx_local;
	int nx       = grid_info.nx;
	int ny       = grid_info.ny;
	int nz       = grid_info.nz;


	//allocate mask
	mask = allocate_real_fftw_grid(grid_info);

	//set mask above threshold
	for(i=0;i<nx_local;++i)
		for(j=0;j<ny;++j)
			for(k=0;k<nz;++k)
			{
				//grid index
				ijk = grid_ijk(i,j,k,grid_info);

				//set mask
				if(u[ijk]>=threshold)
				{
					mask[ijk] = u[ijk];
				}else{
					mask[ijk] = 0;
				}
			}

	//return the answer

	return mask; 
}

/*! \fn double *grid_mask_range(double lower, double upper, double *u, FFTW_Grid_Info grid_info)
 *  \brief Creates a mask based on a range*/
double *grid_mask_range(double lower, double upper, double *u, FFTW_Grid_Info grid_info)
{

	//grid mask 
	double *mask;

	int i, j, k;
	int ijk;

	int nx_local = grid_info.nx_local;
	int nx       = grid_info.nx;
	int ny       = grid_info.ny;
	int nz       = grid_info.nz;


	//allocate mask
	mask = allocate_real_fftw_grid(grid_info);

	//set mask above threshold
	for(i=0;i<nx_local;++i)
		for(j=0;j<ny;++j)
			for(k=0;k<nz;++k)
			{
				//grid index
				ijk = grid_ijk(i,j,k,grid_info);

				//set mask
				if(u[ijk]>=lower && u[ijk]<=upper)
				{
					mask[ijk] = u[ijk];
				}else{
					mask[ijk] = 0;
				}
			}

	//return the answer

	return mask; 
}

/*! \fn double *grid_mask_peak_range(double lower, double upper, double *u, FFTW_Grid_Info grid_info)
 *  \brief Creates a mask based on a range and being a peak*/
double *grid_mask_peak_range(double lower, double upper, double *u, FFTW_Grid_Info grid_info)
{

	//grid mask 
	double *mask;

	int i, j, k;
	int ijk;

	int nx_local = grid_info.nx_local;
	int nx       = grid_info.nx;
	int ny       = grid_info.ny;
	int nz       = grid_info.nz;


	//allocate mask
	mask = allocate_real_fftw_grid(grid_info);

	//set mask above threshold
	for(i=0;i<nx_local;++i)
		for(j=0;j<ny;++j)
			for(k=0;k<nz;++k)
			{
				//grid index
				ijk = grid_ijk(i,j,k,grid_info);

				//set mask
				if(u[ijk]>=lower && u[ijk]<=upper && grid_is_peak(i,j,k,u,grid_info))
				{
					mask[ijk] = u[ijk];
				}else{
					mask[ijk] = 0;
				}
			}

	//return the answer

	return mask; 
}
/*! \fn int grid_mask_count_cells(double threshold, double *u, FFTW_Grid_Info grid_info, MPI_Comm world)
 *  \brief Count unmasked cells */
int grid_mask_count_cells(double *mask, FFTW_Grid_Info grid_info, MPI_Comm world)
{
	int count_local;
	int count;

	//count local cells
	count_local = grid_mask_count_local_cells(mask, grid_info, world);

	//Sum across processors
	MPI_Allreduce(&count_local,&count,1,MPI_INT,MPI_SUM,world);

	//return the answer	
	return count;
}

/*! \fn int grid_mask_count_local_cells(double threshold, double *u, FFTW_Grid_Info grid_info, MPI_Comm world)
 *  \brief Count unmasked local cells */
int grid_mask_count_local_cells(double *mask, FFTW_Grid_Info grid_info, MPI_Comm world)
{
	int i, j, k;
	int ijk;

	int nx_local = grid_info.nx_local;
	int nx       = grid_info.nx;
	int ny       = grid_info.ny;
	int nz       = grid_info.nz;

	int count_local = 0;

	if(grid_info.ndim==2)
		nz = 1;

	//set mask above threshold
	for(i=0;i<nx_local;++i)
		for(j=0;j<ny;++j)
			for(k=0;k<nz;++k)
			{
				//grid index
				ijk = grid_ijk(i,j,k,grid_info);

				//set mask
				if(mask[ijk]>0)
				{
					count_local++;
				}
			}

	return count_local;
}

/*! \fn int *grid_mask_local_cell_indices(double *mask, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world)
 *  \brief Indices to unmasked local cells */
int *grid_mask_local_cell_indices(double *mask, FFTW_Grid_Info grid_info, MPI_Comm world)
{
	int i, j, k;
	int ijk;

	int nx_local = grid_info.nx_local;
	int nx       = grid_info.nx;
	int ny       = grid_info.ny;
	int nz       = grid_info.nz;

	int count_local = 0;
	int noisy=0;
	int *indices;

	int nmask = grid_mask_count_local_cells(mask, grid_info, world);	//get the number of masked cells

	//allocate an int array
	indices = calloc_int_array(nmask);

	//set mask above threshold
	for(i=0;i<nx_local;++i)
		for(j=0;j<ny;++j)
			for(k=0;k<nz;++k)
			{
				ijk = grid_ijk(i,j,k,grid_info);
				//set mask
				if(mask[ijk]>0)
				{
					indices[count_local] = ijk;
					count_local++;
				}
			}

	//return the indices
	return indices;
}

/*! \fn int *grid_mask_direction_indices(int dim, double *mask, FFTW_Grid_Info grid_info, MPI_Comm world) 
 *  \brief Return locations of masked cells in dim direction */
int *grid_mask_direction_indices(int dim, double *mask, FFTW_Grid_Info grid_info, MPI_Comm world)
{
	int i, j, k;
	int ijk;

	int nx_local = grid_info.nx_local;
	int nx       = grid_info.nx;
	int ny       = grid_info.ny;
	int nz       = grid_info.nz;

	int count_local = 0;

	int *indices;
	int noisy=0;

	char variable_name[200];

	int nmask = grid_mask_count_local_cells(mask, grid_info, world);

	//allocate indices
	indices = calloc_int_array(nmask);

	//set mask above threshold
	for(i=0;i<nx_local;++i)
		for(j=0;j<ny;++j)
			for(k=0;k<nz;++k)
			{
				ijk = grid_ijk(i,j,k,grid_info);
				//set mask
				if(mask[ijk]>0)
				{
					switch(dim)
					{
						case 0: indices[count_local] = i;
							break;
						case 1: indices[count_local] = j;
							break;
						case 2: indices[count_local] = k;
							break;
					}
					count_local++;
				}
			}
	//return indices
	return indices;
}



/*! \fn double *grid_mask_apply(double *mask, double *u, FFTW_Grid_Info grid_info)
 *  \brief Copy a grid after applying a mask */
double *grid_mask_apply(double *mask, double *u, FFTW_Grid_Info grid_info)
{

	//grid copy
	double *copy;

	int i, j, k;
	int ijk;

	int nx_local = grid_info.nx_local;
	int nx       = grid_info.nx;
	int ny       = grid_info.ny;
	int nz       = grid_info.nz;

	//copy the mask
	copy = grid_copy(u, grid_info);

	//set apply the mask
	for(i=0;i<nx_local;++i)
		for(j=0;j<ny;++j)
			for(k=0;k<nz;++k)
			{
				//grid index
				ijk = grid_ijk(i,j,k,grid_info);

				//apply mask
				if(!(mask[ijk]>0))
				{
					copy[ijk] = 0;
				}
			}

	//return the answer
	return copy; 
}

/*! \fn double **grid_field_mask_apply(double *mask, double **u, FFTW_Grid_Info grid_info)
 *  \brief Copy a field after applying a mask */
double **grid_field_mask_apply(double *mask, double **u, FFTW_Grid_Info grid_info)
{
	int i, j, k;
	int ijk;

	int nx_local = grid_info.nx_local;
	int nx       = grid_info.nx;
	int ny       = grid_info.ny;
	int nz       = grid_info.nz;

	int ndim = grid_info.ndim;

	char variable_name[200];
	double **copy;

	if(grid_info.ndim==2)
		nz=1;
	//get copy
	copy = grid_field_copy(u, grid_info);

	//apply mask
	for(i=0;i<nx_local;++i)
		for(j=0;j<ny;++j)
			for(k=0;k<nz;++k)
			{
				//grid index
				ijk = grid_ijk(i,j,k,grid_info);
	
				//apply mask
				if(mask[ijk]>0)
					for(int n=0;n<ndim;n++)
						copy[n][ijk] = u[n][ijk];
			}

	//return the copy after applying the mask
	return copy;
}

/*! \fn double grid_surface_area(double *mask, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world)
 *  \brief Find the surface area of a mask */
double grid_surface_area(double *mask, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world)
{
    int ijk;	//cell
    int ijkt;	//adjacent cells
    int nx_local = grid_info.nx_local;
    int ny       = grid_info.ny;
    int nz       = grid_info.nz;

    double dx	= grid_info.dx;
    double dy	= grid_info.dy;
    double dz	= grid_info.dz;

    int dest;	//destination processor for communication
    int source;	//source processor for communication

    MPI_Status status;	//MPI status for communication


    double *ghost_cell_upper;	//ghost cells at i = nx_local
    double *ghost_cell_lower;	//ghost cells at i = -1

    //local surface area
    double A_local = 0;
    
    //total surface area
    double A_total = 0;

    //number of faces contributing to surface
    int n_faces;
    int face[6];

    //perimeter
    double s;

    //triangle sides
    double a, b, c;

    int n_tested = 0;

    //we have to check boundary cells separately
    //so start with inner cells
    for(int i=1;i<nx_local-1;i++)
	for(int j=0;j<ny;j++)
	    for(int k=0;k<nz;k++)
	    {

		//get cell index
		ijk = grid_ijk(i,j,k,grid_info);

		//consider only zero cells near the interface
		n_faces = 0;
		for(int l=0;l<6;l++)
		    face[l] = 0;

		if(mask[ijk]==0)
		{

		    ijkt = grid_ijk(i-1,j,k,grid_info);
		    if(mask[ijkt]!=0)
		    {
			n_faces++;
			face[0] = 1;
		    }

		    ijkt = grid_ijk(i+1,j,k,grid_info);
		    if(mask[ijkt]!=0)
		    {
			n_faces++;
			face[1] = 1;
		    }


		    ijkt = grid_ijk(i,j-1,k,grid_info);
		    if(mask[ijkt]!=0)
		    {
			n_faces++;
			face[2] = 1;
		    }

		    ijkt = grid_ijk(i,j+1,k,grid_info);
		    if(mask[ijkt]!=0)
		    {
			n_faces++;
			face[3] = 1;
		    }

		    ijkt = grid_ijk(i,j,k-1,grid_info);
		    if(mask[ijkt]!=0)
		    {
			n_faces++;
			face[4] = 1;
		    }

		    ijkt = grid_ijk(i,j,k+1,grid_info);
		    if(mask[ijkt]!=0)
		    {
			n_faces++;
			face[5] = 1;
		    }

		    //add area from this cell
		    switch(n_faces)
		    {

			//if no faces contribute, just break
			case 0:	break;


			//if 1 face contributes, add its area
			case 1:	

				//n_tested++;
				if(face[0]||face[1])
				{
				    A_local += dy * dz;
				    break;
				}
				if(face[2]||face[3])
				{
				    A_local += dx * dz;
				    break;
				}
				if(face[4]||face[5])
				{
				    A_local += dx * dy;
				    break;
				}


			//if 2 faces contribute and are adjacent, add their hypotenusal area
			case 2:	

				//opposing faces don't contribute
				if( (face[0]&&face[1])||(face[2]&&face[3])||(face[4]&&face[5]) )
				    break;
				
				if(face[0]||face[1])
				{
				    //an x-face is involved
				    if(face[2]||face[3])
				    {
					A_local += dz * sqrt( dx*dx + dy*dy);
				    }else{
					A_local += dy * sqrt( dx*dx + dz*dz);
				    }
				}else{
				    //only y and z faces are involved
				    A_local += dx * sqrt( dy*dy + dz*dz);
				}
				break;


			//if 3 faces contribute, then treat adjacent and opposing faces separately
			case 3:
				//opposing faces contribute only the non-opposing face
				if( (face[0]&&face[1])||(face[2]&&face[3])||(face[4]&&face[5]) )
				{

				    if( face[0]&&face[1] )
				    {
					if(face[2]||face[3])
					{
					    //return a z face
					    A_local += dx*dy;
					}else{
					    //return a y face
					    A_local += dz*dx;
					}
				    }

				    if( face[2]&&face[3] )
				    {
					if(face[0]||face[1])
					{
					    //return a z face
					    A_local += dx*dy;
					}else{
					    //return an x face
					    A_local += dy*dz;
					}
				    }

				    if( face[4]&&face[5] )
				    {
					if(face[0]||face[1])
					{
					    //return a y face
					    A_local += dx*dz;
					}else{
					    //return an x face
					    A_local += dy*dz;
					}
				    }

				}else{


				    //adjacent faces supply chamfer of triangle of hypotenuses

				    //find triangle sides
				    a = sqrt(dx*dx + dy*dy);
				    b = sqrt(dy*dy + dz*dz);
				    c = sqrt(dz*dz + dx*dx);

				    //half perimeter
				    s = 0.5*(a+b+c);

				    //Area of triangle
				    A_local += sqrt( s*(s-a)*(s-b)*(s-c) );



				}   
				break;


			//if 4 faces contribute, only add their hypotenusal area
			case 4:


				//tunnels don't contribute
				if(face[0]&&face[1]&&face[2]&&face[3])
				    break;
				if(face[2]&&face[3]&&face[4]&&face[5])
				    break;
				if(face[4]&&face[5]&&face[0]&&face[1])
				    break;

				//add open face

				if( face[0]&&face[1] )
				{
				    //surface runs diagonal across the x face
				    A_local += dx * sqrt(dy*dy + dz*dz);
				}else{
				    if( face[2]&&face[3] )
				    {
					//surface runs diagonal across the y face
					A_local += dy * sqrt(dz*dz + dx*dx);
				    }else{
					//surface runs diagonal across the z face
					A_local += dz * sqrt(dx*dx + dy*dy);
				    }
				}
				break;


			//if 5 contribute, only the open face is added
			case 5:
				if((!face[0])||(!face[1]))
				{
				    //add an x face
				    A_local += dy*dz;
				}else{
				    if((!face[2])||(!face[3]))
				    {
					//add a y face
					A_local += dz*dx;
				    }else{
					//add a z face
					A_local += dx*dy;
				    }
				}
				break;

			//if 0 or 6 faces, no area
			default:    break;
		    }
		}
	    }


    //now we have to check boundary cells
    //to do this, we use a sendreceive to get adjacent
    //cells across the boundary, then repeat the checks

    //allocate ghost cells
    ghost_cell_upper = calloc_double_array(ny*nz);
    ghost_cell_lower = calloc_double_array(ny*nz);

    //populate ghost cells
    for(int j=0;j<ny;j++)
	for(int k=0;k<nz;k++)
	{


	    //note nx_local can be zero!
	    if(nx_local)
	    {
		//upper cells to pass from this processor's lower boundary
		ijk = grid_ijk(0,j,k,grid_info);
		ghost_cell_upper[nz*j+k] = mask[ijk];

		//lower cells to pass from this processor's upper boundary
		ijk = grid_ijk(nx_local-1,j,k,grid_info);
		ghost_cell_lower[nz*j+k] = mask[ijk];
	    }else{
		//dummy cells
		ghost_cell_upper[nz*j+k] = 0;
		ghost_cell_lower[nz*j+k] = 0;
	    }
	}

    //sendrecv upper ghost cells
    source = myid+1;
    if(source>numprocs-1)
	source-=numprocs;
    dest = myid-1;
    if(dest<0)
	dest+=numprocs;

    MPI_Sendrecv_replace(ghost_cell_upper, ny*nz, MPI_DOUBLE, dest, myid, source, source, world, &status);
	

    //sendrecv lower ghost cells
    source = myid-1;
    if(source<0)
	source+=numprocs;
    dest = myid+1;
    if(dest>numprocs-1)
	dest-=numprocs;

    MPI_Sendrecv_replace(ghost_cell_lower, ny*nz, MPI_DOUBLE, dest, myid, source, source, world, &status);

    //do lower boundary first
    if(nx_local)
    for(int j=0;j<ny;j++)
	for(int k=0;k<nz;k++)
	{
		//get cell index
		ijk = grid_ijk(0,j,k,grid_info);

		//consider only zero cells near the interface
		n_faces = 0;
		for(int l=0;l<6;l++)
		    face[l] = 0;

		if(mask[ijk]==0)
		{
		    //ijkt = nz*j + k;
		    //if(ghost_cell_lower[ijkt]!=0)
		    ijkt = grid_ijk(1,j,k,grid_info);
		    if(mask[ijkt]!=0)
		    {
			n_faces++;
			face[0] = 1;
		    }

		    //ijkt = grid_ijk(1,j,k,grid_info);
		    //if(mask[ijkt]!=0)
		    ijkt = nz*j + k;
		    if(ghost_cell_lower[ijkt]!=0)
		    {
			n_faces++;
			face[1] = 1;
		    }


		    //ijkt = grid_ijk(0,j-1,k,grid_info);
		    ijkt = grid_ijk(0,j+1,k,grid_info);
		    if(mask[ijkt]!=0)
		    {
			n_faces++;
			face[2] = 1;
		    }

		    //ijkt = grid_ijk(0,j+1,k,grid_info);
		    ijkt = grid_ijk(0,j-1,k,grid_info);
		    if(mask[ijkt]!=0)
		    {
			n_faces++;
			face[3] = 1;
		    }

		    //ijkt = grid_ijk(0,j,k-1,grid_info);
		    ijkt = grid_ijk(0,j,k+1,grid_info);
		    if(mask[ijkt]!=0)
		    {
			n_faces++;
			face[4] = 1;
		    }

		    //ijkt = grid_ijk(0,j,k+1,grid_info);
		    ijkt = grid_ijk(0,j,k-1,grid_info);
		    if(mask[ijkt]!=0)
		    {
			n_faces++;
			face[5] = 1;
		    }

		    //add area from this cell
		    switch(n_faces)
		    {

			//if no faces contribute, just break
			case 0:	break;


			//if 1 face contributes, add its area
			case 1:	

				//n_tested++;
				if(face[0]||face[1])
				{
				    A_local += dy * dz;
				    break;
				}
				if(face[2]||face[3])
				{
				    A_local += dx * dz;
				    break;
				}
				if(face[4]||face[5])
				{
				    A_local += dx * dy;
				    break;
				}


			//if 2 faces contribute and are adjacent, add their hypotenusal area
			case 2:	

				//opposing faces don't contribute
				if( (face[0]&&face[1])||(face[2]&&face[3])||(face[4]&&face[5]) )
				    break;
				
				if(face[0]||face[1])
				{
				    //an x-face is involved
				    if(face[2]||face[3])
				    {
					A_local += dz * sqrt( dx*dx + dy*dy);
				    }else{
					A_local += dy * sqrt( dx*dx + dz*dz);
				    }
				}else{
				    //only y and z faces are involved
				    A_local += dx * sqrt( dy*dy + dz*dz);
				}
				break;


			//if 3 faces contribute, then treat adjacent and opposing faces separately
			case 3:
				//opposing faces contribute only the non-opposing face
				if( (face[0]&&face[1])||(face[2]&&face[3])||(face[4]&&face[5]) )
				{

				    if( face[0]&&face[1] )
				    {
					if(face[2]||face[3])
					{
					    //return a z face
					    A_local += dx*dy;
					}else{
					    //return a y face
					    A_local += dz*dx;
					}
				    }

				    if( face[2]&&face[3] )
				    {
					if(face[0]||face[1])
					{
					    //return a z face
					    A_local += dx*dy;
					}else{
					    //return an x face
					    A_local += dy*dz;
					}
				    }

				    if( face[4]&&face[5] )
				    {
					if(face[0]||face[1])
					{
					    //return a y face
					    A_local += dx*dz;
					}else{
					    //return an x face
					    A_local += dy*dz;
					}
				    }

				}else{


				    //adjacent faces supply chamfer of triangle of hypotenuses

				    //find triangle sides
				    a = sqrt(dx*dx + dy*dy);
				    b = sqrt(dy*dy + dz*dz);
				    c = sqrt(dz*dz + dx*dx);

				    //half perimeter
				    s = 0.5*(a+b+c);

				    //Area of triangle
				    A_local += sqrt( s*(s-a)*(s-b)*(s-c) );



				}   
				break;


			//if 4 faces contribute, only add their hypotenusal area
			case 4:


				//tunnels don't contribute
				if(face[0]&&face[1]&&face[2]&&face[3])
				    break;
				if(face[2]&&face[3]&&face[4]&&face[5])
				    break;
				if(face[4]&&face[5]&&face[0]&&face[1])
				    break;

				//add open face

				if( face[0]&&face[1] )
				{
				    //surface runs diagonal across the x face
				    A_local += dx * sqrt(dy*dy + dz*dz);
				}else{
				    if( face[2]&&face[3] )
				    {
					//surface runs diagonal across the y face
					A_local += dy * sqrt(dz*dz + dx*dx);
				    }else{
					//surface runs diagonal across the z face
					A_local += dz * sqrt(dx*dx + dy*dy);
				    }
				}
				break;


			//if 5 contribute, only the open face is added
			case 5:
				if((!face[0])||(!face[1]))
				{
				    //add an x face
				    A_local += dy*dz;
				}else{
				    if((!face[2])||(!face[3]))
				    {
					//add a y face
					A_local += dz*dx;
				    }else{
					//add a z face
					A_local += dx*dy;
				    }
				}
				break;

			//if 0 or 6 faces, no area
			default:    break;
		    }
		}
	}

    //do upper boundary second
    if(nx_local)
    for(int j=0;j<ny;j++)
	for(int k=0;k<nz;k++)
	{
		//get cell index
		ijk = grid_ijk(nx_local-1,j,k,grid_info);

		//consider only zero cells near the interface
		n_faces = 0;
		for(int l=0;l<6;l++)
		    face[l] = 0;

		if(mask[ijk]==0)
		{

		    ijkt = nz*j + k;
		    if(ghost_cell_upper[ijkt]!=0)
		    {
			n_faces++;
			face[0] = 1;
		    }

		    ijkt = grid_ijk(nx_local-2,j,k,grid_info);
		    if(mask[ijkt]!=0)
		    {
			n_faces++;
			face[1] = 1;
		    }


		    //ijkt = grid_ijk(nx_local-1,j-1,k,grid_info);
		    ijkt = grid_ijk(nx_local-1,j+1,k,grid_info);
		    if(mask[ijkt]!=0)
		    {
			n_faces++;
			face[2] = 1;
		    }

		    //ijkt = grid_ijk(nx_local-1,j+1,k,grid_info);
		    ijkt = grid_ijk(nx_local-1,j-1,k,grid_info);
		    if(mask[ijkt]!=0)
		    {
			n_faces++;
			face[3] = 1;
		    }

		    //ijkt = grid_ijk(nx_local-1,j,k-1,grid_info);
		    ijkt = grid_ijk(nx_local-1,j,k+1,grid_info);
		    if(mask[ijkt]!=0)
		    {
			n_faces++;
			face[4] = 1;
		    }

		    //ijkt = grid_ijk(nx_local-1,j,k+1,grid_info);
		    ijkt = grid_ijk(nx_local-1,j,k-1,grid_info);
		    if(mask[ijkt]!=0)
		    {
			n_faces++;
			face[5] = 1;
		    }

		    //add area from this cell
		    switch(n_faces)
		    {

			//if no faces contribute, just break
			case 0:	break;


			//if 1 face contributes, add its area
			case 1:	

				n_tested++;
				if(face[0]||face[1])
				{
				    A_local += dy * dz;
				    break;
				}
				if(face[2]||face[3])
				{
				    A_local += dx * dz;
				    break;
				}
				if(face[4]||face[5])
				{
				    A_local += dx * dy;
				    break;
				}


			//if 2 faces contribute and are adjacent, add their hypotenusal area
			case 2:	

				//opposing faces don't contribute
				if( (face[0]&&face[1])||(face[2]&&face[3])||(face[4]&&face[5]) )
				    break;
				
				if(face[0]||face[1])
				{
				    //an x-face is involved
				    if(face[2]||face[3])
				    {
					A_local += dz * sqrt( dx*dx + dy*dy);
				    }else{
					A_local += dy * sqrt( dx*dx + dz*dz);
				    }
				}else{
				    //only y and z faces are involved
				    A_local += dx * sqrt( dy*dy + dz*dz);
				}
				break;


			//if 3 faces contribute, then treat adjacent and opposing faces separately
			case 3:
				//opposing faces contribute only the non-opposing face
				if( (face[0]&&face[1])||(face[2]&&face[3])||(face[4]&&face[5]) )
				{

				    if( face[0]&&face[1] )
				    {
					if(face[2]||face[3])
					{
					    //return a y face
					    A_local += dz*dx;
					}else{
					    //return a z face
					    A_local += dx*dy;
					}
				    }

				    if( face[2]&&face[3] )
				    {
					if(face[0]||face[1])
					{
					    //return an x face
					    A_local += dy*dz;
					}else{
					    //return an z face
					    A_local += dy*dx;
					}
				    }

				    if( face[4]&&face[5] )
				    {
					if(face[0]||face[1])
					{
					    //return a x face
					    A_local += dy*dz;
					}else{
					    //return an y face
					    A_local += dx*dz;
					}
				    }

				}else{


				    //adjacent faces supply chamfer of triangle of hypotenuses

				    //find triangle sides
				    a = sqrt(dx*dx + dy*dy);
				    b = sqrt(dy*dy + dz*dz);
				    c = sqrt(dz*dz + dx*dx);

				    //half perimeter
				    s = 0.5*(a+b+c);

				    //Area of triangle
				    A_local += sqrt( s*(s-a)*(s-b)*(s-c) );
				}   
				break;


			//if 4 faces contribute, only add their hypotenusal area
			case 4:


				//tunnels don't contribute
				if(face[0]&&face[1]&&face[2]&&face[3])
				    break;
				if(face[2]&&face[3]&&face[4]&&face[5])
				    break;
				if(face[4]&&face[5]&&face[0]&&face[1])
				    break;

				//add open face

				if( face[0]&&face[1] )
				{
				    //surface runs diagonal across the x face
				    A_local += dx * sqrt(dy*dy + dz*dz);
				}else{
				    if( face[2]&&face[3] )
				    {
					//surface runs diagonal across the y face
					A_local += dy * sqrt(dz*dz + dx*dx);
				    }else{
					//surface runs diagonal across the z face
					A_local += dz * sqrt(dx*dx + dy*dy);
				    }
				}
				break;


			//if 5 contribute, only the open face is added
			case 5:
				if((!face[0])||(!face[1]))
				{
				    //add an x face
				    A_local += dy*dz;
				}else{
				    if((!face[2])||(!face[3]))
				    {
					//add a y face
					A_local += dz*dx;
				    }else{
					//add a z face
					A_local += dx*dy;
				    }
				}
				break;

			//if 0 or 6 faces, no area
			default:    break;
		    }
		}
	}

    //free ghost cells
    free(ghost_cell_upper);
    free(ghost_cell_lower);

    /*for(int i=0;i<numprocs;i++)
    {
	if(myid==i)
	{
	    printf("myid %d A_local %e n_tested %d nx_local %d\n",myid,A_local,n_tested,nx_local);
	    fflush(stdout);
	}
	MPI_Barrier(world);
    }*/


    //sum area across processors
    MPI_Allreduce(&A_local,&A_total,1,MPI_DOUBLE,MPI_SUM,world);


    //find the total surface area
    return A_total;

}


/*! \fn double grid_surface_area_directional(double *mask, double *dir, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world)
 *  \brief Find the approximate surface area of a mask whose normal has a positive dot product with supplied vector*/
double grid_surface_area_directional(double *mask, double *dir, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world)
{
    int ijk;	//cell
    int ijkt;	//adjacent cells
    int nx_local = grid_info.nx_local;
    int nx_local_start = grid_info.nx_local_start;
    int ny       = grid_info.ny;
    int nz       = grid_info.nz;

    double dx	= grid_info.dx;
    double dy	= grid_info.dy;
    double dz	= grid_info.dz;

    int dest;	//destination processor for communication
    int source;	//source processor for communication

    MPI_Status status;	//MPI status for communication


    double *ghost_cell_upper;	//ghost cells at i = nx_local
    double *ghost_cell_lower;	//ghost cells at i = -1

    //local surface area
    double A_local = 0;
    
    //total surface area
    double A_total = 0;

    //number of faces contributing to surface
    int n_faces;
    int face[6];

    //perimeter
    double s;

    //triangle sides
    double a, b, c;

    int n_tested = 0;

    double dir_mag  = 0;
    double d_norm[6];  //dot product of face norm with dir 
    double dir_norm[6];  //direction of input vector

    double face_vector_A[3];  //vector area of the face
    double a_face[3];	      //first vector of the face
    double b_face[3];	      //second vector of the face

    int fcase[6] = {0,0,0,0,0,0};
    double dcase[6] = {0,0,0,0,0,0};

    double A_face;

    for(int i=0;i<3;i++)
	dir_mag += dir[i]*dir[i];
    dir_mag = sqrt(dir_mag);

    for(int i=0;i<3;i++)
	dir_norm[i] = dir[i]/dir_mag;
    
    for(int i=0;i<6;i++)
	d_norm[i] = 0;

    if(dir[0]>0)
    {
	d_norm[0] =  dir_norm[0];

    }else if(dir[0]<0){

	d_norm[1] = -dir_norm[0];
    }

    if(dir[1]>0)
    {
	d_norm[2] =  dir_norm[1];

    }else if(dir[1]<0){

	d_norm[3] = -dir_norm[1];
    }

    if(dir[2]>0)
    {
	d_norm[4] =  dir_norm[2];

    }else if(dir[2]<0){

	d_norm[5] = -dir_norm[2];
    }
    //printf("dir %e %e %e\n",dir[0],dir[1],dir[2]);


    //we have to check boundary cells separately
    //so start with inner cells
    for(int i=1;i<nx_local-1;i++)
	for(int j=0;j<ny;j++)
	    for(int k=0;k<nz;k++)
	    {

		//get cell index
		ijk = grid_ijk(i,j,k,grid_info);

		//consider only zero cells near the interface
		n_faces = 0;
		for(int l=0;l<6;l++)
		    face[l] = 0;

		if(mask[ijk]==0)
		{

		    //ijkt = grid_ijk(i-1,j,k,grid_info);
		    ijkt = grid_ijk(i+1,j,k,grid_info);
		    if(mask[ijkt]!=0)
		    {
			n_faces++;
			face[0] = 1;
		    }

		    ijkt = grid_ijk(i-1,j,k,grid_info);
		    //ijkt = grid_ijk(i+1,j,k,grid_info);
		    if(mask[ijkt]!=0)
		    {
			n_faces++;
			face[1] = 1;
		    }


		    ijkt = grid_ijk(i,j+1,k,grid_info);
		    //ijkt = grid_ijk(i,j-1,k,grid_info);
		    if(mask[ijkt]!=0)
		    {
			n_faces++;
			face[2] = 1;
		    }

		    ijkt = grid_ijk(i,j-1,k,grid_info);
		    //ijkt = grid_ijk(i,j+1,k,grid_info);
		    if(mask[ijkt]!=0)
		    {
			n_faces++;
			face[3] = 1;
		    }

		    ijkt = grid_ijk(i,j,k+1,grid_info);
		    //ijkt = grid_ijk(i,j,k-1,grid_info);
		    if(mask[ijkt]!=0)
		    {
			n_faces++;
			face[4] = 1;
		    }

		    ijkt = grid_ijk(i,j,k-1,grid_info);
		    //ijkt = grid_ijk(i,j,k+1,grid_info);
		    if(mask[ijkt]!=0)
		    {
			n_faces++;
			face[5] = 1;
		    }


		    //surface area contribution of the face
		    //in question
		    A_face = 0;
		    for(int l=0;l<3;l++)
		    {
			face_vector_A[l] = 0;
			a_face[l] = 0;
			b_face[l] = 0;
		    }

		    //add area from this cell
		    switch(n_faces)
		    {

			//if no faces contribute, just break
			case 0:	fcase[0]++;
				break;

			//if 1 face contributes, add its area
			case 1:	

				//n_tested++;
				if(face[0])
				    face_vector_A[0] = -dy * dz;
				
				if(face[1])
				    face_vector_A[0] =  dy * dz;
				
				if(face[2])
				    face_vector_A[1] = -dz * dx;

				if(face[3])
				    face_vector_A[1] =  dz * dx;

				if(face[4])
				    face_vector_A[2] = -dx * dy;
				
				if(face[5])
				    face_vector_A[2] =  dx * dy;
				
				A_face = vector_dot_product(face_vector_A,dir_norm,3);
				if(A_face>0)
				{
				    fcase[1]++;
				    dcase[1] += A_face;
				}
				break;


			//if 2 faces contribute and are adjacent, add their hypotenusal area
			case 2:	fcase[2]++;

				//opposing faces don't contribute
				if( (face[0]&&face[1])||(face[2]&&face[3])||(face[4]&&face[5]) )
				    break;

				//add the hypotenuse only if the dot product is positive
				if(face[0]) //+x face
				{
					if(face[2])
					{
						//+x/+y, checked 05/22
						a_face[2] =  dz;

						b_face[0] = -dx;
						b_face[1] =  dy;

					}else if(face[3]){

						//+x/-y, checked 05/22
						a_face[2] =  dz;

						b_face[0] =  dx;
						b_face[1] =  dy;
					}

					if(face[4])
					{
						//+x/+z,checked 05/22
						a_face[1] =  dy;

						b_face[0] =  dx;
						b_face[2] = -dz;

					}else if(face[5]){

						//+x/-z, checked 05/22
						a_face[1] =  dy;

						b_face[0] =  -dx;
						b_face[2] =  -dz;
					}

				}else if(face[1]){ //-x face

					if(face[2])
					{
						//-x/+y, checked 05/22
						a_face[2] =  dz;

						b_face[0] = -dx;
						b_face[1] = -dy;

					}else if(face[3]){

						//-x/-y, checked 05/22
						a_face[2] =  dz;

						b_face[0] =  dx;
						b_face[1] = -dy;

					}

					if(face[4])
					{
						//-x/+z, checked 05/22
						a_face[1] =  dy;

						b_face[0] =  dx;
						b_face[2] =  dz;

					}else if(face[5]){

						//-x/-z, checked 05/22
						a_face[1] =  dy;

						b_face[0] = -dx;
						b_face[2] =  dz;
					}

				}else if(face[2]){  //+y face, no x-face

					//+y face 
					if(face[4])
					{
						//+y/+z, checked 05/22
						a_face[0] =  dx;

						b_face[1] = -dy;
						b_face[2] =  dz;

					}else if(face[5]){

						//+y/-z, checked 05/22
						a_face[0] =  dx;

						b_face[1] =  dy;
						b_face[2] =  dz;
					}

				}else if(face[3]){ //-y face, no x-face

					//-y face
					if(face[4])
					{
						//-y/+z, checked 05/22
						a_face[0] =  dx;

						b_face[1] = -dy;
						b_face[2] = -dz;

					}else if(face[5]){

						//-y/-z, checked 05/22
						a_face[0] =  dx;

						b_face[1] =  dy;
						b_face[2] = -dz;
					}

				}
				vector_cross_product_in_place(&face_vector_A[0],a_face,b_face,3);
		
				A_face = vector_dot_product(face_vector_A,dir_norm,3);
				if(A_face>0)
				    dcase[2] += A_face;
				break;

			//if 3 faces contribute, then treat adjacent and opposing faces separately
			case 3: fcase[3]++;

				//opposing faces contribute only the non-opposing face
				if( (face[0]&&face[1])||(face[2]&&face[3])||(face[4]&&face[5]) )
				{

				    if( face[0]&&face[1] )
				    {
					//y faces
					if(face[2])
					    face_vector_A[1] = -dz*dx;
					if(face[3])
					    face_vector_A[1] =  dz*dx;
	    
					//z faces
					if(face[4])
					    face_vector_A[2] = -dx*dy;
					if(face[5])
					    face_vector_A[2] =  dx*dy;
				    }

				    if( face[2]&&face[3] )
				    {
					//x faces
					if(face[0])
					    face_vector_A[0] = -dy*dz;
					if(face[1])
					    face_vector_A[0] =  dy*dz;

					//z faces
					if(face[4])
					    face_vector_A[2] = -dx*dy;
					if(face[5])
					    face_vector_A[2] =  dx*dy;
				    }

				    if( face[4]&&face[5] )
				    {
					//x faces
					if(face[0])
					    face_vector_A[0] = -dy*dz;
					if(face[1])
					    face_vector_A[0] =  dy*dz;

					//y faces
					if(face[2])
					    face_vector_A[1] = -dz*dx;
					if(face[3])
					    face_vector_A[1] =  dz*dx;
				    }

				}else{


				    //adjacent faces supply chamfer of triangle of hypotenuses
				    //a_face will lie in the inner most conditioned face

				    //there must be an x face
				    if(face[0])	//+x
				    {
					if(face[2]) //+y
					{
					    if(face[4])	//+z
					    {
						//+x +y +z; triangle points down to (dx, dy, 0)
						a_face[0] =  0;
						a_face[1] = -dy;
						a_face[2] =  dz;

						b_face[0] = -dx;
						b_face[1] =  0;
						b_face[2] =  dz;

					    }else{	//-z

						//+x +y -z;  triangle points up to (dx, dy, dz)
						a_face[0] =  dx;
						a_face[1] = -dy;
						a_face[2] =  0;

						b_face[0] =  dx;
						b_face[1] =  0;
						b_face[2] =  dz;
					    }
					}else{	//-y
					    if(face[4])	//+z
					    {
						//+x -y +z; triangle points down to (dx, 0, 0)
						a_face[0] =  dx;
						a_face[1] =  dy;
						a_face[2] =  0;

						b_face[0] =  dx;
						b_face[1] =  0;
						b_face[2] = -dz;

					    }else{	//-z ?

						//+x -y -z;  triangle points up to (dx, 0, dz)
						a_face[0] =  0;
						a_face[1] =  dy;
						a_face[2] = -dz;

						b_face[0] = -dx;
						b_face[1] =  0;
						b_face[2] = -dz;
					    }
					}
				    }else{	//-x
					if(face[2]) //+y
					{
					    if(face[4])	//+z
					    {
						//-x +y +z; triangle points down to (0, dy, 0)
						a_face[0] =  dx;
						a_face[1] =  0;
						a_face[2] =  dz;

						b_face[0] =  0;
						b_face[1] = -dy;
						b_face[2] =  dz;

					    }else{	//-z

						//-x +y -z;  triangle points up to (0, dy, dz)
						a_face[0] =  dx;
						a_face[1] =  dy;
						a_face[2] =  0;

						b_face[0] =  0;
						b_face[1] =  dy;
						b_face[2] =  dz;
					    }
					}else{	//-y
					    if(face[4])	//+z
					    {
						//-x -y +z; triangle points down to (0, 0, 0)
						a_face[0] =  0;
						a_face[1] =  dy;
						a_face[2] =  dz;

						b_face[0] =  dx;
						b_face[1] =  0;
						b_face[2] =  dz;

					    }else{	//-z

						//-x -y -z;  triangle points up to (0, 0, dz)
						a_face[0] =  0;
						a_face[1] = -dy;
						a_face[2] =  dz;

						b_face[0] =  dx;
						b_face[1] = -dy;
						b_face[2] =  0;
					    }
					}
				    }

				    //vector area of parallelogram
				    vector_cross_product_in_place(&face_vector_A[0],a_face,b_face,3);

				    //vector area of triangle needs to be scaled by 1/2
				    //for(int l=0;l<3;l++)
				//	face_vector_A[l] *= 0.5;
				    //however, we often miss about the same area in the corners
				}   
				
				A_face = vector_dot_product(face_vector_A,dir_norm,3);
				if(A_face>0)
				    dcase[3] += A_face;
				break;


			//if 4 faces contribute, only add their hypotenusal area
			case 4: fcase[4]++;


				//tunnels don't contribute
				if(face[0]&&face[1]&&face[2]&&face[3])
				    break;
				if(face[2]&&face[3]&&face[4]&&face[5])
				    break;
				if(face[4]&&face[5]&&face[0]&&face[1])
				    break;

				//add open face

				if( face[0]&&face[1] )
				{

				    //surface runs diagonal across the x face
				    if(face[2]&&face[4])
				    {
					//+y and +z
					a_face[0] =  dx;
					a_face[1] =  0;
					a_face[2] =  0;

					b_face[0] =  0;
					b_face[1] = -dy;
					b_face[2] =  dz;

				    }else if(face[2]&&face[5]){

					//+y and -z
					a_face[0] =  dx;
					a_face[1] =  0;
					a_face[2] =  0;

					b_face[0] =  0;
					b_face[1] =  dy;
					b_face[2] =  dz;

				    }else if(face[3]&&face[4]){

					//-y and +z
					a_face[0] = -dx;
					a_face[1] =  0;
					a_face[2] =  0;

					b_face[0] =  0;
					b_face[1] = -dy;
					b_face[2] = -dz;

				    }else{

					//-y and -z
					a_face[0] = -dx;
					a_face[1] =  0;
					a_face[2] =  0;

					b_face[0] =  0;
					b_face[1] = -dy;
					b_face[2] =  dz;
				    }

				}else{

				    if( face[2]&&face[3] )
				    {
					//surface runs diagonal across the y face


					if(face[0]&&face[4])
					{
					   //+x and +z
					    a_face[0] =  0;
					    a_face[1] =  dy;
					    a_face[2] =  0;

					    b_face[0] =  dx;
					    b_face[1] =  0;
					    b_face[2] = -dz;

					}else if(face[0]&&face[5]){

					   //+x and -z
					    a_face[0] =  0;
					    a_face[1] =  dy;
					    a_face[2] =  0;

					    b_face[0] = -dx; 
					    b_face[1] =  0;
					    b_face[2] = -dz;

					}else if(face[1]&&face[4]){

					   //-x and +z
					    a_face[0] =  0;
					    a_face[1] = -dy;
					    a_face[2] =  0;

					    b_face[0] = -dx;
					    b_face[1] =  0;
					    b_face[2] = -dz;

					}else{

					    //-x and -z
					    a_face[0] =  0;
					    a_face[1] = -dy;
					    a_face[2] =  0;

					    b_face[0] =  dx;
					    b_face[1] =  0;
					    b_face[2] = -dz;
					}

				    }else{
					//surface runs diagonal across the z face

					if(face[0]&&face[2])
					{
					   //+x and +y
					    a_face[0] =  0;
					    a_face[1] =  0;
					    a_face[2] =  dz;

					    b_face[0] = -dx;
					    b_face[1] =  dy;
					    b_face[2] =  0;

					}else if(face[0]&&face[3]){

					   //+x and -y
					    a_face[0] =  0;
					    a_face[1] =  0;
					    a_face[2] =  dz;

					    b_face[0] =  dx;
					    b_face[1] =  dy;
					    b_face[2] =  0;

					}else if(face[1]&&face[2]){

					   //-x and +y
					    a_face[0] =  0;
					    a_face[1] =  0;
					    a_face[2] = -dz;

					    b_face[0] =  dx;
					    b_face[1] =  dy;
					    b_face[2] =  0;

					}else{

					    //-x and -y
					    a_face[0] =  0;
					    a_face[1] =  0;
					    a_face[2] = -dz;

					    b_face[0] = -dx;
					    b_face[1] =  dy;
					    b_face[2] =  0;
					}

				    }
				}

				//find vector area of parallelogram
				vector_cross_product_in_place(&face_vector_A[0],a_face,b_face,3);

				A_face = vector_dot_product(face_vector_A,dir_norm,3);
				if(A_face>0)
				    dcase[4] += A_face;
				break;


			//if 5 contribute, only the open face is added
			case 5: fcase[5]++;

				if(!face[0])
				    face_vector_A[0] =  dy*dz;
				if(!face[1])
				    face_vector_A[0] = -dy*dz;

				if(!face[2])
				    face_vector_A[1] =  dx*dz;
				if(!face[3])
				    face_vector_A[1] = -dx*dz;

				if(!face[4])
				    face_vector_A[2] =  dx*dy;
				if(!face[5])
				    face_vector_A[2] = -dx*dy;

				A_face = vector_dot_product(face_vector_A,dir_norm,3);
				if(A_face>0)
				    dcase[5] += A_face;
				break;

			//if 0 or 6 faces, no area
			default:    break;
		    }

		    //Add the surface area
		    A_face = vector_dot_product(face_vector_A,dir_norm,3);
		    if(A_face>0)
			A_local += A_face;
		}
	    }

/*
    for(int i=0;i<numprocs;i++)
    {
	if(myid==i)
	{
	    for(int j=0;j<6;j++)
		printf("myid %d fcase[%d] = %d\tdcase[%d] = %e\n",i,j,fcase[j],j,dcase[j]);
	}
	MPI_Barrier(world);
    }
*/

    //now we have to check boundary cells
    //to do this, we use a sendreceive to get adjacent
    //cells across the boundary, then repeat the checks

    //allocate ghost cells
    ghost_cell_upper = calloc_double_array(ny*nz);
    ghost_cell_lower = calloc_double_array(ny*nz);

    //populate ghost cells
    for(int j=0;j<ny;j++)
	for(int k=0;k<nz;k++)
	{


	    //note nx_local can be zero!
	    if(nx_local)
	    {
		//upper cells to pass from this processor's lower boundary
		ijk = grid_ijk(0,j,k,grid_info);
		ghost_cell_upper[nz*j+k] = mask[ijk];

		//lower cells to pass from this processor's upper boundary
		ijk = grid_ijk(nx_local-1,j,k,grid_info);
		ghost_cell_lower[nz*j+k] = mask[ijk];

		//if(nx_local_start+nx_local-1==47)
		//	printf("j %d k %d ghost %e\n",j,k,ghost_cell_lower[nz*j+k]);
	    }else{
		//dummy cells
		ghost_cell_upper[nz*j+k] = 0;
		ghost_cell_lower[nz*j+k] = 0;
	    }
	}

    //sendrecv upper ghost cells
    source = myid+1;
    if(source>numprocs-1)
	source-=numprocs;
    dest = myid-1;
    if(dest<0)
	dest+=numprocs;

    MPI_Sendrecv_replace(ghost_cell_upper, ny*nz, MPI_DOUBLE, dest, myid, source, source, world, &status);
	

    //sendrecv lower ghost cells
    source = myid-1;
    if(source<0)
	source+=numprocs;
    dest = myid+1;
    if(dest>numprocs-1)
	dest-=numprocs;

    MPI_Sendrecv_replace(ghost_cell_lower, ny*nz, MPI_DOUBLE, dest, myid, source, source, world, &status);


/*
    for(int i=0;i<numprocs;i++)
    {
	if(myid==i)
	{
	    printf("Before lower myid %d A_local %e n_tested %d nx_local %d\n",myid,A_local,n_tested,nx_local);
	    fflush(stdout);
	}
	MPI_Barrier(world);
    }
*/
	/*if(nx_local_start==48)
	{
    		for(int j=0;j<ny;j++)
			for(int k=0;k<nz;k++)
				if(ghost_cell_lower[nz*j+k]>0)
					printf("id %d j %d k %d gc %e\n",myid,j,k,ghost_cell_lower[nz*j+k]);
	}*/

    //do lower boundary first
    if(nx_local)
    for(int j=0;j<ny;j++)
	for(int k=0;k<nz;k++)
	{
		//get cell index
		ijk = grid_ijk(0,j,k,grid_info);

		//consider only zero cells near the interface
		n_faces = 0;
		for(int l=0;l<6;l++)
		    face[l] = 0;

		if(mask[ijk]==0)
		{
		    ijkt = grid_ijk(1,j,k,grid_info);
		    if(mask[ijkt]!=0)
		    {
			n_faces++;
			face[0] = 1;
		    }

		    ijkt = nz*j + k;
		    if(ghost_cell_lower[ijkt]!=0)
		    {
			n_faces++;
			face[1] = 1;
		    }


		    ijkt = grid_ijk(0,j+1,k,grid_info);
		    if(mask[ijkt]!=0)
		    {
			n_faces++;
			face[2] = 1;
		    }

		    ijkt = grid_ijk(0,j-1,k,grid_info);
		    if(mask[ijkt]!=0)
		    {
			n_faces++;
			face[3] = 1;
		    }

		    ijkt = grid_ijk(0,j,k+1,grid_info);
		    if(mask[ijkt]!=0)
		    {
			n_faces++;
			face[4] = 1;
		    }

		    ijkt = grid_ijk(0,j,k-1,grid_info);
		    if(mask[ijkt]!=0)
		    {
			n_faces++;
			face[5] = 1;
		    }

		    //surface area contribution of the face
		    //in question
		    A_face = 0;
		    for(int l=0;l<3;l++)
		    {
			face_vector_A[l] = 0;
			a_face[l] = 0;
			b_face[l] = 0;
		    }

		    //add area from this cell
		    switch(n_faces)
		    {
			//if no faces contribute, just break
			case 0:	fcase[0]++;
				break;

			//if 1 face contributes, add its area
			case 1:	fcase[1]++;

				//n_tested++;
				if(face[0])
				    face_vector_A[0] = -dy * dz;
				
				if(face[1])
				    face_vector_A[0] =  dy * dz;
				
				if(face[2])
				    face_vector_A[1] = -dz * dx;

				if(face[3])
				    face_vector_A[1] =  dz * dx;

				if(face[4])
				    face_vector_A[2] = -dx * dy;
				
				if(face[5])
				    face_vector_A[2] =  dx * dy;
				
				A_face = vector_dot_product(face_vector_A,dir_norm,3);
				if(A_face>0)
				    dcase[1] += A_face;
				break;


			//if 2 faces contribute and are adjacent, add their hypotenusal area
			case 2:	fcase[2]++;

				//opposing faces don't contribute
				if( (face[0]&&face[1])||(face[2]&&face[3])||(face[4]&&face[5]) )
				    break;

				//add the hypotenuse only if the dot product is positive
				if(face[0]) //+x face
				{
					if(face[2])
					{
						//+x/+y, checked 05/22
						a_face[2] =  dz;

						b_face[0] = -dx;
						b_face[1] =  dy;

					}else if(face[3]){

						//+x/-y, checked 05/22
						a_face[2] =  dz;

						b_face[0] =  dx;
						b_face[1] =  dy;
					}

					if(face[4])
					{
						//+x/+z,checked 05/22
						a_face[1] =  dy;

						b_face[0] =  dx;
						b_face[2] = -dz;

					}else if(face[5]){

						//+x/-z, checked 05/22
						a_face[1] =  dy;

						b_face[0] =  -dx;
						b_face[2] =  -dz;
					}

				}else if(face[1]){ //-x face

					if(face[2])
					{
						//-x/+y, checked 05/22
						a_face[2] =  dz;

						b_face[0] = -dx;
						b_face[1] = -dy;

					}else if(face[3]){

						//-x/-y, checked 05/22
						a_face[2] =  dz;

						b_face[0] =  dx;
						b_face[1] = -dy;

					}

					if(face[4])
					{
						//-x/+z, checked 05/22
						a_face[1] =  dy;

						b_face[0] =  dx;
						b_face[2] =  dz;

					}else if(face[5]){

						//-x/-z, checked 05/22
						a_face[1] =  dy;

						b_face[0] = -dx;
						b_face[2] =  dz;
					}

				}else if(face[2]){  //+y face, no x-face

					//+y face 
					if(face[4])
					{
						//+y/+z, checked 05/22
						a_face[0] =  dx;

						b_face[1] = -dy;
						b_face[2] =  dz;

					}else if(face[5]){

						//+y/-z, checked 05/22
						a_face[0] =  dx;

						b_face[1] =  dy;
						b_face[2] =  dz;
					}

				}else if(face[3]){ //-y face, no x-face

					//-y face
					if(face[4])
					{
						//-y/+z, checked 05/22
						a_face[0] =  dx;

						b_face[1] = -dy;
						b_face[2] = -dz;

					}else if(face[5]){

						//-y/-z, checked 05/22
						a_face[0] =  dx;

						b_face[1] =  dy;
						b_face[2] = -dz;
					}

				}
				vector_cross_product_in_place(&face_vector_A[0],a_face,b_face,3);
		
				A_face = vector_dot_product(face_vector_A,dir_norm,3);
				if(A_face>0)
				    dcase[2] += A_face;
				break;

			//if 3 faces contribute, then treat adjacent and opposing faces separately
			case 3: fcase[3]++;

				//opposing faces contribute only the non-opposing face
				if( (face[0]&&face[1])||(face[2]&&face[3])||(face[4]&&face[5]) )
				{

				    if( face[0]&&face[1] )
				    {
					//y faces
					if(face[2])
					    face_vector_A[1] = -dz*dx;
					if(face[3])
					    face_vector_A[1] =  dz*dx;
	    
					//z faces
					if(face[4])
					    face_vector_A[2] = -dx*dy;
					if(face[5])
					    face_vector_A[2] =  dx*dy;
				    }

				    if( face[2]&&face[3] )
				    {
					//x faces
					if(face[0])
					    face_vector_A[0] = -dy*dz;
					if(face[1])
					    face_vector_A[0] =  dy*dz;

					//z faces
					if(face[4])
					    face_vector_A[2] = -dx*dy;
					if(face[5])
					    face_vector_A[2] =  dx*dy;
				    }

				    if( face[4]&&face[5] )
				    {
					//x faces
					if(face[0])
					    face_vector_A[0] = -dy*dz;
					if(face[1])
					    face_vector_A[0] =  dy*dz;

					//y faces
					if(face[2])
					    face_vector_A[1] = -dz*dx;
					if(face[3])
					    face_vector_A[1] =  dz*dx;
				    }

				}else{


				    //adjacent faces supply chamfer of triangle of hypotenuses
				    //a_face will lie in the inner most conditioned face

				    //there must be an x face
				    if(face[0])	//+x
				    {
					if(face[2]) //+y
					{
					    if(face[4])	//+z
					    {
						//+x +y +z; triangle points down to (dx, dy, 0)
						a_face[0] =  0;
						a_face[1] = -dy;
						a_face[2] =  dz;

						b_face[0] = -dx;
						b_face[1] =  0;
						b_face[2] =  dz;

					    }else{	//-z

						//+x +y -z;  triangle points up to (dx, dy, dz)
						a_face[0] =  dx;
						a_face[1] = -dy;
						a_face[2] =  0;

						b_face[0] =  dx;
						b_face[1] =  0;
						b_face[2] =  dz;
					    }
					}else{	//-y
					    if(face[4])	//+z
					    {
						//+x -y +z; triangle points down to (dx, 0, 0)
						a_face[0] =  dx;
						a_face[1] =  dy;
						a_face[2] =  0;

						b_face[0] =  dx;
						b_face[1] =  0;
						b_face[2] = -dz;

					    }else{	//-z ?

						//+x -y -z;  triangle points up to (dx, 0, dz)
						a_face[0] =  0;
						a_face[1] =  dy;
						a_face[2] = -dz;

						b_face[0] = -dx;
						b_face[1] =  0;
						b_face[2] = -dz;
					    }
					}
				    }else{	//-x
					if(face[2]) //+y
					{
					    if(face[4])	//+z
					    {
						//-x +y +z; triangle points down to (0, dy, 0)
						a_face[0] =  dx;
						a_face[1] =  0;
						a_face[2] =  dz;

						b_face[0] =  0;
						b_face[1] = -dy;
						b_face[2] =  dz;

					    }else{	//-z

						//-x +y -z;  triangle points up to (0, dy, dz)
						a_face[0] =  dx;
						a_face[1] =  dy;
						a_face[2] =  0;

						b_face[0] =  0;
						b_face[1] =  dy;
						b_face[2] =  dz;
					    }
					}else{	//-y
					    if(face[4])	//+z
					    {
						//-x -y +z; triangle points down to (0, 0, 0)
						a_face[0] =  0;
						a_face[1] =  dy;
						a_face[2] =  dz;

						b_face[0] =  dx;
						b_face[1] =  0;
						b_face[2] =  dz;

					    }else{	//-z

						//-x -y -z;  triangle points up to (0, 0, dz)
						a_face[0] =  0;
						a_face[1] = -dy;
						a_face[2] =  dz;

						b_face[0] =  dx;
						b_face[1] = -dy;
						b_face[2] =  0;
					    }
					}
				    }

				    //vector area of parallelogram
				    vector_cross_product_in_place(&face_vector_A[0],a_face,b_face,3);

				    //vector area of triangle needs to be scaled by 1/2
				    //for(int l=0;l<3;l++)
				//	face_vector_A[l] *= 0.5;
				    //however, we often miss about the same area in the corners
				}   
				
				A_face = vector_dot_product(face_vector_A,dir_norm,3);
				if(A_face>0)
				    dcase[3] += A_face;
				break;


			//if 4 faces contribute, only add their hypotenusal area
			case 4: fcase[4]++;


				//tunnels don't contribute
				if(face[0]&&face[1]&&face[2]&&face[3])
				    break;
				if(face[2]&&face[3]&&face[4]&&face[5])
				    break;
				if(face[4]&&face[5]&&face[0]&&face[1])
				    break;

				//add open face

				if( face[0]&&face[1] )
				{

				    //surface runs diagonal across the x face
				    if(face[2]&&face[4])
				    {
					//+y and +z
					a_face[0] =  dx;
					a_face[1] =  0;
					a_face[2] =  0;

					b_face[0] =  0;
					b_face[1] = -dy;
					b_face[2] =  dz;

				    }else if(face[2]&&face[5]){

					//+y and -z
					a_face[0] =  dx;
					a_face[1] =  0;
					a_face[2] =  0;

					b_face[0] =  0;
					b_face[1] =  dy;
					b_face[2] =  dz;

				    }else if(face[3]&&face[4]){

					//-y and +z
					a_face[0] = -dx;
					a_face[1] =  0;
					a_face[2] =  0;

					b_face[0] =  0;
					b_face[1] = -dy;
					b_face[2] = -dz;

				    }else{

					//-y and -z
					a_face[0] = -dx;
					a_face[1] =  0;
					a_face[2] =  0;

					b_face[0] =  0;
					b_face[1] = -dy;
					b_face[2] =  dz;
				    }

				}else{

				    if( face[2]&&face[3] )
				    {
					//surface runs diagonal across the y face


					if(face[0]&&face[4])
					{
					   //+x and +z
					    a_face[0] =  0;
					    a_face[1] =  dy;
					    a_face[2] =  0;

					    b_face[0] =  dx;
					    b_face[1] =  0;
					    b_face[2] = -dz;

					}else if(face[0]&&face[5]){

					   //+x and -z
					    a_face[0] =  0;
					    a_face[1] =  dy;
					    a_face[2] =  0;

					    b_face[0] = -dx; 
					    b_face[1] =  0;
					    b_face[2] = -dz;

					}else if(face[1]&&face[4]){

					   //-x and +z
					    a_face[0] =  0;
					    a_face[1] = -dy;
					    a_face[2] =  0;

					    b_face[0] = -dx;
					    b_face[1] =  0;
					    b_face[2] = -dz;

					}else{

					    //-x and -z
					    a_face[0] =  0;
					    a_face[1] = -dy;
					    a_face[2] =  0;

					    b_face[0] =  dx;
					    b_face[1] =  0;
					    b_face[2] = -dz;
					}

				    }else{
					//surface runs diagonal across the z face

					if(face[0]&&face[2])
					{
					   //+x and +y
					    a_face[0] =  0;
					    a_face[1] =  0;
					    a_face[2] =  dz;

					    b_face[0] = -dx;
					    b_face[1] =  dy;
					    b_face[2] =  0;

					}else if(face[0]&&face[3]){

					   //+x and -y
					    a_face[0] =  0;
					    a_face[1] =  0;
					    a_face[2] =  dz;

					    b_face[0] =  dx;
					    b_face[1] =  dy;
					    b_face[2] =  0;

					}else if(face[1]&&face[2]){

					   //-x and +y
					    a_face[0] =  0;
					    a_face[1] =  0;
					    a_face[2] = -dz;

					    b_face[0] =  dx;
					    b_face[1] =  dy;
					    b_face[2] =  0;

					}else{

					    //-x and -y
					    a_face[0] =  0;
					    a_face[1] =  0;
					    a_face[2] = -dz;

					    b_face[0] = -dx;
					    b_face[1] =  dy;
					    b_face[2] =  0;
					}

				    }
				}

				//find vector area of parallelogram
				vector_cross_product_in_place(&face_vector_A[0],a_face,b_face,3);

				A_face = vector_dot_product(face_vector_A,dir_norm,3);
				if(A_face>0)
				    dcase[4] += A_face;
				break;


			//if 5 contribute, only the open face is added
			case 5: fcase[5]++;

				if(!face[0])
				    face_vector_A[0] =  dy*dz;
				if(!face[1])
				    face_vector_A[0] = -dy*dz;

				if(!face[2])
				    face_vector_A[1] =  dx*dz;
				if(!face[3])
				    face_vector_A[1] = -dx*dz;

				if(!face[4])
				    face_vector_A[2] =  dx*dy;
				if(!face[5])
				    face_vector_A[2] = -dx*dy;

				A_face = vector_dot_product(face_vector_A,dir_norm,3);
				if(A_face>0)
				    dcase[5] += A_face;
				break;

			//if 0 or 6 faces, no area
			default:    break;

		    }

		    //Add the surface area
		    A_face = vector_dot_product(face_vector_A,dir_norm,3);
		    if(A_face>0)
			A_local += A_face;
		}
	}

/*
    for(int i=0;i<numprocs;i++)
    {
	if(myid==i)
	{
	    printf("Before upper myid %d A_local %e n_tested %d nx_local %d nx_local_start %d nx_local_start + nx_local %d fcase[1] %d dcase[1] %e\n",myid,A_local,n_tested,nx_local,nx_local_start,nx_local_start+nx_local,fcase[1],dcase[1]);
	    fflush(stdout);
	}
	MPI_Barrier(world);
    }
*/
    //do upper boundary second
    if(nx_local)
    for(int j=0;j<ny;j++)
	for(int k=0;k<nz;k++)
	{
		//get cell index
		ijk = grid_ijk(nx_local-1,j,k,grid_info);

		//consider only zero cells near the interface
		n_faces = 0;
		for(int l=0;l<6;l++)
		    face[l] = 0;

		if(mask[ijk]==0)
		{

		    ijkt = nz*j + k;
		    if(ghost_cell_upper[ijkt]!=0)
		    {
			n_faces++;
			face[0] = 1;
		    }

		    //causes double counting for some splits

		    ijkt = grid_ijk(nx_local-2,j,k,grid_info);
		    if(mask[ijkt]!=0)
		    {
			n_faces++;
			face[1] = 1;
		    }

		    //ijkt = grid_ijk(nx_local-1,j-1,k,grid_info);
		    ijkt = grid_ijk(nx_local-1,j+1,k,grid_info);
		    if(mask[ijkt]!=0)
		    {
			n_faces++;
			face[2] = 1;
		    }

		    //ijkt = grid_ijk(nx_local-1,j+1,k,grid_info);
		    ijkt = grid_ijk(nx_local-1,j-1,k,grid_info);
		    if(mask[ijkt]!=0)
		    {
			n_faces++;
			face[3] = 1;
		    }

		    //ijkt = grid_ijk(nx_local-1,j,k-1,grid_info);
		    ijkt = grid_ijk(nx_local-1,j,k+1,grid_info);
		    if(mask[ijkt]!=0)
		    {
			n_faces++;
			face[4] = 1;
		    }

		    //ijkt = grid_ijk(nx_local-1,j,k+1,grid_info);
		    ijkt = grid_ijk(nx_local-1,j,k-1,grid_info);
		    if(mask[ijkt]!=0)
		    {
			n_faces++;
			face[5] = 1;
		    }

		    //surface area contribution of the face
		    //in question
		    A_face = 0;
		    for(int l=0;l<3;l++)
		    {
			face_vector_A[l] = 0;
			a_face[l] = 0;
			b_face[l] = 0;
		    }

		    //add area from this cell
		    switch(n_faces)
		    {


			//if no faces contribute, just break
			case 0:	fcase[0]++;
				break;

			//if 1 face contributes, add its area
			case 1:	fcase[1]++;

				//n_tested++;
				if(face[0])
				    face_vector_A[0] = -dy * dz;
				
				if(face[1])
				    face_vector_A[0] =  dy * dz;
				
				if(face[2])
				    face_vector_A[1] = -dz * dx;

				if(face[3])
				    face_vector_A[1] =  dz * dx;

				if(face[4])
				    face_vector_A[2] = -dx * dy;
				
				if(face[5])
				    face_vector_A[2] =  dx * dy;
				
				A_face = vector_dot_product(face_vector_A,dir_norm,3);
				if(A_face>0)
				    dcase[1] += A_face;
				break;


			//if 2 faces contribute and are adjacent, add their hypotenusal area
			case 2:	fcase[2]++;

				//opposing faces don't contribute
				if( (face[0]&&face[1])||(face[2]&&face[3])||(face[4]&&face[5]) )
				    break;

				//add the hypotenuse only if the dot product is positive
				if(face[0]) //+x face
				{
					if(face[2])
					{
						//+x/+y, checked 05/22
						a_face[2] =  dz;

						b_face[0] = -dx;
						b_face[1] =  dy;

					}else if(face[3]){

						//+x/-y, checked 05/22
						a_face[2] =  dz;

						b_face[0] =  dx;
						b_face[1] =  dy;
					}

					if(face[4])
					{
						//+x/+z,checked 05/22
						a_face[1] =  dy;

						b_face[0] =  dx;
						b_face[2] = -dz;

					}else if(face[5]){

						//+x/-z, checked 05/22
						a_face[1] =  dy;

						b_face[0] =  -dx;
						b_face[2] =  -dz;
					}

				}else if(face[1]){ //-x face

					if(face[2])
					{
						//-x/+y, checked 05/22
						a_face[2] =  dz;

						b_face[0] = -dx;
						b_face[1] = -dy;

					}else if(face[3]){

						//-x/-y, checked 05/22
						a_face[2] =  dz;

						b_face[0] =  dx;
						b_face[1] = -dy;

					}

					if(face[4])
					{
						//-x/+z, checked 05/22
						a_face[1] =  dy;

						b_face[0] =  dx;
						b_face[2] =  dz;

					}else if(face[5]){

						//-x/-z, checked 05/22
						a_face[1] =  dy;

						b_face[0] = -dx;
						b_face[2] =  dz;
					}

				}else if(face[2]){  //+y face, no x-face

					//+y face 
					if(face[4])
					{
						//+y/+z, checked 05/22
						a_face[0] =  dx;

						b_face[1] = -dy;
						b_face[2] =  dz;

					}else if(face[5]){

						//+y/-z, checked 05/22
						a_face[0] =  dx;

						b_face[1] =  dy;
						b_face[2] =  dz;
					}

				}else if(face[3]){ //-y face, no x-face

					//-y face
					if(face[4])
					{
						//-y/+z, checked 05/22
						a_face[0] =  dx;

						b_face[1] = -dy;
						b_face[2] = -dz;

					}else if(face[5]){

						//-y/-z, checked 05/22
						a_face[0] =  dx;

						b_face[1] =  dy;
						b_face[2] = -dz;
					}

				}
				vector_cross_product_in_place(&face_vector_A[0],a_face,b_face,3);
		
				A_face = vector_dot_product(face_vector_A,dir_norm,3);
				if(A_face>0)
				    dcase[2] += A_face;
				break;

			//if 3 faces contribute, then treat adjacent and opposing faces separately
			case 3: fcase[3]++;

				//opposing faces contribute only the non-opposing face
				if( (face[0]&&face[1])||(face[2]&&face[3])||(face[4]&&face[5]) )
				{

				    if( face[0]&&face[1] )
				    {
					//y faces
					if(face[2])
					    face_vector_A[1] = -dz*dx;
					if(face[3])
					    face_vector_A[1] =  dz*dx;
	    
					//z faces
					if(face[4])
					    face_vector_A[2] = -dx*dy;
					if(face[5])
					    face_vector_A[2] =  dx*dy;
				    }

				    if( face[2]&&face[3] )
				    {
					//x faces
					if(face[0])
					    face_vector_A[0] = -dy*dz;
					if(face[1])
					    face_vector_A[0] =  dy*dz;

					//z faces
					if(face[4])
					    face_vector_A[2] = -dx*dy;
					if(face[5])
					    face_vector_A[2] =  dx*dy;
				    }

				    if( face[4]&&face[5] )
				    {
					//x faces
					if(face[0])
					    face_vector_A[0] = -dy*dz;
					if(face[1])
					    face_vector_A[0] =  dy*dz;

					//y faces
					if(face[2])
					    face_vector_A[1] = -dz*dx;
					if(face[3])
					    face_vector_A[1] =  dz*dx;
				    }

				}else{


				    //adjacent faces supply chamfer of triangle of hypotenuses
				    //a_face will lie in the inner most conditioned face

				    //there must be an x face
				    if(face[0])	//+x
				    {
					if(face[2]) //+y
					{
					    if(face[4])	//+z
					    {
						//+x +y +z; triangle points down to (dx, dy, 0)
						a_face[0] =  0;
						a_face[1] = -dy;
						a_face[2] =  dz;

						b_face[0] = -dx;
						b_face[1] =  0;
						b_face[2] =  dz;

					    }else{	//-z

						//+x +y -z;  triangle points up to (dx, dy, dz)
						a_face[0] =  dx;
						a_face[1] = -dy;
						a_face[2] =  0;

						b_face[0] =  dx;
						b_face[1] =  0;
						b_face[2] =  dz;
					    }
					}else{	//-y
					    if(face[4])	//+z
					    {
						//+x -y +z; triangle points down to (dx, 0, 0)
						a_face[0] =  dx;
						a_face[1] =  dy;
						a_face[2] =  0;

						b_face[0] =  dx;
						b_face[1] =  0;
						b_face[2] = -dz;

					    }else{	//-z ?

						//+x -y -z;  triangle points up to (dx, 0, dz)
						a_face[0] =  0;
						a_face[1] =  dy;
						a_face[2] = -dz;

						b_face[0] = -dx;
						b_face[1] =  0;
						b_face[2] = -dz;
					    }
					}
				    }else{	//-x
					if(face[2]) //+y
					{
					    if(face[4])	//+z
					    {
						//-x +y +z; triangle points down to (0, dy, 0)
						a_face[0] =  dx;
						a_face[1] =  0;
						a_face[2] =  dz;

						b_face[0] =  0;
						b_face[1] = -dy;
						b_face[2] =  dz;

					    }else{	//-z

						//-x +y -z;  triangle points up to (0, dy, dz)
						a_face[0] =  dx;
						a_face[1] =  dy;
						a_face[2] =  0;

						b_face[0] =  0;
						b_face[1] =  dy;
						b_face[2] =  dz;
					    }
					}else{	//-y
					    if(face[4])	//+z
					    {
						//-x -y +z; triangle points down to (0, 0, 0)
						a_face[0] =  0;
						a_face[1] =  dy;
						a_face[2] =  dz;

						b_face[0] =  dx;
						b_face[1] =  0;
						b_face[2] =  dz;

					    }else{	//-z

						//-x -y -z;  triangle points up to (0, 0, dz)
						a_face[0] =  0;
						a_face[1] = -dy;
						a_face[2] =  dz;

						b_face[0] =  dx;
						b_face[1] = -dy;
						b_face[2] =  0;
					    }
					}
				    }

				    //vector area of parallelogram
				    vector_cross_product_in_place(&face_vector_A[0],a_face,b_face,3);

				    //vector area of triangle needs to be scaled by 1/2
				    //for(int l=0;l<3;l++)
				//	face_vector_A[l] *= 0.5;
				    //however, we often miss about the same area in the corners
				}   
				
				A_face = vector_dot_product(face_vector_A,dir_norm,3);
				if(A_face>0)
				    dcase[3] += A_face;
				break;


			//if 4 faces contribute, only add their hypotenusal area
			case 4: fcase[4]++;


				//tunnels don't contribute
				if(face[0]&&face[1]&&face[2]&&face[3])
				    break;
				if(face[2]&&face[3]&&face[4]&&face[5])
				    break;
				if(face[4]&&face[5]&&face[0]&&face[1])
				    break;

				//add open face

				if( face[0]&&face[1] )
				{

				    //surface runs diagonal across the x face
				    if(face[2]&&face[4])
				    {
					//+y and +z
					a_face[0] =  dx;
					a_face[1] =  0;
					a_face[2] =  0;

					b_face[0] =  0;
					b_face[1] = -dy;
					b_face[2] =  dz;

				    }else if(face[2]&&face[5]){

					//+y and -z
					a_face[0] =  dx;
					a_face[1] =  0;
					a_face[2] =  0;

					b_face[0] =  0;
					b_face[1] =  dy;
					b_face[2] =  dz;

				    }else if(face[3]&&face[4]){

					//-y and +z
					a_face[0] = -dx;
					a_face[1] =  0;
					a_face[2] =  0;

					b_face[0] =  0;
					b_face[1] = -dy;
					b_face[2] = -dz;

				    }else{

					//-y and -z
					a_face[0] = -dx;
					a_face[1] =  0;
					a_face[2] =  0;

					b_face[0] =  0;
					b_face[1] = -dy;
					b_face[2] =  dz;
				    }

				}else{

				    if( face[2]&&face[3] )
				    {
					//surface runs diagonal across the y face


					if(face[0]&&face[4])
					{
					   //+x and +z
					    a_face[0] =  0;
					    a_face[1] =  dy;
					    a_face[2] =  0;

					    b_face[0] =  dx;
					    b_face[1] =  0;
					    b_face[2] = -dz;

					}else if(face[0]&&face[5]){

					   //+x and -z
					    a_face[0] =  0;
					    a_face[1] =  dy;
					    a_face[2] =  0;

					    b_face[0] = -dx; 
					    b_face[1] =  0;
					    b_face[2] = -dz;

					}else if(face[1]&&face[4]){

					   //-x and +z
					    a_face[0] =  0;
					    a_face[1] = -dy;
					    a_face[2] =  0;

					    b_face[0] = -dx;
					    b_face[1] =  0;
					    b_face[2] = -dz;

					}else{

					    //-x and -z
					    a_face[0] =  0;
					    a_face[1] = -dy;
					    a_face[2] =  0;

					    b_face[0] =  dx;
					    b_face[1] =  0;
					    b_face[2] = -dz;
					}

				    }else{
					//surface runs diagonal across the z face

					if(face[0]&&face[2])
					{
					   //+x and +y
					    a_face[0] =  0;
					    a_face[1] =  0;
					    a_face[2] =  dz;

					    b_face[0] = -dx;
					    b_face[1] =  dy;
					    b_face[2] =  0;

					}else if(face[0]&&face[3]){

					   //+x and -y
					    a_face[0] =  0;
					    a_face[1] =  0;
					    a_face[2] =  dz;

					    b_face[0] =  dx;
					    b_face[1] =  dy;
					    b_face[2] =  0;

					}else if(face[1]&&face[2]){

					   //-x and +y
					    a_face[0] =  0;
					    a_face[1] =  0;
					    a_face[2] = -dz;

					    b_face[0] =  dx;
					    b_face[1] =  dy;
					    b_face[2] =  0;

					}else{

					    //-x and -y
					    a_face[0] =  0;
					    a_face[1] =  0;
					    a_face[2] = -dz;

					    b_face[0] = -dx;
					    b_face[1] =  dy;
					    b_face[2] =  0;
					}

				    }
				}

				//find vector area of parallelogram
				vector_cross_product_in_place(&face_vector_A[0],a_face,b_face,3);

				A_face = vector_dot_product(face_vector_A,dir_norm,3);
				if(A_face>0)
				    dcase[4] += A_face;
				break;


			//if 5 contribute, only the open face is added
			case 5: fcase[5]++;

				if(!face[0])
				    face_vector_A[0] =  dy*dz;
				if(!face[1])
				    face_vector_A[0] = -dy*dz;

				if(!face[2])
				    face_vector_A[1] =  dx*dz;
				if(!face[3])
				    face_vector_A[1] = -dx*dz;

				if(!face[4])
				    face_vector_A[2] =  dx*dy;
				if(!face[5])
				    face_vector_A[2] = -dx*dy;

				A_face = vector_dot_product(face_vector_A,dir_norm,3);
				if(A_face>0)
				    dcase[5] += A_face;
				break;

			//if 0 or 6 faces, no area
			default:    break;
		    }

		    //Add the surface area
		    A_face = vector_dot_product(face_vector_A,dir_norm,3);
		    if(A_face>0)
			A_local += A_face;

		}
	}

    //free ghost cells
    free(ghost_cell_upper);
    free(ghost_cell_lower);

/*
    for(int i=0;i<numprocs;i++)
    {
	if(myid==i)
	{
	    printf("After upper myid %d A_local %e n_tested %d nx_local %d nx_local_start %d nx_local_start + nx_local %d fcase[1] %d dcase[1] %e\n",myid,A_local,n_tested,nx_local,nx_local_start,nx_local_start+nx_local,fcase[1],dcase[1]);
	    fflush(stdout);
	}
	MPI_Barrier(world);
    }
*/

    //sum area across processors
    MPI_Allreduce(&A_local,&A_total,1,MPI_DOUBLE,MPI_SUM,world);


    //find the total surface area
    return A_total;

}

/*! \fn double grid_surface_flux(double *mask, double **vfield, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world)
 *  \brief Find the approximate surface flux of a vector field through the surface at the mask boundary */
double grid_surface_flux(double *mask, double **vfield, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world)
{
    int ijk;	//cell
    int ijkt;	//adjacent cells
    int nx_local = grid_info.nx_local;
    int nx_local_start = grid_info.nx_local_start;
    int ny       = grid_info.ny;
    int nz       = grid_info.nz;

    double dx	= grid_info.dx;
    double dy	= grid_info.dy;
    double dz	= grid_info.dz;

    int ndim = grid_info.ndim;

    int dest;	//destination processor for communication
    int source;	//source processor for communication

    MPI_Status status;	//MPI status for communication


    double *ghost_cell_upper;	//ghost cells at i = nx_local
    double *ghost_cell_lower;	//ghost cells at i = -1

    //local surface area
    double A_local = 0;
    
    //total surface area
    double A_total = 0;

    //number of faces contributing to surface
    int n_faces;
    int face[6];

    //perimeter
    double s;

    //triangle sides
    double a, b, c;

    int n_tested = 0;

    double face_vector_A[3];  //vector area of the face
    double a_face[3];	      //first vector of the face
    double b_face[3];	      //second vector of the face

    double dir_norm[3];	      //value of the vector field at this cell

    int fcase[6] = {0,0,0,0,0,0};
    double dcase[6] = {0,0,0,0,0,0};

    double A_face;


    //we have to check boundary cells separately
    //so start with inner cells
    for(int i=1;i<nx_local-1;i++)
	for(int j=0;j<ny;j++)
	    for(int k=0;k<nz;k++)
	    {

		//get cell index
		ijk = grid_ijk(i,j,k,grid_info);

		//consider only zero cells near the interface
		n_faces = 0;
		for(int l=0;l<6;l++)
		    face[l] = 0;

		//get the value of the vector field at this cell
		for(int l=0;l<ndim;l++)
		    dir_norm[l] = vfield[l][ijk];

		if(mask[ijk]==0)
		{

		    ijkt = grid_ijk(i+1,j,k,grid_info);
		    if(mask[ijkt]!=0)
		    {
			n_faces++;
			face[0] = 1;
		    }

		    ijkt = grid_ijk(i-1,j,k,grid_info);
		    if(mask[ijkt]!=0)
		    {
			n_faces++;
			face[1] = 1;
		    }


		    ijkt = grid_ijk(i,j+1,k,grid_info);
		    if(mask[ijkt]!=0)
		    {
			n_faces++;
			face[2] = 1;
		    }

		    ijkt = grid_ijk(i,j-1,k,grid_info);
		    if(mask[ijkt]!=0)
		    {
			n_faces++;
			face[3] = 1;
		    }

		    ijkt = grid_ijk(i,j,k+1,grid_info);
		    if(mask[ijkt]!=0)
		    {
			n_faces++;
			face[4] = 1;
		    }

		    ijkt = grid_ijk(i,j,k-1,grid_info);
		    if(mask[ijkt]!=0)
		    {
			n_faces++;
			face[5] = 1;
		    }


		    //surface area contribution of the face
		    //in question
		    A_face = 0;
		    for(int l=0;l<3;l++)
		    {
			face_vector_A[l] = 0;
			a_face[l] = 0;
			b_face[l] = 0;
		    }

		    //add area from this cell
		    switch(n_faces)
		    {

			//if no faces contribute, just break
			case 0:	fcase[0]++;
				break;

			//if 1 face contributes, add its area
			case 1:	

				//n_tested++;
				if(face[0])
				    face_vector_A[0] = -dy * dz;
				
				if(face[1])
				    face_vector_A[0] =  dy * dz;
				
				if(face[2])
				    face_vector_A[1] = -dz * dx;

				if(face[3])
				    face_vector_A[1] =  dz * dx;

				if(face[4])
				    face_vector_A[2] = -dx * dy;
				
				if(face[5])
				    face_vector_A[2] =  dx * dy;
				
				A_face = vector_dot_product(face_vector_A,dir_norm,3);
				if(A_face>0)
				{
				    fcase[1]++;
				    dcase[1] += A_face;
				}
				break;


			//if 2 faces contribute and are adjacent, add their hypotenusal area
			case 2:	fcase[2]++;

				//opposing faces don't contribute
				if( (face[0]&&face[1])||(face[2]&&face[3])||(face[4]&&face[5]) )
				    break;

				//add the hypotenuse only if the dot product is positive
				if(face[0]) //+x face
				{
					if(face[2])
					{
						//+x/+y, checked 05/22
						a_face[2] =  dz;

						b_face[0] = -dx;
						b_face[1] =  dy;

					}else if(face[3]){

						//+x/-y, checked 05/22
						a_face[2] =  dz;

						b_face[0] =  dx;
						b_face[1] =  dy;
					}

					if(face[4])
					{
						//+x/+z,checked 05/22
						a_face[1] =  dy;

						b_face[0] =  dx;
						b_face[2] = -dz;

					}else if(face[5]){

						//+x/-z, checked 05/22
						a_face[1] =  dy;

						b_face[0] =  -dx;
						b_face[2] =  -dz;
					}

				}else if(face[1]){ //-x face

					if(face[2])
					{
						//-x/+y, checked 05/22
						a_face[2] =  dz;

						b_face[0] = -dx;
						b_face[1] = -dy;

					}else if(face[3]){

						//-x/-y, checked 05/22
						a_face[2] =  dz;

						b_face[0] =  dx;
						b_face[1] = -dy;

					}

					if(face[4])
					{
						//-x/+z, checked 05/22
						a_face[1] =  dy;

						b_face[0] =  dx;
						b_face[2] =  dz;

					}else if(face[5]){

						//-x/-z, checked 05/22
						a_face[1] =  dy;

						b_face[0] = -dx;
						b_face[2] =  dz;
					}

				}else if(face[2]){  //+y face, no x-face

					//+y face 
					if(face[4])
					{
						//+y/+z, checked 05/22
						a_face[0] =  dx;

						b_face[1] = -dy;
						b_face[2] =  dz;

					}else if(face[5]){

						//+y/-z, checked 05/22
						a_face[0] =  dx;

						b_face[1] =  dy;
						b_face[2] =  dz;
					}

				}else if(face[3]){ //-y face, no x-face

					//-y face
					if(face[4])
					{
						//-y/+z, checked 05/22
						a_face[0] =  dx;

						b_face[1] = -dy;
						b_face[2] = -dz;

					}else if(face[5]){

						//-y/-z, checked 05/22
						a_face[0] =  dx;

						b_face[1] =  dy;
						b_face[2] = -dz;
					}

				}
				vector_cross_product_in_place(&face_vector_A[0],a_face,b_face,3);
		
				A_face = vector_dot_product(face_vector_A,dir_norm,3);
				if(A_face>0)
				    dcase[2] += A_face;
				break;

			//if 3 faces contribute, then treat adjacent and opposing faces separately
			case 3: fcase[3]++;

				//opposing faces contribute only the non-opposing face
				if( (face[0]&&face[1])||(face[2]&&face[3])||(face[4]&&face[5]) )
				{

				    if( face[0]&&face[1] )
				    {
					//y faces
					if(face[2])
					    face_vector_A[1] = -dz*dx;
					if(face[3])
					    face_vector_A[1] =  dz*dx;
	    
					//z faces
					if(face[4])
					    face_vector_A[2] = -dx*dy;
					if(face[5])
					    face_vector_A[2] =  dx*dy;
				    }

				    if( face[2]&&face[3] )
				    {
					//x faces
					if(face[0])
					    face_vector_A[0] = -dy*dz;
					if(face[1])
					    face_vector_A[0] =  dy*dz;

					//z faces
					if(face[4])
					    face_vector_A[2] = -dx*dy;
					if(face[5])
					    face_vector_A[2] =  dx*dy;
				    }

				    if( face[4]&&face[5] )
				    {
					//x faces
					if(face[0])
					    face_vector_A[0] = -dy*dz;
					if(face[1])
					    face_vector_A[0] =  dy*dz;

					//y faces
					if(face[2])
					    face_vector_A[1] = -dz*dx;
					if(face[3])
					    face_vector_A[1] =  dz*dx;
				    }

				}else{


				    //adjacent faces supply chamfer of triangle of hypotenuses
				    //a_face will lie in the inner most conditioned face

				    //there must be an x face
				    if(face[0])	//+x
				    {
					if(face[2]) //+y
					{
					    if(face[4])	//+z
					    {
						//+x +y +z; triangle points down to (dx, dy, 0)
						a_face[0] =  0;
						a_face[1] = -dy;
						a_face[2] =  dz;

						b_face[0] = -dx;
						b_face[1] =  0;
						b_face[2] =  dz;

					    }else{	//-z

						//+x +y -z;  triangle points up to (dx, dy, dz)
						a_face[0] =  dx;
						a_face[1] = -dy;
						a_face[2] =  0;

						b_face[0] =  dx;
						b_face[1] =  0;
						b_face[2] =  dz;
					    }
					}else{	//-y
					    if(face[4])	//+z
					    {
						//+x -y +z; triangle points down to (dx, 0, 0)
						a_face[0] =  dx;
						a_face[1] =  dy;
						a_face[2] =  0;

						b_face[0] =  dx;
						b_face[1] =  0;
						b_face[2] = -dz;

					    }else{	//-z ?

						//+x -y -z;  triangle points up to (dx, 0, dz)
						a_face[0] =  0;
						a_face[1] =  dy;
						a_face[2] = -dz;

						b_face[0] = -dx;
						b_face[1] =  0;
						b_face[2] = -dz;
					    }
					}
				    }else{	//-x
					if(face[2]) //+y
					{
					    if(face[4])	//+z
					    {
						//-x +y +z; triangle points down to (0, dy, 0)
						a_face[0] =  dx;
						a_face[1] =  0;
						a_face[2] =  dz;

						b_face[0] =  0;
						b_face[1] = -dy;
						b_face[2] =  dz;

					    }else{	//-z

						//-x +y -z;  triangle points up to (0, dy, dz)
						a_face[0] =  dx;
						a_face[1] =  dy;
						a_face[2] =  0;

						b_face[0] =  0;
						b_face[1] =  dy;
						b_face[2] =  dz;
					    }
					}else{	//-y
					    if(face[4])	//+z
					    {
						//-x -y +z; triangle points down to (0, 0, 0)
						a_face[0] =  0;
						a_face[1] =  dy;
						a_face[2] =  dz;

						b_face[0] =  dx;
						b_face[1] =  0;
						b_face[2] =  dz;

					    }else{	//-z

						//-x -y -z;  triangle points up to (0, 0, dz)
						a_face[0] =  0;
						a_face[1] = -dy;
						a_face[2] =  dz;

						b_face[0] =  dx;
						b_face[1] = -dy;
						b_face[2] =  0;
					    }
					}
				    }

				    //vector area of parallelogram
				    vector_cross_product_in_place(&face_vector_A[0],a_face,b_face,3);

				    //vector area of triangle needs to be scaled by 1/2
				    //for(int l=0;l<3;l++)
				//	face_vector_A[l] *= 0.5;
				    //however, we often miss about the same area in the corners
				}   
				
				A_face = vector_dot_product(face_vector_A,dir_norm,3);
				if(A_face>0)
				    dcase[3] += A_face;
				break;


			//if 4 faces contribute, only add their hypotenusal area
			case 4: fcase[4]++;


				//tunnels don't contribute
				if(face[0]&&face[1]&&face[2]&&face[3])
				    break;
				if(face[2]&&face[3]&&face[4]&&face[5])
				    break;
				if(face[4]&&face[5]&&face[0]&&face[1])
				    break;

				//add open face

				if( face[0]&&face[1] )
				{

				    //surface runs diagonal across the x face
				    if(face[2]&&face[4])
				    {
					//+y and +z
					a_face[0] =  dx;
					a_face[1] =  0;
					a_face[2] =  0;

					b_face[0] =  0;
					b_face[1] = -dy;
					b_face[2] =  dz;

				    }else if(face[2]&&face[5]){

					//+y and -z
					a_face[0] =  dx;
					a_face[1] =  0;
					a_face[2] =  0;

					b_face[0] =  0;
					b_face[1] =  dy;
					b_face[2] =  dz;

				    }else if(face[3]&&face[4]){

					//-y and +z
					a_face[0] = -dx;
					a_face[1] =  0;
					a_face[2] =  0;

					b_face[0] =  0;
					b_face[1] = -dy;
					b_face[2] = -dz;

				    }else{

					//-y and -z
					a_face[0] = -dx;
					a_face[1] =  0;
					a_face[2] =  0;

					b_face[0] =  0;
					b_face[1] = -dy;
					b_face[2] =  dz;
				    }

				}else{

				    if( face[2]&&face[3] )
				    {
					//surface runs diagonal across the y face


					if(face[0]&&face[4])
					{
					   //+x and +z
					    a_face[0] =  0;
					    a_face[1] =  dy;
					    a_face[2] =  0;

					    b_face[0] =  dx;
					    b_face[1] =  0;
					    b_face[2] = -dz;

					}else if(face[0]&&face[5]){

					   //+x and -z
					    a_face[0] =  0;
					    a_face[1] =  dy;
					    a_face[2] =  0;

					    b_face[0] = -dx; 
					    b_face[1] =  0;
					    b_face[2] = -dz;

					}else if(face[1]&&face[4]){

					   //-x and +z
					    a_face[0] =  0;
					    a_face[1] = -dy;
					    a_face[2] =  0;

					    b_face[0] = -dx;
					    b_face[1] =  0;
					    b_face[2] = -dz;

					}else{

					    //-x and -z
					    a_face[0] =  0;
					    a_face[1] = -dy;
					    a_face[2] =  0;

					    b_face[0] =  dx;
					    b_face[1] =  0;
					    b_face[2] = -dz;
					}

				    }else{
					//surface runs diagonal across the z face

					if(face[0]&&face[2])
					{
					   //+x and +y
					    a_face[0] =  0;
					    a_face[1] =  0;
					    a_face[2] =  dz;

					    b_face[0] = -dx;
					    b_face[1] =  dy;
					    b_face[2] =  0;

					}else if(face[0]&&face[3]){

					   //+x and -y
					    a_face[0] =  0;
					    a_face[1] =  0;
					    a_face[2] =  dz;

					    b_face[0] =  dx;
					    b_face[1] =  dy;
					    b_face[2] =  0;

					}else if(face[1]&&face[2]){

					   //-x and +y
					    a_face[0] =  0;
					    a_face[1] =  0;
					    a_face[2] = -dz;

					    b_face[0] =  dx;
					    b_face[1] =  dy;
					    b_face[2] =  0;

					}else{

					    //-x and -y
					    a_face[0] =  0;
					    a_face[1] =  0;
					    a_face[2] = -dz;

					    b_face[0] = -dx;
					    b_face[1] =  dy;
					    b_face[2] =  0;
					}

				    }
				}

				//find vector area of parallelogram
				vector_cross_product_in_place(&face_vector_A[0],a_face,b_face,3);

				A_face = vector_dot_product(face_vector_A,dir_norm,3);
				if(A_face>0)
				    dcase[4] += A_face;
				break;


			//if 5 contribute, only the open face is added
			case 5: fcase[5]++;

				if(!face[0])
				    face_vector_A[0] =  dy*dz;
				if(!face[1])
				    face_vector_A[0] = -dy*dz;

				if(!face[2])
				    face_vector_A[1] =  dx*dz;
				if(!face[3])
				    face_vector_A[1] = -dx*dz;

				if(!face[4])
				    face_vector_A[2] =  dx*dy;
				if(!face[5])
				    face_vector_A[2] = -dx*dy;

				A_face = vector_dot_product(face_vector_A,dir_norm,3);
				if(A_face>0)
				    dcase[5] += A_face;
				break;

			//if 0 or 6 faces, no area
			default:    break;
		    }

		    //Add the surface area
		    A_face = vector_dot_product(face_vector_A,dir_norm,3);
		    if(A_face>0)
			A_local += A_face;
		}
	    }


    //now we have to check boundary cells
    //to do this, we use a sendreceive to get adjacent
    //cells across the boundary, then repeat the checks

    //allocate ghost cells
    ghost_cell_upper = calloc_double_array(ny*nz);
    ghost_cell_lower = calloc_double_array(ny*nz); 

    //populate ghost cells
    for(int j=0;j<ny;j++)
	for(int k=0;k<nz;k++)
	{


	    //note nx_local can be zero!
	    if(nx_local)
	    {
		//upper cells to pass from this processor's lower boundary
		ijk = grid_ijk(0,j,k,grid_info);
		ghost_cell_upper[nz*j+k] = mask[ijk];

		//lower cells to pass from this processor's upper boundary
		ijk = grid_ijk(nx_local-1,j,k,grid_info);
		ghost_cell_lower[nz*j+k] = mask[ijk];


		//if(nx_local_start+nx_local-1==47)
		//	printf("j %d k %d ghost %e\n",j,k,ghost_cell_lower[nz*j+k]);
	    }else{
		//dummy cells
		ghost_cell_upper[nz*j+k] = 0;
		ghost_cell_lower[nz*j+k] = 0;
	    }
	}

    //sendrecv upper ghost cells
    source = myid+1;
    if(source>numprocs-1)
	source-=numprocs;
    dest = myid-1;
    if(dest<0)
	dest+=numprocs;

    MPI_Sendrecv_replace(ghost_cell_upper, ny*nz, MPI_DOUBLE, dest, myid, source, source, world, &status);
	

    //sendrecv lower ghost cells
    source = myid-1;
    if(source<0)
	source+=numprocs;
    dest = myid+1;
    if(dest>numprocs-1)
	dest-=numprocs;

    MPI_Sendrecv_replace(ghost_cell_lower, ny*nz, MPI_DOUBLE, dest, myid, source, source, world, &status);


    //do lower boundary first
    if(nx_local)
    for(int j=0;j<ny;j++)
	for(int k=0;k<nz;k++)
	{
		//get cell index
		ijk = grid_ijk(0,j,k,grid_info);

		//consider only zero cells near the interface
		n_faces = 0;
		for(int l=0;l<6;l++)
		    face[l] = 0;

		//get the value of the vector field at this cell
		for(int l=0;l<ndim;l++)
		    dir_norm[l] = vfield[l][ijk];

		if(mask[ijk]==0)
		{
		    ijkt = grid_ijk(1,j,k,grid_info);
		    if(mask[ijkt]!=0)
		    {
			n_faces++;
			face[0] = 1;
		    }

		    ijkt = nz*j + k;
		    if(ghost_cell_lower[ijkt]!=0)
		    {
			n_faces++;
			face[1] = 1;
		    }


		    ijkt = grid_ijk(0,j+1,k,grid_info);
		    if(mask[ijkt]!=0)
		    {
			n_faces++;
			face[2] = 1;
		    }

		    ijkt = grid_ijk(0,j-1,k,grid_info);
		    if(mask[ijkt]!=0)
		    {
			n_faces++;
			face[3] = 1;
		    }

		    ijkt = grid_ijk(0,j,k+1,grid_info);
		    if(mask[ijkt]!=0)
		    {
			n_faces++;
			face[4] = 1;
		    }

		    ijkt = grid_ijk(0,j,k-1,grid_info);
		    if(mask[ijkt]!=0)
		    {
			n_faces++;
			face[5] = 1;
		    }

		    //surface area contribution of the face
		    //in question
		    A_face = 0;
		    for(int l=0;l<3;l++)
		    {
			face_vector_A[l] = 0;
			a_face[l] = 0;
			b_face[l] = 0;
		    }

		    //add area from this cell
		    switch(n_faces)
		    {
			//if no faces contribute, just break
			case 0:	fcase[0]++;
				break;

			//if 1 face contributes, add its area
			case 1:	fcase[1]++;

				//n_tested++;
				if(face[0])
				    face_vector_A[0] = -dy * dz;
				
				if(face[1])
				    face_vector_A[0] =  dy * dz;
				
				if(face[2])
				    face_vector_A[1] = -dz * dx;

				if(face[3])
				    face_vector_A[1] =  dz * dx;

				if(face[4])
				    face_vector_A[2] = -dx * dy;
				
				if(face[5])
				    face_vector_A[2] =  dx * dy;
				
				A_face = vector_dot_product(face_vector_A,dir_norm,3);
				if(A_face>0)
				    dcase[1] += A_face;
				break;


			//if 2 faces contribute and are adjacent, add their hypotenusal area
			case 2:	fcase[2]++;

				//opposing faces don't contribute
				if( (face[0]&&face[1])||(face[2]&&face[3])||(face[4]&&face[5]) )
				    break;

				//add the hypotenuse only if the dot product is positive
				if(face[0]) //+x face
				{
					if(face[2])
					{
						//+x/+y, checked 05/22
						a_face[2] =  dz;

						b_face[0] = -dx;
						b_face[1] =  dy;

					}else if(face[3]){

						//+x/-y, checked 05/22
						a_face[2] =  dz;

						b_face[0] =  dx;
						b_face[1] =  dy;
					}

					if(face[4])
					{
						//+x/+z,checked 05/22
						a_face[1] =  dy;

						b_face[0] =  dx;
						b_face[2] = -dz;

					}else if(face[5]){

						//+x/-z, checked 05/22
						a_face[1] =  dy;

						b_face[0] =  -dx;
						b_face[2] =  -dz;
					}

				}else if(face[1]){ //-x face

					if(face[2])
					{
						//-x/+y, checked 05/22
						a_face[2] =  dz;

						b_face[0] = -dx;
						b_face[1] = -dy;

					}else if(face[3]){

						//-x/-y, checked 05/22
						a_face[2] =  dz;

						b_face[0] =  dx;
						b_face[1] = -dy;

					}

					if(face[4])
					{
						//-x/+z, checked 05/22
						a_face[1] =  dy;

						b_face[0] =  dx;
						b_face[2] =  dz;

					}else if(face[5]){

						//-x/-z, checked 05/22
						a_face[1] =  dy;

						b_face[0] = -dx;
						b_face[2] =  dz;
					}

				}else if(face[2]){  //+y face, no x-face

					//+y face 
					if(face[4])
					{
						//+y/+z, checked 05/22
						a_face[0] =  dx;

						b_face[1] = -dy;
						b_face[2] =  dz;

					}else if(face[5]){

						//+y/-z, checked 05/22
						a_face[0] =  dx;

						b_face[1] =  dy;
						b_face[2] =  dz;
					}

				}else if(face[3]){ //-y face, no x-face

					//-y face
					if(face[4])
					{
						//-y/+z, checked 05/22
						a_face[0] =  dx;

						b_face[1] = -dy;
						b_face[2] = -dz;

					}else if(face[5]){

						//-y/-z, checked 05/22
						a_face[0] =  dx;

						b_face[1] =  dy;
						b_face[2] = -dz;
					}

				}
				vector_cross_product_in_place(&face_vector_A[0],a_face,b_face,3);
		
				A_face = vector_dot_product(face_vector_A,dir_norm,3);
				if(A_face>0)
				    dcase[2] += A_face;
				break;

			//if 3 faces contribute, then treat adjacent and opposing faces separately
			case 3: fcase[3]++;

				//opposing faces contribute only the non-opposing face
				if( (face[0]&&face[1])||(face[2]&&face[3])||(face[4]&&face[5]) )
				{

				    if( face[0]&&face[1] )
				    {
					//y faces
					if(face[2])
					    face_vector_A[1] = -dz*dx;
					if(face[3])
					    face_vector_A[1] =  dz*dx;
	    
					//z faces
					if(face[4])
					    face_vector_A[2] = -dx*dy;
					if(face[5])
					    face_vector_A[2] =  dx*dy;
				    }

				    if( face[2]&&face[3] )
				    {
					//x faces
					if(face[0])
					    face_vector_A[0] = -dy*dz;
					if(face[1])
					    face_vector_A[0] =  dy*dz;

					//z faces
					if(face[4])
					    face_vector_A[2] = -dx*dy;
					if(face[5])
					    face_vector_A[2] =  dx*dy;
				    }

				    if( face[4]&&face[5] )
				    {
					//x faces
					if(face[0])
					    face_vector_A[0] = -dy*dz;
					if(face[1])
					    face_vector_A[0] =  dy*dz;

					//y faces
					if(face[2])
					    face_vector_A[1] = -dz*dx;
					if(face[3])
					    face_vector_A[1] =  dz*dx;
				    }

				}else{


				    //adjacent faces supply chamfer of triangle of hypotenuses
				    //a_face will lie in the inner most conditioned face

				    //there must be an x face
				    if(face[0])	//+x
				    {
					if(face[2]) //+y
					{
					    if(face[4])	//+z
					    {
						//+x +y +z; triangle points down to (dx, dy, 0)
						a_face[0] =  0;
						a_face[1] = -dy;
						a_face[2] =  dz;

						b_face[0] = -dx;
						b_face[1] =  0;
						b_face[2] =  dz;

					    }else{	//-z

						//+x +y -z;  triangle points up to (dx, dy, dz)
						a_face[0] =  dx;
						a_face[1] = -dy;
						a_face[2] =  0;

						b_face[0] =  dx;
						b_face[1] =  0;
						b_face[2] =  dz;
					    }
					}else{	//-y
					    if(face[4])	//+z
					    {
						//+x -y +z; triangle points down to (dx, 0, 0)
						a_face[0] =  dx;
						a_face[1] =  dy;
						a_face[2] =  0;

						b_face[0] =  dx;
						b_face[1] =  0;
						b_face[2] = -dz;

					    }else{	//-z ?

						//+x -y -z;  triangle points up to (dx, 0, dz)
						a_face[0] =  0;
						a_face[1] =  dy;
						a_face[2] = -dz;

						b_face[0] = -dx;
						b_face[1] =  0;
						b_face[2] = -dz;
					    }
					}
				    }else{	//-x
					if(face[2]) //+y
					{
					    if(face[4])	//+z
					    {
						//-x +y +z; triangle points down to (0, dy, 0)
						a_face[0] =  dx;
						a_face[1] =  0;
						a_face[2] =  dz;

						b_face[0] =  0;
						b_face[1] = -dy;
						b_face[2] =  dz;

					    }else{	//-z

						//-x +y -z;  triangle points up to (0, dy, dz)
						a_face[0] =  dx;
						a_face[1] =  dy;
						a_face[2] =  0;

						b_face[0] =  0;
						b_face[1] =  dy;
						b_face[2] =  dz;
					    }
					}else{	//-y
					    if(face[4])	//+z
					    {
						//-x -y +z; triangle points down to (0, 0, 0)
						a_face[0] =  0;
						a_face[1] =  dy;
						a_face[2] =  dz;

						b_face[0] =  dx;
						b_face[1] =  0;
						b_face[2] =  dz;

					    }else{	//-z

						//-x -y -z;  triangle points up to (0, 0, dz)
						a_face[0] =  0;
						a_face[1] = -dy;
						a_face[2] =  dz;

						b_face[0] =  dx;
						b_face[1] = -dy;
						b_face[2] =  0;
					    }
					}
				    }

				    //vector area of parallelogram
				    vector_cross_product_in_place(&face_vector_A[0],a_face,b_face,3);

				    //vector area of triangle needs to be scaled by 1/2
				    //for(int l=0;l<3;l++)
				//	face_vector_A[l] *= 0.5;
				    //however, we often miss about the same area in the corners
				}   
				
				A_face = vector_dot_product(face_vector_A,dir_norm,3);
				if(A_face>0)
				    dcase[3] += A_face;
				break;


			//if 4 faces contribute, only add their hypotenusal area
			case 4: fcase[4]++;


				//tunnels don't contribute
				if(face[0]&&face[1]&&face[2]&&face[3])
				    break;
				if(face[2]&&face[3]&&face[4]&&face[5])
				    break;
				if(face[4]&&face[5]&&face[0]&&face[1])
				    break;

				//add open face

				if( face[0]&&face[1] )
				{

				    //surface runs diagonal across the x face
				    if(face[2]&&face[4])
				    {
					//+y and +z
					a_face[0] =  dx;
					a_face[1] =  0;
					a_face[2] =  0;

					b_face[0] =  0;
					b_face[1] = -dy;
					b_face[2] =  dz;

				    }else if(face[2]&&face[5]){

					//+y and -z
					a_face[0] =  dx;
					a_face[1] =  0;
					a_face[2] =  0;

					b_face[0] =  0;
					b_face[1] =  dy;
					b_face[2] =  dz;

				    }else if(face[3]&&face[4]){

					//-y and +z
					a_face[0] = -dx;
					a_face[1] =  0;
					a_face[2] =  0;

					b_face[0] =  0;
					b_face[1] = -dy;
					b_face[2] = -dz;

				    }else{

					//-y and -z
					a_face[0] = -dx;
					a_face[1] =  0;
					a_face[2] =  0;

					b_face[0] =  0;
					b_face[1] = -dy;
					b_face[2] =  dz;
				    }

				}else{

				    if( face[2]&&face[3] )
				    {
					//surface runs diagonal across the y face


					if(face[0]&&face[4])
					{
					   //+x and +z
					    a_face[0] =  0;
					    a_face[1] =  dy;
					    a_face[2] =  0;

					    b_face[0] =  dx;
					    b_face[1] =  0;
					    b_face[2] = -dz;

					}else if(face[0]&&face[5]){

					   //+x and -z
					    a_face[0] =  0;
					    a_face[1] =  dy;
					    a_face[2] =  0;

					    b_face[0] = -dx; 
					    b_face[1] =  0;
					    b_face[2] = -dz;

					}else if(face[1]&&face[4]){

					   //-x and +z
					    a_face[0] =  0;
					    a_face[1] = -dy;
					    a_face[2] =  0;

					    b_face[0] = -dx;
					    b_face[1] =  0;
					    b_face[2] = -dz;

					}else{

					    //-x and -z
					    a_face[0] =  0;
					    a_face[1] = -dy;
					    a_face[2] =  0;

					    b_face[0] =  dx;
					    b_face[1] =  0;
					    b_face[2] = -dz;
					}

				    }else{
					//surface runs diagonal across the z face

					if(face[0]&&face[2])
					{
					   //+x and +y
					    a_face[0] =  0;
					    a_face[1] =  0;
					    a_face[2] =  dz;

					    b_face[0] = -dx;
					    b_face[1] =  dy;
					    b_face[2] =  0;

					}else if(face[0]&&face[3]){

					   //+x and -y
					    a_face[0] =  0;
					    a_face[1] =  0;
					    a_face[2] =  dz;

					    b_face[0] =  dx;
					    b_face[1] =  dy;
					    b_face[2] =  0;

					}else if(face[1]&&face[2]){

					   //-x and +y
					    a_face[0] =  0;
					    a_face[1] =  0;
					    a_face[2] = -dz;

					    b_face[0] =  dx;
					    b_face[1] =  dy;
					    b_face[2] =  0;

					}else{

					    //-x and -y
					    a_face[0] =  0;
					    a_face[1] =  0;
					    a_face[2] = -dz;

					    b_face[0] = -dx;
					    b_face[1] =  dy;
					    b_face[2] =  0;
					}

				    }
				}

				//find vector area of parallelogram
				vector_cross_product_in_place(&face_vector_A[0],a_face,b_face,3);

				A_face = vector_dot_product(face_vector_A,dir_norm,3);
				if(A_face>0)
				    dcase[4] += A_face;
				break;


			//if 5 contribute, only the open face is added
			case 5: fcase[5]++;

				if(!face[0])
				    face_vector_A[0] =  dy*dz;
				if(!face[1])
				    face_vector_A[0] = -dy*dz;

				if(!face[2])
				    face_vector_A[1] =  dx*dz;
				if(!face[3])
				    face_vector_A[1] = -dx*dz;

				if(!face[4])
				    face_vector_A[2] =  dx*dy;
				if(!face[5])
				    face_vector_A[2] = -dx*dy;

				A_face = vector_dot_product(face_vector_A,dir_norm,3);
				if(A_face>0)
				    dcase[5] += A_face;
				break;

			//if 0 or 6 faces, no area
			default:    break;

		    }

		    //Add the surface area
		    A_face = vector_dot_product(face_vector_A,dir_norm,3);
		    if(A_face>0)
			A_local += A_face;
		}
	}

    //do upper boundary second
    if(nx_local)
    for(int j=0;j<ny;j++)
	for(int k=0;k<nz;k++)
	{
		//get cell index
		ijk = grid_ijk(nx_local-1,j,k,grid_info);

		//consider only zero cells near the interface
		n_faces = 0;
		for(int l=0;l<6;l++)
		    face[l] = 0;

		//get the value of the vector field at this cell
		for(int l=0;l<ndim;l++)
		    dir_norm[l] = vfield[l][ijk];

		if(mask[ijk]==0)
		{

		    ijkt = nz*j + k;
		    if(ghost_cell_upper[ijkt]!=0)
		    {
			n_faces++;
			face[0] = 1;
		    }

		    //causes double counting for some splits

		    ijkt = grid_ijk(nx_local-2,j,k,grid_info);
		    if(mask[ijkt]!=0)
		    {
			n_faces++;
			face[1] = 1;
		    }

		    //ijkt = grid_ijk(nx_local-1,j-1,k,grid_info);
		    ijkt = grid_ijk(nx_local-1,j+1,k,grid_info);
		    if(mask[ijkt]!=0)
		    {
			n_faces++;
			face[2] = 1;
		    }

		    //ijkt = grid_ijk(nx_local-1,j+1,k,grid_info);
		    ijkt = grid_ijk(nx_local-1,j-1,k,grid_info);
		    if(mask[ijkt]!=0)
		    {
			n_faces++;
			face[3] = 1;
		    }

		    //ijkt = grid_ijk(nx_local-1,j,k-1,grid_info);
		    ijkt = grid_ijk(nx_local-1,j,k+1,grid_info);
		    if(mask[ijkt]!=0)
		    {
			n_faces++;
			face[4] = 1;
		    }

		    //ijkt = grid_ijk(nx_local-1,j,k+1,grid_info);
		    ijkt = grid_ijk(nx_local-1,j,k-1,grid_info);
		    if(mask[ijkt]!=0)
		    {
			n_faces++;
			face[5] = 1;
		    }

		    //surface area contribution of the face
		    //in question
		    A_face = 0;
		    for(int l=0;l<3;l++)
		    {
			face_vector_A[l] = 0;
			a_face[l] = 0;
			b_face[l] = 0;
		    }

		    //add area from this cell
		    switch(n_faces)
		    {


			//if no faces contribute, just break
			case 0:	fcase[0]++;
				break;

			//if 1 face contributes, add its area
			case 1:	fcase[1]++;

				//n_tested++;
				if(face[0])
				    face_vector_A[0] = -dy * dz;
				
				if(face[1])
				    face_vector_A[0] =  dy * dz;
				
				if(face[2])
				    face_vector_A[1] = -dz * dx;

				if(face[3])
				    face_vector_A[1] =  dz * dx;

				if(face[4])
				    face_vector_A[2] = -dx * dy;
				
				if(face[5])
				    face_vector_A[2] =  dx * dy;
				
				A_face = vector_dot_product(face_vector_A,dir_norm,3);
				if(A_face>0)
				    dcase[1] += A_face;
				break;


			//if 2 faces contribute and are adjacent, add their hypotenusal area
			case 2:	fcase[2]++;

				//opposing faces don't contribute
				if( (face[0]&&face[1])||(face[2]&&face[3])||(face[4]&&face[5]) )
				    break;

				//add the hypotenuse only if the dot product is positive
				if(face[0]) //+x face
				{
					if(face[2])
					{
						//+x/+y, checked 05/22
						a_face[2] =  dz;

						b_face[0] = -dx;
						b_face[1] =  dy;

					}else if(face[3]){

						//+x/-y, checked 05/22
						a_face[2] =  dz;

						b_face[0] =  dx;
						b_face[1] =  dy;
					}

					if(face[4])
					{
						//+x/+z,checked 05/22
						a_face[1] =  dy;

						b_face[0] =  dx;
						b_face[2] = -dz;

					}else if(face[5]){

						//+x/-z, checked 05/22
						a_face[1] =  dy;

						b_face[0] =  -dx;
						b_face[2] =  -dz;
					}

				}else if(face[1]){ //-x face

					if(face[2])
					{
						//-x/+y, checked 05/22
						a_face[2] =  dz;

						b_face[0] = -dx;
						b_face[1] = -dy;

					}else if(face[3]){

						//-x/-y, checked 05/22
						a_face[2] =  dz;

						b_face[0] =  dx;
						b_face[1] = -dy;

					}

					if(face[4])
					{
						//-x/+z, checked 05/22
						a_face[1] =  dy;

						b_face[0] =  dx;
						b_face[2] =  dz;

					}else if(face[5]){

						//-x/-z, checked 05/22
						a_face[1] =  dy;

						b_face[0] = -dx;
						b_face[2] =  dz;
					}

				}else if(face[2]){  //+y face, no x-face

					//+y face 
					if(face[4])
					{
						//+y/+z, checked 05/22
						a_face[0] =  dx;

						b_face[1] = -dy;
						b_face[2] =  dz;

					}else if(face[5]){

						//+y/-z, checked 05/22
						a_face[0] =  dx;

						b_face[1] =  dy;
						b_face[2] =  dz;
					}

				}else if(face[3]){ //-y face, no x-face

					//-y face
					if(face[4])
					{
						//-y/+z, checked 05/22
						a_face[0] =  dx;

						b_face[1] = -dy;
						b_face[2] = -dz;

					}else if(face[5]){

						//-y/-z, checked 05/22
						a_face[0] =  dx;

						b_face[1] =  dy;
						b_face[2] = -dz;
					}

				}
				vector_cross_product_in_place(&face_vector_A[0],a_face,b_face,3);
		
				A_face = vector_dot_product(face_vector_A,dir_norm,3);
				if(A_face>0)
				    dcase[2] += A_face;
				break;

			//if 3 faces contribute, then treat adjacent and opposing faces separately
			case 3: fcase[3]++;

				//opposing faces contribute only the non-opposing face
				if( (face[0]&&face[1])||(face[2]&&face[3])||(face[4]&&face[5]) )
				{

				    if( face[0]&&face[1] )
				    {
					//y faces
					if(face[2])
					    face_vector_A[1] = -dz*dx;
					if(face[3])
					    face_vector_A[1] =  dz*dx;
	    
					//z faces
					if(face[4])
					    face_vector_A[2] = -dx*dy;
					if(face[5])
					    face_vector_A[2] =  dx*dy;
				    }

				    if( face[2]&&face[3] )
				    {
					//x faces
					if(face[0])
					    face_vector_A[0] = -dy*dz;
					if(face[1])
					    face_vector_A[0] =  dy*dz;

					//z faces
					if(face[4])
					    face_vector_A[2] = -dx*dy;
					if(face[5])
					    face_vector_A[2] =  dx*dy;
				    }

				    if( face[4]&&face[5] )
				    {
					//x faces
					if(face[0])
					    face_vector_A[0] = -dy*dz;
					if(face[1])
					    face_vector_A[0] =  dy*dz;

					//y faces
					if(face[2])
					    face_vector_A[1] = -dz*dx;
					if(face[3])
					    face_vector_A[1] =  dz*dx;
				    }

				}else{


				    //adjacent faces supply chamfer of triangle of hypotenuses
				    //a_face will lie in the inner most conditioned face

				    //there must be an x face
				    if(face[0])	//+x
				    {
					if(face[2]) //+y
					{
					    if(face[4])	//+z
					    {
						//+x +y +z; triangle points down to (dx, dy, 0)
						a_face[0] =  0;
						a_face[1] = -dy;
						a_face[2] =  dz;

						b_face[0] = -dx;
						b_face[1] =  0;
						b_face[2] =  dz;

					    }else{	//-z

						//+x +y -z;  triangle points up to (dx, dy, dz)
						a_face[0] =  dx;
						a_face[1] = -dy;
						a_face[2] =  0;

						b_face[0] =  dx;
						b_face[1] =  0;
						b_face[2] =  dz;
					    }
					}else{	//-y
					    if(face[4])	//+z
					    {
						//+x -y +z; triangle points down to (dx, 0, 0)
						a_face[0] =  dx;
						a_face[1] =  dy;
						a_face[2] =  0;

						b_face[0] =  dx;
						b_face[1] =  0;
						b_face[2] = -dz;

					    }else{	//-z ?

						//+x -y -z;  triangle points up to (dx, 0, dz)
						a_face[0] =  0;
						a_face[1] =  dy;
						a_face[2] = -dz;

						b_face[0] = -dx;
						b_face[1] =  0;
						b_face[2] = -dz;
					    }
					}
				    }else{	//-x
					if(face[2]) //+y
					{
					    if(face[4])	//+z
					    {
						//-x +y +z; triangle points down to (0, dy, 0)
						a_face[0] =  dx;
						a_face[1] =  0;
						a_face[2] =  dz;

						b_face[0] =  0;
						b_face[1] = -dy;
						b_face[2] =  dz;

					    }else{	//-z

						//-x +y -z;  triangle points up to (0, dy, dz)
						a_face[0] =  dx;
						a_face[1] =  dy;
						a_face[2] =  0;

						b_face[0] =  0;
						b_face[1] =  dy;
						b_face[2] =  dz;
					    }
					}else{	//-y
					    if(face[4])	//+z
					    {
						//-x -y +z; triangle points down to (0, 0, 0)
						a_face[0] =  0;
						a_face[1] =  dy;
						a_face[2] =  dz;

						b_face[0] =  dx;
						b_face[1] =  0;
						b_face[2] =  dz;

					    }else{	//-z

						//-x -y -z;  triangle points up to (0, 0, dz)
						a_face[0] =  0;
						a_face[1] = -dy;
						a_face[2] =  dz;

						b_face[0] =  dx;
						b_face[1] = -dy;
						b_face[2] =  0;
					    }
					}
				    }

				    //vector area of parallelogram
				    vector_cross_product_in_place(&face_vector_A[0],a_face,b_face,3);

				    //vector area of triangle needs to be scaled by 1/2
				    //for(int l=0;l<3;l++)
				//	face_vector_A[l] *= 0.5;
				    //however, we often miss about the same area in the corners
				}   
				
				A_face = vector_dot_product(face_vector_A,dir_norm,3);
				if(A_face>0)
				    dcase[3] += A_face;
				break;


			//if 4 faces contribute, only add their hypotenusal area
			case 4: fcase[4]++;


				//tunnels don't contribute
				if(face[0]&&face[1]&&face[2]&&face[3])
				    break;
				if(face[2]&&face[3]&&face[4]&&face[5])
				    break;
				if(face[4]&&face[5]&&face[0]&&face[1])
				    break;

				//add open face

				if( face[0]&&face[1] )
				{

				    //surface runs diagonal across the x face
				    if(face[2]&&face[4])
				    {
					//+y and +z
					a_face[0] =  dx;
					a_face[1] =  0;
					a_face[2] =  0;

					b_face[0] =  0;
					b_face[1] = -dy;
					b_face[2] =  dz;

				    }else if(face[2]&&face[5]){

					//+y and -z
					a_face[0] =  dx;
					a_face[1] =  0;
					a_face[2] =  0;

					b_face[0] =  0;
					b_face[1] =  dy;
					b_face[2] =  dz;

				    }else if(face[3]&&face[4]){

					//-y and +z
					a_face[0] = -dx;
					a_face[1] =  0;
					a_face[2] =  0;

					b_face[0] =  0;
					b_face[1] = -dy;
					b_face[2] = -dz;

				    }else{

					//-y and -z
					a_face[0] = -dx;
					a_face[1] =  0;
					a_face[2] =  0;

					b_face[0] =  0;
					b_face[1] = -dy;
					b_face[2] =  dz;
				    }

				}else{

				    if( face[2]&&face[3] )
				    {
					//surface runs diagonal across the y face


					if(face[0]&&face[4])
					{
					   //+x and +z
					    a_face[0] =  0;
					    a_face[1] =  dy;
					    a_face[2] =  0;

					    b_face[0] =  dx;
					    b_face[1] =  0;
					    b_face[2] = -dz;

					}else if(face[0]&&face[5]){

					   //+x and -z
					    a_face[0] =  0;
					    a_face[1] =  dy;
					    a_face[2] =  0;

					    b_face[0] = -dx; 
					    b_face[1] =  0;
					    b_face[2] = -dz;

					}else if(face[1]&&face[4]){

					   //-x and +z
					    a_face[0] =  0;
					    a_face[1] = -dy;
					    a_face[2] =  0;

					    b_face[0] = -dx;
					    b_face[1] =  0;
					    b_face[2] = -dz;

					}else{

					    //-x and -z
					    a_face[0] =  0;
					    a_face[1] = -dy;
					    a_face[2] =  0;

					    b_face[0] =  dx;
					    b_face[1] =  0;
					    b_face[2] = -dz;
					}

				    }else{
					//surface runs diagonal across the z face

					if(face[0]&&face[2])
					{
					   //+x and +y
					    a_face[0] =  0;
					    a_face[1] =  0;
					    a_face[2] =  dz;

					    b_face[0] = -dx;
					    b_face[1] =  dy;
					    b_face[2] =  0;

					}else if(face[0]&&face[3]){

					   //+x and -y
					    a_face[0] =  0;
					    a_face[1] =  0;
					    a_face[2] =  dz;

					    b_face[0] =  dx;
					    b_face[1] =  dy;
					    b_face[2] =  0;

					}else if(face[1]&&face[2]){

					   //-x and +y
					    a_face[0] =  0;
					    a_face[1] =  0;
					    a_face[2] = -dz;

					    b_face[0] =  dx;
					    b_face[1] =  dy;
					    b_face[2] =  0;

					}else{

					    //-x and -y
					    a_face[0] =  0;
					    a_face[1] =  0;
					    a_face[2] = -dz;

					    b_face[0] = -dx;
					    b_face[1] =  dy;
					    b_face[2] =  0;
					}

				    }
				}

				//find vector area of parallelogram
				vector_cross_product_in_place(&face_vector_A[0],a_face,b_face,3);

				A_face = vector_dot_product(face_vector_A,dir_norm,3);
				if(A_face>0)
				    dcase[4] += A_face;
				break;


			//if 5 contribute, only the open face is added
			case 5: fcase[5]++;

				if(!face[0])
				    face_vector_A[0] =  dy*dz;
				if(!face[1])
				    face_vector_A[0] = -dy*dz;

				if(!face[2])
				    face_vector_A[1] =  dx*dz;
				if(!face[3])
				    face_vector_A[1] = -dx*dz;

				if(!face[4])
				    face_vector_A[2] =  dx*dy;
				if(!face[5])
				    face_vector_A[2] = -dx*dy;

				A_face = vector_dot_product(face_vector_A,dir_norm,3);
				if(A_face>0)
				    dcase[5] += A_face;
				break;

			//if 0 or 6 faces, no area
			default:    break;
		    }

		    //Add the surface area
		    A_face = vector_dot_product(face_vector_A,dir_norm,3);
		    if(A_face>0)
			A_local += A_face;

		}
	}

    //free ghost cells
    free(ghost_cell_upper);
    free(ghost_cell_lower);


    //sum area across processors
    MPI_Allreduce(&A_local,&A_total,1,MPI_DOUBLE,MPI_SUM,world);


    //find the total surface area
    return A_total;

}

/*! \fn double **grid_vector_surface_area(double *mask, double **vfield, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world)
 *  \brief Find the approximate vector surface area at the mask boundary, for regions with positive dot product with the vfield */
double **grid_vector_surface_area(double *mask, double **vfield, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world)
{
    int ijk;	//cell
    int ijkt;	//adjacent cells
    int nx_local = grid_info.nx_local;
    int nx_local_start = grid_info.nx_local_start;
    int ny       = grid_info.ny;
    int nz       = grid_info.nz;

    double dx	= grid_info.dx;
    double dy	= grid_info.dy;
    double dz	= grid_info.dz;

    int ndim = grid_info.ndim;

    int dest;	//destination processor for communication
    int source;	//source processor for communication

    double **varea; //vector surface area

    MPI_Status status;	//MPI status for communication


    double *ghost_cell_upper;	//ghost cells at i = nx_local
    double *ghost_cell_lower;	//ghost cells at i = -1

    //local surface area
    double A_local = 0;
    
    //total surface area
    double A_total = 0;

    //number of faces contributing to surface
    int n_faces;
    int face[6];

    //perimeter
    double s;

    //triangle sides
    double a, b, c;

    int n_tested = 0;

    double face_vector_A[3];  //vector area of the face
    double a_face[3];	      //first vector of the face
    double b_face[3];	      //second vector of the face

    double dir_norm[3];	      //value of the vector field at this cell

    int fcase[6] = {0,0,0,0,0,0};
    double dcase[6] = {0,0,0,0,0,0};

    double A_face;

    //allocate the vector surface area
    varea = allocate_field_fftw_grid(grid_info.ndim, grid_info);

    //we have to check boundary cells separately
    //so start with inner cells
    for(int i=1;i<nx_local-1;i++)
	for(int j=0;j<ny;j++)
	    for(int k=0;k<nz;k++)
	    {

		//get cell index
		ijk = grid_ijk(i,j,k,grid_info);

		//consider only zero cells near the interface
		n_faces = 0;
		for(int l=0;l<6;l++)
		    face[l] = 0;

		//get the value of the vector field at this cell
		for(int l=0;l<ndim;l++)
		    dir_norm[l] = vfield[l][ijk];

		if(mask[ijk]==0)
		{

		    ijkt = grid_ijk(i+1,j,k,grid_info);
		    if(mask[ijkt]!=0)
		    {
			n_faces++;
			face[0] = 1;
		    }

		    ijkt = grid_ijk(i-1,j,k,grid_info);
		    if(mask[ijkt]!=0)
		    {
			n_faces++;
			face[1] = 1;
		    }


		    ijkt = grid_ijk(i,j+1,k,grid_info);
		    if(mask[ijkt]!=0)
		    {
			n_faces++;
			face[2] = 1;
		    }

		    ijkt = grid_ijk(i,j-1,k,grid_info);
		    if(mask[ijkt]!=0)
		    {
			n_faces++;
			face[3] = 1;
		    }

		    ijkt = grid_ijk(i,j,k+1,grid_info);
		    if(mask[ijkt]!=0)
		    {
			n_faces++;
			face[4] = 1;
		    }

		    ijkt = grid_ijk(i,j,k-1,grid_info);
		    if(mask[ijkt]!=0)
		    {
			n_faces++;
			face[5] = 1;
		    }


		    //surface area contribution of the face
		    //in question
		    A_face = 0;
		    for(int l=0;l<3;l++)
		    {
			face_vector_A[l] = 0;
			a_face[l] = 0;
			b_face[l] = 0;
		    }

		    //add area from this cell
		    switch(n_faces)
		    {

			//if no faces contribute, just break
			case 0:	fcase[0]++;
				break;

			//if 1 face contributes, add its area
			case 1:	

				//n_tested++;
				if(face[0])
				    face_vector_A[0] = -dy * dz;
				
				if(face[1])
				    face_vector_A[0] =  dy * dz;
				
				if(face[2])
				    face_vector_A[1] = -dz * dx;

				if(face[3])
				    face_vector_A[1] =  dz * dx;

				if(face[4])
				    face_vector_A[2] = -dx * dy;
				
				if(face[5])
				    face_vector_A[2] =  dx * dy;
				
				A_face = vector_dot_product(face_vector_A,dir_norm,3);
				if(A_face>0)
				{
				    fcase[1]++;
				    dcase[1] += A_face;
				}
				break;


			//if 2 faces contribute and are adjacent, add their hypotenusal area
			case 2:	fcase[2]++;

				//opposing faces don't contribute
				if( (face[0]&&face[1])||(face[2]&&face[3])||(face[4]&&face[5]) )
				    break;

				//add the hypotenuse only if the dot product is positive
				if(face[0]) //+x face
				{
					if(face[2])
					{
						//+x/+y, checked 05/22
						a_face[2] =  dz;

						b_face[0] = -dx;
						b_face[1] =  dy;

					}else if(face[3]){

						//+x/-y, checked 05/22
						a_face[2] =  dz;

						b_face[0] =  dx;
						b_face[1] =  dy;
					}

					if(face[4])
					{
						//+x/+z,checked 05/22
						a_face[1] =  dy;

						b_face[0] =  dx;
						b_face[2] = -dz;

					}else if(face[5]){

						//+x/-z, checked 05/22
						a_face[1] =  dy;

						b_face[0] =  -dx;
						b_face[2] =  -dz;
					}

				}else if(face[1]){ //-x face

					if(face[2])
					{
						//-x/+y, checked 05/22
						a_face[2] =  dz;

						b_face[0] = -dx;
						b_face[1] = -dy;

					}else if(face[3]){

						//-x/-y, checked 05/22
						a_face[2] =  dz;

						b_face[0] =  dx;
						b_face[1] = -dy;

					}

					if(face[4])
					{
						//-x/+z, checked 05/22
						a_face[1] =  dy;

						b_face[0] =  dx;
						b_face[2] =  dz;

					}else if(face[5]){

						//-x/-z, checked 05/22
						a_face[1] =  dy;

						b_face[0] = -dx;
						b_face[2] =  dz;
					}

				}else if(face[2]){  //+y face, no x-face

					//+y face 
					if(face[4])
					{
						//+y/+z, checked 05/22
						a_face[0] =  dx;

						b_face[1] = -dy;
						b_face[2] =  dz;

					}else if(face[5]){

						//+y/-z, checked 05/22
						a_face[0] =  dx;

						b_face[1] =  dy;
						b_face[2] =  dz;
					}

				}else if(face[3]){ //-y face, no x-face

					//-y face
					if(face[4])
					{
						//-y/+z, checked 05/22
						a_face[0] =  dx;

						b_face[1] = -dy;
						b_face[2] = -dz;

					}else if(face[5]){

						//-y/-z, checked 05/22
						a_face[0] =  dx;

						b_face[1] =  dy;
						b_face[2] = -dz;
					}

				}
				vector_cross_product_in_place(&face_vector_A[0],a_face,b_face,3);
		
				A_face = vector_dot_product(face_vector_A,dir_norm,3);
				if(A_face>0)
				    dcase[2] += A_face;
				break;

			//if 3 faces contribute, then treat adjacent and opposing faces separately
			case 3: fcase[3]++;

				//opposing faces contribute only the non-opposing face
				if( (face[0]&&face[1])||(face[2]&&face[3])||(face[4]&&face[5]) )
				{

				    if( face[0]&&face[1] )
				    {
					//y faces
					if(face[2])
					    face_vector_A[1] = -dz*dx;
					if(face[3])
					    face_vector_A[1] =  dz*dx;
	    
					//z faces
					if(face[4])
					    face_vector_A[2] = -dx*dy;
					if(face[5])
					    face_vector_A[2] =  dx*dy;
				    }

				    if( face[2]&&face[3] )
				    {
					//x faces
					if(face[0])
					    face_vector_A[0] = -dy*dz;
					if(face[1])
					    face_vector_A[0] =  dy*dz;

					//z faces
					if(face[4])
					    face_vector_A[2] = -dx*dy;
					if(face[5])
					    face_vector_A[2] =  dx*dy;
				    }

				    if( face[4]&&face[5] )
				    {
					//x faces
					if(face[0])
					    face_vector_A[0] = -dy*dz;
					if(face[1])
					    face_vector_A[0] =  dy*dz;

					//y faces
					if(face[2])
					    face_vector_A[1] = -dz*dx;
					if(face[3])
					    face_vector_A[1] =  dz*dx;
				    }

				}else{


				    //adjacent faces supply chamfer of triangle of hypotenuses
				    //a_face will lie in the inner most conditioned face

				    //there must be an x face
				    if(face[0])	//+x
				    {
					if(face[2]) //+y
					{
					    if(face[4])	//+z
					    {
						//+x +y +z; triangle points down to (dx, dy, 0)
						a_face[0] =  0;
						a_face[1] = -dy;
						a_face[2] =  dz;

						b_face[0] = -dx;
						b_face[1] =  0;
						b_face[2] =  dz;

					    }else{	//-z

						//+x +y -z;  triangle points up to (dx, dy, dz)
						a_face[0] =  dx;
						a_face[1] = -dy;
						a_face[2] =  0;

						b_face[0] =  dx;
						b_face[1] =  0;
						b_face[2] =  dz;
					    }
					}else{	//-y
					    if(face[4])	//+z
					    {
						//+x -y +z; triangle points down to (dx, 0, 0)
						a_face[0] =  dx;
						a_face[1] =  dy;
						a_face[2] =  0;

						b_face[0] =  dx;
						b_face[1] =  0;
						b_face[2] = -dz;

					    }else{	//-z ?

						//+x -y -z;  triangle points up to (dx, 0, dz)
						a_face[0] =  0;
						a_face[1] =  dy;
						a_face[2] = -dz;

						b_face[0] = -dx;
						b_face[1] =  0;
						b_face[2] = -dz;
					    }
					}
				    }else{	//-x
					if(face[2]) //+y
					{
					    if(face[4])	//+z
					    {
						//-x +y +z; triangle points down to (0, dy, 0)
						a_face[0] =  dx;
						a_face[1] =  0;
						a_face[2] =  dz;

						b_face[0] =  0;
						b_face[1] = -dy;
						b_face[2] =  dz;

					    }else{	//-z

						//-x +y -z;  triangle points up to (0, dy, dz)
						a_face[0] =  dx;
						a_face[1] =  dy;
						a_face[2] =  0;

						b_face[0] =  0;
						b_face[1] =  dy;
						b_face[2] =  dz;
					    }
					}else{	//-y
					    if(face[4])	//+z
					    {
						//-x -y +z; triangle points down to (0, 0, 0)
						a_face[0] =  0;
						a_face[1] =  dy;
						a_face[2] =  dz;

						b_face[0] =  dx;
						b_face[1] =  0;
						b_face[2] =  dz;

					    }else{	//-z

						//-x -y -z;  triangle points up to (0, 0, dz)
						a_face[0] =  0;
						a_face[1] = -dy;
						a_face[2] =  dz;

						b_face[0] =  dx;
						b_face[1] = -dy;
						b_face[2] =  0;
					    }
					}
				    }

				    //vector area of parallelogram
				    vector_cross_product_in_place(&face_vector_A[0],a_face,b_face,3);

				    //vector area of triangle needs to be scaled by 1/2
				    //for(int l=0;l<3;l++)
				//	face_vector_A[l] *= 0.5;
				    //however, we often miss about the same area in the corners
				}   
				
				A_face = vector_dot_product(face_vector_A,dir_norm,3);
				if(A_face>0)
				    dcase[3] += A_face;
				break;


			//if 4 faces contribute, only add their hypotenusal area
			case 4: fcase[4]++;


				//tunnels don't contribute
				if(face[0]&&face[1]&&face[2]&&face[3])
				    break;
				if(face[2]&&face[3]&&face[4]&&face[5])
				    break;
				if(face[4]&&face[5]&&face[0]&&face[1])
				    break;

				//add open face

				if( face[0]&&face[1] )
				{

				    //surface runs diagonal across the x face
				    if(face[2]&&face[4])
				    {
					//+y and +z
					a_face[0] =  dx;
					a_face[1] =  0;
					a_face[2] =  0;

					b_face[0] =  0;
					b_face[1] = -dy;
					b_face[2] =  dz;

				    }else if(face[2]&&face[5]){

					//+y and -z
					a_face[0] =  dx;
					a_face[1] =  0;
					a_face[2] =  0;

					b_face[0] =  0;
					b_face[1] =  dy;
					b_face[2] =  dz;

				    }else if(face[3]&&face[4]){

					//-y and +z
					a_face[0] = -dx;
					a_face[1] =  0;
					a_face[2] =  0;

					b_face[0] =  0;
					b_face[1] = -dy;
					b_face[2] = -dz;

				    }else{

					//-y and -z
					a_face[0] = -dx;
					a_face[1] =  0;
					a_face[2] =  0;

					b_face[0] =  0;
					b_face[1] = -dy;
					b_face[2] =  dz;
				    }

				}else{

				    if( face[2]&&face[3] )
				    {
					//surface runs diagonal across the y face


					if(face[0]&&face[4])
					{
					   //+x and +z
					    a_face[0] =  0;
					    a_face[1] =  dy;
					    a_face[2] =  0;

					    b_face[0] =  dx;
					    b_face[1] =  0;
					    b_face[2] = -dz;

					}else if(face[0]&&face[5]){

					   //+x and -z
					    a_face[0] =  0;
					    a_face[1] =  dy;
					    a_face[2] =  0;

					    b_face[0] = -dx; 
					    b_face[1] =  0;
					    b_face[2] = -dz;

					}else if(face[1]&&face[4]){

					   //-x and +z
					    a_face[0] =  0;
					    a_face[1] = -dy;
					    a_face[2] =  0;

					    b_face[0] = -dx;
					    b_face[1] =  0;
					    b_face[2] = -dz;

					}else{

					    //-x and -z
					    a_face[0] =  0;
					    a_face[1] = -dy;
					    a_face[2] =  0;

					    b_face[0] =  dx;
					    b_face[1] =  0;
					    b_face[2] = -dz;
					}

				    }else{
					//surface runs diagonal across the z face

					if(face[0]&&face[2])
					{
					   //+x and +y
					    a_face[0] =  0;
					    a_face[1] =  0;
					    a_face[2] =  dz;

					    b_face[0] = -dx;
					    b_face[1] =  dy;
					    b_face[2] =  0;

					}else if(face[0]&&face[3]){

					   //+x and -y
					    a_face[0] =  0;
					    a_face[1] =  0;
					    a_face[2] =  dz;

					    b_face[0] =  dx;
					    b_face[1] =  dy;
					    b_face[2] =  0;

					}else if(face[1]&&face[2]){

					   //-x and +y
					    a_face[0] =  0;
					    a_face[1] =  0;
					    a_face[2] = -dz;

					    b_face[0] =  dx;
					    b_face[1] =  dy;
					    b_face[2] =  0;

					}else{

					    //-x and -y
					    a_face[0] =  0;
					    a_face[1] =  0;
					    a_face[2] = -dz;

					    b_face[0] = -dx;
					    b_face[1] =  dy;
					    b_face[2] =  0;
					}

				    }
				}

				//find vector area of parallelogram
				vector_cross_product_in_place(&face_vector_A[0],a_face,b_face,3);

				A_face = vector_dot_product(face_vector_A,dir_norm,3);
				if(A_face>0)
				    dcase[4] += A_face;
				break;


			//if 5 contribute, only the open face is added
			case 5: fcase[5]++;

				if(!face[0])
				    face_vector_A[0] =  dy*dz;
				if(!face[1])
				    face_vector_A[0] = -dy*dz;

				if(!face[2])
				    face_vector_A[1] =  dx*dz;
				if(!face[3])
				    face_vector_A[1] = -dx*dz;

				if(!face[4])
				    face_vector_A[2] =  dx*dy;
				if(!face[5])
				    face_vector_A[2] = -dx*dy;

				A_face = vector_dot_product(face_vector_A,dir_norm,3);
				if(A_face>0)
				    dcase[5] += A_face;
				break;

			//if 0 or 6 faces, no area
			default:    break;
		    }

		    //find the dot product of the local cell vector area and the
		    //local velocity field direction
		    A_face = vector_dot_product(face_vector_A,dir_norm,3);

		    //if the dot product is positive, record the
		    //vector area
		    if(A_face>0)
			for(int l=0;l<ndim;l++)
			    varea[l][ijk] = face_vector_A[l];
		}
	    }


    //now we have to check boundary cells
    //to do this, we use a sendreceive to get adjacent
    //cells across the boundary, then repeat the checks

    //allocate ghost cells
    ghost_cell_upper = calloc_double_array(ny*nz);
    ghost_cell_lower = calloc_double_array(ny*nz); 

    //populate ghost cells
    for(int j=0;j<ny;j++)
	for(int k=0;k<nz;k++)
	{


	    //note nx_local can be zero!
	    if(nx_local)
	    {
		//upper cells to pass from this processor's lower boundary
		ijk = grid_ijk(0,j,k,grid_info);
		ghost_cell_upper[nz*j+k] = mask[ijk];

		//lower cells to pass from this processor's upper boundary
		ijk = grid_ijk(nx_local-1,j,k,grid_info);
		ghost_cell_lower[nz*j+k] = mask[ijk];


		//if(nx_local_start+nx_local-1==47)
		//	printf("j %d k %d ghost %e\n",j,k,ghost_cell_lower[nz*j+k]);
	    }else{
		//dummy cells
		ghost_cell_upper[nz*j+k] = 0;
		ghost_cell_lower[nz*j+k] = 0;
	    }
	}

    //sendrecv upper ghost cells
    source = myid+1;
    if(source>numprocs-1)
	source-=numprocs;
    dest = myid-1;
    if(dest<0)
	dest+=numprocs;

    MPI_Sendrecv_replace(ghost_cell_upper, ny*nz, MPI_DOUBLE, dest, myid, source, source, world, &status);
	

    //sendrecv lower ghost cells
    source = myid-1;
    if(source<0)
	source+=numprocs;
    dest = myid+1;
    if(dest>numprocs-1)
	dest-=numprocs;

    MPI_Sendrecv_replace(ghost_cell_lower, ny*nz, MPI_DOUBLE, dest, myid, source, source, world, &status);


    //do lower boundary first
    if(nx_local)
    for(int j=0;j<ny;j++)
	for(int k=0;k<nz;k++)
	{
		//get cell index
		ijk = grid_ijk(0,j,k,grid_info);

		//consider only zero cells near the interface
		n_faces = 0;
		for(int l=0;l<6;l++)
		    face[l] = 0;

		//get the value of the vector field at this cell
		for(int l=0;l<ndim;l++)
		    dir_norm[l] = vfield[l][ijk];

		if(mask[ijk]==0)
		{
		    ijkt = grid_ijk(1,j,k,grid_info);
		    if(mask[ijkt]!=0)
		    {
			n_faces++;
			face[0] = 1;
		    }

		    ijkt = nz*j + k;
		    if(ghost_cell_lower[ijkt]!=0)
		    {
			n_faces++;
			face[1] = 1;
		    }


		    ijkt = grid_ijk(0,j+1,k,grid_info);
		    if(mask[ijkt]!=0)
		    {
			n_faces++;
			face[2] = 1;
		    }

		    ijkt = grid_ijk(0,j-1,k,grid_info);
		    if(mask[ijkt]!=0)
		    {
			n_faces++;
			face[3] = 1;
		    }

		    ijkt = grid_ijk(0,j,k+1,grid_info);
		    if(mask[ijkt]!=0)
		    {
			n_faces++;
			face[4] = 1;
		    }

		    ijkt = grid_ijk(0,j,k-1,grid_info);
		    if(mask[ijkt]!=0)
		    {
			n_faces++;
			face[5] = 1;
		    }

		    //surface area contribution of the face
		    //in question
		    A_face = 0;
		    for(int l=0;l<3;l++)
		    {
			face_vector_A[l] = 0;
			a_face[l] = 0;
			b_face[l] = 0;
		    }

		    //add area from this cell
		    switch(n_faces)
		    {
			//if no faces contribute, just break
			case 0:	fcase[0]++;
				break;

			//if 1 face contributes, add its area
			case 1:	fcase[1]++;

				//n_tested++;
				if(face[0])
				    face_vector_A[0] = -dy * dz;
				
				if(face[1])
				    face_vector_A[0] =  dy * dz;
				
				if(face[2])
				    face_vector_A[1] = -dz * dx;

				if(face[3])
				    face_vector_A[1] =  dz * dx;

				if(face[4])
				    face_vector_A[2] = -dx * dy;
				
				if(face[5])
				    face_vector_A[2] =  dx * dy;
				
				A_face = vector_dot_product(face_vector_A,dir_norm,3);
				if(A_face>0)
				    dcase[1] += A_face;
				break;


			//if 2 faces contribute and are adjacent, add their hypotenusal area
			case 2:	fcase[2]++;

				//opposing faces don't contribute
				if( (face[0]&&face[1])||(face[2]&&face[3])||(face[4]&&face[5]) )
				    break;

				//add the hypotenuse only if the dot product is positive
				if(face[0]) //+x face
				{
					if(face[2])
					{
						//+x/+y, checked 05/22
						a_face[2] =  dz;

						b_face[0] = -dx;
						b_face[1] =  dy;

					}else if(face[3]){

						//+x/-y, checked 05/22
						a_face[2] =  dz;

						b_face[0] =  dx;
						b_face[1] =  dy;
					}

					if(face[4])
					{
						//+x/+z,checked 05/22
						a_face[1] =  dy;

						b_face[0] =  dx;
						b_face[2] = -dz;

					}else if(face[5]){

						//+x/-z, checked 05/22
						a_face[1] =  dy;

						b_face[0] =  -dx;
						b_face[2] =  -dz;
					}

				}else if(face[1]){ //-x face

					if(face[2])
					{
						//-x/+y, checked 05/22
						a_face[2] =  dz;

						b_face[0] = -dx;
						b_face[1] = -dy;

					}else if(face[3]){

						//-x/-y, checked 05/22
						a_face[2] =  dz;

						b_face[0] =  dx;
						b_face[1] = -dy;

					}

					if(face[4])
					{
						//-x/+z, checked 05/22
						a_face[1] =  dy;

						b_face[0] =  dx;
						b_face[2] =  dz;

					}else if(face[5]){

						//-x/-z, checked 05/22
						a_face[1] =  dy;

						b_face[0] = -dx;
						b_face[2] =  dz;
					}

				}else if(face[2]){  //+y face, no x-face

					//+y face 
					if(face[4])
					{
						//+y/+z, checked 05/22
						a_face[0] =  dx;

						b_face[1] = -dy;
						b_face[2] =  dz;

					}else if(face[5]){

						//+y/-z, checked 05/22
						a_face[0] =  dx;

						b_face[1] =  dy;
						b_face[2] =  dz;
					}

				}else if(face[3]){ //-y face, no x-face

					//-y face
					if(face[4])
					{
						//-y/+z, checked 05/22
						a_face[0] =  dx;

						b_face[1] = -dy;
						b_face[2] = -dz;

					}else if(face[5]){

						//-y/-z, checked 05/22
						a_face[0] =  dx;

						b_face[1] =  dy;
						b_face[2] = -dz;
					}

				}
				vector_cross_product_in_place(&face_vector_A[0],a_face,b_face,3);
		
				A_face = vector_dot_product(face_vector_A,dir_norm,3);
				if(A_face>0)
				    dcase[2] += A_face;
				break;

			//if 3 faces contribute, then treat adjacent and opposing faces separately
			case 3: fcase[3]++;

				//opposing faces contribute only the non-opposing face
				if( (face[0]&&face[1])||(face[2]&&face[3])||(face[4]&&face[5]) )
				{

				    if( face[0]&&face[1] )
				    {
					//y faces
					if(face[2])
					    face_vector_A[1] = -dz*dx;
					if(face[3])
					    face_vector_A[1] =  dz*dx;
	    
					//z faces
					if(face[4])
					    face_vector_A[2] = -dx*dy;
					if(face[5])
					    face_vector_A[2] =  dx*dy;
				    }

				    if( face[2]&&face[3] )
				    {
					//x faces
					if(face[0])
					    face_vector_A[0] = -dy*dz;
					if(face[1])
					    face_vector_A[0] =  dy*dz;

					//z faces
					if(face[4])
					    face_vector_A[2] = -dx*dy;
					if(face[5])
					    face_vector_A[2] =  dx*dy;
				    }

				    if( face[4]&&face[5] )
				    {
					//x faces
					if(face[0])
					    face_vector_A[0] = -dy*dz;
					if(face[1])
					    face_vector_A[0] =  dy*dz;

					//y faces
					if(face[2])
					    face_vector_A[1] = -dz*dx;
					if(face[3])
					    face_vector_A[1] =  dz*dx;
				    }

				}else{


				    //adjacent faces supply chamfer of triangle of hypotenuses
				    //a_face will lie in the inner most conditioned face

				    //there must be an x face
				    if(face[0])	//+x
				    {
					if(face[2]) //+y
					{
					    if(face[4])	//+z
					    {
						//+x +y +z; triangle points down to (dx, dy, 0)
						a_face[0] =  0;
						a_face[1] = -dy;
						a_face[2] =  dz;

						b_face[0] = -dx;
						b_face[1] =  0;
						b_face[2] =  dz;

					    }else{	//-z

						//+x +y -z;  triangle points up to (dx, dy, dz)
						a_face[0] =  dx;
						a_face[1] = -dy;
						a_face[2] =  0;

						b_face[0] =  dx;
						b_face[1] =  0;
						b_face[2] =  dz;
					    }
					}else{	//-y
					    if(face[4])	//+z
					    {
						//+x -y +z; triangle points down to (dx, 0, 0)
						a_face[0] =  dx;
						a_face[1] =  dy;
						a_face[2] =  0;

						b_face[0] =  dx;
						b_face[1] =  0;
						b_face[2] = -dz;

					    }else{	//-z ?

						//+x -y -z;  triangle points up to (dx, 0, dz)
						a_face[0] =  0;
						a_face[1] =  dy;
						a_face[2] = -dz;

						b_face[0] = -dx;
						b_face[1] =  0;
						b_face[2] = -dz;
					    }
					}
				    }else{	//-x
					if(face[2]) //+y
					{
					    if(face[4])	//+z
					    {
						//-x +y +z; triangle points down to (0, dy, 0)
						a_face[0] =  dx;
						a_face[1] =  0;
						a_face[2] =  dz;

						b_face[0] =  0;
						b_face[1] = -dy;
						b_face[2] =  dz;

					    }else{	//-z

						//-x +y -z;  triangle points up to (0, dy, dz)
						a_face[0] =  dx;
						a_face[1] =  dy;
						a_face[2] =  0;

						b_face[0] =  0;
						b_face[1] =  dy;
						b_face[2] =  dz;
					    }
					}else{	//-y
					    if(face[4])	//+z
					    {
						//-x -y +z; triangle points down to (0, 0, 0)
						a_face[0] =  0;
						a_face[1] =  dy;
						a_face[2] =  dz;

						b_face[0] =  dx;
						b_face[1] =  0;
						b_face[2] =  dz;

					    }else{	//-z

						//-x -y -z;  triangle points up to (0, 0, dz)
						a_face[0] =  0;
						a_face[1] = -dy;
						a_face[2] =  dz;

						b_face[0] =  dx;
						b_face[1] = -dy;
						b_face[2] =  0;
					    }
					}
				    }

				    //vector area of parallelogram
				    vector_cross_product_in_place(&face_vector_A[0],a_face,b_face,3);

				    //vector area of triangle needs to be scaled by 1/2
				    //for(int l=0;l<3;l++)
				//	face_vector_A[l] *= 0.5;
				    //however, we often miss about the same area in the corners
				}   
				
				A_face = vector_dot_product(face_vector_A,dir_norm,3);
				if(A_face>0)
				    dcase[3] += A_face;
				break;


			//if 4 faces contribute, only add their hypotenusal area
			case 4: fcase[4]++;


				//tunnels don't contribute
				if(face[0]&&face[1]&&face[2]&&face[3])
				    break;
				if(face[2]&&face[3]&&face[4]&&face[5])
				    break;
				if(face[4]&&face[5]&&face[0]&&face[1])
				    break;

				//add open face

				if( face[0]&&face[1] )
				{

				    //surface runs diagonal across the x face
				    if(face[2]&&face[4])
				    {
					//+y and +z
					a_face[0] =  dx;
					a_face[1] =  0;
					a_face[2] =  0;

					b_face[0] =  0;
					b_face[1] = -dy;
					b_face[2] =  dz;

				    }else if(face[2]&&face[5]){

					//+y and -z
					a_face[0] =  dx;
					a_face[1] =  0;
					a_face[2] =  0;

					b_face[0] =  0;
					b_face[1] =  dy;
					b_face[2] =  dz;

				    }else if(face[3]&&face[4]){

					//-y and +z
					a_face[0] = -dx;
					a_face[1] =  0;
					a_face[2] =  0;

					b_face[0] =  0;
					b_face[1] = -dy;
					b_face[2] = -dz;

				    }else{

					//-y and -z
					a_face[0] = -dx;
					a_face[1] =  0;
					a_face[2] =  0;

					b_face[0] =  0;
					b_face[1] = -dy;
					b_face[2] =  dz;
				    }

				}else{

				    if( face[2]&&face[3] )
				    {
					//surface runs diagonal across the y face


					if(face[0]&&face[4])
					{
					   //+x and +z
					    a_face[0] =  0;
					    a_face[1] =  dy;
					    a_face[2] =  0;

					    b_face[0] =  dx;
					    b_face[1] =  0;
					    b_face[2] = -dz;

					}else if(face[0]&&face[5]){

					   //+x and -z
					    a_face[0] =  0;
					    a_face[1] =  dy;
					    a_face[2] =  0;

					    b_face[0] = -dx; 
					    b_face[1] =  0;
					    b_face[2] = -dz;

					}else if(face[1]&&face[4]){

					   //-x and +z
					    a_face[0] =  0;
					    a_face[1] = -dy;
					    a_face[2] =  0;

					    b_face[0] = -dx;
					    b_face[1] =  0;
					    b_face[2] = -dz;

					}else{

					    //-x and -z
					    a_face[0] =  0;
					    a_face[1] = -dy;
					    a_face[2] =  0;

					    b_face[0] =  dx;
					    b_face[1] =  0;
					    b_face[2] = -dz;
					}

				    }else{
					//surface runs diagonal across the z face

					if(face[0]&&face[2])
					{
					   //+x and +y
					    a_face[0] =  0;
					    a_face[1] =  0;
					    a_face[2] =  dz;

					    b_face[0] = -dx;
					    b_face[1] =  dy;
					    b_face[2] =  0;

					}else if(face[0]&&face[3]){

					   //+x and -y
					    a_face[0] =  0;
					    a_face[1] =  0;
					    a_face[2] =  dz;

					    b_face[0] =  dx;
					    b_face[1] =  dy;
					    b_face[2] =  0;

					}else if(face[1]&&face[2]){

					   //-x and +y
					    a_face[0] =  0;
					    a_face[1] =  0;
					    a_face[2] = -dz;

					    b_face[0] =  dx;
					    b_face[1] =  dy;
					    b_face[2] =  0;

					}else{

					    //-x and -y
					    a_face[0] =  0;
					    a_face[1] =  0;
					    a_face[2] = -dz;

					    b_face[0] = -dx;
					    b_face[1] =  dy;
					    b_face[2] =  0;
					}

				    }
				}

				//find vector area of parallelogram
				vector_cross_product_in_place(&face_vector_A[0],a_face,b_face,3);

				A_face = vector_dot_product(face_vector_A,dir_norm,3);
				if(A_face>0)
				    dcase[4] += A_face;
				break;


			//if 5 contribute, only the open face is added
			case 5: fcase[5]++;

				if(!face[0])
				    face_vector_A[0] =  dy*dz;
				if(!face[1])
				    face_vector_A[0] = -dy*dz;

				if(!face[2])
				    face_vector_A[1] =  dx*dz;
				if(!face[3])
				    face_vector_A[1] = -dx*dz;

				if(!face[4])
				    face_vector_A[2] =  dx*dy;
				if(!face[5])
				    face_vector_A[2] = -dx*dy;

				A_face = vector_dot_product(face_vector_A,dir_norm,3);
				if(A_face>0)
				    dcase[5] += A_face;
				break;

			//if 0 or 6 faces, no area
			default:    break;

		    }

		    //Add the surface area
		    A_face = vector_dot_product(face_vector_A,dir_norm,3);

		    //if the dot product is positive, record the
		    //vector area
		    if(A_face>0)
			for(int l=0;l<ndim;l++)
			    varea[l][ijk] = face_vector_A[l];
		}
	}

    //do upper boundary second
    if(nx_local)
    for(int j=0;j<ny;j++)
	for(int k=0;k<nz;k++)
	{
		//get cell index
		ijk = grid_ijk(nx_local-1,j,k,grid_info);

		//consider only zero cells near the interface
		n_faces = 0;
		for(int l=0;l<6;l++)
		    face[l] = 0;

		//get the value of the vector field at this cell
		for(int l=0;l<ndim;l++)
		    dir_norm[l] = vfield[l][ijk];

		if(mask[ijk]==0)
		{

		    ijkt = nz*j + k;
		    if(ghost_cell_upper[ijkt]!=0)
		    {
			n_faces++;
			face[0] = 1;
		    }

		    //causes double counting for some splits

		    ijkt = grid_ijk(nx_local-2,j,k,grid_info);
		    if(mask[ijkt]!=0)
		    {
			n_faces++;
			face[1] = 1;
		    }

		    //ijkt = grid_ijk(nx_local-1,j-1,k,grid_info);
		    ijkt = grid_ijk(nx_local-1,j+1,k,grid_info);
		    if(mask[ijkt]!=0)
		    {
			n_faces++;
			face[2] = 1;
		    }

		    //ijkt = grid_ijk(nx_local-1,j+1,k,grid_info);
		    ijkt = grid_ijk(nx_local-1,j-1,k,grid_info);
		    if(mask[ijkt]!=0)
		    {
			n_faces++;
			face[3] = 1;
		    }

		    //ijkt = grid_ijk(nx_local-1,j,k-1,grid_info);
		    ijkt = grid_ijk(nx_local-1,j,k+1,grid_info);
		    if(mask[ijkt]!=0)
		    {
			n_faces++;
			face[4] = 1;
		    }

		    //ijkt = grid_ijk(nx_local-1,j,k+1,grid_info);
		    ijkt = grid_ijk(nx_local-1,j,k-1,grid_info);
		    if(mask[ijkt]!=0)
		    {
			n_faces++;
			face[5] = 1;
		    }

		    //surface area contribution of the face
		    //in question
		    A_face = 0;
		    for(int l=0;l<3;l++)
		    {
			face_vector_A[l] = 0;
			a_face[l] = 0;
			b_face[l] = 0;
		    }

		    //add area from this cell
		    switch(n_faces)
		    {


			//if no faces contribute, just break
			case 0:	fcase[0]++;
				break;

			//if 1 face contributes, add its area
			case 1:	fcase[1]++;

				//n_tested++;
				if(face[0])
				    face_vector_A[0] = -dy * dz;
				
				if(face[1])
				    face_vector_A[0] =  dy * dz;
				
				if(face[2])
				    face_vector_A[1] = -dz * dx;

				if(face[3])
				    face_vector_A[1] =  dz * dx;

				if(face[4])
				    face_vector_A[2] = -dx * dy;
				
				if(face[5])
				    face_vector_A[2] =  dx * dy;
				
				A_face = vector_dot_product(face_vector_A,dir_norm,3);
				if(A_face>0)
				    dcase[1] += A_face;
				break;


			//if 2 faces contribute and are adjacent, add their hypotenusal area
			case 2:	fcase[2]++;

				//opposing faces don't contribute
				if( (face[0]&&face[1])||(face[2]&&face[3])||(face[4]&&face[5]) )
				    break;

				//add the hypotenuse only if the dot product is positive
				if(face[0]) //+x face
				{
					if(face[2])
					{
						//+x/+y, checked 05/22
						a_face[2] =  dz;

						b_face[0] = -dx;
						b_face[1] =  dy;

					}else if(face[3]){

						//+x/-y, checked 05/22
						a_face[2] =  dz;

						b_face[0] =  dx;
						b_face[1] =  dy;
					}

					if(face[4])
					{
						//+x/+z,checked 05/22
						a_face[1] =  dy;

						b_face[0] =  dx;
						b_face[2] = -dz;

					}else if(face[5]){

						//+x/-z, checked 05/22
						a_face[1] =  dy;

						b_face[0] =  -dx;
						b_face[2] =  -dz;
					}

				}else if(face[1]){ //-x face

					if(face[2])
					{
						//-x/+y, checked 05/22
						a_face[2] =  dz;

						b_face[0] = -dx;
						b_face[1] = -dy;

					}else if(face[3]){

						//-x/-y, checked 05/22
						a_face[2] =  dz;

						b_face[0] =  dx;
						b_face[1] = -dy;

					}

					if(face[4])
					{
						//-x/+z, checked 05/22
						a_face[1] =  dy;

						b_face[0] =  dx;
						b_face[2] =  dz;

					}else if(face[5]){

						//-x/-z, checked 05/22
						a_face[1] =  dy;

						b_face[0] = -dx;
						b_face[2] =  dz;
					}

				}else if(face[2]){  //+y face, no x-face

					//+y face 
					if(face[4])
					{
						//+y/+z, checked 05/22
						a_face[0] =  dx;

						b_face[1] = -dy;
						b_face[2] =  dz;

					}else if(face[5]){

						//+y/-z, checked 05/22
						a_face[0] =  dx;

						b_face[1] =  dy;
						b_face[2] =  dz;
					}

				}else if(face[3]){ //-y face, no x-face

					//-y face
					if(face[4])
					{
						//-y/+z, checked 05/22
						a_face[0] =  dx;

						b_face[1] = -dy;
						b_face[2] = -dz;

					}else if(face[5]){

						//-y/-z, checked 05/22
						a_face[0] =  dx;

						b_face[1] =  dy;
						b_face[2] = -dz;
					}

				}
				vector_cross_product_in_place(&face_vector_A[0],a_face,b_face,3);
		
				A_face = vector_dot_product(face_vector_A,dir_norm,3);
				if(A_face>0)
				    dcase[2] += A_face;
				break;

			//if 3 faces contribute, then treat adjacent and opposing faces separately
			case 3: fcase[3]++;

				//opposing faces contribute only the non-opposing face
				if( (face[0]&&face[1])||(face[2]&&face[3])||(face[4]&&face[5]) )
				{

				    if( face[0]&&face[1] )
				    {
					//y faces
					if(face[2])
					    face_vector_A[1] = -dz*dx;
					if(face[3])
					    face_vector_A[1] =  dz*dx;
	    
					//z faces
					if(face[4])
					    face_vector_A[2] = -dx*dy;
					if(face[5])
					    face_vector_A[2] =  dx*dy;
				    }

				    if( face[2]&&face[3] )
				    {
					//x faces
					if(face[0])
					    face_vector_A[0] = -dy*dz;
					if(face[1])
					    face_vector_A[0] =  dy*dz;

					//z faces
					if(face[4])
					    face_vector_A[2] = -dx*dy;
					if(face[5])
					    face_vector_A[2] =  dx*dy;
				    }

				    if( face[4]&&face[5] )
				    {
					//x faces
					if(face[0])
					    face_vector_A[0] = -dy*dz;
					if(face[1])
					    face_vector_A[0] =  dy*dz;

					//y faces
					if(face[2])
					    face_vector_A[1] = -dz*dx;
					if(face[3])
					    face_vector_A[1] =  dz*dx;
				    }

				}else{


				    //adjacent faces supply chamfer of triangle of hypotenuses
				    //a_face will lie in the inner most conditioned face

				    //there must be an x face
				    if(face[0])	//+x
				    {
					if(face[2]) //+y
					{
					    if(face[4])	//+z
					    {
						//+x +y +z; triangle points down to (dx, dy, 0)
						a_face[0] =  0;
						a_face[1] = -dy;
						a_face[2] =  dz;

						b_face[0] = -dx;
						b_face[1] =  0;
						b_face[2] =  dz;

					    }else{	//-z

						//+x +y -z;  triangle points up to (dx, dy, dz)
						a_face[0] =  dx;
						a_face[1] = -dy;
						a_face[2] =  0;

						b_face[0] =  dx;
						b_face[1] =  0;
						b_face[2] =  dz;
					    }
					}else{	//-y
					    if(face[4])	//+z
					    {
						//+x -y +z; triangle points down to (dx, 0, 0)
						a_face[0] =  dx;
						a_face[1] =  dy;
						a_face[2] =  0;

						b_face[0] =  dx;
						b_face[1] =  0;
						b_face[2] = -dz;

					    }else{	//-z ?

						//+x -y -z;  triangle points up to (dx, 0, dz)
						a_face[0] =  0;
						a_face[1] =  dy;
						a_face[2] = -dz;

						b_face[0] = -dx;
						b_face[1] =  0;
						b_face[2] = -dz;
					    }
					}
				    }else{	//-x
					if(face[2]) //+y
					{
					    if(face[4])	//+z
					    {
						//-x +y +z; triangle points down to (0, dy, 0)
						a_face[0] =  dx;
						a_face[1] =  0;
						a_face[2] =  dz;

						b_face[0] =  0;
						b_face[1] = -dy;
						b_face[2] =  dz;

					    }else{	//-z

						//-x +y -z;  triangle points up to (0, dy, dz)
						a_face[0] =  dx;
						a_face[1] =  dy;
						a_face[2] =  0;

						b_face[0] =  0;
						b_face[1] =  dy;
						b_face[2] =  dz;
					    }
					}else{	//-y
					    if(face[4])	//+z
					    {
						//-x -y +z; triangle points down to (0, 0, 0)
						a_face[0] =  0;
						a_face[1] =  dy;
						a_face[2] =  dz;

						b_face[0] =  dx;
						b_face[1] =  0;
						b_face[2] =  dz;

					    }else{	//-z

						//-x -y -z;  triangle points up to (0, 0, dz)
						a_face[0] =  0;
						a_face[1] = -dy;
						a_face[2] =  dz;

						b_face[0] =  dx;
						b_face[1] = -dy;
						b_face[2] =  0;
					    }
					}
				    }

				    //vector area of parallelogram
				    vector_cross_product_in_place(&face_vector_A[0],a_face,b_face,3);

				    //vector area of triangle needs to be scaled by 1/2
				    //for(int l=0;l<3;l++)
				//	face_vector_A[l] *= 0.5;
				    //however, we often miss about the same area in the corners
				}   
				
				A_face = vector_dot_product(face_vector_A,dir_norm,3);
				if(A_face>0)
				    dcase[3] += A_face;
				break;


			//if 4 faces contribute, only add their hypotenusal area
			case 4: fcase[4]++;


				//tunnels don't contribute
				if(face[0]&&face[1]&&face[2]&&face[3])
				    break;
				if(face[2]&&face[3]&&face[4]&&face[5])
				    break;
				if(face[4]&&face[5]&&face[0]&&face[1])
				    break;

				//add open face

				if( face[0]&&face[1] )
				{

				    //surface runs diagonal across the x face
				    if(face[2]&&face[4])
				    {
					//+y and +z
					a_face[0] =  dx;
					a_face[1] =  0;
					a_face[2] =  0;

					b_face[0] =  0;
					b_face[1] = -dy;
					b_face[2] =  dz;

				    }else if(face[2]&&face[5]){

					//+y and -z
					a_face[0] =  dx;
					a_face[1] =  0;
					a_face[2] =  0;

					b_face[0] =  0;
					b_face[1] =  dy;
					b_face[2] =  dz;

				    }else if(face[3]&&face[4]){

					//-y and +z
					a_face[0] = -dx;
					a_face[1] =  0;
					a_face[2] =  0;

					b_face[0] =  0;
					b_face[1] = -dy;
					b_face[2] = -dz;

				    }else{

					//-y and -z
					a_face[0] = -dx;
					a_face[1] =  0;
					a_face[2] =  0;

					b_face[0] =  0;
					b_face[1] = -dy;
					b_face[2] =  dz;
				    }

				}else{

				    if( face[2]&&face[3] )
				    {
					//surface runs diagonal across the y face


					if(face[0]&&face[4])
					{
					   //+x and +z
					    a_face[0] =  0;
					    a_face[1] =  dy;
					    a_face[2] =  0;

					    b_face[0] =  dx;
					    b_face[1] =  0;
					    b_face[2] = -dz;

					}else if(face[0]&&face[5]){

					   //+x and -z
					    a_face[0] =  0;
					    a_face[1] =  dy;
					    a_face[2] =  0;

					    b_face[0] = -dx; 
					    b_face[1] =  0;
					    b_face[2] = -dz;

					}else if(face[1]&&face[4]){

					   //-x and +z
					    a_face[0] =  0;
					    a_face[1] = -dy;
					    a_face[2] =  0;

					    b_face[0] = -dx;
					    b_face[1] =  0;
					    b_face[2] = -dz;

					}else{

					    //-x and -z
					    a_face[0] =  0;
					    a_face[1] = -dy;
					    a_face[2] =  0;

					    b_face[0] =  dx;
					    b_face[1] =  0;
					    b_face[2] = -dz;
					}

				    }else{
					//surface runs diagonal across the z face

					if(face[0]&&face[2])
					{
					   //+x and +y
					    a_face[0] =  0;
					    a_face[1] =  0;
					    a_face[2] =  dz;

					    b_face[0] = -dx;
					    b_face[1] =  dy;
					    b_face[2] =  0;

					}else if(face[0]&&face[3]){

					   //+x and -y
					    a_face[0] =  0;
					    a_face[1] =  0;
					    a_face[2] =  dz;

					    b_face[0] =  dx;
					    b_face[1] =  dy;
					    b_face[2] =  0;

					}else if(face[1]&&face[2]){

					   //-x and +y
					    a_face[0] =  0;
					    a_face[1] =  0;
					    a_face[2] = -dz;

					    b_face[0] =  dx;
					    b_face[1] =  dy;
					    b_face[2] =  0;

					}else{

					    //-x and -y
					    a_face[0] =  0;
					    a_face[1] =  0;
					    a_face[2] = -dz;

					    b_face[0] = -dx;
					    b_face[1] =  dy;
					    b_face[2] =  0;
					}

				    }
				}

				//find vector area of parallelogram
				vector_cross_product_in_place(&face_vector_A[0],a_face,b_face,3);

				A_face = vector_dot_product(face_vector_A,dir_norm,3);
				if(A_face>0)
				    dcase[4] += A_face;
				break;


			//if 5 contribute, only the open face is added
			case 5: fcase[5]++;

				if(!face[0])
				    face_vector_A[0] =  dy*dz;
				if(!face[1])
				    face_vector_A[0] = -dy*dz;

				if(!face[2])
				    face_vector_A[1] =  dx*dz;
				if(!face[3])
				    face_vector_A[1] = -dx*dz;

				if(!face[4])
				    face_vector_A[2] =  dx*dy;
				if(!face[5])
				    face_vector_A[2] = -dx*dy;

				A_face = vector_dot_product(face_vector_A,dir_norm,3);
				if(A_face>0)
				    dcase[5] += A_face;
				break;

			//if 0 or 6 faces, no area
			default:    break;
		    }

		    //Add the surface area
		    A_face = vector_dot_product(face_vector_A,dir_norm,3);

		    //if the dot product is positive, record the
		    //vector area
		    if(A_face>0)
			for(int l=0;l<ndim;l++)
			    varea[l][ijk] = face_vector_A[l];
		}
	}

    //free ghost cells
    free(ghost_cell_upper);
    free(ghost_cell_lower);


    //sum area across processors
    MPI_Allreduce(&A_local,&A_total,1,MPI_DOUBLE,MPI_SUM,world);


    //return the vector surface area of each boundary cell
    return varea;

}

/*! \fn double *grid_surface_area_mask(double *mask, double **vfield, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world)
 *  \brief Define a mask at the surface boundary, for regions with positive dot product with the vfield */
double *grid_surface_area_mask(double *mask, double **vfield, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world)
{
    int ijk;	//cell
    int ijkt;	//adjacent cells
    int nx_local = grid_info.nx_local;
    int nx_local_start = grid_info.nx_local_start;
    int ny       = grid_info.ny;
    int nz       = grid_info.nz;

    double dx	= grid_info.dx;
    double dy	= grid_info.dy;
    double dz	= grid_info.dz;

    int ndim = grid_info.ndim;

    int dest;	//destination processor for communication
    int source;	//source processor for communication

    double *marea; //mask for the surface area

    MPI_Status status;	//MPI status for communication


    double *ghost_cell_upper;	//ghost cells at i = nx_local
    double *ghost_cell_lower;	//ghost cells at i = -1

    //local surface area
    double A_local = 0;
    
    //total surface area
    double A_total = 0;

    //number of faces contributing to surface
    int n_faces;
    int face[6];

    //perimeter
    double s;

    //triangle sides
    double a, b, c;

    int n_tested = 0;

    double face_vector_A[3];  //vector area of the face
    double a_face[3];	      //first vector of the face
    double b_face[3];	      //second vector of the face

    double dir_norm[3];	      //value of the vector field at this cell

    int fcase[6] = {0,0,0,0,0,0};
    double dcase[6] = {0,0,0,0,0,0};

    double A_face;

    //allocate the surface area mask
    marea = allocate_real_fftw_grid(grid_info);

    //we have to check boundary cells separately
    //so start with inner cells
    for(int i=1;i<nx_local-1;i++)
	for(int j=0;j<ny;j++)
	    for(int k=0;k<nz;k++)
	    {

		//get cell index
		ijk = grid_ijk(i,j,k,grid_info);

		//consider only zero cells near the interface
		n_faces = 0;
		for(int l=0;l<6;l++)
		    face[l] = 0;

		//get the value of the vector field at this cell
		for(int l=0;l<ndim;l++)
		    dir_norm[l] = vfield[l][ijk];

		if(mask[ijk]==0)
		{

		    ijkt = grid_ijk(i+1,j,k,grid_info);
		    if(mask[ijkt]!=0)
		    {
			n_faces++;
			face[0] = 1;
		    }

		    ijkt = grid_ijk(i-1,j,k,grid_info);
		    if(mask[ijkt]!=0)
		    {
			n_faces++;
			face[1] = 1;
		    }


		    ijkt = grid_ijk(i,j+1,k,grid_info);
		    if(mask[ijkt]!=0)
		    {
			n_faces++;
			face[2] = 1;
		    }

		    ijkt = grid_ijk(i,j-1,k,grid_info);
		    if(mask[ijkt]!=0)
		    {
			n_faces++;
			face[3] = 1;
		    }

		    ijkt = grid_ijk(i,j,k+1,grid_info);
		    if(mask[ijkt]!=0)
		    {
			n_faces++;
			face[4] = 1;
		    }

		    ijkt = grid_ijk(i,j,k-1,grid_info);
		    if(mask[ijkt]!=0)
		    {
			n_faces++;
			face[5] = 1;
		    }


		    //surface area contribution of the face
		    //in question
		    A_face = 0;
		    for(int l=0;l<3;l++)
		    {
			face_vector_A[l] = 0;
			a_face[l] = 0;
			b_face[l] = 0;
		    }

		    //add area from this cell
		    switch(n_faces)
		    {

			//if no faces contribute, just break
			case 0:	fcase[0]++;
				break;

			//if 1 face contributes, add its area
			case 1:	

				//n_tested++;
				if(face[0])
				    face_vector_A[0] = -dy * dz;
				
				if(face[1])
				    face_vector_A[0] =  dy * dz;
				
				if(face[2])
				    face_vector_A[1] = -dz * dx;

				if(face[3])
				    face_vector_A[1] =  dz * dx;

				if(face[4])
				    face_vector_A[2] = -dx * dy;
				
				if(face[5])
				    face_vector_A[2] =  dx * dy;
				
				A_face = vector_dot_product(face_vector_A,dir_norm,3);
				if(A_face>0)
				{
				    fcase[1]++;
				    dcase[1] += A_face;
				}
				break;


			//if 2 faces contribute and are adjacent, add their hypotenusal area
			case 2:	fcase[2]++;

				//opposing faces don't contribute
				if( (face[0]&&face[1])||(face[2]&&face[3])||(face[4]&&face[5]) )
				    break;

				//add the hypotenuse only if the dot product is positive
				if(face[0]) //+x face
				{
					if(face[2])
					{
						//+x/+y, checked 05/22
						a_face[2] =  dz;

						b_face[0] = -dx;
						b_face[1] =  dy;

					}else if(face[3]){

						//+x/-y, checked 05/22
						a_face[2] =  dz;

						b_face[0] =  dx;
						b_face[1] =  dy;
					}

					if(face[4])
					{
						//+x/+z,checked 05/22
						a_face[1] =  dy;

						b_face[0] =  dx;
						b_face[2] = -dz;

					}else if(face[5]){

						//+x/-z, checked 05/22
						a_face[1] =  dy;

						b_face[0] =  -dx;
						b_face[2] =  -dz;
					}

				}else if(face[1]){ //-x face

					if(face[2])
					{
						//-x/+y, checked 05/22
						a_face[2] =  dz;

						b_face[0] = -dx;
						b_face[1] = -dy;

					}else if(face[3]){

						//-x/-y, checked 05/22
						a_face[2] =  dz;

						b_face[0] =  dx;
						b_face[1] = -dy;

					}

					if(face[4])
					{
						//-x/+z, checked 05/22
						a_face[1] =  dy;

						b_face[0] =  dx;
						b_face[2] =  dz;

					}else if(face[5]){

						//-x/-z, checked 05/22
						a_face[1] =  dy;

						b_face[0] = -dx;
						b_face[2] =  dz;
					}

				}else if(face[2]){  //+y face, no x-face

					//+y face 
					if(face[4])
					{
						//+y/+z, checked 05/22
						a_face[0] =  dx;

						b_face[1] = -dy;
						b_face[2] =  dz;

					}else if(face[5]){

						//+y/-z, checked 05/22
						a_face[0] =  dx;

						b_face[1] =  dy;
						b_face[2] =  dz;
					}

				}else if(face[3]){ //-y face, no x-face

					//-y face
					if(face[4])
					{
						//-y/+z, checked 05/22
						a_face[0] =  dx;

						b_face[1] = -dy;
						b_face[2] = -dz;

					}else if(face[5]){

						//-y/-z, checked 05/22
						a_face[0] =  dx;

						b_face[1] =  dy;
						b_face[2] = -dz;
					}

				}
				vector_cross_product_in_place(&face_vector_A[0],a_face,b_face,3);
		
				A_face = vector_dot_product(face_vector_A,dir_norm,3);
				if(A_face>0)
				    dcase[2] += A_face;
				break;

			//if 3 faces contribute, then treat adjacent and opposing faces separately
			case 3: fcase[3]++;

				//opposing faces contribute only the non-opposing face
				if( (face[0]&&face[1])||(face[2]&&face[3])||(face[4]&&face[5]) )
				{

				    if( face[0]&&face[1] )
				    {
					//y faces
					if(face[2])
					    face_vector_A[1] = -dz*dx;
					if(face[3])
					    face_vector_A[1] =  dz*dx;
	    
					//z faces
					if(face[4])
					    face_vector_A[2] = -dx*dy;
					if(face[5])
					    face_vector_A[2] =  dx*dy;
				    }

				    if( face[2]&&face[3] )
				    {
					//x faces
					if(face[0])
					    face_vector_A[0] = -dy*dz;
					if(face[1])
					    face_vector_A[0] =  dy*dz;

					//z faces
					if(face[4])
					    face_vector_A[2] = -dx*dy;
					if(face[5])
					    face_vector_A[2] =  dx*dy;
				    }

				    if( face[4]&&face[5] )
				    {
					//x faces
					if(face[0])
					    face_vector_A[0] = -dy*dz;
					if(face[1])
					    face_vector_A[0] =  dy*dz;

					//y faces
					if(face[2])
					    face_vector_A[1] = -dz*dx;
					if(face[3])
					    face_vector_A[1] =  dz*dx;
				    }

				}else{


				    //adjacent faces supply chamfer of triangle of hypotenuses
				    //a_face will lie in the inner most conditioned face

				    //there must be an x face
				    if(face[0])	//+x
				    {
					if(face[2]) //+y
					{
					    if(face[4])	//+z
					    {
						//+x +y +z; triangle points down to (dx, dy, 0)
						a_face[0] =  0;
						a_face[1] = -dy;
						a_face[2] =  dz;

						b_face[0] = -dx;
						b_face[1] =  0;
						b_face[2] =  dz;

					    }else{	//-z

						//+x +y -z;  triangle points up to (dx, dy, dz)
						a_face[0] =  dx;
						a_face[1] = -dy;
						a_face[2] =  0;

						b_face[0] =  dx;
						b_face[1] =  0;
						b_face[2] =  dz;
					    }
					}else{	//-y
					    if(face[4])	//+z
					    {
						//+x -y +z; triangle points down to (dx, 0, 0)
						a_face[0] =  dx;
						a_face[1] =  dy;
						a_face[2] =  0;

						b_face[0] =  dx;
						b_face[1] =  0;
						b_face[2] = -dz;

					    }else{	//-z ?

						//+x -y -z;  triangle points up to (dx, 0, dz)
						a_face[0] =  0;
						a_face[1] =  dy;
						a_face[2] = -dz;

						b_face[0] = -dx;
						b_face[1] =  0;
						b_face[2] = -dz;
					    }
					}
				    }else{	//-x
					if(face[2]) //+y
					{
					    if(face[4])	//+z
					    {
						//-x +y +z; triangle points down to (0, dy, 0)
						a_face[0] =  dx;
						a_face[1] =  0;
						a_face[2] =  dz;

						b_face[0] =  0;
						b_face[1] = -dy;
						b_face[2] =  dz;

					    }else{	//-z

						//-x +y -z;  triangle points up to (0, dy, dz)
						a_face[0] =  dx;
						a_face[1] =  dy;
						a_face[2] =  0;

						b_face[0] =  0;
						b_face[1] =  dy;
						b_face[2] =  dz;
					    }
					}else{	//-y
					    if(face[4])	//+z
					    {
						//-x -y +z; triangle points down to (0, 0, 0)
						a_face[0] =  0;
						a_face[1] =  dy;
						a_face[2] =  dz;

						b_face[0] =  dx;
						b_face[1] =  0;
						b_face[2] =  dz;

					    }else{	//-z

						//-x -y -z;  triangle points up to (0, 0, dz)
						a_face[0] =  0;
						a_face[1] = -dy;
						a_face[2] =  dz;

						b_face[0] =  dx;
						b_face[1] = -dy;
						b_face[2] =  0;
					    }
					}
				    }

				    //vector area of parallelogram
				    vector_cross_product_in_place(&face_vector_A[0],a_face,b_face,3);

				    //vector area of triangle needs to be scaled by 1/2
				    //for(int l=0;l<3;l++)
				//	face_vector_A[l] *= 0.5;
				    //however, we often miss about the same area in the corners
				}   
				
				A_face = vector_dot_product(face_vector_A,dir_norm,3);
				if(A_face>0)
				    dcase[3] += A_face;
				break;


			//if 4 faces contribute, only add their hypotenusal area
			case 4: fcase[4]++;


				//tunnels don't contribute
				if(face[0]&&face[1]&&face[2]&&face[3])
				    break;
				if(face[2]&&face[3]&&face[4]&&face[5])
				    break;
				if(face[4]&&face[5]&&face[0]&&face[1])
				    break;

				//add open face

				if( face[0]&&face[1] )
				{

				    //surface runs diagonal across the x face
				    if(face[2]&&face[4])
				    {
					//+y and +z
					a_face[0] =  dx;
					a_face[1] =  0;
					a_face[2] =  0;

					b_face[0] =  0;
					b_face[1] = -dy;
					b_face[2] =  dz;

				    }else if(face[2]&&face[5]){

					//+y and -z
					a_face[0] =  dx;
					a_face[1] =  0;
					a_face[2] =  0;

					b_face[0] =  0;
					b_face[1] =  dy;
					b_face[2] =  dz;

				    }else if(face[3]&&face[4]){

					//-y and +z
					a_face[0] = -dx;
					a_face[1] =  0;
					a_face[2] =  0;

					b_face[0] =  0;
					b_face[1] = -dy;
					b_face[2] = -dz;

				    }else{

					//-y and -z
					a_face[0] = -dx;
					a_face[1] =  0;
					a_face[2] =  0;

					b_face[0] =  0;
					b_face[1] = -dy;
					b_face[2] =  dz;
				    }

				}else{

				    if( face[2]&&face[3] )
				    {
					//surface runs diagonal across the y face


					if(face[0]&&face[4])
					{
					   //+x and +z
					    a_face[0] =  0;
					    a_face[1] =  dy;
					    a_face[2] =  0;

					    b_face[0] =  dx;
					    b_face[1] =  0;
					    b_face[2] = -dz;

					}else if(face[0]&&face[5]){

					   //+x and -z
					    a_face[0] =  0;
					    a_face[1] =  dy;
					    a_face[2] =  0;

					    b_face[0] = -dx; 
					    b_face[1] =  0;
					    b_face[2] = -dz;

					}else if(face[1]&&face[4]){

					   //-x and +z
					    a_face[0] =  0;
					    a_face[1] = -dy;
					    a_face[2] =  0;

					    b_face[0] = -dx;
					    b_face[1] =  0;
					    b_face[2] = -dz;

					}else{

					    //-x and -z
					    a_face[0] =  0;
					    a_face[1] = -dy;
					    a_face[2] =  0;

					    b_face[0] =  dx;
					    b_face[1] =  0;
					    b_face[2] = -dz;
					}

				    }else{
					//surface runs diagonal across the z face

					if(face[0]&&face[2])
					{
					   //+x and +y
					    a_face[0] =  0;
					    a_face[1] =  0;
					    a_face[2] =  dz;

					    b_face[0] = -dx;
					    b_face[1] =  dy;
					    b_face[2] =  0;

					}else if(face[0]&&face[3]){

					   //+x and -y
					    a_face[0] =  0;
					    a_face[1] =  0;
					    a_face[2] =  dz;

					    b_face[0] =  dx;
					    b_face[1] =  dy;
					    b_face[2] =  0;

					}else if(face[1]&&face[2]){

					   //-x and +y
					    a_face[0] =  0;
					    a_face[1] =  0;
					    a_face[2] = -dz;

					    b_face[0] =  dx;
					    b_face[1] =  dy;
					    b_face[2] =  0;

					}else{

					    //-x and -y
					    a_face[0] =  0;
					    a_face[1] =  0;
					    a_face[2] = -dz;

					    b_face[0] = -dx;
					    b_face[1] =  dy;
					    b_face[2] =  0;
					}

				    }
				}

				//find vector area of parallelogram
				vector_cross_product_in_place(&face_vector_A[0],a_face,b_face,3);

				A_face = vector_dot_product(face_vector_A,dir_norm,3);
				if(A_face>0)
				    dcase[4] += A_face;
				break;


			//if 5 contribute, only the open face is added
			case 5: fcase[5]++;

				if(!face[0])
				    face_vector_A[0] =  dy*dz;
				if(!face[1])
				    face_vector_A[0] = -dy*dz;

				if(!face[2])
				    face_vector_A[1] =  dx*dz;
				if(!face[3])
				    face_vector_A[1] = -dx*dz;

				if(!face[4])
				    face_vector_A[2] =  dx*dy;
				if(!face[5])
				    face_vector_A[2] = -dx*dy;

				A_face = vector_dot_product(face_vector_A,dir_norm,3);
				if(A_face>0)
				    dcase[5] += A_face;
				break;

			//if 0 or 6 faces, no area
			default:    break;
		    }

		    //find the dot product of the local cell vector area and the
		    //local velocity field direction
		    A_face = vector_dot_product(face_vector_A,dir_norm,3);

		    //if the dot product is positive, record the
		    //vector area
		    if(A_face>0)
			marea[ijk] = 1.0;
		}
	    }


    //now we have to check boundary cells
    //to do this, we use a sendreceive to get adjacent
    //cells across the boundary, then repeat the checks

    //allocate ghost cells
    ghost_cell_upper = calloc_double_array(ny*nz);
    ghost_cell_lower = calloc_double_array(ny*nz); 

    //populate ghost cells
    for(int j=0;j<ny;j++)
	for(int k=0;k<nz;k++)
	{


	    //note nx_local can be zero!
	    if(nx_local)
	    {
		//upper cells to pass from this processor's lower boundary
		ijk = grid_ijk(0,j,k,grid_info);
		ghost_cell_upper[nz*j+k] = mask[ijk];

		//lower cells to pass from this processor's upper boundary
		ijk = grid_ijk(nx_local-1,j,k,grid_info);
		ghost_cell_lower[nz*j+k] = mask[ijk];


		//if(nx_local_start+nx_local-1==47)
		//	printf("j %d k %d ghost %e\n",j,k,ghost_cell_lower[nz*j+k]);
	    }else{
		//dummy cells
		ghost_cell_upper[nz*j+k] = 0;
		ghost_cell_lower[nz*j+k] = 0;
	    }
	}

    //sendrecv upper ghost cells
    source = myid+1;
    if(source>numprocs-1)
	source-=numprocs;
    dest = myid-1;
    if(dest<0)
	dest+=numprocs;

    MPI_Sendrecv_replace(ghost_cell_upper, ny*nz, MPI_DOUBLE, dest, myid, source, source, world, &status);
	

    //sendrecv lower ghost cells
    source = myid-1;
    if(source<0)
	source+=numprocs;
    dest = myid+1;
    if(dest>numprocs-1)
	dest-=numprocs;

    MPI_Sendrecv_replace(ghost_cell_lower, ny*nz, MPI_DOUBLE, dest, myid, source, source, world, &status);


    //do lower boundary first
    if(nx_local)
    for(int j=0;j<ny;j++)
	for(int k=0;k<nz;k++)
	{
		//get cell index
		ijk = grid_ijk(0,j,k,grid_info);

		//consider only zero cells near the interface
		n_faces = 0;
		for(int l=0;l<6;l++)
		    face[l] = 0;

		//get the value of the vector field at this cell
		for(int l=0;l<ndim;l++)
		    dir_norm[l] = vfield[l][ijk];

		if(mask[ijk]==0)
		{
		    ijkt = grid_ijk(1,j,k,grid_info);
		    if(mask[ijkt]!=0)
		    {
			n_faces++;
			face[0] = 1;
		    }

		    ijkt = nz*j + k;
		    if(ghost_cell_lower[ijkt]!=0)
		    {
			n_faces++;
			face[1] = 1;
		    }


		    ijkt = grid_ijk(0,j+1,k,grid_info);
		    if(mask[ijkt]!=0)
		    {
			n_faces++;
			face[2] = 1;
		    }

		    ijkt = grid_ijk(0,j-1,k,grid_info);
		    if(mask[ijkt]!=0)
		    {
			n_faces++;
			face[3] = 1;
		    }

		    ijkt = grid_ijk(0,j,k+1,grid_info);
		    if(mask[ijkt]!=0)
		    {
			n_faces++;
			face[4] = 1;
		    }

		    ijkt = grid_ijk(0,j,k-1,grid_info);
		    if(mask[ijkt]!=0)
		    {
			n_faces++;
			face[5] = 1;
		    }

		    //surface area contribution of the face
		    //in question
		    A_face = 0;
		    for(int l=0;l<3;l++)
		    {
			face_vector_A[l] = 0;
			a_face[l] = 0;
			b_face[l] = 0;
		    }

		    //add area from this cell
		    switch(n_faces)
		    {
			//if no faces contribute, just break
			case 0:	fcase[0]++;
				break;

			//if 1 face contributes, add its area
			case 1:	fcase[1]++;

				//n_tested++;
				if(face[0])
				    face_vector_A[0] = -dy * dz;
				
				if(face[1])
				    face_vector_A[0] =  dy * dz;
				
				if(face[2])
				    face_vector_A[1] = -dz * dx;

				if(face[3])
				    face_vector_A[1] =  dz * dx;

				if(face[4])
				    face_vector_A[2] = -dx * dy;
				
				if(face[5])
				    face_vector_A[2] =  dx * dy;
				
				A_face = vector_dot_product(face_vector_A,dir_norm,3);
				if(A_face>0)
				    dcase[1] += A_face;
				break;


			//if 2 faces contribute and are adjacent, add their hypotenusal area
			case 2:	fcase[2]++;

				//opposing faces don't contribute
				if( (face[0]&&face[1])||(face[2]&&face[3])||(face[4]&&face[5]) )
				    break;

				//add the hypotenuse only if the dot product is positive
				if(face[0]) //+x face
				{
					if(face[2])
					{
						//+x/+y, checked 05/22
						a_face[2] =  dz;

						b_face[0] = -dx;
						b_face[1] =  dy;

					}else if(face[3]){

						//+x/-y, checked 05/22
						a_face[2] =  dz;

						b_face[0] =  dx;
						b_face[1] =  dy;
					}

					if(face[4])
					{
						//+x/+z,checked 05/22
						a_face[1] =  dy;

						b_face[0] =  dx;
						b_face[2] = -dz;

					}else if(face[5]){

						//+x/-z, checked 05/22
						a_face[1] =  dy;

						b_face[0] =  -dx;
						b_face[2] =  -dz;
					}

				}else if(face[1]){ //-x face

					if(face[2])
					{
						//-x/+y, checked 05/22
						a_face[2] =  dz;

						b_face[0] = -dx;
						b_face[1] = -dy;

					}else if(face[3]){

						//-x/-y, checked 05/22
						a_face[2] =  dz;

						b_face[0] =  dx;
						b_face[1] = -dy;

					}

					if(face[4])
					{
						//-x/+z, checked 05/22
						a_face[1] =  dy;

						b_face[0] =  dx;
						b_face[2] =  dz;

					}else if(face[5]){

						//-x/-z, checked 05/22
						a_face[1] =  dy;

						b_face[0] = -dx;
						b_face[2] =  dz;
					}

				}else if(face[2]){  //+y face, no x-face

					//+y face 
					if(face[4])
					{
						//+y/+z, checked 05/22
						a_face[0] =  dx;

						b_face[1] = -dy;
						b_face[2] =  dz;

					}else if(face[5]){

						//+y/-z, checked 05/22
						a_face[0] =  dx;

						b_face[1] =  dy;
						b_face[2] =  dz;
					}

				}else if(face[3]){ //-y face, no x-face

					//-y face
					if(face[4])
					{
						//-y/+z, checked 05/22
						a_face[0] =  dx;

						b_face[1] = -dy;
						b_face[2] = -dz;

					}else if(face[5]){

						//-y/-z, checked 05/22
						a_face[0] =  dx;

						b_face[1] =  dy;
						b_face[2] = -dz;
					}

				}
				vector_cross_product_in_place(&face_vector_A[0],a_face,b_face,3);
		
				A_face = vector_dot_product(face_vector_A,dir_norm,3);
				if(A_face>0)
				    dcase[2] += A_face;
				break;

			//if 3 faces contribute, then treat adjacent and opposing faces separately
			case 3: fcase[3]++;

				//opposing faces contribute only the non-opposing face
				if( (face[0]&&face[1])||(face[2]&&face[3])||(face[4]&&face[5]) )
				{

				    if( face[0]&&face[1] )
				    {
					//y faces
					if(face[2])
					    face_vector_A[1] = -dz*dx;
					if(face[3])
					    face_vector_A[1] =  dz*dx;
	    
					//z faces
					if(face[4])
					    face_vector_A[2] = -dx*dy;
					if(face[5])
					    face_vector_A[2] =  dx*dy;
				    }

				    if( face[2]&&face[3] )
				    {
					//x faces
					if(face[0])
					    face_vector_A[0] = -dy*dz;
					if(face[1])
					    face_vector_A[0] =  dy*dz;

					//z faces
					if(face[4])
					    face_vector_A[2] = -dx*dy;
					if(face[5])
					    face_vector_A[2] =  dx*dy;
				    }

				    if( face[4]&&face[5] )
				    {
					//x faces
					if(face[0])
					    face_vector_A[0] = -dy*dz;
					if(face[1])
					    face_vector_A[0] =  dy*dz;

					//y faces
					if(face[2])
					    face_vector_A[1] = -dz*dx;
					if(face[3])
					    face_vector_A[1] =  dz*dx;
				    }

				}else{


				    //adjacent faces supply chamfer of triangle of hypotenuses
				    //a_face will lie in the inner most conditioned face

				    //there must be an x face
				    if(face[0])	//+x
				    {
					if(face[2]) //+y
					{
					    if(face[4])	//+z
					    {
						//+x +y +z; triangle points down to (dx, dy, 0)
						a_face[0] =  0;
						a_face[1] = -dy;
						a_face[2] =  dz;

						b_face[0] = -dx;
						b_face[1] =  0;
						b_face[2] =  dz;

					    }else{	//-z

						//+x +y -z;  triangle points up to (dx, dy, dz)
						a_face[0] =  dx;
						a_face[1] = -dy;
						a_face[2] =  0;

						b_face[0] =  dx;
						b_face[1] =  0;
						b_face[2] =  dz;
					    }
					}else{	//-y
					    if(face[4])	//+z
					    {
						//+x -y +z; triangle points down to (dx, 0, 0)
						a_face[0] =  dx;
						a_face[1] =  dy;
						a_face[2] =  0;

						b_face[0] =  dx;
						b_face[1] =  0;
						b_face[2] = -dz;

					    }else{	//-z ?

						//+x -y -z;  triangle points up to (dx, 0, dz)
						a_face[0] =  0;
						a_face[1] =  dy;
						a_face[2] = -dz;

						b_face[0] = -dx;
						b_face[1] =  0;
						b_face[2] = -dz;
					    }
					}
				    }else{	//-x
					if(face[2]) //+y
					{
					    if(face[4])	//+z
					    {
						//-x +y +z; triangle points down to (0, dy, 0)
						a_face[0] =  dx;
						a_face[1] =  0;
						a_face[2] =  dz;

						b_face[0] =  0;
						b_face[1] = -dy;
						b_face[2] =  dz;

					    }else{	//-z

						//-x +y -z;  triangle points up to (0, dy, dz)
						a_face[0] =  dx;
						a_face[1] =  dy;
						a_face[2] =  0;

						b_face[0] =  0;
						b_face[1] =  dy;
						b_face[2] =  dz;
					    }
					}else{	//-y
					    if(face[4])	//+z
					    {
						//-x -y +z; triangle points down to (0, 0, 0)
						a_face[0] =  0;
						a_face[1] =  dy;
						a_face[2] =  dz;

						b_face[0] =  dx;
						b_face[1] =  0;
						b_face[2] =  dz;

					    }else{	//-z

						//-x -y -z;  triangle points up to (0, 0, dz)
						a_face[0] =  0;
						a_face[1] = -dy;
						a_face[2] =  dz;

						b_face[0] =  dx;
						b_face[1] = -dy;
						b_face[2] =  0;
					    }
					}
				    }

				    //vector area of parallelogram
				    vector_cross_product_in_place(&face_vector_A[0],a_face,b_face,3);

				    //vector area of triangle needs to be scaled by 1/2
				    //for(int l=0;l<3;l++)
				//	face_vector_A[l] *= 0.5;
				    //however, we often miss about the same area in the corners
				}   
				
				A_face = vector_dot_product(face_vector_A,dir_norm,3);
				if(A_face>0)
				    dcase[3] += A_face;
				break;


			//if 4 faces contribute, only add their hypotenusal area
			case 4: fcase[4]++;


				//tunnels don't contribute
				if(face[0]&&face[1]&&face[2]&&face[3])
				    break;
				if(face[2]&&face[3]&&face[4]&&face[5])
				    break;
				if(face[4]&&face[5]&&face[0]&&face[1])
				    break;

				//add open face

				if( face[0]&&face[1] )
				{

				    //surface runs diagonal across the x face
				    if(face[2]&&face[4])
				    {
					//+y and +z
					a_face[0] =  dx;
					a_face[1] =  0;
					a_face[2] =  0;

					b_face[0] =  0;
					b_face[1] = -dy;
					b_face[2] =  dz;

				    }else if(face[2]&&face[5]){

					//+y and -z
					a_face[0] =  dx;
					a_face[1] =  0;
					a_face[2] =  0;

					b_face[0] =  0;
					b_face[1] =  dy;
					b_face[2] =  dz;

				    }else if(face[3]&&face[4]){

					//-y and +z
					a_face[0] = -dx;
					a_face[1] =  0;
					a_face[2] =  0;

					b_face[0] =  0;
					b_face[1] = -dy;
					b_face[2] = -dz;

				    }else{

					//-y and -z
					a_face[0] = -dx;
					a_face[1] =  0;
					a_face[2] =  0;

					b_face[0] =  0;
					b_face[1] = -dy;
					b_face[2] =  dz;
				    }

				}else{

				    if( face[2]&&face[3] )
				    {
					//surface runs diagonal across the y face


					if(face[0]&&face[4])
					{
					   //+x and +z
					    a_face[0] =  0;
					    a_face[1] =  dy;
					    a_face[2] =  0;

					    b_face[0] =  dx;
					    b_face[1] =  0;
					    b_face[2] = -dz;

					}else if(face[0]&&face[5]){

					   //+x and -z
					    a_face[0] =  0;
					    a_face[1] =  dy;
					    a_face[2] =  0;

					    b_face[0] = -dx; 
					    b_face[1] =  0;
					    b_face[2] = -dz;

					}else if(face[1]&&face[4]){

					   //-x and +z
					    a_face[0] =  0;
					    a_face[1] = -dy;
					    a_face[2] =  0;

					    b_face[0] = -dx;
					    b_face[1] =  0;
					    b_face[2] = -dz;

					}else{

					    //-x and -z
					    a_face[0] =  0;
					    a_face[1] = -dy;
					    a_face[2] =  0;

					    b_face[0] =  dx;
					    b_face[1] =  0;
					    b_face[2] = -dz;
					}

				    }else{
					//surface runs diagonal across the z face

					if(face[0]&&face[2])
					{
					   //+x and +y
					    a_face[0] =  0;
					    a_face[1] =  0;
					    a_face[2] =  dz;

					    b_face[0] = -dx;
					    b_face[1] =  dy;
					    b_face[2] =  0;

					}else if(face[0]&&face[3]){

					   //+x and -y
					    a_face[0] =  0;
					    a_face[1] =  0;
					    a_face[2] =  dz;

					    b_face[0] =  dx;
					    b_face[1] =  dy;
					    b_face[2] =  0;

					}else if(face[1]&&face[2]){

					   //-x and +y
					    a_face[0] =  0;
					    a_face[1] =  0;
					    a_face[2] = -dz;

					    b_face[0] =  dx;
					    b_face[1] =  dy;
					    b_face[2] =  0;

					}else{

					    //-x and -y
					    a_face[0] =  0;
					    a_face[1] =  0;
					    a_face[2] = -dz;

					    b_face[0] = -dx;
					    b_face[1] =  dy;
					    b_face[2] =  0;
					}

				    }
				}

				//find vector area of parallelogram
				vector_cross_product_in_place(&face_vector_A[0],a_face,b_face,3);

				A_face = vector_dot_product(face_vector_A,dir_norm,3);
				if(A_face>0)
				    dcase[4] += A_face;
				break;


			//if 5 contribute, only the open face is added
			case 5: fcase[5]++;

				if(!face[0])
				    face_vector_A[0] =  dy*dz;
				if(!face[1])
				    face_vector_A[0] = -dy*dz;

				if(!face[2])
				    face_vector_A[1] =  dx*dz;
				if(!face[3])
				    face_vector_A[1] = -dx*dz;

				if(!face[4])
				    face_vector_A[2] =  dx*dy;
				if(!face[5])
				    face_vector_A[2] = -dx*dy;

				A_face = vector_dot_product(face_vector_A,dir_norm,3);
				if(A_face>0)
				    dcase[5] += A_face;
				break;

			//if 0 or 6 faces, no area
			default:    break;

		    }

		    //Add the surface area
		    A_face = vector_dot_product(face_vector_A,dir_norm,3);

		    //if the dot product is positive, record the
		    //vector area
		    if(A_face>0)
			marea[ijk] = 1.0;
		}
	}

    //do upper boundary second
    if(nx_local)
    for(int j=0;j<ny;j++)
	for(int k=0;k<nz;k++)
	{
		//get cell index
		ijk = grid_ijk(nx_local-1,j,k,grid_info);

		//consider only zero cells near the interface
		n_faces = 0;
		for(int l=0;l<6;l++)
		    face[l] = 0;

		//get the value of the vector field at this cell
		for(int l=0;l<ndim;l++)
		    dir_norm[l] = vfield[l][ijk];

		if(mask[ijk]==0)
		{

		    ijkt = nz*j + k;
		    if(ghost_cell_upper[ijkt]!=0)
		    {
			n_faces++;
			face[0] = 1;
		    }

		    //causes double counting for some splits

		    ijkt = grid_ijk(nx_local-2,j,k,grid_info);
		    if(mask[ijkt]!=0)
		    {
			n_faces++;
			face[1] = 1;
		    }

		    //ijkt = grid_ijk(nx_local-1,j-1,k,grid_info);
		    ijkt = grid_ijk(nx_local-1,j+1,k,grid_info);
		    if(mask[ijkt]!=0)
		    {
			n_faces++;
			face[2] = 1;
		    }

		    //ijkt = grid_ijk(nx_local-1,j+1,k,grid_info);
		    ijkt = grid_ijk(nx_local-1,j-1,k,grid_info);
		    if(mask[ijkt]!=0)
		    {
			n_faces++;
			face[3] = 1;
		    }

		    //ijkt = grid_ijk(nx_local-1,j,k-1,grid_info);
		    ijkt = grid_ijk(nx_local-1,j,k+1,grid_info);
		    if(mask[ijkt]!=0)
		    {
			n_faces++;
			face[4] = 1;
		    }

		    //ijkt = grid_ijk(nx_local-1,j,k+1,grid_info);
		    ijkt = grid_ijk(nx_local-1,j,k-1,grid_info);
		    if(mask[ijkt]!=0)
		    {
			n_faces++;
			face[5] = 1;
		    }

		    //surface area contribution of the face
		    //in question
		    A_face = 0;
		    for(int l=0;l<3;l++)
		    {
			face_vector_A[l] = 0;
			a_face[l] = 0;
			b_face[l] = 0;
		    }

		    //add area from this cell
		    switch(n_faces)
		    {


			//if no faces contribute, just break
			case 0:	fcase[0]++;
				break;

			//if 1 face contributes, add its area
			case 1:	fcase[1]++;

				//n_tested++;
				if(face[0])
				    face_vector_A[0] = -dy * dz;
				
				if(face[1])
				    face_vector_A[0] =  dy * dz;
				
				if(face[2])
				    face_vector_A[1] = -dz * dx;

				if(face[3])
				    face_vector_A[1] =  dz * dx;

				if(face[4])
				    face_vector_A[2] = -dx * dy;
				
				if(face[5])
				    face_vector_A[2] =  dx * dy;
				
				A_face = vector_dot_product(face_vector_A,dir_norm,3);
				if(A_face>0)
				    dcase[1] += A_face;
				break;


			//if 2 faces contribute and are adjacent, add their hypotenusal area
			case 2:	fcase[2]++;

				//opposing faces don't contribute
				if( (face[0]&&face[1])||(face[2]&&face[3])||(face[4]&&face[5]) )
				    break;

				//add the hypotenuse only if the dot product is positive
				if(face[0]) //+x face
				{
					if(face[2])
					{
						//+x/+y, checked 05/22
						a_face[2] =  dz;

						b_face[0] = -dx;
						b_face[1] =  dy;

					}else if(face[3]){

						//+x/-y, checked 05/22
						a_face[2] =  dz;

						b_face[0] =  dx;
						b_face[1] =  dy;
					}

					if(face[4])
					{
						//+x/+z,checked 05/22
						a_face[1] =  dy;

						b_face[0] =  dx;
						b_face[2] = -dz;

					}else if(face[5]){

						//+x/-z, checked 05/22
						a_face[1] =  dy;

						b_face[0] =  -dx;
						b_face[2] =  -dz;
					}

				}else if(face[1]){ //-x face

					if(face[2])
					{
						//-x/+y, checked 05/22
						a_face[2] =  dz;

						b_face[0] = -dx;
						b_face[1] = -dy;

					}else if(face[3]){

						//-x/-y, checked 05/22
						a_face[2] =  dz;

						b_face[0] =  dx;
						b_face[1] = -dy;

					}

					if(face[4])
					{
						//-x/+z, checked 05/22
						a_face[1] =  dy;

						b_face[0] =  dx;
						b_face[2] =  dz;

					}else if(face[5]){

						//-x/-z, checked 05/22
						a_face[1] =  dy;

						b_face[0] = -dx;
						b_face[2] =  dz;
					}

				}else if(face[2]){  //+y face, no x-face

					//+y face 
					if(face[4])
					{
						//+y/+z, checked 05/22
						a_face[0] =  dx;

						b_face[1] = -dy;
						b_face[2] =  dz;

					}else if(face[5]){

						//+y/-z, checked 05/22
						a_face[0] =  dx;

						b_face[1] =  dy;
						b_face[2] =  dz;
					}

				}else if(face[3]){ //-y face, no x-face

					//-y face
					if(face[4])
					{
						//-y/+z, checked 05/22
						a_face[0] =  dx;

						b_face[1] = -dy;
						b_face[2] = -dz;

					}else if(face[5]){

						//-y/-z, checked 05/22
						a_face[0] =  dx;

						b_face[1] =  dy;
						b_face[2] = -dz;
					}

				}
				vector_cross_product_in_place(&face_vector_A[0],a_face,b_face,3);
		
				A_face = vector_dot_product(face_vector_A,dir_norm,3);
				if(A_face>0)
				    dcase[2] += A_face;
				break;

			//if 3 faces contribute, then treat adjacent and opposing faces separately
			case 3: fcase[3]++;

				//opposing faces contribute only the non-opposing face
				if( (face[0]&&face[1])||(face[2]&&face[3])||(face[4]&&face[5]) )
				{

				    if( face[0]&&face[1] )
				    {
					//y faces
					if(face[2])
					    face_vector_A[1] = -dz*dx;
					if(face[3])
					    face_vector_A[1] =  dz*dx;
	    
					//z faces
					if(face[4])
					    face_vector_A[2] = -dx*dy;
					if(face[5])
					    face_vector_A[2] =  dx*dy;
				    }

				    if( face[2]&&face[3] )
				    {
					//x faces
					if(face[0])
					    face_vector_A[0] = -dy*dz;
					if(face[1])
					    face_vector_A[0] =  dy*dz;

					//z faces
					if(face[4])
					    face_vector_A[2] = -dx*dy;
					if(face[5])
					    face_vector_A[2] =  dx*dy;
				    }

				    if( face[4]&&face[5] )
				    {
					//x faces
					if(face[0])
					    face_vector_A[0] = -dy*dz;
					if(face[1])
					    face_vector_A[0] =  dy*dz;

					//y faces
					if(face[2])
					    face_vector_A[1] = -dz*dx;
					if(face[3])
					    face_vector_A[1] =  dz*dx;
				    }

				}else{


				    //adjacent faces supply chamfer of triangle of hypotenuses
				    //a_face will lie in the inner most conditioned face

				    //there must be an x face
				    if(face[0])	//+x
				    {
					if(face[2]) //+y
					{
					    if(face[4])	//+z
					    {
						//+x +y +z; triangle points down to (dx, dy, 0)
						a_face[0] =  0;
						a_face[1] = -dy;
						a_face[2] =  dz;

						b_face[0] = -dx;
						b_face[1] =  0;
						b_face[2] =  dz;

					    }else{	//-z

						//+x +y -z;  triangle points up to (dx, dy, dz)
						a_face[0] =  dx;
						a_face[1] = -dy;
						a_face[2] =  0;

						b_face[0] =  dx;
						b_face[1] =  0;
						b_face[2] =  dz;
					    }
					}else{	//-y
					    if(face[4])	//+z
					    {
						//+x -y +z; triangle points down to (dx, 0, 0)
						a_face[0] =  dx;
						a_face[1] =  dy;
						a_face[2] =  0;

						b_face[0] =  dx;
						b_face[1] =  0;
						b_face[2] = -dz;

					    }else{	//-z ?

						//+x -y -z;  triangle points up to (dx, 0, dz)
						a_face[0] =  0;
						a_face[1] =  dy;
						a_face[2] = -dz;

						b_face[0] = -dx;
						b_face[1] =  0;
						b_face[2] = -dz;
					    }
					}
				    }else{	//-x
					if(face[2]) //+y
					{
					    if(face[4])	//+z
					    {
						//-x +y +z; triangle points down to (0, dy, 0)
						a_face[0] =  dx;
						a_face[1] =  0;
						a_face[2] =  dz;

						b_face[0] =  0;
						b_face[1] = -dy;
						b_face[2] =  dz;

					    }else{	//-z

						//-x +y -z;  triangle points up to (0, dy, dz)
						a_face[0] =  dx;
						a_face[1] =  dy;
						a_face[2] =  0;

						b_face[0] =  0;
						b_face[1] =  dy;
						b_face[2] =  dz;
					    }
					}else{	//-y
					    if(face[4])	//+z
					    {
						//-x -y +z; triangle points down to (0, 0, 0)
						a_face[0] =  0;
						a_face[1] =  dy;
						a_face[2] =  dz;

						b_face[0] =  dx;
						b_face[1] =  0;
						b_face[2] =  dz;

					    }else{	//-z

						//-x -y -z;  triangle points up to (0, 0, dz)
						a_face[0] =  0;
						a_face[1] = -dy;
						a_face[2] =  dz;

						b_face[0] =  dx;
						b_face[1] = -dy;
						b_face[2] =  0;
					    }
					}
				    }

				    //vector area of parallelogram
				    vector_cross_product_in_place(&face_vector_A[0],a_face,b_face,3);

				    //vector area of triangle needs to be scaled by 1/2
				    //for(int l=0;l<3;l++)
				//	face_vector_A[l] *= 0.5;
				    //however, we often miss about the same area in the corners
				}   
				
				A_face = vector_dot_product(face_vector_A,dir_norm,3);
				if(A_face>0)
				    dcase[3] += A_face;
				break;


			//if 4 faces contribute, only add their hypotenusal area
			case 4: fcase[4]++;


				//tunnels don't contribute
				if(face[0]&&face[1]&&face[2]&&face[3])
				    break;
				if(face[2]&&face[3]&&face[4]&&face[5])
				    break;
				if(face[4]&&face[5]&&face[0]&&face[1])
				    break;

				//add open face

				if( face[0]&&face[1] )
				{

				    //surface runs diagonal across the x face
				    if(face[2]&&face[4])
				    {
					//+y and +z
					a_face[0] =  dx;
					a_face[1] =  0;
					a_face[2] =  0;

					b_face[0] =  0;
					b_face[1] = -dy;
					b_face[2] =  dz;

				    }else if(face[2]&&face[5]){

					//+y and -z
					a_face[0] =  dx;
					a_face[1] =  0;
					a_face[2] =  0;

					b_face[0] =  0;
					b_face[1] =  dy;
					b_face[2] =  dz;

				    }else if(face[3]&&face[4]){

					//-y and +z
					a_face[0] = -dx;
					a_face[1] =  0;
					a_face[2] =  0;

					b_face[0] =  0;
					b_face[1] = -dy;
					b_face[2] = -dz;

				    }else{

					//-y and -z
					a_face[0] = -dx;
					a_face[1] =  0;
					a_face[2] =  0;

					b_face[0] =  0;
					b_face[1] = -dy;
					b_face[2] =  dz;
				    }

				}else{

				    if( face[2]&&face[3] )
				    {
					//surface runs diagonal across the y face


					if(face[0]&&face[4])
					{
					   //+x and +z
					    a_face[0] =  0;
					    a_face[1] =  dy;
					    a_face[2] =  0;

					    b_face[0] =  dx;
					    b_face[1] =  0;
					    b_face[2] = -dz;

					}else if(face[0]&&face[5]){

					   //+x and -z
					    a_face[0] =  0;
					    a_face[1] =  dy;
					    a_face[2] =  0;

					    b_face[0] = -dx; 
					    b_face[1] =  0;
					    b_face[2] = -dz;

					}else if(face[1]&&face[4]){

					   //-x and +z
					    a_face[0] =  0;
					    a_face[1] = -dy;
					    a_face[2] =  0;

					    b_face[0] = -dx;
					    b_face[1] =  0;
					    b_face[2] = -dz;

					}else{

					    //-x and -z
					    a_face[0] =  0;
					    a_face[1] = -dy;
					    a_face[2] =  0;

					    b_face[0] =  dx;
					    b_face[1] =  0;
					    b_face[2] = -dz;
					}

				    }else{
					//surface runs diagonal across the z face

					if(face[0]&&face[2])
					{
					   //+x and +y
					    a_face[0] =  0;
					    a_face[1] =  0;
					    a_face[2] =  dz;

					    b_face[0] = -dx;
					    b_face[1] =  dy;
					    b_face[2] =  0;

					}else if(face[0]&&face[3]){

					   //+x and -y
					    a_face[0] =  0;
					    a_face[1] =  0;
					    a_face[2] =  dz;

					    b_face[0] =  dx;
					    b_face[1] =  dy;
					    b_face[2] =  0;

					}else if(face[1]&&face[2]){

					   //-x and +y
					    a_face[0] =  0;
					    a_face[1] =  0;
					    a_face[2] = -dz;

					    b_face[0] =  dx;
					    b_face[1] =  dy;
					    b_face[2] =  0;

					}else{

					    //-x and -y
					    a_face[0] =  0;
					    a_face[1] =  0;
					    a_face[2] = -dz;

					    b_face[0] = -dx;
					    b_face[1] =  dy;
					    b_face[2] =  0;
					}

				    }
				}

				//find vector area of parallelogram
				vector_cross_product_in_place(&face_vector_A[0],a_face,b_face,3);

				A_face = vector_dot_product(face_vector_A,dir_norm,3);
				if(A_face>0)
				    dcase[4] += A_face;
				break;


			//if 5 contribute, only the open face is added
			case 5: fcase[5]++;

				if(!face[0])
				    face_vector_A[0] =  dy*dz;
				if(!face[1])
				    face_vector_A[0] = -dy*dz;

				if(!face[2])
				    face_vector_A[1] =  dx*dz;
				if(!face[3])
				    face_vector_A[1] = -dx*dz;

				if(!face[4])
				    face_vector_A[2] =  dx*dy;
				if(!face[5])
				    face_vector_A[2] = -dx*dy;

				A_face = vector_dot_product(face_vector_A,dir_norm,3);
				if(A_face>0)
				    dcase[5] += A_face;
				break;

			//if 0 or 6 faces, no area
			default:    break;
		    }

		    //Add the surface area
		    A_face = vector_dot_product(face_vector_A,dir_norm,3);

		    //if the dot product is positive, record the
		    //vector area
		    if(A_face>0)
			marea[ijk] = 1.0;
		}
	}

    //free ghost cells
    free(ghost_cell_upper);
    free(ghost_cell_lower);


    //return the vector surface area of each boundary cell
    return marea;

}


/*! \fn double *grid_power(double alpha, double *A, FFTW_Grid_Info grid_info)
 *  \brief Return a power of the grid A^alpha. */
double *grid_power(double alpha, double *A, FFTW_Grid_Info grid_info)
{

	//grid power
	double *power;

	int i, j, k;
	int ijk;

	int nx_local = grid_info.nx_local;
	int nx       = grid_info.nx;
	int ny       = grid_info.ny;
	int nz       = grid_info.nz;

	//allocate power
	power = allocate_real_fftw_grid(grid_info);

	//set power to A^alpha
	for(i=0;i<nx_local;++i)
		for(j=0;j<ny;++j)
			for(k=0;k<nz;++k)
			{
				//grid index
				ijk = grid_ijk(i,j,k,grid_info);

				//set power to A^alpha
				power[ijk] = pow( A[ijk], alpha);
			}

	//return the answer

	return power; 
}


/*! \fn double grid_min(double *A, FFTW_Grid_Info grid_info, MPI_Comm world)
 *  \brief Calculate the min of the grid. */
double grid_min(double *A, FFTW_Grid_Info grid_info, MPI_Comm world) 
{

	//min on this slab
	double min_partial = 0;

	//min across slabs
	double min = 0;

	int i, j, k;
	int ijk;

	int nx_local = grid_info.nx_local;
	int nx       = grid_info.nx;
	int ny       = grid_info.ny;
	int nz       = grid_info.nz;

	//Calculate the min on this slab of the grid
	min_partial = 1.0e32;
	for(i=0;i<nx_local;++i)
		for(j=0;j<ny;++j)
			for(k=0;k<nz;++k)
			{
				//grid index
				ijk = grid_ijk(i,j,k,grid_info);

				//local min
				if(A[ijk]<min_partial)
					min_partial = A[ijk];
			}


	//Sum the mean across processors

	MPI_Allreduce(&min_partial,&min,1,MPI_DOUBLE,MPI_MIN,world);

	//return the min
	return min;
}

/*! \fn double grid_max(double *A, FFTW_Grid_Info grid_info, MPI_Comm world)
 *  \brief Calculate the max of the grid. */
double grid_max(double *A, FFTW_Grid_Info grid_info, MPI_Comm world)
{

	//max on this slab
	double max_partial = 0;

	//max across slabs
	double max = 0;

	int i, j, k;
	int ijk;

	int nx_local = grid_info.nx_local;
	int nx       = grid_info.nx;
	int ny       = grid_info.ny;
	int nz       = grid_info.nz;

	//Calculate the min on this slab of the grid
	max_partial = -1.0e32;
	for(i=0;i<nx_local;++i)
		for(j=0;j<ny;++j)
			for(k=0;k<nz;++k)
			{
				//grid index
				ijk = grid_ijk(i,j,k,grid_info);
	
				//find local max
				if(A[ijk]>max_partial)
					max_partial = A[ijk];
			}


	//Sum the mean across processors

	MPI_Allreduce(&max_partial,&max,1,MPI_DOUBLE,MPI_MAX,world);

	//return the max
	return max;
}

/*! \fn double grid_mean(double *A, FFTW_Grid_Info grid_info, MPI_Comm world) 
 *  \brief Calculate the mean of the grid. */
double grid_mean(double *A, FFTW_Grid_Info grid_info, MPI_Comm world) 
{

	//mean on this slab
	double mean_partial = 0;

	//mean across slabs
	double mean = 0;

	int i, j, k;
	int ijk;

	int nx_local = grid_info.nx_local;
	int nx       = grid_info.nx;
	int ny       = grid_info.ny;
	int nz       = grid_info.nz;

	//Calculate the mean on this slab of the grid
	for(i=0;i<nx_local;++i)
		for(j=0;j<ny;++j)
			for(k=0;k<nz;++k)
			{
				//grid index
				ijk = grid_ijk(i,j,k,grid_info);

				//local mean
				mean_partial += A[ijk];
			}


	//Sum the mean across processors

	MPI_Allreduce(&mean_partial,&mean,1,MPI_DOUBLE,MPI_SUM,world);

	//return the mean
	mean /= ((double) nx)*((double) ny)*((double) nz);
	return mean;
}

/*! \fn double grid_weighted_mean(double *A, FFTW_Grid_Info grid_info, MPI_Comm world) 
 *  \brief Calculate the weighted mean of the grid. */
double grid_weighted_mean(double *A, double *w, FFTW_Grid_Info grid_info, MPI_Comm world) 
{

	//mean on this slab
	double partial[2] = {0,0};
	double total[2] = {0,0};


	int i, j, k;
	int ijk;

	int nx_local = grid_info.nx_local;
	int nx       = grid_info.nx;
	int ny       = grid_info.ny;
	int nz       = grid_info.nz;

	//Calculate the mean on this slab of the grid
	for(i=0;i<nx_local;++i)
		for(j=0;j<ny;++j)
			for(k=0;k<nz;++k)
			{
				//grid index
				ijk = grid_ijk(i,j,k,grid_info);

				//local mean
				partial[0] += A[ijk]*w[ijk];
				partial[1] += w[ijk];
			}


	//Sum the mean across processors

	MPI_Allreduce(&partial[0],&total[0],2,MPI_DOUBLE,MPI_SUM,world);

	//return the mean
	total[0]/=total[1];
	//printf("partial %e %e total[0] %e total[1] %e\n",partial[0],partial[1],total[0],total[1]);
	return total[0];
}

/*! \fn double grid_weighted_rms(double *A, double *w, FFTW_Grid_Info grid_info, MPI_Comm world)
 *  \brief Calculate the weighted rms value of the grid. */
double grid_weighted_rms(double *A, double *w, FFTW_Grid_Info grid_info, MPI_Comm world) 
{

	//mean on this slab
	double partial[2] = {0,0};
	double total[2]   = {0,0};

	int i, j, k;
	int ijk;

	int nx_local = grid_info.nx_local;
	int nx       = grid_info.nx;
	int ny       = grid_info.ny;
	int nz       = grid_info.nz;

	if(grid_info.ndim==2)
		nz = 1;

	//Calculate the mean on this slab of the grid
	for(i=0;i<nx_local;++i)
		for(j=0;j<ny;++j)
			for(k=0;k<nz;++k)
			{
				//grid index
				ijk = grid_ijk(i,j,k,grid_info);

				//partial sum
				partial[0] += A[ijk]*A[ijk]*w[ijk];
				partial[1] += w[ijk];
			}


	//Sum the mean across processors

	MPI_Allreduce(&partial[0],&total[0],2,MPI_DOUBLE,MPI_SUM,world);

	//return the mean

	total[0]/=total[1];
	return sqrt(total[0]);
}

/*! \fn double grid_rms(double *A, FFTW_Grid_Info grid_info, MPI_Comm world)
 *  \brief Calculate the rms value of the grid. */
double grid_rms(double *A, FFTW_Grid_Info grid_info, MPI_Comm world) 
{

	//mean on this slab
	double mean_partial = 0;

	//mean across slabs
	double mean = 0;

	int i, j, k;
	int ijk;

	int nx_local = grid_info.nx_local;
	int nx       = grid_info.nx;
	int ny       = grid_info.ny;
	int nz       = grid_info.nz;

	if(grid_info.ndim==2)
		nz = 1;

	//Calculate the mean on this slab of the grid
	for(i=0;i<nx_local;++i)
		for(j=0;j<ny;++j)
			for(k=0;k<nz;++k)
			{
				//grid index
				ijk = grid_ijk(i,j,k,grid_info);

				//partial sum
				mean_partial += A[ijk]*A[ijk];
			}


	//Sum the mean across processors

	MPI_Allreduce(&mean_partial,&mean,1,MPI_DOUBLE,MPI_SUM,world);

	//return the mean

	mean /= ((double) nx)*((double) ny)*((double) nz);
	return sqrt(mean);
}

/*! \fn double grid_field_mean(double **A, FFTW_Grid_Info grid_info, MPI_Comm world)
 *  \brief Calculate the mean value of the field A. */
double grid_field_mean(double **A, FFTW_Grid_Info grid_info, MPI_Comm world)
{

	//mean on this slab
	double mean_partial = 0;

	//mean across slabs
	double mean = 0;

	int ijk;
	int i, j, k;

	int nx_local = grid_info.nx_local;
	int nx       = grid_info.nx;
	int ny       = grid_info.ny;
	int nz       = grid_info.nz;

	if(grid_info.ndim==2)
		nz = 1;

	//Calculate the mean on this slab of the grid
	for(i=0;i<nx_local;++i)
		for(j=0;j<ny;++j)
			for(k=0;k<nz;++k)
			{

				//index
				ijk = grid_ijk(i,j,k,grid_info);

				//find the local magnitude of the field
				mean = 0;	
				for(int n=0;n<grid_info.ndim;n++)
					mean += A[n][ijk]*A[n][ijk];
				mean = sqrt(mean);
	
				//add to local sum	
				mean_partial += mean;
			}


	//Sum the mean across processors

	MPI_Allreduce(&mean_partial,&mean,1,MPI_DOUBLE,MPI_SUM,world);

	//return the mean

	mean /= ((double) nx)*((double) ny)*((double) nz);
	return mean;
}

/*! \fn double grid_field_rms(double **A, FFTW_Grid_Info grid_info, MPI_Comm world)
 *  \brief Calculate the rms value of the field A. */
double grid_field_rms(double **A, FFTW_Grid_Info grid_info, MPI_Comm world)
{

	//mean on this slab
	double mean_partial = 0;

	//mean across slabs
	double mean = 0;

	int ijk;
	int i, j, k;

	int nx_local = grid_info.nx_local;
	int nx       = grid_info.nx;
	int ny       = grid_info.ny;
	int nz       = grid_info.nz;
	//Calculate the mean on this slab of the grid
	for(i=0;i<nx_local;++i)
		for(j=0;j<ny;++j)
			for(k=0;k<nz;++k)
			{
				//grid index
				ijk = grid_ijk(i,j,k,grid_info);


				//add squared mag of local field to sum
				mean = 0;	
				for(int n=0;n<grid_info.ndim;n++)
					mean += A[n][ijk]*A[n][ijk];
	
				//add to local sum	
				mean_partial += mean;
			}


	//Sum the mean across processors

	MPI_Allreduce(&mean_partial,&mean,1,MPI_DOUBLE,MPI_SUM,world);

	//return the rms value

	mean /= ((double) nx)*((double) ny)*((double) nz);
	return sqrt(mean);
}

/*! \fn double grid_variance(double *mean, double *A, FFTW_Grid_Info grid_info, MPI_Comm world)
 *  \brief Calculate the variance of the grid.  Also return the mean. */
double grid_variance(double *mean, double *A, FFTW_Grid_Info grid_info, MPI_Comm world) 
{

	//variance on this slab
	double variance_partial = 0;

	//variance across slabs
	double variance = 0;

	//local difference from mean
	double dx;

	int i, j, k, ijk;

	int nx_local = grid_info.nx_local;
	int nx       = grid_info.nx;
	int ny       = grid_info.ny;
	int nz       = grid_info.nz;

	if(grid_info.ndim==2)
		nz=1;

	//Calculate the mean
	*mean = grid_mean(A, grid_info, world);

	//Calculate the variance on this slab of the grid
	for(i=0;i<nx_local;++i)
		for(j=0;j<ny;++j)
			for(k=0;k<nz;++k)
			{
				//grid index
				ijk = grid_ijk(i,j,k,grid_info);

				//add to variance
				dx = A[ijk] - *mean;
				variance_partial += dx*dx;
			}


	//Sum the variance across processors

	MPI_Allreduce(&variance_partial,&variance,1,MPI_DOUBLE,MPI_SUM,world);

	//return the variance estimator

	variance /= (((double) nx)*((double) ny)*((double) nz));
	return variance;
}


/*! \fn double grid_field_variance(double *mean, double **A, FFTW_Grid_Info grid_info, MPI_Comm world)
 *  \brief Calculate the variance of the field A.  Also return the mean. */
double grid_field_variance(double *mean, double **A, FFTW_Grid_Info grid_info, MPI_Comm world)
{

	//variance on this slab
	double variance_partial = 0;

	//variance across slabs
	double variance = 0;

	//local difference from mean
	double dx, dy, dz;

	int ijk;
	int i, j, k;

	int nx_local = grid_info.nx_local;
	int nx       = grid_info.nx;
	int ny       = grid_info.ny;
	int nz       = grid_info.nz;

	//Calculate the mean
	*mean = grid_field_mean(A, grid_info, world);

	//Calculate the variance on this slab of the grid
	for(i=0;i<nx_local;++i)
		for(j=0;j<ny;++j)
			for(k=0;k<nz;++k)
			{
				//grid index
				ijk = grid_ijk(i,j,k,grid_info);

				dx = 0;
				for(int n=0;n<grid_info.ndim;n++)
				{
					dx = A[n][ijk]*A[n][ijk];
				}
				dx = sqrt(dx);	//find magnitude
				variance_partial += pow(dx - *mean, 2); //find variance contribution
			}

	//Sum the variance across processors

	MPI_Allreduce(&variance_partial,&variance,1,MPI_DOUBLE,MPI_SUM,world);

	//return the variance estimator

	variance /= (((double) nx)*((double) ny)*((double) nz));
	return variance;
}

/*! \fn double grid_variance_estimator(double *mean, double *A, FFTW_Grid_Info grid_info, MPI_Comm world)
 *  \brief Calculate the variance estimator of the grid.  Also return the mean. */
double grid_variance_estimator(double *mean, double *A, FFTW_Grid_Info grid_info, MPI_Comm world)
{

	//variance on this slab
	double variance_partial = 0;

	//variance across slabs
	double variance = 0;

	//local difference from mean
	double dx;

	int i, j, k, ijk;

	int nx_local = grid_info.nx_local;
	int nx       = grid_info.nx;
	int ny       = grid_info.ny;
	int nz       = grid_info.nz;

	//Calculate the mean
	*mean = grid_mean(A, grid_info, world);

	//Calculate the variance on this slab of the grid
	for(i=0;i<nx_local;++i)
		for(j=0;j<ny;++j)
			for(k=0;k<nz;++k)
			{
				//dx = A[ (i*ny + j)*(2*(nz/2+1)) + k] - *mean;
				ijk = grid_ijk(i,j,k,grid_info);
				dx = A[ijk] - *mean;
				variance_partial += dx*dx;
			}


	//Sum the variance across processors

	MPI_Allreduce(&variance_partial,&variance,1,MPI_DOUBLE,MPI_SUM,world);

	//return the variance estimator

	variance /= (((double) nx)*((double) ny)*((double) nz) - 1);
	return variance;
}



/*! \fn void grid_enforce_mean_and_variance(double mean, double variance, double *A, FFTW_Grid_Info grid_info, MPI_Comm world)
 *  \brief Force grid A to have mean mean and variance variance. */
void grid_enforce_mean_and_variance(double mean, double variance, double *A, FFTW_Grid_Info grid_info, MPI_Comm world)
{
	double old_mean;
	double old_variance;
	double mean_test;
	double variance_test;
	int nx_local = grid_info.nx_local;
	int nx       = grid_info.nx;
	int ny       = grid_info.ny;
	int nz       = grid_info.nz;


	
	int i,j,k,ijk;

	//find the input variance
	old_variance = grid_variance(&old_mean, A, grid_info, world);

	//enforce variance
	for(i=0;i<nx_local;++i)
		for(j=0;j<ny;++j)
			for(k=0;k<nz;++k)
			{
				//grid index
				ijk = grid_ijk(i,j,k,grid_info);
			
				//rescale for variance	
				A[ijk] *= sqrt(variance)/sqrt(old_variance);
			}

	//find adjusted mean
	old_mean = grid_mean(A, grid_info, world);

	//enforce mean
	for(i=0;i<nx_local;++i)
		for(j=0;j<ny;++j)
			for(k=0;k<nz;++k)
			{
				//grid index
				ijk = grid_ijk(i,j,k,grid_info);

				//grid mean
				A[ijk] += (mean - old_mean);
			}

	//test variance
	variance_test = grid_variance(&mean_test, A, grid_info, world);

}

/*! \fn void grid_field_enforce_mean_and_variance(double mean, double variance, double **A, FFTW_Grid_Info grid_info, MPI_Comm world)
 *  \brief Force field A to have mean mean and variance variance. */
void grid_field_enforce_mean_and_variance(double mean, double variance, double **A, FFTW_Grid_Info grid_info, MPI_Comm world)
{
	double old_mean;
	double old_variance;
	double mean_test;
	double variance_test;
	int nx_local = grid_info.nx_local;
	int nx       = grid_info.nx;
	int ny       = grid_info.ny;
	int nz       = grid_info.nz;


	
	int i,j,k,ijk;

	//find the input variance
	old_variance = grid_field_variance(&old_mean, A, grid_info, world);

	//enforce variance
	for(i=0;i<nx_local;++i)
		for(j=0;j<ny;++j)
			for(k=0;k<nz;++k)
			{
				ijk = grid_ijk(i,j,k,grid_info);
			
				for(int n=0;n<grid_info.ndim;n++)	
					A[n][ijk] *= sqrt(variance)/sqrt(old_variance);
			}

	//find adjusted mean
	old_mean = grid_field_mean(A, grid_info, world);

	//enforce mean
	for(i=0;i<nx_local;++i)
		for(j=0;j<ny;++j)
			for(k=0;k<nz;++k)
			{
				ijk = grid_ijk(i,j,k,grid_info);

				for(int n=0;n<grid_info.ndim;n++)	
					A[n][ijk] += (mean - old_mean);
			}

	variance_test = grid_field_variance(&mean_test, A, grid_info, world);

}



/*! \fn double *grid_sum(double *A, double *B, FFTW_Grid_Info grid_info)
 *  \brief Performs the sum A + B on two fftw grids. */
double *grid_sum(double *A, double *B, FFTW_Grid_Info grid_info)
{
	int i, j, k;
	int ijk;

	int nx_local = grid_info.nx_local;
	int nx       = grid_info.nx;
	int ny       = grid_info.ny;
	int nz       = grid_info.nz;

	double *sum;

	//allocate sum
	sum = allocate_real_fftw_grid(grid_info);

	//Add A+B
	for(i=0;i<nx_local;++i)
		for(j=0;j<ny;++j)
			for(k=0;k<nz;++k)
			{
				//grid index
				ijk = grid_ijk(i,j,k,grid_info);
		
				//do sum
				sum[ijk] = A[ijk]+B[ijk];
			}

	//return sum
	return sum;
}

/*! \fn void grid_sum_in_place(double *A, double *B, FFTW_Grid_Info grid_info)
 *  \brief Performs the sum A + B on two fftw grids in place, result in A. */
void grid_sum_in_place(double *A, double *B, FFTW_Grid_Info grid_info)
{
	int i, j, k;
	int ijk;

	int nx_local = grid_info.nx_local;
	int nx       = grid_info.nx;
	int ny       = grid_info.ny;
	int nz       = grid_info.nz;

	//Add A+B
	for(i=0;i<nx_local;++i)
		for(j=0;j<ny;++j)
			for(k=0;k<nz;++k)
			{
				//grid index
				ijk = grid_ijk(i,j,k,grid_info);

				//do sum
				A[ijk]+=B[ijk];
			}
}

/*! \fn double *grid_difference(double *A, double *B, FFTW_Grid_Info grid_info)
 *  \brief Performs the difference A - B on two fftw grids. */
double *grid_difference(double *A, double *B, FFTW_Grid_Info grid_info)
{
	int i, j, k;
	int ijk;

	int nx_local = grid_info.nx_local;
	int nx       = grid_info.nx;
	int ny       = grid_info.ny;
	int nz       = grid_info.nz;

	double *difference;

	//allocate difference
	difference = allocate_real_fftw_grid(grid_info);

	//Subtract A-B
	for(i=0;i<nx_local;++i)
		for(j=0;j<ny;++j)
			for(k=0;k<nz;++k)
			{
				//grid index
				ijk = grid_ijk(i,j,k,grid_info);
	
				//find difference
				difference[ijk] = A[ijk]-B[ijk];
			}

	//return difference
	return difference;
}

/*! \fn void grid_difference_in_place(double *A, double *B, FFTW_Grid_Info grid_info)
 *  \brief Performs the difference A - B on two fftw grids in place, result in A. */
void grid_difference_in_place(double *A, double *B, FFTW_Grid_Info grid_info)
{
	int i, j, k;
	int ijk;

	int nx_local = grid_info.nx_local;
	int nx       = grid_info.nx;
	int ny       = grid_info.ny;
	int nz       = grid_info.nz;

	//Subtract A-B
	for(i=0;i<nx_local;++i)
		for(j=0;j<ny;++j)
			for(k=0;k<nz;++k)
			{
				//grid index
				ijk = grid_ijk(i,j,k,grid_info);

				//find difference
				A[ijk]-=B[ijk];
			}
}

/*! \fn double *grid_quotient(double *A, double *B, FFTW_Grid_Info grid_info)
 *  \brief Takes the quotient A / B on two fftw grids. */
double *grid_quotient(double *A, double *B, FFTW_Grid_Info grid_info)
{
	int i, j, k;
	int ijk;

	int nx_local = grid_info.nx_local;
	int nx       = grid_info.nx;
	int ny       = grid_info.ny;
	int nz       = grid_info.nz;

	double *quotient;

	//allocate quotient 
	quotient = allocate_real_fftw_grid(grid_info);

	//Divide A/B
	for(i=0;i<nx_local;++i)
		for(j=0;j<ny;++j)
			for(k=0;k<nz;++k)
			{
				//grid index
				ijk = grid_ijk(i,j,k,grid_info);

				//grid quotient
				quotient[ijk] = A[ijk]/B[ijk];
			}
	//return quotient
	return quotient;
}

/*! \fn void grid_quotient_in_place(double *A, double *B, FFTW_Grid_Info grid_info)
 *  \brief Takes the quotient A / B on two fftw grids in place, result in A. */
void grid_quotient_in_place(double *A, double *B, FFTW_Grid_Info grid_info)
{
	int i, j, k;
	int ijk;

	int nx_local = grid_info.nx_local;
	int nx       = grid_info.nx;
	int ny       = grid_info.ny;
	int nz       = grid_info.nz;

	//Divide A/B
	for(i=0;i<nx_local;++i)
		for(j=0;j<ny;++j)
			for(k=0;k<nz;++k)
			{
				//grid index
				ijk = grid_ijk(i,j,k,grid_info);

				//quotient
				A[ijk]/=B[ijk];
			}
}

/*! \fn double *grid_product(double *A, double *B, FFTW_Grid_Info grid_info)
 *  \brief Performs the product A * B on two fftw grids. */
double *grid_product(double *A, double *B, FFTW_Grid_Info grid_info)
{
	int i, j, k;
	int ijk;

	int nx_local = grid_info.nx_local;
	int nx       = grid_info.nx;
	int ny       = grid_info.ny;
	int nz       = grid_info.nz;

	double *product;

	//allocate product 
	product = allocate_real_fftw_grid(grid_info);

	//Multiply A*B
	for(i=0;i<nx_local;++i)
		for(j=0;j<ny;++j)
			for(k=0;k<nz;++k)
			{
				//grid index
				ijk = grid_ijk(i,j,k,grid_info);
				
				//perform product
				product[ijk] = A[ijk]*B[ijk];
			}

	//return product
	return product;
}

/*! \fn double **grid_field_product(double *A, double **B, FFTW_Grid_Info grid_info)
 *  \brief Performs the product A * B on between a scalar grid and a vector field. */
double **grid_field_product(double *A, double **B, FFTW_Grid_Info grid_info)
{
	int i, j, k;
	int ijk;

	int nx_local = grid_info.nx_local;
	int nx       = grid_info.nx;
	int ny       = grid_info.ny;
	int nz       = grid_info.nz;

	int ndim = grid_info.ndim;

	double **product;

	//allocate product 
	product = allocate_field_fftw_grid(ndim,grid_info);

	//Multiply A*B
	for(i=0;i<nx_local;++i)
		for(j=0;j<ny;++j)
			for(k=0;k<nz;++k)
			{
				//grid index
				ijk = grid_ijk(i,j,k,grid_info);

				//do A*B
				for(int n=0;n<ndim;n++)
					product[n][ijk] = A[ijk]*B[n][ijk];
			}


	//return product
	return product;
}

//operations on 3-dimensional fields

/*! \fn double grid_volume_integral(double *u, FFTW_Grid_Info grid_info, MPI_Comm world) 
 *  \brief Calculate the total volume integral I = \int u dV for a scalar grid.*/
 double grid_volume_integral(double *u, FFTW_Grid_Info grid_info, MPI_Comm world) 
{

	//integral
	double integral = 0;
	double integral_partial=0;

	int i, j, k;
	int ijk;

	int m;

	int nx_local = grid_info.nx_local;
	int nx       = grid_info.nx;
	int ny       = grid_info.ny;
	int nz       = grid_info.nz;

	//Calculate the volume integral on this slab of the grid

	for(i=0;i<nx_local;++i)
		for(j=0;j<ny;++j)
			for(k=0;k<nz;++k)
			{
				//index
				ijk = grid_ijk(i,j,k,grid_info);

				//partial integral
				integral_partial += u[ijk] * grid_info.dV;
			}


	//Sum the volume integral across processes
	
	MPI_Allreduce(&integral_partial,&integral,1,MPI_DOUBLE,MPI_SUM,world);

	
	//return the answer
	return integral;
}

/*! \fn double *grid_field_dot_product(double **A, double **B, FFTW_Grid_Info grid_info)
 *  \brief Calculate the vector dot product A . B at each grid cell -- not matrix dot product.*/
double *grid_field_dot_product(double **A, double **B, FFTW_Grid_Info grid_info)
{
	//dot product
	double *dot_product;

	int ndim = grid_info.ndim;

	int i, j, k;

	int ijk;

	int nx_local = grid_info.nx_local;
	int nx       = grid_info.nx;
	int ny       = grid_info.ny;
	int nz       = grid_info.nz;

	//allocate the dot product
	dot_product = allocate_real_fftw_grid(grid_info);

	//Calculate the product A[i][j][k][*].B[i][j][k][*]

	for(i=0;i<nx_local;++i)
		for(j=0;j<ny;++j)
			for(k=0;k<nz;++k)
			{ 
				//index
				ijk = grid_ijk(i,j,k,grid_info);

				//find local dot product
				dot_product[ijk] = 0;
				for(int n=0;n<grid_info.ndim;n++)
					dot_product[ijk] += A[n][ijk]*B[n][ijk];// + A[1][ijk]*B[1][ijk] + A[2][ijk]*B[2][ijk];
			}

	//return the dot product
	return dot_product;
}

/*! \fn double **grid_field_cross_product(double **A, double **B, FFTW_Grid_Info grid_info)
 *  \brief Calculate the cross product A x B.*/
double **grid_field_cross_product(double **A, double **B, FFTW_Grid_Info grid_info)
{
	//cross product
	double **cross_product;

	int ndim = grid_info.ndim;

	int i, j, k;

	int ijk;

	int nx_local = grid_info.nx_local;
	int nx       = grid_info.nx;
	int ny       = grid_info.ny;
	int nz       = grid_info.nz;

	//allocate cross product field
	cross_product = allocate_field_fftw_grid(ndim,grid_info);

	//Calculate the cross product

	for(i=0;i<nx_local;++i)
		for(j=0;j<ny;++j)
			for(k=0;k<nz;++k)
			{ 
				//grid index
				ijk = grid_ijk(i,j,k,grid_info);

				//find cross product
				cross_product[0][ijk] = A[1][ijk]*B[2][ijk] - A[2][ijk]*B[1][ijk];
				cross_product[1][ijk] = A[2][ijk]*B[0][ijk] - A[0][ijk]*B[2][ijk];
				cross_product[2][ijk] = A[0][ijk]*B[1][ijk] - A[1][ijk]*B[0][ijk];
			}


	//return the cross product
	return cross_product;
}

/*! \fn double **grid_field_norm(double **A, FFTW_Grid_Info grid_info)
 *  \brief Normalize the field by its local magnitude.*/
double **grid_field_norm(double **A, FFTW_Grid_Info grid_info)
{
	//norm
	double **A_hat;

	int ndim = grid_info.ndim;

	int i, j, k;

	int ijk;

	int nx_local = grid_info.nx_local;
	int nx       = grid_info.nx;
	int ny       = grid_info.ny;
	int nz       = grid_info.nz;

	double A_norm;

	//allocate norm
	A_hat = allocate_field_fftw_grid(ndim,grid_info);



	//Calculate the cross product

	for(i=0;i<nx_local;++i)
		for(j=0;j<ny;++j)
			for(k=0;k<nz;++k)
			{ 
				//grid index
				ijk = grid_ijk(i,j,k,grid_info);

				//find norm
				A_norm = 0;

				for(int l=0;l<ndim;l++)
					A_norm += A[l][ijk]*A[l][ijk];

				A_norm = sqrt(A_norm);

				//normalize
				for(int l=0;l<ndim;l++)
					A_hat[l][ijk] = A[l][ijk]/A_norm;

			}


	//return the field norm
	return A_hat;
}

/*! \fn double *grid_field_magnitude(double **A, FFTW_Grid_Info grid_info)
 *  \brief Return the local field magnitude.*/
double *grid_field_magnitude(double **A, FFTW_Grid_Info grid_info)
{

	//magnitude
	double *A_mag;

	int ndim = grid_info.ndim;

	int i, j, k;

	int ijk;

	int nx_local = grid_info.nx_local;
	int nx       = grid_info.nx;
	int ny       = grid_info.ny;
	int nz       = grid_info.nz;

	double A_norm;

	//allocate norm
	A_mag = allocate_real_fftw_grid(grid_info);



	//Calculate the cross product

	for(i=0;i<nx_local;++i)
		for(j=0;j<ny;++j)
			for(k=0;k<nz;++k)
			{ 
				//grid index
				ijk = grid_ijk(i,j,k,grid_info);

				//find norm
				A_norm = 0;

				for(int l=0;l<ndim;l++)
					A_norm += A[l][ijk]*A[l][ijk];

				A_norm = sqrt(A_norm);

				//normalize
				A_mag[ijk] = A_norm;

			}


	//return the field norm
	return A_mag;
}

//derivatives, gradients, and curls

/*! \fn double *grid_derivative(double *u, int direction, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world) 
 *  \brief Calculates the derivative of periodic u along the direction "direction".*/
double *grid_derivative(double *u, int direction, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world) 
{
	//real space derivative 
	//return grid_derivative_real_space(u, direction, grid_info, myid, numprocs, world);

	//k space derivative 
	return grid_derivative_k_space(u, direction, grid_info, myid, numprocs, world);
}


/*! \fn double *grid_derivative_real_space(double *u, int direction, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world)
 *  \brief Calculates the derivative of periodic u along the direction "direction" in real space.*/ 
double *grid_derivative_real_space(double *u, int direction, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world)
{
	//second order	
	//return grid_derivative_second_order(u, direction, grid_info, myid, numprocs, world);

	//fourth order	
	return grid_derivative_fourth_order(u, direction, grid_info, myid, numprocs, world);
}


/*! \fn double *grid_derivative_second_order(double *u, int direction, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world)
 *  \brief Calculates the derivative of periodic u along the direction "direction" using second order finite difference.*/
double *grid_derivative_second_order(double *u, int direction, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world)
{

	//second order finite difference is (u_{i+1} - u_{i-1})/(2*dx)
	//derivative
	double *derivative;


	//we need ghost cells
	//for striding across processor
	//domains if direction = 0.
	double *ghost_cell_lower;
	double *ghost_cell_upper;

	int ndim = grid_info.ndim;

	int i, j, k;

	int ijk;
	int ijkp1;
	int ijkm1;

	int nx_local = grid_info.nx_local;
	int nx       = grid_info.nx;
	int ny       = grid_info.ny;
	int nz       = grid_info.nz;

	int error_flag=0;

	//width of a single grid cell
	double dx;


	//interprocess communication
	MPI_Status status;
	

	int source;	
	int dest;	

	//allocate derivative
	derivative = allocate_real_fftw_grid(grid_info);

	
	//find the grid cell width along the derivative
	//direction
	switch(direction)
	{
		case 0: dx = grid_info.dx;
			break;
		case 1: dx = grid_info.dy;
			break;
		case 2: dx = grid_info.dz;
			break;
	}

	//printf("direction = %d dx = %e\n",direction,dx);
	//fflush(stdout);

	//if the derivative direction is along the x direction
	//then we need to get information from our neighbors about 
	//the value of the grid near the FFTW slab interfaces
	if(direction==0)
	{


		//allocate ghost cell arrays

		//lower slab
		if(!(ghost_cell_lower = (double *) malloc(ny*nz*sizeof(double))))
		{
			error_flag = 1;
		}

		//upper slab
		if(!(ghost_cell_upper = (double *) malloc(ny*nz*sizeof(double))))
		{
			error_flag = 1;
		}

		//AllCheckError(error_flag,myid,numprocs,world);


		//store ghost cells
		for(j=0;j<ny;++j)
			for(k=0;k<nz;++k)
			{
				//grid index at i==0
				i = 0;
				ijk   = grid_ijk(i,j,k,grid_info);

				//save grid at 0
				ghost_cell_upper[ nz*j + k ] = u[ijk];

				//grid index at nx_local-1
				i = nx_local-1;
				ijk   = grid_ijk(i,j,k,grid_info);

				//save grid at nx_local-1
				ghost_cell_lower[ nz*j + k ] = u[ijk];
			}



		//sendrecv upper ghost cells

		source = myid+1;
		if(source>numprocs-1)
			source-=numprocs;
		dest = myid-1;
		if(dest<0)
			dest+=numprocs;
/*
		for(i=0;i<numprocs;i++)
		{
			if(myid==i)
			{
				printf("upper myid %d source %d dest %d\n",myid,source,dest);
				fflush(stdout);
			}
		}
*/
		MPI_Sendrecv_replace(ghost_cell_upper, ny*nz, MPI_DOUBLE, dest, myid, source, source, world, &status);
	

		//sendrecv lower ghost cells

		source = myid-1;
		if(source<0)
			source+=numprocs;
		dest = myid+1;
		if(dest>numprocs-1)
			dest-=numprocs;
/*
		for(i=0;i<numprocs;i++)
		{
			if(myid==i)
			{
				printf("lower myid %d source %d dest %d\n",myid,source,dest);
				fflush(stdout);
			}
		}
*/
		MPI_Sendrecv_replace(ghost_cell_lower, ny*nz, MPI_DOUBLE, dest, myid, source, source, world, &status);

	}//direction==0
	
	//Calculate the derivative
	//but handle i=0 and i=n(x,y,z) - 1
	//differently
	for(i=0;i<nx_local;++i)
		for(j=0;j<ny;++j)
			for(k=0;k<nz;++k)
			{ 

				//index of grid
				ijk   = grid_ijk(i,j,k,grid_info);

				if(direction==0)
				{

					if(i==0)
					{
						//grid index
						ijkp1 = grid_ijk(i+1,j,k,grid_info);

						//find derivative
						derivative[ijk] = (u[ijkp1] - ghost_cell_lower[nz*j+k])/(2*dx);

					}else if(i==nx_local-1){

						//grid index
						ijkm1 = grid_ijk(i-1,j,k,grid_info);

						//find derivative
						derivative[ijk] = (ghost_cell_upper[nz*j + k] - u[ijkm1])/(2*dx);
					}else{
						//grid indicies
						ijkm1 = grid_ijk(i-1,j,k,grid_info);
						ijkp1 = grid_ijk(i+1,j,k,grid_info);
			
						//find derivative
						derivative[ijk] = (u[ijkp1] - u[ijkm1])/(2*dx);
					}
				}else{
					//treat y and z the same, with wrapping
					//find indices
					switch(direction)	
					{
						case 1: ijkm1 = grid_ijk(i,j-1,k,grid_info);
							ijkp1 = grid_ijk(i,j+1,k,grid_info);
							break;
						case 2: ijkm1 = grid_ijk(i,j,k-1,grid_info);
							ijkp1 = grid_ijk(i,j,k+1,grid_info);
							break;
					}

					//find derivative
					derivative[ijk] = (u[ijkp1] - u[ijkm1])/(2*dx);
				}
			}

	//free memory if necessary
	if(direction==0)
	{
		//free ghost cells
		free(ghost_cell_lower);
		free(ghost_cell_upper);
	}


	//return the derivative
	return derivative;
}


/*! \fn double *grid_derivative_fourth_order(double *u, int direction, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world)
 *  \brief Calculates the derivative of periodic u along the direction "direction" using fourth order finite difference.*/
double *grid_derivative_fourth_order(double *u, int direction, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world)
{
	//fourth order finite difference is dudx = ( -u_{i+2} + 8u_{i+1} - 8u_{i-1} + u_{i-2} )/(12*dx)
	//derivative
	double *derivative;


	//we need ghost cells
	//for striding across processor
	//domains if direction = 0.
	double *ghost_cell_lower;
	double *ghost_cell_upper;

	int ndim = grid_info.ndim;

	int i, j, k;

	int ijk;
	int ijkp1;
	int ijkm1;
	int ijkp2;
	int ijkm2;

	int nx_local = grid_info.nx_local;
	int nx       = grid_info.nx;
	int ny       = grid_info.ny;
	int nz       = grid_info.nz;

	int error_flag=0;

	double dx = grid_info.dx;


	double uip2, uip1, uim1, uim2;


	//interprocess communication
	MPI_Status status;
	

	//source and destination for MPI communications
	int source;	
	int dest;	

	//allocate derivative
	derivative = allocate_real_fftw_grid(grid_info);

	//find the grid cell width along the derivative
	//direction
	switch(direction)
	{
		case 0: dx = grid_info.dx;
			break;
		case 1: dx = grid_info.dy;
			break;
		case 2: dx = grid_info.dz;
			break;
	}
	

	//if direction of derivative is along the slab
	//we need to get some other information for
	//the derivative at the slab interface
	if(direction==0)
	{


		//allocate ghost cell arrays
		if(!(ghost_cell_lower = (double *) malloc(2*ny*nz*sizeof(double))))
		{
			error_flag = 1;
		}

		//AllCheckError(error_flag,myid,numprocs,world);

		if(!(ghost_cell_upper = (double *) malloc(2*ny*nz*sizeof(double))))
		{
			error_flag = 1;
		}

		//AllCheckError(error_flag,myid,numprocs,world);


		//store ghost cells
		//two on each side
		for(j=0;j<ny;++j)
			for(k=0;k<nz;++k)
			{
				i = 0;
				ijk   = grid_ijk(i,j,k,grid_info);
				ghost_cell_upper[ nz*j + k ] = u[ijk];

				i = 1;
				ijk   = grid_ijk(i,j,k,grid_info);
				ghost_cell_upper[ nz*ny + nz*j + k ] = u[ijk];

				i = nx_local-2;
				ijk   = grid_ijk(i,j,k,grid_info);
				ghost_cell_lower[ nz*j + k ] = u[ijk];

				i = nx_local-1;
				ijk   = grid_ijk(i,j,k,grid_info);
				ghost_cell_lower[ nz*ny + nz*j + k ] = u[ijk];
			}


		//sendrecv upper ghost cells

		source = myid+1;
		if(source>numprocs-1)
			source-=numprocs;
		dest = myid-1;
		if(dest<0)
			dest+=numprocs;
		MPI_Sendrecv_replace(ghost_cell_upper, 2*ny*nz, MPI_DOUBLE, dest, myid, source, source, world, &status);
	

		//sendrecv lower ghost cells

		source = myid-1;
		if(source<0)
			source+=numprocs;
		dest = myid+1;
		if(dest>numprocs-1)
			dest-=numprocs;
		MPI_Sendrecv_replace(ghost_cell_lower, 2*ny*nz, MPI_DOUBLE, dest, myid, source, source, world, &status);

	}//direction==0
	
	//Calculate the derivative
	//but handle i=0 and i=n(x,y,z) - 1
	//differently
	for(i=0;i<nx_local;++i)
		for(j=0;j<ny;++j)
			for(k=0;k<nz;++k)
			{ 
				ijk   = grid_ijk(i,j,k,grid_info);

				if(direction==0)
				{

					//get u[i-2]
					if(i-2<0)
					{	
						if(i==0)
						{
							uim2  = ghost_cell_lower[nz*j + k];
						}else{
							uim2  = ghost_cell_lower[ny*nz + nz*j + k];
						}

					}else{
						ijkm2 = grid_ijk(i-2,j,k,grid_info);
						uim2  = u[ijkm2];
					}
					//get u[i-1]
					if(i-1<0)
					{	
						uim1  = ghost_cell_lower[ny*nz + nz*j + k];
					}else{
						ijkm1 = grid_ijk(i-1,j,k,grid_info);
						uim1  = u[ijkm1];
					}

					//get u[i+1]
					if(i+1>nx_local-1)
					{
						uip1  = ghost_cell_upper[nz*j + k]; 
					}else{
						ijkp1 = grid_ijk(i+1,j,k,grid_info);
						uip1  = u[ijkp1];
					}

					//get u[i+2]
					if(i+2>nx_local-1)
					{
						if(i==nx_local-1)
						{
							uip2  = ghost_cell_upper[ny*nz + nz*j + k]; 
						}else{
							uip2  = ghost_cell_upper[nz*j + k]; 
						}
					}else{
						ijkp2 = grid_ijk(i+2,j,k,grid_info);
						uip2  = u[ijkp2];
					}


				}else{
					//treat y and z the same, with wrapping
					switch(direction)	
					{
						case 1: ijkm2 = grid_ijk(i,j-2,k,grid_info);
							ijkm1 = grid_ijk(i,j-1,k,grid_info);
							ijkp1 = grid_ijk(i,j+1,k,grid_info);
							ijkp2 = grid_ijk(i,j+2,k,grid_info);
							break;
						case 2: ijkm2 = grid_ijk(i,j,k-2,grid_info);
							ijkm1 = grid_ijk(i,j,k-1,grid_info);
							ijkp1 = grid_ijk(i,j,k+1,grid_info);
							ijkp2 = grid_ijk(i,j,k+2,grid_info);
							break;
					}

					uip2 = u[ijkp2];
					uip1 = u[ijkp1];
					uim1 = u[ijkm1];
					uim2 = u[ijkm2];
				}
			
				//dudx = ( -u_{i+2} + 8u_{i+1} - 8u_{i-1} + u_{i-2} )/(12*dx)

				derivative[ijk] = ( -uip2 + 8*uip1 - 8*uim1 + uim2 )/(12.*dx);
			}


	if(direction==0)
	{
		//free ghost cells
		free(ghost_cell_lower);
		free(ghost_cell_upper);
	}


	//return the derivative
	return derivative;
}

/*! \fn double *grid_derivative_k_space(double *uin, int direction, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world)
 *  \brief Calculates the derivative of periodic u along the direction "direction" in k-space. */
double *grid_derivative_k_space(double *uin, int direction, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world)
{

//#define TEST_DERIVE

  //derivative array
  double *derivative;
	
  //complex data field
  fftw_complex *u;

  //fourier components of vector field
  fftw_complex *uk;

  //indices
  int i, j, k;

  int ijk;

  int nx_local = grid_info.nx_local;
  int nx_local_start = grid_info.nx_local_start;
  int nx       = grid_info.nx;
  int ny       = grid_info.ny;
  int nz       = grid_info.nz;
  int nz_complex = grid_info.nz_complex;
  int nzl;

  double kx, ky, kz;
  double kk;
  double L = grid_info.BoxSize;

  double A, B;

  int ndim = grid_info.ndim;

  //normalization
  double scale = 1./( ((double) grid_info.nx)*((double) grid_info.ny)*((double) grid_info.nz) );

  //forward and reverse FFTW plans
  fftw_plan plan;
  fftw_plan iplan;


  //allocate decomposed field array
  derivative = allocate_real_fftw_grid(grid_info);

  //allocate work and transofrm
  u    = allocate_complex_fftw_grid(grid_info);
  uk   = allocate_complex_fftw_grid(grid_info);


  //create the fftw plans
  plan  = fftw_mpi_plan_dft_3d(grid_info.nx, grid_info.ny, grid_info.nz, u,  u, world, FFTW_FORWARD,  FFTW_ESTIMATE);
  iplan = fftw_mpi_plan_dft_3d(grid_info.nx, grid_info.ny, grid_info.nz, uk,  uk, world, FFTW_BACKWARD, FFTW_ESTIMATE);

  //copy data into u
  grid_copy_real_to_complex_in_place(uin, u, grid_info);



  //perform the forward transform on the components of u
  fftw_execute(plan);


#ifndef TEST_DERIVE
//at this stage, uk contains the Fourier transform of u

  for(int i=0;i<nx_local;++i)
    for(int j=0;j<ny;++j)
    {
      //first, positive kz
      for(int k=0;k<nz;++k)
      {

	//x frequency
	if(i+nx_local_start<nx/2)
	{
	  kx = ((double) (i + nx_local_start));
	}else{
	  kx = -nx + ((double) (i + nx_local_start));
	}
	if(j<ny/2)
	{
	  ky = ((double) j);
	}else{
	  ky = -ny + ((double) j);
	}
	if(k<nz/2)
	{
	  kz = ((double) k);
	}else{
	  kz = -nz + ((double) k);
	}


	kx *= (2*M_PI/L);
	ky *= (2*M_PI/L);
	kz *= (2*M_PI/L);


	//index of uk corresponding to k

	ijk = grid_complex_ijk(i,j,k,grid_info);

	// du/dx = i * kx * U

	switch(direction)
	{
	  case 0: A = ( kx*u[ijk][0] );
		  B = ( kx*u[ijk][1] );
		  break;
	  case 1: A = ( ky*u[ijk][0] );
		  B = ( ky*u[ijk][1] );
		  break;
	  case 2: A = ( kz*u[ijk][0] );
		  B = ( kz*u[ijk][1] );
		  break;
	}

	uk[ijk][0] = -B * scale;
	uk[ijk][1] =  A * scale;
      }
    }
#else  /*TEST_DERIVE*/

  //at this stage, uk contains the Fourier transform of u
  for(int i=0;i<nx_local;++i)
    for(int j=0;j<ny;++j)
    {
      for(int k=0;k<nz;++k)
      {
	//kx = sin( 2.0*M_PI*(((double) (i + nx_local_start))/((double) nx)));
	//ky = sin( 2.0*M_PI*(((double) j)/((double) ny)));
	//kz = sin( 2.0*M_PI*(((double) k)/((double) nz)));

	//x frequency
	if(i+nx_local_start<nx/2)
	{
	  kx = ((double) (i + nx_local_start));
	}else{
	  kx = -nx + ((double) (i + nx_local_start));
	}
	if(j<ny/2)
	{
	  ky = ((double) j);
	}else{
	  ky = -ny + ((double) j);
	}
	if(k<nz/2)
	{
	  kz = ((double) k);
	}else{
	 kz = -nz + ((double) k);
	}
	//kx = sin( 2.0*M_PI*kx/((double) nx/2)) * 2.0*M_PI/L;
	//ky = sin( 2.0*M_PI*ky/((double) ny/2)) * 2.0*M_PI/L;
	//kz = sin( 2.0*M_PI*kz/((double) nz/2)) * 2.0*M_PI/L;
	kx = sin( 2.0*M_PI*kx/((double) nx/2)) * pow(2.0*M_PI/L,3);
	ky = sin( 2.0*M_PI*ky/((double) ny/2)) * pow(2.0*M_PI/L,3);
	kz = sin( 2.0*M_PI*kz/((double) nz/2)) * pow(2.0*M_PI/L,3);


	//index of uk corresponding to k
	ijk = grid_complex_ijk(i,j,k,grid_info);

	// du/dx = i * kx * U

	switch(direction)
	{
	  case 0: A = ( kx*u[ijk][0] );
		  B = ( kx*u[ijk][1] );
		  break;
	  case 1: A = ( ky*u[ijk][0] );
		  B = ( ky*u[ijk][1] );
		  break;
	  case 2: A = ( kz*u[ijk][0] );
		  B = ( kz*u[ijk][1] );
		  break;
	}

	uk[ijk][0] =  -B * scale;
	uk[ijk][1] =   A * scale;

  
	//uk[ijk][0] =   A * scale; //real component is zero
	//uk[ijk][1] =   B * scale;
      }
    }

#endif /*TEST_DERIVE*/


  //perform the inverse transform of the derivative
  fftw_execute(iplan);

  //copy inverse transform into derivative
  grid_copy_complex_to_real_in_place(uk, derivative, grid_info);

  //free the buffer memory
  fftw_free(u);
  fftw_free(uk);

  //destroy the plans
  fftw_destroy_plan(plan);
  fftw_destroy_plan(iplan);

  //return the answer
  return derivative;
}

/*! \fn double **grid_field_curl(double **u, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world)
 *  \brief Calculate the curl = nabla cross u. */
double **grid_field_curl(double **u, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world)
{
  //select real-or k-space
  /*tested on 10/8/2013 using function with known curl*/

  //real space
  //return grid_field_curl_real_space(u, grid_info, myid, numprocs, world);

  /*k space*/
  return grid_field_curl_k_space(u, grid_info, myid, numprocs, world);
}

/*! \fn double **grid_field_curl_real_space(double **u, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world)
 *  \brief Calculate the curl = nabla cross u in real space. */
double **grid_field_curl_real_space(double **u, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world)
{
	//curl
	double **curl;

	int ndim = grid_info.ndim;

	int i, j, k;

	int ijk;

	int nx_local = grid_info.nx_local;
	int nx       = grid_info.nx;
	int ny       = grid_info.ny;
	int nz       = grid_info.nz;

	//field component and derivative direction
	double *dux_dy;
	double *dux_dz;
	double *duy_dx;
	double *duy_dz;
	double *duz_dx;
	double *duz_dy;

	//two grids
	double *A;
	double *B;

	int ndim_loop = grid_info.ndim;


	//allocate the curl
	curl = allocate_field_fftw_grid(ndim,grid_info);

	/* curl = 
	 *	  | i   j   k   |
	 *	  | ddx ddy ddz |
	 *	  | ux  uy  uz  |
	 *	  = (duz/dy - duy/dz)i + (dux/dz - duz/dx)j + (duy/dx - dux/dy)k
	 */

	//get curl components

	dux_dy = grid_derivative_real_space(u[0], 1, grid_info, myid, numprocs, world);
	dux_dz = grid_derivative_real_space(u[0], 2, grid_info, myid, numprocs, world);

	duy_dx = grid_derivative_real_space(u[1], 0, grid_info, myid, numprocs, world);
	duy_dz = grid_derivative_real_space(u[1], 2, grid_info, myid, numprocs, world);

	duz_dx = grid_derivative_real_space(u[2], 0, grid_info, myid, numprocs, world);
	duz_dy = grid_derivative_real_space(u[2], 1, grid_info, myid, numprocs, world);

	//store into the curl
	for(int i=0;i<nx_local;++i)
		for(int j=0;j<ny;++j)
			for(int k=0;k<nz;++k)
			{
				//grid index
				ijk = grid_ijk(i,j,k,grid_info);

				curl[0][ijk] = ( duz_dy[ijk] - duy_dz[ijk] );
				curl[1][ijk] = ( dux_dz[ijk] - duz_dx[ijk] );
				curl[2][ijk] = ( duy_dx[ijk] - dux_dy[ijk] );
			}

	//free memory
	fftw_free(dux_dy);
	fftw_free(dux_dz);
	fftw_free(duy_dx);
	fftw_free(duy_dz);
	fftw_free(duz_dx);
	fftw_free(duz_dy);

	//return the answer
	return curl;
}

/*! \fn double **grid_field_curl_k_space(double **u, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world)
 *  \brief Calculate the curl = nabla cross u in k space. */
double **grid_field_curl_k_space(double **u, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world)
{
	//curl
	double **curl;

	int ndim = grid_info.ndim;

	int ijk;

	int nx_local = grid_info.nx_local;
	int nx       = grid_info.nx;
	int ny       = grid_info.ny;
	int nz       = grid_info.nz;

	//field component and derivative direction
	double *dux_dy;
	double *dux_dz;
	double *duy_dx;
	double *duy_dz;
	double *duz_dx;
	double *duz_dy;

	//allocate the curl
	curl = allocate_field_fftw_grid(ndim,grid_info);

	/* curl = 
	 *	  | i   j   k   |
	 *	  | ddx ddy ddz |
	 *	  | ux  uy  uz  |
	 *	  = (duz/dy - duy/dz)i + (dux/dz - duz/dx)j + (duy/dx - dux/dy)k
	 */

	//get curl components

	dux_dy = grid_derivative_k_space(u[0], 1, grid_info, myid, numprocs, world);
	dux_dz = grid_derivative_k_space(u[0], 2, grid_info, myid, numprocs, world);

	duy_dx = grid_derivative_k_space(u[1], 0, grid_info, myid, numprocs, world);
	duy_dz = grid_derivative_k_space(u[1], 2, grid_info, myid, numprocs, world);

	duz_dx = grid_derivative_k_space(u[2], 0, grid_info, myid, numprocs, world);
	duz_dy = grid_derivative_k_space(u[2], 1, grid_info, myid, numprocs, world);

	//store into the curl
	for(int i=0;i<nx_local;++i)
		for(int j=0;j<ny;++j)
			for(int k=0;k<nz;++k)
			{
				//grid index
				ijk = grid_ijk(i,j,k,grid_info);

				curl[0][ijk] = ( duz_dy[ijk] - duy_dz[ijk] );
				curl[1][ijk] = ( dux_dz[ijk] - duz_dx[ijk] );
				curl[2][ijk] = ( duy_dx[ijk] - dux_dy[ijk] );
			}

	//free memory
	fftw_free(dux_dy);
	fftw_free(dux_dz);
	fftw_free(duy_dx);
	fftw_free(duy_dz);
	fftw_free(duz_dx);
	fftw_free(duz_dy);

	//return the answer
	return curl;
}

/*! \fn double *grid_field_divergence(double **u, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world)
 *  \brief Calculate the div = nabla dot u. */
double *grid_field_divergence(double **u, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world)
{
	//divergence in real space
	//return grid_field_divergence_real_space(u, grid_info, myid, numprocs, world);

  //divergence in k space
  //checked 10/8/13 with function of known divergence 
  return grid_field_divergence_k_space(u, grid_info, myid, numprocs, world);
}

/*! \fn double *grid_field_divergence_real_space(double **u, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world)
 *  \brief Calculate the div = nabla dot u in real space. */
double *grid_field_divergence_real_space(double **u, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world)
{
	//divergence
	double *divergence;

	//derivatives 
	double *duxdx, *duydy, *duzdz;

	//allocate the divergence
	divergence = allocate_real_fftw_grid(grid_info);

	//set divergence to zero

	grid_set_value(0,divergence,grid_info);

	//first calculate the derivatives

	duxdx = grid_derivative_real_space(u[0], 0, grid_info, myid, numprocs, world);
	duydy = grid_derivative_real_space(u[1], 1, grid_info, myid, numprocs, world);
	duzdz = grid_derivative_real_space(u[2], 2, grid_info, myid, numprocs, world);

	//add derivatives to divergence
	grid_sum_in_place(divergence, duxdx, grid_info);
	grid_sum_in_place(divergence, duydy, grid_info);
	grid_sum_in_place(divergence, duzdz, grid_info);

	//free the derivative grids
	fftw_free(duxdx);
	fftw_free(duydy);
	fftw_free(duzdz);

	//return the divergence

	return divergence;
}

/*! \fn double *grid_field_divergence_k_space(double **u, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world)
 *  \brief Calculate the div = nabla dot u in k space. */
double *grid_field_divergence_k_space(double **u, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world)
{
	//divergence
	double *divergence;

	//derivatives 
	double *duxdx, *duydy, *duzdz;

	//allocate the divergence
	divergence = allocate_real_fftw_grid(grid_info);

	//set divergence to zero

	grid_set_value(0,divergence,grid_info);

	//first calculate the derivatives

	duxdx = grid_derivative_k_space(u[0], 0, grid_info, myid, numprocs, world);
	duydy = grid_derivative_k_space(u[1], 1, grid_info, myid, numprocs, world);
	duzdz = grid_derivative_k_space(u[2], 2, grid_info, myid, numprocs, world);

	//add derivatives to divergence
	grid_sum_in_place(divergence, duxdx, grid_info);
	grid_sum_in_place(divergence, duydy, grid_info);
	grid_sum_in_place(divergence, duzdz, grid_info);

	//free the derivative grids
	fftw_free(duxdx);
	fftw_free(duydy);
	fftw_free(duzdz);

	//return the divergence

	return divergence;
}

/*! \fn double **grid_gradient(double *u, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world)
 *  \brief Calculate the grad = nabla u. */
double **grid_gradient(double *u, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world)
{
	//gradient in real space
	//return grid_gradient_real_space(u, grid_info, myid, numprocs, world);

	//gradient in k space
	return grid_gradient_k_space(u, grid_info, myid, numprocs, world);
}

/*! \fn double **grid_gradient_real_space(double *u, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world)
 *  \brief Calculate the grad = nabla u in real space. */
double **grid_gradient_real_space(double *u, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world)
{
	//gradient
	double **grad;

	//derivatives 
	double *dudx;
	double *dudy;
	double *dudz;

	//allocate the gradient
	grad = allocate_field_fftw_grid(grid_info.ndim,grid_info);

	//set gradient to zero
	grid_field_set_value(0,grad,grid_info);

	//first calculate dudx
	dudx = grid_derivative_real_space(u, 0, grid_info, myid, numprocs, world);
	dudy = grid_derivative_real_space(u, 1, grid_info, myid, numprocs, world);
	dudz = grid_derivative_real_space(u, 2, grid_info, myid, numprocs, world);

	//save dudx in grad[0]
	grid_sum_in_place(grad[0], dudx, grid_info);
	grid_sum_in_place(grad[1], dudy, grid_info);
	grid_sum_in_place(grad[2], dudz, grid_info);
	
	//free the derivative grids
	fftw_free(dudx);
	fftw_free(dudy);
	fftw_free(dudz);

	//return the gradient
	return grad;
}

/*! \fn double **grid_gradient_k_space(double *u, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world)
 *  \brief Calculate the grad = nabla u in k space. */
double **grid_gradient_k_space(double *u, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world)
{
	//gradient
	double **grad;

	//derivatives 
	double *dudx;
	double *dudy;
	double *dudz;

	//allocate the gradient
	grad = allocate_field_fftw_grid(grid_info.ndim,grid_info);

	//set gradient to zero
	grid_field_set_value(0,grad,grid_info);

	//first calculate dudx
	dudx = grid_derivative_k_space(u, 0, grid_info, myid, numprocs, world);
	dudy = grid_derivative_k_space(u, 1, grid_info, myid, numprocs, world);
	dudz = grid_derivative_k_space(u, 2, grid_info, myid, numprocs, world);

	//save dudx in grad[0]
	grid_sum_in_place(grad[0], dudx, grid_info);
	grid_sum_in_place(grad[1], dudy, grid_info);
	grid_sum_in_place(grad[2], dudz, grid_info);
	
	//free the derivative grids
	fftw_free(dudx);
	fftw_free(dudy);
	fftw_free(dudz);

	//return the gradient
	return grad;
}

//vorticity, helicity, and dilatation

/*! \fn double **grid_field_vorticity(double **u, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world)
 *  \brief Calculate the vorticity = nabla cross u. */
double **grid_field_vorticity(double **u, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world)
{
	//vorticity
	double **vorticity;

	//find vorticity as omega = nabla cross u
	vorticity = grid_field_curl(u, grid_info, myid, numprocs, world);

	//return the answer
	return vorticity;
}

/*! \fn double *grid_field_helicity(double **u, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world)
 *  \brief Calculate the field helicity h = u . nabla cross u, or h = u dot vorticity. */
double *grid_field_helicity(double **u, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world)
{
	//helicity
	double *helicity;

	//vorticity
	double **vorticity;

	//find vorticity as omega = nabla cross u
	vorticity = grid_field_vorticity(u, grid_info, myid, numprocs, world);

	//find helicity as h = u dot vorticity
	helicity = grid_field_dot_product(u, vorticity, grid_info);

	//free the vorticity
	deallocate_field_fftw_grid(vorticity,grid_info.ndim,grid_info);
	
	//return the answer
	return helicity;

}

/*! \fn double *grid_field_dilatation(double **u, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world)
 *  \brief Calculates the dilatation of field u */
double *grid_field_dilatation(double **u, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world)
{
	//diliataion is the divergence
	return grid_field_divergence(u, grid_info, myid, numprocs, world);
}



//helmholtz decomposition

/*! \fn double **grid_field_dilatational_component(double **uin, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world)
 *  \brief Calculate the dilatational component of field u via Helmholtz decomposition in Fourier space. */
double **grid_field_dilatational_component(double **uin, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world)
{

#define TEST_DIL

  //diltational component
  double **dil;
	
  //complex data field
  fftw_complex *ux;
  fftw_complex *uy;
  fftw_complex *uz;

  //fourier components of vector field
  fftw_complex *ukx;
  fftw_complex *uky;
  fftw_complex *ukz;

	//indices
	int ijk;

	int nx_local = grid_info.nx_local;
	int nx_local_start = grid_info.nx_local_start;
	int nx       = grid_info.nx;
	int ny       = grid_info.ny;
	int nz       = grid_info.nz;
	int nz_complex = grid_info.nz_complex;

	double kx, ky, kz;
	double kk;
	double L = grid_info.BoxSize;

	//normalization
	double scale = 1./( ((double) grid_info.nx)*((double) grid_info.ny)*((double) grid_info.nz) );

	//forward and reverse FFTW plans
	fftw_plan planx;
	fftw_plan plany;
	fftw_plan planz;
	fftw_plan iplanx;
	fftw_plan iplany;
	fftw_plan iplanz;

	//allocate decomposed field array
	dil = allocate_field_fftw_grid(grid_info.ndim,grid_info);

	//allocate work and transofrm
	ux   = allocate_complex_fftw_grid(grid_info);
	uy   = allocate_complex_fftw_grid(grid_info);
	uz   = allocate_complex_fftw_grid(grid_info);
	ukx  = allocate_complex_fftw_grid(grid_info);
	uky  = allocate_complex_fftw_grid(grid_info);
	ukz  = allocate_complex_fftw_grid(grid_info);


	//create the forward fftw plans
	planx  = fftw_mpi_plan_dft_3d(grid_info.nx, grid_info.ny, grid_info.nz, ux,  ux, world, FFTW_FORWARD,  FFTW_ESTIMATE);
	plany  = fftw_mpi_plan_dft_3d(grid_info.nx, grid_info.ny, grid_info.nz, uy,  uy, world, FFTW_FORWARD,  FFTW_ESTIMATE);
	planz  = fftw_mpi_plan_dft_3d(grid_info.nx, grid_info.ny, grid_info.nz, uz,  uz, world, FFTW_FORWARD,  FFTW_ESTIMATE);

	//create the forward fftw plans
	iplanx  = fftw_mpi_plan_dft_3d(grid_info.nx, grid_info.ny, grid_info.nz, ukx,  ukx, world, FFTW_BACKWARD,  FFTW_ESTIMATE);
	iplany  = fftw_mpi_plan_dft_3d(grid_info.nx, grid_info.ny, grid_info.nz, uky,  uky, world, FFTW_BACKWARD,  FFTW_ESTIMATE);
	iplanz  = fftw_mpi_plan_dft_3d(grid_info.nx, grid_info.ny, grid_info.nz, ukz,  ukz, world, FFTW_BACKWARD,  FFTW_ESTIMATE);


	//copy data into u
	grid_copy_real_to_complex_in_place(uin[0], ux, grid_info);
	grid_copy_real_to_complex_in_place(uin[1], uy, grid_info);
	grid_copy_real_to_complex_in_place(uin[2], uz, grid_info);


	//perform the forward transform on the components of uin
	fftw_execute(planx);
	fftw_execute(plany);
	fftw_execute(planz);

	//at this stage, ux contains the Fourier transform of uin[0]
	//at this stage, uy contains the Fourier transform of uin[1]
	//at this stage, uz contains the Fourier transform of uin[2]

#ifndef TEST_DIL
	for(int i=0;i<nx_local;++i)
		for(int j=0;j<ny;++j)
		{
			//first, positive kz
			for(int k=0;k<nz;++k)
			{




				//x frequency
				if(i+nx_local_start<nx/2)
				{
					kx = ((double) (i + nx_local_start));
				}else{
					kx = -nx + ((double) (i + nx_local_start));
				}
				if(j<ny/2)
				{
					ky = ((double) j);
				}else{
					ky = -ny + ((double) j);
				}
				if(k<nz/2)
				{
					kz = ((double) k);
				}else{
					kz = -nz + ((double) k);
				}


				kx *= (2*M_PI/L);
				ky *= (2*M_PI/L);
				kz *= (2*M_PI/L);

				//magnitude of k vector
				kk = sqrt(kx*kx + ky*ky + kz*kz);


				//index of uk corresponding to k

				ijk = grid_complex_ijk(i,j,k,grid_info);


				// need to keep \hat{k}(\hat{k} dot U)
				if(i+nx_local_start==0 && j==0 && k==0)
				{
					
					ukx[ijk][0] = 0;
					ukx[ijk][1] = 0;
				
					uky[ijk][0] = 0;
					uky[ijk][1] = 0;

					ukz[ijk][0] = 0;
					ukz[ijk][1] = 0;
				
				}else{

					//find khat dot U
					ukx[ijk][0] = kx*ux[ijk][0]/kk * scale;
					ukx[ijk][1] = kx*ux[ijk][1]/kk * scale;
				
					uky[ijk][0] = ky*uy[ijk][0]/kk * scale;
					uky[ijk][1] = ky*uy[ijk][1]/kk * scale; 

					ukz[ijk][0] = kz*uz[ijk][0]/kk * scale;
					ukz[ijk][1] = kz*uz[ijk][1]/kk * scale; 

					//need to project along khat
					ukx[ijk][0] *= kx/kk;
					ukx[ijk][1] *= kx/kk;
				
					uky[ijk][0] *= ky/kk;
					uky[ijk][1] *= ky/kk;

					ukz[ijk][0] *= kz/kk;
					ukz[ijk][1] *= kz/kk;
		
				}
			}
		}
#else /*TEST_DIL*/

/*


  // Project off non-solenoidal component of velocity 

//gnx1 is the global x-length
//gnx2 is the global y-length
//gnx3 is the global z-length
//gis+i is the global x position
//gjs+j is the global y position
//gks+k is the global z position

  for (k=0; k<nx3; k++) {
    kap[2] = sin(2.0*PI*(gks+k)/gnx3);
    for (j=0; j<nx2; j++) {
      kap[1] = sin(2.0*PI*(gjs+j)/gnx2);
      for (i=0; i<nx1; i++) {
        if (((gis+i)+(gjs+j)+(gks+k)) != 0) {
          kap[0] = sin(2.0*PI*(gis+i)/gnx1);
          ind = OFST(i,j,k);

          // make kapn a unit vector 
          mag = sqrt(SQR(kap[0]) + SQR(kap[1]) + SQR(kap[2]));
          for (m=0; m<3; m++) kapn[m] = kap[m] / mag;

          // find fv_0 dot kapn 
          dot[0] = fv1[ind][0]*kapn[0]+fv2[ind][0]*kapn[1]+fv3[ind][0]*kapn[2];
          dot[1] = fv1[ind][1]*kapn[0]+fv2[ind][1]*kapn[1]+fv3[ind][1]*kapn[2];

          // fv = fv_0 - (fv_0 dot kapn) * kapn 
          fv1[ind][0] -= dot[0]*kapn[0];
          fv2[ind][0] -= dot[0]*kapn[1];
          fv3[ind][0] -= dot[0]*kapn[2];

          fv1[ind][1] -= dot[1]*kapn[0];
          fv2[ind][1] -= dot[1]*kapn[1];
          fv3[ind][1] -= dot[1]*kapn[2];
        }
      }
    }
 */      
  for(int i=0;i<nx_local;++i)
    for(int j=0;j<ny;++j)
    {
      for(int k=0;k<nz;++k)
      {
	//x frequency
	if(i+nx_local_start<nx/2)
	{
	kx = ((double) (i + nx_local_start));
	}else{
	kx = -nx + ((double) (i + nx_local_start));
	}
	if(j<ny/2)
	{
	ky = ((double) j);
	}else{
	ky = -ny + ((double) j);
	}
	if(k<nz/2)
	{
	kz = ((double) k);
	}else{
	kz = -nz + ((double) k);
	}

	//magnitude of k vector
	kk = sqrt(kx*kx + ky*ky + kz*kz);

	kx /= kk;
	ky /= kk;
	kz /= kk;



	//index of uk corresponding to k
	ijk = grid_complex_ijk(i,j,k,grid_info);


	// need to keep \hat{k}(\hat{k} dot U)
	if(i+nx_local_start==0 && j==0 && k==0)
	{
					
	  ukx[ijk][0] = 0;
	  ukx[ijk][1] = 0;
	  uky[ijk][0] = 0;
	  uky[ijk][1] = 0;
	  ukz[ijk][0] = 0;
	  ukz[ijk][1] = 0;
				
	}else{

	  //find khat dot U
	  ukx[ijk][0] = ux[ijk][0]*kx + uy[ijk][0]*ky + uz[ijk][0]*kz;
	  ukx[ijk][1] = ux[ijk][1]*kx + uy[ijk][1]*ky + uz[ijk][1]*kz; 
	
	  uky[ijk][0] = ukx[ijk][0];
	  uky[ijk][1] = ukx[ijk][1];

	  ukz[ijk][0] = ukx[ijk][0];
	  ukz[ijk][1] = ukx[ijk][1];

					//need to project along khat
	  ukx[ijk][0] *= scale*kx;
	  ukx[ijk][1] *= scale*kx;
	
	  uky[ijk][0] *= scale*ky;
	  uky[ijk][1] *= scale*ky;

	  ukz[ijk][0] *= scale*kz;
	  ukz[ijk][1] *= scale*kz;
	
	}
      }
    }
#endif /*TEST_DIL*/


  //perform the inverse transform of the derivative
  fftw_execute(iplanx);
  fftw_execute(iplany);
  fftw_execute(iplanz);

  //copy inverse transform into derivative
  grid_copy_complex_to_real_in_place(ukx, dil[0], grid_info);
  grid_copy_complex_to_real_in_place(uky, dil[1], grid_info);
  grid_copy_complex_to_real_in_place(ukz, dil[2], grid_info);

  //free the buffer memory
  fftw_free(ux);
  fftw_free(uy);
  fftw_free(uz);
  fftw_free(ukx);
  fftw_free(uky);
  fftw_free(ukz);

  //destroy the plans
  fftw_destroy_plan(planx);
  fftw_destroy_plan(plany);
  fftw_destroy_plan(planz);
  fftw_destroy_plan(iplanx);
  fftw_destroy_plan(iplany);
  fftw_destroy_plan(iplanz);

  //return the answer
  return dil;
}

/*! \fn double **grid_field_solenoidal_component(double **uin, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world)
 *  \brief Calculate the solenoidal component of field u via Helmholtz decomposition in Fourier space. */
double **grid_field_solenoidal_component(double **uin, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world)
{

#define TEST_SOL

	//solenoidal component
	double **sol;
	
	//complex data field
	fftw_complex *ux;
	fftw_complex *uy;
	fftw_complex *uz;

	//fourier components of vector field
	fftw_complex *ukx;
	fftw_complex *uky;
	fftw_complex *ukz;

	//indices
	int ijk;

	int nx_local = grid_info.nx_local;
	int nx_local_start = grid_info.nx_local_start;
	int nx       = grid_info.nx;
	int ny       = grid_info.ny;
	int nz       = grid_info.nz;
	int nz_complex = grid_info.nz_complex;

	double kx, ky, kz;
	double kk;
	double L = grid_info.BoxSize;

	//normalization
	double scale = 1./( ((double) grid_info.nx)*((double) grid_info.ny)*((double) grid_info.nz) );

	//forward and reverse FFTW plans
	fftw_plan planx;
	fftw_plan plany;
	fftw_plan planz;
	fftw_plan iplanx;
	fftw_plan iplany;
	fftw_plan iplanz;

	//allocate decomposed field array
	sol = allocate_field_fftw_grid(grid_info.ndim,grid_info);

	//allocate work and transofrm
	ux   = allocate_complex_fftw_grid(grid_info);
	uy   = allocate_complex_fftw_grid(grid_info);
	uz   = allocate_complex_fftw_grid(grid_info);
	ukx  = allocate_complex_fftw_grid(grid_info);
	uky  = allocate_complex_fftw_grid(grid_info);
	ukz  = allocate_complex_fftw_grid(grid_info);


	//create the forward fftw plans
	planx  = fftw_mpi_plan_dft_3d(grid_info.nx, grid_info.ny, grid_info.nz, ux,  ux, world, FFTW_FORWARD,  FFTW_ESTIMATE);
	plany  = fftw_mpi_plan_dft_3d(grid_info.nx, grid_info.ny, grid_info.nz, uy,  uy, world, FFTW_FORWARD,  FFTW_ESTIMATE);
	planz  = fftw_mpi_plan_dft_3d(grid_info.nx, grid_info.ny, grid_info.nz, uz,  uz, world, FFTW_FORWARD,  FFTW_ESTIMATE);

	//create the forward fftw plans
	iplanx  = fftw_mpi_plan_dft_3d(grid_info.nx, grid_info.ny, grid_info.nz, ukx,  ukx, world, FFTW_BACKWARD,  FFTW_ESTIMATE);
	iplany  = fftw_mpi_plan_dft_3d(grid_info.nx, grid_info.ny, grid_info.nz, uky,  uky, world, FFTW_BACKWARD,  FFTW_ESTIMATE);
	iplanz  = fftw_mpi_plan_dft_3d(grid_info.nx, grid_info.ny, grid_info.nz, ukz,  ukz, world, FFTW_BACKWARD,  FFTW_ESTIMATE);


	//copy data into u
	grid_copy_real_to_complex_in_place(uin[0], ux, grid_info);
	grid_copy_real_to_complex_in_place(uin[1], uy, grid_info);
	grid_copy_real_to_complex_in_place(uin[2], uz, grid_info);


	//perform the forward transform on the components of uin
	fftw_execute(planx);
	fftw_execute(plany);
	fftw_execute(planz);

	//at this stage, ux contains the Fourier transform of uin[0]
	//at this stage, uy contains the Fourier transform of uin[1]
	//at this stage, uz contains the Fourier transform of uin[2]

#ifndef TEST_SOL
	for(int i=0;i<nx_local;++i)
		for(int j=0;j<ny;++j)
		{
			//first, positive kz
			for(int k=0;k<nz;++k)
			{

				//x frequency
				if(i+nx_local_start<nx/2)
				{
					kx = ((double) (i + nx_local_start));
				}else{
					kx = -nx + ((double) (i + nx_local_start));
				}
				if(j<ny/2)
				{
					ky = ((double) j);
				}else{
					ky = -ny + ((double) j);
				}
				if(k<nz/2)
				{
					kz = ((double) k);
				}else{
					kz = -nz + ((double) k);
				}


				kx *= (2*M_PI/L);
				ky *= (2*M_PI/L);
				kz *= (2*M_PI/L);

				//magnitude of k vector
				kk = sqrt(kx*kx + ky*ky + kz*kz);

				//index of uk corresponding to k

				ijk = grid_complex_ijk(i,j,k,grid_info);


				// need to keep \hat{k}(\hat{k} dot U)
				if(i+nx_local_start==0 && j==0 && k==0)
				{
					
					ukx[ijk][0] = ux[ijk][0] * scale;
					ukx[ijk][1] = ux[ijk][1] * scale;
				
					uky[ijk][0] = uy[ijk][0] * scale; 
					uky[ijk][1] = uy[ijk][1] * scale;

					ukz[ijk][0] = uz[ijk][0] * scale; 
					ukz[ijk][1] = uz[ijk][1] * scale;
				
				}else{

					//find U - khat*(khat dot U)
					ukx[ijk][0] = ux[ijk][0]*(1. - pow(kx/kk,2)) * scale; 
					ukx[ijk][1] = ux[ijk][1]*(1. - pow(kx/kk,2)) * scale;
				
					uky[ijk][0] = uy[ijk][0]*(1. - pow(ky/kk,2)) * scale;  
					uky[ijk][1] = uy[ijk][1]*(1. - pow(ky/kk,2)) * scale;  

					ukz[ijk][0] = uz[ijk][0]*(1. - pow(kz/kk,2)) * scale; 
					ukz[ijk][1] = uz[ijk][1]*(1. - pow(kz/kk,2)) * scale; 

				}
			}
		}
#else /*TEST_SOL*/
  for(int i=0;i<nx_local;++i)
    for(int j=0;j<ny;++j)
    {
      for(int k=0;k<nz;++k)
      {
	//x frequency
	if(i+nx_local_start<nx/2)
	{
	  kx = ((double) (i + nx_local_start));
	}else{
	  kx = -nx + ((double) (i + nx_local_start));
	}
	if(j<ny/2)
	{
	  ky = ((double) j);
	}else{
	  ky = -ny + ((double) j);
	}
	if(k<nz/2)
	{
	  kz = ((double) k);
	}else{
	  kz = -nz + ((double) k);
	}

	//magnitude of k vector
	kk = sqrt(kx*kx + ky*ky + kz*kz);

	kx /= kk;
	ky /= kk;
	kz /= kk;



	//index of uk corresponding to k
	ijk = grid_complex_ijk(i,j,k,grid_info);


	// need to keep \hat{k}(\hat{k} dot U)
	if(i+nx_local_start==0 && j==0 && k==0)
	{
					
	  ukx[ijk][0] = 0;
	  ukx[ijk][1] = 0;
	  uky[ijk][0] = 0;
	  uky[ijk][1] = 0;
	  ukz[ijk][0] = 0;
	  ukz[ijk][1] = 0;
				
	}else{

	  //find khat dot U
	  ukx[ijk][0] = ux[ijk][0] - (ux[ijk][0]*kx + uy[ijk][0]*ky + uz[ijk][0]*kz)*kx;
	  ukx[ijk][1] = ux[ijk][1] - (ux[ijk][1]*kx + uy[ijk][1]*ky + uz[ijk][1]*kz)*kx; 
	
	  uky[ijk][0] = uy[ijk][0] - (ux[ijk][0]*kx + uy[ijk][0]*ky + uz[ijk][0]*kz)*ky;
	  uky[ijk][1] = uy[ijk][1] - (ux[ijk][1]*kx + uy[ijk][1]*ky + uz[ijk][1]*kz)*ky;

	  ukz[ijk][0] = uz[ijk][0] - (ux[ijk][0]*kx + uy[ijk][0]*ky + uz[ijk][0]*kz)*kz;
	  ukz[ijk][1] = uz[ijk][1] - (ux[ijk][1]*kx + uy[ijk][1]*ky + uz[ijk][1]*kz)*kz;

					//need to project along khat
	  ukx[ijk][0] *= scale;
	  ukx[ijk][1] *= scale;
	
	  uky[ijk][0] *= scale;
	  uky[ijk][1] *= scale;

	  ukz[ijk][0] *= scale;
	  ukz[ijk][1] *= scale;
	
	}
      }
    }
#endif /*TEST_SOL*/

	//perform the inverse transform of the derivative
	fftw_execute(iplanx);
	fftw_execute(iplany);
	fftw_execute(iplanz);

	//copy inverse transform into derivative
	grid_copy_complex_to_real_in_place(ukx, sol[0], grid_info);
	grid_copy_complex_to_real_in_place(uky, sol[1], grid_info);
	grid_copy_complex_to_real_in_place(ukz, sol[2], grid_info);

	//free the buffer memory
	fftw_free(ux);
	fftw_free(uy);
	fftw_free(uz);
	fftw_free(ukx);
	fftw_free(uky);
	fftw_free(ukz);

	//destroy the plans
	fftw_destroy_plan(planx);
	fftw_destroy_plan(plany);
	fftw_destroy_plan(planz);
	fftw_destroy_plan(iplanx);
	fftw_destroy_plan(iplany);
	fftw_destroy_plan(iplanz);

	//return the answer
	return sol;

}

//grid energies

/*! \fn double grid_total_specific_energy(double *u, FFTW_Grid_Info grid_info, MPI_Comm world)
 *  \brief Calculate the total specific energy E = 1/2 \int u^2 dV on a grid u */
double grid_total_specific_energy(double *u, FFTW_Grid_Info grid_info, MPI_Comm world) 
{
	//total specific energy
	double total_specific_energy = 0;

	//specific energy
	double *specific_energy;

	//find the specific energy
	specific_energy = grid_power(2, u, grid_info);

	//find the total specific energy
	total_specific_energy = 0.5*grid_volume_integral(specific_energy, grid_info, world);

	//free the specific energy
	fftw_free(specific_energy);

	//return the total specific energy = 0.5 \int u^2 dV
	return total_specific_energy;
}

/*! \fn double *grid_field_specific_energy(double **u, FFTW_Grid_Info grid_info)
 *  \brief Calculate the field specific energy = 1/2 |u|^2 */
double *grid_field_specific_energy(double **u, FFTW_Grid_Info grid_info)
{
	//specific energy
	double *specific_energy;

	//find specific energy as epsilon = u dot u
	specific_energy = grid_field_dot_product(u, u, grid_info);

	//multiply by 0.5
	grid_rescale(0.5, specific_energy, grid_info);

	//return the answer
	return specific_energy;
}

/*! \fn double grid_field_total_specific_energy(double **u, FFTW_Grid_Info grid_info, MPI_Comm world) 
 *  \brief Calculate the total specific energy E = 1/2 \int u^2 dV on a field u */
double grid_field_total_specific_energy(double **u, FFTW_Grid_Info grid_info, MPI_Comm world) 
{
	//total specific energy
	double total_specific_energy = 0;

	//specific energy
	double *specific_energy;

	//find the specific energy
	specific_energy = grid_field_specific_energy(u, grid_info);

	//find the total specific energy
	total_specific_energy = grid_volume_integral(specific_energy, grid_info, world);

	//free the specific energy
	fftw_free(specific_energy);

	//return the total specific energy = 0.5 \int u^2 dV
	return total_specific_energy;
}

/*! \fn double *grid_field_specific_dilatational_energy(double **u, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world)
 *  \brief Calculate the specific energy E = 1/2 |u_D|^2 in the dilatational component of the velocity field. */
double *grid_field_specific_dilatational_energy(double **u, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world)
{
	//specific energy
	double *specific_energy;

	//dilatational field
	double **u_dilatational;

	//find the dilatational component
	u_dilatational = grid_field_dilatational_component(u, grid_info, myid, numprocs, world);

	//find specific energy as epsilon = u dot u
	specific_energy = grid_field_dot_product(u_dilatational, u_dilatational, grid_info);

	//multiply by 0.5

	grid_rescale(0.5, specific_energy, grid_info);

	//free the buffer memory
	deallocate_field_fftw_grid(u_dilatational,grid_info.ndim,grid_info);

	//return the answer
	return specific_energy;
}

//specific energies

/*! \fn double grid_field_total_specific_dilatational_energy(double **u, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world)
 *  \brief Calculate the total specific energy E = 1/2 \int u_D^2 dV in the dilatational component of the velocity field u. */
double grid_field_total_specific_dilatational_energy(double **u, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world)
{

	//total specific energy
	double total_specific_energy = 0;

	//specific energy
	double *specific_energy;

	//find the specific energy
	specific_energy = grid_field_specific_dilatational_energy(u, grid_info, myid, numprocs, world);

	//find the total specific energy
	total_specific_energy = grid_volume_integral(specific_energy, grid_info, world);

	//free the specific energy
	fftw_free(specific_energy);

	//return the total specific dilatational energy = 0.5 \int u_D^2 dV
	return total_specific_energy;
}

/*! \fn double *grid_field_specific_solenoidal_energy(double **u, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world)
 *  \brief Calculate the specific energy E = 1/2 |u_S|^2 in the solenoidal component of the velocity field. */
double *grid_field_specific_solenoidal_energy(double **u, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world)
{
	//specific energy
	double *specific_energy;

	//solenoidal field
	double **u_solenoidal;

	//find the solenoidal component
	u_solenoidal = grid_field_solenoidal_component(u, grid_info, myid, numprocs, world);

	//find specific energy as epsilon = u dot u
	specific_energy = grid_field_dot_product(u_solenoidal, u_solenoidal, grid_info);

	//multiply by 0.5
	grid_rescale(0.5, specific_energy, grid_info);

	//free the buffer memory
	deallocate_field_fftw_grid(u_solenoidal,grid_info.ndim,grid_info);

	//return the answer
	return specific_energy;
}


/*! \fn double grid_field_total_specific_solenoidal_energy(double **u, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world)
 *  \brief Calculate the total specific energy E = 1/2 \int u_D^2 dV in the solenoidal component of the velocity field u. */
double grid_field_total_specific_solenoidal_energy(double **u, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world)
{
	//total specific energy
	double total_specific_energy = 0;

	//specific energy
	double *specific_energy;

	//find the specific energy
	specific_energy = grid_field_specific_solenoidal_energy(u, grid_info, myid, numprocs, world);

	//find the total specific energy
	total_specific_energy = grid_volume_integral(specific_energy, grid_info, world);

	//free the specific energy
	fftw_free(specific_energy);

	//return the total specific solenoidal energy = 0.5 \int u_D^2 dV
	return total_specific_energy;
}

//dissipation rates

/*! \fn double dissipation_rate_dilatational(double **u, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world)
 *  \brief Calculate the dilatational contribution to the dissipation rate. */
double dissipation_rate_dilatational(double **u, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world)
{
	double *u_div;
	double **u_curl;

	double *u_curl_abs;
	double *u_div_abs;

	double u_div_ave;
	double u_curl_ave;
	double epsilon_D;

	//calculate divergence and curl
	u_div  = grid_field_divergence(u, grid_info, myid, numprocs, world);
	u_curl = grid_field_curl(u, grid_info, myid, numprocs, world);

	//find |curl u|^2
	u_curl_abs = grid_field_dot_product(u_curl, u_curl, grid_info);

	//find |div u|^2
	u_div_abs = grid_power(2, u_div, grid_info);

	u_div_ave  = grid_mean(u_div_abs,  grid_info, world);
	u_curl_ave = grid_mean(u_curl_abs, grid_info, world);

	//free buffer memory
	deallocate_field_fftw_grid(u_curl,grid_info.ndim,grid_info);
	fftw_free(u_curl_abs);
	fftw_free(u_div);
	fftw_free(u_div_abs);

	//eqn 11 of kritsuk
	epsilon_D = (4./3.) * u_div_ave/( u_curl_ave + (4./3.) * u_div_ave);

	//return the answer
	return epsilon_D;
}

/*! \fn double dissipation_rate_solenoidal(double **u, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world)
 *  \brief Calculate the solenoidal contribution to the dissipation rate. */
double dissipation_rate_solenoidal(double **u, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world)
{
	double *u_div;
	double **u_curl;

	double *u_curl_abs;
	double *u_div_abs;

	double u_div_ave;
	double u_curl_ave;
	double epsilon_S;

	//calculate divergence and curl
	u_div  = grid_field_divergence(u, grid_info, myid, numprocs, world);
	u_curl = grid_field_curl(u, grid_info, myid, numprocs, world);

	//find |curl u|^2
	u_curl_abs = grid_field_dot_product(u_curl, u_curl, grid_info);

	//find |div u|^2
	u_div_abs = grid_power(2, u_div, grid_info);

	u_div_ave  = grid_mean(u_div_abs,  grid_info, world);
	u_curl_ave = grid_mean(u_curl_abs, grid_info, world);

	//free buffer memory
	deallocate_field_fftw_grid(u_curl,grid_info.ndim,grid_info);
	fftw_free(u_curl_abs);
	fftw_free(u_div);
	fftw_free(u_div_abs);

	//eqn 11 of kritsuk
	epsilon_S = u_curl_ave/( u_curl_ave + (4./3.) * u_div_ave);

	//return the answer
	return epsilon_S;
}

/*! \fn double small_scale_compressive_ratio(double **u, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world)
 *  \brief Calculate the small-scale compressive ratio <|div u|^2> / ( <|div u|^2> + <|curl u|^2> ).  */
double small_scale_compressive_ratio(double **u, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world, double *epsilon_S, double *epsilon_D)
{
	double *u_div;		//divergence
	double **u_curl;	//solenoidal

	double *u_curl_abs;	//abs(curl^2)
	double *u_div_abs;	//abs(div^2)

	double u_div_ave;	//ave(abs(curl^2))
	double u_curl_ave;	//ave(abs(div^2))
	double rcs;

	int i,j,k,ijk;
	int nx_local = grid_info.nx_local;
	int nx       = grid_info.nx;
	int ny       = grid_info.ny;
	int nz       = grid_info.nz;

/*
	double u_tot=0;
	double u_dil_tot=0;
	double u_sol_tot=0;
	double u_both_tot=0;


	//dilatational field
	double **u_dilatational;

	//find the dilatational component
	u_dilatational = grid_field_dilatational_component(u, grid_info, myid, numprocs, world);

	//solenoidal field
	double **u_solenoidal;

	//find the solenoidal component
	u_solenoidal = grid_field_solenoidal_component(u, grid_info, myid, numprocs, world);
	
	//set the grid to value
	for(i=0;i<nx_local;++i)
		for(j=0;j<ny;++j)
			for(k=0;k<nz;++k)
			{	
				//get grid index
				ijk = grid_ijk(i,j,k,grid_info);

				//set grid elements to value
				u_tot += u[0][ijk]*u[0][ijk] + u[1][ijk]*u[1][ijk] + u[2][ijk]*u[2][ijk];
				u_dil_tot += u_dilatational[0][ijk]*u_dilatational[0][ijk] + u_dilatational[1][ijk]*u_dilatational[1][ijk] + u_dilatational[2][ijk]*u_dilatational[2][ijk];
				u_sol_tot += u_solenoidal[0][ijk]*u_solenoidal[0][ijk] + u_solenoidal[1][ijk]*u_solenoidal[1][ijk] + u_solenoidal[2][ijk]*u_solenoidal[2][ijk];
				u_both_tot += pow(u_solenoidal[0][ijk]+u_dilatational[0][ijk],2);
				u_both_tot += pow(u_solenoidal[1][ijk]+u_dilatational[1][ijk],2);
				u_both_tot += pow(u_solenoidal[2][ijk]+u_dilatational[2][ijk],2);
			}

	printf("myid %d u_tot %e u_dil %e u_sol %e u_both %e\n",myid,u_tot,u_dil_tot,u_sol_tot,u_both_tot);
*/

	//calculate divergence and curl
	u_div  = grid_field_divergence(u, grid_info, myid, numprocs, world);
	u_curl = grid_field_curl(u, grid_info, myid, numprocs, world);

	//u_div  = grid_field_divergence(u_dilatational, grid_info, myid, numprocs, world);
	//u_curl = grid_field_curl(u_solenoidal, grid_info, myid, numprocs, world);
	//u_div  = grid_field_divergence(u_solenoidal, grid_info, myid, numprocs, world);
	//u_curl = grid_field_curl(u_dilatational, grid_info, myid, numprocs, world);

/*
	u_dil_tot = 0.0;
	u_sol_tot = 0.0;
	//set the grid to value
	for(i=0;i<nx_local;++i)
		for(j=0;j<ny;++j)
			for(k=0;k<nz;++k)
			{	
				//get grid index
				ijk = grid_ijk(i,j,k,grid_info);

				//set grid elements to value
				u_dil_tot += u_div[ijk]*u_div[ijk];
				u_sol_tot += u_curl[0][ijk]*u_curl[0][ijk] + u_curl[1][ijk]*u_curl[1][ijk] + u_curl[2][ijk]*u_curl[2][ijk];
			}

	printf("u_dil_tot = %e u_sol_tot = %e\n",u_dil_tot,u_sol_tot);
*/


	//find |curl u|^2
	u_curl_abs = grid_field_dot_product(u_curl, u_curl, grid_info);

	//find |div u|^2
	u_div_abs = grid_power(2, u_div, grid_info);
	u_div_ave  = grid_mean(u_div_abs,  grid_info, world);
	u_curl_ave = grid_mean(u_curl_abs, grid_info, world);

	//free buffer memory
	deallocate_field_fftw_grid(u_curl,grid_info.ndim,grid_info);
	//deallocate_field_fftw_grid(u_dilatational,grid_info.ndim,grid_info);
	//deallocate_field_fftw_grid(u_solenoidal,grid_info.ndim,grid_info);
	fftw_free(u_curl_abs);
	fftw_free(u_div);
	fftw_free(u_div_abs);

	//eqn 12 of kritsuk
	rcs = u_div_ave/( u_div_ave + u_curl_ave);


	*epsilon_S = u_curl_ave;
	*epsilon_D = u_div_ave;

	//return the answer
	return rcs;
}

//enstrophy and denstrophy

/*! \fn double *grid_field_enstrophy(double **u, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world)
 *  \brief Calculate the field enstrophy = 1/2 |nabla cross u|^2 = 1/2 |vorticity|^2. */
double *grid_field_enstrophy(double **u, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world)
{
	//enstrophy
	double *enstrophy;

	//vorticity
	double **vorticity;

	//find vorticity as omega = nabla cross u
	vorticity = grid_field_vorticity(u, grid_info, myid, numprocs, world);

	//find enstrophy as Omega = vorticity dot vorticity
	enstrophy = grid_field_dot_product(vorticity, vorticity, grid_info);

	//free the vorticity
	deallocate_field_fftw_grid(vorticity,grid_info.ndim,grid_info);

	//multiply by 0.5
	grid_rescale(0.5, enstrophy, grid_info);

	//return the answer enstrophy = 1/2 |nabla cross u|^2
	return enstrophy;
}

/*! \fn double grid_field_total_enstrophy(double **u, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world)
 *  \brief Calculate the field's total enstrophy = 1/2 \int |nabla cross u|^2 dV = 1/2 \int |vorticity|^2 dV. */
double grid_field_total_enstrophy(double **u, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world)
{
	//total enstrophy
	double total_enstrophy;

	//enstrophy
	double *enstrophy;

	//find the enstrophy
	enstrophy = grid_field_enstrophy(u, grid_info, myid, numprocs, world);

	//find the total enstrophy
	total_enstrophy = grid_volume_integral(enstrophy, grid_info, world);

	//free the enstrophy
	fftw_free(enstrophy);

	//return the total enstrophy
	return total_enstrophy;
}


/*! \fn double *grid_field_denstrophy(double *rho, double **u, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world)
 *  \brief Calculate the field denstrophy = 1/2 |nabla cross {sqrt(rho) u}|^2/rho. */
double *grid_field_denstrophy(double *rho, double **u, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world)
{
	//denstrophy
	double *denstrophy;

	//sqrt rho
	double *sqrt_rho;

	//sqrt_rho_u
	double **sqrt_rho_u;
	double **curl_sqrt_rho_u;

	//find sqrt rho
	sqrt_rho = grid_power(0.5, rho, grid_info);

	//find the field sqrt(rho) * u
	sqrt_rho_u = grid_field_product(sqrt_rho, u, grid_info);

	//find denstrophy as Omega = 1/2 | nabla cross {sqrt(rho) u}|^2/rho
	curl_sqrt_rho_u = grid_field_curl(sqrt_rho_u, grid_info, myid, numprocs, world);
	denstrophy      = grid_field_dot_product(curl_sqrt_rho_u, curl_sqrt_rho_u, grid_info);

	//rescale by 1./rho
	grid_quotient_in_place(denstrophy, rho, grid_info);

	//free the field sqrt_rho_u
	deallocate_field_fftw_grid(sqrt_rho_u,      grid_info.ndim, grid_info);
	deallocate_field_fftw_grid(curl_sqrt_rho_u, grid_info.ndim, grid_info);
	
	//free sqrt_rho
	fftw_free(sqrt_rho);

	//multiply by 0.5
	grid_rescale(0.5, denstrophy, grid_info);

	//return the answer denstrophy = 1/2 |nabla cross sqrt(rho)u|^2/rho
	return denstrophy;
}

//tensor operations

/*! \fn double ***grid_transform_tensor(double ***sigma, double ***a, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world)
 *  \brief Transform a tensor using another tensor. */
double ***grid_transform_tensor(double ***sigma_in, double ***a_in, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world)
{
	//number of dimensions
	int ndim = grid_info.ndim;
	int total_local_size = grid_info.n_local_real_size;

	//the tensor transformation law is:
	// sigma'_ij = a sigma a^T

	double ***tensor_transform;

	char tensor_name[200];
	double **x;
	double **sigma;
	double **a;

	//tensors for place by place transformation
	sigma = two_dimensional_array(ndim,ndim);
	a     = two_dimensional_array(ndim,ndim);


	//allocate the transformed tensor
	tensor_transform = allocate_tensor_fftw_grid(grid_info);

	//perform the transformation
	//first, do sigma a^T
	for(int k=0;k<grid_info.n_local_real_size;k++)
	{
		for(int i=0;i<grid_info.ndim;i++)
			for(int j=0;j<grid_info.ndim;j++)
			{
				sigma[i][j] = sigma_in[i][j][k];
				a[i][j]     = a_in[i][j][k];
			}

		//perform the transformation
		x = tensor_transformation(a, sigma, ndim);

		//stor the result
		for(int i=0;i<grid_info.ndim;i++)
			for(int j=0;j<grid_info.ndim;j++)
				tensor_transform[i][j][k] = x[i][j];
		//free the transformation
		deallocate_two_dimensional_array(x,ndim,ndim);
	}
	deallocate_two_dimensional_array(sigma,ndim,ndim);
	deallocate_two_dimensional_array(a,ndim,ndim);


	//return the result
	return tensor_transform;
}


/*! \fn double ***grid_velocity_gradient_tensor(double **u, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world)
 *  \brief Calculate the components of the velocity gradient tensor. */
double ***grid_velocity_gradient_tensor(double **u, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world)
{
	//number of dimensions
	int ndim = grid_info.ndim;
	int total_local_size = grid_info.n_local_real_size;

	//velocity gradient tensor
	double ***velocity_gradient;

	//velocity derivative
	double *dui_dxj;

	//allocate the velocity graident tensor
	velocity_gradient = allocate_tensor_fftw_grid(grid_info);

	//calculate the velocity derivatives in the i and j
	for(int i=0;i<ndim;i++)
		for(int j=0;j<ndim;j++)
		{
			//compute the velocity gradients
			dui_dxj = grid_derivative(u[i], j, grid_info, myid, numprocs, world);

			//velocity gradient is just dui_dxj
			for(int k=0;k<total_local_size;k++)
			{
				velocity_gradient[i][j][k]   = dui_dxj[k];
			}

			//free the velocity gradients
			free(dui_dxj);
		}

	//return the velocity gradient
	return velocity_gradient;
}

/*! \fn double **grid_convective_derivative(double **u, double **v, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world)
 *  \brief Calculate the convective derivative of a vector field. */
double **grid_convective_derivative(double **u, double **v, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world)
{
	//number of dimensions
	int ndim = grid_info.ndim;
	int total_local_size = grid_info.n_local_real_size;


	//convective 
	double **convective_derivative;

	//vector gradient tensor
	double ***vector_gradient;

	//velocity derivative
	double *dui_dxj;

	//allocate the vector graident tensor
	vector_gradient = allocate_tensor_fftw_grid(grid_info);

	//calculate the vector derivatives in the i and j
	for(int i=0;i<ndim;i++)
		for(int j=0;j<ndim;j++)
		{
			//compute the velocity gradients
			dui_dxj = grid_derivative(u[i], j, grid_info, myid, numprocs, world);

			//vector gradient is just dui_dxj
			for(int k=0;k<total_local_size;k++)
			{
				vector_gradient[i][j][k]   = dui_dxj[k];
			}

			//free the vector gradients
			free(dui_dxj);
		}

	//at this point we have the vector gradient of u
	//need to take dot product with v

	//find the field-tensor product
	convective_derivative = grid_field_tensor_product(vector_gradient, v, grid_info, myid, numprocs, world);

	//free tensor
	deallocate_tensor_fftw_grid(vector_gradient, grid_info);
	
	//return the convective_derivative
	return convective_derivative;
}
double **grid_field_tensor_product(double ***u, double **v, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world)
{
	//number of dimensions
	int ndim = grid_info.ndim;
	int total_local_size = grid_info.n_local_real_size;

    double **ft_product;

    //allocate the field_tensor_product
    ft_product = allocate_field_fftw_grid(grid_info.ndim, grid_info);

    //find the dot product
    for(int k=0;k<total_local_size;k++)
    {
	for(int i=0;i<ndim;i++)
	{

	    ft_product[i][k] = 0;

	    for(int j=0;j<ndim;j++)
	    {

		ft_product[i][k] += v[j][k] * u[j][i][k];
	    }

	}
    }

    //return the field-tensor
    return ft_product;
}

/*! \fn double ***grid_strain_tensor(double **u, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world)
 *  \brief Calculate the components of the rate of strain tensor. */
double ***grid_strain_tensor(double **u, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world)
{
	//number of dimensions
	int ndim = grid_info.ndim;
	int total_local_size = grid_info.n_local_real_size;

	//strain tensor
	double ***strain;

	//velocity derivatives
	double *dui_dxj;
	double *duj_dxi;

	//allocate the rate of strain tensor
	strain = allocate_tensor_fftw_grid(grid_info);

	//calculate the velocity derivatives in the i and j
	for(int i=0;i<ndim;i++)
		for(int j=0;j<ndim;j++)
		{

			//compute the velocity gradients
			dui_dxj = grid_derivative(u[i], j, grid_info, myid, numprocs, world);
			duj_dxi = grid_derivative(u[j], i, grid_info, myid, numprocs, world);


			//strain = 0.5*( dui_dxj + dvj_dx)
			for(int k=0;k<total_local_size;k++)
			{
				strain[i][j][k]    = 0.5*(dui_dxj[k] + duj_dxi[k]); 
			}

			//free the velocity gradients
			free(dui_dxj);
			free(duj_dxi);
		}

	//return the strain 
	return strain;
}

/*! \fn double ***grid_shear_tensor(double **u, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world)
 *  \brief Calculate the components of the shear tensor. */
double ***grid_shear_tensor(double **u, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world)
{
	//number of dimensions
	int ndim = grid_info.ndim;
	int total_local_size = grid_info.n_local_real_size;

	//velocity shear tensor
	double ***shear;

	//divergence
	double *divergence;

	//velocity derivatives
	double *dui_dxj;
	double *duj_dxi;

	//allocate the velocity shear tensor
	shear = allocate_tensor_fftw_grid(grid_info);

	//calculate the divergence
	divergence = grid_field_divergence(u, grid_info, myid, numprocs, world);

	//calculate the velocity derivatives in the i and j
	for(int i=0;i<ndim;i++)
		for(int j=0;j<ndim;j++)
		{
			//compute the velocity gradients
			dui_dxj = grid_derivative(u[i], j, grid_info, myid, numprocs, world);
			duj_dxi = grid_derivative(u[j], i, grid_info, myid, numprocs, world);


			//add 0.5*( dui_dxj + dvj_dx) - 1/3 div v 
			for(int k=0;k<total_local_size;k++)
			{
				shear[i][j][k] = 0.5*(dui_dxj[k] + duj_dxi[k]); 
			}
			if(i==j)
				for(int k=0;k<total_local_size;k++)
					shear[i][j][k] -= divergence[k]/3.;

			//free the velocity gradients
			free(dui_dxj);
			free(duj_dxi);
		}

	//free the divergence
	free(divergence);

	//return the shear
	return shear;
}

/*! \fn double ***grid_rotation_tensor(double **u, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world)
 *  \brief Calculate the components of the rotation tensor. */
double ***grid_rotation_tensor(double **u, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world)
{
	//number of dimensions
	int ndim = grid_info.ndim;
	int total_local_size = grid_info.n_local_real_size;

	//velocity rotation tensor
	double ***rotation;

	//velocity derivatives
	double *dui_dxj;
	double *duj_dxi;

	//allocate the velocity rotation tensor
	rotation = allocate_tensor_fftw_grid(grid_info);

	//calculate the velocity derivatives in the i and j
	for(int i=0;i<ndim;i++)
		for(int j=0;j<ndim;j++)
		{
			//compute the velocity gradients
			dui_dxj = grid_derivative(u[i], j, grid_info, myid, numprocs, world);
			duj_dxi = grid_derivative(u[j], i, grid_info, myid, numprocs, world);


			//r_ij = 0.5*( dui_dxj - dvj_dx)
			for(int k=0;k<total_local_size;k++)
			{
				rotation[i][j][k] = 0.5*(dui_dxj[k] - duj_dxi[k]);
			}

			//free the velocity gradients
			free(dui_dxj);
			free(duj_dxi);
		}

	//return the rotation
	return rotation;
}


double *grid_tensor_determinant(double ***A, FFTW_Grid_Info grid_info)
{
	int ijk;
	int ndim = grid_info.ndim;
	int total_local_size = grid_info.n_local_real_size;
	double **a;
	double *det;

	//local tensor
	a = two_dimensional_array(ndim,ndim);

	//determinant
	det = allocate_real_fftw_grid(grid_info);

	for(int k=0;k<total_local_size;k++)
	{
		for(int i=0;i<ndim;i++)
			for(int j=0;j<ndim;j++)
			{
				//get the local tensor
				a[i][j] = A[i][j][k];

			}
		
		det[k] = matrix_determinant(a,ndim);	
	}

	deallocate_two_dimensional_array(a, ndim, ndim);

	return det;
}


//grid interpolation

/*! \fn double *grid_first_slice(double *f, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world);
 *  \brief First y-z slice of slab from right neighbor */
double *grid_first_slice(double *f, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world)
{
	int ijk;
	int ijkp;
	double *fu;	//slice to share
	int source;
	int dest;

	MPI_Status status;


	//allocate the slice
	fu = calloc_double_array(grid_info.ny*grid_info.nz);


	//load info into the slice
	for(int j=0;j<grid_info.ny;j++)
		for(int k=0;k<grid_info.nz;k++)
		{
			ijk  = grid_ijk(0,j,k,grid_info);
			ijkp = grid_info.nz*j + k;

			fu[ijkp] = f[ijk];
		}

	//share the slice
	source = myid + 1;
	if(source>=numprocs)
		source -= numprocs;
	dest = myid - 1;
	if(dest<0)
		dest += numprocs;

	//send data
	MPI_Sendrecv_replace(fu, grid_info.ny*grid_info.nz, MPI_DOUBLE, dest, myid, source, source, world, &status);

	//return the slice
	return fu;
}


/*! \fn double *grid_cubic_lower_slice(double *f, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world);
 *  \brief Lower y-z slice of slab from left neighbor */
double *grid_cubic_lower_slice(double *f, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world)
{
	int ijk;
	int ijkp;
	double *fl;	//slice to share
	int source;
	int dest;

	MPI_Status status;


	//allocate the slice
	fl = calloc_double_array(grid_info.ny*grid_info.nz);


	//load info into the slice
	for(int j=0;j<grid_info.ny;j++)
		for(int k=0;k<grid_info.nz;k++)
		{
			ijk  = grid_ijk(grid_info.nx_local-1,j,k,grid_info);
			ijkp = grid_info.nz*j + k;

			fl[ijkp] = f[ijk];
		}

	//share the slice
	source = myid - 1;
	if(source<0)
		source += numprocs;
	dest = myid + 1;
	if(dest>=numprocs)
		dest -= numprocs;

	//send data
	MPI_Sendrecv_replace(fl, grid_info.ny*grid_info.nz, MPI_DOUBLE, dest, myid, source, source, world, &status);

	//return the slice
	return fl;
}

/*! \fn double *grid_cubic_upper_slice(double *f, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world);
 *  \brief Upper y-z slice of slab from left neighbor */
double *grid_cubic_upper_slice(double *f, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world)
{
	int ijk;
	int ijkp;
	double *fu;	//slice to share
	int source;
	int dest;

	MPI_Status status;


	//allocate the slice
	fu = calloc_double_array(3*grid_info.ny*grid_info.nz);


	//load info into the slice
	for(int i=0;i<3;i++)
	for(int j=0;j<grid_info.ny;j++)
		for(int k=0;k<grid_info.nz;k++)
		{
			ijk  = grid_ijk(i,j,k,grid_info);
			ijkp = grid_info.nz*grid_info.ny*i + grid_info.nz*j + k;

			fu[ijkp] = f[ijk];
		}

	//share the slice
	source = myid + 1;
	if(source>=numprocs)
		source -= numprocs;
	dest = myid - 1;
	if(dest<0)
		dest += numprocs;

	//send data
	MPI_Sendrecv_replace(fu, grid_info.ny*grid_info.nz, MPI_DOUBLE, dest, myid, source, source, world, &status);

	//return the slice
	return fu;
}

/*! \fn double grid_interpolation(double x, double y, double z, double *f, double *fu, FFTW_Grid_Info grid_info)
 *  \brief Trilinear interpolation on the grid */
double grid_interpolation(double x, double y, double z, double *f, double *fu, FFTW_Grid_Info grid_info)
{
	//f      is the grid
	//fu     is the grid slice just right of the slab

	int ijk;	//grid index of the location
	int ijkp;	//grid index of the location

	//find grid index
	int i;
	int j;
	int k;
	int jp1, kp1;

	//coefficients
	double c00;
	double c10;
	double c01;
	double c11;
	double c0;
	double c1;
	double c;

	double c000;
	double c010;
	double c001;
	double c011;
	double c100;
	double c110;
	double c101;
	double c111;

	double xd;
	double yd;
	double zd;

	int i000;
	int i010;
	int i001;
	int i011;
	int i100;
	int i110;
	int i101;
	int i111;

	double xx;
	double yy;
	double zz;

/*
	//centered on cell edges
	//wrap x-direction
	xx = x;
	if(xx>=grid_info.BoxSize)
		xx -= grid_info.BoxSize;
	if(xx<0)
		xx += grid_info.BoxSize;

	//wrap y-direction
	yy = y;
	if(yy>=grid_info.BoxSize)
		yy -= grid_info.BoxSize;
	if(yy<0)
		yy += grid_info.BoxSize;

	//wrap z-direction
	zz = z;
	if(zz>=grid_info.BoxSize)
		zz -= grid_info.BoxSize;
	if(zz<0)
		zz += grid_info.BoxSize;

	i = (int) (xx/grid_info.dx) - grid_info.nx_local_start;	//integer index along x direction
	j = (int) (yy/grid_info.dy);					//integer index along y direction
	k = (int) (zz/grid_info.dz);					//integer index along z direction

	//xd = (xx - (i + grid_info.nx_local_start)*grid_info.dx)/grid_info.dx;
	//yd = (yy - j*grid_info.dy)/grid_info.dy;
	//zd = (zz - k*grid_info.dz)/grid_info.dz;
	//xd = fmod(xx,grid_info.dx);
	//yd = fmod(yy,grid_info.dy);
	//zd = fmod(zz,grid_info.dz);
	xd = xx/grid_info.dx - ((double) ((long) (xx/grid_info.dx)));
	yd = yy/grid_info.dy - ((double) ((long) (yy/grid_info.dy)));
	zd = zz/grid_info.dz - ((double) ((long) (zz/grid_info.dz)));
*/

	//centered on cell centers

	//wrap x-direction
	xx = x;
	if(xx>=grid_info.BoxSize)
		xx -= grid_info.BoxSize;
	if(xx<0)
		xx += grid_info.BoxSize;

	//wrap y-direction
	yy = y;
	if(yy>=grid_info.BoxSize)
		yy -= grid_info.BoxSize;
	if(yy<0)
		yy += grid_info.BoxSize;

	//wrap z-direction
	zz = z;
	if(zz>=grid_info.BoxSize)
		zz -= grid_info.BoxSize;
	if(zz<0)
		zz += grid_info.BoxSize;

	i = (int) ((xx-0.5*grid_info.dx)/grid_info.dx) - grid_info.nx_local_start;		//integer index along x direction
	j = (int) ((yy-0.5*grid_info.dy)/grid_info.dy);					//integer index along y direction
	k = (int) ((zz-0.5*grid_info.dz)/grid_info.dz);					//integer index along z direction

	//find the fractional distance between location
	//and the previous grid point

	//xd = (xx - 0.5*grid_info.dx)/grid_info.dx - ((double) ((long) (xx/grid_info.dx)));
	//yd = (yy - 0.5*grid_info.dy)/grid_info.dy - ((double) ((long) (yy/grid_info.dy)));
	//zd = (zz - 0.5*grid_info.dz)/grid_info.dz - ((double) ((long) (zz/grid_info.dz)));
	xd = (xx - 0.5*grid_info.dx)/grid_info.dx - ((double) (i + grid_info.nx_local_start));
	yd = (yy - 0.5*grid_info.dy)/grid_info.dy - ((double) j);
	zd = (zz - 0.5*grid_info.dz)/grid_info.dz - ((double) k);

	//if the position is not within this slab, then
	//return -1
	if(i < 0 || i >= grid_info.nx_local)
		return 0;


	//if the position does not need 
	//info from another processor
	if(i!=grid_info.nx_local-1)
	{



		//get the ijk of this position
		i000 = grid_ijk(i,  j,k,grid_info);	
		i100 = grid_ijk(i+1,j,k,grid_info);	

		//get coefficient
		c00  = f[i000]*(1 - xd) + f[i100]*xd;

		//get the ijk of this position
		i010 = grid_ijk(i,  j+1, k,grid_info);	
		i110 = grid_ijk(i+1,j+1, k,grid_info);	

		//get coefficient
		c10 = f[i010]*(1 - xd) + f[i110]*xd;

		//get the ijk of this position
		i001 = grid_ijk(i,    j, k+1,grid_info);	
		i101 = grid_ijk(i+1,  j, k+1,grid_info);	

		//get coefficient
		c01 = f[i001]*(1 - xd) + f[i101]*xd;

		//get the ijk of this position
		i011 = grid_ijk(i,    j+1, k+1,grid_info);	
		i111 = grid_ijk(i+1,  j+1, k+1,grid_info);	

		//get coefficient
		c11 = f[i011]*(1 - xd) + f[i111]*xd;

		//get coefficient
		c0 = c00*(1-yd) + c10*yd;
		c1 = c01*(1-yd) + c11*yd;

		//get last coefficient
		c = c0*(1-zd) + c1*zd;

		//printf("xd %e yd %e zd %e c00 %e c01 %e c10 %e c11 %e c0 %e c1 %e c %e\n",xd,yd,zd,c00,c01,c10,c11,c0,c1,c);
		//printf("i %d j %d k %d i000 %d\n",i,j,k,i000);
		//fflush(stdout);
	}else{

		//do wrap if necessary

		if(j+1==grid_info.ny)
		{
			jp1 = 0;
		}else{
			jp1 = j+1;
		}
		if(k+1==grid_info.nz)
		{
			kp1 = 0;
		}else{
			kp1 = k+1;
		}

		//get the ijk of this position
		i000 = grid_ijk(i,  j,k,grid_info);	
		i100 = grid_info.nz*j + k;

		//get coefficient
		c00  = f[i000]*(1 - xd) + fu[i100]*xd;

		//get the ijk of this position
		i010  = grid_ijk(i, jp1, k,grid_info);	
		i110 = grid_info.nz*jp1 + k;

		//get coefficient
		c10 = f[i010]*(1 - xd) + fu[i110]*xd;

		//get the ijk of this position
		i001 = grid_ijk(i,    j, kp1,grid_info);	
		i101 = grid_info.nz*j + kp1;

		//get coefficient
		c01 = f[i001]*(1 - xd) + fu[i101]*xd;

		//get the ijk of this position
		i011 = grid_ijk(i,    jp1, kp1,grid_info);	
		i111 = grid_info.nz*jp1 + kp1;

		//get coefficient
		c11 = f[i011]*(1 - xd) + fu[i111]*xd;

		//get coefficient
		c0 = c00*(1-yd) + c10*yd;
		c1 = c01*(1-yd) + c11*yd;

		//get last coefficient
		c = c0*(1-zd) + c1*zd;


	}

	/*if(c<0)
	{
		printf("C NEG x %e y %e z %e xd %e yd %e zd %e c00 %e c10 %e c01 %e c11 %e c0 %e c1 %e c %e\n",x,y,z,xd,yd,zd,c00,c10,c01,c10,c0,c1,c);
	}*/

	return c;
}

/*! \fn double grid_cubic_log_interpolation(double x, double y, double z, double *f, double *fl, double *fu, FFTW_Grid_Info grid_info)
 *  \brief Tricubic log interpolation on the grid */
double grid_cubic_log_interpolation(double x, double y, double z, double *f, double *fl, double *fu, FFTW_Grid_Info grid_info)
{

  //fl is a single slice
  //fu is three slices

    double s[4][4][4];
    double t[4][4];
    double u[4];
    int ii, jj, kk;
    int ix, jy, kz;

    double xx, yy, zz;

    double f_return;

    int ijk;

	/*
		s(i,j,k) = grid point at ijk
		t(i,j,z) = cint( z, s( i, j,-1), s(i,j,0), s(i,j,1), s(i,j,2) );
		u(i,y,z) = cint( y, t( i,-1, z), t(i,0,z), t(i,1,z), t(i,2,z) );
		f(x,y,z) = cint( x, u(-1, y, z), u(0,y,z), u(1,y,z), u(2,y,z) );


		u(-1, y, z) = cint( y, t(-1, -1, z), t(-1, 0, z), t(-1, 1, z), t(-1, 2, z) );
		u( 0, y, z) = cint( y, t( 0, -1, z), t( 0, 0, z), t( 0, 1, z), t( 0, 2, z) );
		u( 1, y, z) = cint( y, t( 1, -1, z), t( 1, 0, z), t( 1, 1, z), t( 1, 2, z) );
		u( 2, y, z) = cint( y, t( 2, -1, z), t( 2, 0, z), t( 2, 1, z), t( 2, 2, z) );

		t(-1, -1, z) = cint( z, s(-1, -1, -1), s(-1, -1, 0), s(-1, -1, 1), s(-1, -1, 2) );
		t(-1,  0, z) = cint( z, s(-1,  0, -1), s(-1,  0, 0), s(-1,  0, 1), s(-1,  0, 2) );
		t(-1,  1, z) = cint( z, s(-1,  1, -1), s(-1,  1, 0), s(-1,  1, 1), s(-1,  1, 2) );
		t(-1,  2, z) = cint( z, s(-1,  2, -1), s(-1,  2, 0), s(-1,  2, 1), s(-1,  2, 2) );

		t( 0, -1, z) = cint( z, s( 0, -1, -1), s( 0, -1, 0), s( 0, -1, 1), s( 0, -1, 2) );
		t( 0,  0, z) = cint( z, s( 0,  0, -1), s( 0,  0, 0), s( 0,  0, 1), s( 0,  0, 2) );
		t( 0,  1, z) = cint( z, s( 0,  1, -1), s( 0,  1, 0), s( 0,  1, 1), s( 0,  1, 2) );
		t( 0,  2, z) = cint( z, s( 0,  2, -1), s( 0,  2, 0), s( 0,  2, 1), s( 0,  2, 2) );

		t( 1, -1, z) = cint( z, s( 1, -1, -1), s( 1, -1, 0), s( 1, -1, 1), s( 1, -1, 2) );
		t( 1,  0, z) = cint( z, s( 1,  0, -1), s( 1,  0, 0), s( 1,  0, 1), s( 1,  0, 2) );
		t( 1,  1, z) = cint( z, s( 1,  1, -1), s( 1,  1, 0), s( 1,  1, 1), s( 1,  1, 2) );
		t( 1,  2, z) = cint( z, s( 1,  2, -1), s( 1,  2, 0), s( 1,  2, 1), s( 1,  2, 2) );

		t( 2, -1, z) = cint( z, s( 2, -1, -1), s( 2, -1, 0), s( 2, -1, 1), s( 2, -1, 2) );
		t( 2,  0, z) = cint( z, s( 2,  0, -1), s( 2,  0, 0), s( 2,  0, 1), s( 2,  0, 2) );
		t( 2,  1, z) = cint( z, s( 2,  1, -1), s( 2,  1, 0), s( 2,  1, 1), s( 2,  1, 2) );
		t( 2,  2, z) = cint( z, s( 2,  2, -1), s( 2,  2, 0), s( 2,  2, 1), s( 2,  2, 2) );
	*/


	//centered on cell centers

	//wrap x-direction
	xx = x;
	if(xx>=grid_info.BoxSize)
		xx -= grid_info.BoxSize;
	if(xx<0)
		xx += grid_info.BoxSize;

	//wrap y-direction
	yy = y;
	if(yy>=grid_info.BoxSize)
		yy -= grid_info.BoxSize;
	if(yy<0)
		yy += grid_info.BoxSize;

	//wrap z-direction
	zz = z;
	if(zz>=grid_info.BoxSize)
		zz -= grid_info.BoxSize;
	if(zz<0)
		zz += grid_info.BoxSize;

	ii = (int) ((xx-0.5*grid_info.dx)/grid_info.dx) - grid_info.nx_local_start;		//integer index along x direction
	jj = (int) ((yy-0.5*grid_info.dy)/grid_info.dy);					//integer index along y direction
	kk = (int) ((zz-0.5*grid_info.dz)/grid_info.dz);					//integer index along z direction

	//find the fractional distance between location
	//and the previous grid point

	xx = (xx - 0.5*grid_info.dx)/grid_info.dx - ((double) (ii + grid_info.nx_local_start));
	yy = (yy - 0.5*grid_info.dy)/grid_info.dy - ((double) jj);
	zz = (zz - 0.5*grid_info.dz)/grid_info.dz - ((double) kk);

	//if the position is not within this slab, then
	//return -1
	if(ii < 0 || ii >= grid_info.nx_local)
		return 0;



	if(ii==0)
	{
	  for(int i=0;i<1;i++)
	   for(int j=0;j<4;j++)
	     for(int k=0;k<4;k++)
	     {
		jy = jj - 1 + j;
		if(jy>=grid_info.ny)
		    jy -= grid_info.ny;
		if(jy<0)
		    jy += grid_info.ny;

		kz = kk - 1 + k;
		if(kz>=grid_info.nz)
		    kz -= grid_info.nz;
		if(kz<0)
		    kz += grid_info.nz;

	       ijk = grid_info.nz*jy + kz;
	       s[i][j][k] = log(fl[ijk]);

	     }

	  for(int i=1;i<4;i++)
	   for(int j=0;j<4;j++)
	     for(int k=0;k<4;k++)
	     {
	       ijk = grid_ijk(ii-1+i,jj-1+j,kk-1+k,grid_info);
	       s[i][j][k] = log(f[ijk]);
	     }
	}else if (ii >= grid_info.nx_local-3){

	  for(int i=0;i<4;i++)
	   for(int j=0;j<4;j++)
	     for(int k=0;k<4;k++)
	     {

		ix = ii - 1 + i;
	
		if(ix<grid_info.nx_local)
		{
		  //normal
		  ijk = grid_ijk(ii-1+i,jj-1+j,kk-1+k,grid_info);
		  s[i][j][k] = log(f[ijk]);
		}else{
      
		  ix -= grid_info.nx_local;
  
		  jy = jj - 1 + j;
		  if(jy>=grid_info.ny)
		     jy -= grid_info.ny;
		  if(jy<0)
		     jy += grid_info.ny;

		  kz = kk - 1 + k;
		  if(kz>=grid_info.nz)
		     kz -= grid_info.nz;
		  if(kz<0)
		     kz += grid_info.nz;

		  ijk = grid_info.ny*grid_info.nz*ix + grid_info.nz*jy + kz;

		  s[i][j][k] = log(fu[ijk]);
	      }
	  }
      }else{
      
	  for(int i=0;i<4;i++)
	   for(int j=0;j<4;j++)
	     for(int k=0;k<4;k++)
	     {
	       ijk = grid_ijk(ii-1+i,jj-1+j,kk-1+k,grid_info);
	       s[i][j][k] = log(f[ijk]);
	     }
      }

	//find t(i,j,z)
	for(int i=0;i<4;i++)
		for(int j=0;j<4;j++)
			t[i][j] = cint( zz, s[i][j][0], s[i][j][1], s[i][j][2], s[i][j][3] );
	//find u(i,y,z)
	for(int i=0;i<4;i++)
		u[i] = cint( yy, t[i][0], t[i][1], t[i][2], t[i][3] );


	//return f(x,y,z)
	f_return = cint( xx, u[0], u[1], u[2], u[3] );

	//return f_return;
	return exp(f_return);
}


/*! \fn double grid_cubic_interpolation(double x, double y, double z, double *f, double *fl, double *fu, FFTW_Grid_Info grid_info)
 *  \brief Tricubic interpolation on the grid */
double grid_cubic_interpolation(double x, double y, double z, double *f, double *fl, double *fu, FFTW_Grid_Info grid_info)
{

  //fl is a single slice
  //fu is three slices

    double s[4][4][4];
    double t[4][4];
    double u[4];
    int ii, jj, kk;
    int ix, jy, kz;

    double xx, yy, zz;

    double f_return;

    int ijk;

	/*
		s(i,j,k) = grid point at ijk
		t(i,j,z) = cint( z, s( i, j,-1), s(i,j,0), s(i,j,1), s(i,j,2) );
		u(i,y,z) = cint( y, t( i,-1, z), t(i,0,z), t(i,1,z), t(i,2,z) );
		f(x,y,z) = cint( x, u(-1, y, z), u(0,y,z), u(1,y,z), u(2,y,z) );


		u(-1, y, z) = cint( y, t(-1, -1, z), t(-1, 0, z), t(-1, 1, z), t(-1, 2, z) );
		u( 0, y, z) = cint( y, t( 0, -1, z), t( 0, 0, z), t( 0, 1, z), t( 0, 2, z) );
		u( 1, y, z) = cint( y, t( 1, -1, z), t( 1, 0, z), t( 1, 1, z), t( 1, 2, z) );
		u( 2, y, z) = cint( y, t( 2, -1, z), t( 2, 0, z), t( 2, 1, z), t( 2, 2, z) );

		t(-1, -1, z) = cint( z, s(-1, -1, -1), s(-1, -1, 0), s(-1, -1, 1), s(-1, -1, 2) );
		t(-1,  0, z) = cint( z, s(-1,  0, -1), s(-1,  0, 0), s(-1,  0, 1), s(-1,  0, 2) );
		t(-1,  1, z) = cint( z, s(-1,  1, -1), s(-1,  1, 0), s(-1,  1, 1), s(-1,  1, 2) );
		t(-1,  2, z) = cint( z, s(-1,  2, -1), s(-1,  2, 0), s(-1,  2, 1), s(-1,  2, 2) );

		t( 0, -1, z) = cint( z, s( 0, -1, -1), s( 0, -1, 0), s( 0, -1, 1), s( 0, -1, 2) );
		t( 0,  0, z) = cint( z, s( 0,  0, -1), s( 0,  0, 0), s( 0,  0, 1), s( 0,  0, 2) );
		t( 0,  1, z) = cint( z, s( 0,  1, -1), s( 0,  1, 0), s( 0,  1, 1), s( 0,  1, 2) );
		t( 0,  2, z) = cint( z, s( 0,  2, -1), s( 0,  2, 0), s( 0,  2, 1), s( 0,  2, 2) );

		t( 1, -1, z) = cint( z, s( 1, -1, -1), s( 1, -1, 0), s( 1, -1, 1), s( 1, -1, 2) );
		t( 1,  0, z) = cint( z, s( 1,  0, -1), s( 1,  0, 0), s( 1,  0, 1), s( 1,  0, 2) );
		t( 1,  1, z) = cint( z, s( 1,  1, -1), s( 1,  1, 0), s( 1,  1, 1), s( 1,  1, 2) );
		t( 1,  2, z) = cint( z, s( 1,  2, -1), s( 1,  2, 0), s( 1,  2, 1), s( 1,  2, 2) );

		t( 2, -1, z) = cint( z, s( 2, -1, -1), s( 2, -1, 0), s( 2, -1, 1), s( 2, -1, 2) );
		t( 2,  0, z) = cint( z, s( 2,  0, -1), s( 2,  0, 0), s( 2,  0, 1), s( 2,  0, 2) );
		t( 2,  1, z) = cint( z, s( 2,  1, -1), s( 2,  1, 0), s( 2,  1, 1), s( 2,  1, 2) );
		t( 2,  2, z) = cint( z, s( 2,  2, -1), s( 2,  2, 0), s( 2,  2, 1), s( 2,  2, 2) );
	*/


	//centered on cell centers

	//wrap x-direction
	xx = x;
	if(xx>=grid_info.BoxSize)
		xx -= grid_info.BoxSize;
	if(xx<0)
		xx += grid_info.BoxSize;

	//wrap y-direction
	yy = y;
	if(yy>=grid_info.BoxSize)
		yy -= grid_info.BoxSize;
	if(yy<0)
		yy += grid_info.BoxSize;

	//wrap z-direction
	zz = z;
	if(zz>=grid_info.BoxSize)
		zz -= grid_info.BoxSize;
	if(zz<0)
		zz += grid_info.BoxSize;

	ii = (int) ((xx-0.5*grid_info.dx)/grid_info.dx) - grid_info.nx_local_start;		//integer index along x direction
	jj = (int) ((yy-0.5*grid_info.dy)/grid_info.dy);					//integer index along y direction
	kk = (int) ((zz-0.5*grid_info.dz)/grid_info.dz);					//integer index along z direction

	//find the fractional distance between location
	//and the previous grid point

	xx = (xx - 0.5*grid_info.dx)/grid_info.dx - ((double) (ii + grid_info.nx_local_start));
	yy = (yy - 0.5*grid_info.dy)/grid_info.dy - ((double) jj);
	zz = (zz - 0.5*grid_info.dz)/grid_info.dz - ((double) kk);

	//if the position is not within this slab, then
	//return -1
	if(ii < 0 || ii >= grid_info.nx_local)
		return 0;



	if(ii==0)
	{
	  for(int i=0;i<1;i++)
	   for(int j=0;j<4;j++)
	     for(int k=0;k<4;k++)
	     {
		jy = jj - 1 + j;
		if(jy>=grid_info.ny)
		    jy -= grid_info.ny;
		if(jy<0)
		    jy += grid_info.ny;

		kz = kk - 1 + k;
		if(kz>=grid_info.nz)
		    kz -= grid_info.nz;
		if(kz<0)
		    kz += grid_info.nz;

	       ijk = grid_info.nz*jy + kz;
	       s[i][j][k] = fl[ijk];

	     }

	  for(int i=1;i<4;i++)
	   for(int j=0;j<4;j++)
	     for(int k=0;k<4;k++)
	     {
	       ijk = grid_ijk(ii-1+i,jj-1+j,kk-1+k,grid_info);
	       s[i][j][k] = f[ijk];
	     }
	}else if (ii >= grid_info.nx_local-3){

	  for(int i=0;i<4;i++)
	   for(int j=0;j<4;j++)
	     for(int k=0;k<4;k++)
	     {

		ix = ii - 1 + i;
	
		if(ix<grid_info.nx_local)
		{
		  //normal
		  ijk = grid_ijk(ii-1+i,jj-1+j,kk-1+k,grid_info);
		  s[i][j][k] = f[ijk];
		}else{
      
		  ix -= grid_info.nx_local;
  
		  jy = jj - 1 + j;
		  if(jy>=grid_info.ny)
		     jy -= grid_info.ny;
		  if(jy<0)
		     jy += grid_info.ny;

		  kz = kk - 1 + k;
		  if(kz>=grid_info.nz)
		     kz -= grid_info.nz;
		  if(kz<0)
		     kz += grid_info.nz;

		  ijk = grid_info.ny*grid_info.nz*ix + grid_info.nz*jy + kz;

		  s[i][j][k] = fu[ijk];
	      }
	  }
      }else{
      
	  for(int i=0;i<4;i++)
	   for(int j=0;j<4;j++)
	     for(int k=0;k<4;k++)
	     {
	       ijk = grid_ijk(ii-1+i,jj-1+j,kk-1+k,grid_info);
	       s[i][j][k] = f[ijk];
	     }
      }

	//find t(i,j,z)
	for(int i=0;i<4;i++)
		for(int j=0;j<4;j++)
			t[i][j] = cint( zz, s[i][j][0], s[i][j][1], s[i][j][2], s[i][j][3] );
	//find u(i,y,z)
	for(int i=0;i<4;i++)
		u[i] = cint( yy, t[i][0], t[i][1], t[i][2], t[i][3] );


	//return f(x,y,z)
	f_return = cint( xx, u[0], u[1], u[2], u[3] );

	return f_return;
}

/*! \fn double cint(double x, double pm1, double p0, double pp1, double pp2)
 *  \brief Hermite interpolation in one direction. */
double cint(double x, double pm1, double p0, double pp1, double pp2)
{
	double c = 0;

	c  = 0.5*(-1*x*x*x + 2*x*x - x)*pm1;
	c += 0.5*( 3*x*x*x - 5*x*x + 2)*p0;
	c += 0.5*(-3*x*x*x + 4*x*x + x)*pp1;
	c += 0.5*(   x*x*x -   x*x    )*pp2;

	return c;
}


/*! \fn double *grid_normal_white_noise(double mu, double sigma_squared, int iseed, FFTW_Grid_Info grid_info) 
 *  \brief Produces a fft grid of normal variates with white noise, with mean mu and variance sigma_squared, with seed iseed. */
double *grid_normal_white_noise(double mu, double variance, int iseed, FFTW_Grid_Info grid_info)
{
	//checked this produces correct 
	//mean and variance

	int i, j, k, ijk;

	int nx_local = grid_info.nx_local;
	int nx       = grid_info.nx;
	int ny       = grid_info.ny;
	int nz       = grid_info.nz;


	double *field;





	//allocate field
	field = allocate_real_fftw_grid(grid_info);

	if(grid_info.ndim==2)
		nz=1;

	//generate field
	for(i=0;i<nx_local;++i)
		for(j=0;j<ny;++j)
			for(k=0;k<nz;++k)
			{
				ijk = grid_ijk(i,j,k,grid_info);
				//draw a random number
				field[ijk] = gsl_ran_gaussian(r, sqrt(variance)) + mu;
			}



	//return the field
	return field;
}



/*! \fn double *grid_uniform_white_noise(double mu, double sigma_squared, int iseed, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world) 
 *  \brief Produces a fft grid of uniform variates with white noise, with mean mu and variance sigma_squared, with seed iseed. */
double *grid_uniform_white_noise(double mu, double variance, int iseed, FFTW_Grid_Info grid_info)
{
	//checked this produces correct 
	//mean and variance
	int i, j, k, ijk;

	int nx_local = grid_info.nx_local;
	int nx       = grid_info.nx;
	int ny       = grid_info.ny;
	int nz       = grid_info.nz;

	double *field;

	//random number generator
	gsl_rng *r;
	const gsl_rng_type *T;

	//extend of uniform distribution
	double a, b;	


	//tests
	double mean_test;
	double variance_test;

	//set up random number generator
	gsl_rng_env_setup();
	T = gsl_rng_default;
	r = gsl_rng_alloc(T);

	//set rng seed
	gsl_rng_set(r,iseed);

	//allocate field
	field = allocate_real_fftw_grid(grid_info);


	//Se the extent of the uniform distribution
	//for a uniform distribution
	// mu = 0.5*( a + b )
	// variance = (1/12) *(  b - a )^2
	// a = 2*mu - b
	// variance = (1/12) * ( 2*b - 2*mu )^2
	// 2*b - 2*mu = sqrt(12*variance)
	// b = mu + 0.5 * sqrt(12*variance)
	// a = mu - 0.5 * sqrt(12*variance)
	
	a = mu - 0.5 * sqrt(12*variance);
	b = mu + 0.5 * sqrt(12*variance);

	if(grid_info.ndim==2)
		nz=1;

	//generate field
	for(i=0;i<nx_local;++i)
		for(j=0;j<ny;++j)
			for(k=0;k<nz;++k)
			{
				ijk = grid_ijk(i,j,k,grid_info);
				//draw a random number
				field[ijk] = gsl_ran_flat(r, a, b);
			}
	//free the gsl rng.
    gsl_rng_free(r);

	//return the field
	return field;
}
//OK ABOVE HERE

/*! \fn double *generate_gaussian_field(double (*Pk_power_spectrum)(double,void *), void *params, int iseed, FFTW_Grid_Info grid_info, rfftwnd_mpi_plan plan, rfftwnd_mpi_plan iplan, int myid, int numprocs, MPI_Comm world)
 *  \brief  Given a power spectrum Pk_power_spectrum that depends on wavenumber and some parameters *params, and a random number generator 
            seed iseed, return a gaussian random field with power spectrum Pk_power_spectrum. */
double *generate_gaussian_field(double (*Pk_power_spectrum)(double,void *), void *params, int iseed, FFTW_Grid_Info grid_info, MPI_Comm world)
{

	//forward and reverse FFTW plans
	fftw_plan plan_z;
	fftw_plan iplan_z;

	int nx = grid_info.nx;
	int ny = grid_info.ny;
	int nz = grid_info.nz;

	double sigma_m = 0;
	double mean_m = 0;
	//mean for initial white noise gaussian field
	double mu = 0;

	int ijk, i, j;
	if(grid_info.ndim==2)
		nz=1;
	
	//variance for initial white noise gaussian field
	//double sigma_squared = ((double) nx)*((double) ny)*((double) nz);
	double sigma_squared = 1.0;
	double scale = 1./( ((double) grid_info.nx)*((double) grid_info.ny)*((double) grid_info.nz) );
	double total = 0;

	//initial white noise gaussian field and
	//final gaussian field with power spectrum P(k)
	double *zeta;
	fftw_complex *zeta_t;


	//transform of white noise gaussian; white noise gaussian x transfer function
	fftw_complex *zeta_k;

	//generate a white noise gaussian field
	zeta = grid_normal_white_noise(mu, sigma_squared, iseed, grid_info);


	//allocate a complex grid to contain the forward transform
	zeta_k = allocate_complex_fftw_grid(grid_info);


	//get complex version of zeta
	//grid_copy_real_to_complex_in_place(zeta, zeta_t, grid_info);

	//create the fftw plan
	switch(grid_info.ndim)
	{
	  case 2:
		plan_z  = fftw_mpi_plan_dft_r2c_2d(grid_info.nx, grid_info.ny, zeta, zeta_k, world,  FFTW_ESTIMATE);
		iplan_z = fftw_mpi_plan_dft_c2r_2d(grid_info.nx, grid_info.ny, zeta_k, zeta, world, FFTW_ESTIMATE);
		break;
	  case 3:
	  	break;
		//plan_z  = fftw_mpi_plan_dft_3d(grid_info.nx, grid_info.ny, grid_info.nz, zeta_t, zeta_k, world, FFTW_FORWARD,  FFTW_ESTIMATE);
		//iplan_z = fftw_mpi_plan_dft_3d(grid_info.nx, grid_info.ny, grid_info.nz, zeta_t, zeta_k, world, FFTW_BACKWARD, FFTW_ESTIMATE);
	}

	//forward transform the white noise gaussian field to k-space
	fftw_execute(plan_z);

	//apply the transfer function to zeta_k

	if(grid_info.ndim==2)
	{	
		grid_transform_apply_transfer_function_2d(Pk_power_spectrum,params,zeta_k,grid_info);
	}else{
		grid_transform_apply_transfer_function(Pk_power_spectrum,params,zeta_k,grid_info);
	}

	//perform the inverse transform
	fftw_execute(iplan_z);

	//copy inverse transform into convolved field
	//grid_copy_complex_to_real_in_place(zeta_t, zeta, grid_info);

	//free the complex transform
	fftw_free(zeta_k);
	//fftw_free(zeta_t);

	//destroy the plans
	fftw_destroy_plan(plan_z);
	fftw_destroy_plan(iplan_z);

	//rescale after transform
	grid_rescale(scale, zeta, grid_info);


	//enforce variance
	for(int i=0;i<nx;i++)
		for(int j=0;j<ny;j++)
			for(int k=0;k<nz;k++)
			{
				ijk = grid_ijk(i,j,k,grid_info);
				mean_m += zeta[ijk];
				total  += 1.0;
			}
	mean_m /= total;
	for(int i=0;i<nx;i++)
		for(int j=0;j<ny;j++)
			for(int k=0;k<nz;k++)
			{
				ijk = grid_ijk(i,j,k,grid_info);
				sigma_m += pow(zeta[ijk] - mean_m, 2.);
				zeta[ijk] -= mean_m;
			}
	sigma_m /= total;
	scale = sqrt(sigma_squared/sigma_m);


	//rescale the fields
	grid_rescale(scale, zeta, grid_info);

	//return the answer
	return zeta;
}


/*! \fn void grid_transform_apply_transfer_function(double (*Pk_power_spectrum)(double,void*),void *params, fftw_complex *cdata, FFTW_Grid_Info grid_info)
 *  \brief Given a power spectrum Pk, that depends on wavenumber and some parameters *params, multiply complex grid transform cdata by the 
 *         transfer function T(k) = sqrt( (2*pi/L)^3 Pk)*/
void grid_transform_apply_transfer_function(double (*Pk_power_spectrum)(double,void*),void *params, fftw_complex *cdata, FFTW_Grid_Info grid_info)
{

	//multiplies cdata x T(k)

	int ijk;

	int nx                            = grid_info.nx;
	int ny                            = grid_info.ny;
	int nz                            = grid_info.nz;
	int ny_local_transposed      = grid_info.ny_local_transposed;
	int ny_local_start_transposed = grid_info.ny_local_start_transposed;

	double kx, ky, kz;
	double kk;
	double L = grid_info.BoxSize;

	//int nzl;

	//nzl = nz/2 +1;
	//if(grid_info.ndim==2)
	//	nzl = 1;

	for(int j=0;j<ny_local_transposed;j++)
		for(int i=0;i<nx;i++)
			for(int k=0;k<(nz/2+1);k++)
			{

				if(i>nx/2)
				{
					kx = ((double) (i-nx));
				}else{
					kx = ((double) i);
				}
				if(ny_local_start_transposed+j>ny/2)
				{
					ky = ((double) (ny_local_start_transposed+j-ny) );
				}else{
					ky = ((double) (ny_local_start_transposed+j) );
				}
				kz = ((double) k);



				// the magnitude of the k-vector

				kk = (2*M_PI/L) * sqrt( kx*kx + ky*ky + kz*kz );


				//index of cdata corresponding to k

				//ijk = (j*nx + i)*(nz/2+1) + k;
				//ijk = grid_transposed_ijk(i,j,k,grid_info);


				// multiply cdata * T(k)
				// T(k) = \sqrt{ (2*pi/L)^3 * P(k) }
				// see eqn. 8 of Bertschinger 2001, ApJS, 137, 1-20

				//need to check real / imaginary multiplication

				//cdata[ijk].re*= sqrt( pow(2*M_PI/L,3)*Pk_power_spectrum(kk,params) );
				//cdata[ijk].im*= sqrt( pow(2*M_PI/L,3)*Pk_power_spectrum(kk,params) );
				cdata[ijk][0]*= sqrt(Pk_power_spectrum(kk,params) );
				cdata[ijk][1]*= sqrt(Pk_power_spectrum(kk,params) );
			}
}


/*! \fn void grid_transform_apply_transfer_function_2d(double (*Pk_power_spectrum)(double,void*),void *params, fftw_complex *cdata, FFTW_Grid_Info grid_info)
 *  \brief Given a power spectrum Pk, that depends on wavenumber and some parameters *params, multiply complex grid transform cdata by the 
 *         transfer function T(k) = sqrt( (2*pi/L)^3 Pk)*/
void grid_transform_apply_transfer_function_2d(double (*Pk_power_spectrum)(double,void*),void *params, fftw_complex *cdata, FFTW_Grid_Info grid_info)
{

	//multiplies cdata x T(k)

	int ijk;

	int nx                            = grid_info.nx;
	int ny                            = grid_info.ny;
	int nx_local_start 			  = grid_info.nx_local_start;
	/*int ny_local_transposed       = grid_info.ny_local_transposed;
	int ny_local_start_transposed = grid_info.ny_local_start_transposed;
	int nx_local_transposed       = grid_info.nx_local_transposed;
	int nx_local_start_transposed = grid_info.nx_local_start_transposed;*/

	double kx, ky;
	double kk;
	double L = 1.0;

	int nzl;

	printf("HEREHERE\n");

	for(int i=0;i<nx;i++)
		for(int j=0;j<ny;j++)
		{

			if(j>ny/2)
			{
				ky = ((double) (j-ny/2));
			}else{
				ky = ((double) j);
			}

			kx = ((double) (nx_local_start+i) );

			// the magnitude of the k-vector

			kk = (2*M_PI/L) * sqrt( kx*kx + ky*ky );
			//printf("i %d j %d kx %e ky %e kk %e Pk %e\n",i,j,kx,ky,kk,Pk_power_spectrum(kk,params));


			//index of cdata corresponding to k

			//ijk = (j*nx + i)*(nz/2+1) + k;
			ijk = grid_complex_ijk(i,j,0,grid_info);


			// multiply cdata * T(k)
			// T(k) = \sqrt{ (2*pi/L)^3 * P(k) }
			// see eqn. 8 of Bertschinger 2001, ApJS, 137, 1-20

			//need to check real / imaginary multiplication

			//cdata[ijk].re*= sqrt( pow(2*M_PI/L,3)*Pk_power_spectrum(kk,params) );
			//cdata[ijk].im*= sqrt( pow(2*M_PI/L,3)*Pk_power_spectrum(kk,params) );


			cdata[ijk][0]*= sqrt(Pk_power_spectrum(kk,params) );
			cdata[ijk][1]*= sqrt(Pk_power_spectrum(kk,params) );
			//printf("ijk %d kx %e ky %e kk %e\n",ijk,kx,ky,kk);
			//cdata[ijk][0]*= 1.0;
			//cdata[ijk][1]*= 1.0;
		}
}

//TESTING ABOVE HERE





/*! \fn void check_parsevals_theorem(double *u, FFTW_Grid_Info grid_info, rfftwnd_mpi_plan plan, rfftwnd_mpi_plan iplan, int myid, int numprocs, MPI_Comm world)
 *  \brief Check for energy conservation in power spectra for a scalar grid. 
void check_parsevals_theorem(double *u, FFTW_Grid_Info grid_info, rfftwnd_mpi_plan plan, rfftwnd_mpi_plan iplan, int myid, int numprocs, MPI_Comm world)
{
	if(grid_info.ndim==2)
	{
		//2-d
		check_parsevals_theorem_2d(u, grid_info, plan, iplan, myid, numprocs, world);
	}else{
		//3-d
		check_parsevals_theorem_3d(u, grid_info, plan, iplan, myid, numprocs, world);
	}
}

/*! \fn void check_parsevals_theorem_3d(double *u, FFTW_Grid_Info grid_info, rfftwnd_mpi_plan plan, rfftwnd_mpi_plan iplan, int myid, int numprocs, MPI_Comm world)
 *  \brief Check for energy conservation in power spectra for a 3-d scalar grid. 
void check_parsevals_theorem_3d(double *u, FFTW_Grid_Info grid_info, rfftwnd_mpi_plan plan, rfftwnd_mpi_plan iplan, int myid, int numprocs, MPI_Comm world)
{	
	double *work;
	double *ukx;
	fftw_complex *cEk;
	fftw_complex *cukx;

	double energy, energy_proc;
	int ijk;
	int nx = grid_info.nx;
	int ny = grid_info.ny;
	int nz = grid_info.nz;
	int nx_local = grid_info.nx_local;
	int local_ny_after_transpose = grid_info.local_ny_after_transpose;
	int local_y_start_after_transpose = grid_info.local_y_start_after_transpose;

	char variable_name[200];

	double total_specific_energy = grid_total_specific_energy(u, grid_info, plan, iplan, myid, numprocs, world);

	double *k_array;
	double *P_k_array;


	double kx, ky, kz, kk, wk;
	int nk, ik;
	double energy_check;

	//initialize k array
	initialize_power_spectrum_k_array(k_array, &nk, grid_info, myid, numprocs, world);

	//allocate fft workspace 
	sprintf(variable_name,"work");
	work   = allocate_real_fftw_grid(grid_info.total_local_size, variable_name, myid, numprocs, world,0);

	//allocate fft workspace 
	//sprintf(variable_name,"Ek");
	//Ek = allocate_real_fftw_grid(grid_info.total_local_size, variable_name, myid, numprocs, world,0);
	cEk = allocate_complex_fftw_grid(grid_info.total_local_size, variable_name, myid, numprocs, world,0);
	

	//copy the grid to preserve it
	ukx = grid_copy(u, grid_info, plan, iplan, myid, numprocs, world);

	//perform the forward transform on the components of u
	forward_transform_fftw_grid(ukx, work, plan, cukx, grid_info, myid, numprocs, world);

	//typecast the complex transform pointers
	cukx = (fftw_complex *) ukx;


	if(grid_info.ndim==2)
	{
		if(myid==0)
		{
			printf("check_parsevals_theorem is not implemented for 2 dimensions.\n");
			fflush(stdout);
		}
		MPI_Abort(world,-1);
		exit(-1);
	}
	if(myid==0)
	{
		printf("check_parsevals_theorem : total specfic energy %10.9e\n",total_specific_energy);
	}

	//check cukx contents
	energy_proc = 0;
	for(int j=0;j<local_ny_after_transpose;j++)
		for(int i=0;i<nx;i++)
		{
			//first, positive kz
			for(int k=0;k<(nz/2+1);k++)
			{
				ijk = grid_transpose_ijk(i,j,k,grid_info);
				energy_proc += 0.5*( cukx[ijk].re*cukx[ijk].re +  cukx[ijk].im*cukx[ijk].im);

				cEk[ijk].re  = 0.5*( cukx[ijk].re*cukx[ijk].re +  cukx[ijk].im*cukx[ijk].im);
			}

			//then for negative kz, needed because we sum over all frequencies
			for(int k=1;k<(nz/2);k++)
			{
				ijk = grid_transpose_ijk(i,j,k,grid_info);
				energy_proc += 0.5*( cukx[ijk].re*cukx[ijk].re +  cukx[ijk].im*cukx[ijk].im);
			}
		}	

	energy = 0;
	MPI_Allreduce(&energy_proc,&energy,1,MPI_DOUBLE,MPI_SUM,world);
	if(myid==0)
	{
		printf("check_parsevals_theorem : total specfic energy check 1 %10.9e\n",energy * grid_info.dVk);
	}


	//Check energy in cEk

	energy_proc = 0;
	for(int j=0;j<local_ny_after_transpose;j++)
		for(int i=0;i<nx;i++)
		{
			//first, positive kz
			for(int k=0;k<(nz/2+1);k++)
			{
				ijk = grid_transpose_ijk(i,j,k,grid_info);
				energy_proc += cEk[ijk].re;
			}

			//then for negative kz, needed because we sum over all frequencies
			for(int k=1;k<(nz/2);k++)
			{
				ijk = grid_transpose_ijk(i,j,k,grid_info);
				energy_proc += cEk[ijk].re;
			}
		}	

	energy = 0;
	MPI_Allreduce(&energy_proc,&energy,1,MPI_DOUBLE,MPI_SUM,world);
	if(myid==0)
	{
		printf("check_parsevals_theorem : total specfic energy check 2 %10.9e\n",energy * grid_info.dVk);
	}

	energy_check = energy;

	//convert to 3-d spectrum
	P_k_array = power_spectrum(cukx, k_array, nk, grid_info, myid, numprocs, world);
	for(ik = 0;ik<nk;ik++)
		P_k_array[ik]*=0.5;

	//check total energy, volume properly accounted for
	energy= 0;
	for(ik = 0;ik<nk;ik++)
		energy+= P_k_array[ik]*P_k_array[nk+ik];

	if(myid==0)
	{
		printf("check_parsevals_theorem : total specfic energy check 3 %10.9e (error = %e) (nk = %d)\n",energy,(energy-total_specific_energy)/total_specific_energy,nk);
	}


	free(k_array);
	free(P_k_array);

	
	free(work);
	free(ukx);
	free(cEk);
}

/*! \fn void check_parsevals_theorem_2d(double *u, FFTW_Grid_Info grid_info, rfftwnd_mpi_plan plan, rfftwnd_mpi_plan iplan, int myid, int numprocs, MPI_Comm world)
 *  \brief Check for energy conservation in power spectra for a scalar grid. 
void check_parsevals_theorem_2d(double *u, FFTW_Grid_Info grid_info, rfftwnd_mpi_plan plan, rfftwnd_mpi_plan iplan, int myid, int numprocs, MPI_Comm world)
{	
	double *work;
	double *ukx;
	fftw_complex *cEk;
	fftw_complex *cukx;

	double energy, energy_proc;
	int ijk, jlim;
	int nx = grid_info.nx;
	int ny = grid_info.ny;
	int nz = grid_info.nz;
	int nx_local = grid_info.nx_local;
	int local_ny_after_transpose = grid_info.local_ny_after_transpose;
	int local_y_start_after_transpose = grid_info.local_y_start_after_transpose;

	char variable_name[200];

	double total_specific_energy = grid_total_specific_energy(u, grid_info, plan, iplan, myid, numprocs, world);

	double *k_array;
	double *P_k_array;


	double kx, ky, kk, wk;
	int nk, ik;
	double energy_check;

	//initialize k array
	initialize_power_spectrum_k_array(k_array, &nk, grid_info, myid, numprocs, world);

	//allocate fft workspace 
	sprintf(variable_name,"work");
	work   = allocate_real_fftw_grid(grid_info.total_local_size, variable_name, myid, numprocs, world,0);

	//allocate fft workspace 
	cEk = allocate_complex_fftw_grid(grid_info.total_local_size, variable_name, myid, numprocs, world,0);
	

	//copy the grid to preserve it
	ukx = grid_copy(u, grid_info, plan, iplan, myid, numprocs, world);

	//perform the forward transform on the components of u
	forward_transform_fftw_grid(ukx, work, plan, cukx, grid_info, myid, numprocs, world);

	//typecast the complex transform pointers
	cukx = (fftw_complex *) ukx;


	//check cukx contents
	energy_proc = 0;
	for(int j=0;j<local_ny_after_transpose;j++)
		for(int i=0;i<nx;i++)
		{
			ijk = grid_transpose_ijk(i,j,0,grid_info);
			energy_proc += 0.5*( cukx[ijk].re*cukx[ijk].re +  cukx[ijk].im*cukx[ijk].im);

			cEk[ijk].re  = 0.5*( cukx[ijk].re*cukx[ijk].re +  cukx[ijk].im*cukx[ijk].im);
		}
	if(myid==0)
	{
		jlim = 1;
	}else{
		jlim = 0;
	}
	for(int j=jlim;j<local_ny_after_transpose;j++)
		for(int i=0;i<nx;i++)
		{
			//then for negative ky, needed because we sum over all frequencies
			ijk = grid_transpose_ijk(i,j,0,grid_info);
			energy_proc += 0.5*( cukx[ijk].re*cukx[ijk].re +  cukx[ijk].im*cukx[ijk].im);

			cEk[ijk].re  = 0.5*( cukx[ijk].re*cukx[ijk].re +  cukx[ijk].im*cukx[ijk].im);
		}


	energy = 0;
	MPI_Allreduce(&energy_proc,&energy,1,MPI_DOUBLE,MPI_SUM,world);
	if(myid==0)
	{
		printf("check_parsevals_theorem_2d : total specfic energy check 1 %10.9e\n",energy * grid_info.dVk);
	}


	//Check energy in cEk

	energy_proc = 0;
	for(int j=0;j<local_ny_after_transpose;j++)
		for(int i=0;i<nx;i++)
		{
			//first, positive ky
			ijk = grid_transpose_ijk(i,j,0,grid_info);
			energy_proc += cEk[ijk].re;
		}
	for(int j=jlim;j<local_ny_after_transpose;j++)
		for(int i=0;i<nx;i++)
		{
			//then for negative ky, needed because we sum over all frequencies
			ijk = grid_transpose_ijk(i,j,0,grid_info);
			energy_proc += cEk[ijk].re;
		}	

	energy = 0;
	MPI_Allreduce(&energy_proc,&energy,1,MPI_DOUBLE,MPI_SUM,world);
	if(myid==0)
	{
		printf("check_parsevals_theorem_2d : total specfic energy check 2 %10.9e\n",energy * grid_info.dVk);
	}

	energy_check = energy;

	//convert to 2-d spectrum
	P_k_array = power_spectrum(cukx, k_array, nk, grid_info, myid, numprocs, world);
	for(ik = 0;ik<nk;ik++)
		P_k_array[ik]*=0.5;

	//weight by volume
	energy= 0;
	for(ik = 0;ik<nk;ik++)
		energy+= P_k_array[ik]*P_k_array[nk+ik];

	if(myid==0)
	{
		printf("check_parsevals_theorem_2d : total specfic energy check 3 %10.9e (error = %e) (nk = %d)\n",energy,(energy-energy_check)/energy_check,nk);
	}


	free(k_array);
	free(P_k_array);

	
	free(work);
	free(ukx);
	free(cEk);
}

/*! \fn void check_parsevals_theorem_field(double **u, FFTW_Grid_Info grid_info, rfftwnd_mpi_plan plan, rfftwnd_mpi_plan iplan, int myid, int numprocs, MPI_Comm world)
 *  \brief Check for energy conservation in power spectra for a vector field. 
void check_parsevals_theorem_field(double **u, FFTW_Grid_Info grid_info, rfftwnd_mpi_plan plan, rfftwnd_mpi_plan iplan, int myid, int numprocs, MPI_Comm world)
{
	if(grid_info.ndim==2)
	{
		//2-d
		check_parsevals_theorem_field_2d(u, grid_info, plan, iplan, myid, numprocs, world);
	}else{
		//3-d
		check_parsevals_theorem_field_3d(u, grid_info, plan, iplan, myid, numprocs, world);
	}
}

/*! \fn void check_parsevals_theorem_field_3d(double **u, FFTW_Grid_Info grid_info, rfftwnd_mpi_plan plan, rfftwnd_mpi_plan iplan, int myid, int numprocs, MPI_Comm world)
 *  \brief Check for energy conservation in power spectra for a 3-d vector field. 
void check_parsevals_theorem_field_3d(double **u, FFTW_Grid_Info grid_info, rfftwnd_mpi_plan plan, rfftwnd_mpi_plan iplan, int myid, int numprocs, MPI_Comm world)
{

	double *work;
	double *ukx, *uky, *ukz;
	double *Ek;

	fftw_complex *cukx, *cuky, *cukz;
	fftw_complex *cEk;

	double total_specific_energy = grid_field_total_specific_energy(u, grid_info, plan, iplan, myid, numprocs, world);

	double energy;
	int ijk;
	int nx = grid_info.nx;
	int ny = grid_info.ny;
	int nz = grid_info.nz;
	int nx_local = grid_info.nx_local;
	int local_ny_after_transpose = grid_info.local_ny_after_transpose;

	
	double *k_array;
	double *P_k_array;
	double Pk;
	double kx, ky, kz, kk;
	int nk, ik;
	double wk;
	int local_y_start_after_transpose = grid_info.local_y_start_after_transpose;

	char variable_name[200];

	double energy_check;

	//initialize k array
	//initialize_power_spectrum_k_array(k_array, &nk, grid_info, myid, numprocs, world);

	if(grid_info.ndim==2)
	{
		if(myid==0)
		{
			printf("check_parsevals_theorem_field is not implemented for 2 dimensions.\n");
			fflush(stdout);
		}
		MPI_Abort(world,-1);
		exit(-1);
	}

	if(myid==0)
	{
		printf("check_parsevals_theorem_field : total specfic energy %10.9e\n",total_specific_energy);
	}
	energy_check = total_specific_energy;

	//allocate fft workspace 
	sprintf(variable_name,"work");
	work   = allocate_real_fftw_grid(grid_info.total_local_size, variable_name, myid, numprocs, world,0);

	//allocate fft workspace 
	sprintf(variable_name,"cEk");
	cEk = allocate_complex_fftw_grid(grid_info.total_local_size, variable_name, myid, numprocs, world,0);
	
	//copy u[] into ukx, uky, ukz
	ukx = grid_copy(u[0], grid_info, plan, iplan, myid, numprocs, world);
	uky = grid_copy(u[1], grid_info, plan, iplan, myid, numprocs, world);
	ukz = grid_copy(u[2], grid_info, plan, iplan, myid, numprocs, world);

	//perform the forward transform on the components of u
	forward_transform_fftw_grid(ukx, work, plan, cukx, grid_info, myid, numprocs, world);
	forward_transform_fftw_grid(uky, work, plan, cuky, grid_info, myid, numprocs, world);
	forward_transform_fftw_grid(ukz, work, plan, cukz, grid_info, myid, numprocs, world);

	//typecast the complex transform pointers
	cukx = (fftw_complex *) ukx;
	cuky = (fftw_complex *) uky;
	cukz = (fftw_complex *) ukz;


	total_specific_energy = 0;
	for(int j=0;j<local_ny_after_transpose;j++)
		for(int i=0;i<nx;i++)
		{
			//first, positive kz
			for(int k=0;k<(nz/2+1);k++)
			{
				ijk = grid_transpose_ijk(i,j,k,grid_info);
				total_specific_energy += 0.5*( cukx[ijk].re*cukx[ijk].re +  cukx[ijk].im*cukx[ijk].im);
				total_specific_energy += 0.5*( cuky[ijk].re*cuky[ijk].re +  cuky[ijk].im*cuky[ijk].im);
				total_specific_energy += 0.5*( cukz[ijk].re*cukz[ijk].re +  cukz[ijk].im*cukz[ijk].im);

				cEk[ijk].re  = 0.5*( cukx[ijk].re*cukx[ijk].re +  cukx[ijk].im*cukx[ijk].im);
				cEk[ijk].re += 0.5*( cuky[ijk].re*cuky[ijk].re +  cuky[ijk].im*cuky[ijk].im);
				cEk[ijk].re += 0.5*( cukz[ijk].re*cukz[ijk].re +  cukz[ijk].im*cukz[ijk].im);
			}

			//then for negative kz, needed because we sum over all frequencies
			for(int k=1;k<(nz/2);k++)
			{
				ijk = grid_transpose_ijk(i,j,k,grid_info);
				total_specific_energy += 0.5*( cukx[ijk].re*cukx[ijk].re +  cukx[ijk].im*cukx[ijk].im);
				total_specific_energy += 0.5*( cuky[ijk].re*cuky[ijk].re +  cuky[ijk].im*cuky[ijk].im);
				total_specific_energy += 0.5*( cukz[ijk].re*cukz[ijk].re +  cukz[ijk].im*cukz[ijk].im);
			}
		}	

	energy = 0;
	MPI_Allreduce(&total_specific_energy,&energy,1,MPI_DOUBLE,MPI_SUM,world);
	if(myid==0)
	{
		printf("check_parsevals_theorem_field : total specfic energy check 1 %10.9e\n",energy * grid_info.dVk);
	}

	
	total_specific_energy = 0;
	for(int j=0;j<local_ny_after_transpose;j++)
		for(int i=0;i<nx;i++)
		{
			//first, positive kz
			for(int k=0;k<(nz/2+1);k++)
			{
				ijk = grid_transpose_ijk(i,j,k,grid_info);
				total_specific_energy += cEk[ijk].re;
			}

			//then for negative kz, needed because we sum over all frequencies
			for(int k=1;k<(nz/2);k++)
			{
				ijk = grid_transpose_ijk(i,j,k,grid_info);
				total_specific_energy += cEk[ijk].re;
			}
		}	

	energy = 0;
	MPI_Allreduce(&total_specific_energy,&energy,1,MPI_DOUBLE,MPI_SUM,world);
	if(myid==0)
	{
		printf("check_parsevals_theorem_field : total specfic energy check 2 %10.9e\n",energy * grid_info.dVk);
	}



	//get energy power spectrum

	energy_power_spectrum(k_array, P_k_array, &nk, u, grid_info, plan, iplan, myid, numprocs, world);

	//check energy power spectrum
	energy= 0;
	for(int i=0;i<nk;i++)
	{
		energy+= P_k_array[i]*P_k_array[nk+i];
	}
	if(myid==0)
	{
		printf("check_parsevals_theorem_field : total specfic energy check 3 %10.9e (error = %e) (nk = %d)\n",energy,(energy-energy_check)/energy_check,nk);
	}


	free(P_k_array);
	free(k_array);


	free(ukx);
	free(uky);
	free(ukz);
	free(work);
	free(cEk);

	
}
/*! \fn void check_parsevals_theorem_field_2d(double **u, FFTW_Grid_Info grid_info, rfftwnd_mpi_plan plan, rfftwnd_mpi_plan iplan, int myid, int numprocs, MPI_Comm world)
 *  \brief Check for energy conservation in power spectra for a vector field. 
void check_parsevals_theorem_field_2d(double **u, FFTW_Grid_Info grid_info, rfftwnd_mpi_plan plan, rfftwnd_mpi_plan iplan, int myid, int numprocs, MPI_Comm world)
{
	double *work;
	double *ukx, *uky;
	double *Ek;

	fftw_complex *cukx, *cuky;
	fftw_complex *cEk;

	double total_specific_energy = grid_field_total_specific_energy(u, grid_info, plan, iplan, myid, numprocs, world);

	double energy;
	int ijk;
	int nx = grid_info.nx;
	int ny = grid_info.ny;
	int nx_local = grid_info.nx_local;
	int local_ny_after_transpose = grid_info.local_ny_after_transpose;

	
	double *k_array;
	double *P_k_array;
	double Pk;
	double kx, ky, kk;
	int nk, ik;
	double wk;
	int local_y_start_after_transpose = grid_info.local_y_start_after_transpose;
	int jmin;

	char variable_name[200];

	double energy_check;

	//initialize k array
	//initialize_power_spectrum_k_array(k_array, &nk, grid_info, myid, numprocs, world);

	if(myid==0)
	{
		printf("check_parsevals_theorem_field_2d : total specfic energy %10.9e\n",total_specific_energy);
	}
	energy_check = total_specific_energy;

	//allocate fft workspace 
	sprintf(variable_name,"work");
	work   = allocate_real_fftw_grid(grid_info.total_local_size, variable_name, myid, numprocs, world,0);

	//allocate fft workspace 
	sprintf(variable_name,"cEk");
	cEk = allocate_complex_fftw_grid(grid_info.total_local_size, variable_name, myid, numprocs, world,0);
	
	//copy u[] into ukx, uky, ukz
	ukx = grid_copy(u[0], grid_info, plan, iplan, myid, numprocs, world);
	uky = grid_copy(u[1], grid_info, plan, iplan, myid, numprocs, world);

	//perform the forward transform on the components of u
	forward_transform_fftw_grid(ukx, work, plan, cukx, grid_info, myid, numprocs, world);
	forward_transform_fftw_grid(uky, work, plan, cuky, grid_info, myid, numprocs, world);

	//typecast the complex transform pointers
	cukx = (fftw_complex *) ukx;
	cuky = (fftw_complex *) uky;


	total_specific_energy = 0;
	for(int j=0;j<local_ny_after_transpose;j++)
		for(int i=0;i<nx;i++)
		{
			//first, positive ky
			ijk = grid_transpose_ijk(i,j,0,grid_info);
			total_specific_energy += 0.5*( cukx[ijk].re*cukx[ijk].re +  cukx[ijk].im*cukx[ijk].im);
			total_specific_energy += 0.5*( cuky[ijk].re*cuky[ijk].re +  cuky[ijk].im*cuky[ijk].im);

			cEk[ijk].re  = 0.5*( cukx[ijk].re*cukx[ijk].re +  cukx[ijk].im*cukx[ijk].im);
			cEk[ijk].re += 0.5*( cuky[ijk].re*cuky[ijk].re +  cuky[ijk].im*cuky[ijk].im);
		}

	if(myid==0)
	{
		jmin = 1;
	}else{
		jmin = 0;
	}

	for(int j=jmin;j<local_ny_after_transpose;j++)
		for(int i=0;i<nx;i++)
		{
			//then for negative ky, needed because we sum over all frequencies
			ijk = grid_transpose_ijk(i,j,0,grid_info);
			total_specific_energy += 0.5*( cukx[ijk].re*cukx[ijk].re +  cukx[ijk].im*cukx[ijk].im);
			total_specific_energy += 0.5*( cuky[ijk].re*cuky[ijk].re +  cuky[ijk].im*cuky[ijk].im);
		}	

	energy = 0;
	MPI_Allreduce(&total_specific_energy,&energy,1,MPI_DOUBLE,MPI_SUM,world);
	if(myid==0)
	{
		printf("check_parsevals_theorem_field_2d : total specfic energy check 1 %10.9e\n",energy);
	}

	
	total_specific_energy = 0;
	for(int j=0;j<local_ny_after_transpose;j++)
		for(int i=0;i<nx;i++)
		{
			//first, positive ky
			ijk = grid_transpose_ijk(i,j,0,grid_info);
			total_specific_energy += cEk[ijk].re;
		}

	for(int j=jmin;j<local_ny_after_transpose;j++)
		for(int i=0;i<nx;i++)
		{
			//then for negative ky, needed because we sum over all frequencies
			ijk = grid_transpose_ijk(i,j,0,grid_info);
			total_specific_energy += cEk[ijk].re;
		}	

	energy = 0;
	MPI_Allreduce(&total_specific_energy,&energy,1,MPI_DOUBLE,MPI_SUM,world);
	if(myid==0)
	{
		printf("check_parsevals_theorem_field_2d : total specfic energy check 2 %10.9e\n",energy);
	}



	//get energy power spectrum

	energy_power_spectrum(k_array, P_k_array, &nk, u, grid_info, plan, iplan, myid, numprocs, world);


	//check energy power spectrum
	energy= 0;
	for(int i=0;i<nk;i++)
	{
		energy+= P_k_array[i]*P_k_array[nk+i];
	}
	if(myid==0)
	{
		printf("check_parsevals_theorem_field_2d : total specfic energy check 3 %10.9e (error = %e) (nk = %d)\n",energy,(energy-energy_check)/energy_check,nk);
	}


	free(P_k_array);
	free(k_array);


	free(ukx);
	free(uky);
	free(work);
	free(cEk);
}
	

void energy_power_spectrum_1d(double *&k_array, double *&P_k_array, int *nk_return, double **u, FFTW_Grid_Info grid_info, rfftwnd_mpi_plan plan, rfftwnd_mpi_plan iplan, int myid, int numprocs, MPI_Comm world)
{
	//CHECK dk^3 normalization
	//get the n-d spectrum
	energy_power_spectrum(k_array, P_k_array, nk_return, u, grid_info, plan, iplan, myid, numprocs, world);

	if(grid_info.ndim==2)
	{
		if(myid==0)
			printf("ndim = %d Ek\n",grid_info.ndim);
		//get the 2-d spectrum

		//multiply by 2 pi k to get the 1-d spectrum
		for(int i=0;i<*nk_return;i++)
			P_k_array[i] *= 2*M_PI*(2*M_PI*k_array[i]/grid_info.BoxSize);
	}else{
		//get the 3-d spectrum
		//energy_power_spectrum(k_array, P_k_array, nk_return, u, grid_info, plan, iplan, myid, numprocs, world);

		//multiply by 4 pi k^2 to get the 1-d spectrum
		for(int i=0;i<*nk_return;i++)
			P_k_array[i] *= 4*M_PI*pow(2*M_PI*k_array[i]/grid_info.BoxSize,2);
	}
}

void energy_power_spectrum(double *&k_array, double *&P_k_array, int *nk_return, double **u, FFTW_Grid_Info grid_info, rfftwnd_mpi_plan plan, rfftwnd_mpi_plan iplan, int myid, int numprocs, MPI_Comm world)
{
	double *work;
	double **uk;

	fftw_complex **cdata;

	char variable_name[200];
	int nk;
	double dk;



	//initialize k array
	initialize_power_spectrum_k_array(k_array, &nk, grid_info, myid, numprocs, world);
	*nk_return = nk;

	//allocate fft workspace 
	sprintf(variable_name,"work");
	work   = allocate_real_fftw_grid(grid_info.total_local_size, variable_name, myid, numprocs, world,0);

	//allocate uk workspace 
	sprintf(variable_name,"work");
	uk = allocate_field_fftw_grid(grid_info.ndim, grid_info.total_local_size, variable_name, myid, numprocs, world,0);




	//copy u[] into ukx, uky, ukz
	for(int n=0;n<grid_info.ndim;n++)
		grid_copy_in_place(u[n], uk[n], grid_info, plan, iplan, myid, numprocs, world);

	cdata = new fftw_complex *[grid_info.ndim];



	//perform the forward transform on the components of u
	for(int n=0;n<grid_info.ndim;n++)
		forward_transform_fftw_grid(uk[n], work, plan, cdata[n], grid_info, myid, numprocs, world);



	//typecast the complex transform pointers
	for(int n=0;n<grid_info.ndim;n++)
		cdata[n] = (fftw_complex *) uk[n];


	//get power spectrum
	P_k_array = power_spectrum_field(cdata, k_array, nk, grid_info, myid, numprocs, world);

	fflush(stdout);
	MPI_Finalize();
	exit(0);

	
	//normalize by 1/2 because it's 1/2 |u(k)|^2
	for(int i=0;i<nk;i++)
		P_k_array[i]*=0.5;




	//free memory
	free(cdata);
	free(work);
	deallocate_field_fftw_grid(uk, grid_info.ndim, grid_info.total_local_size, myid, numprocs, world);
}


/*
void energy_power_spectrum_2d(double *&k_array, double *&P_k_array, int *nk_return, double **u, FFTW_Grid_Info grid_info, rfftwnd_mpi_plan plan, rfftwnd_mpi_plan iplan, int myid, int numprocs, MPI_Comm world)
{
	double *work;
	double *ukx, *uky;

	fftw_complex *cukx, *cuky;

	char variable_name[200];
	int nk;


	//initialize k array
	initialize_power_spectrum_k_array(k_array, &nk, grid_info, myid, numprocs, world);
	*nk_return = nk;

	//allocate fft workspace 
	sprintf(variable_name,"work");
	work   = allocate_real_fftw_grid(grid_info.total_local_size, variable_name, myid, numprocs, world,0);

	//copy u[] into ukx, uky, ukz
	ukx = grid_copy(u[0], grid_info, plan, iplan, myid, numprocs, world);
	uky = grid_copy(u[1], grid_info, plan, iplan, myid, numprocs, world);

	//perform the forward transform on the components of u
	forward_transform_fftw_grid(ukx, work, plan, cukx, grid_info, myid, numprocs, world);
	forward_transform_fftw_grid(uky, work, plan, cuky, grid_info, myid, numprocs, world);

	//typecast the complex transform pointers
	cukx = (fftw_complex *) ukx;
	cuky = (fftw_complex *) uky;


	//get power spectrum
	P_k_array = power_spectrum_field_2d(cukx, cuky, k_array, nk, grid_info, myid, numprocs, world);
	
	//normalize by 1/2 because it's 1/2 |u(k)|^2
	for(int i=0;i<nk;i++)
		P_k_array[i]*=0.5;


	free(ukx);
	free(uky);
	free(work);

	
}


/*! \fn void construct_histogram_log10(double *&x_array, double *&P_x_array, int *nx_return, double *u, FFTW_Grid_Info grid_info, rfftwnd_mpi_plan plan, rfftwnd_mpi_plan iplan, int myid, int numprocs, MPI_Comm world)
 *  \brief Histogram the grid u w/a log10 abcissa.
void construct_histogram_log10(double *&x_array, double *&P_x_array, int *nx_return, double *u, FFTW_Grid_Info grid_info, rfftwnd_mpi_plan plan, rfftwnd_mpi_plan iplan, int myid, int numprocs, MPI_Comm world)
{
	int nx;

	double x_min;
	double x_max;
	char variable_name[200];


	x_min = grid_min(u, grid_info, plan, iplan, myid, numprocs, world);
	x_max = grid_max(u, grid_info, plan, iplan, myid, numprocs, world);

	x_min = log10(0.9*x_min);
	x_max = log10(1.1*x_max);


	//MPI_Allreduce(&,S_l_array_sum,nl,MPI_DOUBLE,MPI_SUM,world);

	//initialize x array
	initialize_histogram_x_array_log10(x_array, &nx, x_min, x_max, grid_info, myid, numprocs, world);

	*nx_return = nx;


	//get the histogram
	P_x_array  = grid_histogram_log10(u, x_array, *nx_return, grid_info, myid, numprocs, world);

}

/*! \fn double *grid_histogram_log10(double *u, double *x_array, int nx_return, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world)
 *  \brief Histogram a grid in log10 bins 
double *grid_histogram_log10(double *u, double *x_array, int nx_return, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world)
{
	int i, j, k, ijk;

	int ix;
	double x, dx;
	double *P_x, *P_x_local;
	double nxx, nxtotal;
	int nx_local = grid_info.nx_local;
	int nx       = grid_info.nx;
	int ny       = grid_info.ny;
	int nz       = grid_info.nz;

	char variable_name[200];

	if(grid_info.ndim==2)
		nz=1;

	dx = x_array[1]-x_array[0];

	//allocate pdf
	sprintf(variable_name,"P_x");
	P_x       = allocate_double_array(nx_return,variable_name,myid,numprocs,world,0);
	P_x_local = allocate_double_array(nx_return,variable_name,myid,numprocs,world,0);

	//set the grid to value
	for(i=0;i<nx_local;++i)
		for(j=0;j<ny;++j)
			for(k=0;k<nz;++k)
			{
				ijk = grid_ijk(i,j,k,grid_info);

				if(u[ijk]>0)
				{
					x = (log10(u[ijk]) - x_array[0])/dx;
					ix = (int) x;
					P_x_local[ix] += 1.0;	
				}
			}

	//sum components across particles
	MPI_Allreduce(P_x_local,P_x,nx_return,MPI_DOUBLE,MPI_SUM,world);

	free(P_x_local);

	//normalize histogram
	nxx=0;
	for(i=0;i<nx_return;++i)
		nxx+=P_x[i]*dx;
	for(i=0;i<nx_return;++i)
		P_x[i]/=nxx;

	//return histogram
	return P_x;
}



/*! \fn void initialize_histogram_x_array_log10(double *&x_array, int *nx, double log10_x_min, double log10_x_max, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world)
 *  \brief Routine to initialize the x_array for a histogram
void initialize_histogram_x_array_log10(double *&x_array, int *nx, double log10_x_min, double log10_x_max, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world)
{
	char variable_name[200];
	
	double dx;

	*nx = 100;

	dx = (log10_x_max - log10_x_min)/((double) *nx);

	sprintf(variable_name,"x_array");
	x_array = allocate_double_array( *nx,   variable_name, myid, numprocs, world, 0);

	for(int i=0;i<(*nx);i++)
	{
		x_array[i] = dx * ((double) i) + log10_x_min + 0.5*dx;
	}
}


/*! \fn void construct_grid_power_spectrum(double *&k_array, double *&P_k, int *nk_return, double *u, FFTW_Grid_Info grid_info, rfftwnd_mpi_plan plan, rfftwnd_mpi_plan iplan, int myid, int numprocs, MPI_Comm world)
 *  \brief Calculate a power spectrum for a grid.
void construct_grid_power_spectrum(double *&k_array, double *&P_k, int *nk_return, double *u, FFTW_Grid_Info grid_info, rfftwnd_mpi_plan plan, rfftwnd_mpi_plan iplan, int myid, int numprocs, MPI_Comm world)
{

	double *uk;
	double *work;

	fftw_complex *cdata;

	int nk;

	char variable_name[200];

	//allocate fft workspace 
	sprintf(variable_name,"work");
	work   = allocate_real_fftw_grid(grid_info.total_local_size, variable_name, myid, numprocs, world,0);
	
	//copy u[] into ukx, uky, ukz
	uk = grid_copy(u, grid_info, plan, iplan, myid, numprocs, world);

	//perform the forward transform on the components of u
	forward_transform_fftw_grid(uk, work, plan, cdata, grid_info, myid, numprocs, world);

	//typecast the complex transform pointers
	cdata = (fftw_complex *) uk;

	//get a properly zeroed array

	if(myid==0)
	{
		printf("\n");
		printf("In construct_grid_power_spectrum.\n");
		fflush(stdout);
	}

	//initialize k array
	initialize_power_spectrum_k_array(k_array, &nk, grid_info, myid, numprocs, world);

	*nk_return = nk;

	//get the power spectrum
	P_k = power_spectrum(cdata, k_array, nk, grid_info, myid, numprocs, world);


	//free memory
	free(uk);
	free(work);
}


/*! \fn void construct_grid_power_spectrum_field(double *&k_array, double *&P_k, int *nk_return, double **u, FFTW_Grid_Info grid_info, rfftwnd_mpi_plan plan, rfftwnd_mpi_plan iplan, int myid, int numprocs, MPI_Comm world)
 *  \brief Calculate a power spectrum for a field.
void construct_grid_power_spectrum_field(double *&k_array, double *&P_k, int *nk_return, double **u, FFTW_Grid_Info grid_info, rfftwnd_mpi_plan plan, rfftwnd_mpi_plan iplan, int myid, int numprocs, MPI_Comm world)
{

	double **uk;
	double *work;

	fftw_complex **cdata;

	int nk;

	char variable_name[200];

	//allocate fft workspace 
	sprintf(variable_name,"work");
	work   = allocate_real_fftw_grid(grid_info.total_local_size, variable_name, myid, numprocs, world,0);


	//allocate fft grid
	sprintf(variable_name,"uk");
	uk = allocate_field_fftw_grid(grid_info.ndim, grid_info.total_local_size, variable_name, myid, numprocs, world, 0);

	//copy u[] into ukx, uky, ukz
	//uk = grid_copy(u, grid_info, plan, iplan, myid, numprocs, world);
	for(int n=0;n<grid_info.ndim;n++)
		grid_copy_in_place(u[n], uk[n], grid_info, plan, iplan, myid, numprocs, world);	

	cdata = new fftw_complex *[grid_info.ndim];

	//perform the forward transform on the components of u
	for(int n=0;n<grid_info.ndim;n++)
		forward_transform_fftw_grid(uk[n], work, plan, cdata[n], grid_info, myid, numprocs, world);


	//typecast the complex transform pointers
	for(int n=0;n<grid_info.ndim;n++)
		cdata[n] = (fftw_complex *) uk[n];

	//get a properly zeroed array

	if(myid==0)
	{
		printf("\n");
		printf("In construct_grid_power_spectrum_field.\n");
		fflush(stdout);
	}

	//initialize k array
	initialize_power_spectrum_k_array(k_array, &nk, grid_info, myid, numprocs, world);

	*nk_return = nk;

	//get the power spectrum
	/*if(grid_info.ndim==2)
	{
		P_k = power_spectrum_field_2d(cdata[0], cdata[1], k_array, nk, grid_info, myid, numprocs, world);
	}else{
		P_k = power_spectrum_field(cdata[0], cdata[1], cdata[2], k_array, nk, grid_info, myid, numprocs, world);
	}

	P_k = power_spectrum_field(cdata, k_array, nk, grid_info, myid, numprocs, world);

	//free memory
	//free(uk);
	free(cdata);
	deallocate_field_fftw_grid(uk, grid_info.ndim, grid_info.total_local_size, myid, numprocs, world);
	free(work);
}


/*! \fn void construct_field_structure_function(double alpha, double *&l_array, double *&S_l_array, int *nl, double **u, double *x, double *y, double *z, FFTW_Grid_Info grid_info, rfftwnd_mpi_plan plan, rfftwnd_mpi_plan iplan, int myid, int numprocs, MPI_Comm world)
 *  \brief Calculate the parallel structure function for a field.
void construct_field_structure_function(double alpha, double *&l_array, double *&S_l_array, int *nl_save, double **u, double *x, double *y, double *z, FFTW_Grid_Info grid_info, rfftwnd_mpi_plan plan, rfftwnd_mpi_plan iplan, int myid, int numprocs, MPI_Comm world)
{

	if(grid_info.ndim==2)
	{
		//get the parallel structure function
		construct_field_structure_function_mode_2d(0, alpha, l_array, S_l_array, nl_save, u, x, y, z, grid_info, plan, iplan, myid, numprocs, world);
	}else{
		//get the parallel structure function
		construct_field_structure_function_mode(0, alpha, l_array, S_l_array, nl_save, u, x, y, z, grid_info, plan, iplan, myid, numprocs, world);
	}
}

/*! \fn void construct_field_perpendicular_structure_function(double alpha, double *&l_array, double *&S_l_array, int *nl, double **u, double *x, double *y, double *z, FFTW_Grid_Info grid_info, rfftwnd_mpi_plan plan, rfftwnd_mpi_plan iplan, int myid, int numprocs, MPI_Comm world)
 *  \brief Calculate the perpendicular structure function for a field.
void construct_field_perpendicular_structure_function(double alpha, double *&l_array, double *&S_l_array, int *nl_save, double **u, double *x, double *y, double *z, FFTW_Grid_Info grid_info, rfftwnd_mpi_plan plan, rfftwnd_mpi_plan iplan, int myid, int numprocs, MPI_Comm world)
{
	if(grid_info.ndim==2)
	{
		//get the perpendicular structure function
		construct_field_structure_function_mode_2d(1, alpha, l_array, S_l_array, nl_save, u, x, y, z, grid_info, plan, iplan, myid, numprocs, world);
	}else{
		//get the perpendicular structure function
		construct_field_structure_function_mode(1, alpha, l_array, S_l_array, nl_save, u, x, y, z, grid_info, plan, iplan, myid, numprocs, world);
	}
}

/*! \fn void construct_field_magnitude_structure_function(double alpha, double *&l_array, double *&S_l_array, int *nl, double **u, double *x, double *y, double *z, FFTW_Grid_Info grid_info, rfftwnd_mpi_plan plan, rfftwnd_mpi_plan iplan, int myid, int numprocs, MPI_Comm world)
 *  \brief Calculate the perpendicular structure function for a field.
void construct_field_magnitude_structure_function(double alpha, double *&l_array, double *&S_l_array, int *nl_save, double **u, double *x, double *y, double *z, FFTW_Grid_Info grid_info, rfftwnd_mpi_plan plan, rfftwnd_mpi_plan iplan, int myid, int numprocs, MPI_Comm world)
{
	if(grid_info.ndim==2)
	{
		//get the magnitude structure function
		construct_field_structure_function_mode_2d(2, alpha, l_array, S_l_array, nl_save, u, x, y, z, grid_info, plan, iplan, myid, numprocs, world);
	}else{
		//get the magnitude structure function
		construct_field_structure_function_mode(2, alpha, l_array, S_l_array, nl_save, u, x, y, z, grid_info, plan, iplan, myid, numprocs, world);
	}
}

/*! \fn void construct_field_structure_function_mode(int mode, double alpha, double *&l_array, double *&S_l_array, int *nl, double **u, double *x, double *y, double *z, FFTW_Grid_Info grid_info, rfftwnd_mpi_plan plan, rfftwnd_mpi_plan iplan, int myid, int numprocs, MPI_Comm world)
 *  \brief Driving function to calculate the structure functions for a field.
 *
 *   mode==0 / default -> parallel structure function
 *   mode==1           -> perpendicular structure function
 *   mode==2           -> velocity difference magnitude structure function
 
void construct_field_structure_function_mode_tmp(int mode, double alpha, double *&l_array, double *&S_l_array, int *nl_save, double **u, double *x, double *y, double *z, FFTW_Grid_Info grid_info, rfftwnd_mpi_plan plan, rfftwnd_mpi_plan iplan, int myid, int numprocs, MPI_Comm world)
{
	int nx_local      = grid_info.nx_local;
	int nx_local_start = grid_info.nx_local_start;
	int total_local_size = grid_info.total_local_size;
	int nx            = grid_info.nx;
	int ny            = grid_info.ny;
	int nz            = grid_info.nz;

	int ijk;
	int iijjkk;

	int ibuf;	

	double l_min;
	double l_max;
	int nl = 100;

	int *nx_local_array;
	int *nx_local_start_array;
	int *total_local_size_array;

	char variable_name[200];

	int i, j, k;
	int il;

	int ips, ipr;

	double S_l, l, dl, xx, yy, zz;
	double dv[3];
	double lhat[3];

	double **ul;

	FFTW_Grid_Info gi;

	gi.nx = nx;
	gi.ny = ny;
	gi.nz = nz;

	MPI_Status status;	

	double BoxSizeX;
	double BoxSizeY;
	double BoxSizeZ;

	double *S_l_array_proc;
	double *S_l_array_sum;
	double *norm_proc;
	double *norm;

	double *vperp;
	double wk;
	double norm_tot;

	int x_bw_flag=0;
	int y_bw_flag=0;
	int z_bw_flag=0;

	int nskip;

	if(grid_info.ndim==2)
	{
		if(myid==0)
		{
			printf("structure function not implmented in 2-d.\n");
			fflush(stdout);
		}
		MPI_Abort(world,-1);
		exit(-1);
	}


	BoxSizeX = x[nx-1]-x[0] + (x[1]-x[0]);
	BoxSizeY = y[ny-1]-y[0] + (y[1]-y[0]);
	BoxSizeZ = z[nz-1]-z[0] + (z[1]-z[0]);

	sprintf(variable_name,"nx_local_array");
	nx_local_array = allocate_int_array(numprocs,variable_name,myid,numprocs,world,0);

	sprintf(variable_name,"nx_local_start_array");
	nx_local_start_array = allocate_int_array(numprocs,variable_name,myid,numprocs,world,0);

	sprintf(variable_name,"total_local_size_array");
	total_local_size_array = allocate_int_array(numprocs,variable_name,myid,numprocs,world,0);

	//record the nx_local and nx_local_start of each process

	for(int ip=0;ip<numprocs;ip++)
	{
		if(ip==myid)
			ibuf = nx_local;
		MPI_Bcast(&ibuf,1,MPI_INT,ip,world);
		nx_local_array[ip] = ibuf;
		if(ip==myid)
			ibuf = nx_local_start;
		MPI_Bcast(&ibuf,1,MPI_INT,ip,world);
		nx_local_start_array[ip] = ibuf;
		if(ip==myid)
			ibuf = total_local_size;
		MPI_Bcast(&ibuf,1,MPI_INT,ip,world);
		total_local_size_array[ip] = ibuf;
	}


	//doing full (n^3)^2 is ridiculous
	//instead, we'll translate along only
	//the x-direction, but at two special
	//locations -- j==0, k==0 and j==(ny/2),
	//k==(nz/2).  This will let us make use
	//of the smallest and largest displacements.

	//allocate the l_array and S_l_arrays

	sprintf(variable_name,"S_l_array");
	S_l_array = allocate_double_array(2*nl,variable_name,myid,numprocs,world,0);

	sprintf(variable_name,"S_l_array_proc");
	S_l_array_proc = allocate_double_array(nl,variable_name,myid,numprocs,world,0);
	sprintf(variable_name,"S_l_array_sum");
	S_l_array_sum = allocate_double_array(nl,variable_name,myid,numprocs,world,0);

	sprintf(variable_name,"norm_proc");
	norm_proc = allocate_double_array(nl,variable_name,myid,numprocs,world,0);
	sprintf(variable_name,"norm");
	norm = allocate_double_array(nl,variable_name,myid,numprocs,world,0);


	//initialize the l array
	initialize_structure_function_l_array(l_array, &nl, &l_min, &l_max, x, y, z, grid_info, myid, numprocs, world);
	*nl_save = nl;


	//begin calculation.  The loop is over the number of processors.
	//Each processor sends its data to ips and receives its data from
	//processor ipr.	

	//just do proc 0
	ipr = 0;


	//initialize the grid_info for the partner processor ipr
	gi.nx_local = nx_local_array[ipr];
	gi.nx_local_start = nx_local_start_array[ipr];

	//allocate the velocity field buffer to receive the u field
	//from ipr
	sprintf(variable_name,"u");
	ul = allocate_field_fftw_grid(grid_info.ndim,total_local_size_array[ipr],variable_name,myid,numprocs,world,0);

	if(myid==0)
		grid_field_copy_in_place(u, ul, grid_info, plan, iplan, myid, numprocs, world);

	//We sendrecv to send and receive the data to/from ips/ipr.
	for(int n=0;n<grid_info.ndim;n++)
		MPI_Bcast(ul[n],total_local_size_array[ipr]*sizeof(double),MPI_BYTE,ipr,world);

	//small scale
	for(int i=0;i<nx_local;i++)
	{
		if(myid==0)
		{
			printf(".");
			fflush(stdout);
		}
		for(int j=0;j<ny;j++)
			for(int k=0;k<nz;k++)
				for(int ii=0;ii<nx_local_array[ipr];ii++)
				{
					int jj=j;
					int kk=k;
							//if(! ( (ii==i)&&(jj==j)&&(kk==k) ) )	
							if(1)
							{
  
								//the x-direction separation between our cell and ipr's cell
								xx = x[nx_local_start_array[ipr]+ii]-x[nx_local_start + i];

								//the y-direction separation between our cell and ipr's cell
								yy = y[jj]-y[j];
						
								//the z-direction separation between our cell and ipr's cell
								zz = z[kk]-z[k];

								//since the box is periodic, then the maximum separation
								//can only be 1/2 the box size in each x,y,z direction.
								//so perform a box wrap if necessary

								//the separation vector magnitude, simply
								//the distance between cells
								l  = sqrt(xx*xx + yy*yy + zz*zz);
		
								//find the location of the separation vector
					 			//magnitude in l_array
								//il = gsl_interp_bsearch(l_array,l,0,nl);
								il = gsl_interp_bsearch(l_array,l,0,nl-1);

								//if the separation is not zero, take the
								//dot product of lhat with the velocity
								//difference
								if(l!=0.0)
								{
									//project of l along unit vector
									lhat[0] = xx/l;
									lhat[1] = yy/l;
									lhat[2] = zz/l;

									//this should never happen

									ijk    = grid_ijk(i,j,k,grid_info);
									//iijjkk = grid_ijk(ii,jj,kk,grid_info);
									iijjkk = grid_ijk(ii,jj,kk,gi);

									//S_l_|| = <{ [v(r+l) - v(r)] dot l/|l|}^p>
									//S_l_|| = <(dvx*lx + dvy*ly + dvz*lz)^p>

									for(int n=0;n<grid_info.ndim;n++)	
										dv[n] = ul[n][iijjkk] - u[n][ijk];
										//dv[n] = u[n][iijjkk] - u[n][ijk];

									switch(mode)
									{
											/////////////////////
											//find the perpendicular velocity difference
											/////////////////////
										case 1: vperp = (double *) calloc(grid_info.ndim,sizeof(double));
											S_l = vector_dot_product(dv, lhat, grid_info.ndim);
	
											//subtract the parallel projection from dv to leave the perp v-difference
											for(int n=0;n<grid_info.ndim;n++)	
												vperp[n] = dv[n] - S_l*lhat[n];

											//find the magnitude of the perpendicular velocity difference
											//S_l = vector_magnitude(vperp, grid_info.ndim); 
											S_l = vector_magnitude(vperp, grid_info.ndim)/sqrt(2); 
											free(vperp);
											break;

											/////////////////////
											//find the absolute velocity difference
											/////////////////////
										case 2: S_l = vector_magnitude(dv, grid_info.ndim); 
											break;

											/////////////////////
											//find the parallel velocity difference
											/////////////////////
										default: S_l = vector_dot_product(dv, lhat, grid_info.ndim);
											break;
									}

									//interpolate onto the separation vector array 
									//S_l_array_proc[il] += pow( S_l, alpha);
									if((l>=l_array[0])&&(l<=l_array[nl-1]))
									{
										wk = (l - l_array[il])/(l_array[il+1]-l_array[il]);
		
										if(il>=0 && il<nl)
				 						{		

											S_l_array_proc[il]   += ( 1.0-wk )*pow(fabs(S_l), alpha);
											norm_proc[il]        += ( 1.0-wk );
										}
										if((il+1)>=0 && (il+1)<nl)
										{			
											S_l_array_proc[il+1] += (     wk )*pow(fabs(S_l), alpha);
											norm_proc[il+1]      += (     wk );
										}
									}
								}
							}
			}
	}
	if(myid==0)
	{
		printf("done!\n");
		fflush(stdout);
	}



	//free memory
	deallocate_field_fftw_grid(ul,grid_info.ndim,total_local_size_array[ipr],myid,numprocs,world);

	//normalize
	//for(il= 0;il<nl;il++)
	//	if(norm_proc[il]>0)
	//		S_l_array_proc[il] /= norm_proc[il];

	//Sum structure function
	MPI_Allreduce(S_l_array_proc,S_l_array_sum,nl,MPI_DOUBLE,MPI_SUM,world);
	MPI_Allreduce(norm_proc,     norm,     nl,MPI_DOUBLE,MPI_SUM,world);
	//MPI_Allreduce(S_l_array_proc,S_l_array_sum,nl,MPI_DOUBLE,MPI_SUM,world);
	//MPI_Allreduce(norm_proc,     norm,     nl,MPI_DOUBLE,MPI_SUM,world);

	//normalize
	//for(il= 0;il<nl;il++)
	//	if(norm[il]>0)
	//		S_l_array_sum[il] /= (double) numprocs;
	for(il= 0;il<nl;il++)
		if(norm[il]>0)
			S_l_array_sum[il] /= norm[il];


	for(il=0;il<nl;il++)
	{	
		//l_array[il] += 0.5*dl;//shift to bin center
		S_l_array[il] = S_l_array_sum[il];
		S_l_array[nl+il] = norm[il];
	}

	norm_tot=0;
	for(i=0;i<nl;i++)
	{
		norm_tot+= norm[i];
		//if(myid==0)
		//	printf("i %d il %15.f norm %e\n",i,norm_tot,norm[i]);
	}

	if(myid==0)
		printf("total n = %15f\n",norm_tot);
	

	//free memory
	free(S_l_array_proc);
	free(S_l_array_sum);
	free(norm_proc);
	free(norm);

	free(nx_local_array);
	free(nx_local_start_array);
	free(total_local_size_array);

}


/*! \fn void construct_structure_function_random(double alpha, double *&l_array, double *&S_l_array, int *nl, double *u, double *x, double *y, double *z, FFTW_Grid_Info grid_info, rfftwnd_mpi_plan plan, rfftwnd_mpi_plan iplan, int myid, int numprocs, MPI_Comm world)
 *  \brief Function to calculate the structure functions for a scalar for random pairs.
 
void construct_structure_function_random(double alpha, double *&l_array, double *&S_l_array, int *nl_save, double *u, double *x, double *y, double *z, FFTW_Grid_Info grid_info, rfftwnd_mpi_plan plan, rfftwnd_mpi_plan iplan, int myid, int numprocs, MPI_Comm world)
{
	int nx_local      = grid_info.nx_local;
	int nx_local_start = grid_info.nx_local_start;
	int total_local_size = grid_info.total_local_size;
	int nx            = grid_info.nx;
	int ny            = grid_info.ny;
	int nz            = grid_info.nz;

	int ijk;
	int iijjkk;

	int ibuf;	

	double l_min;
	double l_max;
	int nl = 100;

	int *nx_local_array;
	int *nx_local_start_array;
	int *total_local_size_array;

	char variable_name[200];

	int i, j, k;
	int ii, jj, kk;
	int il;

	int ips, ipr;

	double S_l, l, dl, xx, yy, zz;
	double dv[3];
	double lhat[3];

	double *ul;

	FFTW_Grid_Info gi;

	gi.nx = nx;
	gi.ny = ny;
	gi.nz = nz;

	MPI_Status status;	

	double BoxSizeX;
	double BoxSizeY;
	double BoxSizeZ;

	double *S_l_array_proc;
	double *S_l_array_sum;
	double *norm_proc;
	double *norm;

	double *vperp;
	double wk;
	double norm_tot;

	int x_bw_flag=0;
	int y_bw_flag=0;
	int z_bw_flag=0;

	int nskip;

	int *flag_calc;

	//random numbers

	int n_samples = (int) pow(nx_local*ny*nz,1.35);
	const gsl_rng_type *T;
	gsl_rng *r;

	if(grid_info.ndim==2)
	{
		if(myid==0)
		{
			printf("structure function not implmented in 2-d.\n");
			fflush(stdout);
		}
		MPI_Abort(world,-1);
		exit(-1);
	}

	gsl_rng_env_setup();

	T = gsl_rng_default;
	r = gsl_rng_alloc(T);

	BoxSizeX = x[nx-1]-x[0] + (x[1]-x[0]);
	BoxSizeY = y[ny-1]-y[0] + (y[1]-y[0]);
	BoxSizeZ = z[nz-1]-z[0] + (z[1]-z[0]);

	sprintf(variable_name,"nx_local_array");
	nx_local_array = allocate_int_array(numprocs,variable_name,myid,numprocs,world,0);

	sprintf(variable_name,"nx_local_start_array");
	nx_local_start_array = allocate_int_array(numprocs,variable_name,myid,numprocs,world,0);

	sprintf(variable_name,"total_local_size_array");
	total_local_size_array = allocate_int_array(numprocs,variable_name,myid,numprocs,world,0);

	//record the nx_local and nx_local_start of each process

	for(int ip=0;ip<numprocs;ip++)
	{
		if(ip==myid)
			ibuf = nx_local;
		MPI_Bcast(&ibuf,1,MPI_INT,ip,world);
		nx_local_array[ip] = ibuf;
		if(ip==myid)
			ibuf = nx_local_start;
		MPI_Bcast(&ibuf,1,MPI_INT,ip,world);
		nx_local_start_array[ip] = ibuf;
		if(ip==myid)
			ibuf = total_local_size;
		MPI_Bcast(&ibuf,1,MPI_INT,ip,world);
		total_local_size_array[ip] = ibuf;
	}


	//doing full (n^3)^2 is ridiculous
	//instead, we'll translate along only
	//the x-direction, but at two special
	//locations -- j==0, k==0 and j==(ny/2),
	//k==(nz/2).  This will let us make use
	//of the smallest and largest displacements.

	//allocate the l_array and S_l_arrays

	sprintf(variable_name,"S_l_array");
	S_l_array = allocate_double_array(2*nl,variable_name,myid,numprocs,world,0);

	sprintf(variable_name,"S_l_array_proc");
	S_l_array_proc = allocate_double_array(nl,variable_name,myid,numprocs,world,0);
	sprintf(variable_name,"S_l_array_sum");
	S_l_array_sum = allocate_double_array(nl,variable_name,myid,numprocs,world,0);

	sprintf(variable_name,"norm_proc");
	norm_proc = allocate_double_array(nl,variable_name,myid,numprocs,world,0);
	sprintf(variable_name,"norm");
	norm = allocate_double_array(nl,variable_name,myid,numprocs,world,0);


	//initialize the l array
	initialize_structure_function_l_array(l_array, &nl, &l_min, &l_max, x, y, z, grid_info, myid, numprocs, world);
	*nl_save = nl;


	dl = (l_max - l_min)/((double) (nl)); 

	sprintf(variable_name,"flag_calc");
	flag_calc = allocate_int_array(numprocs,variable_name,myid,numprocs,world,0);
	for(int ip=0;ip<numprocs;ip++)
		flag_calc[ip]=0;

	//begin calculation.  The loop is over the number of processors.
	//Each processor sends its data to ips and receives its data from
	//processor ipr.	
	for(int ip=0;ip<numprocs/2+1;ip++)
	{

		//find send/receive processor partner	
		ips = myid + ip;
		if(ips>=numprocs)
			ips-=numprocs;
		ipr = myid - ip;
		if(ipr<0)
			ipr+=numprocs;

		//printf("ip %d myid %d ips %d ipr %d\n",ip,myid,ips,ipr);
		//fflush(stdout);
		if(myid==0)
		{
			printf(".");
			fflush(stdout);
		}

		//initialize the grid_info for the partner processor ipr
		gi.nx_local = nx_local_array[ipr];
		gi.nx_local_start = nx_local_start_array[ipr];

		//allocate the velocity field buffer to receive the u field
		//from ipr
		sprintf(variable_name,"ul");
		//ul = allocate_field_fftw_grid(grid_info.ndim,total_local_size_array[ipr],variable_name,myid,numprocs,world,0);
		ul = allocate_real_fftw_grid(total_local_size_array[ipr],variable_name,myid,numprocs,world,0);


		//We sendrecv to send and receive the data to/from ips/ipr.
		if(ip==0)
		{

			//use our own velocity field
			//grid_field_copy_in_place(u, ul, grid_info, plan, iplan, myid, numprocs, world);
			grid_copy_in_place(u, ul, grid_info, plan, iplan, myid, numprocs, world);
		}else{

			//send and receive velocity field
			MPI_Sendrecv(u,total_local_size*sizeof(double),MPI_BYTE,ips,myid,ul,total_local_size_array[ipr]*sizeof(double),MPI_BYTE,ipr,ipr,world,&status);
			//MPI_Sendrecv(u[n],total_local_size*sizeof(double),MPI_BYTE,ips,myid,ul[n],total_local_size_array[ipr]*sizeof(double),MPI_BYTE,ipr,ipr,world,&status);

		}


		//limit double counting	
		if( (!flag_calc[ipr])&&(! ( (ipr==ips)&&(myid>ipr) ) ) )
		{


			//instead, we will loop over random samplings of
			//each dimension

			for(int isamp=0;isamp<n_samples;isamp++)
			{

				i  = gsl_rng_uniform_int(r,nx_local);
				j  = gsl_rng_uniform_int(r,ny);
				k  = gsl_rng_uniform_int(r,nz);
				ii = gsl_rng_uniform_int(r,nx_local_array[ipr]);
				jj = gsl_rng_uniform_int(r,ny);
				kk = gsl_rng_uniform_int(r,nz);

				//prevent comparing the same cell
				if(! ((nx_local_start+i==nx_local_start_array[ipr]+ii)&&(j==jj)&&(k==kk)) )
				{
	
					//the x-direction separation between our cell and ipr's cell
					xx = x[nx_local_start_array[ipr]+ii]-x[nx_local_start + i];

					//the y-direction separation between our cell and ipr's cell
					yy = y[jj]-y[j];
						
					//the z-direction separation between our cell and ipr's cell
					zz = z[kk]-z[k];
				
					//since the box is periodic, then the maximum separation
					//can only be 1/2 the box size in each x,y,z direction.
					//so perform a box wrap if necessary
					if(xx>0.5*BoxSizeX)
					{
						xx-= BoxSizeX;
						x_bw_flag = 1;
					}
					if(yy>0.5*BoxSizeY)
					{
						yy-= BoxSizeY;
						y_bw_flag = 1;
					}
					if(zz>0.5*BoxSizeZ)
					{
						zz-= BoxSizeZ;
						z_bw_flag = 1;
					}
					if(xx<-0.5*BoxSizeX)
					{
						xx+= BoxSizeX;
						x_bw_flag = 1;
					}
					if(yy<-0.5*BoxSizeY)
					{
						yy+= BoxSizeY;
						y_bw_flag = 1;
					}
					if(zz<-0.5*BoxSizeZ)
					{
						zz+= BoxSizeZ;
						z_bw_flag = 1;
					}

					//the separation vector magnitude, simply
					//the distance between cells
					l  = sqrt(xx*xx + yy*yy + zz*zz);
				
	
					//find the location of the separation vector
					//magnitude in l_array
					//il = gsl_interp_bsearch(l_array,l,0,nl);
					il = gsl_interp_bsearch(l_array,l,0,nl-1);

					//if the separation is not zero, take the
					//dot product of lhat with the velocity
					//difference
					if(l!=0.0)
					{
						ijk    = grid_ijk(i,j,k,grid_info);
						iijjkk = grid_ijk(ii,jj,kk,gi);

						//the grid is scalar, only need the absolute difference
						S_l = fabs(ul[iijjkk] - u[ijk]);

						//interpolate onto the separation vector array 
						if((l>=l_array[0])&&(l<=l_array[nl-1]))
						{
							wk = (l - l_array[il])/(l_array[il+1]-l_array[il]);
	
							if(il>=0 && il<nl)
 							{		

								S_l_array_proc[il]   += ( 1.0-wk )*pow(fabs(S_l), alpha);
								norm_proc[il]        += ( 1.0-wk );
							}
							if((il+1)>=0 && (il+1)<nl)
							{			
								S_l_array_proc[il+1] += (     wk )*pow(fabs(S_l), alpha);
								norm_proc[il+1]      += (     wk );
							}
						}
					}
				}
			}
		}


		//free memory
		//deallocate_field_fftw_grid(ul,grid_info.ndim,total_local_size_array[ipr],myid,numprocs,world);
		free(ul);
		//remember which pairs have been calculated
		flag_calc[ips] = 1;
		flag_calc[ipr] = 1;
	}
	if(myid==0)
		printf("\n");

	//normalize
	//for(il= 0;il<nl;il++)
	//	if(norm_proc[il]>0)
	//		S_l_array_proc[il] /= norm_proc[il];

	//Sum structure function
	MPI_Allreduce(S_l_array_proc,S_l_array_sum,nl,MPI_DOUBLE,MPI_SUM,world);
	MPI_Allreduce(norm_proc,     norm,     nl,MPI_DOUBLE,MPI_SUM,world);

	//normalize
	for(il= 0;il<nl;il++)
		if(norm[il]>0)
			S_l_array_sum[il] /= norm[il];


	for(il=0;il<nl;il++)
	{	
		l_array[il] += 0.5*dl;//shift to bin center
		S_l_array[il] = S_l_array_sum[il];
		S_l_array[nl+il] = norm[il];
	}

	norm_tot=0;
	for(i=0;i<nl;i++)
	{
		norm_tot+= norm[i];
		//if(myid==0)
		//	printf("i %d il %15.f norm %e\n",i,norm_tot,norm[i]);
	}

	if(myid==0)
		printf("total n = %15f\n",norm_tot);
	

	//free memory
	free(S_l_array_proc);
	free(S_l_array_sum);
	free(norm_proc);
	free(norm);

	free(nx_local_array);
	free(nx_local_start_array);
	free(total_local_size_array);

	free(flag_calc);
	gsl_rng_free(r);

}

/*! \fn void construct_field_structure_function_mode_random(int mode, double alpha, double *&l_array, double *&S_l_array, int *nl, double **u, double *x, double *y, double *z, FFTW_Grid_Info grid_info, rfftwnd_mpi_plan plan, rfftwnd_mpi_plan iplan, int myid, int numprocs, MPI_Comm world)
 *  \brief Driving function to calculate the structure functions for a field for all pairs.
 *
 *   mode==0 / default -> parallel structure function
 *   mode==1           -> perpendicular structure function
 *   mode==2           -> velocity difference magnitude structure function
 *
 *   Note that the full calculation and the random sampling calculations are nearly identical.
 
void construct_field_structure_function_mode_random(int mode, double alpha, double *&l_array, double *&S_l_array, int *nl_save, double **u, double *x, double *y, double *z, FFTW_Grid_Info grid_info, rfftwnd_mpi_plan plan, rfftwnd_mpi_plan iplan, int myid, int numprocs, MPI_Comm world)
{
	int nx_local      = grid_info.nx_local;
	int nx_local_start = grid_info.nx_local_start;
	int total_local_size = grid_info.total_local_size;
	int nx            = grid_info.nx;
	int ny            = grid_info.ny;
	int nz            = grid_info.nz;

	int ijk;
	int iijjkk;

	int ibuf;	

	double l_min;
	double l_max;
	int nl = 100;

	int *nx_local_array;
	int *nx_local_start_array;
	int *total_local_size_array;

	char variable_name[200];

	int i, j, k;
	int ii, jj, kk;
	int il;

	int ips, ipr;

	double S_l, l, dl, xx, yy, zz;
	double dv[3];
	double lhat[3];

	double **ul;

	FFTW_Grid_Info gi;

	gi.nx = nx;
	gi.ny = ny;
	gi.nz = nz;

	MPI_Status status;	

	double BoxSizeX;
	double BoxSizeY;
	double BoxSizeZ;

	double *S_l_array_proc;
	double *S_l_array_sum;
	double *norm_proc;
	double *norm;

	double *vperp;
	double wk;
	double norm_tot;

	int x_bw_flag=0;
	int y_bw_flag=0;
	int z_bw_flag=0;

	int nskip;

	int *flag_calc;

	//random numbers

	int n_samples = (int) pow(nx_local*ny*nz,1.35);
	const gsl_rng_type *T;
	gsl_rng *r;

	if(grid_info.ndim==2)
	{
		if(myid==0)
		{
			printf("structure function not implmented in 2-d.\n");
			fflush(stdout);
		}
		MPI_Abort(world,-1);
		exit(-1);
	}

	gsl_rng_env_setup();

	T = gsl_rng_default;
	r = gsl_rng_alloc(T);

	BoxSizeX = x[nx-1]-x[0] + (x[1]-x[0]);
	BoxSizeY = y[ny-1]-y[0] + (y[1]-y[0]);
	BoxSizeZ = z[nz-1]-z[0] + (z[1]-z[0]);

	sprintf(variable_name,"nx_local_array");
	nx_local_array = allocate_int_array(numprocs,variable_name,myid,numprocs,world,0);

	sprintf(variable_name,"nx_local_start_array");
	nx_local_start_array = allocate_int_array(numprocs,variable_name,myid,numprocs,world,0);

	sprintf(variable_name,"total_local_size_array");
	total_local_size_array = allocate_int_array(numprocs,variable_name,myid,numprocs,world,0);

	//record the nx_local and nx_local_start of each process

	for(int ip=0;ip<numprocs;ip++)
	{
		if(ip==myid)
			ibuf = nx_local;
		MPI_Bcast(&ibuf,1,MPI_INT,ip,world);
		nx_local_array[ip] = ibuf;
		if(ip==myid)
			ibuf = nx_local_start;
		MPI_Bcast(&ibuf,1,MPI_INT,ip,world);
		nx_local_start_array[ip] = ibuf;
		if(ip==myid)
			ibuf = total_local_size;
		MPI_Bcast(&ibuf,1,MPI_INT,ip,world);
		total_local_size_array[ip] = ibuf;
	}


	//doing full (n^3)^2 is ridiculous
	//instead, we'll translate along only
	//the x-direction, but at two special
	//locations -- j==0, k==0 and j==(ny/2),
	//k==(nz/2).  This will let us make use
	//of the smallest and largest displacements.

	//allocate the l_array and S_l_arrays

	sprintf(variable_name,"S_l_array");
	S_l_array = allocate_double_array(2*nl,variable_name,myid,numprocs,world,0);

	sprintf(variable_name,"S_l_array_proc");
	S_l_array_proc = allocate_double_array(nl,variable_name,myid,numprocs,world,0);
	sprintf(variable_name,"S_l_array_sum");
	S_l_array_sum = allocate_double_array(nl,variable_name,myid,numprocs,world,0);

	sprintf(variable_name,"norm_proc");
	norm_proc = allocate_double_array(nl,variable_name,myid,numprocs,world,0);
	sprintf(variable_name,"norm");
	norm = allocate_double_array(nl,variable_name,myid,numprocs,world,0);


	//initialize the l array
	initialize_structure_function_l_array(l_array, &nl, &l_min, &l_max, x, y, z, grid_info, myid, numprocs, world);
	*nl_save = nl;


	dl = (l_max - l_min)/((double) (nl)); 

	sprintf(variable_name,"flag_calc");
	flag_calc = allocate_int_array(numprocs,variable_name,myid,numprocs,world,0);
	for(int ip=0;ip<numprocs;ip++)
		flag_calc[ip]=0;

	//begin calculation.  The loop is over the number of processors.
	//Each processor sends its data to ips and receives its data from
	//processor ipr.	
	for(int ip=0;ip<numprocs/2+1;ip++)
	{

		//find send/receive processor partner	
		ips = myid + ip;
		if(ips>=numprocs)
			ips-=numprocs;
		ipr = myid - ip;
		if(ipr<0)
			ipr+=numprocs;

		//printf("ip %d myid %d ips %d ipr %d\n",ip,myid,ips,ipr);
		//fflush(stdout);
		if(myid==0)
		{
			printf(".");
			fflush(stdout);
		}

		//initialize the grid_info for the partner processor ipr
		gi.nx_local = nx_local_array[ipr];
		gi.nx_local_start = nx_local_start_array[ipr];

		//allocate the velocity field buffer to receive the u field
		//from ipr
		sprintf(variable_name,"u");
		ul = allocate_field_fftw_grid(grid_info.ndim,total_local_size_array[ipr],variable_name,myid,numprocs,world,0);


		//We sendrecv to send and receive the data to/from ips/ipr.
		if(ip==0)
		{

			//use our own velocity field
			grid_field_copy_in_place(u, ul, grid_info, plan, iplan, myid, numprocs, world);
		}else{

			//send and receive velocity field
			for(int n=0;n<grid_info.ndim;n++)
				MPI_Sendrecv(u[n],total_local_size*sizeof(double),MPI_BYTE,ips,myid,ul[n],total_local_size_array[ipr]*sizeof(double),MPI_BYTE,ipr,ipr,world,&status);

		}


		//limit double counting	
		if( (!flag_calc[ipr])&&(! ( (ipr==ips)&&(myid>ipr) ) ) )
		{

			//instead, we will loop over random samplings of
			//each dimension

			for(int isamp=0;isamp<n_samples;isamp++)
			{

				i  = gsl_rng_uniform_int(r,nx_local);
				j  = gsl_rng_uniform_int(r,ny);
				k  = gsl_rng_uniform_int(r,nz);
				ii = gsl_rng_uniform_int(r,nx_local_array[ipr]);
				jj = gsl_rng_uniform_int(r,ny);
				kk = gsl_rng_uniform_int(r,nz);

				//prevent comparing the same cell
				if(! ((nx_local_start+i==nx_local_start_array[ipr]+ii)&&(j==jj)&&(k==kk)) )
				{
	
					//the x-direction separation between our cell and ipr's cell
					xx = x[nx_local_start_array[ipr]+ii]-x[nx_local_start + i];

					//the y-direction separation between our cell and ipr's cell
					yy = y[jj]-y[j];
						
					//the z-direction separation between our cell and ipr's cell
					zz = z[kk]-z[k];
				
					//since the box is periodic, then the maximum separation
					//can only be 1/2 the box size in each x,y,z direction.
					//so perform a box wrap if necessary
					if(xx>0.5*BoxSizeX)
					{
						xx-= BoxSizeX;
						x_bw_flag = 1;
					}
					if(yy>0.5*BoxSizeY)
					{
						yy-= BoxSizeY;
						y_bw_flag = 1;
					}
					if(zz>0.5*BoxSizeZ)
					{
						zz-= BoxSizeZ;
						z_bw_flag = 1;
					}
					if(xx<-0.5*BoxSizeX)
					{
						xx+= BoxSizeX;
						x_bw_flag = 1;
					}
					if(yy<-0.5*BoxSizeY)
					{
						yy+= BoxSizeY;
						y_bw_flag = 1;
					}
					if(zz<-0.5*BoxSizeZ)
					{
						zz+= BoxSizeZ;
						z_bw_flag = 1;
					}

					//the separation vector magnitude, simply
					//the distance between cells
					l  = sqrt(xx*xx + yy*yy + zz*zz);
				
	
					//find the location of the separation vector
					//magnitude in l_array
					//il = gsl_interp_bsearch(l_array,l,0,nl);
					il = gsl_interp_bsearch(l_array,l,0,nl-1);

					//if the separation is not zero, take the
					//dot product of lhat with the velocity
					//difference
					if(l!=0.0)
					{
						//project of l along unit vector
						lhat[0] = xx/l;
						lhat[1] = yy/l;
						lhat[2] = zz/l;

						//this should never happen
						if(l>l_max)
						{
							printf("? myid %d ii %d jj %d kk %d i %d j %d k %d xx %e yy %e zz %e l %e lmax %e\n",myid,ii,jj,kk,i,j,k,xx,yy,zz,l,l_max);
						}

						ijk    = grid_ijk(i,j,k,grid_info);
						iijjkk = grid_ijk(ii,jj,kk,gi);

						//S_l_|| = <{ [v(r+l) - v(r)] dot l/|l|}^p>
						//S_l_|| = <(dvx*lx + dvy*ly + dvz*lz)^p>

						for(int n=0;n<grid_info.ndim;n++)	
							dv[n] = ul[n][iijjkk] - u[n][ijk];

						switch(mode)
						{
								/////////////////////
								//find the perpendicular velocity difference
								/////////////////////
							case 1: vperp = (double *) calloc(grid_info.ndim,sizeof(double));
								S_l = vector_dot_product(dv, lhat, grid_info.ndim);
	
								//subtract the parallel projection from dv to leave the perp v-difference
								for(int n=0;n<grid_info.ndim;n++)	
									vperp[n] = dv[n] - S_l*lhat[n];

								//find the magnitude of the perpendicular velocity difference
								S_l = vector_magnitude(vperp, grid_info.ndim)/sqrt(2);
								free(vperp);
								break;

								/////////////////////
								//find the absolute velocity difference
								/////////////////////
							case 2: S_l = vector_magnitude(dv, grid_info.ndim); 
								break;

								/////////////////////
								//find the parallel velocity difference
								/////////////////////
							default: S_l = vector_dot_product(dv, lhat, grid_info.ndim);
								break;
						}

						//interpolate onto the separation vector array 
						if((l>=l_array[0])&&(l<=l_array[nl-1]))
						{
							wk = (l - l_array[il])/(l_array[il+1]-l_array[il]);
	
							if(il>=0 && il<nl)
 							{		

								S_l_array_proc[il]   += ( 1.0-wk )*pow(fabs(S_l), alpha);
								norm_proc[il]        += ( 1.0-wk );
							}
							if((il+1)>=0 && (il+1)<nl)
							{			
								S_l_array_proc[il+1] += (     wk )*pow(fabs(S_l), alpha);
								norm_proc[il+1]      += (     wk );
							}
						}
					}
				}
			}
		}


		//free memory
		deallocate_field_fftw_grid(ul,grid_info.ndim,total_local_size_array[ipr],myid,numprocs,world);
		//remember which pairs have been calculated
		flag_calc[ips] = 1;
		flag_calc[ipr] = 1;
	}
	if(myid==0)
		printf("\n");

	//normalize
	//for(il= 0;il<nl;il++)
	//	if(norm_proc[il]>0)
	//		S_l_array_proc[il] /= norm_proc[il];

	//Sum structure function
	MPI_Allreduce(S_l_array_proc,S_l_array_sum,nl,MPI_DOUBLE,MPI_SUM,world);
	MPI_Allreduce(norm_proc,     norm,     nl,MPI_DOUBLE,MPI_SUM,world);

	//normalize
	for(il= 0;il<nl;il++)
		if(norm[il]>0)
			S_l_array_sum[il] /= norm[il];


	for(il=0;il<nl;il++)
	{	
		l_array[il] += 0.5*dl;//shift to bin center
		S_l_array[il] = S_l_array_sum[il];
		S_l_array[nl+il] = norm[il];
	}

	norm_tot=0;
	for(i=0;i<nl;i++)
	{
		norm_tot+= norm[i];
		//if(myid==0)
		//	printf("i %d il %15.f norm %e\n",i,norm_tot,norm[i]);
	}

	if(myid==0)
		printf("total n = %15f\n",norm_tot);
	

	//free memory
	free(S_l_array_proc);
	free(S_l_array_sum);
	free(norm_proc);
	free(norm);

	free(nx_local_array);
	free(nx_local_start_array);
	free(total_local_size_array);

	free(flag_calc);
	gsl_rng_free(r);

}

/*! \fn void construct_field_structure_function_mode_full(int mode, double alpha, double *&l_array, double *&S_l_array, int *nl, double **u, double *x, double *y, double *z, FFTW_Grid_Info grid_info, rfftwnd_mpi_plan plan, rfftwnd_mpi_plan iplan, int myid, int numprocs, MPI_Comm world)
 *  \brief Driving function to calculate the structure functions for a field for all pairs.
 *
 *   mode==0 / default -> parallel structure function
 *   mode==1           -> perpendicular structure function
 *   mode==2           -> velocity difference magnitude structure function
 *
 *   Note that the full calculation and the random sampling calculations are nearly identical.
 
void construct_field_structure_function_mode_full(int mode, double alpha, double *&l_array, double *&S_l_array, int *nl_save, double **u, double *x, double *y, double *z, FFTW_Grid_Info grid_info, rfftwnd_mpi_plan plan, rfftwnd_mpi_plan iplan, int myid, int numprocs, MPI_Comm world)
{
	int nx_local      = grid_info.nx_local;
	int nx_local_start = grid_info.nx_local_start;
	int total_local_size = grid_info.total_local_size;
	int nx            = grid_info.nx;
	int ny            = grid_info.ny;
	int nz            = grid_info.nz;

	int ijk;
	int iijjkk;

	int ibuf;	

	double l_min;
	double l_max;
	int nl = 100;

	int *nx_local_array;
	int *nx_local_start_array;
	int *total_local_size_array;

	char variable_name[200];

	int i, j, k;
	int il;

	int ips, ipr;

	double S_l, l, dl, xx, yy, zz;
	double dv[3];
	double lhat[3];

	double **ul;

	FFTW_Grid_Info gi;

	gi.nx = nx;
	gi.ny = ny;
	gi.nz = nz;

	MPI_Status status;	

	double BoxSizeX;
	double BoxSizeY;
	double BoxSizeZ;

	double *S_l_array_proc;
	double *S_l_array_sum;
	double *norm_proc;
	double *norm;

	double *vperp;
	double wk;
	double norm_tot;

	int x_bw_flag=0;
	int y_bw_flag=0;
	int z_bw_flag=0;

	int nskip;

	int *flag_calc;

	if(grid_info.ndim==2)
	{
		if(myid==0)
		{
			printf("structure function not implmented in 2-d.\n");
			fflush(stdout);
		}
		MPI_Abort(world,-1);
		exit(-1);
	}


	BoxSizeX = x[nx-1]-x[0] + (x[1]-x[0]);
	BoxSizeY = y[ny-1]-y[0] + (y[1]-y[0]);
	BoxSizeZ = z[nz-1]-z[0] + (z[1]-z[0]);

	sprintf(variable_name,"nx_local_array");
	nx_local_array = allocate_int_array(numprocs,variable_name,myid,numprocs,world,0);

	sprintf(variable_name,"nx_local_start_array");
	nx_local_start_array = allocate_int_array(numprocs,variable_name,myid,numprocs,world,0);

	sprintf(variable_name,"total_local_size_array");
	total_local_size_array = allocate_int_array(numprocs,variable_name,myid,numprocs,world,0);

	//record the nx_local and nx_local_start of each process

	for(int ip=0;ip<numprocs;ip++)
	{
		if(ip==myid)
			ibuf = nx_local;
		MPI_Bcast(&ibuf,1,MPI_INT,ip,world);
		nx_local_array[ip] = ibuf;
		if(ip==myid)
			ibuf = nx_local_start;
		MPI_Bcast(&ibuf,1,MPI_INT,ip,world);
		nx_local_start_array[ip] = ibuf;
		if(ip==myid)
			ibuf = total_local_size;
		MPI_Bcast(&ibuf,1,MPI_INT,ip,world);
		total_local_size_array[ip] = ibuf;
	}


	//doing full (n^3)^2 is ridiculous
	//instead, we'll translate along only
	//the x-direction, but at two special
	//locations -- j==0, k==0 and j==(ny/2),
	//k==(nz/2).  This will let us make use
	//of the smallest and largest displacements.

	//allocate the l_array and S_l_arrays

	sprintf(variable_name,"S_l_array");
	S_l_array = allocate_double_array(2*nl,variable_name,myid,numprocs,world,0);

	sprintf(variable_name,"S_l_array_proc");
	S_l_array_proc = allocate_double_array(nl,variable_name,myid,numprocs,world,0);
	sprintf(variable_name,"S_l_array_sum");
	S_l_array_sum = allocate_double_array(nl,variable_name,myid,numprocs,world,0);

	sprintf(variable_name,"norm_proc");
	norm_proc = allocate_double_array(nl,variable_name,myid,numprocs,world,0);
	sprintf(variable_name,"norm");
	norm = allocate_double_array(nl,variable_name,myid,numprocs,world,0);


	//initialize the l array
	initialize_structure_function_l_array(l_array, &nl, &l_min, &l_max, x, y, z, grid_info, myid, numprocs, world);
	*nl_save = nl;


	dl = (l_max - l_min)/((double) (nl)); 

	sprintf(variable_name,"flag_calc");
	flag_calc = allocate_int_array(numprocs,variable_name,myid,numprocs,world,0);
	for(int ip=0;ip<numprocs;ip++)
		flag_calc[ip]=0;

	//begin calculation.  The loop is over the number of processors.
	//Each processor sends its data to ips and receives its data from
	//processor ipr.	
	for(int ip=0;ip<numprocs/2+1;ip++)
	{

		//find send/receive processor partner	
		ips = myid + ip;
		if(ips>=numprocs)
			ips-=numprocs;
		ipr = myid - ip;
		if(ipr<0)
			ipr+=numprocs;

		printf("ip %d myid %d ips %d ipr %d\n",ip,myid,ips,ipr);
		fflush(stdout);

		//initialize the grid_info for the partner processor ipr
		gi.nx_local = nx_local_array[ipr];
		gi.nx_local_start = nx_local_start_array[ipr];

		//allocate the velocity field buffer to receive the u field
		//from ipr
		sprintf(variable_name,"u");
		ul = allocate_field_fftw_grid(grid_info.ndim,total_local_size_array[ipr],variable_name,myid,numprocs,world,0);


		//We sendrecv to send and receive the data to/from ips/ipr.
		if(ip==0)
		{

			//use our own velocity field
			grid_field_copy_in_place(u, ul, grid_info, plan, iplan, myid, numprocs, world);
		}else{

			//send and receive velocity field
			for(int n=0;n<grid_info.ndim;n++)
				MPI_Sendrecv(u[n],total_local_size*sizeof(double),MPI_BYTE,ips,myid,ul[n],total_local_size_array[ipr]*sizeof(double),MPI_BYTE,ipr,ipr,world,&status);

		}

		//send and receive velocity field
		//for(int n=0;n<grid_info.ndim;n++)
		//	MPI_Sendrecv(u[n],total_local_size*sizeof(double),MPI_BYTE,ips,myid,ul[n],total_local_size_array[ipr]*sizeof(double),MPI_BYTE,ipr,ipr,world,&status);

	//small scale

		//if( (!flag_calc[ipr]) )
		if( (!flag_calc[ipr])&&(! ( (ipr==ips)&&(myid>ipr) ) ) )
		for(int i=0;i<nx_local;i++)
		{
			if(myid==0)
			{
				printf(".");
				fflush(stdout);
			}
			for(int j=0;j<ny;j++)
				for(int k=0;k<nz;k++)
					for(int ii=0;ii<nx_local_array[ip];ii++)
						for(int jj=0;jj<ny;jj++)
							for(int kk=0;kk<nz;kk++)
							{
								if(! ((nx_local_start+i==nx_local_start_array[ipr]+ii)&&(j==jj)&&(k==kk)) )
								{
									//the x-direction separation between our cell and ipr's cell
									xx = x[nx_local_start_array[ipr]+ii]-x[nx_local_start + i];

									//the y-direction separation between our cell and ipr's cell
									yy = y[jj]-y[j];
						
									//the z-direction separation between our cell and ipr's cell
									zz = z[kk]-z[k];
				
									//since the box is periodic, then the maximum separation
									//can only be 1/2 the box size in each x,y,z direction.
									//so perform a box wrap if necessary
									if(xx>0.5*BoxSizeX)
									{
										xx-= BoxSizeX;
										x_bw_flag = 1;
									}
									if(yy>0.5*BoxSizeY)
									{
										yy-= BoxSizeY;
										y_bw_flag = 1;
									}
									if(zz>0.5*BoxSizeZ)
									{
										zz-= BoxSizeZ;
										z_bw_flag = 1;
									}

									if(xx<-0.5*BoxSizeX)
									{
										xx+= BoxSizeX;
										x_bw_flag = 1;
									}
									if(yy<-0.5*BoxSizeY)
									{
										yy+= BoxSizeY;
										y_bw_flag = 1;
									}
									if(zz<-0.5*BoxSizeZ)
									{
										zz+= BoxSizeZ;
										z_bw_flag = 1;
									}

									//the separation vector magnitude, simply
									//the distance between cells
									l  = sqrt(xx*xx + yy*yy + zz*zz);
					
		
									//find the location of the separation vector
					 				//magnitude in l_array
									//il = gsl_interp_bsearch(l_array,l,0,nl);
									il = gsl_interp_bsearch(l_array,l,0,nl-1);

									//if the separation is not zero, take the
									//dot product of lhat with the velocity
									//difference
									if(l!=0.0)
									{
										//project of l along unit vector
										lhat[0] = xx/l;
										lhat[1] = yy/l;
										lhat[2] = zz/l;

										//this should never happen
										if(l>l_max)
										{
											printf("? myid %d ii %d jj %d kk %d i %d j %d k %d xx %e yy %e zz %e l %e lmax %e\n",myid,ii,jj,kk,i,j,k,xx,yy,zz,l,l_max);
										}

										ijk    = grid_ijk(i,j,k,grid_info);
										iijjkk = grid_ijk(ii,jj,kk,gi);

										//S_l_|| = <{ [v(r+l) - v(r)] dot l/|l|}^p>
										//S_l_|| = <(dvx*lx + dvy*ly + dvz*lz)^p>

										for(int n=0;n<grid_info.ndim;n++)	
											dv[n] = ul[n][iijjkk] - u[n][ijk];

										switch(mode)
										{
												/////////////////////
												//find the perpendicular velocity difference
												/////////////////////
											case 1: vperp = (double *) calloc(grid_info.ndim,sizeof(double));
												S_l = vector_dot_product(dv, lhat, grid_info.ndim);
	
												//subtract the parallel projection from dv to leave the perp v-difference
												for(int n=0;n<grid_info.ndim;n++)	
													vperp[n] = dv[n] - S_l*lhat[n];

												//find the magnitude of the perpendicular velocity difference
												//S_l = vector_magnitude(vperp, grid_info.ndim); 
												S_l = vector_magnitude(vperp, grid_info.ndim)/sqrt(2); 
												free(vperp);
												break;

												/////////////////////
												//find the absolute velocity difference
												/////////////////////
											case 2: S_l = vector_magnitude(dv, grid_info.ndim); 
												break;

												/////////////////////
												//find the parallel velocity difference
												/////////////////////
											default: S_l = vector_dot_product(dv, lhat, grid_info.ndim);
												break;
										}

										//interpolate onto the separation vector array 
										if((l>=l_array[0])&&(l<=l_array[nl-1]))
										{
											wk = (l - l_array[il])/(l_array[il+1]-l_array[il]);
		
											if(il>=0 && il<nl)
				 							{		

												S_l_array_proc[il]   += ( 1.0-wk )*pow(fabs(S_l), alpha);
												norm_proc[il]        += ( 1.0-wk );
											}
											if((il+1)>=0 && (il+1)<nl)
											{			
												S_l_array_proc[il+1] += (     wk )*pow(fabs(S_l), alpha);
												norm_proc[il+1]      += (     wk );
											}
										}
									}
								}
							}
		}


		//free memory
		deallocate_field_fftw_grid(ul,grid_info.ndim,total_local_size_array[ipr],myid,numprocs,world);
		//remember which pairs have been calculated
		flag_calc[ips] = 1;
		flag_calc[ipr] = 1;
	}
	if(myid==0)
		printf("\n");

	//normalize
	//for(il= 0;il<nl;il++)
	//	if(norm_proc[il]>0)
	//		S_l_array_proc[il] /= norm_proc[il];

	//Sum structure function
	MPI_Allreduce(S_l_array_proc,S_l_array_sum,nl,MPI_DOUBLE,MPI_SUM,world);
	MPI_Allreduce(norm_proc,     norm,     nl,MPI_DOUBLE,MPI_SUM,world);
	//MPI_Allreduce(S_l_array_proc,S_l_array_sum,nl,MPI_DOUBLE,MPI_SUM,world);
	//MPI_Allreduce(norm_proc,     norm,     nl,MPI_DOUBLE,MPI_SUM,world);

	//normalize
	//for(il= 0;il<nl;il++)
	//	if(norm[il]>0)
	//		S_l_array_sum[il] /= (double) numprocs;
	for(il= 0;il<nl;il++)
		if(norm[il]>0)
			S_l_array_sum[il] /= norm[il];


	for(il=0;il<nl;il++)
	{	
		l_array[il] += 0.5*dl;//shift to bin center
		S_l_array[il] = S_l_array_sum[il];
		S_l_array[nl+il] = norm[il];
	}

	norm_tot=0;
	for(i=0;i<nl;i++)
	{
		norm_tot+= norm[i];
		//if(myid==0)
		//	printf("i %d il %15.f norm %e\n",i,norm_tot,norm[i]);
	}

	if(myid==0)
		printf("total n = %15f\n",norm_tot);
	

	//free memory
	free(S_l_array_proc);
	free(S_l_array_sum);
	free(norm_proc);
	free(norm);

	free(nx_local_array);
	free(nx_local_start_array);
	free(total_local_size_array);

}

/*! \fn void construct_field_structure_function_mode(int mode, double alpha, double *&l_array, double *&S_l_array, int *nl, double **u, double *x, double *y, double *z, FFTW_Grid_Info grid_info, rfftwnd_mpi_plan plan, rfftwnd_mpi_plan iplan, int myid, int numprocs, MPI_Comm world)
 *  \brief Driving function to calculate the structure functions for a field.
 *
 *   mode==0 / default -> parallel structure function
 *   mode==1           -> perpendicular structure function
 *   mode==2           -> velocity difference magnitude structure function
 
void construct_field_structure_function_mode(int mode, double alpha, double *&l_array, double *&S_l_array, int *nl_save, double **u, double *x, double *y, double *z, FFTW_Grid_Info grid_info, rfftwnd_mpi_plan plan, rfftwnd_mpi_plan iplan, int myid, int numprocs, MPI_Comm world)
{
	int nx_local      = grid_info.nx_local;
	int nx_local_start = grid_info.nx_local_start;
	int total_local_size = grid_info.total_local_size;
	int nx            = grid_info.nx;
	int ny            = grid_info.ny;
	int nz            = grid_info.nz;

	int ijk;
	int iijjkk;

	int ibuf;	

	double l_min;
	double l_max;
	int nl = 100;

	int *nx_local_array;
	int *nx_local_start_array;
	int *total_local_size_array;

	char variable_name[200];

	int i, j, k;
	int il;

	int ips, ipr;

	double S_l, l, dl, xx, yy, zz;
	double dv[3];
	double lhat[3];

	double **ul;

	FFTW_Grid_Info gi;

	gi.nx = nx;
	gi.ny = ny;
	gi.nz = nz;

	MPI_Status status;	

	double BoxSizeX;
	double BoxSizeY;
	double BoxSizeZ;

	double *S_l_array_proc;
	double *S_l_array_sum;
	double *norm_proc;
	double *norm;

	double *vperp;
	double wk;
	double norm_tot;

	int x_bw_flag=0;
	int y_bw_flag=0;
	int z_bw_flag=0;

	int nskip;

	if(grid_info.ndim==2)
	{
		if(myid==0)
		{
			printf("structure function not implmented in 2-d.\n");
			fflush(stdout);
		}
		MPI_Abort(world,-1);
		exit(-1);
	}


	BoxSizeX = x[nx-1]-x[0] + (x[1]-x[0]);
	BoxSizeY = y[ny-1]-y[0] + (y[1]-y[0]);
	BoxSizeZ = z[nz-1]-z[0] + (z[1]-z[0]);

	sprintf(variable_name,"nx_local_array");
	nx_local_array = allocate_int_array(numprocs,variable_name,myid,numprocs,world,0);

	sprintf(variable_name,"nx_local_start_array");
	nx_local_start_array = allocate_int_array(numprocs,variable_name,myid,numprocs,world,0);

	sprintf(variable_name,"total_local_size_array");
	total_local_size_array = allocate_int_array(numprocs,variable_name,myid,numprocs,world,0);

	//record the nx_local and nx_local_start of each process

	for(int ip=0;ip<numprocs;ip++)
	{
		if(ip==myid)
			ibuf = nx_local;
		MPI_Bcast(&ibuf,1,MPI_INT,ip,world);
		nx_local_array[ip] = ibuf;
		if(ip==myid)
			ibuf = nx_local_start;
		MPI_Bcast(&ibuf,1,MPI_INT,ip,world);
		nx_local_start_array[ip] = ibuf;
		if(ip==myid)
			ibuf = total_local_size;
		MPI_Bcast(&ibuf,1,MPI_INT,ip,world);
		total_local_size_array[ip] = ibuf;
	}


	//doing full (n^3)^2 is ridiculous
	//instead, we'll translate along only
	//the x-direction, but at two special
	//locations -- j==0, k==0 and j==(ny/2),
	//k==(nz/2).  This will let us make use
	//of the smallest and largest displacements.

	//allocate the l_array and S_l_arrays

	sprintf(variable_name,"S_l_array");
	S_l_array = allocate_double_array(2*nl,variable_name,myid,numprocs,world,0);

	sprintf(variable_name,"S_l_array_proc");
	S_l_array_proc = allocate_double_array(nl,variable_name,myid,numprocs,world,0);
	sprintf(variable_name,"S_l_array_sum");
	S_l_array_sum = allocate_double_array(nl,variable_name,myid,numprocs,world,0);

	sprintf(variable_name,"norm_proc");
	norm_proc = allocate_double_array(nl,variable_name,myid,numprocs,world,0);
	sprintf(variable_name,"norm");
	norm = allocate_double_array(nl,variable_name,myid,numprocs,world,0);


	//initialize the l array
	initialize_structure_function_l_array(l_array, &nl, &l_min, &l_max, x, y, z, grid_info, myid, numprocs, world);
	*nl_save = nl;


	dl = (l_max - l_min)/((double) (nl)); 

	//begin calculation.  The loop is over the number of processors.
	//Each processor sends its data to ips and receives its data from
	//processor ipr.	
	for(int ip=0;ip<numprocs;ip++)
	{

		//find send/receive processor partner	
		ips = myid + ip;
		if(ips>=numprocs)
			ips-=numprocs;
		ipr = myid - ip;
		if(ipr<0)
			ipr+=numprocs;


		//initialize the grid_info for the partner processor ipr
		gi.nx_local = nx_local_array[ipr];
		gi.nx_local_start = nx_local_start_array[ipr];

		//allocate the velocity field buffer to receive the u field
		//from ipr
		sprintf(variable_name,"u");
		ul = allocate_field_fftw_grid(grid_info.ndim,total_local_size_array[ipr],variable_name,myid,numprocs,world,0);


		//We sendrecv to send and receive the data to/from ips/ipr.
		if(ip==0)
		{

			//use our own velocity field
			grid_field_copy_in_place(u, ul, grid_info, plan, iplan, myid, numprocs, world);
		}else{

			//send and receive velocity field
			for(int n=0;n<grid_info.ndim;n++)
				MPI_Sendrecv(u[n],total_local_size*sizeof(double),MPI_BYTE,ips,myid,ul[n],total_local_size_array[ipr]*sizeof(double),MPI_BYTE,ipr,ipr,world,&status);

		}

		//send and receive velocity field
		//for(int n=0;n<grid_info.ndim;n++)
		//	MPI_Sendrecv(u[n],total_local_size*sizeof(double),MPI_BYTE,ips,myid,ul[n],total_local_size_array[ipr]*sizeof(double),MPI_BYTE,ipr,ipr,world,&status);

	//small scale
	nskip = 3;
	for(int i=nskip;i<nx_local-nskip;i+=nskip)
		for(int j=nskip;j<ny-nskip;j+=nskip)
			for(int k=nskip;k<nz-nskip;k+=nskip)
				for(int ii=i-nskip;ii<=i+nskip;ii++)
					for(int jj=j-nskip;jj<=j+nskip;jj++)
						for(int kk=k-nskip;kk<=k+nskip;kk++)
							if(! ( (ii==i)&&(jj==j)&&(kk==k) ) )
							{
								//the x-direction separation between our cell and ipr's cell
								xx = x[nx_local_start_array[ipr]+ii]-x[nx_local_start + i];

								//the y-direction separation between our cell and ipr's cell
								yy = y[jj]-y[j];
						
								//the z-direction separation between our cell and ipr's cell
								zz = z[kk]-z[k];

						//since the box is periodic, then the maximum separation
						//can only be 1/2 the box size in each x,y,z direction.
						//so perform a box wrap if necessary
						if(xx>0.5*BoxSizeX)
						{
							xx-= BoxSizeX;
							x_bw_flag = 1;
						}
						if(yy>0.5*BoxSizeY)
						{
							yy-= BoxSizeY;
							y_bw_flag = 1;
						}
						if(zz>0.5*BoxSizeZ)
						{
							zz-= BoxSizeZ;
							z_bw_flag = 1;
						}

						if(xx<-0.5*BoxSizeX)
						{
							xx+= BoxSizeX;
							x_bw_flag = 1;
						}
						if(yy<-0.5*BoxSizeY)
						{
							yy+= BoxSizeY;
							y_bw_flag = 1;
						}
						if(zz<-0.5*BoxSizeZ)
						{
							zz+= BoxSizeZ;
							z_bw_flag = 1;
						}
								//the separation vector magnitude, simply
								//the distance between cells
								l  = sqrt(xx*xx + yy*yy + zz*zz);
		
								//find the location of the separation vector
					 			//magnitude in l_array
								//il = gsl_interp_bsearch(l_array,l,0,nl);
								il = gsl_interp_bsearch(l_array,l,0,nl-1);

								//if the separation is not zero, take the
								//dot product of lhat with the velocity
								//difference
								if(l!=0.0)
								{
									//project of l along unit vector
									lhat[0] = xx/l;
									lhat[1] = yy/l;
									lhat[2] = zz/l;

									//this should never happen
									if(l>l_max)
									{
										printf("? myid %d ii %d jj %d kk %d i %d j %d k %d xx %e yy %e zz %e l %e lmax %e\n",myid,ii,jj,kk,i,j,k,xx,yy,zz,l,l_max);
									}

									ijk    = grid_ijk(i,j,k,grid_info);
									//iijjkk = grid_ijk(ii,jj,kk,grid_info);
									iijjkk = grid_ijk(ii,jj,kk,gi);

									//S_l_|| = <{ [v(r+l) - v(r)] dot l/|l|}^p>
									//S_l_|| = <(dvx*lx + dvy*ly + dvz*lz)^p>

									for(int n=0;n<grid_info.ndim;n++)	
										dv[n] = ul[n][iijjkk] - u[n][ijk];
										//dv[n] = u[n][iijjkk] - u[n][ijk];

									switch(mode)
									{
											/////////////////////
											//find the perpendicular velocity difference
											/////////////////////
										case 1: vperp = (double *) calloc(grid_info.ndim,sizeof(double));
											S_l = vector_dot_product(dv, lhat, grid_info.ndim);
	
											//subtract the parallel projection from dv to leave the perp v-difference
											for(int n=0;n<grid_info.ndim;n++)	
												vperp[n] = dv[n] - S_l*lhat[n];

											//find the magnitude of the perpendicular velocity difference
											//S_l = vector_magnitude(vperp, grid_info.ndim); 
											S_l = vector_magnitude(vperp, grid_info.ndim)/sqrt(2); 
											free(vperp);
											break;

											/////////////////////
											//find the absolute velocity difference
											/////////////////////
										case 2: S_l = vector_magnitude(dv, grid_info.ndim); 
											break;

											/////////////////////
											//find the parallel velocity difference
											/////////////////////
										default: S_l = vector_dot_product(dv, lhat, grid_info.ndim);
											break;
									}

									//interpolate onto the separation vector array 
									//S_l_array_proc[il] += pow( S_l, alpha);
									if((l>=l_array[0])&&(l<=l_array[nl-1]))
									{
										wk = (l - l_array[il])/(l_array[il+1]-l_array[il]);
		
										if(il>=0 && il<nl)
				 						{		

											S_l_array_proc[il]   += ( 1.0-wk )*pow(fabs(S_l), alpha);
											norm_proc[il]        += ( 1.0-wk );
										}
										if((il+1)>=0 && (il+1)<nl)
										{			
											S_l_array_proc[il+1] += (     wk )*pow(fabs(S_l), alpha);
											norm_proc[il+1]      += (     wk );
										}
									}
								}
							}


		//free memory
		deallocate_field_fftw_grid(ul,grid_info.ndim,total_local_size_array[ipr],myid,numprocs,world);
	}

	//normalize
	//for(il= 0;il<nl;il++)
	//	if(norm_proc[il]>0)
	//		S_l_array_proc[il] /= norm_proc[il];

	//Sum structure function
	MPI_Allreduce(S_l_array_proc,S_l_array_sum,nl,MPI_DOUBLE,MPI_SUM,world);
	MPI_Allreduce(norm_proc,     norm,     nl,MPI_DOUBLE,MPI_SUM,world);
	//MPI_Allreduce(S_l_array_proc,S_l_array_sum,nl,MPI_DOUBLE,MPI_SUM,world);
	//MPI_Allreduce(norm_proc,     norm,     nl,MPI_DOUBLE,MPI_SUM,world);

	//normalize
	//for(il= 0;il<nl;il++)
	//	if(norm[il]>0)
	//		S_l_array_sum[il] /= (double) numprocs;
	for(il= 0;il<nl;il++)
		if(norm[il]>0)
			S_l_array_sum[il] /= norm[il];


	for(il=0;il<nl;il++)
	{	
		l_array[il] += 0.5*dl;//shift to bin center
		S_l_array[il] = S_l_array_sum[il];
		S_l_array[nl+il] = norm[il];
	}

	norm_tot=0;
	for(i=0;i<nl;i++)
	{
		norm_tot+= norm[i];
		//if(myid==0)
		//	printf("i %d il %15.f norm %e\n",i,norm_tot,norm[i]);
	}

	if(myid==0)
		printf("total n = %15f\n",norm_tot);
	

	//free memory
	free(S_l_array_proc);
	free(S_l_array_sum);
	free(norm_proc);
	free(norm);

	free(nx_local_array);
	free(nx_local_start_array);
	free(total_local_size_array);

}

/*! \fn void construct_field_structure_function_mode(int mode, double alpha, double *&l_array, double *&S_l_array, int *nl, double **u, double *x, double *y, double *z, FFTW_Grid_Info grid_info, rfftwnd_mpi_plan plan, rfftwnd_mpi_plan iplan, int myid, int numprocs, MPI_Comm world)
 *  \brief Driving function to calculate the structure functions for a field.
 *
 *   mode==0 / default -> parallel structure function
 *   mode==1           -> perpendicular structure function
 *   mode==2           -> velocity difference magnitude structure function
 
void construct_field_structure_function_mode_save(int mode, double alpha, double *&l_array, double *&S_l_array, int *nl_save, double **u, double *x, double *y, double *z, FFTW_Grid_Info grid_info, rfftwnd_mpi_plan plan, rfftwnd_mpi_plan iplan, int myid, int numprocs, MPI_Comm world)
{
	int nx_local      = grid_info.nx_local;
	int nx_local_start = grid_info.nx_local_start;
	int total_local_size = grid_info.total_local_size;
	int nx            = grid_info.nx;
	int ny            = grid_info.ny;
	int nz            = grid_info.nz;

	int ijk;
	int iijjkk;

	int ibuf;	

	double l_min;
	double l_max;
	int nl = 100;

	int *nx_local_array;
	int *nx_local_start_array;
	int *total_local_size_array;

	char variable_name[200];

	int i, j, k;
	int il;

	int ips, ipr;

	double S_l, l, dl, xx, yy, zz;
	double dv[3];
	double lhat[3];

	double **ul;

	FFTW_Grid_Info gi;

	gi.nx = nx;
	gi.ny = ny;
	gi.nz = nz;

	MPI_Status status;	

	double BoxSizeX;
	double BoxSizeY;
	double BoxSizeZ;

	double *S_l_array_proc;
	double *S_l_array_sum;
	double *norm_proc;
	double *norm;

	double *vperp;
	double wk;
	double norm_tot;

	int x_bw_flag=0;
	int y_bw_flag=0;
	int z_bw_flag=0;

	if(grid_info.ndim==2)
	{
		if(myid==0)
		{
			printf("structure function not implmented in 2-d.\n");
			fflush(stdout);
		}
		MPI_Abort(world,-1);
		exit(-1);
	}


	BoxSizeX = x[nx-1]-x[0] + (x[1]-x[0]);
	BoxSizeY = y[ny-1]-y[0] + (y[1]-y[0]);
	BoxSizeZ = z[nz-1]-z[0] + (z[1]-z[0]);

	sprintf(variable_name,"nx_local_array");
	nx_local_array = allocate_int_array(numprocs,variable_name,myid,numprocs,world,0);

	sprintf(variable_name,"nx_local_start_array");
	nx_local_start_array = allocate_int_array(numprocs,variable_name,myid,numprocs,world,0);

	sprintf(variable_name,"total_local_size_array");
	total_local_size_array = allocate_int_array(numprocs,variable_name,myid,numprocs,world,0);

	//record the nx_local and nx_local_start of each process

	for(int ip=0;ip<numprocs;ip++)
	{
		if(ip==myid)
			ibuf = nx_local;
		MPI_Bcast(&ibuf,1,MPI_INT,ip,world);
		nx_local_array[ip] = ibuf;
		if(ip==myid)
			ibuf = nx_local_start;
		MPI_Bcast(&ibuf,1,MPI_INT,ip,world);
		nx_local_start_array[ip] = ibuf;
		if(ip==myid)
			ibuf = total_local_size;
		MPI_Bcast(&ibuf,1,MPI_INT,ip,world);
		total_local_size_array[ip] = ibuf;
	}


	//doing full (n^3)^2 is ridiculous
	//instead, we'll translate along only
	//the x-direction, but at two special
	//locations -- j==0, k==0 and j==(ny/2),
	//k==(nz/2).  This will let us make use
	//of the smallest and largest displacements.

	//allocate the l_array and S_l_arrays

	sprintf(variable_name,"S_l_array");
	S_l_array = allocate_double_array(2*nl,variable_name,myid,numprocs,world,0);

	sprintf(variable_name,"S_l_array_proc");
	S_l_array_proc = allocate_double_array(nl,variable_name,myid,numprocs,world,0);
	sprintf(variable_name,"S_l_array_sum");
	S_l_array_sum = allocate_double_array(nl,variable_name,myid,numprocs,world,0);

	sprintf(variable_name,"norm_proc");
	norm_proc = allocate_double_array(nl,variable_name,myid,numprocs,world,0);
	sprintf(variable_name,"norm");
	norm = allocate_double_array(nl,variable_name,myid,numprocs,world,0);


	//initialize the l array
	initialize_structure_function_l_array(l_array, &nl, &l_min, &l_max, x, y, z, grid_info, myid, numprocs, world);
	*nl_save = nl;


	//begin calculation.  The loop is over the number of processors.
	//Each processor sends its data to ips and receives its data from
	//processor ipr.	
	for(int ip=0;ip<numprocs;ip++)
	{

		//find send/receive processor partner	
		ips = myid + ip;
		if(ips>=numprocs)
			ips-=numprocs;
		ipr = myid - ip;
		if(ipr<0)
			ipr+=numprocs;


		//initialize the grid_info for the partner processor ipr
		gi.nx_local = nx_local_array[ipr];
		gi.nx_local_start = nx_local_start_array[ipr];

		//allocate the velocity field buffer to receive the u field
		//from ipr
		sprintf(variable_name,"u");
		ul = allocate_field_fftw_grid(grid_info.ndim,total_local_size_array[ipr],variable_name,myid,numprocs,world,0);


		//For the first iteration, we use our own data.  Otherwise
		//we sendrecv to send and receive the data to/from ips/ipr.
		if(ip==0)
		{

			//use our own velocity field
			grid_field_copy_in_place(u, ul, grid_info, plan, iplan, myid, numprocs, world);
		}else{

			//send and receive velocity field
			for(int n=0;n<grid_info.ndim;n++)
				MPI_Sendrecv(u[n],total_local_size*sizeof(double),MPI_BYTE,ips,myid,ul[n],total_local_size_array[ipr]*sizeof(double),MPI_BYTE,ipr,ipr,world,&status);

		}



		//first we will compare our j=0, k=0 x-column
		//with all the ii,jj,kk cells of partner processor
		//velocity field

		j=0;	
		k=0;	
		for(int ii=0;ii<nx_local_array[ipr];ii++)
			for(int jj=0;jj<ny;jj++)
				for(int kk=0;kk<nz;kk++)
					for(i=0;i<nx_local;i++)
					{
						//the x-direction separation between our cell and ipr's cell
						xx = x[nx_local_start_array[ipr]+ii]-x[nx_local_start + i];

						//the y-direction separation between our cell and ipr's cell
						yy = y[jj]-y[j];

						//the z-direction separation between our cell and ipr's cell
						zz = z[kk]-z[k];

						//since the box is periodic, then the maximum separation
						//can only be 1/2 the box size in each x,y,z direction.
						//so perform a box wrap if necessary
						if(xx>0.5*BoxSizeX)
						{
							xx-= BoxSizeX;
							x_bw_flag = 1;
						}
						if(yy>0.5*BoxSizeY)
						{
							yy-= BoxSizeY;
							y_bw_flag = 1;
						}
						if(zz>0.5*BoxSizeZ)
						{
							zz-= BoxSizeZ;
							z_bw_flag = 1;
						}

						if(xx<-0.5*BoxSizeX)
						{
							xx+= BoxSizeX;
							x_bw_flag = 1;
						}
						if(yy<-0.5*BoxSizeY)
						{
							yy+= BoxSizeY;
							y_bw_flag = 1;
						}
						if(zz<-0.5*BoxSizeZ)
						{
							zz+= BoxSizeZ;
							z_bw_flag = 1;
						}

						//the separation vector magnitude, simply
						//the distance between cells
						l  = sqrt(xx*xx + yy*yy + zz*zz);
		
						//find the location of the separation vector
						//magnitude in l_array
						//il = gsl_interp_bsearch(l_array,l,0,nl);
						il = gsl_interp_bsearch(l_array,l,0,nl-1);

						//if the separation is not zero, take the
						//dot product of lhat with the velocity
						//difference
						if(l!=0.0)
						{
							//project of l along unit vector
							lhat[0] = xx/l;
							lhat[1] = yy/l;
							lhat[2] = zz/l;

							//this should never happen
							if(l>l_max)
							{
								printf("? myid %d ii %d jj %d kk %d i %d j %d k %d xx %e yy %e zz %e l %e lmax %e\n",myid,ii,jj,kk,i,j,k,xx,yy,zz,l,l_max);
							}

							ijk    = grid_ijk(i,j,k,grid_info);
							iijjkk = grid_ijk(ii,jj,kk,gi);

							//S_l_|| = <{ [v(r+l) - v(r)] dot l/|l|}^p>
							//S_l_|| = <(dvx*lx + dvy*ly + dvz*lz)^p>

							for(int n=0;n<grid_info.ndim;n++)	
								dv[n] = ul[n][iijjkk] - u[n][ijk];

							if(x_bw_flag)
							{
								//dv[0] *= -1;
								//dv[0] = -1*ul[0][iijjkk] - u[0][ijk]; //didn't work
								x_bw_flag = 0;
							}
							if(y_bw_flag)
							{
								//dv[1] *= -1;
								//dv[1] = -1*ul[1][iijjkk] - u[1][ijk];
								y_bw_flag = 0;
							}
							if(z_bw_flag)
							{
								//dv[2] *= -1;
								//dv[2] = -1*ul[2][iijjkk] - u[2][ijk];
								z_bw_flag = 0;
							}

							switch(mode)
							{
									/////////////////////
									//find the perpendicular velocity difference
									/////////////////////
								case 1: vperp = (double *) calloc(grid_info.ndim,sizeof(double));
									S_l = vector_dot_product(dv, lhat, grid_info.ndim);
	
									//subtract the parallel projection from dv to leave the perp v-difference
									for(int n=0;n<grid_info.ndim;n++)	
										vperp[n] = dv[n] - S_l*lhat[n];

									//find the magnitude of the perpendicular velocity difference
									S_l = vector_magnitude(vperp, grid_info.ndim); 
									free(vperp);
									break;

									/////////////////////
									//find the absolute velocity difference
									/////////////////////
								case 2: S_l = vector_magnitude(dv, grid_info.ndim); 
									break;

									/////////////////////
									//find the parallel velocity difference
									/////////////////////
								default: S_l = vector_dot_product(dv, lhat, grid_info.ndim);
									break;
							}

							//interpolate onto the separation vector array 
							//S_l_array_proc[il] += pow( S_l, alpha);
							if((l>=l_array[0])&&(l<=l_array[nl-1]))
							{
								wk = (l - l_array[il])/(l_array[il+1]-l_array[il]);

								if(il>=0 && il<nl)
								{		

									S_l_array_proc[il]   += ( 1.0-wk )*pow(fabs(S_l), alpha);
									norm_proc[il]        += ( 1.0-wk );
								}
								if((il+1)>=0 && (il+1)<nl)
								{	
									S_l_array_proc[il+1] += (     wk )*pow(fabs(S_l), alpha);
									norm_proc[il+1]      += (     wk );
								}
							}
						}
						//norm_proc[il]      += 1.0;
					}


		//repeat for the j=ny/2, k=nz/2 x-column, which is
		//the most separated from j=0, k=0.  Adds another
		//probe
	
		if(!(ny%2))
		{
			j=ny/2;	
		}else{
			j=(ny-1)/2;	
		}
		if(!(nz%2))
		{
			k=nz/2;	
		}else{
			k=(nz-1)/2;	
		}
		for(int ii=0;ii<nx_local_array[ipr];ii++)
			for(int jj=0;jj<ny;jj++)
				for(int kk=0;kk<nz;kk++)
					for(i=0;i<nx_local;i++)
					{
						//find the separation in the x, y, and z directions
						xx = x[nx_local_start_array[ipr]+ii]-x[nx_local_start + i];
						yy = y[jj]-y[j];
						zz = z[kk]-z[k];


						//this appears to be required in some form
						//box wrap if necessary
						if(xx>0.5*BoxSizeX)
						{
							xx-= BoxSizeX;
							x_bw_flag = 1;
						}
						if(yy>0.5*BoxSizeY)
						{
							yy-= BoxSizeY;
							y_bw_flag = 1;
						}
						if(zz>0.5*BoxSizeZ)
						{
							zz-= BoxSizeZ;
							z_bw_flag = 1;
						}

						if(xx<-0.5*BoxSizeX)
						{
							xx+= BoxSizeX;
							x_bw_flag = 1;
						}
						if(yy<-0.5*BoxSizeY)
						{
							yy+= BoxSizeY;
							y_bw_flag = 1;
						}
						if(zz<-0.5*BoxSizeZ)
						{
							zz+= BoxSizeZ;
							z_bw_flag = 1;
						}

						//find the magnitude of the separation
						//vector and its location within l_array
						l  = sqrt(xx*xx + yy*yy + zz*zz);
			
						il = gsl_interp_bsearch(l_array,l,0,nl-1);

						

						if(l!=0.0)
						{	
						
							//project of l along unit vector
							lhat[0] = xx/l;
							lhat[1] = yy/l;
							lhat[2] = zz/l;

							//this should never happen
							if(l>l_max)
							{
								printf("? myid %d ii %d jj %d kk %d i %d j %d k %d xx %e yy %e zz %e l %e lmax %e\n",myid,ii,jj,kk,i,j,k,xx,yy,zz,l,l_max);
							}

							ijk    = grid_ijk(i,j,k,grid_info);
							iijjkk = grid_ijk(ii,jj,kk,gi);

							//S_l_|| = <{ [v(r+l) - v(r)] dot l/|l|}^p>
							//S_l_|| = <(dvx*lx + dvy*ly + dvz*lz)^p>
							for(int n=0;n<grid_info.ndim;n++)	
								dv[n] = ul[n][iijjkk] - u[n][ijk];

							if(x_bw_flag)
							{
								//dv[0] *= -1;
								//dv[0] = -1*ul[0][iijjkk] - u[0][ijk];
								x_bw_flag = 0;
							}
							if(y_bw_flag)
							{
								//dv[1] *= -1;
								//dv[1] = -1*ul[1][iijjkk] - u[1][ijk];
								y_bw_flag = 0;
							}
							if(z_bw_flag)
							{
								//dv[2] *= -1;
								//dv[2] = -1*ul[2][iijjkk] - u[2][ijk];
								z_bw_flag = 0;
							}
							//S_l = fabs(vector_dot_product(dv, lhat));
							switch(mode)
							{
									/////////////////////
									//find the perpendicular velocity difference
									/////////////////////
								case 1: vperp = (double *) calloc(grid_info.ndim,sizeof(double));
									S_l = vector_dot_product(dv, lhat, grid_info.ndim);
									for(int n=0;n<grid_info.ndim;n++)
										vperp[n] = dv[n] - S_l*lhat[n];
									S_l = vector_magnitude(vperp, grid_info.ndim); 
									free(vperp);
									break;

									/////////////////////
									//find the absolute velocity difference
									/////////////////////
								case 2: S_l = vector_magnitude(dv, grid_info.ndim); 
									break;

									/////////////////////
									//find the parallel velocity difference
									/////////////////////
								default: S_l = vector_dot_product(dv, lhat, grid_info.ndim);
									break;
							}
							//S_l_array_proc[il] += pow( S_l, alpha);
							if((l>=l_array[0])&&(l<=l_array[nl-1]))
							{
								wk = (l - l_array[il])/(l_array[il+1]-l_array[il]);

								if(il>=0 && il<nl)
								{		

									S_l_array_proc[il]   += ( 1.0-wk )*pow(fabs(S_l), alpha);
									norm_proc[il]        += ( 1.0-wk );
								}
								if((il+1)>=0 && (il+1)<nl)
								{	
									S_l_array_proc[il+1] += (     wk )*pow(fabs(S_l), alpha);
									norm_proc[il+1]      += (     wk );
								}
							}
						}
						//norm_proc[il]      += 1.0;
					}
	




		//free memory
		deallocate_field_fftw_grid(ul,grid_info.ndim,total_local_size_array[ipr],myid,numprocs,world);
	}


	//Sum structure function
	MPI_Allreduce(S_l_array_proc,S_l_array_sum,nl,MPI_DOUBLE,MPI_SUM,world);
	MPI_Allreduce(norm_proc,     norm,     nl,MPI_DOUBLE,MPI_SUM,world);

	//normalize
	for(il= 0;il<nl;il++)
		if(norm[il]>0)
			S_l_array_sum[il] /= norm[il];


	for(il=0;il<nl;il++)
	{	
		//l_array[il] += 0.5*dl;//shift to bin center
		S_l_array[il] = S_l_array_sum[il];
		S_l_array[nl+il] = norm[il];
	}

	norm_tot=0;
	for(i=0;i<nl;i++)
	{
		norm_tot+= norm[i];
		//if(myid==0)
		//	printf("i %d il %15.f norm %e\n",i,norm_tot,norm[i]);
	}

	if(myid==0)
		printf("total n = %15f\n",norm_tot);
	

	//free memory
	free(S_l_array_proc);
	free(S_l_array_sum);
	free(norm_proc);
	free(norm);

	free(nx_local_array);
	free(nx_local_start_array);
	free(total_local_size_array);

}

/*! \fn void construct_field_ksf(double *&l_array, double *&S_l_array, int *nl, double **u, double *x, double *y, double *z, FFTW_Grid_Info grid_info, rfftwnd_mpi_plan plan, rfftwnd_mpi_plan iplan, int myid, int numprocs, MPI_Comm world)
 *  \brief Calculate the parallel structure function for a field in k-space.
 *
 *   Note that the k-volume weighting makes little difference.  For consistency with the parseval's
 *   theorem routine, we stick to the cubic (non-spherical) weighting.
void construct_field_ksf(double *&l_array, double *&S_l_array, int *nl_save, double **u, double *x, double *y, double *z, FFTW_Grid_Info grid_info, rfftwnd_mpi_plan plan, rfftwnd_mpi_plan iplan, int myid, int numprocs, MPI_Comm world)
{



	double *work;
	double *ukx, *uky, *ukz;
	double *Ek;

	fftw_complex *cukx, *cuky, *cukz;
	fftw_complex *cEk;

	double total_specific_energy = (2./3.)*grid_field_total_specific_energy(u, grid_info, plan, iplan, myid, numprocs, world);

	double kscale = 2.0*M_PI/grid_info.BoxSize;
	double energy;
	int ijk;
	int nx = grid_info.nx;
	int ny = grid_info.ny;
	int nz = grid_info.nz;
	int nx_local = grid_info.nx_local;
	int local_ny_after_transpose = grid_info.local_ny_after_transpose;

	int nl;
	double l_min, l_max;
	double l;	

	double *k_array;
	double *P_k_array;
	double Pk;
	double kx, ky, kz, kk;
	int nk, ik;
	double wk;
	double dk;
	double dl;
	int local_y_start_after_transpose = grid_info.local_y_start_after_transpose;

	char variable_name[200];

	double energy_check;

	double *E_k_array;

	//get energy power spectrum

	energy_power_spectrum(k_array, P_k_array, &nk, u, grid_info, plan, iplan, myid, numprocs, world);

	sprintf(variable_name,"E_k_array");
	E_k_array = allocate_double_array(nk, variable_name, myid, numprocs, world, 0);
	for(int i=0;i<nk;i++)
	{
		k_array[i] += 1.0e-6;
		E_k_array[i] = P_k_array[i]*P_k_array[nk+i];
		//E_k_array[i] = 4.0*M_PI*kscale*kscale*k_array[i]*k_array[i]*P_k_array[i]/pow(2*M_PI,3);
	}

	//check energy power spectrum
	energy= 0;
	for(int i=0;i<nk;i++)
	{
		energy+= E_k_array[i];
		//k_array[i] += 1.0e-6;
		//k_array[i] += 1.0e-9;
	}
	if(myid==0)
	{	
		printf("here\n");
		printf("check_parsevals_theorem_field : total specfic energy check 3 %10.9e (error = %e) (nk = %d)\n",energy,(energy-1.5*total_specific_energy)/(1.5*total_specific_energy),nk);
	}

	//initialize the l array
	initialize_structure_function_l_array(l_array, &nl, &l_min, &l_max, x, y, z, grid_info, myid, numprocs, world);
	*nl_save = nl;

	sprintf(variable_name,"S_l_array");
	S_l_array = allocate_double_array(2*nl,variable_name,myid,numprocs,world,0);

	if(myid==0)
	{
		printf("ka0 %e Ek0 %e Pk0 %e Pkw0 %e\n",k_array[0],E_k_array[0],P_k_array[0],P_k_array[nk]);
		printf("ka1 %e Ek1 %e Pk1 %e Pkw1 %e\n",k_array[1],E_k_array[1],P_k_array[1],P_k_array[nk+1]);
	}

	dl = l_array[1]-l_array[0];
	dk = k_array[1]-k_array[0];
	for(int i=0;i<nl;i++)
	{
		l_array[i] += 0.5*dl;
		l = l_array[i];
		S_l_array[i] = 0.0;
		
		for(int k=0;k<nk;k++)
		{
			kk = kscale*k_array[k];

		  	S_l_array[i] += ( pow(l,3)/3. + l*cos(kk*l)/(kk*kk) - sin(kk*l)/pow(kk,3) )*E_k_array[k];
		  	//S_l_array[i] += ( pow(l,3)/3. + l*cos(kk*l)/(kk*kk) - sin(kk*l)/pow(kk,3) )*E_k_array[k]*dk*kscale;
			
		}

		S_l_array[i] *= 4*pow(l,-3);

		//if(myid==0)
		//	printf("i %d l %e Sl %e\n",i,l,S_l_array[i]);
	}

	free(E_k_array);
	free(P_k_array);
	free(k_array);
}

/*! \fn void construct_field_structure_function_mode_2d(int mode, double alpha, double *&l_array, double *&S_l_array, int *nl, double **u, double *x, double *y, double *z, FFTW_Grid_Info grid_info, rfftwnd_mpi_plan plan, rfftwnd_mpi_plan iplan, int myid, int numprocs, MPI_Comm world)
 *  \brief Driving function to calculate the structure functions for a field.
 *
 *   mode==0 / default -> parallel structure function
 *   mode==1           -> perpendicular structure function
 *   mode==2           -> velocity difference magnitude structure function
 
void construct_field_structure_function_mode_2d(int mode, double alpha, double *&l_array, double *&S_l_array, int *nl_save, double **u, double *x, double *y, double *z, FFTW_Grid_Info grid_info, rfftwnd_mpi_plan plan, rfftwnd_mpi_plan iplan, int myid, int numprocs, MPI_Comm world)
{
	int nx_local      = grid_info.nx_local;
	int nx_local_start = grid_info.nx_local_start;
	int total_local_size = grid_info.total_local_size;
	int nx            = grid_info.nx;
	int ny            = grid_info.ny;
	int nz            = 1;

	int ijk;
	int iijjkk;

	int ibuf;	

	double l_min;
	double l_max;
	int nl = 100;

	int *nx_local_array;
	int *nx_local_start_array;
	int *total_local_size_array;

	char variable_name[200];

	int i, j, k;
	int il;

	int ips, ipr;

	double S_l, l, dl, xx, yy, zz;
	double dv[2];
	double lhat[2];

	double **ul;

	FFTW_Grid_Info gi;

	gi.nx = nx;
	gi.ny = ny;
	gi.nz = nz;
	gi.ndim = grid_info.ndim;

	MPI_Status status;	

	double BoxSizeX;
	double BoxSizeY;

	double *S_l_array_proc;
	double *S_l_array_sum;
	double *norm_proc;
	double *norm;

	double *vperp;
	double wk;
	double norm_tot;

	int x_bw_flag=0;
	int y_bw_flag=0;

	BoxSizeX = x[nx-1]-x[0] + (x[1]-x[0]);
	BoxSizeY = y[ny-1]-y[0] + (y[1]-y[0]);

	sprintf(variable_name,"nx_local_array");
	nx_local_array = allocate_int_array(numprocs,variable_name,myid,numprocs,world,0);

	sprintf(variable_name,"nx_local_start_array");
	nx_local_start_array = allocate_int_array(numprocs,variable_name,myid,numprocs,world,0);

	sprintf(variable_name,"total_local_size_array");
	total_local_size_array = allocate_int_array(numprocs,variable_name,myid,numprocs,world,0);

	//record the nx_local and nx_local_start of each process

	for(int ip=0;ip<numprocs;ip++)
	{
		if(ip==myid)
			ibuf = nx_local;
		MPI_Bcast(&ibuf,1,MPI_INT,ip,world);
		nx_local_array[ip] = ibuf;
		if(ip==myid)
			ibuf = nx_local_start;
		MPI_Bcast(&ibuf,1,MPI_INT,ip,world);
		nx_local_start_array[ip] = ibuf;
		if(ip==myid)
			ibuf = total_local_size;
		MPI_Bcast(&ibuf,1,MPI_INT,ip,world);
		total_local_size_array[ip] = ibuf;
	}


	//doing full (n^3)^2 is ridiculous
	//instead, we'll translate along only
	//the x-direction, but at two special
	//locations -- j==0, k==0 and j==(ny/2),
	//k==(nz/2).  This will let us make use
	//of the smallest and largest displacements.

	//allocate the l_array and S_l_arrays

	sprintf(variable_name,"S_l_array");
	S_l_array = allocate_double_array(2*nl,variable_name,myid,numprocs,world,0);

	sprintf(variable_name,"S_l_array_proc");
	S_l_array_proc = allocate_double_array(nl,variable_name,myid,numprocs,world,0);
	sprintf(variable_name,"S_l_array_sum");
	S_l_array_sum = allocate_double_array(nl,variable_name,myid,numprocs,world,0);

	sprintf(variable_name,"norm_proc");
	norm_proc = allocate_double_array(nl,variable_name,myid,numprocs,world,0);
	sprintf(variable_name,"norm");
	norm = allocate_double_array(nl,variable_name,myid,numprocs,world,0);


	//initialize the l array
	initialize_structure_function_l_array(l_array, &nl, &l_min, &l_max, x, y, z, grid_info, myid, numprocs, world);
	*nl_save = nl;


	//begin calculation.  The loop is over the number of processors.
	//Each processor sends its data to ips and receives its data from
	//processor ipr.	
	for(int ip=0;ip<numprocs;ip++)
	{

		//find send/receive processor partner	
		ips = myid + ip;
		if(ips>=numprocs)
			ips-=numprocs;
		ipr = myid - ip;
		if(ipr<0)
			ipr+=numprocs;


		//initialize the grid_info for the partner processor ipr
		gi.nx_local = nx_local_array[ipr];
		gi.nx_local_start = nx_local_start_array[ipr];

		//allocate the velocity field buffer to receive the u field
		//from ipr
		sprintf(variable_name,"u");
		ul = allocate_field_fftw_grid(grid_info.ndim,total_local_size_array[ipr],variable_name,myid,numprocs,world,0);


		//For the first iteration, we use our own data.  Otherwise
		//we sendrecv to send and receive the data to/from ips/ipr.
		if(ip==0)
		{

			//use our own velocity field
			grid_field_copy_in_place(u, ul, grid_info, plan, iplan, myid, numprocs, world);
		}else{

			//send and receive velocity field
			for(int n=0;n<grid_info.ndim;n++)
				MPI_Sendrecv(u[n],total_local_size*sizeof(double),MPI_BYTE,ips,myid,ul[n],total_local_size_array[ipr]*sizeof(double),MPI_BYTE,ipr,ipr,world,&status);

		}



		//first we will compare our j=0, k=0 x-column
		//with all the ii,jj,kk cells of partner processor
		//velocity field

		j=0;	
		k=0;	
		for(int ii=0;ii<nx_local_array[ipr];ii++)
			for(int jj=0;jj<ny;jj++)
				for(int kk=0;kk<nz;kk++)
					for(i=0;i<nx_local;i++)
					{
						//the x-direction separation between our cell and ipr's cell
						xx = x[nx_local_start_array[ipr]+ii]-x[nx_local_start + i];

						//the y-direction separation between our cell and ipr's cell
						yy = y[jj]-y[j];


						//since the box is periodic, then the maximum separation
						//can only be 1/2 the box size in each x,y,z direction.
						//so perform a box wrap if necessary
						if(xx>0.5*BoxSizeX)
						{
							xx-= BoxSizeX;
							x_bw_flag = 1;
						}
						if(yy>0.5*BoxSizeY)
						{
							yy-= BoxSizeY;
							y_bw_flag = 1;
						}

						if(xx<-0.5*BoxSizeX)
						{
							xx+= BoxSizeX;
							x_bw_flag = 1;
						}
						if(yy<-0.5*BoxSizeY)
						{
							yy+= BoxSizeY;
							y_bw_flag = 1;
						}

						//the separation vector magnitude, simply
						//the distance between cells
						l  = sqrt(xx*xx + yy*yy);
		
						//find the location of the separation vector
						//magnitude in l_array
						//il = gsl_interp_bsearch(l_array,l,0,nl);
						il = gsl_interp_bsearch(l_array,l,0,nl-1);

						//if the separation is not zero, take the
						//dot product of lhat with the velocity
						//difference
						if(l!=0.0)
						{
							//project of l along unit vector
							lhat[0] = xx/l;
							lhat[1] = yy/l;

							//this should never happen
							if(l>l_max)
							{
								printf("? myid %d ii %d jj %d kk %d i %d j %d k %d xx %e yy %e zz %e l %e lmax %e\n",myid,ii,jj,kk,i,j,k,xx,yy,zz,l,l_max);
							}

							ijk    = grid_ijk(i,j,0,grid_info);
							iijjkk = grid_ijk(ii,jj,0,gi);

							//S_l_|| = <{ [v(r+l) - v(r)] dot l/|l|}^p>
							//S_l_|| = <(dvx*lx + dvy*ly + dvz*lz)^p>

							for(int n=0;n<grid_info.ndim;n++)	
								dv[n] = ul[n][iijjkk] - u[n][ijk];

							if(x_bw_flag)
							{
								//dv[0] *= -1;
								//dv[0] = -1*ul[0][iijjkk] - u[0][ijk]; //didn't work
								x_bw_flag = 0;
							}
							if(y_bw_flag)
							{
								//dv[1] *= -1;
								//dv[1] = -1*ul[1][iijjkk] - u[1][ijk];
								y_bw_flag = 0;
							}

							switch(mode)
							{
									/////////////////////
									//find the perpendicular velocity difference
									/////////////////////
								case 1: vperp = (double *) calloc(grid_info.ndim,sizeof(double));
									S_l = vector_dot_product(dv, lhat, grid_info.ndim);
	
									//subtract the parallel projection from dv to leave the perp v-difference
									for(int n=0;n<grid_info.ndim;n++)	
										vperp[n] = dv[n] - S_l*lhat[n];

									//find the magnitude of the perpendicular velocity difference
									S_l = vector_magnitude(vperp, grid_info.ndim); 
									free(vperp);
									break;

									/////////////////////
									//find the absolute velocity difference
									/////////////////////
								case 2: S_l = vector_magnitude(dv, grid_info.ndim); 
									break;

									/////////////////////
									//find the parallel velocity difference
									/////////////////////
								default: S_l = vector_dot_product(dv, lhat, grid_info.ndim);
									break;
							}

							//interpolate onto the separation vector array 
							//S_l_array_proc[il] += pow( S_l, alpha);
							if((l>=l_array[0])&&(l<=l_array[nl-1]))
							{
								wk = (l - l_array[il])/(l_array[il+1]-l_array[il]);

								if(il>=0 && il<nl)
								{		

									S_l_array_proc[il]   += ( 1.0-wk )*pow(fabs(S_l), alpha);
									norm_proc[il]        += ( 1.0-wk );
								}
								if((il+1)>=0 && (il+1)<nl)
								{	
									S_l_array_proc[il+1] += (     wk )*pow(fabs(S_l), alpha);
									norm_proc[il+1]      += (     wk );
								}
							}
						}
						//norm_proc[il]      += 1.0;
					}


		//repeat for the j=ny/2, k=nz/2 x-column, which is
		//the most separated from j=0, k=0.  Adds another
		//probe

		if(!(ny%2))
		{
			j=ny/2;	
		}else{
			j=(ny-1)/2;	
		}
		k = 0;
		for(int ii=0;ii<nx_local_array[ipr];ii++)
			for(int jj=0;jj<ny;jj++)
				for(int kk=0;kk<nz;kk++)
					for(i=0;i<nx_local;i++)
					{
						//find the separation in the x, y, and z directions
						xx = x[nx_local_start_array[ipr]+ii]-x[nx_local_start + i];
						yy = y[jj]-y[j];


						//this appears to be required in some form
						//box wrap if necessary
						if(xx>0.5*BoxSizeX)
						{
							xx-= BoxSizeX;
							x_bw_flag = 1;
						}
						if(yy>0.5*BoxSizeY)
						{
							yy-= BoxSizeY;
							y_bw_flag = 1;
						}

						if(xx<-0.5*BoxSizeX)
						{
							xx+= BoxSizeX;
							x_bw_flag = 1;
						}
						if(yy<-0.5*BoxSizeY)
						{
							yy+= BoxSizeY;
							y_bw_flag = 1;
						}

						//find the magnitude of the separation
						//vector and its location within l_array
						l  = sqrt(xx*xx + yy*yy); 
			
						il = gsl_interp_bsearch(l_array,l,0,nl-1);

						

						if(l!=0.0)
						{	
						
							//project of l along unit vector
							lhat[0] = xx/l;
							lhat[1] = yy/l;

							//this should never happen
							if(l>l_max)
							{
								printf("? myid %d ii %d jj %d kk %d i %d j %d k %d xx %e yy %e zz %e l %e lmax %e\n",myid,ii,jj,kk,i,j,k,xx,yy,zz,l,l_max);
							}

							ijk    = grid_ijk(i,j,k,grid_info);
							iijjkk = grid_ijk(ii,jj,kk,gi);

							//S_l_|| = <{ [v(r+l) - v(r)] dot l/|l|}^p>
							//S_l_|| = <(dvx*lx + dvy*ly + dvz*lz)^p>
							for(int n=0;n<grid_info.ndim;n++)	
								dv[n] = ul[n][iijjkk] - u[n][ijk];

							if(x_bw_flag)
							{
								//dv[0] *= -1;
								//dv[0] = -1*ul[0][iijjkk] - u[0][ijk];
								x_bw_flag = 0;
							}
							if(y_bw_flag)
							{
								//dv[1] *= -1;
								//dv[1] = -1*ul[1][iijjkk] - u[1][ijk];
								y_bw_flag = 0;
							}
							//S_l = fabs(vector_dot_product(dv, lhat));
							switch(mode)
							{
									/////////////////////
									//find the perpendicular velocity difference
									/////////////////////
								case 1: vperp = (double *) calloc(grid_info.ndim,sizeof(double));
									S_l = vector_dot_product(dv, lhat, grid_info.ndim);
									for(int n=0;n<grid_info.ndim;n++)
										vperp[n] = dv[n] - S_l*lhat[n];
									S_l = vector_magnitude(vperp, grid_info.ndim); 
									free(vperp);
									break;

									/////////////////////
									//find the absolute velocity difference
									/////////////////////
								case 2: S_l = vector_magnitude(dv, grid_info.ndim); 
									break;

									/////////////////////
									//find the parallel velocity difference
									/////////////////////
								default: S_l = vector_dot_product(dv, lhat, grid_info.ndim);
									break;
							}
							//S_l_array_proc[il] += pow( S_l, alpha);
							if((l>=l_array[0])&&(l<=l_array[nl-1]))
							{
								wk = (l - l_array[il])/(l_array[il+1]-l_array[il]);

								if(il>=0 && il<nl)
								{		

									S_l_array_proc[il]   += ( 1.0-wk )*pow(fabs(S_l), alpha);
									norm_proc[il]        += ( 1.0-wk );
								}
								if((il+1)>=0 && (il+1)<nl)
								{	
									S_l_array_proc[il+1] += (     wk )*pow(fabs(S_l), alpha);
									norm_proc[il+1]      += (     wk );
								}
							}
						}
						//norm_proc[il]      += 1.0;
					}
	


	

		//free memory
		deallocate_field_fftw_grid(ul,grid_info.ndim,total_local_size_array[ipr],myid,numprocs,world);
	}


	//Sum structure function
	MPI_Allreduce(S_l_array_proc,S_l_array_sum,nl,MPI_DOUBLE,MPI_SUM,world);
	MPI_Allreduce(norm_proc,     norm,     nl,MPI_DOUBLE,MPI_SUM,world);

	//normalize
	for(il= 0;il<nl;il++)
		if(norm[il]>0)
			S_l_array_sum[il] /= norm[il];


	for(il=0;il<nl;il++)
	{	
		//l_array[il] += 0.5*dl;//shift to bin center
		S_l_array[il] = S_l_array_sum[il];
		S_l_array[nl+il] = norm[il];
	}

	norm_tot=0;
	for(i=0;i<nl;i++)
	{
		norm_tot+= norm[i];
		//if(myid==0)
		//	printf("i %d il %15.f norm %e\n",i,norm_tot,norm[i]);
	}

	if(myid==0)
		printf("total n = %15f\n",norm_tot);
	

	//free memory
	free(S_l_array_proc);
	free(S_l_array_sum);
	free(norm_proc);
	free(norm);

	free(nx_local_array);
	free(nx_local_start_array);
	free(total_local_size_array);

}

/*! \fn double *power_spectrum(fftw_complex *cdata, double *k_array, int nk, int local_y_start_after_transpose, int local_ny_after_transpose, int nx, int ny, int nz, double BoxSize, int myid, int numprocs, MPI_Comm world)
 *  \brief This function takes a forward transform (e.g. a complex grid), already normalized by 1./(nx)^ndim and returns a n-dimensional power spectrum vs. k
 
double *power_spectrum(fftw_complex *cdata, double *k_array, int nk, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world)
{
	if(grid_info.ndim==2)
	{
		//2-d
		return power_spectrum_2d(cdata, k_array, nk, grid_info, myid, numprocs, world);
	}else{
		//3-d
		return power_spectrum_3d(cdata, k_array, nk, grid_info, myid, numprocs, world);
	}
}

/*! \fn double *power_spectrum_3d(fftw_complex *cdata, double *k_array, int nk, int local_y_start_after_transpose, int local_ny_after_transpose, int nx, int ny, int nz, double BoxSize, int myid, int numprocs, MPI_Comm world)
 *  \brief This function takes a forward transform (e.g. a complex grid), already normalized by 1./(nx*ny*nz) and returns a 3-dimensional power spectrum vs. k
 *
 *  Note that the last 1/2 of the 2*nk long array that is returned provides the volume of each k mode in the cube.  This volume
 *  is not 4*pi*k^2 dk, since the fft grid is not a sphere.  For comparison purposes using parseval's theorem, it's this volume
 *  element that needs to be used, rather than 4 pi k^2 dk.
  
double *power_spectrum_3d(fftw_complex *cdata, double *k_array, int nk, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world)
{

	double pi = M_PI;

	int nx = grid_info.nx;
	int ny = grid_info.ny;
	int nz = grid_info.nz;
	int local_y_start_after_transpose = grid_info.local_y_start_after_transpose;
	int local_ny_after_transpose = grid_info.local_ny_after_transpose;

	double *P_k_array_proc; //power spectrum P(k)
	double *norm_proc;      //normalization for averaging

	double *P_k_array;      //power spectrum P(k)
	double *norm;           //normalization for averaging

	double *P_k; //power spectrum P(k)

	double kk, kx, ky, kz;

	int ijk, ik;

	double wk, Pk; //linear interp weighting

	int bin1, bin2;


	char variable_name[200];


	if(grid_info.ndim==2)
	{
		if(myid==0)
		{
			printf("power_spectrum not implemented in 2-d.\n");
			fflush(stdout);
		}
		MPI_Abort(world,-1);
		exit(-1);
	}

	//get a properly zeroed array

	sprintf(variable_name,"P_k_array_proc");
	P_k_array_proc = allocate_double_array(nk,   variable_name, myid, numprocs, world, 0);
	sprintf(variable_name,"norm_proc");
	norm_proc      = allocate_double_array(nk,   variable_name, myid, numprocs, world, 0);

	sprintf(variable_name,"P_k_array");
	P_k_array      = allocate_double_array(nk,   variable_name, myid, numprocs, world, 0);
	sprintf(variable_name,"norm");
	norm           = allocate_double_array(nk,   variable_name, myid, numprocs, world, 0);

	sprintf(variable_name,"P_k");
	P_k            = allocate_double_array(2*nk, variable_name, myid, numprocs, world, 0);


	for(int j=0;j<local_ny_after_transpose;j++)
		for(int i=0;i<nx;i++)
		{
			for(int k=0;k<(nz/2+1);k++)
			{

				if(i>nx/2)
				{
					kx = ((double) (i-nx));
				}else{
					kx = ((double) i);
				}
				if(local_y_start_after_transpose+j>ny/2)
				{
					ky = ((double) (local_y_start_after_transpose+j-ny) );
				}else{
					ky = ((double) (local_y_start_after_transpose+j) );
				}
				kz = ((double) k);



				// the magnitude of the k-vector

				kk = sqrt( kx*kx + ky*ky + kz*kz );


				ijk = (j*nx + i)*(nz/2+1) + k;


				// Pk = |delta_k|^2

				Pk = cdata[ijk].re*cdata[ijk].re + cdata[ijk].im*cdata[ijk].im;


				// add the contribution to the binned power spectrum

				if((kk>=k_array[0])&&(kk<=k_array[nk-1]))
				{
					ik = gsl_interp_bsearch(k_array,kk,0,nk);

					wk = (kk - k_array[ik])/(k_array[ik+1]-k_array[ik]);

					if(ik>=0 && ik<nk)
					{	

						P_k_array_proc[ik]   += ( 1.0-wk )*Pk;
						norm_proc[ik]        += ( 1.0-wk );
					}
					if((ik+1)>=0 && (ik+1)<nk)
					{	
						P_k_array_proc[ik+1] += (     wk )*Pk;
						norm_proc[ik+1]      += (     wk );
					}
				}

			}

			//negative kz
			for(int k=1;k<(nz/2);k++)
			{

				if(i>nx/2)
				{
					kx = ((double) (i-nx));
				}else{
					kx = ((double) i);
				}
				if(local_y_start_after_transpose+j>ny/2)
				{
					ky = ((double) (local_y_start_after_transpose+j-ny) );
				}else{
					ky = ((double) (local_y_start_after_transpose+j) );
				}
				kz = -((double) k);



				// the magnitude of the k-vector

				kk = sqrt( kx*kx + ky*ky + kz*kz );


				ijk = (j*nx + i)*(nz/2+1) + k;


				// Pk = |delta_k|^2

				Pk = cdata[ijk].re*cdata[ijk].re + cdata[ijk].im*cdata[ijk].im;


				// add the contribution to the binned power spectrum

				if((kk>=k_array[0])&&(kk<=k_array[nk-1]))
				{
					ik = gsl_interp_bsearch(k_array,kk,0,nk);

					wk = (kk - k_array[ik])/(k_array[ik+1]-k_array[ik]);

					if(ik>=0 && ik<nk)
					{	

						P_k_array_proc[ik]   += ( 1.0-wk )*Pk;
						norm_proc[ik]        += ( 1.0-wk );
					}
					if((ik+1)>=0 && (ik+1)<nk)
					{	
						P_k_array_proc[ik+1] += (     wk )*Pk;
						norm_proc[ik+1]      += (     wk );
					}
				}

			}
		}

	//Sum P(k) contributions from each processor

	MPI_Allreduce(P_k_array_proc,P_k_array,nk,MPI_DOUBLE,MPI_SUM,world);
	MPI_Allreduce(norm_proc,     norm,     nk,MPI_DOUBLE,MPI_SUM,world);

	//normalize

	for(ik = 0;ik<nk;ik++)
		if(norm[ik]>0)
			P_k_array[ik] /= norm[ik];


	for(int i=0;i<nk;i++)
	{	
		//P_k[i] = P_k_array[i];
		P_k[i] = P_k_array[i] * grid_info.dVk;
		P_k[nk+i] = norm[i];
	}

	free(norm);
	free(P_k_array);
	free(norm_proc);
	free(P_k_array_proc);

	return P_k;

}

/*! \fn double *power_spectrum_2d(fftw_complex *cdata, double *k_array, int nk, int local_y_start_after_transpose, int local_ny_after_transpose, int nx, int ny, int nz, double BoxSize, int myid, int numprocs, MPI_Comm world)
 *  \brief This function takes a forward transform (e.g. a complex grid), already normalized by 1./(nx*ny) and returns a 1-dimensional power spectrum vs. k
 *
 *  Note that the last 1/2 of the 2*nk long array that is returned provides the volume of each k mode in the cube.  This volume
 *  is not 2*pi*k dk, since the fft grid is not a disk.  For comparison purposes using parseval's theorem, it's this volume
 *  element that needs to be used, rather than 2 pi k dk.
  
double *power_spectrum_2d(fftw_complex *cdata, double *k_array, int nk, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world)
{

	double pi = M_PI;

	int nx = grid_info.nx;
	int ny = grid_info.ny;
	int local_y_start_after_transpose = grid_info.local_y_start_after_transpose;
	int local_ny_after_transpose = grid_info.local_ny_after_transpose;

	double *P_k_array_proc; //power spectrum P(k)
	double *norm_proc;      //normalization for averaging

	double *P_k_array;      //power spectrum P(k)
	double *norm;           //normalization for averaging

	double *P_k; //power spectrum P(k)

	double kk, kx, ky;

	int ijk, ik, jmin;

	double wk, Pk; //linear interp weighting

	int bin1, bin2;


	char variable_name[200];

	//get a properly zeroed array

	sprintf(variable_name,"P_k_array_proc");
	P_k_array_proc = allocate_double_array(nk,   variable_name, myid, numprocs, world, 0);
	sprintf(variable_name,"norm_proc");
	norm_proc      = allocate_double_array(nk,   variable_name, myid, numprocs, world, 0);

	sprintf(variable_name,"P_k_array");
	P_k_array      = allocate_double_array(nk,   variable_name, myid, numprocs, world, 0);
	sprintf(variable_name,"norm");
	norm           = allocate_double_array(nk,   variable_name, myid, numprocs, world, 0);

	sprintf(variable_name,"P_k");
	P_k            = allocate_double_array(2*nk, variable_name, myid, numprocs, world, 0);


	for(int j=0;j<local_ny_after_transpose;j++)
		for(int i=0;i<nx;i++)
		{
			//positive ky

			if(i>nx/2)
			{
				kx = ((double) (i-nx));
			}else{
				kx = ((double) i);
			}
			ky = ((double) (local_y_start_after_transpose+j) );

			// the magnitude of the k-vector

			kk = sqrt( kx*kx + ky*ky );


			//ijk = i*(ny/2+1) + j;
			//ijk = i*(local_ny_after_transpose) + j;
			ijk = grid_transpose_ijk(i,j,0,grid_info);


			// Pk = |delta_k|^2

			Pk = cdata[ijk].re*cdata[ijk].re + cdata[ijk].im*cdata[ijk].im;


			// add the contribution to the binned power spectrum

			if((kk>=k_array[0])&&(kk<=k_array[nk-1]))
			{
				ik = gsl_interp_bsearch(k_array,kk,0,nk);

				wk = (kk - k_array[ik])/(k_array[ik+1]-k_array[ik]);

				if(ik>=0 && ik<nk)
				{	

					P_k_array_proc[ik]   += ( 1.0-wk )*Pk;
					norm_proc[ik]        += ( 1.0-wk );
				}
				if((ik+1)>=0 && (ik+1)<nk)
				{	
					P_k_array_proc[ik+1] += (     wk )*Pk;
					norm_proc[ik+1]      += (     wk );
				}
			}

		}

	if(myid==0)
	{
		jmin = 1;//avoid DC component
	}else{
		jmin = 0;
	}
	for(int j=jmin;j<local_ny_after_transpose;j++)
		for(int i=0;i<nx;i++)
		{
			//negative ky
			if(i>nx/2)
			{
				kx = ((double) (i-nx));
			}else{
				kx = ((double) i);
			}
			ky = ((double) (local_y_start_after_transpose+j) );
			ky *= -1;



			// the magnitude of the k-vector

			kk = sqrt( kx*kx + ky*ky );


			//ijk = i*(ny/2+1) + j;
			ijk = grid_transpose_ijk(i,j,0,grid_info);


			// Pk = |delta_k|^2

			Pk = cdata[ijk].re*cdata[ijk].re + cdata[ijk].im*cdata[ijk].im;


			// add the contribution to the binned power spectrum

			if((kk>=k_array[0])&&(kk<=k_array[nk-1]))
			{
				ik = gsl_interp_bsearch(k_array,kk,0,nk);

				wk = (kk - k_array[ik])/(k_array[ik+1]-k_array[ik]);

				if(ik>=0 && ik<nk)
				{	

					P_k_array_proc[ik]   += ( 1.0-wk )*Pk;
					norm_proc[ik]        += ( 1.0-wk );
				}
				if((ik+1)>=0 && (ik+1)<nk)
				{	
					P_k_array_proc[ik+1] += (     wk )*Pk;
					norm_proc[ik+1]      += (     wk );
				}
			}

		}

	//Sum P(k) contributions from each processor

	MPI_Allreduce(P_k_array_proc,P_k_array,nk,MPI_DOUBLE,MPI_SUM,world);
	MPI_Allreduce(norm_proc,     norm,     nk,MPI_DOUBLE,MPI_SUM,world);

	//normalize

	for(ik = 0;ik<nk;ik++)
		if(norm[ik]>0)
			P_k_array[ik] /= norm[ik];


	for(int i=0;i<nk;i++)
	{	
		//P_k[i] = P_k_array[i];
		P_k[i] = P_k_array[i] * grid_info.dVk;
		P_k[nk+i] = norm[i];
	}

	free(norm);
	free(P_k_array);
	free(norm_proc);
	free(P_k_array_proc);

	return P_k;
}


/*! \fn double *power_spectrum_field(fftw_complex **cdata, double *k_array, int nk, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world)
 *  \brief This function takes a forward transform (e.g. a complex grid), already normalized by 1./(nx)^ndim and returns a n-dimensional power spectrum vs. k
  
double *power_spectrum_field(fftw_complex **cuk, double *k_array, int nk, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world)
{
	if(grid_info.ndim==2)
	{
		//2-d
		power_spectrum_field_2d(cuk[0], cuk[1], k_array, nk, grid_info, myid, numprocs, world);
	}else{
		//3-d
		power_spectrum_field_3d(cuk[0], cuk[1], cuk[2], k_array, nk, grid_info, myid, numprocs, world);
	}
}

/*! \fn double *power_spectrum_field_3d(fftw_complex *cukx, fftw_complex *cuky, fftw_complex *cukz, double *k_array, int nk, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world)
 *  \brief This function takes a forward transform (e.g. a complex grid), already normalized by 1./(nx*ny*nz) and returns a 3-dimensional power spectrum vs. k
  
double *power_spectrum_field_3d(fftw_complex *cukx, fftw_complex *cuky, fftw_complex *cukz, double *k_array, int nk, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world)
{

	double pi = M_PI;

	int nx = grid_info.nx;
	int ny = grid_info.ny;
	int nz = grid_info.nz;
	int local_y_start_after_transpose = grid_info.local_y_start_after_transpose;
	int local_ny_after_transpose = grid_info.local_ny_after_transpose;

	double *P_k_array_proc; //power spectrum P(k)
	double *norm_proc;      //normalization for averaging

	double *P_k_array;      //power spectrum P(k)
	double *norm;           //normalization for averaging

	double *P_k; //power spectrum P(k)

	double kk, kx, ky, kz;

	int ijk, ik;

	double wk, Pk; //linear interp weighting

	int bin1, bin2;


	char variable_name[200];


	if(grid_info.ndim==2)
	{
		if(myid==0)
		{
			printf("power_spectrum_field_3d not implemented in 2-d.\n");
			fflush(stdout);
		}
		MPI_Abort(world,-1);
		exit(-1);
	}

	//get a properly zeroed array


	sprintf(variable_name,"P_k_array_proc");
	P_k_array_proc = allocate_double_array(nk,   variable_name, myid, numprocs, world, 0);
	sprintf(variable_name,"norm_proc");
	norm_proc      = allocate_double_array(nk,   variable_name, myid, numprocs, world, 0);

	sprintf(variable_name,"P_k_array");
	P_k_array      = allocate_double_array(nk,   variable_name, myid, numprocs, world, 0);
	sprintf(variable_name,"norm");
	norm           = allocate_double_array(nk,   variable_name, myid, numprocs, world, 0);

	sprintf(variable_name,"P_k");
	P_k            = allocate_double_array(2*nk, variable_name, myid, numprocs, world, 0);



	for(int j=0;j<local_ny_after_transpose;j++)
		for(int i=0;i<nx;i++)
		{
			for(int k=0;k<(nz/2+1);k++)
			{

				if(i>nx/2)
				{
					kx = ((double) (i-nx));
				}else{
					kx = ((double) i);
				}
				if(local_y_start_after_transpose+j>ny/2)
				{
					ky = ((double) (local_y_start_after_transpose+j-ny) );
				}else{
					ky = ((double) (local_y_start_after_transpose+j) );
				}
				kz = ((double) k);



				// the magnitude of the k-vector

				kk = sqrt( kx*kx + ky*ky + kz*kz );


				//ijk = (j*nx + i)*(nz/2+1) + k;
				ijk = grid_transpose_ijk(i,j,k,grid_info);


				// Pk = |delta_k|^2

				Pk   = cukx[ijk].re*cukx[ijk].re + cukx[ijk].im*cukx[ijk].im;
				Pk  += cuky[ijk].re*cuky[ijk].re + cuky[ijk].im*cuky[ijk].im;
				Pk  += cukz[ijk].re*cukz[ijk].re + cukz[ijk].im*cukz[ijk].im;


				// add the contribution to the binned power spectrum

				if((kk>=k_array[0])&&(kk<=k_array[nk-1]))
				{
					ik = gsl_interp_bsearch(k_array,kk,0,nk);

					wk = (kk - k_array[ik])/(k_array[ik+1]-k_array[ik]);

					if(ik>=0 && ik<nk)
					{	

						P_k_array_proc[ik]   += ( 1.0-wk )*Pk;
						norm_proc[ik]        += ( 1.0-wk );
					}
					if((ik+1)>=0 && (ik+1)<nk)
					{	
						P_k_array_proc[ik+1] += (     wk )*Pk;
						norm_proc[ik+1]      += (     wk );
					}
				}

			}

			//negative kz
			for(int k=1;k<(nz/2);k++)
			{

				if(i>nx/2)
				{
					kx = ((double) (i-nx));
				}else{
					kx = ((double) i);
				}
				if(local_y_start_after_transpose+j>ny/2)
				{
					ky = ((double) (local_y_start_after_transpose+j-ny) );
				}else{
					ky = ((double) (local_y_start_after_transpose+j) );
				}
				kz = -((double) k);



				// the magnitude of the k-vector

				kk = sqrt( kx*kx + ky*ky + kz*kz );


				//ijk = (j*nx + i)*(nz/2+1) + k;
				ijk = grid_transpose_ijk(i,j,k,grid_info);


				// Pk = |delta_k|^2

				Pk   = cukx[ijk].re*cukx[ijk].re + cukx[ijk].im*cukx[ijk].im;
				Pk  += cuky[ijk].re*cuky[ijk].re + cuky[ijk].im*cuky[ijk].im;
				Pk  += cukz[ijk].re*cukz[ijk].re + cukz[ijk].im*cukz[ijk].im;


				// add the contribution to the binned power spectrum

				if((kk>=k_array[0])&&(kk<=k_array[nk-1]))
				{
					ik = gsl_interp_bsearch(k_array,kk,0,nk);

					wk = (kk - k_array[ik])/(k_array[ik+1]-k_array[ik]);

					if(ik>=0 && ik<nk)
					{	

						P_k_array_proc[ik]   += ( 1.0-wk )*Pk;
						norm_proc[ik]        += ( 1.0-wk );
					}
					if((ik+1)>=0 && (ik+1)<nk)
					{	
						P_k_array_proc[ik+1] += (     wk )*Pk;
						norm_proc[ik+1]      += (     wk );
					}
				}

			}
		}

	//Sum P(k) contributions from each processor

	MPI_Allreduce(P_k_array_proc,P_k_array,nk,MPI_DOUBLE,MPI_SUM,world);
	MPI_Allreduce(norm_proc,     norm,     nk,MPI_DOUBLE,MPI_SUM,world);

	//normalize

	for(ik = 0;ik<nk;ik++)
		if(norm[ik]>0)
			P_k_array[ik] /= norm[ik];


	for(int i=0;i<nk;i++)
	{	
		//P_k[i] = P_k_array[i];
		P_k[i] = P_k_array[i] * grid_info.dVk;
		P_k[nk+i] = norm[i];
		printf("i %d P_k[%d] %e norm[%d] %e dVk %e\n",i,i,P_k[i],i,P_k[nk+i],grid_info.dVk); //
	}

	free(norm);
	free(P_k_array);
	free(norm_proc);
	free(P_k_array_proc);

	return P_k;

}

/*! \fn double *power_spectrum_field_2d(fftw_complex *cukx, fftw_complex *cuky, fftw_complex *cukz, double *k_array, int nk, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world)
 *  \brief This function takes a forward transform (e.g. a complex grid), already normalized by 1./(nx*ny) and returns a 1-dimensional power spectrum vs. k
  
double *power_spectrum_field_2d(fftw_complex *cukx, fftw_complex *cuky, double *k_array, int nk, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world)
{

	double pi = M_PI;

	int nx = grid_info.nx;
	int ny = grid_info.ny;
	int local_y_start_after_transpose = grid_info.local_y_start_after_transpose;
	int local_ny_after_transpose = grid_info.local_ny_after_transpose;

	double *P_k_array_proc; //power spectrum P(k)
	double *norm_proc;      //normalization for averaging

	double *P_k_array;      //power spectrum P(k)
	double *norm;           //normalization for averaging

	double *P_k; //power spectrum P(k)

	double kk, kx, ky;

	int ijk, ik, jmin;

	double wk, Pk; //linear interp weighting

	int bin1, bin2;


	char variable_name[200];


	//get a properly zeroed array


	sprintf(variable_name,"P_k_array_proc");
	P_k_array_proc = allocate_double_array(nk,   variable_name, myid, numprocs, world, 0);
	sprintf(variable_name,"norm_proc");
	norm_proc      = allocate_double_array(nk,   variable_name, myid, numprocs, world, 0);

	sprintf(variable_name,"P_k_array");
	P_k_array      = allocate_double_array(nk,   variable_name, myid, numprocs, world, 0);
	sprintf(variable_name,"norm");
	norm           = allocate_double_array(nk,   variable_name, myid, numprocs, world, 0);

	sprintf(variable_name,"P_k");
	P_k            = allocate_double_array(2*nk, variable_name, myid, numprocs, world, 0);


	for(int j=0;j<local_ny_after_transpose;j++)
		for(int i=0;i<nx;i++)
		{
			//positive ky
			if(i>nx/2)
			{
				kx = ((double) (i-nx));
			}else{
				kx = ((double) i);
			}
			ky = ((double) (local_y_start_after_transpose+j) );


			// the magnitude of the k-vector

			kk = sqrt( kx*kx + ky*ky );


			//ijk = i*(ny/2+1) + j;
			ijk = grid_transpose_ijk(i,j,0,grid_info);


			// Pk = |delta_k|^2

			Pk   = cukx[ijk].re*cukx[ijk].re + cukx[ijk].im*cukx[ijk].im;
			Pk  += cuky[ijk].re*cuky[ijk].re + cuky[ijk].im*cuky[ijk].im;


			// add the contribution to the binned power spectrum

			if((kk>=k_array[0])&&(kk<=k_array[nk-1]))
			{
				ik = gsl_interp_bsearch(k_array,kk,0,nk);

				wk = (kk - k_array[ik])/(k_array[ik+1]-k_array[ik]);

				if(ik>=0 && ik<nk)
				{	

					P_k_array_proc[ik]   += ( 1.0-wk )*Pk;
					norm_proc[ik]        += ( 1.0-wk );
				}
				if((ik+1)>=0 && (ik+1)<nk)
				{	
					P_k_array_proc[ik+1] += (     wk )*Pk;
					norm_proc[ik+1]      += (     wk );
				}
			}

		}

	if(myid==0)
	{
		jmin = 1; //avoid DC component
	}else{
		jmin = 0;
	}
	for(int j=jmin;j<local_ny_after_transpose;j++)
		for(int i=0;i<nx;i++)
		{
			//negative ky

			if(i>nx/2)
			{
				kx = ((double) (i-nx));
			}else{
				kx = ((double) i);
			}
			ky = ((double) (local_y_start_after_transpose+j) );
			ky *= -1;



			// the magnitude of the k-vector

			kk = sqrt( kx*kx + ky*ky );


			//ijk = i*(ny/2+1) + j;
			ijk = grid_transpose_ijk(i,j,0,grid_info);


			// Pk = |delta_k|^2

			Pk   = cukx[ijk].re*cukx[ijk].re + cukx[ijk].im*cukx[ijk].im;
			Pk  += cuky[ijk].re*cuky[ijk].re + cuky[ijk].im*cuky[ijk].im;


			// add the contribution to the binned power spectrum

			if((kk>=k_array[0])&&(kk<=k_array[nk-1]))
			{
				ik = gsl_interp_bsearch(k_array,kk,0,nk);

				wk = (kk - k_array[ik])/(k_array[ik+1]-k_array[ik]);

				if(ik>=0 && ik<nk)
				{	

					P_k_array_proc[ik]   += ( 1.0-wk )*Pk;
					norm_proc[ik]        += ( 1.0-wk );
				}
				if((ik+1)>=0 && (ik+1)<nk)
				{	
					P_k_array_proc[ik+1] += (     wk )*Pk;
					norm_proc[ik+1]      += (     wk );
				}
			}

		}

	//Sum P(k) contributions from each processor

	MPI_Allreduce(P_k_array_proc,P_k_array,nk,MPI_DOUBLE,MPI_SUM,world);
	MPI_Allreduce(norm_proc,     norm,     nk,MPI_DOUBLE,MPI_SUM,world);

	//normalize

	for(ik = 0;ik<nk;ik++)
		if(norm[ik]>0)
			P_k_array[ik] /= norm[ik];


	for(int i=0;i<nk;i++)
	{	
		//P_k[i] = P_k_array[i];
		P_k[i] = P_k_array[i] * grid_info.dVk;
		P_k[nk+i] = norm[i];
	}

	free(norm);
	free(P_k_array);
	free(norm_proc);
	free(P_k_array_proc);

	return P_k;

}
/*! \fn int find_power_spectrum_nk(FFTW_Grid_Info grid_info)
 *  \brief Routine to find the extent of 1-d power spectrum arrays.
 
int find_power_spectrum_nk(FFTW_Grid_Info grid_info)
{
	double nkd = sqrt( 0.25*grid_info.nx*grid_info.nx + 0.25*grid_info.ny*grid_info.ny + 0.25*grid_info.nz*grid_info.nz );
	int nk = ((int) nkd) + 2;
	return nk;
}

/*! \fn void initialize_power_spectrum_k_array(double *&k_array, int *nk, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world)
 *  \brief Routine to initialize the k_array for power spectra 
void initialize_power_spectrum_k_array(double *&k_array, int *nk, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world)
{
	char variable_name[200];

	*nk = find_power_spectrum_nk(grid_info);

	sprintf(variable_name,"k_array");
	k_array = allocate_double_array( *nk,   variable_name, myid, numprocs, world, 0);

	for(int i=0;i<(*nk);i++)
	{
		k_array[i] = ((double) i);

	}
}


/*! \fn int find_structure_function_nl(FFTW_Grid_Info grid_info)
 *  \brief Routine to find the extent of 1-d structure function arrays.
 
int find_structure_function_nl(FFTW_Grid_Info grid_info)
{
	//double nld = sqrt( 0.25*grid_info.nx*grid_info.nx + 0.25*grid_info.ny*grid_info.ny + 0.25*grid_info.nz*grid_info.nz );
	int nl = 100;
	//int nl = 50;
	//int nl = nx-1;
	return nl;
}
/*! \fn void initialize_structure_function_l_array(double *&l_array, int *nl, double *l_min, double *l_max, double *x, double *y, double *z, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world)
 *  \brief Routine to initialize the l_array for structure functions
void initialize_structure_function_l_array(double *&l_array, int *nl, double *l_min, double *l_max, double *x, double *y, double *z, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world)
{	
	int nx = grid_info.nx;
	int ny = grid_info.ny;
	int nz = grid_info.nz;

	int il;
	int i,j,k;
	double dl;

	char variable_name[200];

	//find length of the l array
	*nl = find_structure_function_nl(grid_info);

	//allocate l array
	sprintf(variable_name,"l_array");
	l_array = allocate_double_array( *nl,   variable_name, myid, numprocs, world, 0);

	//the minimum displacement is zero
	i = 0;
	j = 0;
	k = 0;



	*l_min = 0.5*fabs(x[1]-x[0]);
	*l_max = 1.01*sqrt(3*grid_info.BoxSize*grid_info.BoxSize/4.);
	//l_min = fabs(x[1]-x[0]);
	//l_max = fabs(x[nx-1]-x[0]);

	//linear mapping
	dl = (*l_max - *l_min)/((double) (*nl)); 
	//dl = (log10(*l_max) - log10(*l_min))/((double) (*nl)); 
	for(il=0;il<(*nl);il++)
		l_array[il]    = dl*((double) il) + *l_min;
		//l_array[il]    = pow(10.0, dl*((double) il) + log10(*l_min));
}




/*! \fn double **generate_forcing_field(double (*Pk_power_spectrum)(double,void *), void *params, int iseed, int ndim, FFTW_Grid_Info grid_info, rfftwnd_mpi_plan plan, rfftwnd_mpi_plan iplan, int myid, int numprocs, MPI_Comm world)
 *  \brief  Given a power spectrum Pk_power_spectrum that depends on wavenumber and some parameters *params, and a random number generator 
            seed iseed, return a ndim-dimensional gaussian random field with power spectrum Pk_power_spectrum.
 *
 *	    Note that the field power spectrum is normalized such that the FFT of the *entire field* has the same normalization
 *          as the n-d supplied Pk_power_spectrum.
double **generate_forcing_field(double (*Pk_power_spectrum)(double,void *), void *params, int iseed, FFTW_Grid_Info grid_info, rfftwnd_mpi_plan plan, rfftwnd_mpi_plan iplan, int myid, int numprocs, MPI_Comm world)
{
	int nx = grid_info.nx;
	int ny = grid_info.ny;
	int nz = grid_info.nz;

	int ndim = grid_info.ndim;
	int iseed_iter;

	//mean for initial white noise gaussian field
	double mu = 0;

	if(grid_info.ndim==2)
		nz=1;	
	//variance for initial white noise gaussian field
	double sigma_squared = ((double) nx)*((double) ny)*((double) nz);


	//ndim-dimensional field to return
	double **field;

	//initial white noise gaussian field and
	//final gaussian field with power spectrum P(k)
	double *zeta;

	//fftw work space
	double *work;

	//transform of white noise gaussian; white noise gaussian x transfer function
	fftw_complex *zeta_k;

	//variable name
	char variable_name[200];


	//allocate field
	sprintf(variable_name,"field");
	field = allocate_field_fftw_grid(ndim, grid_info.total_local_size, variable_name, myid, numprocs, world, 0);


	//allocate fft workspace 
	sprintf(variable_name,"work");
	work   = allocate_real_fftw_grid(grid_info.total_local_size, variable_name, myid, numprocs, world,0);



	for(int i=0;i<ndim;i++)
	{
		//use a different iseed for each dimension, so they are different
		iseed_iter = iseed + i*iseed;
		
		//generate a white noise gaussian field
		zeta = grid_normal_white_noise(mu, sigma_squared, iseed_iter, grid_info, plan, iplan, myid, numprocs, world);


		//forward transform the white noise gaussian field to k-space
		forward_transform_fftw_grid(zeta, work, plan, zeta_k, grid_info, myid, numprocs, world);


		//the data is now complex, so typecast a pointer

		zeta_k = (fftw_complex *) zeta;	


		//apply the transfer function to zeta_k

		if(grid_info.ndim==2)
		{
			grid_transform_apply_transfer_function_2d(Pk_power_spectrum,params,zeta_k,grid_info,plan,iplan,myid,numprocs,world);
		}else{
			grid_transform_apply_transfer_function(Pk_power_spectrum,params,zeta_k,grid_info,plan,iplan,myid,numprocs,world);
		}


		//perform the inverse transform
		inverse_transform_fftw_grid(zeta, work, iplan, zeta_k, grid_info, myid, numprocs, world);

		//copy grid into field

		grid_copy_in_place(zeta, field[i], grid_info, plan, iplan, myid, numprocs, world);

		//normalize for n-dimensionality, so n-d P(k) has proper normalization
		grid_rescale(1./sqrt(grid_info.ndim), field[i], grid_info, plan, iplan, myid, numprocs, world);

		//free zeta
		free(zeta);

	}


	//free work
	free(work);


	//return the answer
	return field;
}



*/
