/*! \file grid_operations.h
 *  \brief Function declarations for mathematical operations
 *	   on fft grids. */
#include<mpi.h>
#ifndef NO_FFTW
#include<fftw3-mpi.h>
#include"grid_fft.h"
#endif
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_interp.h>
 
//#include"routines.h"
#ifndef  GRID_OPERATIONS
#define  GRID_OPERATIONS
//random number generator
extern gsl_rng *r;
extern const gsl_rng_type *T;

/*! \fn void grid_set_value(double value, double *A, FFTW_Grid_Info grid_info);
 *  \brief Set the grid to a single value. */
void grid_set_value(double value, double *A, FFTW_Grid_Info grid_info);

/*! \fn void grid_field_set_value(double value, double **A, FFTW_Grid_Info grid_info);
 *  \brief Set the field to a single value. */
void grid_field_set_value(double value, double **A, FFTW_Grid_Info grid_info);

/*! \fn double *grid_copy(double *A, FFTW_Grid_Info grid_info)
 *  \brief Produces a copy of a double grid. */
double *grid_copy(double *A, FFTW_Grid_Info grid_info);

/*! \fn double **grid_field_copy(double **source, FFTW_Grid_Info grid_info)
 *  \brief Produces a copy of a double field in place. */
double **grid_field_copy(double **source, FFTW_Grid_Info grid_info);

/*! \fn double *grid_copy_in_place(double *source, double *copy, FFTW_Grid_Info grid_info)
 *  \brief Produces a copy of a double grid in place. */
void grid_copy_in_place(double *source, double *copy, FFTW_Grid_Info grid_info);

/*! \fn void grid_field_copy_in_place(double **source, double **copy, FFTW_Grid_Info grid_info)
 *  \brief Produces a copy of a double field in place. */
void grid_field_copy_in_place(double **source, double **copy, FFTW_Grid_Info grid_info);

/*! \fn double *grid_scaled_copy_in_place(double *source, double scale, double *copy, FFTW_Grid_Info grid_info)
 *  \brief Produces a scaled copy of a double grid in place. */
void grid_scaled_copy_in_place(double *source, double scale, double *copy, FFTW_Grid_Info grid_info);

/*! \fn double *grid_copy_real_to_complex_in_place(double *source, fftw_complex *copy, FFTW_Grid_Info grid_info)
 *  \brief Produces a copy of a double grid in place into the real elements of a complex grid. */
void grid_copy_real_to_complex_in_place(double *source, fftw_complex *copy, FFTW_Grid_Info grid_info);

/*! \fn double *grid_copy_complex_to_real_in_place(fftw_complex *source, double *copy, FFTW_Grid_Info grid_info)
 *  \brief Produces a copy of the real components of a complex grid in place into a real grid. */
void grid_copy_complex_to_real_in_place(fftw_complex *source, double *copy, FFTW_Grid_Info grid_info);

/*! \fn void grid_set_function(double (*setting_function)(double, double, double, void *), void *params, double *A, FFTW_Grid_Info grid_info);
 *  \brief Set the grid to the value of the supplied function at the cell centers. */
void grid_set_function(double (*setting_function)(double,double,double,void *), void *params, double *A, FFTW_Grid_Info grid_info);

/*! \fn void grid_field_set_function(double (*setting_function)(double, double, double, int, void *), void *params, double **A, FFTW_Grid_Info grid_info);
 *  \brief Set the grid field to the value of the supplied function at the cell centers. */
void grid_field_set_function(double (*setting_function)(double,double,double,int,void *), void *params, double **A, FFTW_Grid_Info grid_info);

/*! \fn void grid_tensor_set_function(double (*setting_function)(double, double, double, int, int, void *), void *params, double ***A, FFTW_Grid_Info grid_info);
 *  \brief Set the grid tensor to the value of the supplied function at the cell centers. */
void grid_tensor_set_function(double (*setting_function)(double,double,double,int,int,void *), void *params, double ***A, FFTW_Grid_Info grid_info);

/*! \fn void grid_rescale(double scale, double *A, FFTW_Grid_Info grid_info)
 *  \brief Rescale the grid by a factor of scale. */
void grid_rescale(double scale, double *A, FFTW_Grid_Info grid_info);

/*! \fn int grid_is_peak(int i, int j, int k, double *u, FFTW_Grid_Info grid_info)
 *  \brief Is this location a peak in u ? */
int grid_is_peak(int ii, int jj, int kk, double *u, FFTW_Grid_Info grid_info);

/*! \fn double *grid_mask_threshold(double threshold, double *u, FFTW_Grid_Info grid_info);
 *  \brief Creates a mask based on a threshold */
double *grid_mask_threshold(double threshold, double *u, FFTW_Grid_Info grid_info);

/*! \fn double *grid_mask_range(double lower, double upper, double *u, FFTW_Grid_Info grid_info);
 *  \brief Creates a mask based on a range*/
double *grid_mask_range(double lower, double upper, double *u, FFTW_Grid_Info grid_info);

/*! \fn double *grid_mask_peak_range(double lower, double upper, double *u, FFTW_Grid_Info grid_info)
 *  \brief Creates a mask based on a range and being a peak*/
double *grid_mask_peak_range(double lower, double upper, double *u, FFTW_Grid_Info grid_info);

/*! \fn int grid_mask_count_cells(double *mask, FFTW_Grid_Info grid_info, MPI_Comm world);
 *  \brief Count unmasked cells*/
int grid_mask_count_cells(double *mask, FFTW_Grid_Info grid_info, MPI_Comm world); 

/*! \fn int grid_mask_count_local_cells(double *mask, FFTW_Grid_Info grid_info, MPI_Comm world);
 *  \brief Count unmasked local cells*/
int grid_mask_count_local_cells(double *mask, FFTW_Grid_Info grid_info, MPI_Comm world);

/*! \fn int *grid_mask_local_cell_indices(double *mask, FFTW_Grid_Info grid_info, MPI_Comm world);
 *  \brief Indices to unmasked local cells */
int *grid_mask_local_cell_indices(double *mask, FFTW_Grid_Info grid_info, MPI_Comm world);

/*! \fn int *grid_mask_direction_indices(int dim, double *mask, FFTW_Grid_Info grid_info, MPI_Comm world);
 *  \brief Return locations of masked cells in dim direction */
int *grid_mask_direction_indices(int dim, double *mask, FFTW_Grid_Info grid_info, MPI_Comm world);

/*! \fn double *grid_mask_apply(double *mask, double *u, FFTW_Grid_Info grid_info)
 *  \brief Copy a grid after applying a mask */
double *grid_mask_apply(double *mask, double *u, FFTW_Grid_Info grid_info);

/*! \fn double **grid_field_mask_apply(double *mask, double **u, FFTW_Grid_Info grid_info)
 *  \brief Copy a grid after applying a mask */
double *grid_field_mask_apply_(double *mask, double **u, FFTW_Grid_Info grid_info);

/*! \fn double grid_surface_area(double *mask, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world)
 *  \brief Find the approximate surface area of a mask */
double grid_surface_area(double *mask, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world);

/*! \fn double grid_surface_area(double *mask, double *dir, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world)
 *  \brief Find the approximate surface area of a mask whose normal has a positive dot product with supplied vector*/
double grid_surface_area_directional(double *mask, double *dir, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world);

/*! \fn double grid_surface_area(double *mask, double **vfield, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world)
 *  \brief Find the approximate surface flux of a vector field through the surface at the mask boundary */
double grid_surface_flux(double *mask, double **vfield, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world);

/*! \fn double **grid_vector_surface_area(double *mask, double **vfield, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world)
 *  \brief Find the approximate vector surface area at the mask boundary, for regions with positive dot product with the vfield */
double **grid_vector_surface_area(double *mask, double **vfield, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world);

/*! \fn double *grid_surface_area_mask(double *mask, double **vfield, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world)
 *  \brief Define a mask at the surface boundary, for regions with positive dot product with the vfield */
double *grid_surface_area_mask(double *mask, double **vfield, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world);


/*! \fn double *grid_power(double alpha, double *A, FFTW_Grid_Info grid_info);
 *  \brief Return a power of the grid A^alpha. */
double *grid_power(double alpha, double *A, FFTW_Grid_Info grid_info);

/*! \fn double grid_min(double *A, FFTW_Grid_Info grid_info, MPI_Comm world)
 *  \brief Calculate the min of the grid. */
double grid_min(double *A, FFTW_Grid_Info grid_info, MPI_Comm world); 

/*! \fn double grid_max(double *A, FFTW_Grid_Info grid_info, MPI_Comm world); 
 *  \brief Calculate the max of the grid. */
double grid_max(double *A, FFTW_Grid_Info grid_info, MPI_Comm world);  

/*! \fn double grid_mean(double *A, FFTW_Grid_Info grid_info, MPI_Comm world); 
 *  \brief Calculate the mean of the grid. */
double grid_mean(double *A, FFTW_Grid_Info grid_info, MPI_Comm world); 

/*! \fn double grid_weighted_mean(double *A, double *w, FFTW_Grid_Info grid_info, MPI_Comm world)
 *  \brief Calculate the weighted mean of the grid. */
double grid_weighted_mean(double *A, double *w, FFTW_Grid_Info grid_info, MPI_Comm world);


/*! \fn double grid_rms(double *A, FFTW_Grid_Info grid_info, MPI_Comm world)
 *  \brief Calculate the RMS value of the grid. */
double grid_rms(double *A, FFTW_Grid_Info grid_info, MPI_Comm world); 

/*! \fn double grid_weighted_rms(double *A, FFTW_Grid_Info grid_info, MPI_Comm world)
 *  \brief Calculate the weighted RMS value of the grid. */
double grid_weighted_rms(double *A, double *w, FFTW_Grid_Info grid_info, MPI_Comm world); 

/*! \fn double grid_field_mean(double **A, FFTW_Grid_Info grid_info, MPI_Comm world) 
 *  \brief Calculate the mean of the field A. */
double grid_field_mean(double **A, FFTW_Grid_Info grid_info, MPI_Comm world);

/*! \fn double grid_field_rms(double **A, FFTW_Grid_Info grid_info, MPI_Comm world) 
 *  \brief Calculate the rms vlaue of the field A. */
double grid_field_rms(double **A, FFTW_Grid_Info grid_info, MPI_Comm world);

/*! \fn double grid_variance(double *mean, double *A, FFTW_Grid_Info grid_info, MPI_Comm world) 
 *  \brief Calculate the variance of the grid.  Also return the mean. */
double grid_variance(double *mean, double *A, FFTW_Grid_Info grid_info, MPI_Comm world);

/*! \fn double grid_field_variance(double *mean, double *A, FFTW_Grid_Info grid_info, MPI_Comm world)
 *  \brief Calculate the variance of the field A.  Also return the mean. */
double grid_field_variance(double *mean, double **A, FFTW_Grid_Info grid_info, MPI_Comm world); 

/*! \fn double grid_variance_estimator(double *mean, double *A, FFTW_Grid_Info grid_info, MPI_Comm world)
 *  \brief Calculate the variance estimator of the grid.  Also return the mean. */
double grid_variance_estimator(double *mean, double *A, FFTW_Grid_Info grid_info, MPI_Comm world);

/*! \fn void grid_enforce_mean_and_variance(double mean, double variance, double *A, FFTW_Grid_Info grid_info, MPI_Comm world)
 *  \brief Force grid A to have mean mean and variance variance. */
void grid_enforce_mean_and_variance(double mean, double variance, double *A, FFTW_Grid_Info grid_info, MPI_Comm world);

/*! \fn void grid_field_enforce_mean_and_variance(double mean, double variance, double **A, FFTW_Grid_Info grid_info, MPI_Comm world)
 *  \brief Force field A to have mean mean and variance variance. */
void grid_field_enforce_mean_and_variance(double mean, double variance, double **A, FFTW_Grid_Info grid_info, MPI_Comm world);

/*! \fn double *grid_sum(double *A, double *B, FFTW_Grid_Info grid_info)
 *  \brief Performs the sum A + B on two fftw grids. */
double *grid_sum(double *A, double *B, FFTW_Grid_Info grid_info);

/*! \fn void grid_sum_in_place(double *A, double *B, FFTW_Grid_Info grid_info)
 *  \brief Performs the sum A + B on two fftw grids in place, result in A. */
void grid_sum_in_place(double *A, double *B, FFTW_Grid_Info grid_info);

/*! \fn double *grid_difference(double *A, double *B, FFTW_Grid_Info grid_info)
 *  \brief Performs the difference A - B on two fftw grids. */
double *grid_difference(double *A, double *B, FFTW_Grid_Info grid_info);

/*! \fn void grid_difference_in_place(double *A, double *B, FFTW_Grid_Info grid_info);
 *  \brief Performs the difference A - B on two fftw grids in place, result in A. */
void grid_difference_in_place(double *A, double *B, FFTW_Grid_Info grid_info);

/*! \fn double *grid_quotient(double *A, double *B, FFTW_Grid_Info grid_info)
 *  \brief Takes the quotient A / B on two fftw grids. */
double *grid_quotient(double *A, double *B, FFTW_Grid_Info grid_info);

/*! \fn void grid_quotient_in_place(double *A, double *B, FFTW_Grid_Info grid_info)
 *  \brief Takes the quotient A / B on two fftw grids in place, result in A. */
void grid_quotient_in_place(double *A, double *B, FFTW_Grid_Info grid_info);

/*! \fn double *grid_product(double *A, double *B, FFTW_Grid_Info grid_info)
 *  \brief Performs the product A * B on two fftw grids. */
double *grid_product(double *A, double *B, FFTW_Grid_Info grid_info);

/*! \fn double **grid_field_product(double *A, double **B, FFTW_Grid_Info grid_info);
 *  \brief Performs the product A * B on between a scalar grid and a vector field. */
double **grid_field_product(double *A, double **B, FFTW_Grid_Info grid_info);

/*! \fn double grid_volume_integral(double *u, FFTW_Grid_Info grid_info, MPI_Comm world) 
 *  \brief Calculate the total volume integral I = \int u dV for a scalar grid.*/
double grid_volume_integral(double *u, FFTW_Grid_Info grid_info, MPI_Comm world);

/*! \fn double *grid_field_dot_product(double **A, double **B, FFTW_Grid_Info grid_info)
 *  \brief Calculate the vector dot product A . B at each grid cell -- not matrix dot product.*/
double *grid_field_dot_product(double **A, double **B, FFTW_Grid_Info grid_info);

/*! \fn double **grid_field_cross_product(double **A, double **B, FFTW_Grid_Info grid_info)
 *  \brief Calculate the cross product A x B.*/
double **grid_field_cross_product(double **A, double **B, FFTW_Grid_Info grid_info);

/*! \fn double **grid_field_norm(double **A, FFTW_Grid_Info grid_info)
 *  \brief Normalize the field by its local magnitude.*/
double **grid_field_norm(double **A, FFTW_Grid_Info grid_info);

/*! \fn double *grid_field_magnitude(double **A, FFTW_Grid_Info grid_info)
 *  \brief Return the local field magnitude.*/
double *grid_field_magnitude(double **A, FFTW_Grid_Info grid_info);

//derivatives, curls, and gradients


/*! \fn double *grid_derivative(double *u, int direction, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world)
 *  \brief Calculates the derivative of periodic u along the direction "direction". */
double *grid_derivative(double *u, int direction, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world);

/*! \fn double *grid_derivative_real_space(double *u, int direction, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world)
 *  \brief Calculates the derivative of periodic u along the direction "direction" in real space. */
double *grid_derivative_real_space(double *u, int direction, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world);

/*! \fn double *grid_derivative_fourth_order(double *u, int direction, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world)
 *  \brief Calculates the derivative of periodic u along the direction "direction" using fourth order finite difference. */
double *grid_derivative_fourth_order(double *u, int direction, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world);

/*! \fn double *grid_derivative_second_order(double *u, int direction, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world)
 *  \brief Calculates the derivative of periodic u along the direction "direction" using second order finite difference. */
double *grid_derivative_second_order(double *u, int direction, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world);

/*! \fn double *grid_derivative_k_space(double *u, int direction, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world)
 *  \brief Calculates the derivative of periodic u along the direction "direction" in k space. */
double *grid_derivative_k_space(double *u, int direction, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world);

/*! \fn double **grid_field_curl(double **u, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world)
 *  \brief Calculate the curl = nabla cross u. */
double **grid_field_curl(double **u, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world);

/*! \fn double **grid_field_curl_real_space(double **u, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world)
 *  \brief Calculate the curl = nabla cross u in real space.*/
double **grid_field_curl_real_space(double **u, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world);

/*! \fn double **grid_field_curl_k_space(double **u, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world)
 *  \brief Calculate the curl in k-space. */
double **grid_field_curl_k_space(double **u, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world);

/*! \fn double *grid_field_divergence(double **u, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world)
 *  \brief Calculate the div = nabla dot u.*/
double *grid_field_divergence(double **u, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world);

/*! \fn double *grid_field_divergence_real_space(double **u, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world)
 *  \brief Calculate the div = nabla dot u in real space. */
double *grid_field_divergence_real_space(double **u, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world);

/*! \fn double *grid_field_divergence_k_space(double **u, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world)
 *  \brief Calculate the div = nabla dot u in k space. */
double *grid_field_divergence_k_space(double **u, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world);

/*! \fn double **grid_gradient(double *u, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world)
 *  \brief Calculates the gradient nabla u = [dudx,dudy,dudz] */
double **grid_gradient(double *u, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world);

/*! \fn double **grid_gradient_real_space(double *u, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world)
 *  \brief Calculates the gradient nabla u = [dudx,dudy,dudz] in real space */
double **grid_gradient_real_space(double *u, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world);

/*! \fn double **grid_gradient_k_space(double *u, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world)
 *  \brief Calculates the gradient nabla u = [dudx,dudy,dudz] in k space */
double **grid_gradient_k_space(double *u, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world);

//vorticity, helicity, and dilatation

/*! \fn double **grid_field_vorticity(double **u, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world)
 *  \brief Calculate the vorticity = nabla cross u. */
double **grid_field_vorticity(double **u, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world);

/*! \fn double *grid_field_helicity(double **u, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world)
 *  \brief Calculate the field helicity h = u . nabla cross u, or h = u dot vorticity.  */
double *grid_field_helicity(double **u, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world);

/*! \fn double *grid_field_dilatation(double **u, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world)
 *  \brief Calculate the dilatation of the field u. */
double *grid_field_dilatation(double **u, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world);

/*! \fn double **grid_field_dilatational_component(double **u, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world)
 *  \brief Calculate the dilatational component of field u via Helmholtz decomposition in Fourier space. */
double **grid_field_dilatational_component(double **u, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world);

/*! \fn double **grid_field_solenoidal_component(double **u, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world)
 *  \brief Calculate the solenoidal component of field u via Helmholtz decomposition in Fourier space. */
double **grid_field_solenoidal_component(double **u, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world);

//grid energies

/*! \fn double grid_total_specific_energy(double *u, FFTW_Grid_Info grid_info, MPI_Comm world)
 *  \brief Calculate the total specific energy E = 1/2 \int u^2 dV on a grid u */
double grid_total_specific_energy(double *u, FFTW_Grid_Info grid_info, MPI_Comm world);

/*! \fn double *grid_field_specific_energy(double **u, FFTW_Grid_Info grid_info)
 *  \brief Calculate the specific energy E = 1/2 |u|^2.*/
double *grid_field_specific_energy(double **u, FFTW_Grid_Info grid_info);

/*! \fn double grid_field_total_specific_energy(double **u, FFTW_Grid_Info grid_info, MPI_Comm world)
 *  \brief Calculate the total specific energy E = 1/2 \int u^2 dV on a field u */
double grid_field_total_specific_energy(double **u, FFTW_Grid_Info grid_info, MPI_Comm world);

/*! \fn double *grid_field_specific_dilatational_energy(double **u, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world)
 *  \brief Calculate the specific energy E = 1/2 |u_D|^2 in the dilatational component of the velocity field. */
double *grid_field_specific_dilatational_energy(double **u, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world);

/*! \fn double grid_field_total_specific_dilatational_energy(double **u, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world)
 *  \brief Calculate the total specific energy E = 1/2 \int u_D^2 dV in the dilatational component of the velocity field u. */
double grid_field_total_specific_dilatational_energy(double **u, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world);

/*! \fn double *grid_field_specific_solenoidal_energy(double **u, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world)
 *  \brief Calculate the specific energy E = 1/2 |u_S|^2 in the solenoidal component of the velocity field. */
double *grid_field_specific_solenoidal_energy(double **u, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world);

/*! \fn double grid_field_total_specific_solenoidal_energy(double **u, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world)
 *  \brief Calculate the total specific energy E = 1/2 \int u_D^2 dV in the solenoidal component of the velocity field u. */
double grid_field_total_specific_solenoidal_energy(double **u, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world);

//dissipation rates

/*! \fn double dissipation_rate_dilatational(double **u, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world)
 *  \brief Calculate the dilatational contribution to the dissipation rate. */
double dissipation_rate_dilatational(double **u, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world);

/*! \fn double dissipation_rate_solenoidal(double **u, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world)
 *  \brief Calculate the solenoidal contribution to the dissipation rate. */
double dissipation_rate_solenoidal(double **u, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world);

/*! \fn double small_scale_compressive_ratio(double **u, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world)
 *  \brief Calculate the small-scale compressive ratio <|div u|^2> / ( <|div u|^2> + <|curl u|^2> ).  */
double small_scale_compressive_ratio(double **u, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world, double *epsilon_S, double *epsilon_D);

//enstrophy and denstrophy

/*! \fn double *grid_field_enstrophy(double **u, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world)
 *  \brief Calculate the field enstrophy = |nabla cross u|^2 = |vorticity|^2. */
double *grid_field_enstrophy(double **u, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world);

/*! \fn double grid_field_total_enstrophy(double **u, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world)
 *  \brief Calculate the field's total enstrophy = 1/2 \int |nabla cross u|^2 dV = 1/2 \int |vorticity|^2 dV. */
double grid_field_total_enstrophy(double **u, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world);

/*! \fn double *grid_field_denstrophy(double *rho, double **u, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world)
 *  \brief Calculate the field denstrophy = 1/2 |nabla cross {sqrt(rho) u}|^2/rho. */
double *grid_field_denstrophy(double *rho, double **u, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world);

//tensor operations

/*! \fn double ***grid_transform_tensor(double ***sigma, double ***a, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world)
 *  \brief Transform a tensor using another tensor. */
double ***grid_transform_tensor(double ***sigma_in, double ***a_in, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world);

/*! \fn double ***grid_velocity_gradient_tensor(double **u, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world)
 *  \brief Calculate the components of the velocity gradient tensor. */
double ***grid_velocity_gradient_tensor(double **u, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world);


/*! \fn double **grid_convective_derivative(double **u, double **velocity, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world)
 *  \brief Calculate the convective derivative of a vector field. */
double **grid_convective_derivative(double **u, double **velocity, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world);

double **grid_field_tensor_product(double ***u, double **v, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world);

/*! \fn double ***grid_strain_tensor(double **u, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world)
 *  \brief Calculate the components of the rate of strain tensor. */
double ***grid_strain_tensor(double **u, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world);

/*! \fn double ***grid_shear_tensor(double **u, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world)
 *  \brief Calculate the components of the shear tensor. */
double ***grid_shear_tensor(double **u, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world);

/*! \fn double ***grid_rotation_tensor(double **u, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world)
 *  \brief Calculate the components of the rotation tensor. */
double ***grid_rotation_tensor(double **u, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world);

double *grid_tensor_determinant(double ***A, FFTW_Grid_Info grid_info);


//grid interpolation

/*! \fn double *grid_first_slice(double *f, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world);
 *  \brief First y-z slice of slab from right neighbor */
double *grid_first_slice(double *f, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world);

/*! \fn double *grid_cubic_lower_slice(double *f, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world);
 *  \brief Lower y-z slice of slab from left neighbor */
double *grid_cubic_lower_slice(double *f, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world);

/*! \fn double *grid_cubic_upper_slice(double *f, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world);
 *  \brief Upper y-z slice of slab from left neighbor */
double *grid_cubic_upper_slice(double *f, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world);

/*! \fn double grid_interpolation(double x, double y, double z, double *y, double *fu, FFTW_Grid_Info grid_info);
 *  \brief Interpolation on the grid */
double grid_interpolation(double x, double y, double z, double *f, double *fu, FFTW_Grid_Info grid_info);

/*! \fn double grid_cubic_log_interpolation(double x, double y, double z, double *f, double *fl, double *fu, FFTW_Grid_Info grid_info)
 *  \brief Tricubic log interpolation on the grid */
double grid_cubic_log_interpolation(double x, double y, double z, double *f, double *fl, double *fu, FFTW_Grid_Info grid_info);

/*! \fn double grid_cubic_interpolation(double x, double y, double z, double *f, double *fl, double *fu, FFTW_Grid_Info grid_info)
 *  \brief Tricubic interpolation on the grid */
double grid_cubic_interpolation(double x, double y, double z, double *f, double *fl, double *fu, FFTW_Grid_Info grid_info);


/*! \fn double cint(double x, double pm1, double p0, double pp1, double pp2)
 *  \brief Hermite interpolation in one direction. */
double cint(double x, double pm1, double p0, double pp1, double pp2);

//OK ABOVE HERE

/*! \fn double **grid_field_smooth(double **u, double *kernel, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world)
 *  \brief smooth a field using a kernel */
double **grid_field_smooth(double **u, double *kernel, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world);

/*! \fn double *grid_make_gaussian_kernel(double r_cells, FFTW_Grid_Info grid_info)
 *  \brief make a gaussian kernel with stddev r_cells */
double *grid_make_gaussian_kernel(double r_cells, FFTW_Grid_Info grid_info);


/*! \fn double **grid_field_smooth(double **u, double *kernel, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world)
 *  \brief convolve grid A with grid B */
double *grid_convolve(double *A, double *B, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world);


/*! \fn double *generate_gaussian_field(double (*Pk_power_spectrum)(double,void *), void *params, int iseed, FFTW_Grid_Info grid_info)
 *  \brief  Given a power spectrum Pk_power_spectrum that depends on wavenumber and some parameters *params, and a random number generator 
 *          seed iseed, return a gaussian random field with power spectrum Pk_power_spectrum. */
double *generate_gaussian_field(double (*Pk_power_spectrum)(double,void *), void *params, int iseed, FFTW_Grid_Info grid_info, MPI_Comm world);


/*! \fn double *grid_uniform_white_noise(double mu, double sigma_squared, int iseed, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world) 
 *  \brief Produces a fft grid of uniform variates with white noise, with mean mu and variance sigma_squared, with seed iseed.*/
double *grid_uniform_white_noise(double mu, double sigma_squared, int iseed, FFTW_Grid_Info grid_info);

/*! \fn double *grid_normal_white_noise(double mu, double sigma_squared, int iseed, FFTW_Grid_Info grid_info) 
 *  \brief Produces a fft grid of normal variates with white noise, with mean mu and variance sigma_squared, with seed iseed. */
double *grid_normal_white_noise(double mu, double variance, int iseed, FFTW_Grid_Info grid_info);


/*! \fn void grid_transform_apply_transfer_function(double (*Pk_power_spectrum)(double,void*),void *params, fftw_complex *cdata, FFTW_Grid_Info grid_info)
 *  \brief Given a power spectrum Pk, that depends on wavenumber and some parameters *params, multiply complex grid transform cdata by the 
 *         transfer function T(k) = sqrt( (2*pi/L)^3 Pk) */
void grid_transform_apply_transfer_function(double (*Pk_power_spectrum)(double,void*),void *params, fftw_complex *cdata, FFTW_Grid_Info grid_info);

/*! \fn void grid_transform_apply_transfer_function_2d(double (*Pk_power_spectrum)(double,void*),void *params, fftw_complex *cdata, FFTW_Grid_Info grid_info)
 *  \brief Given a power spectrum Pk, that depends on wavenumber and some parameters *params, multiply complex grid transform cdata by the 
 *         transfer function T(k) = sqrt( (2*pi/L)^3 Pk) */
void grid_transform_apply_transfer_function_2d(double (*Pk_power_spectrum)(double,void*),void *params, fftw_complex *cdata, FFTW_Grid_Info grid_info);


//routines for multidimensional fields on grids


/*! \fn double **generate_forcing_field(double (*Pk_power_spectrum)(double,void *), void *params, int iseed, FFTW_Grid_Info grid_info, rfftwnd_mpi_plan plan, rfftwnd_mpi_plan iplan, int myid, int numprocs, MPI_Comm world)
 *  \brief  Given a power spectrum Pk_power_spectrum that depends on wavenumber and some parameters *params, and a random number generator 
            seed iseed, return a ndim-dimensional gaussian random field with power spectrum Pk_power_spectrum. 
double **generate_forcing_field(double (*Pk_power_spectrum)(double,void *), void *params, int iseed, FFTW_Grid_Info grid_info, rfftwnd_mpi_plan plan, rfftwnd_mpi_plan iplan, int myid, int numprocs, MPI_Comm world);











/*! \fn void energy_power_spectrum(double *&k_array, double *&P_k_array, int *nk, double **u, FFTW_Grid_Info grid_info, rfftwnd_mpi_plan plan, rfftwnd_mpi_plan iplan, int myid, int numprocs, MPI_Comm world)
 *  \brief Calculate the 3-d energy power spectrum.
void energy_power_spectrum(double *&k_array, double *&P_k, int *nk_return, double **u, FFTW_Grid_Info grid_info, rfftwnd_mpi_plan plan, rfftwnd_mpi_plan iplan, int myid, int numprocs, MPI_Comm world);

void energy_power_spectrum_2d(double *&k_array, double *&P_k_array, int *nk_return, double **u, FFTW_Grid_Info grid_info, rfftwnd_mpi_plan plan, rfftwnd_mpi_plan iplan, int myid, int numprocs, MPI_Comm world);


/*! \fn void energy_power_spectrum_1d(double *&k_array, double *&P_k_array, int *nk, double **u, FFTW_Grid_Info grid_info, rfftwnd_mpi_plan plan, rfftwnd_mpi_plan iplan, int myid, int numprocs, MPI_Comm world)
 *  \brief Calculate the 1-d energy power spectrum.
void energy_power_spectrum_1d(double *&k_array, double *&P_k, int *nk_return, double **u, FFTW_Grid_Info grid_info, rfftwnd_mpi_plan plan, rfftwnd_mpi_plan iplan, int myid, int numprocs, MPI_Comm world);


/*! \fn void construct_histogram_log10(double *&x_array, double *&P_x_array, int *nx, double *u, FFTW_Grid_Info grid_info, rfftwnd_mpi_plan plan, rfftwnd_mpi_plan iplan, int myid, int numprocs, MPI_Comm world)
 *  \brief Histogram the grid u w/a log10 abcissa.
void construct_histogram_log10(double *&x_array, double *&P_x_array, int *nx, double *u, FFTW_Grid_Info grid_info, rfftwnd_mpi_plan plan, rfftwnd_mpi_plan iplan, int myid, int numprocs, MPI_Comm world);

/*! \fn double *grid_histogram_log10(double *u, double *x_array, int nx, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world)
 *  \brief Calculate a histogram w/a log10 abcissa.
double *grid_histogram_log10(double *u, double *x_array, int nx, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world);

/*! \fn void initialize_histogram_x_array_log10(double *&x_array, int *nx, double log10_x_min, double log10_x_max, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world)
 *  \brief Routine to initialize the x_array for a histogram
void initialize_histogram_x_array_log10(double *&x_array, int *nx, double log10_x_min, double log10_x_max, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world);

/*! \fn void construct_power_spectrum(double *&k_array, double *&P_k_array, int *nk, double *u, FFTW_Grid_Info grid_info, rfftwnd_mpi_plan plan, rfftwnd_mpi_plan iplan, int myid, int numprocs, MPI_Comm world)
 *  \brief Calculate a power spectrum for a grid.
void construct_grid_power_spectrum(double *&k_array, double *&P_k_array, int *nk, double *u, FFTW_Grid_Info grid_info, rfftwnd_mpi_plan plan, rfftwnd_mpi_plan iplan, int myid, int numprocs, MPI_Comm world);

/*! \fn void construct_grid_power_spectrum_field(double *&k_array, double *&P_k, int *nk_return, double **u, FFTW_Grid_Info grid_info, rfftwnd_mpi_plan plan, rfftwnd_mpi_plan iplan, int myid, int numprocs, MPI_Comm world)
 *  \brief Calculate a power spectrum for a field.
void construct_grid_power_spectrum_field(double *&k_array, double *&P_k, int *nk_return, double **u, FFTW_Grid_Info grid_info, rfftwnd_mpi_plan plan, rfftwnd_mpi_plan iplan, int myid, int numprocs, MPI_Comm world);

/*! \fn void construct_field_ksf(double *&l_array, double *&S_l_array, int *nl, double **u, double *x, double *y, double *z, FFTW_Grid_Info grid_info, rfftwnd_mpi_plan plan, rfftwnd_mpi_plan iplan, int myid, int numprocs, MPI_Comm world)
 *  \brief Calculate the parallel structure function for a field in k-space.
void construct_field_ksf(double *&l_array, double *&S_l_array, int *nl_save, double **u, double *x, double *y, double *z, FFTW_Grid_Info grid_info, rfftwnd_mpi_plan plan, rfftwnd_mpi_plan iplan, int myid, int numprocs, MPI_Comm world);

/*! \fn void construct_field_structure_function_mode_full(double alpha, double *&l_array, double *&S_l_array, int *nl, double **u, double *x, double *y, double *z, FFTW_Grid_Info grid_info, rfftwnd_mpi_plan plan, rfftwnd_mpi_plan iplan, int myid, int numprocs, MPI_Comm world)
 *  \brief Driving function to calculate the structure functions for a field using the full N^6 operation. 
void construct_field_structure_function_mode_full(int mode, double alpha, double *&l_array, double *&S_l_array, int *nl_save, double **u, double *x, double *y, double *z, FFTW_Grid_Info grid_info, rfftwnd_mpi_plan plan, rfftwnd_mpi_plan iplan, int myid, int numprocs, MPI_Comm world);

/*! \fn void construct_field_structure_function_mode_random(double alpha, double *&l_array, double *&S_l_array, int *nl, double **u, double *x, double *y, double *z, FFTW_Grid_Info grid_info, rfftwnd_mpi_plan plan, rfftwnd_mpi_plan iplan, int myid, int numprocs, MPI_Comm world)
 *  \brief Driving function to calculate the structure functions for a field. 
void construct_field_structure_function_mode_random(int mode, double alpha, double *&l_array, double *&S_l_array, int *nl_save, double **u, double *x, double *y, double *z, FFTW_Grid_Info grid_info, rfftwnd_mpi_plan plan, rfftwnd_mpi_plan iplan, int myid, int numprocs, MPI_Comm world);

/*! \fn void construct_structure_function_random(double alpha, double *&l_array, double *&S_l_array, int *nl, double *u, double *x, double *y, double *z, FFTW_Grid_Info grid_info, rfftwnd_mpi_plan plan, rfftwnd_mpi_plan iplan, int myid, int numprocs, MPI_Comm world)
 *  \brief Function to calculate the structure functions for a scalar. 
void construct_structure_function_random(double alpha, double *&l_array, double *&S_l_array, int *nl_save, double *u, double *x, double *y, double *z, FFTW_Grid_Info grid_info, rfftwnd_mpi_plan plan, rfftwnd_mpi_plan iplan, int myid, int numprocs, MPI_Comm world);


/*! \fn void construct_field_structure_function_mode(double alpha, double *&l_array, double *&S_l_array, int *nl, double **u, double *x, double *y, double *z, FFTW_Grid_Info grid_info, rfftwnd_mpi_plan plan, rfftwnd_mpi_plan iplan, int myid, int numprocs, MPI_Comm world)
 *  \brief Driving function to calculate the structure functions for a field. 
void construct_field_structure_function_mode(int mode, double alpha, double *&l_array, double *&S_l_array, int *nl_save, double **u, double *x, double *y, double *z, FFTW_Grid_Info grid_info, rfftwnd_mpi_plan plan, rfftwnd_mpi_plan iplan, int myid, int numprocs, MPI_Comm world);

/*! \fn void construct_field_structure_function_mode_2d(double alpha, double *&l_array, double *&S_l_array, int *nl, double **u, double *x, double *y, double *z, FFTW_Grid_Info grid_info, rfftwnd_mpi_plan plan, rfftwnd_mpi_plan iplan, int myid, int numprocs, MPI_Comm world)
 *  \brief Driving function to calculate the structure functions for a field. 
void construct_field_structure_function_mode_2d(int mode, double alpha, double *&l_array, double *&S_l_array, int *nl_save, double **u, double *x, double *y, double *z, FFTW_Grid_Info grid_info, rfftwnd_mpi_plan plan, rfftwnd_mpi_plan iplan, int myid, int numprocs, MPI_Comm world);

/*! \fn void construct_field_structure_function(double alpha, double *&l_array, double *&S_l_array, int *nl, double **u, double *x, double *y, double *z, FFTW_Grid_Info grid_info, rfftwnd_mpi_plan plan, rfftwnd_mpi_plan iplan, int myid, int numprocs, MPI_Comm world)
 *  \brief Calculate the parallel structure function for a field.
void construct_field_structure_function(double alpha, double *&l_array, double *&S_l_array, int *nl_save, double **u, double *x, double *y, double *z, FFTW_Grid_Info grid_info, rfftwnd_mpi_plan plan, rfftwnd_mpi_plan iplan, int myid, int numprocs, MPI_Comm world);

/*! \fn void construct_field_perpendicular_structure_function(double alpha, double *&l_array, double *&S_l_array, int *nl, double **u, double *x, double *y, double *z, FFTW_Grid_Info grid_info, rfftwnd_mpi_plan plan, rfftwnd_mpi_plan iplan, int myid, int numprocs, MPI_Comm world)
 *  \brief Calculate the parallel structure function for a field.
void construct_field_perpendicular_structure_function(double alpha, double *&l_array, double *&S_l_array, int *nl_save, double **u, double *x, double *y, double *z, FFTW_Grid_Info grid_info, rfftwnd_mpi_plan plan, rfftwnd_mpi_plan iplan, int myid, int numprocs, MPI_Comm world);

/*! \fn void construct_field_magnitude_structure_function(double alpha, double *&l_array, double *&S_l_array, int *nl, double **u, double *x, double *y, double *z, FFTW_Grid_Info grid_info, rfftwnd_mpi_plan plan, rfftwnd_mpi_plan iplan, int myid, int numprocs, MPI_Comm world)
 *  \brief Calculate the velocity difference magnitude structure function for a field.
void construct_field_magnitude_structure_function(double alpha, double *&l_array, double *&S_l_array, int *nl_save, double **u, double *x, double *y, double *z, FFTW_Grid_Info grid_info, rfftwnd_mpi_plan plan, rfftwnd_mpi_plan iplan, int myid, int numprocs, MPI_Comm world);

/*! \fn void check_parsevals_theorem(double *u, FFTW_Grid_Info grid_info, rfftwnd_mpi_plan plan, rfftwnd_mpi_plan iplan, int myid, int numprocs, MPI_Comm world)
 *  \brief Check for energy conservation in power spectra for a scalar grid. 
void check_parsevals_theorem(double *u, FFTW_Grid_Info grid_info, rfftwnd_mpi_plan plan, rfftwnd_mpi_plan iplan, int myid, int numprocs, MPI_Comm world);

/*! \fn void check_parsevals_theorem_3d(double *u, FFTW_Grid_Info grid_info, rfftwnd_mpi_plan plan, rfftwnd_mpi_plan iplan, int myid, int numprocs, MPI_Comm world)
 *  \brief Check for energy conservation in power spectra for a 3-d scalar grid. 
void check_parsevals_theorem_3d(double *u, FFTW_Grid_Info grid_info, rfftwnd_mpi_plan plan, rfftwnd_mpi_plan iplan, int myid, int numprocs, MPI_Comm world);


/*! \fn void check_parsevals_theorem_2d(double *u, FFTW_Grid_Info grid_info, rfftwnd_mpi_plan plan, rfftwnd_mpi_plan iplan, int myid, int numprocs, MPI_Comm world)
 *  \brief Check for energy conservation in power spectra for a scalar grid. 
void check_parsevals_theorem_2d(double *u, FFTW_Grid_Info grid_info, rfftwnd_mpi_plan plan, rfftwnd_mpi_plan iplan, int myid, int numprocs, MPI_Comm world);

/*! \fn void check_parsevals_theorem_field(double **u, FFTW_Grid_Info grid_info, rfftwnd_mpi_plan plan, rfftwnd_mpi_plan iplan, int myid, int numprocs, MPI_Comm world)
 *  \brief Check for energy conservation in power spectra for a vector field. 
void check_parsevals_theorem_field(double **u, FFTW_Grid_Info grid_info, rfftwnd_mpi_plan plan, rfftwnd_mpi_plan iplan, int myid, int numprocs, MPI_Comm world);

/*! \fn void check_parsevals_theorem_field_3d(double **u, FFTW_Grid_Info grid_info, rfftwnd_mpi_plan plan, rfftwnd_mpi_plan iplan, int myid, int numprocs, MPI_Comm world)
 *  \brief Check for energy conservation in power spectra for a 3-d vector field. 
void check_parsevals_theorem_field_3d(double **u, FFTW_Grid_Info grid_info, rfftwnd_mpi_plan plan, rfftwnd_mpi_plan iplan, int myid, int numprocs, MPI_Comm world);


/*! \fn void check_parsevals_theorem_field_2d(double **u, FFTW_Grid_Info grid_info, rfftwnd_mpi_plan plan, rfftwnd_mpi_plan iplan, int myid, int numprocs, MPI_Comm world)
 *  \brief Check for energy conservation in power spectra for a vector field. 
void check_parsevals_theorem_field_2d(double **u, FFTW_Grid_Info grid_info, rfftwnd_mpi_plan plan, rfftwnd_mpi_plan iplan, int myid, int numprocs, MPI_Comm world);

/*! \fn double *power_spectrum(fftw_complex *cdata, double *k_array, int nk, int local_y_start_after_transpose, int local_ny_after_transpose, int nx, int ny, int nz, double BoxSize, int myid, int numprocs, MPI_Comm world)
 *  \brief This function takes a forward transform (e.g. a complex grid), already normalized by 1./(nx)^ndim and returns a n-dimensional power spectrum vs. k
  
double *power_spectrum(fftw_complex *cdata, double *k_array, int nk, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world);

/*! \fn double *power_spectrum_3d(fftw_complex *cdata, double *k_array, int nk, int local_y_start_after_transpose, int local_ny_after_transpose, int nx, int ny, int nz, double BoxSize, int myid, int numprocs, MPI_Comm world)
 *  \brief This function takes a forward transform (e.g. a complex grid), already normalized by 1./(nx*ny*nz) and returns a 3-dimensional power spectrum vs. k
  
double *power_spectrum_3d(fftw_complex *cdata, double *k_array, int nk, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world);


/*! \fn double *power_spectrum_field_3d(fftw_complex *cukx, fftw_complex *cuky, fftw_complex *cukz, double *k_array, int nk, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world)
 *  \brief This function takes a forward transform (e.g. a complex grid), already normalized by 1./(nx)^dim and returns a n-dimensional power spectrum vs. k
  
double *power_spectrum_field_3d(fftw_complex *cukx, fftw_complex *cuky, fftw_complex *cukz, double *k_array, int nk, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world);


/*! \fn double *power_spectrum_2d(fftw_complex *cdata, double *k_array, int nk, int local_y_start_after_transpose, int local_ny_after_transpose, int nx, int ny, int nz, double BoxSize, int myid, int numprocs, MPI_Comm world)
 *  \brief This function takes a forward transform (e.g. a complex grid), already normalized by 1./(nx*ny) and returns a 2-dimensional power spectrum vs. k
 
double *power_spectrum_2d(fftw_complex *cdata, double *k_array, int nk, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world);

/*! \fn double *power_spectrum_field(fftw_complex *cdata, double *k_array, int nk, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world)
 *  \brief This function takes a forward transform (e.g. a complex grid), already normalized by 1./(nx)^dim and returns a n-dimensional power spectrum vs. k
  
double *power_spectrum_field(fftw_complex **cdata, double *k_array, int nk, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world);


/*! \fn double *power_spectrum_field_2d(fftw_complex *cukx, fftw_complex *cuky, fftw_complex *cukz, double *k_array, int nk, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world)
 *  \brief This function takes a forward transform (e.g. a complex grid), already normalized by 1./(nx*ny) and returns a 1-dimensional power spectrum vs. k
  
double *power_spectrum_field_2d(fftw_complex *cukx, fftw_complex *cuky, double *k_array, int nk, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world);

/*! \fn int find_power_spectrum_nk(FFTW_Grid_Info grid_info)
 *  \brief Routine to find the extent of 1-d power spectrum arrays.
 
int find_power_spectrum_nk(FFTW_Grid_Info grid_info);

/*! \fn void initialize_power_spectrum_k_array(double *&k_array, int *nk, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world)
 *  \brief Routine to initialize the k_array for power spectra 
void initialize_power_spectrum_k_array(double *&k_array, int *nk, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world);


/*! \fn int find_structure_function_nl(FFTW_Grid_Info grid_info)
 *  \brief Routine to find the extent of 1-d structure function arrays.
 
int find_structure_function_nl(FFTW_Grid_Info grid_info);

/*! \fn void initialize_structure_function_l_array(double *&l_array, int *nl, double *l_min, double *l_max, double *x, double *y, double *z, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world)
 *  \brief Routine to initialize the l_array for structure functions
void initialize_structure_function_l_array(double *&l_array, int *nl, double *l_min, double *l_max, double *x, double *y, double *z, FFTW_Grid_Info grid_info, int myid, int numprocs, MPI_Comm world);


/*! \fn double *grid_normal_white_noise(double mu, double sigma_squared, int iseed, FFTW_Grid_Info grid_info, rfftwnd_mpi_plan plan, rfftwnd_mpi_plan iplan, int myid, int numprocs, MPI_Comm world) 
 *  \brief Produces a fft grid of normal variates with white noise, with mean mu and variance sigma_squared, with seed iseed. 
double *grid_normal_white_noise(double mu, double variance, int iseed, FFTW_Grid_Info grid_info, rfftwnd_mpi_plan plan, rfftwnd_mpi_plan iplan, int myid, int numprocs, MPI_Comm world);

*/


#endif //GRID_OPERATIONS
