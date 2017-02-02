#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#ifndef NO_FFTW
#include <fftw3-mpi.h>
#include <fftw3.h>
#endif /*NO_FFTW*/
#include "read_athena_binary.h"
#include "routines.h"


int main(int argc, char **argv)
{
	int ijk;
	char fdir[200];
	char filename[200];
	int flag_error = 0;
	int flag_error_tot = 1;

	double *xout;

	int isnap;

	AthenaBinary A;

	MPI_Init(&argc,&argv);
	
	//A.world = MPI_COMM_WORLD;
	//MPI_Comm_rank(A.world,&A.myid);
	//MPI_Comm_size(A.world,&A.numprocs);

	A.Initialize();

	//check number of arguments
	if(!((argc==3)||(argc==5)))
	{
		if(A.myid==0)
		{
			printf("./cat_athena_mpi_bin fdir filename(base) [nfiles isnap]\n");
			fflush(stdout);
			flag_error = 1;
		}

	}

	sprintf(fdir,"%s",argv[1]);
	sprintf(filename,"%s",argv[2]);
	

	if(argc==5)
	{
		isnap    = atoi(argv[3]);
		A.nfiles = atoi(argv[4]);
	}else{
		A.nfiles = 1;
	}


	if(A.nfiles==1)
	{
		A.ReadAthenaBinary(filename);
	}else{

		A.ReadAthenaBinarySplit(fdir,filename,isnap);
	}

		/*data does not seems to be in place here*/
		/*for(int ip=0;ip<A.numprocs;ip++)
		{
			if(A.myid==ip)
			{
				for(int i=0;i<A.grid_info.local_nx;i++)
				{	
					ijk = grid_ijk(i,191,0,A.grid_info);
					printf("main %d %e\n",i+A.grid_info.local_x_start,A.b_field[2][ijk]);
					fflush(stdout);
				}
			}
			MPI_Barrier(A.world);
		}*/

	xout = (double *) calloc(A.nx*A.ny,sizeof(double *));

	//density first
	for(int i=0;i<A.nx;i++)
		for(int j=0;j<A.ny;j++)
			for(int k=0;k<A.nz;k++)
			{
				ijk = grid_ijk(i,j,k,A.grid_info);
				xout[i*ny + j] += A.density[ijk];
			}

	FILE *fp;
	char fxout[200];

	sprintf(fxout,"density.%04d.dat",isnap);
	fp = fopen(fxout,"w");
	fwrite(fp,&A.nx,1,sizeof(int));
	fwrite(fp,&A.ny,1,sizeof(int));
	fwrite(fp,xout,A.nx*A.ny,sizeof(double));
	fclose(fp);

	//total velocity next
	for(int i=0;i<A.nx;i++)
		for(int j=0;j<A.ny;j++)
			xout[i*ny+j] = 0.0;

	double vx, vy, vz;
	for(int i=0;i<A.nx;i++)
		for(int j=0;j<A.ny;j++)
			for(int k=0;k<A.nz;k++)
			{
				ijk = grid_ijk(i,j,k,A.grid_info);
				vx = A.velocity[0][ijk];
				vy = A.velocity[1][ijk];
				vz = A.velocity[2][ijk];

				xout[i*ny + j] += sqrt(vx*vx + vy*vy + vz*vz);
			}

	sprintf(fxout,"vtot.%04d.dat",isnap);
	fp = fopen(fxout,"w");
	fwrite(fp,&A.nx,1,sizeof(int));
	fwrite(fp,&A.ny,1,sizeof(int));
	fwrite(fp,xout,A.nx*A.ny,sizeof(double));
	fclose(fp);

	//x velocity next
	for(int i=0;i<A.nx;i++)
		for(int j=0;j<A.ny;j++)
			xout[i*ny+j] = 0.0;

	for(int i=0;i<A.nx;i++)
		for(int j=0;j<A.ny;j++)
			for(int k=0;k<A.nz;k++)
			{
				ijk = grid_ijk(i,j,k,A.grid_info);
				vx = A.velocity[0][ijk];

				xout[i*ny + j] += vx;
			}

	sprintf(fxout,"vx.%04d.dat",isnap);
	fp = fopen(fxout,"w");
	fwrite(fp,&A.nx,1,sizeof(int));
	fwrite(fp,&A.ny,1,sizeof(int));
	fwrite(fp,xout,A.nx*A.ny,sizeof(double));
	fclose(fp);


	//y velocity next
	for(int i=0;i<A.nx;i++)
		for(int j=0;j<A.ny;j++)
			xout[i*ny+j] = 0.0;

	for(int i=0;i<A.nx;i++)
		for(int j=0;j<A.ny;j++)
			for(int k=0;k<A.nz;k++)
			{
				ijk = grid_ijk(i,j,k,A.grid_info);
				vx = A.velocity[1][ijk];

				xout[i*ny + j] += vy;
			}

	sprintf(fxout,"vy.%04d.dat",isnap);
	fp = fopen(fxout,"w");
	fwrite(fp,&A.nx,1,sizeof(int));
	fwrite(fp,&A.ny,1,sizeof(int));
	fwrite(fp,xout,A.nx*A.ny,sizeof(double));
	fclose(fp);


	//z velocity next
	for(int i=0;i<A.nx;i++)
		for(int j=0;j<A.ny;j++)
			xout[i*ny+j] = 0.0;
	memset(xout,0,A.nx*A.ny*sizeof(double));

	for(int i=0;i<A.nx;i++)
		for(int j=0;j<A.ny;j++)
			for(int k=0;k<A.nz;k++)
			{
				ijk = grid_ijk(i,j,k,A.grid_info);
				vz = A.velocity[2][ijk];

				xout[i*ny + j] += vz;
			}

	sprintf(fxout,"vz.%04d.dat",isnap);
	fp = fopen(fxout,"w");
	fwrite(fp,&A.nx,1,sizeof(int));
	fwrite(fp,&A.ny,1,sizeof(int));
	fwrite(fp,xout,A.nx*A.ny,sizeof(double));
	fclose(fp);

	free(xout);

	//sprintf(filename,"test.dat");
	//A.WriteAthenaBinary(filename);

	MPI_Finalize();
	return 0;
}
