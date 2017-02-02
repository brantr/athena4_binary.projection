#include<mpi.h>
#include<math.h>
#include"read_athena_binary.h"
#include"routines.h"
#ifndef NO_FFTW
#include<fftw3-mpi.h>
#include<fftw3.h>
#include"grid_fft.h"
#endif /*NO_FFTW*/

AthenaBinary::AthenaBinary(void)
{
}
AthenaBinary::~AthenaBinary(void)
{
	/*free memory*/
	DestroyData();
}
void AthenaBinary::Initialize(void)
{
	flag_memory = 0;
	world = MPI_COMM_WORLD;
	MPI_Comm_rank(world,&myid);
	MPI_Comm_size(world,&numprocs);
	grid_info.ndim = 3;

}
void AthenaBinary::AllocateData(void)
{
	char variable_name[200];
	int nxi; 

	//printf("HERE\n"); 
	//fflush(stdout);
	if(!flag_memory)
	{
		//printf("Allocating data... myid %d nx %d ny %d nz %d\n",myid,grid_info.nx,grid_info.ny,grid_info.nz);
		//fflush(stdout);

		//allocate grid coordinates
		x = calloc_double_array(grid_info.nx);
		y = calloc_double_array(grid_info.ny); 
		z = calloc_double_array(grid_info.nz); 

		//data grids

		sprintf(variable_name,"density");
		density = allocate_real_fftw_grid(grid_info);
	
		if(!flag_isothermal)
		{	
			sprintf(variable_name,"etot");
			etot = allocate_real_fftw_grid(grid_info);
		}

		sprintf(variable_name,"velocity");	
		velocity = allocate_field_fftw_grid(grid_info.ndim,grid_info);

		if(flag_mhd)
		{
			sprintf(variable_name,"b_field");  
			b_field = allocate_field_fftw_grid(grid_info.ndim,grid_info);
		}

		if(flag_scalars)
		{
			sprintf(variable_name,"scalars");
			scalars = allocate_field_fftw_grid(nscalars,grid_info);
		}

		if(flag_gravity)
		{
			sprintf(variable_name,"phi");
			phi = allocate_real_fftw_grid(grid_info);
		}

		/*printf("numprocs %d\n",numprocs);*/
		//fflush(stdout);


		//share the local size on each processor
		nx_local_array          = calloc_int_array(numprocs);
		nx_local_start_array    = calloc_int_array(numprocs);
		n_local_real_size_array = calloc_int_array(numprocs);

		for(int i=0;i<numprocs;i++)
		{
			nxi = grid_info.nx_local;

			MPI_Bcast(&nxi,1,MPI_INT,i,world);

			nx_local_array[i] = nxi;


			nxi = grid_info.nx_local_start;

			MPI_Bcast(&nxi,1,MPI_INT,i,world);

			nx_local_start_array[i] = nxi;

			nxi = grid_info.n_local_real_size;

			MPI_Bcast(&nxi,1,MPI_INT,i,world);

			n_local_real_size_array[i] = nxi;

			/*if(myid==i)
			{
				printf("myid %d total local size %d tlsa[%d] %d nxi %d\n",myid,grid_info.n_local_real_size,i,n_local_real_size_array[i],nxi);
				fflush(stdout);
			}*/
		}
		



		//remember that memory was allocated
		flag_memory = 1;
	}
}
void AthenaBinary::DestroyData(void)
{
	if(flag_memory)
	{

		//free grid coordinates
		free(x);
		free(y);
		free(z);

		//free data
		free(density); 
		if(!flag_isothermal)
			free(etot);
		deallocate_field_fftw_grid(velocity, ndim, grid_info);

		if(flag_gravity)
			free(phi);

		if(flag_scalars)
			deallocate_field_fftw_grid(scalars, nscalars, grid_info);
		
		if(flag_mhd)
			deallocate_field_fftw_grid(b_field, ndim, grid_info);

		free(nx_local_array);
		free(nx_local_start_array);
		free(n_local_real_size_array);

	  	if(nfiles>1)
		{
			//free some memory
			/*free(nx_file);
			free(ny_file);
			free(nz_file);
			free(nx_start);
			free(ny_start);
			free(nz_start);*/
		}

		//remember that memory is unallocated
		flag_memory = 0;
	}
}

double AthenaBinary::energy(int i, int j, int k)
{
	double IE;
	double KE;
	int ijk;

	ijk = grid_ijk(i,j,k,grid_info);

	if(flag_isothermal)
	{
		IE = pressure(i,j,k);
		KE = 0.5*( velocity[0][ijk]*velocity[0][ijk] + velocity[1][ijk]*velocity[1][ijk] + velocity[2][ijk]*velocity[2][ijk]);
		KE *= density[ijk];

		return IE+KE;
	}else{
		//return total energy
		return etot[ijk];
	}
}

double AthenaBinary::pressure(int i, int j, int k)
{
	double KE, ME;
	int ijk;

	ijk = grid_ijk(i,j,k,grid_info);

	if(flag_isothermal)
	{
		//isothermal pressure
		return c_s_iso*c_s_iso*density[ijk];
	}else{
		if(flag_pressure)
		{
			//internal energy is tracked as an advected
			//quantity
			return gamma_minus_1*scalars[ipressure][ijk];
		}
		//internal energy -- energy less kinetic energy less mag pressure
	
		ME = 0.5*( b_field[0][ijk]*b_field[0][ijk] + b_field[1][ijk]*b_field[1][ijk] + b_field[2][ijk]*b_field[2][ijk]);
		KE = 0.5*( velocity[0][ijk]*velocity[0][ijk] + velocity[1][ijk]*velocity[1][ijk] + velocity[2][ijk]*velocity[2][ijk]);
		KE *= density[ijk];

	 	return gamma_minus_1*( etot[ijk] - KE - ME );
	}
}

void AthenaBinary::ShowAthenaHeader(void)
{
	if(myid==0)
	{
		printf("files    = %d\n",nfiles);
#ifdef ATHENA4
		printf("CoordSys = %d\n",CoordinateSystem);
#endif /*ATHENA4*/
		printf("nx       = %d\n",nx);
		printf("ny       = %d\n",ny);
		printf("nz       = %d\n",nz);
		printf("nvar     = %d\n",nvar);
		printf("nscalars = %d\n",nscalars);
		printf("ngrav    = %d\n",ngrav);

#ifdef ATHENA4
		printf("flag_tracers    = %d\n",flag_tracers);
#endif /*ATHENA4*/
		printf("flag_scalars    = %d\n",flag_scalars);
		printf("flag_isothermal = %d\n",flag_isothermal);
		printf("flag_gravity    = %d\n",flag_gravity);
		printf("flag_mhd        = %d\n",flag_mhd);


		printf("ndim            = %d\n",ndim);


		printf("gamma_minus_1   = %e\n",gamma_minus_1);
		printf("c_s_iso         = %e\n",c_s_iso);
		printf("t               = %e\n",t);
		printf("dt              = %e\n",dt);
	}
}
void AthenaBinary::ShowAthenaBinary(void)
{
	if(myid==0)
	{
		printf("files    = %d\n",nfiles);
		printf("nx       = %d\n",nx);
		printf("ny       = %d\n",ny);
		printf("nz       = %d\n",nz);
		printf("nvar     = %d\n",nvar);
		printf("nscalars = %d\n",nscalars);
		printf("ngrav    = %d\n",ngrav);

		printf("flag_scalars    = %d\n",flag_scalars);
		printf("flag_isothermal = %d\n",flag_isothermal);
		printf("flag_gravity    = %d\n",flag_gravity);
		printf("flag_mhd        = %d\n",flag_mhd);


		printf("ndim            = %d\n",ndim);


		printf("gamma_minus_1   = %e\n",gamma_minus_1);
		printf("c_s_iso         = %e\n",c_s_iso);
		printf("t               = %e\n",t);
		printf("dt              = %e\n",dt);
	
		ShowAthenaHeader();



		printf("x[%d]              = %e\n",0,x[0]);
		printf("x[%d]              = %e\n",nx-1,x[nx-1]);
		printf("y[%d]              = %e\n",0,y[0]);
		printf("y[%d]              = %e\n",ny-1,y[ny-1]);
		printf("z[%d]              = %e\n",0,z[0]);
		printf("z[%d]              = %e\n",nz-1,z[nz-1]);
		
		printf("BoxSize            = %e\n",grid_info.BoxSize);

		fflush(stdout);
	}
}
void AthenaBinary::ReadAthenaHeader(char *fname)
{
	FILE *fp;
	float fb;

	if(myid==0)
	{
		printf("Reading header %s....\n",fname);
		fflush(stdout);

		//open the athena binary file
		fp = fopen_brant(fname, "r");

		//start reading in data

#ifdef   ATHENA4
		fread(&CoordinateSystem,sizeof(int),1,fp);
#endif /*ATHENA4*/
		fread(&nx,sizeof(int),1,fp);
		fread(&ny,sizeof(int),1,fp);
		fread(&nz,sizeof(int),1,fp);
		fread(&nvar,sizeof(int),1,fp);
		fread(&nscalars,sizeof(int),1,fp);
		fread(&ngrav,sizeof(int),1,fp);
#ifdef   ATHENA4
		fread(&flag_tracers,sizeof(int),1,fp);
#endif /*ATHENA4*/

		//set flags

		//scalars?
		flag_scalars = 0;
		if(nscalars)
			flag_scalars = 1;

		//isothermal?
		flag_isothermal = 1;
		if( ((nvar-nscalars)==5)||((nvar-nscalars)==8) )
		{
			printf("nvar %d nscal %d diff %d\n",nvar,nscalars,nvar-nscalars);
			fflush(stdout);
			flag_isothermal = 0;
		}
			

		//mhd?
		flag_mhd = 0;
		if( ((nvar-nscalars)==7)||((nvar-nscalars)==8) )
			flag_mhd = 1;

		//gravity?
		flag_gravity = 0;
		if(ngrav)
			flag_gravity = 1;

		//count number of dimensions
		ndim = 3;//always 3
		/*if(nx>1)
			ndim++;
		if(ny>1)
			ndim++;
		if(nz>1)
			ndim++;*/

		fread(&fb,sizeof(float),1,fp);
		gamma_minus_1 = fb;
		fread(&fb,sizeof(float),1,fp);
		c_s_iso       = fb;

		fread(&fb,sizeof(float),1,fp);
		t    = fb;
		fread(&fb,sizeof(float),1,fp);
		dt   = fb;

#ifdef   READ_PRESSURE
		flag_pressure = 1;
#else  //READ_PRESSURE
		flag_pressure = 0;
#endif //READ_PRESSURE



		//pressure is always the first scalar
		ipressure = 0;

		fclose(fp);
	}

	//broadcast information
#ifdef ATHENA4
	MPI_Bcast(&CoordinateSystem,1,MPI_INT,0,world);
#endif /*ATHENA4*/
	MPI_Bcast(&nx,1,MPI_INT,0,world);
	MPI_Bcast(&ny,1,MPI_INT,0,world);
	MPI_Bcast(&nz,1,MPI_INT,0,world);
	MPI_Bcast(&ndim,1,MPI_INT,0,world);
	MPI_Bcast(&nvar,1,MPI_INT,0,world);
	MPI_Bcast(&nscalars,1,MPI_INT,0,world);
	MPI_Bcast(&ngrav,1,MPI_INT,0,world);
#ifdef ATHENA4
	MPI_Bcast(&flag_tracers,1,MPI_INT,0,world);
#endif /*ATHENA4*/

	MPI_Bcast(&flag_gravity,1,MPI_INT,0,world);
	MPI_Bcast(&flag_scalars,1,MPI_INT,0,world);
	MPI_Bcast(&flag_mhd,1,MPI_INT,0,world);
	MPI_Bcast(&flag_isothermal,1,MPI_INT,0,world);
	MPI_Bcast(&flag_memory,1,MPI_INT,0,world);
	MPI_Bcast(&flag_pressure,1,MPI_INT,0,world);
	MPI_Bcast(&ipressure,1,MPI_INT,0,world);

	MPI_Bcast(&gamma_minus_1,1,MPI_DOUBLE,0,world);
	MPI_Bcast(&c_s_iso,1,MPI_DOUBLE,0,world);
	MPI_Bcast(&t,1,MPI_DOUBLE,0,world);
	MPI_Bcast(&dt,1,MPI_DOUBLE,0,world);

	//set grid info
	grid_info.nx = nx;
	grid_info.ny = ny;
	grid_info.nz = nz;
	if(grid_info.ndim == 2)
		grid_info.nz = 0;
}
void AthenaBinary::ReadAthenaBinary(char *fname)
{
	FILE *fp;

	float fb;

	int   nxi;

	int ijk;
	
	int nzl;

	float *xb;


	if(myid==0)
	{
#ifdef LOUD
		printf("Reading file %s (pid %d)....\n",fname,myid);
		fflush(stdout);
#endif

		//open the athena binary file
		fp = fopen_brant(fname, "r");

		if(!fp)
		{	
			printf("Error opening %s\n",fname);
			fflush(stdout);
		}

		//start reading in data

#ifdef ATHENA4
		fread(&CoordinateSystem,sizeof(int),1,fp);
#endif /*ATHENA4*/
		fread(&nx,sizeof(int),1,fp);
		fread(&ny,sizeof(int),1,fp);
		fread(&nz,sizeof(int),1,fp);
		fread(&nvar,sizeof(int),1,fp);
		fread(&nscalars,sizeof(int),1,fp);
		fread(&ngrav,sizeof(int),1,fp);
#ifdef ATHENA4
		fread(&flag_tracers,sizeof(int),1,fp);
#endif /*ATHENA4*/

#ifdef LOUD
		printf("nx %d ny %d nz %d nvar %d ns %d ng %d\n",nx,ny,nz,nvar,nscalars,ngrav);
#endif

		//set flags

		//scalars?
		flag_scalars = 0;
		if(nscalars)
			flag_scalars = 1;

		//isothermal?
		flag_isothermal = 1;
		if( ((nvar-nscalars)==5)||((nvar-nscalars)==8) )
		{
#ifdef LOUD
			printf("nvar %d nscal %d diff %d\n",nvar,nscalars,nvar-nscalars);
			fflush(stdout);
#endif
			flag_isothermal = 0;
		}
			

		//mhd?
		flag_mhd = 0;
		if( ((nvar-nscalars)==7)||((nvar-nscalars)==8) )
			flag_mhd = 1;

		//gravity?
		flag_gravity = 0;
		if(ngrav)
			flag_gravity = 1;

		//count number of dimensions
		ndim = 3;//always 3
		/*if(nx>1)
			ndim++;
		if(ny>1)
			ndim++;
		if(nz>1)
			ndim++;*/

		fread(&fb,sizeof(float),1,fp);
		gamma_minus_1 = fb;
		fread(&fb,sizeof(float),1,fp);
		c_s_iso       = fb;

		fread(&fb,sizeof(float),1,fp);
		t    = fb;
		fread(&fb,sizeof(float),1,fp);
		dt   = fb;

#ifdef   READ_PRESSURE
		flag_pressure = 1;
#else  //READ_PRESSURE
		flag_pressure = 0;
#endif //READ_PRESSURE



		//pressure is always the first scalar
		ipressure = 0;

	}

	//broadcast information
#ifdef ATHENA4
	MPI_Bcast(&CoordinateSystem,1,MPI_INT,0,world);
#endif /*ATHENA4*/
	MPI_Bcast(&nx,1,MPI_INT,0,world);
	MPI_Bcast(&ny,1,MPI_INT,0,world);
	MPI_Bcast(&nz,1,MPI_INT,0,world);
	MPI_Bcast(&ndim,1,MPI_INT,0,world);
	MPI_Bcast(&nvar,1,MPI_INT,0,world);
	MPI_Bcast(&nscalars,1,MPI_INT,0,world);
	MPI_Bcast(&ngrav,1,MPI_INT,0,world);
#ifdef ATHENA4
	MPI_Bcast(&flag_tracers,1,MPI_INT,0,world);
#endif /*ATHENA4*/

	MPI_Bcast(&flag_gravity,1,MPI_INT,0,world);
	MPI_Bcast(&flag_scalars,1,MPI_INT,0,world);
	MPI_Bcast(&flag_mhd,1,MPI_INT,0,world);
	MPI_Bcast(&flag_isothermal,1,MPI_INT,0,world);
	MPI_Bcast(&flag_memory,1,MPI_INT,0,world);
	MPI_Bcast(&flag_pressure,1,MPI_INT,0,world);
	MPI_Bcast(&ipressure,1,MPI_INT,0,world);

	MPI_Bcast(&gamma_minus_1,1,MPI_DOUBLE,0,world);
	MPI_Bcast(&c_s_iso,1,MPI_DOUBLE,0,world);
	MPI_Bcast(&t,1,MPI_DOUBLE,0,world);
	MPI_Bcast(&dt,1,MPI_DOUBLE,0,world);

	//set grid info
	grid_info.nx = nx;
	grid_info.ny = ny;
	grid_info.nz = nz;
	if(grid_info.ndim == 2)
		grid_info.nz = 0;

	//printf("About to initialize fftw (pid %d, nx %d ny %d nz %d)\n",myid,grid_info.nx,grid_info.ny,grid_info.nz);
	//fflush(stdout);


#ifndef NO_FFTW

	//initialize fftw plans
        initialize_mpi_local_sizes(&grid_info, world);

#else	//NO_FFTW

	//initialize local sizes
        athena_initialize_mpi_local_sizes(&grid_info, myid, numprocs, world);

#endif	//NO_FFTW


	//allocate memory
	
	AllocateData();	


	//printf("Data allocated...\n");
	//fflush(stdout);
	

	
	/*for(int i=0;i<numprocs;i++)
	{
		if(myid==i)
		{
			printf("id %d nx_local %d ny %d nz %d\n",i,nx_local_array[i]);
			printf("id %d nx_locala %d nx_local %d nx %d ny %d nz %d\n",i,nx_local_array[i],grid_info.nx_local,grid_info.nx,grid_info.ny,grid_info.nz);
			printf("id %d nx_locala %d nx_local %d nx %d ny %d nz %d tls %d\n",i,nx_local_array[i],grid_info.nx_local,grid_info.nx,grid_info.ny,grid_info.nz,grid_info.n_local_real_size);
			fflush(stdout);
		}
	}*/

	//begin reading in data

	//read in x,y,z

	if(myid==0)
	{
		//x
		for(int i=0;i<nx;i++)
		{
			fread(&fb,sizeof(float),1,fp);
			x[i] = fb;
		}
		//y
		for(int i=0;i<ny;i++)
		{
			fread(&fb,sizeof(float),1,fp);
			y[i] = fb;
		}
		//z
		for(int i=0;i<nz;i++)
		{
			fread(&fb,sizeof(float),1,fp);
			z[i] = fb;
		}

		
	}
	//send x,y,z
	MPI_Bcast(x,nx,MPI_DOUBLE,0,world);
	MPI_Bcast(y,ny,MPI_DOUBLE,0,world);
	MPI_Bcast(z,nz,MPI_DOUBLE,0,world);


	grid_info.BoxSize  = (x[nx-1]-x[0]) + (x[1]-x[0]);
	grid_info.dx       = (x[1]-x[0]);
	grid_info.dy       = (y[1]-y[0]);
	grid_info.dz       = (z[1]-z[0]);
	grid_info.dV       = grid_info.dx*grid_info.dy*grid_info.dz;
	grid_info.dVk      = 1.0;

	ShowAthenaBinary();


	//read in the density

	ReadAthenaBinaryField(fp, 0);


	//read in the x-velocity

	ReadAthenaBinaryField(fp, 1);

	//read in the y-velocity

	ReadAthenaBinaryField(fp, 2);

	//read in the z-velocity

	ReadAthenaBinaryField(fp, 3);


	if(!flag_isothermal)
	{
		//read in the total energy

		ReadAthenaBinaryField(fp, 4);
	}

	if(flag_mhd)
	{
		//read in the Bx

		ReadAthenaBinaryField(fp, 5);

		//read in the By

		ReadAthenaBinaryField(fp, 6);

		//read in the Bz

		ReadAthenaBinaryField(fp, 7);

	}
	//MPI_Abort(world,-1);
	

	if(flag_scalars)
	{
		for(int i=0;i<nscalars;i++)
		{
			//read in the scalars

			ReadAthenaBinaryField(fp, 8+i);
		}
	}

	if(flag_gravity)
	{
		//read in the gravitational potential

		ReadAthenaBinaryField(fp, 8);
	}



	if(!grid_info.nz)
	{
		nzl = 1;
	}else{
		nzl = grid_info.nz;
	}
	//Convert momenta into velocities
	for(int i=0;i<grid_info.nx_local;i++)
		for(int j=0;j<grid_info.ny;j++)
			for(int k=0;k<nzl;k++)
				for(int n=0;n<grid_info.ndim;n++)
				{
					ijk = grid_ijk(i,j,k,grid_info);
					velocity[n][ijk]/=density[ijk];
				}

	if(myid==0)
	{
		//printf("Done reading file.\n");
		//fflush(stdout);
		fclose(fp);
	}
}
void AthenaBinary::ReadAthenaInfoBinarySplit(char *fdir, char *fbase, int isnap)
{

	char  fname[200];
	FILE *fp;
	float fb;

	int   nxi;

	int ijk;

	int nzl;

	float *xmin;
	float *ymin;
	float *zmin;
	float *xmax;
	float *ymax;
	float *zmax;

	float xmin_cur;
	float ymin_cur;
	float zmin_cur;
	float xmax_cur;
	float ymax_cur;
	float zmax_cur;

	float xtest;
	float ytest;
	float ztest;

	fpos_t *file_position;

	nx = 0;
	ny = 0;
	nz = 0;


	//information for use with MPI output files
	nx_file = calloc_int_array(nfiles);
	ny_file = calloc_int_array(nfiles);
	nz_file = calloc_int_array(nfiles);
	nx_start = calloc_int_array(nfiles);
	ny_start = calloc_int_array(nfiles);
	nz_start = calloc_int_array(nfiles);

	xmin = calloc_float_array(nfiles);
	ymin = calloc_float_array(nfiles);
	zmin = calloc_float_array(nfiles);
	xmax = calloc_float_array(nfiles);
	ymax = calloc_float_array(nfiles);
	zmax = calloc_float_array(nfiles);

	file_position = (fpos_t *) malloc(nfiles*sizeof(fpos_t));

	//ensure that the min and max's will be reset
	//by setting them to large absolute values
	for(int i=0;i<nfiles;i++)
	{
		xmin[i] =  1.0e32;
		xmax[i] = -1.0e32;
		ymin[i] =  1.0e32;
		ymax[i] = -1.0e32;
		zmin[i] =  1.0e32;
		zmax[i] = -1.0e32;
	}

	if(myid==0)
	{

		

		for(int i=0;i<nfiles;i++)
		{

			if(i==0)
			{
#ifdef ATHENA4
				sprintf(fname,"%s/id%d/%s.%04d.bin",fdir,i,fbase,isnap);
#else /*ATHENA4*/
				sprintf(fname,"%s/%s.%04d.bin",fdir,fbase,isnap);
#endif /*ATHENA4*/
			}else{
#ifdef ATHENA4
				sprintf(fname,"%s/id%d/%s-id%d.%04d.bin",fdir,i,fbase,i,isnap);
#else /*ATHENA4*/
				sprintf(fname,"%s/%s-id%d.%04d.bin",fdir,i,fbase,i,isnap);
#endif /*ATHENA4*/
			}


			//open the athena binary file
			//printf("fname %s\n",fname);
			//fflush(stdout);
			fp = fopen_brant(fname, "r");

			//start reading in data

#ifdef ATHENA4
			fread(&nxi,sizeof(int),1,fp);
			CoordinateSystem = nxi;
			//printf("Coordinate system = %d\n",CoordinateSystem);
#endif //ATHENA4

			fread(&nxi,sizeof(int),1,fp);
			nx_file[i] = nxi;

			fread(&nxi,sizeof(int),1,fp);
			ny_file[i] = nxi;

			fread(&nxi,sizeof(int),1,fp);
			nz_file[i] = nxi;

			//printf("file %d nx %d ny %d nz %d\n",i,nx_file[i],ny_file[i],ny_file[i]);
			//fflush(stdout);

			//printf("file : %s\t\tnx %d ny %d nz %d\n",fname,nx_file[i],ny_file[i],ny_file[i]);
			//fflush(stdout);

#ifdef LOUD
			if(i==0)
			{
				printf("file : %s\n\n",fname);
				fflush(stdout);
			}
#endif

			//if(i==0)
			//{
				fread(&nvar,sizeof(int),1,fp);
				fread(&nscalars,sizeof(int),1,fp);
				fread(&ngrav,sizeof(int),1,fp);
#ifdef ATHENA4
			fread(&flag_tracers,sizeof(int),1,fp);
#endif //ATHENA4

				//set flags

				//scalars?
				flag_scalars = 0;
				if(nscalars)
					flag_scalars = 1;

				//isothermal?
				flag_isothermal = 1;
				if( ((nvar-nscalars)==5)||((nvar-nscalars)==8) )
				{
					printf("file %s nvar %d nsca %d diff %d\n",fname,nvar,nscalars,nvar-nscalars);
					fflush(stdout);
					flag_isothermal = 0;
				}
			

				//mhd?
				flag_mhd = 0;
				if( ((nvar-nscalars)==7)||((nvar-nscalars)==8) )
					flag_mhd = 1;

				//gravity?
				flag_gravity = 0;
				if(ngrav)
					flag_gravity = 1;

				//count number of dimensions
				ndim = 3;//always 3

				fread(&fb,sizeof(float),1,fp);
				gamma_minus_1 = fb;
				fread(&fb,sizeof(float),1,fp);
				c_s_iso       = fb;

				fread(&fb,sizeof(float),1,fp);
				t    = fb;
				fread(&fb,sizeof(float),1,fp);
				dt   = fb;

#ifdef   READ_PRESSURE
				flag_pressure = 1;
#else  /*READ_PRESSURE*/
				flag_pressure = 0;
#endif /*READ_PRESSURE*/
	

				//pressure is always the first scalar
		 		ipressure = 0;
			//}

			//remember this position in the file

			fgetpos(fp,&file_position[i]);

			//x
			for(int j=0;j<nx_file[i];j++)
			{
				fread(&fb,sizeof(float),1,fp);
				if(fb<xmin[i])
					xmin[i] = fb;
				if(fb>xmax[i])
					xmax[i] = fb;
			}
			//y
			for(int j=0;j<ny_file[i];j++)
			{
				fread(&fb,sizeof(float),1,fp);
				if(fb<ymin[i])
					ymin[i] = fb;
				if(fb>ymax[i])
					ymax[i] = fb;
			}
			//z
			for(int j=0;j<nz_file[i];j++)
			{
				fread(&fb,sizeof(float),1,fp);
				if(fb<zmin[i])
					zmin[i] = fb;
				if(fb>zmax[i])
					zmax[i] = fb;
			}


			//close this file
			fclose(fp);
		}


		//determine nx, ny, and nz from the extent
		//of each file in the x,y,z-directions

		nx = nx_file[0];
		ny = ny_file[0];
		nz = nz_file[0];


		nx_start[0] = 0;
		ny_start[0] = 0;
		nz_start[0] = 0;

		xmin_cur = xmin[0];
		ymin_cur = ymin[0];
		zmin_cur = zmin[0];
		xmax_cur = xmax[0];
		ymax_cur = ymax[0];
		zmax_cur = zmax[0];

		for(int i=1;i<nfiles;i++)
		{
			nx_start[i] = 0;
			ny_start[i] = 0;
			nz_start[i] = 0;

			//if the file has a different x,y,z
			//extent than file 0, it's local nx,
			//ny, nz should be added
			//if((xmin[i]<xmin[i-1])||(xmax[i]>xmax[i-1]))
			if((xmin[i]<xmin_cur)||(xmax[i]>xmax_cur))
			{
				nx+=nx_file[i];
			
				//if(xmin[i]<xmin[i-1])
				if(xmin[i]<xmin_cur)
				{
					//nx_start[i] = nx_start[i-1] - nx_file[i];
					xmin_cur = xmin[i];
				}
				//if(xmax[i]>xmax[i-1])
				if(xmax[i]>xmax_cur)
				{
					//nx_start[i] = nx_start[i-1] + nx_file[i-1];
					xmax_cur = xmax[i];
				}
								
			}
			//if((ymin[i]<ymin[i-1])||(ymax[i]>ymax[i-1]))
			if((ymin[i]<ymin_cur)||(ymax[i]>ymax_cur))
			{
				ny+=ny_file[i];

				//if(ymin[i]<ymin[i-1])
				if(ymin[i]<ymin_cur)
				{
					//ny_start[i] = ny_start[i-1] - ny_file[i];
					ymin_cur = ymin[i];
				}
				//if(ymax[i]>ymax[i-1])
				if(ymax[i]>ymax_cur)
				{
					//ny_start[i] = ny_start[i-1] + ny_file[i-1];
					ymax_cur = ymax[i];
				}
								
			}
			//if((zmin[i]<zmin[i-1])||(zmax[i]>zmax[i-1]))
			if((zmin[i]<zmin_cur)||(zmax[i]>zmax_cur))
			{
				nz+=nz_file[i];

				//if(zmin[i]<zmin[i-1])
				if(zmin[i]<zmin_cur)
				{
					//nz_start[i] = nz_start[i-1] - nz_file[i];
					zmin_cur = zmin[i];
				}
				//if(zmax[i]>zmax[i-1])
				if(zmax[i]>zmax_cur)
				{
					//nz_start[i] = nz_start[i-1] + nz_file[i-1];
					zmax_cur = zmax[i];
				}
			}
		}

		//get start array
		for(int i=0;i<nfiles;i++)
		{
			//xtest = xmin[0];
			xtest = -1.0e8;
			ytest = -1.0e8;
			ztest = -1.0e8;

			for(int j=0;j<nfiles;j++)
			{
				if(i!=j)
				{
					if((xmin[i]>xmin[j])&&(xmin[j]>xtest))
					{
						xtest = xmin[j];
						nx_start[i] += nx_file[j];
					}
					if((ymin[i]>ymin[j])&&(ymin[j]>ytest))
					{
						ytest = ymin[j];
						ny_start[i] += ny_file[j];
					}
					if((zmin[i]>zmin[j])&&(zmin[j]>ztest))
					{
						ztest = zmin[j];
						nz_start[i] += nz_file[j];
					}
				}
			}
		}

		//shift so minimum nx_start is 0
		for(int i=0;i<nfiles;i++)
		{
			if(nx_start[i]<0)
			{
				nxi = nx_start[i];
				for(int j=0;j<nfiles;j++)
					nx_start[j] -= nxi;
			}
			if(ny_start[i]<0)
			{
				nxi = ny_start[i];
				for(int j=0;j<nfiles;j++)
					ny_start[j] -= nxi;
			}
			if(nz_start[i]<0)
			{
				nxi = nz_start[i];
				for(int j=0;j<nfiles;j++)
					nz_start[j] -= nxi;
			}
							
		}
		//these are the nx, ny, nz for the whole simulation
		/*printf("nx = %d ny = %d nz = %d\n",nx,ny,nz);
		for(int i=0;i<nfiles;i++)
		{
			printf("i %4d nx %5d nxs %5d xmin % 4e xmax % 4e ny %5d nys %5d ymin % 4e ymax % 4e nz %5d nzs %5d\n",i,nx_file[i],nx_start[i],xmin[i],xmax[i],ny_file[i],ny_start[i],ymin[i],ymax[i],nz_file[i],nz_start[i]);
		}
		fflush(stdout);*/
	}

	//broadcast information

#ifdef ATHENA4
	MPI_Bcast(&CoordinateSystem,1,MPI_INT,0,world);
#endif /*ATHENA4*/
	MPI_Bcast(&nx,1,MPI_INT,0,world);
	MPI_Bcast(&ny,1,MPI_INT,0,world);
	MPI_Bcast(&nz,1,MPI_INT,0,world);
	MPI_Bcast(&ndim,1,MPI_INT,0,world);
	MPI_Bcast(&nvar,1,MPI_INT,0,world);
	MPI_Bcast(&nscalars,1,MPI_INT,0,world);
	MPI_Bcast(&ngrav,1,MPI_INT,0,world);
#ifdef ATHENA4
	MPI_Bcast(&flag_tracers,1,MPI_INT,0,world);
#endif /*ATHENA4*/

	MPI_Bcast(&flag_gravity,1,MPI_INT,0,world);
	MPI_Bcast(&flag_scalars,1,MPI_INT,0,world);
	MPI_Bcast(&flag_mhd,1,MPI_INT,0,world);
	MPI_Bcast(&flag_isothermal,1,MPI_INT,0,world);
	MPI_Bcast(&flag_memory,1,MPI_INT,0,world);
	MPI_Bcast(&flag_pressure,1,MPI_INT,0,world);
	MPI_Bcast(&ipressure,1,MPI_INT,0,world);

	MPI_Bcast(&gamma_minus_1,1,MPI_DOUBLE,0,world);
	MPI_Bcast(&c_s_iso,1,MPI_DOUBLE,0,world);
	MPI_Bcast(&t,1,MPI_DOUBLE,0,world);
	MPI_Bcast(&dt,1,MPI_DOUBLE,0,world);

	//set grid info
	grid_info.nx = nx;
	grid_info.ny = ny;
	grid_info.nz = nz;


#ifndef NO_FFTW
	//initialize plans
        initialize_mpi_local_sizes(&grid_info, world);
#else  //NO_FFTW
        athena_initialize_mpi_local_sizes(&grid_info, myid, numprocs, world);
#endif //NO_FFTW


	//printf("About to allocate data %d flag_memory %d \n",myid,flag_memory);
	//fflush(stdout);

	//allocate memory
	
	AllocateData();	


	/*now actually start reading in data*/

	/*share information about the file extents*/
	/*with the other processors		  */
	
	MPI_Bcast(nx_file,nfiles,MPI_INT,0,world);
	MPI_Bcast(ny_file,nfiles,MPI_INT,0,world);
	MPI_Bcast(nz_file,nfiles,MPI_INT,0,world);
	MPI_Bcast(nx_start,nfiles,MPI_INT,0,world);
	MPI_Bcast(ny_start,nfiles,MPI_INT,0,world);
	MPI_Bcast(nz_start,nfiles,MPI_INT,0,world);
	

	//printf("Info bcasted %d\n",myid);
	//fflush(stdout);

	
	//read in x,y,z

	if(myid==0)
	{

		for(int i=0;i<nfiles;i++)
		{


			if(i==0)
			{
#ifdef ATHENA4
				sprintf(fname,"%s/id%d/%s.%04d.bin",fdir,i,fbase,isnap);
#else  /*ATHENA4*/
				sprintf(fname,"%s/%s.%04d.bin",fdir,i,fbase,isnap);
#endif /*ATHENA4*/
			}else{
#ifdef ATHENA4
				sprintf(fname,"%s/id%d/%s-id%d.%04d.bin",fdir,i,fbase,i,isnap);
#else  /*ATHENA4*/
				sprintf(fname,"%s/%s-id%d.%04d.bin",fdir,i,fbase,i,isnap);
#endif /*ATHENA4*/
			}

			printf("file : %5d\t%s\t\tnx_start %d nx_file %d\n",i,fname,nx_start[i],nx_file[i]);	
			fflush(stdout);


			//open the athena binary file
			fp = fopen_brant(fname, "r");
		
			//set the position to just before x,y,z arrays	
			fsetpos(fp,&file_position[i]);

			for(int j=0;j<nx_file[i];j++)
			{
				fread(&fb,sizeof(float),1,fp);
				x[nx_start[i] + j] = fb;
				//printf("x[%d] = %e\n",nx_start[i]+j,x[nx_start[i]+j]);
				//fflush(stdout);
			}

			for(int j=0;j<ny_file[i];j++)
			{
				fread(&fb,sizeof(float),1,fp);
				y[ny_start[i] + j] = fb;
			}

			for(int j=0;j<nz_file[i];j++)
			{
				fread(&fb,sizeof(float),1,fp);
				z[nz_start[i] + j] = fb;
			}


			//remember this position in the file

			fgetpos(fp,&file_position[i]);

			
			//close the file
			fclose(fp);
		}
	}
	
	//send x,y,z

	MPI_Bcast(x,nx,MPI_DOUBLE,0,world);
	MPI_Bcast(y,ny,MPI_DOUBLE,0,world);
	MPI_Bcast(z,nz,MPI_DOUBLE,0,world);


	//set box size

	grid_info.BoxSize  = (x[nx-1]-x[0]) + (x[1]-x[0]);
	grid_info.dx       = (x[1]-x[0]);
	grid_info.dy       = (y[1]-y[0]);
	grid_info.dz       = (z[1]-z[0]);
	grid_info.dV       = grid_info.dx*grid_info.dy*grid_info.dz;
	grid_info.dVk      = 1.0;


	//show information about the binary
	ShowAthenaBinary();
	

}
void AthenaBinary::ReadAthenaBinarySplit(char *fdir, char *fbase, int isnap)
{

	char  fname[200];
	FILE *fp;
	float fb;

	int   nxi;

	int ijk;

	int nzl;

	float *xmin;
	float *ymin;
	float *zmin;
	float *xmax;
	float *ymax;
	float *zmax;

	float xmin_cur;
	float ymin_cur;
	float zmin_cur;
	float xmax_cur;
	float ymax_cur;
	float zmax_cur;

	float xtest;
	float ytest;
	float ztest;

	fpos_t *file_position;

	nx = 0;
	ny = 0;
	nz = 0;


	//information for use with MPI output files
	nx_file = calloc_int_array(nfiles);
	ny_file = calloc_int_array(nfiles);
	nz_file = calloc_int_array(nfiles);
	nx_start = calloc_int_array(nfiles);
	ny_start = calloc_int_array(nfiles);
	nz_start = calloc_int_array(nfiles);

	xmin = calloc_float_array(nfiles);
	ymin = calloc_float_array(nfiles);
	zmin = calloc_float_array(nfiles);
	xmax = calloc_float_array(nfiles);
	ymax = calloc_float_array(nfiles);
	zmax = calloc_float_array(nfiles);

	file_position = (fpos_t *) malloc(nfiles*sizeof(fpos_t));

	//ensure that the min and max's will be reset
	//by setting them to large absolute values
	for(int i=0;i<nfiles;i++)
	{
		xmin[i] =  1.0e32;
		xmax[i] = -1.0e32;
		ymin[i] =  1.0e32;
		ymax[i] = -1.0e32;
		zmin[i] =  1.0e32;
		zmax[i] = -1.0e32;
	}

	if(myid==0)
	{

		

		for(int i=0;i<nfiles;i++)
		{

			if(i==0)
			{
#ifdef ATHENA4
				sprintf(fname,"%s/id%d/%s.%04d.bin",fdir,i,fbase,isnap);
#else /*ATHENA4*/
				sprintf(fname,"%s/%s.%04d.bin",fdir,fbase,isnap);
#endif /*ATHENA4*/
			}else{
#ifdef ATHENA4
				sprintf(fname,"%s/id%d/%s-id%d.%04d.bin",fdir,i,fbase,i,isnap);
#else /*ATHENA4*/
				sprintf(fname,"%s/%s-id%d.%04d.bin",fdir,i,fbase,i,isnap);
#endif /*ATHENA4*/
			}


			//open the athena binary file
			//printf("fname %s\n",fname);
			//fflush(stdout);
			fp = fopen_brant(fname, "r");

			//start reading in data

#ifdef ATHENA4
			fread(&nxi,sizeof(int),1,fp);
			CoordinateSystem = nxi;
			//printf("Coordinate system = %d\n",CoordinateSystem);
#endif //ATHENA4

			fread(&nxi,sizeof(int),1,fp);
			nx_file[i] = nxi;

			fread(&nxi,sizeof(int),1,fp);
			ny_file[i] = nxi;

			fread(&nxi,sizeof(int),1,fp);
			nz_file[i] = nxi;

			//printf("file %d nx %d ny %d nz %d\n",i,nx_file[i],ny_file[i],ny_file[i]);
			//fflush(stdout);

			//printf("file : %s\t\tnx %d ny %d nz %d\n",fname,nx_file[i],ny_file[i],ny_file[i]);
			//fflush(stdout);

#ifdef LOUD
			if(i==0)
			{
				printf("file : %s\n\n",fname);
				fflush(stdout);
			}
#endif

			//if(i==0)
			//{
				fread(&nvar,sizeof(int),1,fp);
				fread(&nscalars,sizeof(int),1,fp);
				fread(&ngrav,sizeof(int),1,fp);
#ifdef ATHENA4
			fread(&flag_tracers,sizeof(int),1,fp);
#endif //ATHENA4

				//set flags

				//scalars?
				flag_scalars = 0;
				if(nscalars)
					flag_scalars = 1;

				//isothermal?
				flag_isothermal = 1;
				if( ((nvar-nscalars)==5)||((nvar-nscalars)==8) )
				{
					printf("file %s nvar %d nsca %d diff %d\n",fname,nvar,nscalars,nvar-nscalars);
					fflush(stdout);
					flag_isothermal = 0;
				}
			

				//mhd?
				flag_mhd = 0;
				if( ((nvar-nscalars)==7)||((nvar-nscalars)==8) )
					flag_mhd = 1;

				//gravity?
				flag_gravity = 0;
				if(ngrav)
					flag_gravity = 1;

				//count number of dimensions
				ndim = 3;//always 3

				fread(&fb,sizeof(float),1,fp);
				gamma_minus_1 = fb;
				fread(&fb,sizeof(float),1,fp);
				c_s_iso       = fb;

				fread(&fb,sizeof(float),1,fp);
				t    = fb;
				fread(&fb,sizeof(float),1,fp);
				dt   = fb;

#ifdef   READ_PRESSURE
				flag_pressure = 1;
#else  /*READ_PRESSURE*/
				flag_pressure = 0;
#endif /*READ_PRESSURE*/
	

				//pressure is always the first scalar
		 		ipressure = 0;
			//}

			//remember this position in the file

			fgetpos(fp,&file_position[i]);

			//x
			for(int j=0;j<nx_file[i];j++)
			{
				fread(&fb,sizeof(float),1,fp);
				if(fb<xmin[i])
					xmin[i] = fb;
				if(fb>xmax[i])
					xmax[i] = fb;
			}
			//y
			for(int j=0;j<ny_file[i];j++)
			{
				fread(&fb,sizeof(float),1,fp);
				if(fb<ymin[i])
					ymin[i] = fb;
				if(fb>ymax[i])
					ymax[i] = fb;
			}
			//z
			for(int j=0;j<nz_file[i];j++)
			{
				fread(&fb,sizeof(float),1,fp);
				if(fb<zmin[i])
					zmin[i] = fb;
				if(fb>zmax[i])
					zmax[i] = fb;
			}


			//close this file
			fclose(fp);
		}


		//determine nx, ny, and nz from the extent
		//of each file in the x,y,z-directions

		nx = nx_file[0];
		ny = ny_file[0];
		nz = nz_file[0];


		nx_start[0] = 0;
		ny_start[0] = 0;
		nz_start[0] = 0;

		xmin_cur = xmin[0];
		ymin_cur = ymin[0];
		zmin_cur = zmin[0];
		xmax_cur = xmax[0];
		ymax_cur = ymax[0];
		zmax_cur = zmax[0];

		for(int i=1;i<nfiles;i++)
		{
			nx_start[i] = 0;
			ny_start[i] = 0;
			nz_start[i] = 0;

			//if the file has a different x,y,z
			//extent than file 0, it's local nx,
			//ny, nz should be added
			//if((xmin[i]<xmin[i-1])||(xmax[i]>xmax[i-1]))
			if((xmin[i]<xmin_cur)||(xmax[i]>xmax_cur))
			{
				nx+=nx_file[i];
			
				//if(xmin[i]<xmin[i-1])
				if(xmin[i]<xmin_cur)
				{
					//nx_start[i] = nx_start[i-1] - nx_file[i];
					xmin_cur = xmin[i];
				}
				//if(xmax[i]>xmax[i-1])
				if(xmax[i]>xmax_cur)
				{
					//nx_start[i] = nx_start[i-1] + nx_file[i-1];
					xmax_cur = xmax[i];
				}
								
			}
			//if((ymin[i]<ymin[i-1])||(ymax[i]>ymax[i-1]))
			if((ymin[i]<ymin_cur)||(ymax[i]>ymax_cur))
			{
				ny+=ny_file[i];

				//if(ymin[i]<ymin[i-1])
				if(ymin[i]<ymin_cur)
				{
					//ny_start[i] = ny_start[i-1] - ny_file[i];
					ymin_cur = ymin[i];
				}
				//if(ymax[i]>ymax[i-1])
				if(ymax[i]>ymax_cur)
				{
					//ny_start[i] = ny_start[i-1] + ny_file[i-1];
					ymax_cur = ymax[i];
				}
								
			}
			//if((zmin[i]<zmin[i-1])||(zmax[i]>zmax[i-1]))
			if((zmin[i]<zmin_cur)||(zmax[i]>zmax_cur))
			{
				nz+=nz_file[i];

				//if(zmin[i]<zmin[i-1])
				if(zmin[i]<zmin_cur)
				{
					//nz_start[i] = nz_start[i-1] - nz_file[i];
					zmin_cur = zmin[i];
				}
				//if(zmax[i]>zmax[i-1])
				if(zmax[i]>zmax_cur)
				{
					//nz_start[i] = nz_start[i-1] + nz_file[i-1];
					zmax_cur = zmax[i];
				}
			}
		}

		//get start array
		for(int i=0;i<nfiles;i++)
		{
			//xtest = xmin[0];
			xtest = -1.0e8;
			ytest = -1.0e8;
			ztest = -1.0e8;

			for(int j=0;j<nfiles;j++)
			{
				if(i!=j)
				{
					if((xmin[i]>xmin[j])&&(xmin[j]>xtest))
					{
						xtest = xmin[j];
						nx_start[i] += nx_file[j];
					}
					if((ymin[i]>ymin[j])&&(ymin[j]>ytest))
					{
						ytest = ymin[j];
						ny_start[i] += ny_file[j];
					}
					if((zmin[i]>zmin[j])&&(zmin[j]>ztest))
					{
						ztest = zmin[j];
						nz_start[i] += nz_file[j];
					}
				}
			}
		}

		//shift so minimum nx_start is 0
		for(int i=0;i<nfiles;i++)
		{
			if(nx_start[i]<0)
			{
				nxi = nx_start[i];
				for(int j=0;j<nfiles;j++)
					nx_start[j] -= nxi;
			}
			if(ny_start[i]<0)
			{
				nxi = ny_start[i];
				for(int j=0;j<nfiles;j++)
					ny_start[j] -= nxi;
			}
			if(nz_start[i]<0)
			{
				nxi = nz_start[i];
				for(int j=0;j<nfiles;j++)
					nz_start[j] -= nxi;
			}
							
		}
		//these are the nx, ny, nz for the whole simulation
		/*printf("nx = %d ny = %d nz = %d\n",nx,ny,nz);
		for(int i=0;i<nfiles;i++)
		{
			printf("i %4d nx %5d nxs %5d xmin % 4e xmax % 4e ny %5d nys %5d ymin % 4e ymax % 4e nz %5d nzs %5d\n",i,nx_file[i],nx_start[i],xmin[i],xmax[i],ny_file[i],ny_start[i],ymin[i],ymax[i],nz_file[i],nz_start[i]);
		}
		fflush(stdout);*/
	}

	//broadcast information

#ifdef ATHENA4
	MPI_Bcast(&CoordinateSystem,1,MPI_INT,0,world);
#endif /*ATHENA4*/
	MPI_Bcast(&nx,1,MPI_INT,0,world);
	MPI_Bcast(&ny,1,MPI_INT,0,world);
	MPI_Bcast(&nz,1,MPI_INT,0,world);
	MPI_Bcast(&ndim,1,MPI_INT,0,world);
	MPI_Bcast(&nvar,1,MPI_INT,0,world);
	MPI_Bcast(&nscalars,1,MPI_INT,0,world);
	MPI_Bcast(&ngrav,1,MPI_INT,0,world);
#ifdef ATHENA4
	MPI_Bcast(&flag_tracers,1,MPI_INT,0,world);
#endif /*ATHENA4*/

	MPI_Bcast(&flag_gravity,1,MPI_INT,0,world);
	MPI_Bcast(&flag_scalars,1,MPI_INT,0,world);
	MPI_Bcast(&flag_mhd,1,MPI_INT,0,world);
	MPI_Bcast(&flag_isothermal,1,MPI_INT,0,world);
	MPI_Bcast(&flag_memory,1,MPI_INT,0,world);
	MPI_Bcast(&flag_pressure,1,MPI_INT,0,world);
	MPI_Bcast(&ipressure,1,MPI_INT,0,world);

	MPI_Bcast(&gamma_minus_1,1,MPI_DOUBLE,0,world);
	MPI_Bcast(&c_s_iso,1,MPI_DOUBLE,0,world);
	MPI_Bcast(&t,1,MPI_DOUBLE,0,world);
	MPI_Bcast(&dt,1,MPI_DOUBLE,0,world);

	//set grid info
	grid_info.nx = nx;
	grid_info.ny = ny;
	grid_info.nz = nz;


#ifndef NO_FFTW
	//initialize plans
        initialize_mpi_local_sizes(&grid_info, world);
#else  //NO_FFTW
        athena_initialize_mpi_local_sizes(&grid_info, myid, numprocs, world);
#endif //NO_FFTW


	//printf("About to allocate data %d flag_memory %d \n",myid,flag_memory);
	//fflush(stdout);

	//allocate memory
	
	AllocateData();	


	/*now actually start reading in data*/

	/*share information about the file extents*/
	/*with the other processors		  */
	
	MPI_Bcast(nx_file,nfiles,MPI_INT,0,world);
	MPI_Bcast(ny_file,nfiles,MPI_INT,0,world);
	MPI_Bcast(nz_file,nfiles,MPI_INT,0,world);
	MPI_Bcast(nx_start,nfiles,MPI_INT,0,world);
	MPI_Bcast(ny_start,nfiles,MPI_INT,0,world);
	MPI_Bcast(nz_start,nfiles,MPI_INT,0,world);
	

	//printf("Info bcasted %d\n",myid);
	//fflush(stdout);

	
	//read in x,y,z

	if(myid==0)
	{

		for(int i=0;i<nfiles;i++)
		{


			if(i==0)
			{
#ifdef ATHENA4
				sprintf(fname,"%s/id%d/%s.%04d.bin",fdir,i,fbase,isnap);
#else  /*ATHENA4*/
				sprintf(fname,"%s/%s.%04d.bin",fdir,i,fbase,isnap);
#endif /*ATHENA4*/
			}else{
#ifdef ATHENA4
				sprintf(fname,"%s/id%d/%s-id%d.%04d.bin",fdir,i,fbase,i,isnap);
#else  /*ATHENA4*/
				sprintf(fname,"%s/%s-id%d.%04d.bin",fdir,i,fbase,i,isnap);
#endif /*ATHENA4*/
			}

			printf("file : %5d\t%s\t\tnx_start %d nx_file %d\n",i,fname,nx_start[i],nx_file[i]);	
			fflush(stdout);


			//open the athena binary file
			fp = fopen_brant(fname, "r");
		
			//set the position to just before x,y,z arrays	
			fsetpos(fp,&file_position[i]);

			for(int j=0;j<nx_file[i];j++)
			{
				fread(&fb,sizeof(float),1,fp);
				x[nx_start[i] + j] = fb;
				//printf("x[%d] = %e\n",nx_start[i]+j,x[nx_start[i]+j]);
				//fflush(stdout);
			}

			for(int j=0;j<ny_file[i];j++)
			{
				fread(&fb,sizeof(float),1,fp);
				y[ny_start[i] + j] = fb;
			}

			for(int j=0;j<nz_file[i];j++)
			{
				fread(&fb,sizeof(float),1,fp);
				z[nz_start[i] + j] = fb;
			}


			//remember this position in the file

			fgetpos(fp,&file_position[i]);

			
			//close the file
			fclose(fp);
		}
	}
	
	//send x,y,z

	MPI_Bcast(x,nx,MPI_DOUBLE,0,world);
	MPI_Bcast(y,ny,MPI_DOUBLE,0,world);
	MPI_Bcast(z,nz,MPI_DOUBLE,0,world);


	//set box size

	grid_info.BoxSize  = (x[nx-1]-x[0]) + (x[1]-x[0]);
	grid_info.dx       = (x[1]-x[0]);
	grid_info.dy       = (y[1]-y[0]);
	grid_info.dz       = (z[1]-z[0]);
	grid_info.dV       = grid_info.dx*grid_info.dy*grid_info.dz;
	grid_info.dVk      = 1.0;

	printf("A %d made it here\n",myid);
	MPI_Barrier(world);

	//show information about the binary
	ShowAthenaBinary();

	printf("B %d made it here\n",myid);
	MPI_Barrier(world);
	

	//read in the density

	//ReadAthenaBinaryField(fp, 0);
	ReadAthenaBinaryFieldSplit(fdir,fbase, isnap, file_position, 0);


	//MPI_Finalize();
	//exit(0);

	//read in the x-velocity

	//ReadAthenaBinaryField(fp, 1);
	ReadAthenaBinaryFieldSplit(fdir,fbase, isnap, file_position, 1);

	//read in the y-velocity

	//ReadAthenaBinaryField(fp, 2);
	ReadAthenaBinaryFieldSplit(fdir,fbase, isnap, file_position, 2);

	//read in the z-velocity

	//ReadAthenaBinaryField(fp, 3);
	ReadAthenaBinaryFieldSplit(fdir,fbase, isnap, file_position, 3);

	if(!flag_isothermal)
	{
		//read in the total energy

		//ReadAthenaBinaryField(fp, 4);
		ReadAthenaBinaryFieldSplit(fdir,fbase, isnap, file_position, 4);
	}

	if(flag_mhd)
	{
		//read in the Bx

		//ReadAthenaBinaryField(fp, 5);
		ReadAthenaBinaryFieldSplit(fdir, fbase, isnap, file_position, 5);

		//read in the By

		//ReadAthenaBinaryField(fp, 6);
		ReadAthenaBinaryFieldSplit(fdir, fbase, isnap, file_position, 6);

		//read in the Bz

		//ReadAthenaBinaryField(fp, 7);
		ReadAthenaBinaryFieldSplit(fdir, fbase, isnap, file_position, 7);

	}
	//MPI_Abort(world,-1);
	

	if(flag_scalars)
	{
		for(int i=0;i<nscalars;i++)
		{
			//read in the scalars

			//ReadAthenaBinaryField(fp, 8+i);
			ReadAthenaBinaryFieldSplit(fdir, fbase, isnap, file_position, 8+i);
		}
	}

	if(flag_gravity)
	{
		//read in the gravitational potential

		//ReadAthenaBinaryField(fp, 8);
	
		ReadAthenaBinaryFieldSplit(fdir, fbase, isnap, file_position, 8);
	}

	if(!grid_info.nz)
	{
		nzl = 1;
	}else{
		nzl = grid_info.nz;
	}
	//Convert momenta into velocities
	for(int i=0;i<grid_info.nx_local;i++)
		for(int j=0;j<grid_info.ny;j++)
			for(int k=0;k<nzl;k++)
				for(int n=0;n<grid_info.ndim;n++)
				{
					ijk = grid_ijk(i,j,k,grid_info);
					velocity[n][ijk]/=density[ijk];
				}


	free(xmin);
	free(ymin);
	free(zmin);
	free(xmax);
	free(ymax);
	free(zmax);

	free(file_position);

	if(myid==0)
	{
		printf("Done reading file.\n");
		fflush(stdout);
	}
}

void AthenaBinary::WriteAthenaBinaryLimits(char *fname, int nx_min, int nx_max, int ny_min, int ny_max, int nz_min, int nz_max)
{
	FILE *fp;

	float fb;

	int   nxi;

	int ijk;
	int nzl;

  int nx_out = nx_max-nx_min;
  int ny_out = ny_max-ny_min;
  int nz_out = nz_max-nz_min;

	if(myid==0)
	{

		printf("Writing file %s (size = %d)...\n",fname,(int) (6*sizeof(int) + 4*sizeof(float)));
		fflush(stdout);
		//open the athena binary file
		fp = fopen_brant(fname, "w");

		//start reading in data
#ifdef   ATHENA4
		fwrite(&CoordinateSystem,sizeof(int),1,fp);
#endif /*ATHENA4*/

		fwrite(&nx_out,sizeof(int),1,fp);
		fwrite(&ny_out,sizeof(int),1,fp);
		fwrite(&nz_out,sizeof(int),1,fp);
		fwrite(&nvar,sizeof(int),1,fp);
		fwrite(&nscalars,sizeof(int),1,fp);
		fwrite(&ngrav,sizeof(int),1,fp);
#ifdef   ATHENA4
		fwrite(&flag_tracers,sizeof(int),1,fp);
#endif /*ATHENA4*/


		fb = gamma_minus_1;
		fwrite(&fb,sizeof(float),1,fp);
		fb = c_s_iso;
		fwrite(&fb,sizeof(float),1,fp);

		fb = t;
		fwrite(&fb,sizeof(float),1,fp);
		fb = dt;
		fwrite(&fb,sizeof(float),1,fp);

		//write x,y,z
		for(int i=nx_min;i<nx_max;i++)
		{
			fb = x[i];
			fwrite(&fb,sizeof(float),1,fp);
		}
		for(int i=ny_min;i<ny_max;i++)
		{
			fb = y[i];
			fwrite(&fb,sizeof(float),1,fp);
		}
		for(int i=nz_min;i<nz_max;i++)
		{
			fb = z[i];
			fwrite(&fb,sizeof(float),1,fp);
		}

		//printf("This far\n");
		//fflush(stdout);
	}


	if(!grid_info.nz)
	{
		nzl = 1;
	}else{
		nzl = grid_info.nz;
	}
	//Convert velocities into momenta
	for(int i=0;i<grid_info.nx_local;i++)
		for(int j=0;j<grid_info.ny;j++)
			for(int k=0;k<nzl;k++)
				for(int n=0;n<grid_info.ndim;n++)
				{
					ijk = grid_ijk(i,j,k,grid_info);
					velocity[n][ijk]*=density[ijk];
				}


	//write out the density


	WriteAthenaBinaryFieldLimits(fp, 0, nx_min, nx_max, ny_min, ny_max, nz_min, nz_max);

	//write out the x-velocity

	WriteAthenaBinaryFieldLimits(fp, 1, nx_min, nx_max, ny_min, ny_max, nz_min, nz_max);

	//write out the y-velocity

	WriteAthenaBinaryFieldLimits(fp, 2, nx_min, nx_max, ny_min, ny_max, nz_min, nz_max);

	//write out the z-velocity

	WriteAthenaBinaryFieldLimits(fp, 3, nx_min, nx_max, ny_min, ny_max, nz_min, nz_max);

	if(!flag_isothermal)
	{
		//write out the total energy

	  WriteAthenaBinaryFieldLimits(fp, 4, nx_min, nx_max, ny_min, ny_max, nz_min, nz_max);
	}

	if(flag_mhd)
	{
		//write out the Bx

	  WriteAthenaBinaryFieldLimits(fp, 5, nx_min, nx_max, ny_min, ny_max, nz_min, nz_max);

		//write out the By

	  WriteAthenaBinaryFieldLimits(fp, 6, nx_min, nx_max, ny_min, ny_max, nz_min, nz_max);

		//write out the Bz

	  WriteAthenaBinaryFieldLimits(fp, 7, nx_min, nx_max, ny_min, ny_max, nz_min, nz_max);
	}

	if(flag_scalars)
	{
		for(int i=0;i<nscalars;i++)
		{
			//write out the scalars

		  WriteAthenaBinaryFieldLimits(fp, 8+i, nx_min, nx_max, ny_min, ny_max, nz_min, nz_max);
		}
	}

	if(flag_gravity)
	{
		//write out the gravitational potential

		WriteAthenaBinaryFieldLimits(fp, 8, nx_min, nx_max, ny_min, ny_max, nz_min, nz_max);
	}

	if(myid==0)
		fclose(fp);
}

void AthenaBinary::WriteAthenaBinary(char *fname)
{
	FILE *fp;

	float fb;

	int   nxi;

	int ijk;
	int nzl;

	if(myid==0)
	{

		printf("Writing file %s (size = %d)...\n",fname,(int) (6*sizeof(int) + 4*sizeof(float)));
		fflush(stdout);
		//open the athena binary file
		fp = fopen_brant(fname, "w");

		//start reading in data
#ifdef   ATHENA4
		fwrite(&CoordinateSystem,sizeof(int),1,fp);
#endif /*ATHENA4*/

		fwrite(&nx,sizeof(int),1,fp);
		fwrite(&ny,sizeof(int),1,fp);
		fwrite(&nz,sizeof(int),1,fp);
		fwrite(&nvar,sizeof(int),1,fp);
		fwrite(&nscalars,sizeof(int),1,fp);
		fwrite(&ngrav,sizeof(int),1,fp);
#ifdef   ATHENA4
		fwrite(&flag_tracers,sizeof(int),1,fp);
#endif /*ATHENA4*/


		fb = gamma_minus_1;
		fwrite(&fb,sizeof(float),1,fp);
		fb = c_s_iso;
		fwrite(&fb,sizeof(float),1,fp);

		fb = t;
		fwrite(&fb,sizeof(float),1,fp);
		fb = dt;
		fwrite(&fb,sizeof(float),1,fp);

		//write x,y,z
		for(int i=0;i<nx;i++)
		{
			fb = x[i];
			fwrite(&fb,sizeof(float),1,fp);
		}
		for(int i=0;i<ny;i++)
		{
			fb = y[i];
			fwrite(&fb,sizeof(float),1,fp);
		}
		for(int i=0;i<nz;i++)
		{
			fb = z[i];
			fwrite(&fb,sizeof(float),1,fp);
		}

		//printf("This far\n");
		//fflush(stdout);
	}


	if(!grid_info.nz)
	{
		nzl = 1;
	}else{
		nzl = grid_info.nz;
	}
	//Convert velocities into momenta
	for(int i=0;i<grid_info.nx_local;i++)
		for(int j=0;j<grid_info.ny;j++)
			for(int k=0;k<nzl;k++)
				for(int n=0;n<grid_info.ndim;n++)
				{
					ijk = grid_ijk(i,j,k,grid_info);
					velocity[n][ijk]*=density[ijk];
				}


	//write out the density


	WriteAthenaBinaryField(fp, 0);

	//write out the x-velocity

	WriteAthenaBinaryField(fp, 1);

	//write out the y-velocity

	WriteAthenaBinaryField(fp, 2);

	//write out the z-velocity

	WriteAthenaBinaryField(fp, 3);

	if(!flag_isothermal)
	{
		//write out the total energy

		WriteAthenaBinaryField(fp, 4);
	}

	if(flag_mhd)
	{
		//write out the Bx

		WriteAthenaBinaryField(fp, 5);

		//write out the By

		WriteAthenaBinaryField(fp, 6);

		//write out the Bz

		WriteAthenaBinaryField(fp, 7);
	}

	if(flag_scalars)
	{
		for(int i=0;i<nscalars;i++)
		{
			//write out the scalars

			WriteAthenaBinaryField(fp, 8+i);
		}
	}

	if(flag_gravity)
	{
		//write out the gravitational potential

		WriteAthenaBinaryField(fp, 8);
	}

	if(myid==0)
		fclose(fp);
}

void AthenaBinary::ReadAthenaBinaryField(FILE *fp, int nfield)
{

	int   nxi;
	int   *nx_array;
	float *data;

	fpos_t file_position;

	double *frbuf;

	float *datax;

	char variable_name[200];

	int ijk;

	int nzl = nz;

	//save current position
	if(myid==0)
	{
		fgetpos(fp,&file_position);

		//allocate buffers
		datax = calloc_float_array(nx);

		//printf("Reading field %d\n",nfield);
		//fflush(stdout);

	}

	if(grid_info.ndim==2)
		nzl = 1;

	//allocate a buffer
	sprintf(variable_name,"frbuf");
	if(myid!=0)
		frbuf = allocate_real_fftw_grid(grid_info);

	for(int ip=0;ip<numprocs;ip++)
	{

		if(myid==0)	
		{
			//printf("reading data for process %d, tlsa %d\n",ip,n_local_real_size_array[ip]);
			//fflush(stdout);

			sprintf(variable_name,"frbuf");
			frbuf = allocate_real_fftw_grid_sized(n_local_real_size_array[ip]);

			//printf("data is allocated for process %d, tlsa %d\n",ip,n_local_real_size_array[ip]);
			//fflush(stdout);
			

			//reset the file pointer position
			//to the beginning of the data field

			fsetpos(fp,&file_position);
		
			//printf("File pointer position reset.\n");
			//fflush(stdout);

			//read in the data
			for(int k=0;k<nzl;k++)
			{
				for(int j=0;j<ny;j++)
				{
					//read data array
					fread(datax,sizeof(float),nx,fp);
					for(int i=0;i<nx_local_array[ip];i++)
					{	
						//ijk = (i*ny + j)*(2*(nz/2+1)) + k;
						ijk = grid_ijk(i,j,k,grid_info);

						frbuf[ijk] = datax[nx_local_start_array[ip] + i];
					}
			
					/*if((nfield==7)&&(j==2))
					{	
						for(int i=0;i<nx;i++)
							printf("in %d %d %d %e\n",nfield,j,i,datax[i]);
						fflush(stdout);
					}*/
				}
			}



		}

		//printf("id %d is waiting\n",myid);
		//fflush(stdout);
		MPI_Barrier(world);

		//store the binary field
		StoreAthenaBinaryField(frbuf,ip,nfield);

		if(myid==0)
			free(frbuf);
	}
	if(myid!=0)
		free(frbuf);

	if(myid==0)
	{
		free(datax);

		//printf("Done reading field %d\n",nfield);
		//fflush(stdout);
	}

	MPI_Barrier(world);
}

void AthenaBinary::ReadAthenaBinaryFieldSplit(char *fdir, char *fbase, int isnap, fpos_t *file_position, int nfield)
{
	
	char fname[200];

	double *frbuf;
	double *buffer;

	float *datax;

	int ijkt;
	int ijk;
	int nt=0;

	char variable_name[200];

	FILE *fp;

	if(myid==0)
	{
		for(int ip=0;ip<numprocs;ip++)
			nt += n_local_real_size_array[ip];
		sprintf(variable_name,"buffer");
		buffer = allocate_real_fftw_grid_sized(nt);

		for(int il=0;il<nfiles;il++)
		{
			//allocate datax buffer
			datax = calloc_float_array(nx_file[il]);

			if(il==0)
			{
#ifdef ATHENA4
				sprintf(fname,"%s/id%d/%s.%04d.bin",fdir,il,fbase,isnap);
#else  /*ATHENA4*/
				sprintf(fname,"%s/%s.%04d.bin",fdir,fbase,isnap);
#endif /*ATHENA4*/
			}else{
#ifdef ATHENA4
				sprintf(fname,"%s/id%d/%s-id%d.%04d.bin",fdir,il,fbase,il,isnap);
#else  /*ATHENA4*/
				sprintf(fname,"%s/%s-id%d.%04d.bin",fdir,fbase,il,isnap);
#endif /*ATHENA4*/
			}

			//printf("file : %s (%d/%d)\n",fname,il,nfiles);	
			//fflush(stdout);


			//open the athena binary file
			fp = fopen_brant(fname, "r");

	
			//reset the file pointer position
			//to the beginning of the data field
			fsetpos(fp,&file_position[il]);

			//printf("nfield %d il %d fp %d nx_file %d ny_file %d nz_file %d nt %d\n",nfield,il,(int) file_position[il],nx_file[il],ny_file[il],nz_file[il],nt);

			//read in the data
			//for(int k=0;k<nz;k++)
			for(int k=0;k<nz_file[il];k++)
			{
				for(int j=0;j<ny_file[il];j++)
				{
					//read data array
					fread(datax,sizeof(float),nx_file[il],fp);
					for(int i=0;i<nx_file[il];i++)
					{	
						ijkt = ((nx_start[il]+i)*ny + ny_start[il]+j)*(2*(nz/2+1)) + nz_start[il]+k;

						if(ijkt>nt)
						{
							printf("Error here! ijkt %d nt %d nx_start %d i %d ny_start %d ny %d j %d nz %d k %d nz_start %d\n",ijkt,nt,nx_start[il],i,ny_start[il],ny,j,nz,k,nz_start[il]);
							fflush(stdout);
						}

						buffer[ijkt] = datax[i];
					}
				}
			}


			//remember the file pointer position
			//after this field
			fgetpos(fp,&file_position[il]);

			//free buffer data
			free(datax);

			fclose(fp);
		}

		//at this point, the large buffer "buffer"
		//has the whole data field in it.
	}

	if(myid!=0)
	{
		sprintf(variable_name,"frbuf");
		frbuf  = allocate_real_fftw_grid(grid_info);
	}
	for(int ip=0;ip<numprocs;ip++)
	{

		if(myid==0)
		{
			sprintf(variable_name,"frbuf");
			frbuf = allocate_real_fftw_grid_sized(n_local_real_size_array[ip]);

			//printf("ip %d size %d\n",ip,n_local_real_size_array[ip]);

			for(int i=0;i<nx_local_array[ip];i++)
				for(int j=0;j<ny;j++)
					for(int k=0;k<nz;k++)
					{
						ijkt = ((nx_local_start_array[ip]+i)*ny + j)*(2*(nz/2+1)) + k;
						ijk  = (i*ny + j)*(2*(nz/2+1)) + k;

						frbuf[ijk] = buffer[ijkt];
					}
		}	


		//store the data across processors
		StoreAthenaBinaryField(frbuf, ip, nfield);


		if(myid==0)
			free(frbuf);
	}

	//free buffers
	if(myid!=0)
		free(frbuf);

	if(myid==0)
		free(buffer);
		
}


void AthenaBinary::WriteAthenaBinaryField(FILE *fp, int nfield)
{

	int   nxi;
	int   *nx_array;
	float *data;

	fpos_t file_position;

	double *frbuf;
	double *buffer;

	float *datax;

	char variable_name[200];

	int ijk, ijkt;

	int nt = 0;

	//save current position
	if(myid==0)
	{
		fgetpos(fp,&file_position);

		//allocate buffers
		datax = calloc_float_array(nx);

		printf("Writing field %d\n",nfield);
		fflush(stdout);

	}

	//allocate a buffer
	if(myid!=0)
	{
		sprintf(variable_name,"frbuf");
		frbuf  = allocate_real_fftw_grid(grid_info);
	}else{
		for(int ip=0;ip<numprocs;ip++)
			nt += n_local_real_size_array[ip];
		sprintf(variable_name,"buffer");
		buffer = allocate_real_fftw_grid_sized(nt);
	}


	MPI_Barrier(world);


	for(int ip=0;ip<numprocs;ip++)
	{

		if(myid==0)
		{
			//printf("about to allocate ip %d size %d\n",ip,n_local_real_size_array[ip]);
			//fflush(stdout);

			sprintf(variable_name,"frbuf");
			frbuf = allocate_real_fftw_grid_sized(n_local_real_size_array[ip]);

			//printf("about to recover ip %d size %d\n",ip,n_local_real_size_array[ip]);
			//fflush(stdout);
		}

		//recover the binary field
		RecoverAthenaBinaryField(frbuf,ip,nfield);

		if(myid==0)
		{

			//read in the data
			for(int i=0;i<nx_local_array[ip];i++)
			{
				for(int j=0;j<ny;j++)
				{
					//read data array
					for(int k=0;k<nz;k++)
					{	
						ijkt = ((i+nx_local_start_array[ip])*ny + j)*(2*(nz/2+1)) + k;
						ijk  = (i*ny + j)*(2*(nz/2+1)) + k;

						buffer[ijkt] = frbuf[ijk];
						//buffer[ijkt] = ip;
					}
				}
			}

			free(frbuf);
		}

		/*if(myid==0)
		{
			printf("**************\n");
			fflush(stdout);
		}*/
	
		MPI_Barrier(world);
	}


	if(myid==0)
	{	
		//printf("field %d size %d\n",nfield,(int) sizeof(float)*nx*nz*ny);
		//fflush(stdout);
		for(int k=0;k<nz;k++)
		{
			for(int j=0;j<ny;j++)
			{
				//write the data array
				for(int i=0;i<nx;i++)
				{	
					ijk = (i*ny + j)*(2*(nz/2+1)) + k;

					datax[i] = buffer[ijk];
					if((nfield==7)&&(j==ny-1))
					{
						//printf("%d %d %d %e\n",nfield,j,i,datax[i]);
						//fflush(stdout);
					}
				}
				
				fwrite(datax,sizeof(float),nx,fp);
			}
		}

		free(buffer);
		free(datax);
	}

	if(myid!=0)
		free(frbuf);
	//printf("proc %d finished\n",myid);
	//fflush(stdout);
	MPI_Barrier(world);
}

void AthenaBinary::WriteAthenaBinaryFieldLimits(FILE *fp, int nfield, int nx_min, int nx_max, int ny_min, int ny_max, int nz_min, int nz_max)
{

	int   nxi;
	int   *nx_array;
	float *data;

	fpos_t file_position;

	double *frbuf;
	double *buffer;

	float *datax;

	char variable_name[200];

	int ijk, ijkt;

	int nt = 0;

	int nx_out = nx_max - nx_min;
	int ny_out = ny_max - ny_min;
	int nz_out = nz_max - nz_min;

	//save current position
	if(myid==0)
	{
		fgetpos(fp,&file_position);

		//allocate buffers
		datax = calloc_float_array(nx_out);

		printf("Writing field %d\n",nfield);
		fflush(stdout);

	}

	//allocate a buffer
	if(myid!=0)
	{
		sprintf(variable_name,"frbuf");
		frbuf  = allocate_real_fftw_grid(grid_info);
	}else{
		for(int ip=0;ip<numprocs;ip++)
			nt += n_local_real_size_array[ip];
		sprintf(variable_name,"buffer");
		buffer = allocate_real_fftw_grid_sized(nt);
	}


	MPI_Barrier(world);


	for(int ip=0;ip<numprocs;ip++)
	{

		if(myid==0)
		{
			//printf("about to allocate ip %d size %d\n",ip,n_local_real_size_array[ip]);
			//fflush(stdout);

			sprintf(variable_name,"frbuf");
			frbuf = allocate_real_fftw_grid_sized(n_local_real_size_array[ip]);

			//printf("about to recover ip %d size %d\n",ip,n_local_real_size_array[ip]);
			//fflush(stdout);
		}

		//recover the binary field
		RecoverAthenaBinaryField(frbuf,ip,nfield);

		if(myid==0)
		{

			//read in the data
			for(int i=0;i<nx_local_array[ip];i++)
			{
				for(int j=0;j<ny;j++)
				{
					//read data array
					for(int k=0;k<nz;k++)
					{	
						ijkt = ((i+nx_local_start_array[ip])*ny + j)*(2*(nz/2+1)) + k;
						ijk  = (i*ny + j)*(2*(nz/2+1)) + k;

						buffer[ijkt] = frbuf[ijk];
						//buffer[ijkt] = ip;
					}
				}
			}

			free(frbuf);
		}

		/*if(myid==0)
		{
			printf("**************\n");
			fflush(stdout);
		}*/
	
		MPI_Barrier(world);
	}


	if(myid==0)
	{	
		//printf("field %d size %d\n",nfield,(int) sizeof(float)*nx*nz*ny);
		//fflush(stdout);
		//for(int k=0;k<nz;k++)
		for(int k=nz_min;k<nz_max;k++)
		{
			for(int j=ny_min;j<ny_max;j++)
			{
				//write the data array
				for(int i=nx_min;i<nx_max;i++)
				{	
					ijk = (i*ny + j)*(2*(nz/2+1)) + k;

					datax[i - nx_min] = buffer[ijk];
					if((nfield==7)&&(j==ny-1))
					{
						//printf("%d %d %d %e\n",nfield,j,i,datax[i]);
						//fflush(stdout);
					}
				}
				
				fwrite(datax,sizeof(float),nx_out,fp);
			}
		}

		free(buffer);
		free(datax);
	}

	if(myid!=0)
		free(frbuf);
	//printf("proc %d finished\n",myid);
	//fflush(stdout);
	MPI_Barrier(world);
}
void AthenaBinary::StoreAthenaBinaryField(double *frbuf, int ip, int nfield)
{
	int ijk;

	MPI_Status status;

	double *data_select;

	/*printf("myid %d is in Store nfield = %d\n",myid,nfield);
	fflush(stdout);*/

	//select grid to keep
	switch(nfield)
	{
		case 0: data_select = density;
			break;
		case 1: data_select = velocity[0];
			break;
		case 2: data_select = velocity[1];
			break;
		case 3: data_select = velocity[2];
			break;
		case 4: data_select = etot;
			break;
		case 5: data_select = b_field[0];
			break;
		case 6: data_select = b_field[1];
			break;
		case 7: data_select = b_field[2];
			break;
		case 8: data_select = phi;
			break;
		default: data_select = scalars[nfield-8];
			break;
	}
		
	if(myid==0)
	{
		if(myid!=ip)
		{
			//send the data

			MPI_Send(frbuf,n_local_real_size_array[ip]*sizeof(double),MPI_BYTE,ip,0,world);
		}
	}else{

		if(myid==ip)
		{
			//receive the data
	
			MPI_Recv(frbuf,grid_info.n_local_real_size*sizeof(double),MPI_BYTE,0,0,world,&status);
		}
	}
	MPI_Barrier(world);
	/*if(myid==ip)
	{
		printf("myid %d data received for field %d\n",ip,nfield);
		fflush(stdout);
	}*/
	if(myid==ip)
	{
		//store the data
		for(int i=0;i<grid_info.nx_local;i++)
			for(int j=0;j<ny;j++)
				for(int k=0;k<nz;k++)
				{	
					ijk = grid_ijk(i,j,k,grid_info);
	
					data_select[ijk] = frbuf[ijk];
				}
	}

}
void AthenaBinary::RecoverAthenaBinaryField(double *frbuf, int ip, int nfield)
{
	int ijk;

	MPI_Status status;

	double *data_select;

	//printf("myid %d is in Recover nfield = %d\n",myid,nfield);
	//fflush(stdout);

	//select grid to keep
	switch(nfield)
	{
		case 0: data_select = density;
			break;
		case 1: data_select = velocity[0];
			break;
		case 2: data_select = velocity[1];
			break;
		case 3: data_select = velocity[2];
			break;
		case 4: data_select = etot;
			break;
		case 5: data_select = b_field[0];
			break;
		case 6: data_select = b_field[1];
			break;
		case 7: data_select = b_field[2];
			break;
		case 8: data_select = phi;
			break;
		default: data_select = scalars[nfield-8];
			break;
	}

	if(myid==ip)
	{
		//store the data
		for(int i=0;i<grid_info.nx_local;i++)
			for(int j=0;j<ny;j++)
				for(int k=0;k<nz;k++)
				{	
					ijk = grid_ijk(i,j,k,grid_info);
	
					frbuf[ijk] = data_select[ijk];
				}
	}


		
	if(myid==0)
	{
		if(myid!=ip)
		{
			//send the data

			MPI_Recv(frbuf,n_local_real_size_array[ip]*sizeof(double),MPI_BYTE,ip,0,world,&status);
		}
	}else{

		if(myid==ip)
		{
			//receive the data
			MPI_Send(frbuf,n_local_real_size_array[ip]*sizeof(double),MPI_BYTE,0,0,world);
	
		}
	}
	MPI_Barrier(world);
	/*if(myid==ip)
	{
		printf("myid %d data sent for field %d\n",ip,nfield);
		fflush(stdout);
	}*/
}




//////////////////////////////////////
//Only AthenaBinarySlice below here
//////////////////////////////////////
AthenaBinarySlice *AthenaBinary::ExchangeAthenaBinarySlices(int dir)
{
	int islice;
	int ijk;
	int jk;
	int source;
	int dest;
	double *xbuf;
	double *xrec;
	AthenaBinarySlice *slice;
	MPI_Status status;

	slice = (AthenaBinarySlice *) malloc(sizeof(AthenaBinarySlice));

	slice->flag_memory = 0;

	slice->ny = grid_info.ny;
	slice->nz = grid_info.nz;
	slice->nvar = nvar;
	slice->nscalars = nscalars;
	slice->ngrav = ngrav;
	slice->flag_gravity    = flag_gravity;
	slice->flag_scalars    = flag_scalars;
	slice->flag_mhd        = flag_mhd;
	slice->flag_isothermal = flag_isothermal;
	slice->flag_pressure    = flag_pressure;

	slice->AllocateData();

	if(dir==0)
	{
		//shift to the right
		islice = grid_info.nx_local-1;
		source = myid - 1;
		if(source<0)
			source = numprocs-1;
		dest = myid+1;
		if(dest>=numprocs)
			dest = 0;
	}else{
		//shift to the left
		islice = 0;
		dest = myid- 1;
		if(dest<0)
			dest= numprocs-1;
		source = myid+1;
		if(source>=numprocs)
			source= 0;
	}


	//first density
	xbuf = calloc_double_array(ny*nz);
	xrec = calloc_double_array(ny*nz);
	for(int j=0;j<slice->ny;j++)
		for(int k=0;k<slice->nz;k++)
		{
			jk  = j*slice->nz + k;
			ijk = grid_ijk(islice,j,k,grid_info);
			xbuf[jk] = density[ijk];
		}
	MPI_Sendrecv(xbuf,slice->ny*slice->nz,MPI_DOUBLE,dest,myid,xrec,slice->ny*slice->nz,MPI_DOUBLE,source,source,world,&status);	
	for(int j=0;j<slice->ny;j++)
		for(int k=0;k<slice->nz;k++)
		{
			jk  = j*slice->nz + k;
			slice->density[j][k] = xrec[jk];
		}
	free(xbuf);
	free(xrec);


	if(!flag_isothermal)
	{
		//second energy
		xbuf = calloc_double_array(ny*nz);
		xrec = calloc_double_array(ny*nz);
		for(int j=0;j<slice->ny;j++)
			for(int k=0;k<slice->nz;k++)
			{
				jk  = j*slice->nz + k;
				ijk = grid_ijk(islice,j,k,grid_info);
				xbuf[jk] = etot[ijk];
			}
		MPI_Sendrecv(xbuf,slice->ny*slice->nz,MPI_DOUBLE,dest,myid,xrec,slice->ny*slice->nz,MPI_DOUBLE,source,source,world,&status);	
		for(int j=0;j<slice->ny;j++)
			for(int k=0;k<slice->nz;k++)
			{
				jk  = j*slice->nz + k;
				slice->etot[j][k] = xrec[jk];
			}
		free(xbuf);
		free(xrec);
	}


	//third velocity 
	xbuf = calloc_double_array(3*ny*nz);
	xrec = calloc_double_array(3*ny*nz);
	for(int i=0;i<3;i++)
		for(int j=0;j<slice->ny;j++)
			for(int k=0;k<slice->nz;k++)
			{
				jk  = i*slice->ny*slice->nz + j*slice->nz + k;
				ijk = grid_ijk(islice,j,k,grid_info);
				xbuf[jk] = velocity[i][ijk];
			}
	MPI_Sendrecv(xbuf,3*slice->ny*slice->nz,MPI_DOUBLE,dest,myid,xrec,3*slice->ny*slice->nz,MPI_DOUBLE,source,source,world,&status);	
	for(int i=0;i<3;i++)
		for(int j=0;j<slice->ny;j++)
			for(int k=0;k<slice->nz;k++)
			{
				jk  = i*slice->ny*slice->nz + j*slice->nz + k;
				slice->velocity[i][j][k] = xrec[jk];
			}
	free(xbuf);
	free(xrec);


	if(flag_mhd)
	{
		//fourth bfield
		xbuf = calloc_double_array(3*ny*nz);
		xrec = calloc_double_array(3*ny*nz);
		for(int i=0;i<3;i++)
			for(int j=0;j<slice->ny;j++)
				for(int k=0;k<slice->nz;k++)
				{
					jk  = i*slice->ny*slice->nz + j*slice->nz + k;
					ijk = grid_ijk(islice,j,k,grid_info);
					xbuf[jk] = b_field[i][ijk];
				}
		MPI_Sendrecv(xbuf,3*slice->ny*slice->nz,MPI_DOUBLE,dest,myid,xrec,3*slice->ny*slice->nz,MPI_DOUBLE,source,source,world,&status);	
		for(int i=0;i<3;i++)
			for(int j=0;j<slice->ny;j++)
				for(int k=0;k<slice->nz;k++)
				{
					jk  = i*slice->ny*slice->nz + j*slice->nz + k;
					slice->b_field[i][j][k] = xrec[jk];
				}
		free(xbuf);
		free(xrec);
	}


	if(flag_scalars)
	{
		//fifth scalars
		xbuf = calloc_double_array(nscalars*ny*nz);
		xrec = calloc_double_array(nscalars*ny*nz);
		for(int i=0;i<nscalars;i++)
			for(int j=0;j<slice->ny;j++)
				for(int k=0;k<slice->nz;k++)
				{
					jk  = i*slice->ny*slice->nz + j*slice->nz + k;
					ijk = grid_ijk(islice,j,k,grid_info);
					xbuf[jk] = scalars[i][ijk];
				}
		MPI_Sendrecv(xbuf,nscalars*slice->ny*slice->nz,MPI_DOUBLE,dest,myid,xrec,nscalars*slice->ny*slice->nz,MPI_DOUBLE,source,source,world,&status);	
		for(int i=0;i<nscalars;i++)
			for(int j=0;j<slice->ny;j++)
				for(int k=0;k<slice->nz;k++)
				{
					jk  = i*slice->ny*slice->nz + j*slice->nz + k;
					slice->scalars[i][j][k] = xrec[jk];
				}
		free(xbuf);
		free(xrec);
	}

	if(flag_gravity)
	{
		//sixth phi
		xbuf = calloc_double_array(ny*nz);
		xrec = calloc_double_array(ny*nz);
		for(int j=0;j<slice->ny;j++)
			for(int k=0;k<slice->nz;k++)
			{
				jk  = j*slice->nz + k;
				ijk = grid_ijk(islice,j,k,grid_info);
				xbuf[jk] = phi[ijk];
			}
		MPI_Sendrecv(xbuf,slice->ny*slice->nz,MPI_DOUBLE,dest,myid,xrec,slice->ny*slice->nz,MPI_DOUBLE,source,source,world,&status);	
		for(int j=0;j<slice->ny;j++)
			for(int k=0;k<slice->nz;k++)
			{
				jk  = j*slice->nz + k;
				slice->phi[j][k] = xrec[jk];
			}
		free(xbuf);
		free(xrec);
	}

	return slice;
}

AthenaBinarySlice::AthenaBinarySlice(void)
{
	flag_memory = 0;
}
AthenaBinarySlice::~AthenaBinarySlice(void)
{
	DestroyData();
}
void AthenaBinarySlice::AllocateData(void)
{
	if(!flag_memory)
	{
		//data grids

		density = two_dimensional_array(ny,nz);
	
		if(!flag_isothermal)
		{	
			etot = two_dimensional_array(ny,nz);
		}

		velocity = three_dimensional_array(3,ny,nz);

		if(flag_mhd)
		{
			b_field = three_dimensional_array(3,ny,nz);
		}

		if(flag_scalars)
		{
			scalars = three_dimensional_array(nscalars,ny,nz);
		}

		if(flag_gravity)
		{
			phi = two_dimensional_array(ny,nz);
		}

		//remember that memory was allocated
		flag_memory = 1;
	}
}
void AthenaBinarySlice::DestroyData(void)
{
	if(flag_memory)
	{

		//free data
		free(density);
		if(!flag_isothermal)
			free(etot);
		deallocate_three_dimensional_array(velocity, 3, ny, nz);

		if(flag_gravity)
			free(phi);

		if(flag_scalars)
			deallocate_three_dimensional_array(scalars, nscalars, ny, nz);
		
		if(flag_mhd)
			deallocate_three_dimensional_array(b_field, 3, ny, nz);

		//remember that memory is unallocated
		flag_memory = 0;
	}
}

double AthenaBinarySlice::energy(int j, int k)
{
	double IE;
	double KE;
	int ijk;

	ijk = j*nz + k;

	if(flag_isothermal)
	{
		IE = pressure(j,k);
		//KE = 0.5*( velocity[0][ijk]*velocity[0][ijk] + velocity[1][ijk]*velocity[1][ijk] + velocity[2][ijk]*velocity[2][ijk]);
		KE = 0.5*( velocity[0][j][k]*velocity[0][j][k] + velocity[1][j][k]*velocity[1][j][k] + velocity[2][j][k]*velocity[2][j][k]);
		//KE *= density[ijk];
		KE *= density[j][k];

		return IE+KE;
	}else{
		//return total energy
		return etot[j][k];
	}
}

double AthenaBinarySlice::pressure(int j, int k)
{
	double KE, ME;
	int ijk;

	ijk = j*nz + k;

	if(flag_isothermal)
	{
		//isothermal pressure
		return c_s_iso*c_s_iso*density[j][k];
	}else{
		if(flag_pressure)
		{
			//internal energy is tracked as an advected
			//quantity
			return gamma_minus_1*scalars[ipressure][j][k];
		}
		//internal energy -- energy less kinetic energy less mag pressure
	
		//jME = 0.5*( b_field[0][ijk]*b_field[0][ijk] + b_field[1][ijk]*b_field[1][ijk] + b_field[2][ijk]*b_field[2][ijk]);
		//KE = 0.5*( velocity[0][ijk]*velocity[0][ijk] + velocity[1][ijk]*velocity[1][ijk] + velocity[2][ijk]*velocity[2][ijk]);
		//KE *= density[ijk];
		ME = 0.5*( b_field[0][j][k]*b_field[0][j][k] + b_field[1][j][k]*b_field[1][j][k] + b_field[2][j][k]*b_field[2][j][k]);
		KE = 0.5*( velocity[0][j][k]*velocity[0][j][k] + velocity[1][j][k]*velocity[1][j][k] + velocity[2][j][k]*velocity[2][j][k]);
		KE *= density[j][k];

	 	return gamma_minus_1*( etot[j][k] - KE - ME );
	}
}
void AthenaBinarySlice::CopySlice(AthenaBinarySlice *source)
{

	//copy integers
	nvar     = source->nvar;
	nscalars = source->nscalars;
	ngrav    = source->ngrav;

	//copy flags
	flag_gravity = source->flag_gravity;
	flag_scalars = source->flag_scalars;
	flag_mhd     = source->flag_mhd;
	flag_isothermal = source->flag_isothermal;
	flag_pressure = source->flag_pressure;
	ipressure = source->ipressure;
	gamma_minus_1 = source->gamma_minus_1;
	c_s_iso = source->c_s_iso;
	ny = source->ny;
	nz = source->nz;

	//if necessary
	AllocateData();

	//copy arrays
	for(int j=0;j<ny;j++)
		for(int k=0;k<nz;k++)
		{
			
			density[j][k] = source->density[j][k];
			if(!flag_isothermal)
				etot[j][k] = source->etot[j][k];
			for(int i=0;i<3;i++)	
				velocity[i][j][k] = source->velocity[i][j][k];
			if(flag_mhd)
				for(int i=0;i<3;i++)	
					b_field[i][j][k] = source->b_field[i][j][k];
			if(flag_scalars)
				for(int i=0;i<nscalars;i++)	
					scalars[i][j][k] = source->scalars[i][j][k];
			if(flag_gravity)
				phi[j][k] = source->phi[j][k];
			
		}
}



#ifdef NO_FFTW
/*! \fn int grid_ijk(int i, int j, int k, FFTW_Grid_Info grid_info)
 *  \brief Array index for fftw grid based on coordinates i,j,k. */
int grid_ijk(int i, int j, int k, FFTW_Grid_Info grid_info)
{
	int jj;	//wrap safe y index
	int kk;	//wrap safe z index

	//wrap safely in y and z
	jj = j;
	if(jj<0)
		jj += grid_info.ny;
	if(jj>(grid_info.ny-1))
		jj -= grid_info.ny;

	kk = k;
	if(kk<0)
		kk += grid_info.nz;
	if(kk>(grid_info.nz-1))
		kk -= grid_info.nz;

	//see page 61 of fftw3 manual
	return (i*grid_info.ny + jj)*(2*(grid_info.nz/2+1)) + kk;
}

/*! \fn int grid_index(double x, double y, double z, FFTW_Grid_Info grid_info)
 *  \brief Given a position, return the grid index. */
int grid_index(double x, double y, double z, FFTW_Grid_Info grid_info)
{
	int i = (int) (x/grid_info.dx) - grid_info.nx_local_start;	//integer index along x direction
	int j = (int) (y/grid_info.dy);					//integer index along y direction
	int k = (int) (z/grid_info.dz);					//integer index along z direction

	//if the position is not within this slab, then
	//return -1
	if(i < 0 || i >= grid_info.nx_local)
		return -1;

	//return the ijk of this position
	return grid_ijk(i,j,k,grid_info);	
}

/*! \fn void initialize_mpi_local_sizes(FFTW_Grid_Info *grid_info, int myid, int numprocs, MPI_Comm world);
 *  \brief Function to determine local grid sizes for parallel FFT. */
void athena_initialize_mpi_local_sizes(FFTW_Grid_Info *grid_info, int myid, int numprocs, MPI_Comm world)
{
	//unsure if this works properly
	ptrdiff_t nx_local;	
	ptrdiff_t nx_local_start;	
	ptrdiff_t n_local_complex_size;	

	//set the z-index size
	grid_info->nz_complex = grid_info->nz/2 + 1;

	if(myid==numprocs-1)
	{
		nx_local = (grid_info->nx/numprocs) +  (grid_info->nx % numprocs);
	}else{
		nx_local = (grid_info->nx/numprocs);
	}

	nx_local_start = myid*(grid_info->nx/numprocs);

	//find the local sizes for complex arrays
	//n_local_complex_size = fftw_mpi_local_size_3d(grid_info->nx, grid_info->ny, grid_info->nz, world, &nx_local, &nx_local_start);
	n_local_complex_size = nx_local * grid_info->ny * 2*(grid_info->nz/2+1);


	//remember the size
	grid_info->nx_local             = nx_local;
	grid_info->nx_local_start       = nx_local_start;
	grid_info->n_local_complex_size = n_local_complex_size;
	grid_info->n_local_real_size    = 2*n_local_complex_size;
}

/*! \fn double *allocate_real_fftw_grid_sized(int n_size)
 *  \brief Allocates a pre-sized 3-d real grid for use with fftw.*/
double *allocate_real_fftw_grid_sized(int n_size)
{
	double *data;

	//allocate data
	data = (double *) malloc(n_size*sizeof(double));

	//return data
	return data;
}

/*! \fn double *allocate_real_fftw_grid(FFTW_Grid_Info grid_info)
 *  \brief Allocates a 3-d real grid for use with fftw.*/
double *allocate_real_fftw_grid(FFTW_Grid_Info grid_info)
{
	double *data;

	//allocate data
	//data = fftw_alloc_real(grid_info.n_local_real_size);
	data = (double *) malloc(grid_info.n_local_real_size*sizeof(double));

	//return data
	return data;
}


/*! \fn double **allocate_field_fftw_grid(int nd, FFTW_Grid_Info grid_info);
 *  \brief Allocates a field[ndim][n_local_real_size] (of dimension ndim) of  3-d real grids for use with fftw.*/
double **allocate_field_fftw_grid(int nd, FFTW_Grid_Info grid_info)
{
	double **data;

	//allocate the field
	data = new double *[nd];

	//each field element is an fftw grid
	for(int i=0;i<nd;i++)
		data[i] = allocate_real_fftw_grid(grid_info);

	//return the field
	return data;
}

/*! \fn void deallocate_field_fftw_grid(double **field, int nd, FFTW_Grid_Info grid_info)
 *  \brief De-allocates a field[ndim][n_local_size] (of dimension ndim) of  3-d real grids for use with fftw.*/
void deallocate_field_fftw_grid(double **field, int nd, FFTW_Grid_Info grid_info)
{
	//free field elements
	for(int i=0;i<nd;i++)
		free(field[i]);
	//free field pointer
	delete field;
}

#endif
