EXEC   = athena4_binary.projection

OPTIMIZE =  -O2  -DATHENA4


OBJS   = main.o constants.o routines.o grid_fft.o grid_operations.o read_athena_binary.o

CC     = mpicxx

INCL   = constants.h routines.h grid_fft.h grid_operations.h read_athena_binary.h

LIBS   = -lgsl -lgslcblas -lfftw3_mpi -lfftw3 -lm -lmpi -L/home/brant/code/fftw/lib -stdlib=libstdc++ 
#-L/pfs/sw/parallel/impi_intel/fftw-3.3.4/lib/ #

CFLAGS = $(OPTIMIZE) -I/home/brant/code/fftw/include -stdlib=libstdc++
#-I/home/brant/code/gsl/include 

$(EXEC): $(OBJS) 
	 $(CC) $(OBJS) $(LIBS) -o $(EXEC)   

$(OBJS): $(INCL) 

.PHONY : clean

clean:
	 rm -f $(OBJS) $(EXEC)

