ifndef CPPC
	CPPC=g++
endif

CCFLAGS=-O3 -ffast-math
# CCFLAGS=-g

LIBS = -lm -lOpenCL -fopenmp

MMUL_OBJS = matmul.o cl_util.o matrix.o
EXEC = mult

all: $(EXEC)

mult: $(MMUL_OBJS)
	$(CPPC) $(MMUL_OBJS) $(CCFLAGS) $(LIBS) -o $(EXEC)

.cpp.o:
	$(CPPC) -c $< $(CCFLAGS) $(INC) -o $@

clean:
	rm -f $(MMUL_OBJS) $(EXEC)
