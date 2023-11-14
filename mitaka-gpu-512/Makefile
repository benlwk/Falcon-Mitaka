###########################################################

## USER SPECIFIC DIRECTORIES ##

# CUDA directory:
CUDA_ROOT_DIR=/usr/local/cuda-12.1

##########################################################

## CC COMPILER OPTIONS ##

# CC compiler options:
CC=gcc
CC_FLAGS=-Iinclude
CC_LIBS=

##########################################################

## NVCC COMPILER OPTIONS ##

# NVCC compiler options:
NVCC=/usr/local/cuda-12.1/bin/nvcc
NVCC_FLAGS=-arch sm_86
# -rdc=true
NVCC_LIBS=

# CUDA library directory:
CUDA_LIB_DIR= -L$(CUDA_ROOT_DIR)/lib64
# CUDA include directory:
CUDA_INC_DIR= -I$(CUDA_ROOT_DIR)/include
# CUDA linking libraries:
CUDA_LINK_LIBS= -lcudart

##########################################################

## Project file structure ##

# Source file directory:
SRC_DIR = src

# Object file directory:
OBJ_DIR = bin

# Include header file diretory:
INC_DIR = include

##########################################################

## Make variables ##

# Target executable name:
EXE = mitaka_gpu

# Object files:
OBJS = $(OBJ_DIR)/main.o  $(OBJ_DIR)/cuda_kernel.o $(OBJ_DIR)/fft.o $(OBJ_DIR)/shake.o $(OBJ_DIR)/samplerZ.o 
##########################################################

## Compile ##

# Link c and CUDA compiled object files to target executable:
$(EXE) : $(OBJS)
	$(NVCC) $(NVCC_FLAGS) $(OBJS) -o $@ $(CUDA_INC_DIR) $(CUDA_LIB_DIR) $(CUDA_LINK_LIBS) 

# Compile main file to object files:
$(OBJ_DIR)/%.o : %.cu
	$(NVCC) $(CC_FLAGS) -c $< -o $@

# Compile C source files to object files:
$(OBJ_DIR)/%.o : $(SRC_DIR)/%.cu 
	$(NVCC) $(CC_FLAGS) -c $< -o $@

# Compile CUDA source files to object files:
$(OBJ_DIR)/%.o : $(SRC_DIR)/%.cu $(INC_DIR)/%.cuh
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@ $(NVCC_LIBS)

# Clean objects in object directory.
clean:
	$(RM) bin/* *.o $(EXE)




