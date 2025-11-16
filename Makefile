BDIR = ./bin
EDIR = ./examples
IDIR = ./include
LDIR = ./lib
ODIR = ./obj
SDIR = ./src
OUTDIR = ./output

CC = g++
CFLAGS = -I$(IDIR) -L$(LDIR) -Wall -std=c++17

LIBS = -llibfftw3-3

# list of headers (dependencies) and rule to format them as [INCLUDE_DIR]/[HEADER]
_DEPS = fftw3.h fourier.h image.h lodepng.h
DEPS = $(patsubst %,$(IDIR)/%,$(_DEPS))

# list of objects and rule to format them as [OBJECT_DIR]/[OBJECT]
_OBJ = example_blue_noise.o example_brown_noise.o example_white_noise.o fourier.o image.o lodepng.o
OBJ = $(patsubst %,$(ODIR)/%,$(_OBJ))

# rule to make each object with corresponding named cpp file and headers as dependencies
$(ODIR)/%.o : $(EDIR)/%.cpp $(SDIR)/%.cpp $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

# rule to make all objects and build the result as main.exe
main : $(OBJ)
	$(CC) -o $(BDIR)/$@ $^ $(CFLAGS) $(LIBS)

# rule to make example_blue_noise.cpp and build the result as blue_noise.exe
blue_noise: ./examples/example_blue_noise.cpp ./src/fourier.cpp ./src/image.cpp ./src/lodepng.cpp
	$(CC) -o $(BDIR)/$@ $^ $(CFLAGS) $(LIBS)

# rule to make example_brown_noise.cpp and build the result as brown_noise.exe
brown_noise: ./examples/example_brown_noise.cpp ./src/fourier.cpp ./src/image.cpp ./src/lodepng.cpp
	$(CC) -o $(BDIR)/$@ $^ $(CFLAGS) $(LIBS)

# rule to make example_white_noise.cpp and build the result as white_noise.exe
white_noise: ./examples/example_white_noise.cpp ./src/fourier.cpp ./src/image.cpp ./src/lodepng.cpp
	$(CC) -o $(BDIR)/$@ $^ $(CFLAGS) $(LIBS)

# rule to make example_perlin_noise.cpp and build the result as perlin_noise.exe
perlin_noise: ./examples/example_perlin_noise.cpp ./src/fourier.cpp ./src/image.cpp ./src/lodepng.cpp
	$(CC) -o $(BDIR)/$@ $^ $(CFLAGS) $(LIBS)

# rule to make example_fractal_noise.cpp and build the result as fractak_noise.exe
fractal_noise: ./examples/example_fractal_noise.cpp ./src/fourier.cpp ./src/image.cpp ./src/lodepng.cpp
	$(CC) -o $(BDIR)/$@ $^ $(CFLAGS) $(LIBS)

.PHONY : clean

clean:
	rm -f $(ODIR)/*.o $(OUTDIR)/*.png $(BDIR)/*.exe
