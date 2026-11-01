BDIR = ./bin
EDIR = ./examples
IDIR = ./include
ODIR = ./obj
SDIR = ./src
TDIR = ./tests
OUTDIR = ./output

CC = g++
CFLAGS = -I$(IDIR) -Wall -std=c++17 -O2

# set CPPFLAGS and LDFLAGS to point at FFTW if it is not installed in a default location
CPPFLAGS =
LDFLAGS =
LIBS = -lfftw3

# sanitizers are unavailable with MinGW; SANITIZE can be overridden on the command line
ifneq (,$(findstring mingw,$(shell $(CC) -dumpmachine)))
EXE = .exe
SANITIZE =
else
EXE =
SANITIZE = -fsanitize=address,undefined
endif

# list of headers (dependencies) and rule to format them as [INCLUDE_DIR]/[HEADER]
_DEPS = fourier.h image.h lodepng.h noise2d.h
DEPS = $(patsubst %,$(IDIR)/%,$(_DEPS)) $(EDIR)/common.h

# list of objects shared by the examples and rule to format them as [OBJECT_DIR]/[OBJECT]
_OBJ = fourier.o image.o lodepng.o
OBJ = $(patsubst %,$(ODIR)/%,$(_OBJ))

EXAMPLES = blue_noise brown_noise white_noise perlin_noise fractal_noise

# rule to build every example
all : $(EXAMPLES)

# rule for each example name to build the corresponding executable
$(EXAMPLES) : % : $(BDIR)/%$(EXE)

# rule to make each shared object with corresponding named cpp file and headers as dependencies
$(ODIR)/%.o : $(SDIR)/%.cpp $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS) $(CPPFLAGS)

# rule to make each example object with corresponding named cpp file and headers as dependencies
$(ODIR)/example_%.o : $(EDIR)/example_%.cpp $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS) $(CPPFLAGS)

# rule to link each example object against the shared objects
$(BDIR)/%$(EXE) : $(ODIR)/example_%.o $(OBJ)
	$(CC) -o $@ $^ $(LDFLAGS) $(LIBS)

# rule to build and run the tests
test :
	$(CC) -std=c++17 -Wall -Wextra -Werror -g $(SANITIZE) -I$(IDIR) $(TDIR)/*.cpp -o $(BDIR)/tests$(EXE)
	$(BDIR)/tests$(EXE)

# rule to build and run the benchmark
bench :
	$(CC) -std=c++17 -Wall -Wextra -O2 -I$(IDIR) ./bench/bench.cpp -o $(BDIR)/bench$(EXE)
	$(BDIR)/bench$(EXE)

.PHONY : all clean test bench $(EXAMPLES)
.SECONDARY :

# requires a POSIX shell (e.g. MSYS2 on Windows)
clean :
	rm -f $(ODIR)/*.o $(OUTDIR)/*.png $(BDIR)/*.exe $(patsubst %,$(BDIR)/%,$(EXAMPLES)) $(BDIR)/tests $(BDIR)/bench
