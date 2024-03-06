CC = gcc
CFLAGS = -Wall -Wextra

# List of source files
SRCS = main.c

# List of object files (generated from source files)
OBJS = $(SRCS:.c=.o)

# Name of the final executable
TARGET = main

# Rule to compile each source file into its corresponding object file
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# Rule to link all object files together to create the final executable
$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) $(OBJS) -o $(TARGET)

# Rule to clean up generated files
clean:
	rm -f $(OBJS) $(TARGET)
