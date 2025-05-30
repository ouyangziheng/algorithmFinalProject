CC = g++
CFLAGS = -std=c++17 -Wall -Wextra -O2
LDFLAGS = -lz -pthread

SRCS = max_similarity_frames.cpp cnpy.cpp
OBJS = $(SRCS:.cpp=.o)
TARGET = max_similarity_frames

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.cpp
	$(CC) $(CFLAGS) -c $< -o $@

cnpy.o: cnpy.cpp cnpy.h
	$(CC) $(CFLAGS) -c $< -o $@

max_similarity_frames.o: max_similarity_frames.cpp cnpy.h
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET) 