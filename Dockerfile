# Stage 1: Build figures
FROM FROM python:3.12-slim AS build-artifacts
# Copy your repository
WORKDIR /app
COPY . .
  
# Stage 1: Build dependencies
FROM texlive/texlive:latest AS build-paper

# Install additional dependencies
RUN apt-get update && apt-get install -y \
    make \
    python3.12 \
    python3-pip \
    && apt-get clean

# Copy your repository
WORKDIR /app
COPY . .

# Install Python dependencies
RUN pip3 install -r requirements.txt

# Compile LaTeX documents to ensure dependencies are met
RUN make pdf
