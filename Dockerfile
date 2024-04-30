FROM ubuntu:22.04

# Install any additional packages that your application requires
RUN apt-get update && apt-get install -y software-properties-common
RUN apt install nano -y
RUN apt install pip -y
#RUN apt install openjdk-18-jdk -y
#RUN apt install maven -y
RUN apt install curl wget -y

# Copy your application code into the image
COPY SOGA /root/SOGA

# Set the working directory
WORKDIR /root/SOGA
RUN pip install -r  requirements.txt

# Set the command to run when the container is started
#CMD ["<command>"]