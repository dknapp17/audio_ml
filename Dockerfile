# Use a Windows base image with Python
FROM mcr.microsoft.com/windows/servercore:ltsc2022

# Set the working directory in the container
WORKDIR /app

# Install Python
RUN powershell -Command `
    Invoke-WebRequest -Uri https://www.python.org/ftp/python/3.9.6/python-3.9.6-amd64.exe -OutFile python-installer.exe; `
    Start-Process python-installer.exe -ArgumentList '/quiet InstallAllUsers=1 PrependPath=1' -Wait; `
    Remove-Item -Force python-installer.exe

# Copy the requirements file into the container
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the source code into the container
COPY src/ ./src

# Define the command to run your application
CMD ["python", "src/train.py"]
