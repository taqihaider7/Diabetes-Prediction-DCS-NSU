# Use a lightweight official Python image as the base
FROM continuumio/miniconda3:latest
#FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file and install dependencies
#COPY requirements.txt .
#RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code and the serialized model/scaler files
COPY app.py .
COPY best_diabetes_model.pkl .
COPY scaler.pkl .

# Expose the port the Flask app runs on
EXPOSE 5000

# Define the command to run the application
CMD ["python", "app.py"]