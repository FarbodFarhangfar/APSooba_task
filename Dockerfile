# Use an official Python runtime as a parent image
FROM python:3.12

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY ./app /app

# Install any needed dependencies specified in requirements.txt
COPY requirements.txt requirements.txt
RUN pip install tensorflow
RUN pip install xgboost



RUN pip install --no-cache-dir -r requirements.txt

# Expose the FastAPI default port (8000)
EXPOSE 8000

# Define environment variables
ENV PYTHONUNBUFFERED=1

# Command to run FastAPI with Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
