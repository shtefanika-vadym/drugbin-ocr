FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install the required packages
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 9876

# Define environment variable
ENV FLASK_APP=app.py

# Run the application using Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:9876", "app:app"]
