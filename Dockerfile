FROM tensorflow/tensorflow:latest-gpu

WORKDIR /usr/src/app

# Install required lib
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt


# Copy the current directory contents into the container at /usr/src/app
COPY . .


EXPOSE 8000
CMD ["python", "main.py"]