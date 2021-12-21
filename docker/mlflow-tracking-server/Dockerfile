FROM python:3.7-slim-buster  
ENV VIRTUAL_ENV=/opt/venv  
RUN python3 -m venv ${VIRTUAL_ENV}  
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}" 
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt  
RUN rm requirements.txt  
