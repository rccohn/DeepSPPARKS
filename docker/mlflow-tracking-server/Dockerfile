FROM python:3.7-slim-buster  

RUN apt-get update && apt-get -y --no-install-recommends install \
		ca-certificates \
		git \
		openssh-client \
		wget \ 
		build-essential && \
	apt-get clean && \
	apt-get autoremove && \
	rm -rf /var/lib/apt/lists/*

USER root
RUN mkdir -p /root/.ssh
RUN chown root -R /root/.ssh # not sure if this applies to mounted ssh files

ENV VIRTUAL_ENV=/opt/venv  
RUN python3 -m venv ${VIRTUAL_ENV}  
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}" 
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt  
#RUN rm requirements.txt  
