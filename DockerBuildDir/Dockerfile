FROM ubuntu:16.04

RUN apt-get update
RUN apt-get install --no-install-recommends -y apt-utils software-properties-common curl nano unzip openssh-server
RUN apt-get install -y python3 python3-dev python-distribute python3-pip git

# main python packages
RUN pip3 install --upgrade pip
RUN pip3 install --upgrade numpy scipy matplotlib scikit-learn pandas seaborn plotly jupyter statsmodels
RUN pip3 install --upgrade nose tqdm pydot pydotplus watermark geopy joblib pillow
RUN pip3 install --upgrade memory_profiler line_profiler

# Graphviz, visualizing trees
RUN apt-get -y install graphviz 

# Jupyter configs
RUN jupyter notebook --allow-root --generate-config -y
RUN echo "c.NotebookApp.password = ''" >> ~/.jupyter/jupyter_notebook_config.py
RUN echo "c.NotebookApp.token = ''" >> ~/.jupyter/jupyter_notebook_config.py
RUN jupyter nbextension enable --py --sys-prefix widgetsnbextension

# PyTorch
RUN pip3 install https://download.pytorch.org/whl/cpu/torch-1.0.1.post2-cp35-cp35m-linux_x86_64.whl 
RUN pip3 install torchvision

COPY entry-point.sh /

# Final setup: directories, permissions, ssh login, symlinks, etc
RUN mkdir -p /home/user && \
    mkdir -p /var/run/sshd && \
    echo 'root:12345' | chpasswd && \
    sed -i 's/PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config && \
    sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd && \
    chmod a+x /entry-point.sh

WORKDIR /home/user
EXPOSE 22 4545

ENTRYPOINT ["/entry-point.sh"]
CMD ["shell"]
