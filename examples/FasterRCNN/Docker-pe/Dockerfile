FROM awssamples/mask-rcnn-tensorflow:latest


ARG user
ARG password

RUN apt-get update && apt-get install -y openssh-server
RUN mkdir -p /var/run/sshd
RUN echo "$user:$password" | chpasswd
RUN sed -i 's/PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config

RUN echo "PermitRootLogin yes" >> /etc/ssh/sshd_config
#RUN echo "prohibit-password/PermitRootLogin yes" >> /etc/ssh/sshd_config
#RUN echo "Subsystem sftp internal-sftp" >> /etc/ssh/sshd_config

#RUN echo "PubkeyAuthentication yes" >> /etc/ssh/sshd_config

# SSH login fix. Otherwise user is kicked off after login
RUN sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd

ENV NOTVISIBLE "in users profile"
RUN echo "export VISIBLE=now" >> /etc/profile

RUN apt-get update && apt-get install -y tmux


# ready to go
ENV LC_ALL "C.UTF-8"
ENV LANG "C.UTF-8"

EXPOSE 22
EXPOSE 5000
EXPOSE 6006
EXPOSE 8888
CMD ["/usr/sbin/sshd", "-D"]


