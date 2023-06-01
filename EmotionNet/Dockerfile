FROM pytorch/pytorch
RUN apt -y update

COPY . /workspace

WORKDIR /workspace
RUN     pip install -r requirements.txt

RUN export PS1="[\d \t] \[\e]0;\u@\h:　\w\a\]${debian_chroot:+($debian_chroot)}\[\033[01;32m\]\u@\h\[\033[00m\]:\[\033[01;34m\]\w\[\033[00m\]\$"
