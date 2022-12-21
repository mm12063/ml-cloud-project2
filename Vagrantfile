# -*- mode: ruby -*-
# vi: set ft=ruby :

Vagrant.configure("2") do |config|

  file_loc = "~/.gitconfig"
  if File.exists?(File.expand_path(file_loc))
    config.vm.provision "file",
        source: file_loc, destination: file_loc
  end

  file_loc = "~/.ssh"
  if File.exists?(File.expand_path(file_loc))
    config.vm.provision "file", source: file_loc, destination: file_loc
  end

  file_loc = "~/.docker/access_token"
  if File.exists?(File.expand_path(file_loc))
    config.vm.provision "file",
        source: file_loc, destination: file_loc
  end

  file_loc = "~/.bash_aliases"
  if File.exists?(File.expand_path(file_loc))
    config.vm.provision "file",
        source: file_loc, destination: file_loc
  end

#  api_key_loc = "~/.bluemix/apiKey.json"
 api_key_loc = "~/.bluemix/cloudml_apiKey.json"
 if File.exists?(File.expand_path(api_key_loc))
   config.vm.provision "file",
        source: api_key_loc, destination: "~/.bluemix/apiKey.json"
 end

  # Use Ubuntu 18.04
  config.vm.box = "ubuntu/bionic64"
  config.disksize.size = '50GB'

  # Set up network port forwarding
  config.vm.network "forwarded_port", guest: 5000, host: 5000
  config.vm.network "forwarded_port", guest: 90, host: 9000
  config.vm.network "forwarded_port", guest: 8080, host: 8080
  config.vm.network "private_network", ip: "192.168.33.10"
  config.vm.hostname = "ibmcloudml"

  # Keep the VM as lean as possible
  config.vm.provider "virtualbox" do |vb|
    # Customize the amount of memory on the VM:
    vb.memory = "12288"
    vb.cpus = 4

    # Fixes DNS issues on some networks
    vb.customize ["modifyvm", :id, "--natdnshostresolver1", "on"]
    vb.customize ["modifyvm", :id, "--natdnsproxy1", "on"]
  end

  # Open the vm via vagrant ssh in /vagrant/
  config.ssh.extra_args = ["-t", "cd /vagrant; bash --login"]

#######################################################################
# Make sure that git and other dev utilities are available
######################################################################
  config.vm.provision "shell", inline: <<-SHELL
    apt-get update
    sudo dpkg --configure -a
    apt-get install -y git curl wget zip tree
    # Set up a Python 3 environment
    apt-get install -y python3-dev python3-pip python3-venv python3-flask
    python3 -m venv venv
    apt-get -y autoremove
    echo "Y" | apt install jq
  SHELL

######################################################################
# Add Minikube and Kubectl
######################################################################
  config.vm.provision "shell", inline: <<-SHELL
    echo "\n************************************"
    echo " Installing Minikube..."
    echo "************************************\n"
    sudo apt-get update -y
    sudo apt-get upgrade -y
    sudo apt-get install curl
    sudo apt-get install apt-transport-https
    echo "Y" | sudo apt install virtualbox
    sudo wget https://download.virtualbox.org/virtualbox/5.2.42/Oracle_VM_VirtualBox_Extension_Pack-5.2.42.vbox-extpack -P /usr/share/virtualbox-ext-pack
    echo "Y" | sudo VBoxManage extpack install --accept-license=56be48f923303c8cababb0bb4c478284b688ed23f16d775d729b89a2e8e5f9eb --replace /usr/share/virtualbox-ext-pack/Oracle_VM_VirtualBox_Extension_Pack-5.2.42.vbox-extpack
    wget https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64
    sudo mv minikube-linux-amd64 /usr/local/bin/minikube
    sudo chmod 755 /usr/local/bin/minikube
    echo "\n************************************"
    echo " Installing Kubectl..."
    echo "************************************\n"
    curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
    chmod +x kubectl
    sudo mv kubectl /usr/local/bin/kubectl
  SHELL


######################################################################
# Add docker image before IBM Cloud
######################################################################
  config.vm.provision "docker" do |d|
    d.pull_images "alpine:latest"
  end

######################################################################
# Setup a IBM Cloud CLI
######################################################################
  config.vm.provision "shell", inline: <<-SHELL
    echo "**********************************************************************"
    echo "Installing IBM Cloud CLI..."
    echo "**********************************************************************"
    curl -fsSL https://clis.cloud.ibm.com/install/linux | sh
    echo "source /usr/local/ibmcloud/autocomplete/bash_autocomplete" >> $HOME/.bashrc
    echo "Done!"
  SHELL


######################################################################
# Install Kind
######################################################################
  config.vm.provision "shell", inline: <<-SHELL
    echo "**********************************************************************"
    echo "Installing Kind..."
    echo "**********************************************************************"
    curl -Lo ./kind https://kind.sigs.k8s.io/dl/v0.17.0/kind-linux-amd64
    chmod +x ./kind
    sudo mv ./kind /usr/local/bin/kind
    echo "Done!"
  SHELL

######################################################################
# Install Kustomize
######################################################################
  config.vm.provision "shell", inline: <<-SHELL
    echo "**********************************************************************"
    echo "Installing Kustomize..."
    echo "**********************************************************************"
    curl -LO https://github.com/kubernetes-sigs/kustomize/releases/download/v3.2.0/kustomize_3.2.0_linux_amd64
    chmod +x kustomize_3.2.0_linux_amd64
    sudo mv kustomize_3.2.0_linux_amd64 /usr/local/bin/kustomize
    echo "Done!"
  SHELL

######################################################################
# Setup a IBM Cloud plugins
######################################################################
  config.vm.provision "shell", inline: <<-SHELL
    echo "**********************************************************************"
    echo "Installing IBM Cloud Plugins..."
    echo "**********************************************************************"
    ibmcloud plugin install container-service -r 'IBM Cloud'
    ibmcloud plugin install container-registry -r 'IBM Cloud'
    ibmcloud plugin install vpc-infrastructure -r 'IBM Cloud'
    cp -r /root/.bluemix/plugins/ /home/vagrant/.bluemix/plugins/
    sudo chown -R vagrant /home/vagrant/.bluemix/
    echo "Done!"
  SHELL

end
