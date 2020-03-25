CR=$(conda --version | grep -Po 'conda \d.\d');
echo $CR
if ! [ "$CR" ]; then
    echo "Conda is not installed"
    read -p "Install conda (Y/n)? " CONT
    if [ "$CONT" = "y" ] || ! [ "$CONT" ]; then
        
        CR=$(curl --version | grep -Po 'curl \d.\d');
        if ! [ "$CR" ]; then
            echo "Curl is required and is not installed"
            read -p "Install curl (Y/n)? " CONT
            if [ "$CONT" = "y" ] || ! [ "$CONT" ]; then
                apt update -y
                apt install curl -y
                echo "Curl is installed, if any problem occurs, try running the script again"
            else
                echo "Run the script again in another terminal after curl or conda is installed"
                exit 0
            fi        
        fi
        cd /tmp
        curl -O https://repo.anaconda.com/archive/Anaconda3-2019.03-Linux-x86_64.sh
        bash Anaconda3-2019.03-Linux-x86_64.sh
        source ~/.bashrc
	conda config --set auto_activate_base false
        echo "Check if anaconda is currently intalled"
        
    else
        echo "Operation canceled"
    fi
    echo "Run the script again in another terminal after conda is installed"
    exit 0
else
    echo "Conda is already installed"
fi

VR=$(nvidia-smi | grep -Po '(?<=Version: )\d\d\d');
if ! [ "$VR" ]; then 
    echo "You should install the correct nvidia drivers, version >=410.x for version 1.12"
    echo "If you have already installed nvidia drivers and still see this message please reboot and try again"
    echo "This script will add the official nvidia repository and install drivers"
    read -p "Continue (Y/n)? " CONT
    if [ "$CONT" = "y" ] || ! [ "$CONT" ]; then
        apt purge nvidia*
        add-apt-repository ppa:graphics-drivers/ppa
        apt update -y
        VR=$(ubuntu-drivers devices | grep "nvidia-driver" | grep 'recommended' | grep -Eo '[0-9]{3,3}')
        apt install nvidia-driver-$VR -y
        apt install nvidia-cuda-toolkit -y
        echo "Now reboot and execute the script again"
        read -p "Reboot (y/n)? " CONT
        if [ "$CONT" = "y" ]; then
            reboot
        else
            echo "Reboot canceled, you need to reboot in order for the drivers have effect."
        fi
    else
        echo "Operation canceled";
    fi
    
elif [ "$VR" -gt "409" ]; then 
    echo "Installing version 1.12 for nvidia driver version $VR"
    conda create --name tf_gpu tensorflow-gpu==1.12
    echo "Try it in a python script"
else  
    echo "Its recommended to update your drivers version, if you want to upgrade your drivers and tensorflow-gpu version, run 'sudo apt purge nvidia*' and rerun the script"
    echo "Installing version 1.9 for nvidia driver version $VR"
    conda create --name tf_gpu tensorflow-gpu==1.9
    echo "Try it in a python script"
fi
