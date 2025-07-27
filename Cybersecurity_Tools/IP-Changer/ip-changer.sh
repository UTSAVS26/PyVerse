#!/bin/bash
# insuring script executed as root
[[ "$UID" -ne 0 ]] && { 
    echo "Script must run as root. use sudo "
    exit 1
}

install_pacakage_according_to_OS() {
    local distro
    distro=$(awk -F= '/^NAME/{print $2}' /etc/os-release)
    distro=${distro//\"/}
    
    case "$distro" in
        *"Ubuntu"* | *"Debian"*)
            apt-get update
            apt-get install -y curl tor
            ;;
        *"Fedora"* | *"CentOS"* | *"Red Hat"* | *"Amazon Linux"*)
            yum update
            yum install -y curl tor
            ;;
        *"Arch"*)
            pacman -S --noconfirm curl tor
            ;;
        *) 
        # give warning if not in above distro to install torand curl manually
            echo "Unsupported distribution: $distro. Please install curl and tor manually."
            exit 1
            ;;
    esac
}

if ! command -v curl &> /dev/null || ! command -v tor &> /dev/null; then
    echo "Installing curl and tor"
    install_pacakage_according_to_OS
fi

if ! systemctl --quiet is-active tor.service; then
    echo "Starting tor service"
    systemctl start tor.service
fi
 # this gets current Ip
get_ip() {
    local url get_ip ip
    url="https://checkip.amazonaws.com"
    get_ip=$(curl -s -x socks5h://127.0.0.1:9050 "$url")
    ip=$(echo "$get_ip" | grep -oP '\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}')
    echo "$ip"
}
# this function changes ip by reloading tor services
change_ip() {
    echo "Reloading tor service"
    systemctl reload tor.service
    sleep 5
    echo -e "\033[34mNew IP address: $(get_ip)\033[0m"
}

clear
cat << EOF
    ___ ____        ____ _   _    _    _   _  ____ _____ ____  
   |_ _|  _ \      / ___| | | |  / \  | \ | |/ ___| ____|  _ \ 
    | || |_) |____| |   | |_| | / _ \ |  \| | |  _|  _| | |_) |
    | ||  __/_____| |___|  _  |/ ___ \| |\  | |_| | |___|  _ < 
   |___|_|         \____|_| |_/_/   \_\_| \_|\____|_____|_| \_\
EOF
# main modifcation done below
while true; do
    read -rp $'\033[34mEnter time interval in seconds (type 0 for irandom interval): \033[0m' interval
    read -rp $'\033[34mEnter number of times to change IP address (type 0 for infinite IP changes): \033[0m' times
    # now if user give 0 for interval a random interval assigned
    # and if 0 for how many changes then infinite changes
    if [ "$interval" -eq "0" ] || [ "$times" -eq "0" ]; then
        echo "Starting infinite IP changes"
        while true; do
            change_ip
            interval=$(shuf -i 10-20 -n 1)
            sleep "$interval"
        done
    else
        for ((i=0; i< times; i++)); do
            change_ip
            sleep "$interval"
        done
    fi
done