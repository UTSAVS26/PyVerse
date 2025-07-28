#!/bin/bash
# insuring script executed as root
[[ "$UID" -ne 0 ]] && { 
    echo "Script must run as root. use sudo "
    exit 1
}

package_installer() {
    local distro
    distro=$(awk -F= '/^NAME/{print $2}' /etc/os-release)
    distro=${distro//\"/}
    
    case "$distro" in
        *"Ubuntu"* | *"Debian"*)
            apt-get update
            apt-get install -y curl tor
            ;;
        *"Fedora"* | *"CentOS"* | *"Red Hat"* | *"Amazon Linux"*)
            yum -y update
            yum -y install curl tor
            ;;
        *"Arch"*)
            pacman -Sy --noconfirm curl tor
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
    package_installer
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
    read -rp $'\033[34m  Enter time interval in seconds (0 for random 5–30 sec): \033[0m' interval
    read -rp $'\033[34m Enter number of IP changes (0 for infinite): \033[0m' times

    echo ""
    echo "✅ Starting with settings: "
    [[ "$interval" -eq 0 ]] && echo "• Random interval between 5–30 seconds" || echo "• Interval: $interval seconds"
    [[ "$times" -eq 0 ]] && echo "• Infinite IP changes" || echo "• Number of changes: $times"
    echo ""

    if [ "$times" -eq 0 ]; then
        # Infinite changes
        while true; do
            change_ip
            sleep_interval=$interval
            if [ "$interval" -eq 0 ]; then
                sleep_interval=$(shuf -i 5-30 -n 1)
            fi
            sleep "$sleep_interval"
        done
    else
        # Finite changes
        for ((i = 1; i <= times; i++)); do
            change_ip
            if [ "$i" -lt "$times" ]; then
                sleep_interval=$interval
                if [ "$interval" -eq 0 ]; then
                    sleep_interval=$(shuf -i 5-30 -n 1)
                fi
                sleep "$sleep_interval"
            fi
        done
        echo -e "\n✅ Completed $times IP change(s)."
        echo -e "🌐 Final IP: \033[32m$(get_ip)\033[0m"
        echo "👋 Exiting script."
        exit 0
    fi
done