#!/bin/bash

# launch the dnsmasq service
srv_status_dnsmasq=`service dnsmasq status | grep 'Active' | awk '{print $3}'`
if [ $srv_status_dnsmasq == "(running)" ]; then
    echo "The status of service dnsmasq is running and dns configuration gets ready."
elif [ $srv_status_dnsmasq == "(dead)" ]; then
    echo "The status of service dnsmasq is dead and launch the service for dns configuration."
    service dnsmasq start
fi

# launch the mosquitto service
srv_status_mosquitto=`service mosquitto status | grep 'Active' | awk '{print $3}'`
if [ $srv_status_mosquitto == "(running)" ]; then
    echo "The status of service mosquitto is running and broker gets ready."
elif [ $srv_status_mosquitto == "(dead)" ]; then
    echo "The status of service mosquitto is dead and launch the service for broker."
    service mosquitto start
fi

# launch the human keypoints detection script
if [ $# == 0 ]; then
    echo "No arguments and run the 3D scripts by default..."
    python $(dirname $0)/Pacific_HumanPose3D.py
elif [ "$1" == "3d" ] || [ "$1" == "3D" ]; then
    echo "Run the Pacific_HumanPose3D.py file and get the human 3D keypoints..."
    python $(dirname $0)/Pacific_HumanPose3D.py
else
    echo "You should run the script like this(choose one option):"
    echo "./Pacific_HumanPose3D.sh"
    echo "./Pacific_HumanPose3D.sh 3d(or 3D)"
fi
