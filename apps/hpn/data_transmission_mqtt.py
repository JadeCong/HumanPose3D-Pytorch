import sys, os
import paho.mqtt.client as mqtt


# get the current host ip address
ips = os.popen("/sbin/ifconfig | grep 'inet addr' | awk '{print $2}'").read()
ip = ips[ips.find(':')+1 : ips.find('\n')]

# create mqtt client
mqtt_client = mqtt.Client()
# mqtt_client.connect(ip, 1883, 60) # or set the remote server ip address
# mqtt_client.connect("PacificFuture-Broker", 1883, 60) # domain name of the broker
# mqtt_client.connect("127.0.0.1", 1883, 60)
mqtt_client.connect("192.168.20.62", 1883, 60)
mqtt_client.loop_start()

# publish the topic to remote broker
data = "data need to be published"
mqtt_client.publish("/pacific/avatar/human_keypoints_3d", bytes(repr(json.dumps(data ,sort_keys=False)).encode('utf-8')))
