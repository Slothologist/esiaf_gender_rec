#!/usr/bin/env python

from esiaf_gender_rec.gender_rec import Gender_rec
import pyesiaf
import rospy
from esiaf_ros.msg import RecordingTimeStamps, AugmentedAudio, GenderInfo

# config
import yaml
import sys

def msg_from_string(msg, data):
    msg.deserialize(data)

nodename = 'esiaf_gender_rec'

# initialize rosnode
rospy.init_node(nodename)
pyesiaf.roscpp_init(nodename, [])

# read config
rospy.loginfo('Loading config...')
argv = sys.argv
if len(argv) < 2:
    rospy.logerr('Need path to configfile as first parameter!')
    exit('1')
path_to_config = argv[1]
data = yaml.safe_load(open(path_to_config))

rospy.loginfo('Creating emotion recognizer instance...')

wrapper = Gender_rec(data)

gender_publisher = rospy.Publisher(nodename + '/' + 'GenderInfo', GenderInfo, queue_size=10)

rospy.loginfo('Creating esiaf handler...')
handler = pyesiaf.Esiaf_Handler(nodename, pyesiaf.NodeDesignation.Gender, sys.argv)

rospy.loginfo('Setting up esiaf...')
esiaf_format = pyesiaf.EsiafAudioFormat()
esiaf_format.rate = pyesiaf.Rate.RATE_16000
esiaf_format.bitrate = pyesiaf.Bitrate.BIT_INT_16_SIGNED
esiaf_format.endian = pyesiaf.Endian.LittleEndian
esiaf_format.channels = 1

esiaf_audio_info = pyesiaf.EsiafAudioTopicInfo()
esiaf_audio_info.topic = data['esiaf_input_topic']
esiaf_audio_info.allowedFormat = esiaf_format

rospy.loginfo('adding input topic...')


def input_callback(audio, timeStamps):
    # deserialize inputs
    _recording_timestamps = RecordingTimeStamps()
    msg_from_string(_recording_timestamps, timeStamps)

    # gender rec call
    gender, probability = wrapper.recognize_gender(audio), 1.0

    # assemble output
    output = GenderInfo()
    output.duration = _recording_timestamps
    output.gender = gender
    output.probability = probability

    # publish output
    gender_publisher.publish(output)
    rospy.loginfo('Person is ' + gender)



handler.add_input_topic(esiaf_audio_info, input_callback)
rospy.loginfo('input topic added')
handler.start_esiaf()

rospy.loginfo('Gender recognizer ready!')
rospy.spin()

handler.quit_esiaf()
