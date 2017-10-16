import oss2

auth = oss2.Auth('LTAIbHBeGsbrDmqv', 'KuwKLM7eqJN4BI4bSpoxBg8y4WrVcy')
bucket = oss2.Bucket(auth, 'oss-cn-shanghai.aliyuncs.com', 'mlearn')

bucket.put_object_from_file('vocal_track/train.zip', 'train.zip')
print "train script uploaded to 'vocal_track/train.zip'"
