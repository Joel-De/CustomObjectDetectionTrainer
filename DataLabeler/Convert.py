import xml.etree.ElementTree as ET
import os
import json




JSONDIR = 'Json'
XMLDIR = 'Labels'


if not os.path.exists(JSONDIR):
    os.mkdir(JSONDIR)



xmlList = os.listdir(XMLDIR)

for file in xmlList:
    Labels = ET.parse((os.path.join(XMLDIR,file)))
    root = Labels.getroot()

    JsonFile = {}
    FileName = root.find('filename').text
    for item in root.iter('object'):




        Name = item.find('name').text
        BoundingBox = item.find('bndbox')

        if Name not in JsonFile.keys():
            JsonFile[Name] = []


        BoundingBox = [int(value.text) for value in BoundingBox]
        JsonFile[Name].append(BoundingBox)


        with open(os.path.join(JSONDIR, FileName + '.json'), 'w') as JsonWriter:

            json.dump( JsonFile,JsonWriter)






