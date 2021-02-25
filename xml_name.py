#-*- coding:utf-8 –*-
import os
import os.path 
import sys
from xml.etree.ElementTree import parse, Element

pathsrc="F:\\论文\\IP102\\Detection\\VOC2007\\Annotations"
pathdet="F:\\论文\\IP102\\Detection\\VOC2007\\Annotationsnew"


		
def changeName(pathsrc, pathdet,origin_name, new_name):
    '''
    xml_fold: xml存放文件夹
    origin_name: 原始名字，比如弄错的名字，原先要cow,不小心打成cwo
    new_name: 需要改成的正确的名字，在上个例子中就是cow
    '''
    files = os.listdir(pathsrc)
    cnt = 0 
    for xmlFile in files:
        if xmlFile[0:5]=="IP093":
	        file_path = os.path.join(pathsrc, xmlFile)
	        dom = parse(file_path)
	        root = dom.getroot()
	        for obj in root.iter('object'):#获取object节点中的name子节点
	            tmp_name = obj.find('name').text
	            print("tmp_name",tmp_name)
	            if tmp_name == origin_name: # 修改
	                obj.find('name').text = new_name
	                print(xmlFile,obj)
	                print("change %s to %s." % (origin_name, new_name))
	                cnt += 1
	        dom.write(os.path.join(pathdet,xmlFile),encoding="utf-8", xml_declaration=True)#保存到指定文件
    print("有%d个文件被成功修改。" % cnt)

changeName(pathsrc,pathdet,"92","茶黄蓟马")