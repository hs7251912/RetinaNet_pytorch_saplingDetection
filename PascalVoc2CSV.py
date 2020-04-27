import csv
import os
import glob
import sys

class PascalVOC2CSV(object):
    def __init__(self,xml=[],ann_path='./test_annotations.csv',classes_path='./classes.csv'):
        '''
        :param xml: 所有Pascal VOC的xml文件路径组成的列表
        :param ann_path: ann_path
        :param classes_path: classes_path
        '''

        self.xml = xml
        self.ann_path = ann_path
        self.classes_path = classes_path
        self.label = []
        self.annotations = []

        self.data_transfer()
        self.write_file()

    def data_transfer(self):
        for num, xml_file in enumerate(self.xml):
            try:
                sys.stdout.write('\r>> Converting image %d/%d' % (
                    num + 1, len(self.xml)))
                sys.stdout.flush()

                with open(xml_file,'r') as fp:
                    for p in fp:
                        # <filename>IMG_1104_4.jpeg</filename>
                        if '<filename>' in p:
                            self.file_name = p.split('>')[1].split('<')[0]

                        if '<object>' in p:
                            # 类别
                            d = [next(fp).split('>')[1].split('<')[0] for _ in range(9)]
                            self.supercategory = d[0]
                            if self.supercategory not in self.label:
                                self.label.append(self.supercategory)

                            # 边界框
                            x1 = int(d[-4])
                            y1 = int(d[-3])
                            x2 = int(d[-2])
                            y2 = int(d[-1])

                            self.annotations.append([os.path.join('./data/selected_cropped_imgs',self.file_name),x1,y1,x2,y2,self.supercategory])
            except:
                continue

        sys.stdout.write('\n')
        sys.stdout.flush()

    def write_file(self):
        with open(self.ann_path,'w',newline='') as fp:
            csv_writer = csv.writer(fp,dialect='excel')
            csv_writer.writerows(self.annotations)

        class_name = sorted(self.label)
        class_ = []
        for num,name in enumerate(class_name):
            class_.append([name,num])
        with open(self.classes_path,'w',newline='') as fp:
            csv_writer = csv.writer(fp,dialect='excel')
            csv_writer.writerows(class_)


xml_file = glob.glob('./data/selected_cropped_imgs/*.xml')

PascalVOC2CSV(xml_file)