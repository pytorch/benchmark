from utils import create_data_lists

if __name__ == '__main__':
    create_data_lists(voc07_path='./datasets/VOCdevkit/VOC2007/',
                      voc12_path='./datasets/VOCdevkit/VOC2012/',
                      output_folder='./datasets')
