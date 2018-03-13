
from skimage.io import imread

files_1= ['/opt/ml_data/cifar/cifar-10/images/train/im_3_4456_8.jpg',
 '/opt/ml_data/cifar/cifar-10/images/train/im_2_7935_8.jpg',
 '/opt/ml_data/cifar/cifar-10/images/train/im_4_937_8.jpg',
 '/opt/ml_data/cifar/cifar-10/images/train/im_4_5364_1.jpg',
 '/opt/ml_data/cifar/cifar-10/images/train/im_3_9264_1.jpg',
 '/opt/ml_data/cifar/cifar-10/images/train/im_3_1040_2.jpg',
 '/opt/ml_data/cifar/cifar-10/images/train/im_3_7782_7.jpg',
 '/opt/ml_data/cifar/cifar-10/images/train/im_1_3962_0.jpg',
 '/opt/ml_data/cifar/cifar-10/images/train/im_4_226_4.jpg',
 '/opt/ml_data/cifar/cifar-10/images/train/im_4_7824_9.jpg',
 '/opt/ml_data/cifar/cifar-10/images/train/im_3_6762_2.jpg',
 '/opt/ml_data/cifar/cifar-10/images/train/im_4_7824_9.jpg',
 '/opt/ml_data/cifar/cifar-10/images/train/im_1_1449_3.jpg',
 '/opt/ml_data/cifar/cifar-10/images/train/im_1_2671_9.jpg',
 '/opt/ml_data/cifar/cifar-10/images/train/im_4_6866_9.jpg',
 '/opt/ml_data/cifar/cifar-10/images/train/im_1_5327_0.jpg',
 '/opt/ml_data/cifar/cifar-10/images/train/im_4_937_8.jpg',
 '/opt/ml_data/cifar/cifar-10/images/train/im_5_111_4.jpg',
 '/opt/ml_data/cifar/cifar-10/images/train/im_1_6419_5.jpg',
 '/opt/ml_data/cifar/cifar-10/images/train/im_1_5714_1.jpg',
 '/opt/ml_data/cifar/cifar-10/images/train/im_2_6532_8.jpg',
 '/opt/ml_data/cifar/cifar-10/images/train/im_4_7824_9.jpg',
 '/opt/ml_data/cifar/cifar-10/images/train/im_2_4839_2.jpg',
 '/opt/ml_data/cifar/cifar-10/images/train/im_1_4612_9.jpg',
 '/opt/ml_data/cifar/cifar-10/images/train/im_3_4350_8.jpg',
 '/opt/ml_data/cifar/cifar-10/images/train/im_1_8701_4.jpg',
 '/opt/ml_data/cifar/cifar-10/images/train/im_5_1972_2.jpg',
 '/opt/ml_data/cifar/cifar-10/images/train/im_3_2552_1.jpg',
 '/opt/ml_data/cifar/cifar-10/images/train/im_2_7001_7.jpg',
 '/opt/ml_data/cifar/cifar-10/images/train/im_4_6039_1.jpg',
 '/opt/ml_data/cifar/cifar-10/images/train/im_2_2051_5.jpg',
 '/opt/ml_data/cifar/cifar-10/images/train/im_4_8935_1.jpg']

files_2 = ['/opt/ml_data/cifar/cifar-10/images/train/im_1_6419_5.jpg',
 '/opt/ml_data/cifar/cifar-10/images/train/im_1_744_7.jpg',
 '/opt/ml_data/cifar/cifar-10/images/train/im_4_1959_1.jpg',
 '/opt/ml_data/cifar/cifar-10/images/train/im_2_9079_1.jpg',
 '/opt/ml_data/cifar/cifar-10/images/train/im_1_530_9.jpg',
 '/opt/ml_data/cifar/cifar-10/images/train/im_4_5488_2.jpg',
 '/opt/ml_data/cifar/cifar-10/images/train/im_5_2141_3.jpg',
 '/opt/ml_data/cifar/cifar-10/images/train/im_5_182_7.jpg',
 '/opt/ml_data/cifar/cifar-10/images/train/im_4_7142_7.jpg',
 '/opt/ml_data/cifar/cifar-10/images/train/im_1_5818_2.jpg',
 '/opt/ml_data/cifar/cifar-10/images/train/im_1_5204_2.jpg',
 '/opt/ml_data/cifar/cifar-10/images/train/im_1_9078_2.jpg',
 '/opt/ml_data/cifar/cifar-10/images/train/im_4_4721_1.jpg',
 '/opt/ml_data/cifar/cifar-10/images/train/im_3_9474_5.jpg',
 '/opt/ml_data/cifar/cifar-10/images/train/im_5_6999_9.jpg',
 '/opt/ml_data/cifar/cifar-10/images/train/im_1_5580_0.jpg',
 '/opt/ml_data/cifar/cifar-10/images/train/im_3_9197_9.jpg',
 '/opt/ml_data/cifar/cifar-10/images/train/im_4_3685_4.jpg',
 '/opt/ml_data/cifar/cifar-10/images/train/im_4_1782_5.jpg',
 '/opt/ml_data/cifar/cifar-10/images/train/im_3_5363_1.jpg',
 '/opt/ml_data/cifar/cifar-10/images/train/im_3_6580_8.jpg',
 '/opt/ml_data/cifar/cifar-10/images/train/im_1_7338_1.jpg',
 '/opt/ml_data/cifar/cifar-10/images/train/im_1_2878_2.jpg',
 '/opt/ml_data/cifar/cifar-10/images/train/im_5_6712_3.jpg',
 '/opt/ml_data/cifar/cifar-10/images/train/im_5_4375_8.jpg',
 '/opt/ml_data/cifar/cifar-10/images/train/im_2_3663_4.jpg',
 '/opt/ml_data/cifar/cifar-10/images/train/im_4_8351_4.jpg',
 '/opt/ml_data/cifar/cifar-10/images/train/im_1_9878_1.jpg',
 '/opt/ml_data/cifar/cifar-10/images/train/im_5_2386_7.jpg',
 '/opt/ml_data/cifar/cifar-10/images/train/im_1_7338_1.jpg',
 '/opt/ml_data/cifar/cifar-10/images/train/im_1_6358_5.jpg',
 '/opt/ml_data/cifar/cifar-10/images/train/im_1_8695_0.jpg']

if __name__ == '__main__':

    for f in files_1:

        im = imread( f )
        print im.shape

    for f in files_2:

        im = imread( f )
        print im.shape

    print len( files_1+files_2 )
