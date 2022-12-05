import os
from glob import glob


def gen_acdc():
    data_dir = '/root/dataset/acdc'
    city_data_dir = '/root/dataset/cityscapes'

    full_city = True

    if (full_city):
        train_file = 'acdc_cityfull_train.txt'
        val_file = 'acdc_cityfull_val.txt'
        test_file = 'acdc_cityfull_test.txt'
    else:
        train_file = 'acdc_city_train.txt'
        val_file = 'acdc_city_val.txt'
        test_file = 'acdc_city_test.txt'
    dir_name = 'rgb_anon'
    dir_name_city = 'leftImg8bit'


    # Split the training set with 4:1 raito (160 for training, 40 for validation)
    write_file(train_file, data_dir, city_data_dir, dir_name, dir_name_city, mode='train', full_city=full_city)
    write_file(val_file, data_dir, city_data_dir, dir_name, dir_name_city, mode='val', full_city=full_city)
    write_file_test(test_file, data_dir, city_data_dir, dir_name, dir_name_city, mode='test', full_city=full_city)


def write_file(file, data_dir, city_data_dir, dir_name, dir_name_city, mode='train', full_city=False):
    weather = ['fog', 'night', 'rain', 'snow']
    with open(file, 'w') as f:
        for wea in weather:
            left_dir = os.path.join(data_dir, dir_name, wea, mode)
            left_imgs = recursive_glob(left_dir, suffix='.png')
            left_imgs.sort()
            print('Number of {} images in {} weather: {}'.format(mode, wea, len(left_imgs)))

            for left_img in left_imgs:
                label_path = left_img.replace(dir_name, 'gt')
                label_path = label_path.replace('.png', '_labelIds.png')

                f.write(left_img + ' ')
                f.write(wea + ' ')
                f.write(label_path + '\n')

        weather_img_num = len(left_imgs)

        left_dir = os.path.join(city_data_dir, dir_name_city, mode)
        left_imgs = recursive_glob(left_dir, suffix='.png')
        left_imgs.sort()
        if full_city:
            print('Number of {} images in cityscapes {}, sunny weather '.format(mode, len(left_imgs)))
        else:
            print('Number of {} images in cityscapes: {}, we select only {} numbers '.format(mode, len(left_imgs), weather_img_num))

        for i, left_img in enumerate(left_imgs):
            if i > weather_img_num and (full_city == False):
                break
            label_path = left_img.replace(dir_name_city, 'gtFine')
            label_path = label_path.replace('.png', '_labelIds.png')

            f.write(left_img + ' ')
            f.write('sunny ')
            f.write(label_path + '\n')


def write_file_test(file, data_dir, city_data_dir, dir_name, dir_name_city, mode='train', full_city=False):
    weather = ['fog', 'night', 'rain', 'snow']
    with open(file, 'w') as f:
        for wea in weather:
            left_dir = os.path.join(data_dir, dir_name, wea, mode)
            left_imgs = recursive_glob(left_dir, suffix='.png')
            left_imgs.sort()
            print('Number of {} images in {} weather: {}'.format(mode, wea, len(left_imgs)))

            for left_img in left_imgs:
                label_path = left_img.replace(dir_name, 'gt')
                label_path = label_path.replace('.png', '_labelIds.png')
                f.write(left_img+ ' ')
                f.write(wea + '\n')

        weather_img_num = len(left_imgs)

        left_dir = os.path.join(city_data_dir, dir_name_city, mode)
        left_imgs = recursive_glob(left_dir, suffix='.png')
        left_imgs.sort()
        if full_city:
            print('Number of {} images in cityscapes {}, sunny weather '.format(mode, len(left_imgs)))
        else:
            print('Number of {} images in cityscapes: {}, we select only {} numbers '.format(mode, len(left_imgs), weather_img_num))

        for i, left_img in enumerate(left_imgs):
            if i > weather_img_num and (full_city == False) :
                break

            label_path = left_img.replace(dir_name_city, 'gtFine')
            label_path = label_path.replace('.png', '_labelIds.png')

            f.write(left_img+ ' ')
            f.write('sunny' + '\n')


def recursive_glob( rootdir='.', suffix=''):
    """Performs recursive glob with given suffix and rootdir
        :param rootdir is the root directory
        :param suffix is the suffix to be searched
    """
    return [os.path.join(looproot, filename)
            for looproot, _, filenames in os.walk(rootdir)
            for filename in filenames if filename.endswith(suffix)]

if __name__ == '__main__':
    gen_acdc()
