import numpy as np
from sklearn.metrics import confusion_matrix
import torch

class _StreamMetrics(object):
    def __init__(self):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def update(self, gt, pred):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def get_results(self):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def to_str(self, metrics):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def reset(self):
        """ Overridden by subclasses """
        raise NotImplementedError()      

class StreamSegMetrics(_StreamMetrics):
    """
    Stream Metrics for Semantic Segmentation Task
    """
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist( lt.flatten(), lp.flatten() )
    
    @staticmethod
    def to_str(results):
        string = "\n"
        for k, v in results.items():
            if k!="Class IoU":
                string += "%s: %f\n"%(k, v)
        
        #string+='Class IoU:\n'
        #for k, v in results['Class IoU'].items():
        #    string += "\tclass %d: %f\n"%(k, v)
        return string

    def _fast_hist(self, label_true, label_pred):
        mask = (label_true >= 0) & (label_true < self.n_classes)
        hist = np.bincount(
            self.n_classes * label_true[mask].astype(int) + label_pred[mask],
            minlength=self.n_classes ** 2,
        ).reshape(self.n_classes, self.n_classes)
        return hist

    def get_results(self):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        """
        hist = self.confusion_matrix
                
        acc = np.diag(hist).sum() / hist.sum()
        acc_cls = np.diag(hist) / hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        mean_iu = np.nanmean(iu)
        freq = hist.sum(axis=1) / hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        cls_iu = dict(zip(range(self.n_classes), iu))

        return {
                "Overall Acc": acc,
                "Mean Acc": acc_cls,
                "FreqW Acc": fwavacc,
                "Mean IoU": mean_iu,
                "Class IoU": cls_iu,   
            }
        
    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))


class AverageMeter(object):
    """Computes average values"""
    def __init__(self):
        self.book = dict()

    def reset_all(self):
        self.book.clear()
    
    def reset(self, id):
        item = self.book.get(id, None)
        if item is not None:
            item[0] = 0
            item[1] = 0

    def update(self, id, val):
        record = self.book.get(id, None)
        if record is None:
            self.book[id] = [val, 1]
        else:
            record[0]+=val
            record[1]+=1

    def get_results(self, id):
        record = self.book.get(id, None)
        assert record is not None
        return record[0] / record[1]


class TimeAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



class Evaluator(object):
    def __init__(self, num_class, weather_num):
        self.num_class = num_class
        self.weather_num = weather_num

        self.confusion_matrix = np.zeros((self.num_class,)*2)  # shape:(num_class, num_class)
        self.confusion_matrix_sem_weather = {}
        for wea in range(self.weather_num):
            self.confusion_matrix_sem_weather[str(wea)] = np.zeros((self.num_class,)*2)
        self.confusion_matrix_weather = np.zeros((self.weather_num,)*2)

        self.weather_acc = torch.tensor([])
        # self.confusion_matrix_depth = {'20': np.zeros((self.num_class,) * 2),
        #                                '40': np.zeros((self.num_class,) * 2),
        #                                '60': np.zeros((self.num_class,) * 2),
        #                                '80': np.zeros((self.num_class,) * 2),
        #                                '100': np.zeros((self.num_class,) * 2)}
        self.weather_dict = {
            '0': 'fog',
            '1': 'night',
            '2': 'rain',
            '3': 'snow',
            '4': 'sunny'
        }

    def Pixel_Accuracy(self):
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc

    def Pixel_Accuracy_Class(self):
        acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        print('-----------Acc of each class-----------')
        print("road         : %.6f" % (acc[0] * 100.0), "%\t")
        print("sidewalk     : %.6f" % (acc[1] * 100.0), "%\t")
        print("building     : %.6f" % (acc[2] * 100.0), "%\t")
        print("wall         : %.6f" % (acc[3] * 100.0), "%\t")
        print("fence        : %.6f" % (acc[4] * 100.0), "%\t")
        print("pole         : %.6f" % (acc[5] * 100.0), "%\t")
        print("traffic light: %.6f" % (acc[6] * 100.0), "%\t")
        print("traffic sign : %.6f" % (acc[7] * 100.0), "%\t")
        print("vegetation   : %.6f" % (acc[8] * 100.0), "%\t")
        print("terrain      : %.6f" % (acc[9] * 100.0), "%\t")
        print("sky          : %.6f" % (acc[10] * 100.0), "%\t")
        print("person       : %.6f" % (acc[11] * 100.0), "%\t")
        print("rider        : %.6f" % (acc[12] * 100.0), "%\t")
        print("car          : %.6f" % (acc[13] * 100.0), "%\t")
        print("truck        : %.6f" % (acc[14] * 100.0), "%\t")
        print("bus          : %.6f" % (acc[15] * 100.0), "%\t")
        print("train        : %.6f" % (acc[16] * 100.0), "%\t")
        print("motorcycle   : %.6f" % (acc[17] * 100.0), "%\t")
        print("bicycle      : %.6f" % (acc[18] * 100.0), "%\t")
        acc = np.nanmean(acc)
        return acc

    def Mean_Intersection_over_Union(self, save_filename):
        MIoU = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))

        # print MIoU of each class
        print('-----------IoU of each class-----------')
        print("road         : %.6f" % (MIoU[0] * 100.0), "%\t")
        print("sidewalk     : %.6f" % (MIoU[1] * 100.0), "%\t")
        print("building     : %.6f" % (MIoU[2] * 100.0), "%\t")
        print("wall         : %.6f" % (MIoU[3] * 100.0), "%\t")
        print("fence        : %.6f" % (MIoU[4] * 100.0), "%\t")
        print("pole         : %.6f" % (MIoU[5] * 100.0), "%\t")
        print("traffic light: %.6f" % (MIoU[6] * 100.0), "%\t")
        print("traffic sign : %.6f" % (MIoU[7] * 100.0), "%\t")
        print("vegetation   : %.6f" % (MIoU[8] * 100.0), "%\t")
        print("terrain      : %.6f" % (MIoU[9] * 100.0), "%\t")
        print("sky          : %.6f" % (MIoU[10] * 100.0), "%\t")
        print("person       : %.6f" % (MIoU[11] * 100.0), "%\t")
        print("rider        : %.6f" % (MIoU[12] * 100.0), "%\t")
        print("car          : %.6f" % (MIoU[13] * 100.0), "%\t")
        print("truck        : %.6f" % (MIoU[14] * 100.0), "%\t")
        print("bus          : %.6f" % (MIoU[15] * 100.0), "%\t")
        print("train        : %.6f" % (MIoU[16] * 100.0), "%\t")
        print("motorcycle   : %.6f" % (MIoU[17] * 100.0), "%\t")
        print("bicycle      : %.6f" % (MIoU[18] * 100.0), "%\t")
        if self.num_class == 20:
            print("small obstacles: %.6f" % (MIoU[19] * 100.0), "%\t")

        # Save validation results
        with open(save_filename, 'a') as f:
            f.write('-----------IoU of each class-----------\n')
            f.write("road         : %.6f \n" % (MIoU[0] * 100.0))
            f.write("sidewalk     : %.6f \n" % (MIoU[1] * 100.0))
            f.write("building     : %.6f\n" % (MIoU[2] * 100.0))
            f.write("wall         : %.6f\n" % (MIoU[3] * 100.0))
            f.write("fence        : %.6f\n" % (MIoU[4] * 100.0))
            f.write("pole         : %.6f\n" % (MIoU[5] * 100.0))
            f.write("traffic light: %.6f\n" % (MIoU[6] * 100.0))
            f.write("traffic sign : %.6f\n" % (MIoU[7] * 100.0))
            f.write("vegetation   : %.6f\n" % (MIoU[8] * 100.0))
            f.write("terrain      : %.6f\n" % (MIoU[9] * 100.0))
            f.write("sky          : %.6f\n" % (MIoU[10] * 100.0))
            f.write("person       : %.6f\n" % (MIoU[11] * 100.0))
            f.write("rider        : %.6f\n" % (MIoU[12] * 100.0))
            f.write("car          : %.6f\n" % (MIoU[13] * 100.0))
            f.write("truck        : %.6f\n" % (MIoU[14] * 100.0))
            f.write("bus          : %.6f\n" % (MIoU[15] * 100.0))
            f.write("train        : %.6f\n" % (MIoU[16] * 100.0))
            f.write("motorcycle   : %.6f\n" % (MIoU[17] * 100.0))
            f.write("bicycle      : %.6f\n" % (MIoU[18] * 100.0))
            if self.num_class == 20:
                f.write("small obstacles: %.6f\n" % (MIoU[19] * 100.0))

        MIoU = np.nanmean(MIoU)
        return MIoU

    def Mean_Intersection_over_Union_each_weather(self, save_filename):
        weather_MIoU = {}
        for wea in range(self.weather_num):
            cf_mat = self.confusion_matrix_sem_weather[str(wea)]

            MIoU = np.diag(cf_mat) / (
                        np.sum(cf_mat, axis=1) + np.sum(cf_mat, axis=0) -
                        np.diag(cf_mat)) * 100.0

            # print MIoU of each class
            print('--------------------------------IoU of each class in {}------------------------------------\n'.format(self.weather_dict[str(wea)]))
            print("road            sidewalk        building        wall            fence           pole  ")
            print("%-15.3f %-15.3f %-15.3f %-15.3f %-15.3f %-15.3f  "
                  % (MIoU[0],       MIoU[1],          MIoU[2],          MIoU[3],       MIoU[4],     MIoU[5]))

            print("traffic light   traffic sign    vegetation      terrain         sky             person")
            print("%-15.3f %-15.3f %-15.3f %-15.3f %-15.3f %-15.3f  "
                  % (MIoU[6],       MIoU[7],          MIoU[8],          MIoU[9],       MIoU[10],    MIoU[11]))

            print("rider           car             truck           bus             train           motorcycle")
            print("%-15.3f %-15.3f %-15.3f %-15.3f %-15.3f %-15.3f  "
                  % (MIoU[12],      MIoU[13],         MIoU[14],         MIoU[15],      MIoU[16],    MIoU[17]))

            print("bicycle  ")
            print("%-15.3f     " % MIoU[18])

            if self.num_class == 20:
                print("small obstacles: %-15.3f" % (MIoU[19]), "%\t")

            # Save validation results
            with open(save_filename, 'a') as f:
                f.write(
                    '--------------------------------IoU of each class in {}------------------------------------\n'.format(
                        self.weather_dict[str(wea)]))
                f.write("road            sidewalk        building        wall            fence           pole  \n")
                f.write("%-15.3f %-15.3f %-15.3f %-15.3f %-15.3f %-15.3f  \n"
                      % (MIoU[0], MIoU[1], MIoU[2], MIoU[3], MIoU[4], MIoU[5]))
                f.write("traffic light   traffic sign    vegetation      terrain         sky             person\n")
                f.write("%-15.3f %-15.3f %-15.3f %-15.3f %-15.3f %-15.3f  \n"
                      % (MIoU[6], MIoU[7], MIoU[8], MIoU[9], MIoU[10], MIoU[11]))
                f.write("rider           car             truck           bus             train           motorcycle\n")
                f.write("%-15.3f %-15.3f %-15.3f %-15.3f %-15.3f %-15.3f  \n"
                      % (MIoU[12], MIoU[13], MIoU[14], MIoU[15], MIoU[16], MIoU[17]))
                f.write("bicycle  \n")
                f.write("%-15.3f     \n" % MIoU[18])
                if self.num_class == 20:
                    f.write("small obstacles: %-15.3f\n" % (MIoU[19]))

            MIoU = np.nanmean(MIoU)
            print("mIoU in {} : {}".format(self.weather_dict[str(wea)], MIoU))
            with open(save_filename, 'a') as f:
                f.write("mIoU in {} : {} \n".format(self.weather_dict[str(wea)], MIoU))

            weather_MIoU[str(wea)] = MIoU
        return weather_MIoU

    # def Mean_Intersection_over_Union_with_depth(self, d_range, save_filename):
    #     cf_matrix = self.confusion_matrix_depth[str(d_range)]

    #     MIoU = np.diag(cf_matrix) / (
    #                 np.sum(cf_matrix, axis=1) + np.sum(cf_matrix, axis=0) -
    #                 np.diag(cf_matrix))

    #     # print MIoU of each class
    #     print('-----------IoU of each classes-----------')
    #     if self.num_class == 20:
    #         print("small obstacles: %.6f" % (MIoU[19] * 100.0), "%\t")

    #         with open(save_filename, 'a') as f:
    #             f.write("small obstacles: %.6f \n" % (MIoU[19] * 100.0))

    #     MIoU = np.nanmean(MIoU)
    #     return MIoU

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image, gt_weather):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

        for i, wea in enumerate(gt_weather):
            self.confusion_matrix_sem_weather[str(wea)] += self._generate_matrix(gt_image[i], pre_image[i])


    def add_batch_weather(self, gt_weather, weather_pred):
        _, preds = torch.max(weather_pred, dim=1)
        acc = torch.tensor([torch.sum(preds == gt_weather).item() / len(preds)])

        for t, p in zip(gt_weather.view(-1), preds.view(-1)):
            self.confusion_matrix_weather[t.int(), p.int()] += 1

        self.weather_acc = torch.cat((self.weather_acc, acc))


    def _generate_matrix_with_depth(self, gt_image, pre_image, mask):
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)

        return confusion_matrix

    # def add_batch_with_depth(self, gt_image, pre_image, disp):
    #     assert gt_image.shape == pre_image.shape
    #     mask = (gt_image >= 0) & (gt_image < self.num_class)
    #     depth_range = [20, 40, 60, 80, 100]

    #     self.confusion_matrix += self._generate_matrix_with_depth(gt_image, pre_image, mask)

    #     for d_range in depth_range:
    #         if d_range == 20:
    #             disp_mask = mask & (disp >= 483 / d_range)  # focal_length*base_line : 2300pixel * 0.21m
    #         else:
    #             disp_mask = mask & (disp >= 483 / d_range) & (disp < (483 / (d_range - 20)))
    #         self.confusion_matrix_depth[str(d_range)] += self._generate_matrix_with_depth(gt_image, pre_image, disp_mask)


    def get_results(self):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        """
        hist = self.confusion_matrix
        print(hist)
        
        acc = np.diag(hist).sum() / hist.sum()
        acc_cls = np.diag(hist) / hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        mean_iu = np.nanmean(iu)
        freq = hist.sum(axis=1) / hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        cls_iu = dict(zip(range(self.num_class), iu))

        return {
                "Overall Acc": acc,
                "Mean Acc": acc_cls,
                "FreqW Acc": fwavacc,
                "Mean IoU": mean_iu,
                "Class IoU": cls_iu,
            }

    def get_weather_results(self, save_filename, gan_based=False):
        cf = self.confusion_matrix_weather
        print('\nweather confusion_matrix:')
        print('|fog|night|rain|snow|sunny|')
        print(cf)

        with open(save_filename, 'a') as f:
            if gan_based:
                f.write("\n--------- GAN-based results -------")
            f.write('weather confusion_matrix:\n')
            f.write('|fog|night|rain|snow|sunny|\n')
            np.savetxt(f, cf, fmt='%-5.0f')

        purity = np.sum(np.trace(cf)) / np.sum(cf)
        print('purity score:{}'.format(purity))

        acc_mean = self.weather_acc.mean()
        print("weather accuracy: {}".format(acc_mean))

        with open(save_filename, 'a') as f:
            f.write('purity score: %.5f \n' % (purity))
            f.write("weather accuracy: %.5f \n"% (acc_mean))

        return acc_mean


    @staticmethod
    def to_str(results):
        string = "\n"
        for k, v in results.items():
            if k != "Class IoU":
                string += "%s: %f\n" % (k, v)

        return string

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)
        self.confusion_matrix_weather = np.zeros((self.weather_num,)*2)
        # self.confusion_matrix_depth = {'20': np.zeros((self.num_class,) * 2),
        #                                '40': np.zeros((self.num_class,) * 2),
        #                                '60': np.zeros((self.num_class,) * 2),
        #                                '80': np.zeros((self.num_class,) * 2),
        #                                '100': np.zeros((self.num_class,) * 2)}
        self.weather_acc = torch.tensor([])

        self.confusion_matrix_sem_weather = {}
        for wea in range(self.weather_num):
            self.confusion_matrix_sem_weather[str(wea)] = np.zeros((self.num_class,)*2)
