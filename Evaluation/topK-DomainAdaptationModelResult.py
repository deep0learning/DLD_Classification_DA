import os


def getInformation(file):
    epoch = []
    training_accuracy = []
    validation_accuracy = []
    training_loss = []
    validation_loss = []
    source_test_accuracy = []
    source_test_fscore = []
    target_test_accuracy = []
    target_test_fscore = []

    summarized_list = []

    with open(file, 'r') as f:
        lines = f.readlines()
        for l in range(len(lines)):
            _line_information = lines[l].lower()

            if _line_information.startswith('epoch:'):
                splitted_info = _line_information.split(',')
                _eps = int(splitted_info[0][splitted_info[0].find('[') + 1:splitted_info[0].rfind(']')])
                _training_accuracy = float(splitted_info[2][splitted_info[2].find('[') + 1:splitted_info[2].rfind(']')])
                _validation_accuracy = float(
                    splitted_info[3][splitted_info[3].find('[') + 1:splitted_info[3].rfind(']')])
                _training_loss = float(splitted_info[4][splitted_info[4].find('[') + 1:splitted_info[4].rfind(']')])
                _validation_loss = float(splitted_info[5][splitted_info[5].find('[') + 1:splitted_info[5].rfind(']')])

                epoch.append(_eps)
                training_accuracy.append(_training_accuracy)
                validation_accuracy.append(_validation_accuracy)
                training_loss.append(_training_loss)
                validation_loss.append(_validation_loss)

            if _line_information.startswith('mode: source'):
                _tst_accuracy_info = lines[l + 1].lower()
                _tst_fscore_info = lines[l + 14].lower()
                _source_tst_accuracy = float(_tst_accuracy_info.split(':')[1][1:])
                _source_tst_fscore = float(_tst_fscore_info.split(':')[1][1:])
                source_test_accuracy.append(_source_tst_accuracy)
                source_test_fscore.append(_source_tst_fscore)

            if _line_information.startswith('mode: target'):
                _tst_accuracy_info = lines[l + 1].lower()
                _tst_fscore_info = lines[l + 14].lower()
                _target_tst_accuracy = float(_tst_accuracy_info.split(':')[1][1:])
                _target_tst_fscore = float(_tst_fscore_info.split(':')[1][1:])
                target_test_accuracy.append(_target_tst_accuracy)
                target_test_fscore.append(_target_tst_fscore)

    for e, tr_acc, val_acc, tr_loss, val_loss, src_tst_acc, src_tst_fsc, tar_tst_acc, tar_tst_fsc in zip(epoch,
                                                                                                         training_accuracy,
                                                                                                         validation_accuracy,
                                                                                                         training_loss,
                                                                                                         validation_loss,
                                                                                                         source_test_accuracy,
                                                                                                         source_test_fscore,
                                                                                                         target_test_accuracy,
                                                                                                         target_test_fscore):
        summarized_list.append(
            [e, tr_acc, val_acc, tr_loss, val_loss, src_tst_acc, src_tst_fsc, tar_tst_acc, tar_tst_fsc])

    return summarized_list


def sortList(summarized_list):
    sorted_list = sorted(summarized_list, key=lambda s: s[7], reverse=True)

    return sorted_list


def showResults(sorted_list, count):
    for i in range(count):
        # print(
        #     'Epoch [{}], Training Accuracy [{}], Validation Accuracy [{}], Training Loss [{}], Validation Loss [{}], '
        #     'Source Test Accuracy [{}], Source Test FScore [{}], Target Test Accuracy [{}], Target Test FScore [{}]'.format(
        #         sorted_list[i][0], sorted_list[i][1], sorted_list[i][2], sorted_list[i][3], sorted_list[i][4],
        #         sorted_list[i][5], sorted_list[i][6], sorted_list[i][7], sorted_list[i][8]))

        print(
            'Epoch [%d], Source Test Accuracy [%.4f], Source Test FScore [%.4f], Target Test Accuracy [%.4f], '
            'Target Test FScore [%.4f]' % (
                sorted_list[i][0], sorted_list[i][5], sorted_list[i][6], sorted_list[i][7], sorted_list[i][8]))
    print()


def main():
    top_k = 20
    file_dir = 'D:/2/experimental_logs/DA_EXP_G5/'
    file_name_list = os.listdir(file_dir)
    print(file_name_list)
    for file in file_name_list:
        print('File Name {}'.format(file))
        info_list = getInformation(file_dir + file)
        sort_list = sortList(info_list)
        showResults(sort_list, top_k)


if __name__ == '__main__':
    main()
