import numpy as np
import os
import re


def is_float(element):
    if isinstance(element, list) or isinstance(element, dict) or isinstance(element, tuple):
        return False

    try:
        float(element)
        return True
    except ValueError:
        return False


def getFileList(path, filter='', fileFullPath=False):
    fileList = []
    with os.scandir(path) as it:
        for entry in it:
            if filter:
                if entry.is_file() and filter in entry.name:
                    fileList.append(entry.name)
            else:
                if entry.is_file():
                    fileList.append(entry.name)

    files = sorted(fileList)
    if fileFullPath:
        for i in range(len(files)):
            files[i] = path + '/' + files[i]

    return files


def user_select_files(available_files):
    print("可用的文件列表:")
    for idx, file in enumerate(available_files):
        print(f"{idx}: {file}")

    selected_indexes = input("请输入要选择的文件索引（以逗号分隔，例如 '0,1'）: ")
    try:
        selected_indexes = list(map(int, selected_indexes.split(',')))
        selected_files = [available_files[i] for i in selected_indexes if 0 <= i < len(available_files)]

        if selected_files:
            return selected_files
        else:
            print("未选择有效文件。")
            return []
    except ValueError:
        print("请输入有效的索引。")
        return []


def readColumn(filePath, iColumn, rowSkip=0, keyStr='', sep=''):
    f = open(filePath, 'r')
    dataList = []

    iRow = -1
    for line in f:
        iRow += 1

        if line[0] == '#' or iRow < rowSkip:
            continue
        if keyStr and keyStr not in line:
            continue

        if not sep:
            datas = list(filter(None, re.split(r'[\s\,\:]', line)))
        else:
            datas = list(filter(None, re.split(sep, line)))

        # print(line, datas, keyStr, iRow, (not keyStr), (keyStr not in line))
        # exit()

        if len(datas) > iColumn:
            dataList.append(datas[iColumn])

    dataListNP = np.empty(len(dataList))
    for i in range(len(dataList)):
        if dataList[i] == "nan" or dataList[i] == "Nan" or not is_float(dataList[i]):
            dataListNP[i] = 0
        else:
            dataListNP[i] = dataList[i]

    f.close()
    return dataListNP
