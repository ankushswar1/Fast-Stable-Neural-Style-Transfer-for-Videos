import os
import argparse
import errorScript
import csv

parser = argparse.ArgumentParser()
parser.add_argument("--gtDirectory", type=str, required=True, dest="gtDirectory",
 help="Directory path to folder with ground truth images")
parser.add_argument("--mdlDirectory", type=str, required=True, dest="mdlDirectory",
  help="Directory path to folder with stylized model images")
parser.add_argument("--bslDirectory", type=str, required=True, dest="bslDirectory",
  help="Directory path to folder with baseline model images")
parser.add_argument("--csvFile", type=str, required=True, dest="csvFile",
  help="save to CSV file")
args = parser.parse_args()

def get_subdirectory_names(dir):
    return [d for d in os.listdir(dir) if os.path.isdir(dir + '/' + d)];

def get_file_names(dir, ext):
    return [d for d in os.listdir(dir) if d.endswith(ext)];

def savetoCSV(mass_list):
    with open(args.csvFile, 'w') as csvFile:
        writer = csv.writer(csvFile)
        for list in mass_list:
            writer.writerow(list)
    csvFile.close()

if __name__ == '__main__':
    vids = get_subdirectory_names(args.gtDirectory)
    print(args.gtDirectory)
    styles = get_subdirectory_names(args.mdlDirectory + '/' + vids[0])

    mass_list = []

    for vid in vids:
        for style in styles:
            print(vid, style)
            gt = errorScript.readPaths(args.gtDirectory + '/' + vid)
            mdl = errorScript.readPaths(args.mdlDirectory + '/' + vid + '/' + style)
            bsl = errorScript.readPaths(args.bslDirectory + '/' + vid + '/' + style)

            mse_bsl = errorScript.temporalError(gt, bsl)
            mse_mdl = errorScript.temporalError(gt, mdl)

            ssim_bsl = errorScript.SSIM(gt, bsl)
            ssim_mdl = errorScript.SSIM(gt, mdl)

            mass_list.append([vid, style, mse_bsl, mse_mdl, ssim_bsl, ssim_mdl])

    savetoCSV(mass_list)
