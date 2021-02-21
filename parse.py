import re
import sys, getopt
import csv

prog = []
devname = []
blksz = []
version = []
img = []
duration = []
block = []
name = []
applicationstring = "Device Name,Input Image"

def parse_applications(line):
    for i in range(0,8):
        if re.search("NVPROF is profiling process",line):
            prog.append((re.search("gblur_[0-9]+_[0-9]+",line)).group(0))
            blksz.append((re.search("[0-9]+$",prog[-1])).group(0))
            try:
                images = (re.search(r'\W[a-z]+_[0-9]+.*jpeg\b',line)).group(0)
            except:
                images = (re.search(r'\W[a-z]+_[0-9]+.*png\b',line)).group(0)

            if ((re.search("gblur_[0-9]+_[0-9]+",line)).group(0))[6:8] == '35':
                devname.append("Tesla K20m")
            elif ((re.search("gblur_[0-9]+_[0-9]+",line)).group(0))[6:8] == '60':
                devname.append("Tesla P100-PCIE")
            elif ((re.search("gblur_[0-9]+_[0-9]+",line)).group(0))[6:8] == '70':
                devname.append("Tesla V100-PCIE")

            if ((re.search("gblur_[0-9]+_[0-9]+",line)).group(0))[9:] == '4':
                block.append(4)
            elif ((re.search("gblur_[0-9]+_[0-9]+",line)).group(0))[9:] == '8':
                block.append(8)
            elif ((re.search("gblur_[0-9]+_[0-9]+",line)).group(0))[9:] == '16':
                block.append(16)
            elif ((re.search("gblur_[0-9]+_[0-9]+",line)).group(0))[9:] == '32':
                block.append(32)
            image  = (images.split(' '))[1]
            img.append(image[11:])

def parse_nvprof(line):
    if re.search("^GPU",line):
        line = line.split(' ')
        line = [l for l in line if l]
        if len(line) == 11 :
            print(line)
            if line[3][-2:] == "ns":
                num = float(line[3][:-2])
                num = 10**-9 * num
                duration.append(num)
                fun = ' '.join([i for i in line[8:]])
                name.append(fun)
            elif line[3][-2:] == "us":
                num = float(line[3][:-2])
                num = 10**-6 * num
                duration.append(num)
                fun = ' '.join([i for i in line[8:]])
                name.append(fun)
            elif line[3][-2:] == "ms":
                num = float(line[3][:-2])
                num = 10**-3 * num
                duration.append(num)
                fun = ' '.join([i for i in line[8:]])
                name.append(fun)
            version.append("CUDA")

        elif len(line) == 17 :
            print(line)
            if line[3][-2:] == "ns":
                num = float(line[3][:-2])
                num = 10**-9 * num
                duration.append(num)
                name.append(line[8][:-9])
            elif line[3][-2:] == "us":
                num = float(line[3][:-2])
                num = 10**-6 * num
                duration.append(num)
                name.append(line[8][:-9])
            elif line[3][-2:] == "ms":
                num = float(line[3][:-2])
                num = 10**-3 * num
                duration.append(num)
                name.append(line[8][:-9])
            version.append("CUDA")

        else:
            print(line)
            if line[3][-2:] == "ns":
                num = float(line[3][:-2])
                num = 10**-9 * num
                duration.append(num)
                name.append(line[8][:-9])
            elif line[3][-2:] == "us":
                num = float(line[3][:-2])
                num = 10**-6 * num
                duration.append(num)
                name.append(line[8][:-9])
            elif line[3][-2:] == "ms":
                num = float(line[3][:-2])
                num = 10**-3 * num
                duration.append(num)
                name.append(line[8][:-9])
            version.append("CUDA")

    elif re.search("^[0-9]",line):
        line = line.split(' ')
        line = [l for l in line if l]
        if len(line) <= 3 :
            print(line)
            if line[1][:2] == "μs":
                num = float(line[0])
                num = 10**-6 * num
                duration.append(num)
                name.append(line[2][3:])
            elif line[1][:2] == "ms":
                num = float(line[0])
                num = 10**-3 * num
                duration.append(num)
                name.append(line[2][3:])
            elif line[1][:1] == "s":
                num = float(line[0])
                duration.append(num)
                name.append(line[2][2:])
            version.append("serial")

        elif len(line) == 9 :
            print(line)
            if line[1][-2:] == "ns":
                num = float(line[1][:-2])
                num = 10**-9 * num
                duration.append(num)
                fun = ' '.join([i for i in line[6:]])
                name.append(fun)
            elif line[1][-2:] == "us":
                num = float(line[1][:-2])
                num = 10**-6 * num
                duration.append(num)
                fun = ' '.join([i for i in line[6:]])
                name.append(fun)
            elif line[1][-2:] == "ms":
                num = float(line[1][:-2])
                num = 10**-3 * num
                duration.append(num)
                fun = ' '.join([i for i in line[6:]])
                name.append(fun)
            version.append("CUDA")

        elif len(line) == 15 :
            print(line)
            if line[1][-2:] == "ns":
                num = float(line[1][:-2])
                num = 10**-9 * num
                duration.append(num)
                name.append(line[6][:-9])
            elif line[1][-2:] == "us":
                num = float(line[1][:-2])
                num = 10**-6 * num
                duration.append(num)
                name.append(line[6][:-9])
            elif line[1][-2:] == "ms":
                num = float(line[1][:-2])
                num = 10**-3 * num
                duration.append(num)
                name.append(line[6][:-9])
            version.append("CUDA")

        elif len(line) == 14 :
            print(line)
            if line[1][-2:] == "ns":
                num = float(line[1][:-2])
                num = 10**-9 * num
                duration.append(num)
                name.append(line[6][:-9])
            elif line[1][-2:] == "us":
                num = float(line[1][:-2])
                num = 10**-6 * num
                duration.append(num)
                name.append(line[6][:-9])
            elif line[1][-2:] == "ms":
                num = float(line[1][:-2])
                num = 10**-3 * num
                duration.append(num)
                name.append(line[6][:-9])
            version.append("CUDA")

def read_file(ifile):
    with open(ifile,"r") as file_object:
        for lines in file_object:
            line = lines.strip()
            parse_applications(line)
            parse_nvprof(line)

def write_file(ofile):
    with open(ofile,"w") as fp:
       fp.write(applicationstring+",Version,Duration,Block Size,Name\n")
       csvwriter = csv.writer(fp)
       for a,b,c,d,e,f in zip(devname,img,version,duration,block,name):
           row = [a,b,c,d,e,f]
           csvwriter.writerow(row)


def main(argv):
    inputfile = ''
    outputfile = ''
    try:
        opts, args = getopt.getopt(argv,"hi:o:",["ifile=","ofile="])
    except getopt.GetoptError:
        print("parse.py -i <inputfile> -o <outputfile>")
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print("parse.py -i <inputfile> -o <outputfile>")
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inputfile = arg
        elif opt in ("-o", "--ofile"):
            outputfile = arg
    print('Input file is ', inputfile)
    print('Output file is ', outputfile)
    read_file(inputfile)
    write_file(outputfile)


if __name__ == "__main__":
    main(sys.argv[1:])
