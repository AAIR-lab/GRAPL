import sys

if __name__ == "__main__":

    print(sys.argv[1])
    file_handle = open(sys.argv[1], "r")
    lines = []
    line_dict = {}
    for line in file_handle:
    
        line = line.strip()
        if line in line_dict:
            continue
        else:
            lines.append(line)
            line_dict[line] = True
            
    file_handle.close()
    file_handle = open(sys.argv[1], "w")
    for line in lines:
        file_handle.write(line + "\n")
    file_handle.close()
    
