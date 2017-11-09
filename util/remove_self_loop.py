#!/usr/bin/env python3
import sys

if len(sys.argv) != 3:
    print ("usage:",sys.argv[0],"input output")
    sys.exit(0)

with open(sys.argv[1], 'r') as input_file, open(sys.argv[2],'w') as output_file:

    nodi = 0
    archi = 0

    start = []
    end = []
    for line in input_file:

        if line.find('%') != -1:
            start.append(line)
            continue

        token = line.split()
        nodi  = int(token[0])
        archi = int(token[2])
        print(nodi,archi)
        break

    archi_rimossi = 0

    for line in input_file:
        if line.find('%') != -1:
            end.append(line)
            continue

        token = line.split()
        partenza  = int(token[0])
        arrivo = int(token[1])

        if partenza != arrivo:
            end.append(line)
        else:
            archi_rimossi+=1

    print("rimossi",archi_rimossi,"archi")

    for line in start:
        output_file.write(line)

    archi = archi - archi_rimossi

    output_file.write(str(nodi) +" "+str(nodi)+" "+str(archi)+"\n")

    for line in end:
        output_file.write(line)


