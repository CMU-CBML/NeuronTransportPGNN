
#%%
import numpy as np
import torch
from mesh.datastructure import *

def ReadBEXT(filename, degree):
    pt_count = 0
    ele_count = 0

    num_cpt = -1
    num_ele = -1
    nrow = -1
    nbzpt = 64
    ele_first = -1
    ele_end = -1
    tmp_mat = []
    pts = []
    bzmesh = []
    element_type = 'hex'
    count_belem = 0
    with open(filename) as fp:
        for count, line in enumerate(fp):
            tmp_old = line.rstrip('\n').split(" ")
            tmp = [elem for elem in tmp_old if elem.strip()]
            if count == 0:
                num_cpt = int(tmp[0])
                num_ele = int(tmp[1])
            elif count < num_cpt + 1:
                tmp_pt = []
                tmp_pt.append(float(tmp[0]))
                tmp_pt.append(float(tmp[1]))
                tmp_pt.append(float(tmp[2]))
                pts.append(tmp_pt)
            else:
                if len(tmp) == 3:
                    ele_idx = int(tmp[0])
                    nrow = int(tmp[1])
                    ele_first = count + 1
                    ele_end = ele_first + nrow
                else:
                    if len(tmp_mat) == nrow:
                        tmp_mat = []
                    if count >= ele_first and count <= ele_end:
                        if count == ele_first:
                            tmp_IEN = np.fromstring(line, dtype=int, sep=' ')
                        else:
                            if element_type == 'hex':
                                tmp_matrow = np.fromstring(line, dtype=float, sep=' ')
                                tmp_mat.append(np.float32(tmp_matrow))
                    if count == ele_end:
                        cpts = torch.tensor(pts)
                        tmp_cpts_e = cpts[tmp_IEN[:],:]
                        tmp_bzele = BezierElement(tmp_IEN,
                                              np.array(tmp_mat),
                                              tmp_cpts_e,
                                              ele_type=element_type)
                        tmp_mat = []
                        bzmesh.append(tmp_bzele)
                        print('Element {}: bz_element load!'.format(ele_count))
                        ele_count+=1
    return pts, bzmesh

def ReadBEXT_old(filename, degree):
    pt_count = 0
    ele_count = 0
    nrow = -1
    nbzpt = -1
    ele_first = -1
    ele_end = -1
    tmp_mat = []
    pts = []
    bzmesh = []
    element_type = ''
    count_belem = 0
    with open(filename) as fp:
        for count, line in enumerate(fp):
            tmp_old = line.rstrip('\n').split(" ")
            tmp = [elem for elem in tmp_old if elem.strip()]

            if tmp[1] == 'plane':
                nbzpt = 16
                element_type = 'plane'
            elif tmp[1] == 'hex':
                nbzpt = 64
                element_type = 'hex'

            if tmp[0] == 'nodeN':
                npts = int(tmp[1])

            if tmp[0] == 'elemN':
                nele = int(tmp[1])
                # if element_type == 'plane':
                #     nele = int(nele/(degree + 1))

            if tmp[0] == 'gnode':
                tmp_pt = []
                tmp_pt.append(float(tmp[1]))
                tmp_pt.append(float(tmp[2]))
                tmp_pt.append(float(tmp[3]))
                pts.append(tmp_pt)
                pt_count += 1

            if tmp[0] == 'belem':
                nrow = int(tmp[1])
                ele_first = count + 1
                ele_end = ele_first + nrow
                count_belem += 1

            if len(tmp_mat) == nrow:
                tmp_mat = []
            if count >= ele_first and count <= ele_end:
                if count == ele_first:
                    tmp_IEN = np.fromstring(line, dtype=int, sep=' ')
                    # if element_type == 'plane':
                    #     tmp_layer = np.arange(degree + 1)
                    #     idx_layer = (count_belem-1)//nele
                    #     tmp_layer = tmp_layer - tmp_layer[idx_layer]
                    #     shift_layer = np.delete(tmp_layer, idx_layer)

                    #     for idx_deg in range(0,degree):
                    #         tmp_IENappend = np.fromstring(line, dtype=int, sep=' ')
                    #         tmp_IENappend = tmp_IENappend + shift_layer[idx_deg] * int(npts/(degree + 1))
                    #         if shift_layer[idx_deg] > 0:
                    #             tmp_IEN = np.append(tmp_IEN,tmp_IENappend)
                    #         else:
                    #             tmp_IEN = np.append(tmp_IENappend,tmp_IEN)

                else:
                    if element_type == 'plane':
                        tmp_matrow = np.fromstring(line, dtype=float, sep=' ')
                        tmp_mat.append(tmp_matrow)
                    elif element_type == 'hex':
                        tmp_matrow = []
                        if tmp[0] == 'd':
                            for ii in range(0, nbzpt):
                                tmp_matrow.append(float(tmp[ii + 1]))
                        elif tmp[0] == 's':
                            for ii in range(0, int(tmp[1])):
                                tmp_matrow.append(int(tmp[ii * 2 + 2]))
                                tmp_matrow.append(float(tmp[ii * 2 + 3]))
                        tmp_mat.append(tmp_matrow)
                # if count == ele_end and count_belem <= nele:
                if count == ele_end:
                    tmp_bzele = BezierElement(tmp_IEN,
                                              tmp_mat,
                                              ele_type=element_type)
                    bzmesh.append(tmp_bzele)
    return pts, bzmesh


# root = "./io/Mesh2"
# filename = root + "/BotSurf/BotSurf_extrude_CM_NewInterface_extrude_CM_NewInterface_extrude_BEXT_layer.txt"
# pts, bzmesh = ReadBEXT(filename, 2)



#%%
def ReadMergeMapping(filename):
    mapping = []
    with open(filename) as fp:
        for count, line in enumerate(fp):
            if line.strip():
                tmp_old = line.split(" ")
                tmp = [elem for elem in tmp_old if elem.strip()]
                mapping.append(int(tmp[0]))
    return mapping


def ReadMergeVertices(filename):
    pts = []
    with open(filename) as fp:
        for count, line in enumerate(fp):
            if line.strip():
                tmp_old = line.split(" ")
                tmp = [elem for elem in tmp_old if elem.strip()]
                tmp_pt = []
                tmp_pt.append(float(tmp[0]))
                tmp_pt.append(float(tmp[1]))
                tmp_pt.append(float(tmp[2]))
                pts.append(tmp_pt)
    return pts


def ReadVTK_CM(filename):
    pt_count = 0
    ele_count = 0
    n_ele = -1
    ele_start = -1
    ele_end = -1
    tmp_mat = []
    pts = []
    tmesh = []
    with open(filename) as fp:
        for count, line in enumerate(fp):
            if line.strip():
                tmp_old = line.split(" ")
                tmp = [elem for elem in tmp_old if elem.strip()]

                if tmp[0] == 'CELLS':
                    n_ele = int(tmp[1])
                    ele_start = count + 1
                    ele_end = ele_start + n_ele

                if count >= ele_start and count < ele_end:
                    tmp_cnct = np.fromstring(line, dtype=int, sep=' ')
                    tmp_cnct = tmp_cnct[1:]
                    tmp_cmele = ControlElement(tmp_cnct)
                    tmesh.append(tmp_cmele)
    return pts, tmesh


def WriteVTK_CM(filename, pts, tmesh):

    nPoint = 0
    nElement = 0

    nPoint += len(pts)
    nElement += len(tmesh)

    outF = open(filename, 'w')
    outF.write('# vtk DataFile Version 4.2\n')
    outF.write('vtk output\n')
    outF.write('ASCII\n')
    outF.write('DATASET UNSTRUCTURED_GRID\n')
    print('POINTS {} double'.format(nPoint), file=outF)

    for i in range(0, len(pts)):
        print(' '.join(str(x) for x in pts[i]), file=outF)

    print('CELLS {} {}'.format(nElement, nElement * 9), file=outF)
    for i in range(0, len(tmesh)):
        print('8', ' '.join(str(x) for x in tmesh[i].cnct), file=outF)

    print('CELL_TYPES {}'.format(nElement), file=outF)
    for i in range(0, nElement):
        outF.write('12\n')

    outF.close()


def WriteABAQUS_tsplines(filename,
                         pts,
                         bzmesh,
                         thickness,
                         num_layer,
                         num_layer_wdir,
                         output_mode,
                         modelname=None):

    nPoint = 0
    nElement = 0

    nPoint += len(pts)
    nElement += len(bzmesh)

    outF = open(filename, 'w')
    print('**this is the T-spline input file for {}'.format(modelname),
          file=outF)
    print('*thickness', file=outF)
    print(thickness, file=outF)

    print('*number of layers', file=outF)
    print(num_layer, file=outF)

    print('*fiber angle', file=outF)
    print(0.0, file=outF)

    print('*knotW', file=outF)
    tmplist = []
    for ii in range(0, num_layer_wdir):
        tmplist.insert(0, 0)
        tmplist.append(1)
    for ii in range(int((len(tmplist) - 1) / 8) + 1):
        if ii < int((len(tmplist) - 1) / 8):
            print(','.join(
                str(x) for x in (tmplist[ii * 8:(ii + 1) * 8])),
                  end=',',
                  file=outF)
        else:
            print(','.join(
                str(x) for x in (tmplist[ii * 8:(ii + 1) * 8])),
                  end='',
                  file=outF)
        print('', file=outF)

    # print('0,0,0,1,1,1', file=outF)

    print('*elRangeW', file=outF)
    print('0,1', file=outF)

    print('*coordinates', file=outF)
    for i in range(0, len(pts)):
        # print('\t'.join(str(x) for x in pts[i]), file=outF)
        for x in pts[i]:
            print('{0:>10}'.format(round(x, 8)), end='\t', file=outF)
        print('', file=outF)

    print('*weights', file=outF)
    for i in range(0, len(pts)):
        print('1', file=outF)

    print('*bext', file=outF)

    for i in range(0, len(bzmesh)):
        # for j in range(0, 1):
        for j in range(0, len(bzmesh[i])):
            if bzmesh[i][j].type == 'plane' and j < len(
                    bzmesh[i]) / num_layer_wdir and (output_mode == 'both' or
                                                     output_mode == 'plane'):
                print('belem {} 3 3'.format(len(bzmesh[i][j].IEN)), file=outF)

                # print('\t'.join(str(x) for x in bzmesh[i][j].IEN), file=outF)
                for x in bzmesh[i][j].IEN:
                    print('{0:>10}'.format(round(x, 8)), end='\t', file=outF)
                print('', file=outF)

                for k in range(0, len(bzmesh[i][j].mat)):
                    # print('\t'.join(str(x) for x in bzmesh[i][j].mat[k]), file=outF)
                    for x in bzmesh[i][j].mat[k]:
                        xx = round(x, 8)
                        if (xx == 0.):
                            print('{0:>10}'.format(int(xx)),
                                  end='\t',
                                  file=outF)
                        else:
                            print('{0:>10}'.format(round(x, 8)),
                                  end='\t',
                                  file=outF)
                    print('', file=outF)

            if bzmesh[i][j].type == 'hex' and (output_mode == 'both'
                                               or output_mode == 'hex'):
                print('belem {} 3 3 3'.format(len(bzmesh[i][j].IEN)),
                      file=outF)

                for x in bzmesh[i][j].IEN:
                    print('{0:>10}'.format(round(x, 8)), end='\t', file=outF)
                print('', file=outF)

                # print(len(bzmesh[i][j].mat))

                for k in range(0, len(bzmesh[i][j].mat)):
                    # print('\t'.join(str(x) for x in bzmesh[i][j].mat[k]), file=outF)
                    ncol = len(bzmesh[i][j].mat[k])
                    if ncol == 64:
                        tmp_mat = bzmesh[i][j].mat[k]
                    else:
                        tmp_mat = [0] * 64
                        for ii in range(0, int(ncol / 2)):
                            tmp_mat[bzmesh[i][j].mat[k][
                                ii * 2]] = bzmesh[i][j].mat[k][ii * 2 + 1]

                    # print(k, len(tmp_mat))

                    for x in tmp_mat:
                        xx = round(x, 8)
                        if (xx == 0.):
                            print('{0:>10}'.format(int(xx)),
                                  end='\t',
                                  file=outF)
                        else:
                            print('{0:>10}'.format(round(x, 8)),
                                  end='\t',
                                  file=outF)
                    print('', file=outF)

    # for j in range(0, len(bzmesh[0])):
    #     print('belem {} 3 3 3'.format(len(bzmesh[0][j].IEN)), file=outF)

    # ! Shell BEXT
    # for i in range(1, len(bzmesh)):
    #     for j in range(0, len(bzmesh[i])):
    #         print('belem {} 3 3'.format(len(bzmesh[i][j].IEN)), file=outF)

    #         # print('\t'.join(str(x) for x in bzmesh[i][j].IEN), file=outF)
    #         for x in bzmesh[i][j].IEN:
    #             print('{0:>10}'.format(round(x, 8)), end='\t', file=outF)
    #         print('', file=outF)

    #         for k in range(0, len(bzmesh[i][j].mat)):
    #             # print('\t'.join(str(x) for x in bzmesh[i][j].mat[k]), file=outF)
    #             for x in bzmesh[i][j].mat[k]:
    #                 xx = round(x, 8)
    #                 if (xx == 0.):
    #                     print('{0:>10}'.format(int(xx)), end='\t', file=outF)
    #                 else:
    #                     print('{0:>10}'.format(round(x, 8)),
    #                           end='\t',
    #                           file=outF)
    #             print('', file=outF)

    outF.close()


def WriteABAQUS_inp(filename,
                    pts,
                    bzmesh,
                    num_layer_wdir,
                    output_mode=None,
                    modelname=None):

    nPoint = 0
    nElement = 0

    nPoint += len(pts)
    nElement += len(bzmesh)

    bzmesh_shell_dict = {}
    bzmesh_solid_dict = {}
    shift = []
    num_bzmesh = []
    tmp_sum = 0

    for i in range(0, len(bzmesh)):
        shift.append(tmp_sum)
        if bzmesh[i][0].type == 'plane':
            num_bzmesh.append(int(len(bzmesh[i]) / num_layer_wdir))
            tmp_sum += int(len(bzmesh[i]) / num_layer_wdir)
        else:
            num_bzmesh.append(len(bzmesh[i]))
            tmp_sum += len(bzmesh[i])
    shift.append(tmp_sum)
    print(tmp_sum)
    print('belem shift: ', shift)

    count = 0
    for i in range(0, len(bzmesh)):
        for j in range(0, len(bzmesh[i])):
            if bzmesh[i][j].type == 'plane' and j < len(
                    bzmesh[i]) / num_layer_wdir and (output_mode == 'both' or
                                                     output_mode == 'plane'):
                tmp_num_cp = len(bzmesh[i][j].IEN)
                if str(tmp_num_cp) not in bzmesh_shell_dict.keys():
                    bzmesh_shell_dict[str(tmp_num_cp)] = [j + shift[i]]
                else:
                    bzmesh_shell_dict[str(tmp_num_cp)].append(j + shift[i])
            if bzmesh[i][j].type == 'hex' and (output_mode == 'both'
                                               or output_mode == 'hex'):
                tmp_num_cp = len(bzmesh[i][j].IEN)
                if str(tmp_num_cp) not in bzmesh_solid_dict.keys():
                    bzmesh_solid_dict[str(tmp_num_cp)] = [j + shift[i]]
                else:
                    bzmesh_solid_dict[str(tmp_num_cp)].append(j + shift[i])

    outF = open(filename, 'w')

    # print(bzmesh_shell_dict, file=outF)

    print('*Heading', file=outF)
    print('*Part, name=Part-1', file=outF)
    print('*Node', file=outF)

    for i in range(0, len(pts)):
        print(i + 1, end=',', file=outF)
        print(','.join(str(round(x, 8)) for x in pts[i]), file=outF)

    for i in range(0, len(bzmesh)):
        for j in range(0, len(bzmesh[i])):
            if bzmesh[i][j].type == 'plane' and j < len(
                    bzmesh[i]) / num_layer_wdir and (output_mode == 'both' or
                                                     output_mode == 'plane'):
                tmp_num_cp = len(bzmesh[i][j].IEN)
                if str(tmp_num_cp) not in bzmesh_shell_dict.keys():
                    bzmesh_shell_dict[str(tmp_num_cp)] = [j + shift[i]]
                else:
                    bzmesh_shell_dict[str(tmp_num_cp)].append(j + shift[i])
            # if bzmesh[i][j].type == 'hex' and (output_mode == 'both'
            #                                    or output_mode == 'hex'):
            #     tmp_num_cp = len(bzmesh[i][j].IEN)
            #     if str(tmp_num_cp) not in bzmesh_solid_dict.keys():
            #         bzmesh_solid_dict[str(tmp_num_cp)] = [j + shift[i]]
            #     else:
            #         bzmesh_solid_dict[str(tmp_num_cp)].append(j + shift[i])

    if output_mode == 'plane':
        for shell_key in bzmesh_shell_dict.keys():
            # ! Heading of UEL for different num of nodes
            print(
                '*USER ELEMENT, NODES={}, TYPE=U{}, PROP={}, COORD=3, VARIABLES=216'
                .format(
                    int(shell_key) * num_layer_wdir, int(shell_key),
                    3 * num_layer_wdir),
                file=outF)

            tmplist = []
            for ii in range(0, num_layer_wdir):
                tmplist.append(ii + 1)
            for ii in range(int((len(tmplist) - 1) / 8) + 1):
                if ii < int((len(tmplist) - 1) / 8):
                    print(','.join(str(x) for x in (tmplist[ii * 8:(ii + 1) * 8])),
                      end=',',
                      file=outF)
                else:
                    print(','.join(str(x) for x in (tmplist[ii * 8:(ii + 1) * 8])),
                      end='',
                      file=outF)
                print('', file=outF)
            # print('1,2,3', file=outF)

            # ! Heading of UEL property
            print('*UEL PROPERTY, ELSET=uel{}'.format(int(shell_key)),
                  file=outF)
            tmplist = []
            for ii in range(0, num_layer_wdir * 3):
                tmplist.append(ii + 1)
            for ii in range(int((len(tmplist) - 1) / 8) + 1):
                if ii < int((len(tmplist) - 1) / 8):
                    print(','.join(
                        str(x) for x in (tmplist[ii * 8:(ii + 1) * 8])),
                          end=',',
                          file=outF)
                else:
                    print(','.join(
                        str(x) for x in (tmplist[ii * 8:(ii + 1) * 8])),
                          end='',
                          file=outF)
                print('', file=outF)

            # print(
            #     '6.825e7, 6.825e7, 6.825e7, 2.625e7, 2.625e7, 2.625e7, 0.3, 0.3,',
            #     file=outF)
            # print('0.3', file=outF)

            # ! Detail connectivity of each element
            print('*Element, type=U{}, ELSET=uel{}'.format(
                int(shell_key), int(shell_key)),
                  file=outF)
            for idx in bzmesh_shell_dict[shell_key]:
                tmp_output = [idx + 1]
                for idx_bzmesh in range(0, len(shift) - 1):
                    if idx >= shift[idx_bzmesh] and idx < shift[idx_bzmesh +
                                                                1]:
                        for nw in range(0, num_layer_wdir):
                            tmp_output = tmp_output + (bzmesh[idx_bzmesh][
                                idx - shift[idx_bzmesh] +
                                int(nw * num_bzmesh[idx_bzmesh])].IEN +
                                                       1).tolist()

                for ii in range(int((len(tmp_output) -1) / 16) + 1):
                    if ii < int((len(tmp_output) - 1) / 16):
                        print(','.join(
                            str(x)
                            for x in (tmp_output[ii * 16:(ii + 1) * 16])),
                              end=',',
                              file=outF)
                    else:
                        print(','.join(
                            str(x)
                            for x in (tmp_output[ii * 16:(ii + 1) * 16])),
                              end='',
                              file=outF)
                    print('', file=outF)
    # for j in range(0, len(bzmesh[0])):

    # for i in range(1, len(bzmesh)):
    #     for j in range(0, len(bzmesh[i])):

    print('*End Part', file=outF)
    print('**', file=outF)
    print('**', file=outF)
    print('** ASSEMBLY', file=outF)
    print('**', file=outF)
    print('*Assembly, name=Assembly', file=outF)
    print('**', file=outF)
    print('*Instance, name=Part-1-1, part=Part-1', file=outF)
    print('*End Instance', file=outF)
    print('**', file=outF)

    outF.close()