import numpy as np
import torch


class Point():
    coor = np.array([0., 0., 0.])
    List_edge = []
    List_face = []
    List_hex = []

    def __init__(self, idx, coor):
        self.idx = idx
        self.coor = coor


class Edge():
    cnct = np.array([0, 0])

    def __init__(self, cnct, bext_idx):
        self.cnct = cnct


class ElementQuad():
    cnct = np.array([0, 0, 0, 0])
    bext_idx = 0

    def __init__(self, cnct, bext_idx):
        self.cnct = cnct
        self.bext_idx = bext_idx


class ElementHex():
    cnct = np.array([0, 0, 0, 0, 0, 0, 0, 0])
    bext_idx = 0

    def __init__(self, cnct, bext_idx):
        self.cnct = cnct
        self.bext_idx = bext_idx


class BezierElement():
    def __init__(self, IEN, cmat, cpts_e ,ele_type):
        self.IEN = IEN
        self.cmat = torch.from_numpy(cmat)
        self.type = ele_type
        self.num_cpts = len(IEN)
        self.cpts_e = cpts_e


        if ele_type =='hex':
            self.Nx, self.dNdx, self.wght_all, self.detJ = self.IGA_basis_hex()
        if ele_type == 'quad':
            self.Nx, self.dNdx, self.wght_all, self.detJ = self.IGA_basis_quad()

    def IGA_basis_hex(self):
        # ! Gaussian quadrature
        Gpt = torch.tensor([0.06943184420297371, 0.33000947820757187, 0.6699905217924281,0.9305681557970262])
        wght = torch.tensor([0.3478548451374539, 0.6521451548625461, 0.6521451548625461, 0.3478548451374539])
        Nt = torch.zeros(64,64)
        dNdt = torch.zeros(64,3,64)
        wght_all = torch.zeros(64)

        count_int = 0
        for ii in range(4):
            for jj in range(4):
                for kk in range(4):
                    u_i = Gpt[ii]
                    v_i = Gpt[jj]
                    w_i = Gpt[kk]
                    Nu = torch.tensor([(1.-u_i)*(1.-u_i)*(1.-u_i), 3*u_i* (1.-u_i)*(1.-u_i), 3*u_i*u_i*(1.-u_i), u_i*u_i*u_i])
                    Nv = torch.tensor([(1.-v_i)*(1.-v_i)*(1.-v_i), 3*v_i* (1.-v_i)*(1.-v_i), 3*v_i*v_i*(1.-v_i), v_i*v_i*v_i])
                    Nw = torch.tensor([(1.-w_i)*(1.-w_i)*(1.-w_i), 3*w_i* (1.-w_i)*(1.-w_i), 3*w_i*w_i*(1.-w_i), w_i*w_i*w_i])
                    dNdu = torch.tensor([-3.*(1.-u_i)*(1.-u_i), 3.- 12.*u_i + 9.*u_i*u_i, 3.*(2. - 3.*u_i)*u_i, 3*u_i*u_i])
                    dNdv = torch.tensor([-3.*(1.-v_i)*(1.-v_i), 3.- 12.*v_i + 9.*v_i*v_i, 3.*(2. - 3.*v_i)*v_i, 3*v_i*v_i])
                    dNdw = torch.tensor([-3.*(1.-w_i)*(1.-w_i), 3.- 12.*w_i + 9.*w_i*w_i, 3.*(2. - 3.*w_i)*w_i, 3*w_i*w_i])
                    count_bz = 0
                    for i in range(4):
                        for j in range(4):
                            for k in range(4):
                                Nt[count_bz][count_int] = Nu[k]*Nv[j]*Nw[i]
                                dNdt[count_bz][0][count_int] = dNdu[k]*Nv[j]*Nw[i]
                                dNdt[count_bz][1][count_int] = Nu[k]*dNdv[j]*Nw[i]
                                dNdt[count_bz][2][count_int] = Nu[k]*Nv[j]*dNdw[i]
                                count_bz=count_bz+1
                    wght_all[count_int] = 0.125 * wght[ii] * wght[jj] * wght[kk]
                    count_int = count_int +1

        num_cpts = self.IEN.size
        dtdx = torch.zeros(3,3,64)
        detJ = torch.zeros(64)
        Nx = torch.matmul(self.cmat, Nt)
        dNdx = torch.matmul(self.cmat, dNdt.flatten(1)).reshape([-1,3,64])
        dxdt = torch.matmul(self.cpts_e.T,dNdx.flatten(1)).reshape([3,3,64])

        for i in range(64):
            dtdx[:,:,i] = torch.linalg.inv(dxdt[:,:,i].squeeze())
            # detJ[i] = torch.linalg.det(dtdx[:,:,i])
            detJ[i] = dtdx[:,:,i].det()

        self.cmat = []

        return Nx, dNdx, wght_all, detJ
    def IGA_basis_quad(self):
        # ! Gaussian quadrature
        Gpt = torch.tensor([0.06943184420297371, 0.33000947820757187, 0.6699905217924281,0.9305681557970262])
        wght = torch.tensor([0.3478548451374539, 0.6521451548625461, 0.6521451548625461, 0.3478548451374539])
        Nt = torch.zeros(16,16)
        dNdt = torch.zeros(16,2,16)
        wght_all = torch.zeros(16)

        count_int = 0
        for ii in range(4):
            for jj in range(4):
                u_i = Gpt[ii]
                v_i = Gpt[jj]
                Nu = torch.tensor([(1.-u_i)*(1.-u_i)*(1.-u_i), 3*u_i* (1.-u_i)*(1.-u_i), 3*u_i*u_i*(1.-u_i), u_i*u_i*u_i])
                Nv = torch.tensor([(1.-v_i)*(1.-v_i)*(1.-v_i), 3*v_i* (1.-v_i)*(1.-v_i), 3*v_i*v_i*(1.-v_i), v_i*v_i*v_i])
                dNdu = torch.tensor([-3.*(1.-u_i)*(1.-u_i), 3.- 12.*u_i + 9.*u_i*u_i, 3.*(2. - 3.*u_i)*u_i, 3*u_i*u_i])
                dNdv = torch.tensor([-3.*(1.-v_i)*(1.-v_i), 3.- 12.*v_i + 9.*v_i*v_i, 3.*(2. - 3.*v_i)*v_i, 3*v_i*v_i])
                count_bz = 0
                for j in range(4):
                    for k in range(4):
                        Nt[count_bz][count_int] = Nu[k]*Nv[j]
                        dNdt[count_bz][0][count_int] = dNdu[k]*Nv[j]
                        dNdt[count_bz][1][count_int] = Nu[k]*dNdv[j]
                        count_bz=count_bz+1
                wght_all[count_int] = 0.25 * wght[ii] * wght[jj]
                count_int = count_int +1

        num_cpts = self.IEN.size
        dtdx = torch.zeros(2,2,64)
        detJ = torch.zeros(64)
        Nx = torch.matmul(self.cmat, Nt)
        dNdx = torch.matmul(self.cmat, dNdt.flatten(1)).reshape([-1,2,64])
        dxdt = torch.matmul(self.cpts_e.T,dNdx.flatten(1)).reshape([2,2,64])

        for i in range(16):
            dtdx[:,:,i] = torch.linalg.inv(dxdt[:,:,i].squeeze())
            # detJ[i] = torch.linalg.det(dtdx[:,:,i])
            detJ[i] = dtdx[:,:,i].det()

        self.cmat = []

        return Nx, dNdx, wght_all, detJ

class ControlElement():
    def __init__(self, cnct):
        self.cnct = cnct
        # self.type = ele_type
