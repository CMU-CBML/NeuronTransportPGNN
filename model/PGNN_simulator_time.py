# import torch

# import torch.nn.init as init
from multiprocessing import reduction
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops

from torch_geometric.nn import (
    SplineConv,
    SAGEConv,
    GCNConv,
    MetaLayer,
)
from mesh.datastructure import *
from mesh.MeshIO import *


# from blocks import EdgeBlock, NodeBlock, GlobalBlock


class EdgeModel(torch.nn.Module):
    def __init__(self, net=None):
        super(EdgeModel, self).__init__()
        self.net = net

    def _forward_one_net(self, net, src, dest, edge_attr):
        disp_r = dest[:, :3] - src[:, :3]
        f_src = src[:, -1].reshape((-1, 1))
        f_dest = dest[:, -1].reshape((-1, 1))
        # f_r = dest[:, -1] - src[:, -1]
        # f_r = f_r.reshape((-1, 1))
        norm_r = torch.norm(disp_r, dim=-1).reshape((-1, 1))

        # print()
        # print(edge_attr.shape)

        net_in = torch.cat([disp_r, norm_r, edge_attr, f_src, f_dest], dim=-1)
        # print(net_in.shape)
        # net_in = torch.cat([net_in, edge_attr], dim=-1)
        # print(net_in.shape)
        out = net(net_in)
        out = edge_attr + out

        return out

    def forward(self, src, dest, edge_attr, u, batch):
        return self._forward_one_net(self.net, src, dest, edge_attr)


class NodeModel(MessagePassing):
    def __init__(self, net=None, aggr="mean", flow="source_to_target"):
        super(NodeModel, self).__init__(aggr=aggr)
        self.net = net

    def _message_one_net(self, net, x_i, x_j, edge_attr):
        return edge_attr

    def message(self, x_i, x_j, edge_attr):
        return self._message_one_net(self.net, x_i, x_j, edge_attr)

    def update(self, aggr_out, x):
        # net_input = torch.cat((x[:, -1].unsqueeze(-1), aggr_out), dim=-1)
        net_input = torch.cat((x[:, -2:], aggr_out), dim=-1)
        net_out = self.net(net_input)
        out = x
        out[:, -1] = out[:, -1] + net_out.squeeze()

        return out

    def forward(self, x, edge_index, edge_attr, u, batch):
        return self.propagate(
            edge_index, size=(x.size(0), x.size(0)), x=x, edge_attr=edge_attr
        )

# ! GN simulator
class Simulator(torch.nn.Module):
    def __init__(self, dataset, num_layers, num_hidden):
        super(Simulator, self).__init__()

        self.num_features = dataset.feature.shape[-1]
        self.num_targets = dataset.target.shape[-1]

        # ! New GN
        self.mlp1 = nn.Sequential(
            nn.Linear(5, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, 5),
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(self.num_features + 2, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, self.num_features),
        )

        # self.eb = EdgeModel(net=self.mlp1)
        self.eb = EdgeModel()
        # self.eb = EdgeGradientLayer()
        self.nb = NodeModel(net=self.mlp2)
        self.gn = MetaLayer(edge_model=self.eb, node_model=self.nb)

        self.processors = torch.nn.ModuleList()

        # Shared parameter
        for i in range(num_layers - 1):
            self.processors.append(self.gn)

        # Not Shared parameter
        # for i in range(num_layers - 1):
        #    self.processors.append(MetaLayer(edge_model=EdgeModel(), node_model=NodeModel(net=self.mlp2)))

        self.decoder = nn.Sequential(
            nn.Linear(self.num_features, num_hidden),
            nn.ReLU(True),
            nn.Linear(num_hidden, num_hidden),
            nn.ReLU(True),
            nn.Linear(num_hidden, num_hidden),
            nn.ReLU(True),
            nn.Linear(num_hidden, self.num_targets),
        )

    def forward(self, data, mode):
        # x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x, edge_index = data.x, data.edge_index

        # print(edge_attr.shape)

        if mode == 1:
            # ! New GEN + Decoder

            for processor in self.processors:
                x_res = processor(x, edge_index)[0]
                x = x + F.relu(x_res)
            out = self.decoder(x)

        return out
        # return x


# ! PGNN Simulator change init
class SimulatorModel(torch.nn.Module):
    def __init__(self, num_features, num_targets, num_layers, num_hidden):
        # def __init__(self):
        super(SimulatorModel, self).__init__()

        self.num_features = num_features
        self.num_targets = num_targets

        # Gaussian quadrature
        # self.Gpt = torch.tensor([0.06943184420297371, 0.33000947820757187, 0.6699905217924281,0.9305681557970262])
        # self.wght = torch.tensor([0.3478548451374539, 0.6521451548625461, 0.6521451548625461, 0.3478548451374539])
        # self.Nt = torch.zeros(64,64)
        # self.dNdt = torch.zeros(64,3,64)
        # self.wght_all = torch.zeros(64)
        # self.IGA_basis()

        # ! Parameter
        self.D = 1.0
        self.k_attach = 1.0
        self.k_detach = 0.1
        self.mu = 0.1
        self.l = 1.0
        self.force = 0.0

       
        # ! New GN
        self.eb = EdgeModel(
            net=nn.Sequential(
                nn.Linear(3 + 6, num_hidden),
                nn.ReLU(),
                nn.Linear(num_hidden, num_hidden),
                nn.ReLU(),
                nn.Linear(num_hidden, 3),
            )
        )
        self.nb = NodeModel(
            net=nn.Sequential(
                nn.Linear(5, num_hidden),
                nn.ReLU(),
                nn.Linear(num_hidden, num_hidden),
                nn.ReLU(),
                nn.Linear(num_hidden, 1),
            )
        )

        # self.eb = EdgeModel(
        #     net=nn.Sequential(
        #         nn.Linear(3 + 6, num_hidden),
        #         nn.LayerNorm(num_hidden),
        #         nn.ReLU(),
        #         nn.Linear(num_hidden, num_hidden),
        #         nn.LayerNorm(num_hidden),
        #         nn.ReLU(),
        #         nn.Linear(num_hidden, 3),
        #     )
        # )
        # self.nb = NodeModel(
        #     net=nn.Sequential(
        #         nn.Linear(5, num_hidden),
        #         nn.LayerNorm(num_hidden),
        #         nn.ReLU(),
        #         nn.Linear(num_hidden, num_hidden),
        #         nn.LayerNorm(num_hidden),
        #         nn.ReLU(),
        #         nn.Linear(num_hidden, 1),
        #     )
        # )
        self.gn = MetaLayer(edge_model=self.eb, node_model=self.nb)

        self.processors = torch.nn.ModuleList()

        # Shared parameter
        for i in range(num_layers - 1):
            self.processors.append(MetaLayer(edge_model=self.eb, node_model=self.nb))

        # Not Shared parameter
        # for i in range(num_layers - 1):
        #    self.processors.append(MetaLayer(edge_model=EdgeModel(), node_model=NodeModel(net=self.mlp2)))

        self.decoder = nn.Sequential(
            nn.Linear(self.num_features - 1, num_hidden),
            nn.ReLU(True),
            nn.Linear(num_hidden, num_hidden),
            nn.ReLU(True),
            nn.Linear(num_hidden, num_hidden),
            nn.ReLU(True),
            nn.Linear(num_hidden, self.num_targets),
        )

    def _compute_edge_attr(self, x, edge_index):
        x_src = x[edge_index[0, :], :]
        x_dest = x[edge_index[1, :], :]

        disp_r = x_dest[:, :3] - x_src[:, :3]
        f_r = x_dest[:, -1] - x_src[:, -1]
        f_r = f_r.reshape((-1, 1))
        f_r3 = torch.cat((f_r, f_r, f_r), dim=-1)

        norm_r = torch.norm(disp_r, dim=-1).reshape((-1, 1))
        grad_r = torch.mul(f_r3, disp_r)
        edge_attr = grad_r


        return edge_attr

    def forward(self, X_curr, edge, y_prev, mode):
        # x, edge_index, edge_attr = X_k.x, X_k.edge_index, X_k.edge_attr
        x, edge_index = X_curr, edge

        x_in = torch.cat((x[:, [0, 1, 2, 4]], y_prev.unsqueeze(-1)), dim=-1)
        # x_in = torch.cat((x[:, [0, 1, 2]], y_prev.unsqueeze(-1)), dim=-1)
        # x_in = torch.cat((x, y_prev.unsqueeze(-1)), dim=-1)
        edge_attr = self._compute_edge_attr(x_in, edge_index)
        for processor in self.processors:
            x_in, edge_attr, _ = processor(x_in, edge_index, edge_attr)
        x_out = self.decoder(x_in)

        out = y_prev + x_out.squeeze()

        return out

    # def IGA_basis(self):
    #     # ! Gaussian quadrature
    #     Gpt = torch.tensor([0.06943184420297371, 0.33000947820757187, 0.6699905217924281,0.9305681557970262])
    #     wght = torch.tensor([0.3478548451374539, 0.6521451548625461, 0.6521451548625461, 0.3478548451374539])
    #     # self.Nt = torch.zeros(64,64)
    #     # self.dNdt = torch.zeros(64,3,64)
    #     # self.wght_all = torch.zeros(64)

    #     count_int = 0
    #     for ii in range(4):
    #         for jj in range(4):
    #             for kk in range(4):
    #                 u_i = Gpt[ii]
    #                 v_i = Gpt[jj]
    #                 w_i = Gpt[kk]
    #                 Nu = torch.tensor([(1.-u_i)*(1.-u_i)*(1.-u_i), 3*u_i* (1.-u_i)*(1.-u_i), 3*u_i*u_i*(1.-u_i), u_i*u_i*u_i])
    #                 Nv = torch.tensor([(1.-v_i)*(1.-v_i)*(1.-v_i), 3*v_i* (1.-v_i)*(1.-v_i), 3*v_i*v_i*(1.-v_i), v_i*v_i*v_i])
    #                 Nw = torch.tensor([(1.-w_i)*(1.-w_i)*(1.-w_i), 3*w_i* (1.-w_i)*(1.-w_i), 3*w_i*w_i*(1.-w_i), w_i*w_i*w_i])
    #                 dNdu = torch.tensor([-3.*(1.-u_i)*(1.-u_i), 3.- 12.*u_i + 9.*u_i*u_i, 3.*(2. - 3.*u_i)*u_i, 3*u_i*u_i])
    #                 dNdv = torch.tensor([-3.*(1.-v_i)*(1.-v_i), 3.- 12.*v_i + 9.*v_i*v_i, 3.*(2. - 3.*v_i)*v_i, 3*v_i*v_i])
    #                 dNdw = torch.tensor([-3.*(1.-w_i)*(1.-w_i), 3.- 12.*w_i + 9.*w_i*w_i, 3.*(2. - 3.*w_i)*w_i, 3*w_i*w_i])
    #                 count_bz = 0
    #                 for i in range(4):
    #                     for j in range(4):
    #                         for k in range(4):
    #                             self.Nt[count_bz][count_int] = Nu[k]*Nv[j]*Nw[i]
    #                             self.dNdt[count_bz][0][count_int] = dNdu[k]*Nv[j]*Nw[i]
    #                             self.dNdt[count_bz][1][count_int] = Nu[k]*dNdv[j]*Nw[i]
    #                             self.dNdt[count_bz][2][count_int] = Nu[k]*Nv[j]*dNdw[i]
    #                             count_bz=count_bz+1
    #                 self.wght_all[count_int] = 0.125 * wght[ii] * wght[jj] * wght[kk]
    #                 count_int = count_int +1

    def loss_cpt(self, y_pred, y_target):
        eta = 1e-4
        # predict_out = self.forward(X_curr, edge, y_prev, mode)
        num_pt = y_pred.size(dim=0)

        n_0 = y_target[:,0]
        n_plus = y_target[:, 1]
        v_plus = y_target[:, 2:]

        n_0_predict = y_pred[:, 0]
        n_plus_predict = y_pred[:, 1]
        v_plus_predict = y_pred[:, 2:]

        v_plus_dir = torch.div(v_plus.T, torch.norm(v_plus, dim=1)+eta).T
        v_plus_pred_dir = torch.div(v_plus_predict.T, torch.norm(v_plus_predict, dim=1)+eta).T

        mse = torch.nn.MSELoss(reduction='sum')
        mse_val = (mse(n_0, n_0_predict) + mse(n_plus, n_plus_predict) + mse(v_plus, v_plus_predict))/num_pt
        mse_vel_dir = mse(v_plus_dir, v_plus_pred_dir)
        mse_predict = mse_val + mse_vel_dir
        return mse_predict

    def loss_equation(self, bzmesh, y_pred):
        num_equation = 4

        n_0 = y_pred[:,0]
        n_plus = y_pred[:, 1]
        v_plus = y_pred[:, 2:]

        num_cpt = y_pred.size(dim=0)

        Re = torch.zeros(num_cpt, 5)

        for e in range(len(bzmesh)):
            ele = bzmesh[e]
            IEN = ele.IEN

            num_ele_cpt = IEN.size

            n_0_ele = n_0[IEN[:]]
            n_plus_ele = n_plus[IEN[:]]
            v_plus_ele =v_plus[IEN[:], :]
            
            
            Re_ele = torch.zeros(num_ele_cpt, 5)

            Nx = ele.Nx
            dNdx = ele.dNdx
            wght_all = ele.wght_all
            detJ = ele.detJ

            n_0_int = torch.matmul(n_0_ele, Nx)
            n_plus_int = torch.matmul(n_plus_ele, Nx)
            v_plus_int =torch.matmul(v_plus_ele.T, Nx)

            n_0_der1_int = torch.matmul(n_0_ele, dNdx.view(num_ele_cpt, -1)).reshape([3,64])
            n_plus_der1_int = torch.matmul(n_plus_ele, dNdx.view(num_ele_cpt, -1)).reshape([3,64])
            v_plus_der1_int =torch.matmul(v_plus_ele.T, dNdx.view(num_ele_cpt, -1)).T.reshape([3,3,-1])
            
            for i in range (64):
                for j in range(num_ele_cpt):
                    Re_ele[j, 0] += wght_all[i] * Nx[j,i] * ((1+0) * n_0_int[i] - 0.5*n_plus_int[i])*detJ[i]
                    Re_ele[j, 1] += wght_all[i] * (Nx[j,i] * ((v_plus_int[0,i]*n_0_der1_int[0,i]+v_plus_int[1,i]*n_0_der1_int[1,i]+v_plus_int[2,i]*n_0_der1_int[2,i]) - 1.0 *n_0_int[i] + 0.5*n_plus_int[i])+1.0*(dNdx[j,0,i] * n_0_der1_int[0,i]+dNdx[j,1,i] * n_0_der1_int[1,i]+dNdx[j,2,i] * n_0_der1_int[2,i]))*detJ[i]
                    Re_ele[j, 2] += wght_all[i] * (Nx[j,i] * (v_plus_int[0,i]*v_plus_der1_int[0,0,i] + v_plus_int[1,i]*v_plus_der1_int[1,0,i] + v_plus_int[2,i]*v_plus_der1_int[2,0,i]) + Nx[j,i] * n_plus_der1_int[0,i] + 1.0*(dNdx[j,0,i] * v_plus_der1_int[0,0,i]+dNdx[j,1,i] * v_plus_der1_int[1,0,i]+dNdx[j,2,i] * v_plus_der1_int[2,0,i]))*detJ[i]
                    Re_ele[j, 3] += wght_all[i] * (Nx[j,i] * (v_plus_int[0,i]*v_plus_der1_int[0,1,i] + v_plus_int[1,i]*v_plus_der1_int[1,1,i] + v_plus_int[2,i]*v_plus_der1_int[2,1,i]) + Nx[j,i] * n_plus_der1_int[1,i] + 1.0*(dNdx[j,0,i] * v_plus_der1_int[0,1,i]+dNdx[j,1,i] * v_plus_der1_int[1,1,i]+dNdx[j,2,i] * v_plus_der1_int[2,1,i]))*detJ[i]
                    Re_ele[j, 4] += wght_all[i] * (Nx[j,i] * (v_plus_int[0,i]*v_plus_der1_int[0,2,i] + v_plus_int[1,i]*v_plus_der1_int[1,2,i] + v_plus_int[2,i]*v_plus_der1_int[2,2,i]) + Nx[j,i] * n_plus_der1_int[2,i] + 1.0*(dNdx[j,0,i] * v_plus_der1_int[0,2,i]+dNdx[j,1,i] * v_plus_der1_int[1,2,i]+dNdx[j,2,i] * v_plus_der1_int[2,2,i]))*detJ[i]
            for i in range(5):
                Re[IEN[:],i] += Re_ele[:,i]
        
        mse_predict = 0
        for i in range(5):
            mse_predict += torch.norm(Re[:,i])
        
        return mse_predict



