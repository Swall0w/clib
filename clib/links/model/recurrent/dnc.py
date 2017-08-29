import chainer
import numpy as np
from chainer import functions as F
from chainer import links as L
from chainer import Chain, Link, Variable, optimizers


# controller of DNC
class SimpleLSTM(Chain): 
    def __init__(self, d_in, d_hidden, d_out):
        super(SimpleLSTM, self).__init__()
        with self.init_scope():
            self.l1 = L.LSTM(d_in, d_hidden)
            self.l2 = L.Linear(d_hidden, d_out)

    def __call__(self, x):
        return self.l2(self.l1(x))

    def reset_state(self):
        self.l1.reset_state()


class DNC(Chain):

    def __init__(self, X, Y, N, W, R):
        self.X = X # input dimension
        self.Y = Y # output dimension
        self.N = N # number of memory slot
        self.W = W # dimension of one memory slot
        self.R = R # number of read heads
        self.d_ctr_in = W*R+X # input dimension into the controller
        self.d_ctr_out = Y+W*R+3*W+5*R+3 # output dimension from the controller
        self.d_ctr_hidden = self.d_ctr_out # dimension of hidden unit of the controller
        self.d_interface = W*R+3*W+5*R+3 # dimension of interface vector

        self.controller = SimpleLSTM(self.d_ctr_in, self.d_ctr_hidden, self.d_ctr_out)

        super(DNC, self).__init__(
                l_ctr = self.controller, 
                l_Wy = L.Linear(self.d_ctr_out, self.Y),
                l_Wxi = L.Linear(self.d_ctr_out, self.d_interface), 
                l_Wr = L.Linear(self.R * self.W, self.Y), 
                )

        self.reset_state()

    def reset_state(self):
# initialize all the recurrent state
        self.l_ctr.reset_state() # controller
        self.u = Variable(np.zeros((self.N, 1)).astype(np.float32)) # usage vector (N, 1)
        self.p = Variable(np.zeros((self.N, 1)).astype(np.float32)) # precedence weighting (N, 1)
        self.L = Variable(np.zeros((self.N, self.N)).astype(np.float32)) # temporal memory linkage (N, N)                     
        self.Mem = Variable(np.zeros((self.N, self.W)).astype(np.float32)) # memory (N, W)
        self.r = Variable(np.zeros((1, self.R*self.W)).astype(np.float32)) # read vector (1, R * W)
        self.wr = Variable(np.zeros((self.N, self.R)).astype(np.float32)) # read weighting (N, R)
        self.ww = Variable(np.zeros((self.N, 1)).astype(np.float32)) # write weighting (N, 1)


# utility functions

    def _cosine_similarity(self, u, v):
# cosine similarity as a distance of two vectors
# u, v: (1, -) Variable  -> (1, 1) Variable
        denominator = F.sqrt(F.batch_l2_norm_squared(u) * F.batch_l2_norm_squared(v))
        if (np.array_equal(denominator.data, np.array([0]))):
            return F.matmul(u, F.transpose(v))
        return F.matmul(u, F.transpose(v)) / F.reshape(denominator, (1, 1))


    def _C(self, Mem, k, beta):
# similarity between rows of matrix Mem and vector k
# Mem:(N, W) Variable, k:(1, W) Variable, beta:(1, 1) Variable -> (N, 1) Variable
        N, W = Mem.shape 
        ret_list = [0] * N
        for i in range(N):
# calculate distance between i-th row of Mem and k
            ret_list[i] = self._cosine_similarity(F.reshape(Mem[i,:], (1, W)), k) * beta 
# concat horizontally because softmax operates along the direction of axis=1 
        return F.transpose(F.softmax(F.concat(ret_list, 1))) 


    def _u2a(self, u):
# convert usage vector u to allocation weighting a
# u, a: (N, 1) Variable
        N = u.shape[0]
        phi = np.argsort(u.data.flatten()) # u.data[phi]: ascending
        a_list = [0] * N    
        cumprod = Variable(np.array([[1.0]]).astype(np.float32)) 
        for i in range(N): 
            a_list[phi[i]] = cumprod * (1.0 - F.reshape(u[phi[i]], (1, 1)))
            cumprod *= F.reshape(u[phi[i]], (1, 1))
        return F.concat(a_list, 0) 


# operations of the DNC system

    def _controller_io(self, x):
# input data from the Data Set : x (1, X) Variable  
# out-put from the controller h is split into two ways : v (1, Y), xi(1, W*R+3*W+5*R+3) Variable
        chi = F.concat([x, self.r], 1) # total input to the controller 
        h = self.l_ctr(chi) # total out-put from the controller
        self.v = self.l_Wy(h)
        self.xi = self.l_Wxi(h)

# interface vector xi is split into several components
        (self.kr, self.beta_r, self.kw, self.beta_w,
         self.e, self.nu, self.f, self.ga, self.gw, self.pi
         ) = F.split_axis(self.xi, np.cumsum(
            [self.W*self.R, self.R, self.W, 1, self.W, self.W, self.R, 1, 1]), 1) # details of the interface vector

# rescale components
        self.kr = F.reshape(self.kr, (self.R, self.W)) # read key (R, W)
        self.beta_r = 1 + F.softplus(self.beta_r) # read strength (1, R)
# self.kw : write key (1, W)
        self.beta_w = 1 + F.softplus(self.beta_w) # write strength (1, 1)
        self.e = F.sigmoid(self.e) # erase vector (1, W)
# self.nu : write vector (1, W)
        self.f = F.sigmoid(self.f) #  free gate (1, R)
        self.ga = F.sigmoid(self.ga) # allcation gate (1, 1)
        self.gw = F.sigmoid(self.gw) # write gate (1, 1)
        self.pi = F.softmax(F.reshape(self.pi, (self.R, 3))) # read mode (R, 3)


    def _up_date_write_weighting(self):
# calculate retention vector : psi (N, 1) 
# here, read weighting : wr (N, R) must retain state one step former
        psi_mat = 1 - F.matmul(Variable(np.ones((self.N, 1)).astype(np.float32)), self.f) * self.wr # (N, R)
        self.psi = Variable(np.ones((self.N, 1)).astype(np.float32)) 
        for i in range(self.R):
            self.psi = self.psi * F.reshape(psi_mat[:,i],(self.N,1)) # (N, 1)
# up date usage vector : u (N, 1)
# here, write weighting : ww (N, 1) must retain state one step former
        self.u = (self.u + self.ww - (self.u * self.ww)) * self.psi 
# calculate allocation weighting : a (N, 1)
        self.a = self._u2a(self.u) 
# calculate write content weighting : cw (N, 1)
        self.cw = self._C(self.Mem, self.kw, self.beta_w) 
# up date write weighting : ww (N, 1)
        self.ww = F.matmul(F.matmul(self.a, self.ga) + F.matmul(self.cw, 1.0 - self.ga), self.gw) 


    def _write_to_memory(self):
# erase vector : e (1, W) deletes information on the Memory : Mem (N, W) 
# and write vector : nu (1, W) is written there
# write weighting : ww (N, 1) must be up-dated before this step
        self.Mem = self.Mem * (np.ones((self.N, self.W)).astype(np.float32) - F.matmul(self.ww, self.e)) + F.matmul(self.ww, self.nu)


    def _up_date_read_weighting(self):    
# up date temporal memory linkage : L (N, N)
        ww_mat = F.matmul(self.ww, Variable(np.ones((1, self.N)).astype(np.float32))) # (N, N)
# here, precedence wighting : p (N, 1) must retain state one step former
        self.L = (1.0 - ww_mat - F.transpose(ww_mat)) * self.L + F.matmul(self.ww, F.transpose(self.p)) # (N, N)
        self.L = self.L * (np.ones((self.N, self.N)) - np.eye(self.N)) # constrain L[i,i] == 0   
# up date prcedence weighting : p (N, 1)
        self.p = (1.0 - F.matmul(Variable(np.ones((self.N, 1)).astype(np.float32)), F.reshape(F.sum(self.ww),(1, 1)))) * self.p + self.ww 
# calculate forward weighting : fw (N, R)
# here, read wighting : wr (N, R) must retain state one step former
        self.fw = F.matmul(self.L, self.wr) 
# calculate backward weighting : bw (N, R)
        self.bw = F.matmul(F.transpose(self.L), self.wr)
# calculate read content weighting : cr (N, R)
        self.cr_list = [0] * self.R
        for i in range(self.R):
            self.cr_list[i] = self._C(self.Mem, F.reshape(self.kr[i,:], (1, self.W)), F.reshape(self.beta_r[0,i],(1, 1))) # (N, 1)
        self.cr = F.concat(self.cr_list, 1) # (1, N * R)       
# compose up-dated read weighting : wr (N, R)
        bcf_tensor = F.concat([
            F.reshape(F.transpose(self.bw), (self.R, self.N, 1)),
            F.reshape(F.transpose(self.cr), (self.R, self.N, 1)),
            F.reshape(F.transpose(self.fw), (self.R, self.N, 1))
            ], 2) # (R, N, 3)
        self.pi = F.reshape(self.pi, (self.R, 3, 1)) # (R, 3, 1)
        self.wr = F.transpose(F.reshape(F.batch_matmul(bcf_tensor, self.pi), (self.R, self.N))) # (N, R)


    def _read_from_memory(self): 
# read information from the memory : Mem (N, W) and compose read vector : r (W, R) to reshape (1, W * R)
# read weighting : wr (N, R) must be up-dated before this step
        self.r = F.reshape(F.matmul(F.transpose(self.Mem), self.wr), (1, self.R * self.W))


    def __call__(self, x):
        self._controller_io(x) # input data is processed through the controller
        self._up_date_write_weighting() 
        self._write_to_memory() # memory up-date
        self._up_date_read_weighting()  
        self._read_from_memory() # extract information from the memory
        self.y = self.l_Wr(self.r) + self.v # compose total out put y : (1, Y)
        return self.y
