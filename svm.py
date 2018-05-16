# -*- coding:utf-8 -*-

import numpy as np
import random

class SVM:
    """
文件是一个0/1支持向量机, 翻译自Andrej Karpathy的svm.js, SVM用SMO算法进行训练
参考: "The Simplified SMO Algorithm" (http://math.unt.edu/~hsp0009/smo.pdf)
使用方法:
    svm = SVM()
    svm.train(data, labels)
    testlabels = svm.predict(testdata)
"""
    def __init__(self, options={}):
        self.options = options

    def _get_default(self, d, key, val):
        result = val
        try:
            result = d[key]
        except:
            pass
        return result

    # Kernels
    def _make_rbf_kernel(self, sigma):
        def f(v1, v2):
            s=0
            for q in range(v1.shape[0]):
                s += (v1[q] - v2[q])*(v1[q] - v2[q])             
            return np.exp(-s/(2.0*sigma*sigma))
        return f
  
    def linear_kernel(self, v1, v2):
        s = 0
        for q in range(v1.shape[0]):
            s += v1[q] * v2[q] 
        return s
        #return np.dot(v1, v2)


    # data is NxD array of floats. labels are 1 or -1.
    def train(self, data, labels):
        # we need these in helper functions
        self.data = data
        self.labels = labels
    
        # parameters
        options = self.options
        C = self._get_default(options, "C", 1.0) # C value. Decrease for more regularization
        tol = self._get_default(options, "tol", 1e-4) # numerical tolerance. Don't touch unless you're pro
        alphatol = self._get_default(options, "alphatol", 1e-7) # non-support vectors for space and time efficiency are truncated. To guarantee correct result set this to 0 to do no truncating. If you want to increase efficiency, experiment with setting this little higher, up to maybe 1e-4 or so.
        maxiter = self._get_default(options, "maxiter", 10000) # max number of iterations
        numpasses = self._get_default(options, "numpasses", 10) # how many passes over data with no change before we halt? Increase for more precision.
        
        # instantiate kernel according to options. kernel can be given as string or as a custom function
        kernel = self.linear_kernel
        self.kernelType = "linear"
      
        if "kernel" in options:
            if type(options["kernel"]) == type("str"):
            # kernel was specified as a string. Handle these special cases appropriately
                if options["kernel"] == "linear": 
                    self.kernelType = "linear"
                    kernel = self.linear_kernel
                if options["kernel"] == "rbf": 
                    rbfSigma = self._get_default(options, "rbfsigma", 0.5)
                    self.rbfSigma = rbfSigma # back this up
                    self.kernelType = "rbf"
                    kernel = self._make_rbf_kernel(rbfSigma)
            else:
                # assume kernel was specified as a function. Let's just use it
                self.kernelType = "custom"
                kernel = options["kernel"]


        # initializations
        self.kernel = kernel
        self.N, N = data.shape[0], data.shape[0]
        self.D, D = data.shape[1], data.shape[1]
        self.alpha = np.zeros(N)
        self.b = 0.0
        self.usew_ = False # internal efficiency flag

        # Cache kernel computations to avoid expensive recomputation.
        # This could use too much memory if N is large.
        self.kernelResults = np.zeros([N, N])
        if "memoize" in options and options["memoize"]:
            for i in range(N):
                for j in range(N):
                    self.kernelResults[i][j] = kernel(data[i], data[j])

        # run SMO algorithm
        passes, iter = 0, 0
        while (passes < numpasses and iter < maxiter):
            alphaChanged = 0
            for i in range(N):
                ## Calculate $E_i = f(x^{(i)})-y^{(i)}$ using (2)
                Ei = self.margin_one(data[i]) - labels[i]
                if( (labels[i]*Ei < -tol and self.alpha[i] < C) or 
                    (labels[i]*Ei >  tol and self.alpha[i] > 0)):
            
                    # alpha_i needs updating! Pick a j to update it with
                    j = i
                    ## select j != i randomly
                    while(j == i):
                        j= random.randint(0, self.N-1)
                    ## Calculate $E_j = f(x^{(j)})-y^{(j)}$ using (2)     
                    ## (2) $f(x) = \sum\limits_{i=1}^m{\alpha_iy^{(i)}(x^{(i)},x)+b}$
                    Ej= self.margin_one(data[j]) - labels[j]
            
                    # calculate L and H bounds for j to ensure we're in [0 C]x[0 C] box
                    ## Save old $\alpha$'s: $\alpha_i^{(old)} = \alpha_i, $\alpha_j^{(old)} = \alpha_j$ 
                    ai= self.alpha[i]
                    aj= self.alpha[j]
                    
                    ## Compute L and H by (10) and (11)
                    L, H = 0., C
                    if(labels[i] == labels[j]):
                        ## (11)
                        L, H = max(0, ai+aj-C), min(C, ai+aj)
                    else:
                        ## (10)
                        L, H = max(0, aj-ai), min(C, C+aj-ai)
                    
                    ## If L == H continue to next i
                    if(abs(L - H) < 1e-4):
                        continue
                    
                    ## Compute $\eta$ by (14), if $\eta$>=0 continue to next i
                    eta = 2 * self.kernel_result(i,j) - self.kernel_result(i,i) - self.kernel_result(j,j)
                    if(eta >= 0):
                        continue
        
                    # compute new alpha_j and clip it inside [0 C]x[0 C] box
                    # then compute alpha_i based on it.
                    ## compute and clip new value for $\alpha_j$ using (12) and (15).
                    ## (12) $\alpha_j := \alpha_j - \frac{y^{(i)}(E_i-E_j)}{\eta}$
                    newaj = aj - labels[j] * (Ei-Ej) / eta
                    if(newaj>H):
                        newaj = H
                    if(newaj<L):
                        newaj = L
                        
                    ## if $(|\alpha_j-\alpha_j^{(old)}| <10^{-5})$, continue to next i
                    if(abs(aj - newaj) < 1e-4):
                        continue 
                    
                    ## Determine value for $\alpha_i$ using (16).
                    ## $\alpha_i := \alpha_i + y^{(i)}y^{(j)}(\alpha_j^{(old)}-\alpha_j)$
                    self.alpha[j] = newaj
                    newai = ai + labels[i]*labels[j]*(aj - newaj)
                    self.alpha[i] = newai
        
                    # update the bias term
                    ## Compute $b_1$ and $b_2$ using (17) and (18) respectively
                    ## $b1 =b-E_i-y^{(i)}(\alpha_i-\alpha_i^{(old)})(x^{(i)},x^{(i)})-y^{(j)}(\alpha_j-\alpha_j^{(old)})(x^{(i)},x^{(j)})$
                    ## $b1 =b-E_i-y^{(i)}(\alpha_i-\alpha_i^{(old)})(x^{(i)},x^{(j)})-y^{(j)}(\alpha_j-\alpha_j^{(old)})(x^{(j)},x^{(j)})$
                    b1 = self.b - Ei - labels[i]*(newai-ai)*self.kernel_result(i,i) - labels[j]*(newaj-aj)*self.kernel_result(i,j)
                    b2 = self.b - Ej - labels[i]*(newai-ai)*self.kernel_result(i,j) - labels[j]*(newaj-aj)*self.kernel_result(j,j)
                    ## Compute b by (19)
                    self.b = 0.5*(b1+b2)
                    if(newai > 0 and newai < C):
                        self.b= b1
                    if(newaj > 0 and newaj < C):
                        self.b= b2
        
                    ## num changed alphas := num changed alphas + 1.
                    alphaChanged += 1
            
                # end alpha_i needed updating
            # end for i=1..N
        
            iter += 1
            #print("iter number %d, alphaChanged = %d", iter, alphaChanged);
            if(alphaChanged == 0):
                passes += 1
            else:
                passes= 0
        
        # end outer loop
      
        # if the user was using a linear kernel, lets also compute and store the
        # weights. This will speed up evaluations during testing time
        if(self.kernelType == "linear"):

            # compute weights and store them
            self.w = np.zeros(self.D)
            for j in range(self.D):
                s= 0.0
                for i in range(self.D):
                    s+= self.alpha[i] * labels[i] * data[i][j]
              
                self.w[j] = s
                self.usew_ = True
        else:

            # okay, we need to retain all the support vectors in the training data,
            # we can't just get away with computing the weights and throwing it out
    
            # But! We only need to store the support vectors for evaluation of testing
            # instances. So filter here based on self.alpha[i]. The training data
            # for which self.alpha[i] = 0 is irrelevant for future. 
            newdata = []
            newlabels = []
            newalpha = []
            for i in range(self.N):
                #print("alpha=%f", self.alpha[i]);
                if(self.alpha[i] > alphatol):
                    newdata.append(self.data[i])
                    newlabels.append(self.labels[i])
                    newalpha.append(self.alpha[i])
            
    
            # store data and labels
            self.data = np.array(newdata)
            self.labels = np.array(newlabels)
            self.alpha = np.array(newalpha)
            self.N = self.data.shape[0]
            #print("filtered training data from %d to %d support vectors.", data.length, self.data.length)
        
        trainstats = {}
        trainstats["iters"] = iter
        return trainstats
    
    
    # inst is an array of length D. Returns margin of given example
    # this is the core prediction function. All others are for convenience mostly
    # and end up calling this one somehow.
    def margin_one(self, inst):
        f = self.b
        # if the linear kernel was used and w was computed and stored,
        # (i.e. the svm has fully finished training)
        # the internal class variable usew_ will be set to True.
        if(self.usew_):
            # we can speed this up a lot by using the computed weights
            # we computed these during train(). This is significantly faster
            # than the version below
            for j in range(self.D):
                f += inst[j] * self.w[j]
        else:
            for i in range(self.N):
                f += self.alpha[i] * self.labels[i] * self.kernel(inst, self.data[i])
        return f
    
    def predict_one(self, inst):
        if (self.margin_one(inst) > 0):
            return 1
        else:
            return -1
    
    # data is an NxD array. Returns array of margins.
    def margins(self, data):
        # go over support vectors and accumulate the prediction. 
        N = data.shape[0]
        margins = np.zeros(N)
        for i in range(N):
            margins[i] = self.margin_one(data[i])
        
        return margins
      
    def kernel_result(self, i, j):
        if ("memoize" in self.options and 
            self.options["memoize"]):
            return self.kernelResults[i][j]
        return self.kernel(self.data[i], self.data[j])
    

    # data is NxD array. Returns array of 1 or -1, predictions
    def predict(self, data):
        margs = self.margins(data)
        for i in range(margs.shape[0]):
            if margs[i] > 0:
                margs[i] = 1
            else:
                margs[i] = -1
        return margs



if __name__ == "__main__":
    data = np.array([
        [1.05,1.05],
        [1,1.2],
        [1.1,2.1],
        [1.3,1.2],
        [5,6.0],
        [4.5,6.2]], dtype=np.float32)
    labels = np.array([1.,1.,1.,1.,-1.,-1.], dtype=np.float32)
    
    testdata = np.array([
        [1.05,1.05],
        [1,1.2],
        [1.1,2.1],
        [1.3,1.2],
        [5,6.0],
        [4.5,6.2]], dtype=np.float32)
    
    svm = SVM({"kernel":"rbf"})
    svm.train(data, labels)
    testlabels = svm.predict(testdata)
    print(testlabels)

    
