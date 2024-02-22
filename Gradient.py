import numpy as np

class GradientDescent:
    def __init__(self, learning_rate=0.01, max_iters=1000, tol=1e-6):
        self.learning_rate = learning_rate
        self.max_iters = max_iters
        self.tol = tol
        self.theta = np.random.rand(3,1)
        self.history = {'loss': [], 'theta': []}

    def _compute_gradient(self, x, y, theta):
        new_x = np.insert(x, 0,1,axis = 1)   
        y_pred = np.dot(new_x,self.theta)
        gradient = np.zeros_like(self.theta,dtype = float)

        for i in range(self.theta.shape[0]):
            if i == 0:
                gradient[0] = 2 * np.mean(y_pred - y)
            else:
                gradient[i] = 2 * np.dot(new_x[:,i],(y_pred - y)) / self.theta.shape[0]
        return gradient

    def _compute_loss(self, x, y, theta):
        self.x = x
        self.y = y
        self.theta = theta
        new_x = np.insert(x, 0,1,axis = 1)   
        y_pred = np.dot(new_x,self.theta)    
        loss = np.mean((y_pred - y)**2)
        return loss

    def fit(self, x, y):
        self.x = x
        self.y = y
        for i in range(self.max_iters):
            if self._compute_loss(self.x,self.y,self.theta) < self.tol:
                break
            self.history['loss'].append(self._compute_loss(self.x,self.y,self.theta))
            self.history['theta'].append(self.theta)
            gradient = self._compute_gradient(self.x,self.y,self.theta)
            self.theta = self.theta - self.learning_rate * gradient
n = int(input("Number of data points: "))
X = np.random.randn(n, 2)
y = np.random.randn(n, 1)

a = GradientDescent()
a.fit(X, y)
print(a.history)
