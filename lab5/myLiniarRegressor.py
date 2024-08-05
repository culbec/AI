class MyLiniarRegressor:
    def __init__(self):
        self.intercept_ = 0.0
        self.coef_ = []
    
    def __transpose(self, matrix):
        '''
            Transposes a given matrix.
            
            @param matrix: Matrix to transpose.
        '''
        return [[matrix[j][i] for j in range(len(matrix))] for i in range(len(matrix[0]))]
   
    def __eliminate(self, r1, r2, col, target=0):    
        fac = (r2[col] - target) / r1[col]
        
        for i in range(len(r2)):
            r2[i] -= fac * r1[i]
            
    def __gauss(self, matrix):
        '''
            Performs Gauss-Jordan elimination on a matrix.
            
            @param matrix: Matrix to perform the Gauss-Jordan elimination on.
        '''
        
        for i in range(len(matrix)):
            if matrix[i][i] == 0:
                for j in range(i + 1, len(matrix)):
                    if matrix[i][j] != 0:
                        matrix[i], matrix[j] = matrix[j], matrix[i]
                        break
                    else:
                        raise ValueError('Matrix is not invertible')
            for j in range(i+1, len(matrix)):
                self.__eliminate(matrix[i], matrix[j], i)
        for i in range(len(matrix)-1, -1, -1):
            for j in range(i - 1, -1, -1):
                self.__eliminate(matrix[i], matrix[j], i)
        for i in range(len(matrix)):
            self.__eliminate(matrix[i], matrix[i], i, target=1)
        
        return matrix

    def __inverse(self, matrix):
        '''
            Computes the inverse of a matrix.
            
            @param matrix: Matrix to inverse.
        '''
        
        tmp = [[] for _ in matrix]
        
        for i, row in enumerate(matrix):
            assert len(row) == len(matrix)
            tmp[i].extend(row + [0] * i + [1] + [0] * (len(matrix) - i - 1))
            
        self.__gauss(tmp)
        return [tmp[i][len(tmp[i])//2:] for i in range(len(tmp))]    
   
    def __dot_product(self, m1, m2):
        '''
            Performs the dot product on two matrices.
            
            @param m1: First matrix.
            @param m2: Second matrix.
        '''
        rows_m1, cols_m1 = len(m1), len(m1[0])
        rows_m2, cols_m2 = len(m2), len(m2[0])

        #check if the matrices can be multiplied
        if cols_m1 != rows_m2:
                raise ValueError("Cannot multiply matrices:  incompatible dimensions.")
        #create the result matrix 
        c =[[0 for _ in range(cols_m2)] for _ in range(rows_m1)]
        for i in range(rows_m1):
            for j in range(cols_m2):
                for k in range(cols_m1):
                    c[i][j] += m1[i][k] * m2[k][j]
        return c 
    
    def fit(self, x, y):
        '''
            Determines the intercept and coefficients of a linear regression.
            
            @param x: Input data.
            @param y: Output data.
        '''
        x = [[1] + row for row in x]
        yy = [[y[i]] for i in range(len(y))]
        
        xT = self.__transpose(x)
        xTx = self.__dot_product(xT, x)
        xTx_inverse = self.__inverse(xTx)
        xTx_inverse_xT = self.__dot_product(xTx_inverse, xT)
        weights = self.__dot_product(xTx_inverse_xT, yy)
        weights = [weights[i][0] for i in range(len(weights))]
        
        self.intercept_ = weights[0]
        self.coef_ = weights[1:]
        
    def predict(self, x):
        '''
            Predicts outputs based on the given inputs.
            
            @param x: Inputs given.
        '''
        return [self.intercept_ + sum(self.coef_[i] * val[i] for i in range(len(val))) for val in x]