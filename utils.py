import torch
import numpy as np
import torch.nn.functional as F
import torch.utils.data as data_utils
from sklearn.metrics import auc
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch

Scaler = StandardScaler()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def add_iqr_annotation(ax, Q1, Q2, Q3, w_position=0.85, W_position=0.85, width=0.05, color='k', label_w='IQR', label_W='Wp', fontsize=11):
    arrow_style = '<|-|>'
    arrow = FancyArrowPatch((w_position, Q1), (w_position, Q3), 
                            arrowstyle=arrow_style, color='black', lw=1.5, 
                            mutation_scale=10)
    ax.add_patch(arrow)
    ax.add_line(Line2D([w_position - width/2, w_position + width/2], [Q1, Q1], color='black', lw=1.5))
    ax.add_line(Line2D([w_position - width/2, w_position + width/2], [Q3, Q3], color='black', lw=1.5))
    ax.text(w_position, (Q1 + Q3) / 2, label_w, ha='center', va='center', fontsize=fontsize, color=color,
            bbox=dict(facecolor='white', edgecolor='white'))  
    ax.text(W_position, Q2+15, label_W, ha='center', va='center', fontsize=fontsize-1, color='k')


# Exponential growth model (EGM)
def growth_func(age, a, b):
    return a * np.exp(b * age)

# Pinball loss for quantile regression、 (Koenker & Bassett, 1978)
def tilted_loss(q, y, f):
    e = y - f
    return torch.mean(torch.max(q * e, (q - 1) * e))

# Convert categorical maturity status to binary numerical values
def maturity_01(value):
    if value == 'M':
        return 1
    elif value == 'I':
        return 0
    else:
        raise ValueError(f"Invalid maturity status: {value}. Expected 'M' or 'I'")


# Deep quantile growth model (deep QGM) architecture
class DeepQModel(nn.Module):
    def __init__(self, ndim=1, qs=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]):
        super(DeepQModel, self).__init__()
        self.fc1 = nn.Linear(ndim, 20)
        self.fc2 = nn.Linear(20, 10)
        self.outputs = nn.ModuleList([nn.Linear(10, 1) for _ in qs])
        self.qs = qs

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        outs = [output_layer(x) for output_layer in self.outputs]
        return outs

# Train deep QGM
def train_DeepQModel(x, y, batch_size=16, epochs=200):
    dataset = TensorDataset(torch.tensor(x).float(), torch.tensor(y).float())
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = DeepQModel()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    for epoch in range(epochs):
        total_loss = 0.0
        for inputs, targets in dataloader:

            outputs = model(inputs)
            
            loss = 0
            for i, q in enumerate(model.qs):
                loss += tilted_loss(q, targets, outputs[i])
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()      
#         if epoch % 100 == 0:
#             print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}")
    
    return model


def calculate_intersections(x, y):
    intersections = []
    for i in range(len(y) - 1):
        if (y[i] < 0 and y[i + 1] > 0) or (y[i] > 0 and y[i + 1] < 0):
            x_intersection = x[i] + (x[i + 1] - x[i]) * (0 - y[i]) / (y[i + 1] - y[i])
            intersections.append(x_intersection)
    return np.mean(intersections)


def kernel(u):
    return np.exp(-0.5*(u**2))/np.sqrt((2*np.pi))

def smoothed_quantile(tau, available_tau, available_quantiles, h=0.1):
    result = []
    for point in range(len(available_quantiles[0])):
        numerator = 0
        denominator = 0
        for t in range(len(available_tau)):
            weight = kernel((tau-available_tau[t])/h)/h
            numerator += weight*available_quantiles[t][point]
            denominator += weight
        result.append(numerator/denominator)  
    return result

# Deep binary quantile regression (BQR)
def BQR(data_path, batch_is, x_cols,  y_col, attribute_index, attribute_name, latent_name, total_epochs, ndim, q=0.1):

    random_seed = 111
    lr_is = 1e-2
    all_qs = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    if q==0.05:
        all_qs = np.linspace(0.05, 0.95, 19)
    # all_qs = torch.Tensor(all_qs).to(device)
    mean_is = 0
    std_is = 1
    penalty = 1
    alpha = 0.0


    class Network(nn.Module):
        def __init__(self, indim):
            super(Network,self).__init__()
            self.fc1 = nn.Linear(indim,100)
            self.fc2 = nn.Linear(100,10)
            self.outputs = nn.Linear(10,len(all_qs))
        
        def forward(self,x):
            x = F.leaky_relu(self.fc1(x))
            x = F.leaky_relu(self.fc2(x))
            x = self.outputs(x)
            return x

  
    def create_xy(dataset, attribute_columns, target_column, delim, split_ratio, ditch_head=True):
        with open(dataset, 'r') as f:
            lines = f.readlines()
        if ditch_head:
            lines = lines[1:]
        X = []
        Y = []
        for line in lines:
            while len(line) > 0 and line[-1] == "\n":
                line = line[:len(line)-1]
            split_array = line.split(delim)
            all_columns = []
            for value in split_array:
                if value !="" and value !=" ":
                    all_columns.append(value)
            if len(all_columns)==0:
                break
            point = []
            for i in attribute_columns:
                point.append(float(all_columns[i]))
            X.append(point)
            Y.append(float(all_columns[target_column]))
        X_arr = np.asarray(X)
        X_unscaled = np.asarray(X)
        Scaler.fit(X_arr)
        X_arr = Scaler.transform(X_arr)
        Y_arr = np.asarray(Y)
        thresh = 0
        Y_arr_binary = np.where(Y_arr<=thresh,0,1)
        unique, counts = np.unique(Y_arr_binary, return_counts=True)
        x_train, x_test, y_train, y_test = train_test_split(X_arr, Y_arr_binary, test_size = split_ratio, random_state=42)
        return x_train, x_test, y_train, y_test, Y_arr, X_arr, X_unscaled
    
    
    
    # Loss and Accuracy Computation functions
    
    def cumLaplaceDistribution(y_pred,mean,standard_deviation,all_qs):
        term1 = ((1-all_qs) * (y_pred - mean))/standard_deviation
        term1.clamp_(max = 0) # Prevents NaN - Only one of term 1 or 2 is used, whichever is -ve
        lesser_term = all_qs * torch.exp(term1)
        term2 = (-1.0 * all_qs * (y_pred - mean))/standard_deviation
        term2.clamp_(max = 0) # Again, Prevents NaN
        greater_term = 1 - ((1-all_qs) * torch.exp(term2))
        mean_tensor = torch.ones_like(mean)
        y_mask = torch.div(y_pred,mean_tensor)
        y_mask[y_pred >= mean] = 1.0
        y_mask[y_pred < mean] = 0.0
        return ((1 - y_mask) * lesser_term )+  (y_mask * greater_term)
    
    
    def logLikelihoodLoss(y_true,y_pred,mean,standard_deviation,all_qs):
        new_pred = y_pred
        prob = cumLaplaceDistribution(0.0,mean = new_pred,
                                      standard_deviation = standard_deviation,all_qs = all_qs)
        prob.clamp_(min = 1e-7,max = 1 - 1e-7)
        if_one = y_true * torch.log(1 - prob)
        if_zero = (1 - y_true) * torch.log(prob)
        final_loss = - 1 * torch.mean(if_one + if_zero)
        return final_loss
    
    def customLoss(y_true, y_pred, mean, standard_deviation, all_qs, penalty):
        ind_losses = []
        for i,j in enumerate(all_qs):
            single_quantile_loss = logLikelihoodLoss(y_true[:,0],y_pred[:,i] ,
                                                     mean, standard_deviation, j)
            ind_losses.append(single_quantile_loss)
        zero = torch.Tensor([0]).to(device)
        dummy1 = y_pred[:,1:] - y_pred[:,:-1]
        dummy2 = penalty * torch.mean(torch.max(zero,-1.0 * dummy1))
        total_loss  = torch.mean(torch.stack(ind_losses)) +dummy2
        return total_loss
    
    def customTestPred(y_pred,mean,standard_deviation,all_qs,batch_size = 1):
        acc = []
        cdfs = []
        val = (y_pred - mean)/standard_deviation 
        
        for xx in range(batch_size):
            if(y_pred < mean[xx]):
                lesser_term = all_qs * torch.exp((1.0 - all_qs) * torch.tensor(val[xx], dtype=torch.double)) 
                # Typecast above needed for some versions of torch
                lesser_term  = 1 - lesser_term
                cdfs.append(lesser_term.item())
                if(lesser_term.item() >= 0.5):
                    acc.append([1])
                else:
                    acc.append([0])
            
            elif(y_pred >= mean[xx]):
                greater_term = 1.0 - ((1.0-all_qs) * torch.exp(-1.0 * all_qs * torch.tensor(val[xx], dtype=torch.double)))
                # Typecast above needed for some versions of torch
                greater_term = 1 - greater_term
                cdfs.append(greater_term.item())
                if(greater_term.item() >= 0.5):
                    acc.append([1])
                else:
                    acc.append([0])
        return torch.Tensor(acc).to(device).reshape(-1,1),torch.Tensor(cdfs).to(device).reshape(-1,1)
    
    def acc_tests(test_preds,test_labels):
        test_preds = np.array(test_preds).reshape(-1,1)
        test_labels = np.array(test_labels).reshape(-1,1)
        cdfs_acc,_ = customTestPred(0,test_preds,standard_deviation = 1,all_qs = torch.Tensor([0.5]),
                                    batch_size = test_preds.shape[0])
    
        count = 0
        for i,j in zip(cdfs_acc,test_labels):
            if(i.item() == j[0]):
                count += 1
        return count/test_labels.shape[0]
    
    
    
    # Training and Testing Methods
    def train(model,loader,epochs, verbose=False):
        optimizer = torch.optim.Adam(model.parameters(), lr = lr_is)
        train_preds_Q = []
        train_labels = []
        model.train()
        
        for i,j in enumerate(loader):
            inputs,labels = j[0],j[1]
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            op_qs = model(inputs)
            lossQ = customLoss(labels.reshape(-1,1),op_qs, mean_is,std_is,all_qs,penalty)
            lossQ.backward()
            optimizer.step()
            
            for lag in op_qs[:,int(0.5*(len(all_qs)-1))].detach().reshape(-1,1):
                train_preds_Q.append(lag.item())
            for lag in labels.reshape(-1,1):
                train_labels.append(lag.item())
                
        acc_is_Q = acc_tests(train_preds_Q,train_labels)
        
        if verbose:
            print("[%d/%d] Train Acc Q : %f "%(epochs,total_epochs,acc_is_Q))
        return acc_is_Q
    
    def test(model,loader,epochs,verbose=False):
        model.eval()
        test_preds_Q = []
        test_preds_bce = []
        test_labels = []
        with torch.no_grad():
            for i,j in enumerate(loader):
                inputs,labels = j[0],j[1]
                inputs = inputs.to(device)
                labels = labels.to(device)
                op_qs = model(inputs)
                
                for lag in op_qs[:,int(0.5*(len(all_qs)-1))].detach().reshape(-1,1):
                    test_preds_Q.append(lag.item())
                for lag in labels.reshape(-1,1):
                    test_labels.append(lag.item())
                    
        acc_is_Q = acc_tests(test_preds_Q,test_labels)
        
        if verbose:
            if (epochs+1)%5 ==0:
                print("[%d/%d] Test Acc Q : %f  "%(epochs+1,total_epochs,acc_is_Q))
        return acc_is_Q
    
    def quantileCDF(x, tau=0.5):
        if x>0:
            return 1 - tau*np.exp((tau-1)*x)
        else:
            return (1 - tau)*np.exp(tau*x)
    

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.deterministic=True
    print("Torch Device:",device)
    torch.set_default_dtype(torch.double)
    
    torch.manual_seed(random_seed)
    
    X_train,X_val,y_train,y_val, data_Y, data_X_scaled, data_X_unscaled = create_xy(data_path, x_cols, y_col, ",", 0.2)
    shap_x_train = X_train.copy()
    shap_x_val = X_val.copy()
    X_train = torch.Tensor(X_train)
    y_train = torch.Tensor(y_train)
    X_val = torch.Tensor(X_val)
    y_val = torch.Tensor(y_val)
    train_dataset = data_utils.TensorDataset(X_train, y_train)
    test_dataset = data_utils.TensorDataset(X_val, y_val)
    train_loader = data_utils.DataLoader(train_dataset, batch_size =batch_is, pin_memory=True,shuffle=True,num_workers = 1)
    test_loader = data_utils.DataLoader(test_dataset,batch_size =batch_is,pin_memory=True,shuffle = False,num_workers = 1)
    
    indim = X_train.shape[1]
    model = Network(indim)
    model = model.to(device)
    
    
    for i in range(total_epochs):
        # print("Epoch:",str(i+1))
        acc_train = train(model,train_loader,i)
        acc_test = test(model,test_loader,i,verbose=True)
    
    if ndim==1:
        new_scaler = StandardScaler()
        new_scaler.fit(data_X_unscaled)
        
        col_index = attribute_index
        attribute_array = np.arange(min(data_X_unscaled[:,col_index]), max(data_X_unscaled[:,col_index])+1, 1)
        
        test_inputs  = []
        med_values_left = []
        med_values_right = []
        
        for i in range(len(data_X_unscaled[0])):
            if i<col_index:
                med_values_left.append(np.mean(data_X_unscaled[:,i]))
            elif i>col_index:
                med_values_right.append(np.mean(data_X_unscaled[:,i]))
        
        for i in range(len(attribute_array)):
            test_inputs.append(np.concatenate([med_values_left,attribute_array[i],med_values_right], axis=None))
        
        scaled_input = new_scaler.transform(test_inputs)
        X_tens = torch.Tensor(scaled_input)
        y_tens = torch.Tensor([0]*len(scaled_input))
        
        func_dataset = data_utils.TensorDataset(X_tens, y_tens)
        func_loader = data_utils.DataLoader(func_dataset, batch_size =64, pin_memory=True, shuffle=False, num_workers = 1)
        
        
        model.eval()
        outputs = [[] for i in range(len(all_qs))]
        probs = [[] for i in range(len(all_qs))]
        avg = []
        with torch.no_grad():
            for i,j in func_loader:
                inputs,labels = i.to(device),j.to(device)
                op_qs = model(inputs)
                for itemset in op_qs.detach():
                    total = 0
                    for q in range(len(all_qs)):
                        val = itemset[q].item()
                        total+=val
                        outputs[q].append(val)
                        probs[q].append(quantileCDF(val))
                    avg.append(total/len(all_qs))
        return attribute_array, outputs

    elif ndim==2:
        new_scaler = StandardScaler()
        new_scaler.fit(data_X_unscaled)
        
        col_index = attribute_index
        attribute_array_1_ = np.arange(min(data_X_unscaled[:, col_index[0]]), max(data_X_unscaled[:, col_index[0]])+1, 1)
        attribute_array_2_ = np.arange(min(data_X_unscaled[:, col_index[1]]), max(data_X_unscaled[:, col_index[1]])+1, 1)
        attribute_array_1, attribute_array_2 = np.meshgrid(attribute_array_1_, attribute_array_2_)
        
        flat_attr_1 = attribute_array_1.flatten()
        flat_attr_2 = attribute_array_2.flatten()
        
        test_array = np.vstack((flat_attr_1, flat_attr_2)).T
        
        scaled_input = new_scaler.transform(test_array)
        X_tens = torch.Tensor(scaled_input)
        y_tens = torch.Tensor([0] * len(scaled_input))
        
        func_dataset = data_utils.TensorDataset(X_tens, y_tens)
        func_loader = data_utils.DataLoader(func_dataset, batch_size=64, pin_memory=True, shuffle=False, num_workers=1)
        
        model.eval()
        outputs = [[] for _ in range(len(all_qs))]
        probs = [[] for _ in range(len(all_qs))]
        avg = []
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        X_tens = X_tens.to(device)
        
        with torch.no_grad():
            for i, j in func_loader:
                inputs, labels = i.to(device), j.to(device)
                op_qs = model(inputs)
                for itemset in op_qs:
                    total = 0
                    for q in range(len(all_qs)):
                        val = itemset[q].item()
                        total += val
                        outputs[q].append(val)
                        probs[q].append(quantileCDF(val))
                    avg.append(total / len(all_qs))
        
        outputs = np.array(outputs)
        outputs = outputs.reshape((len(all_qs), attribute_array_1.shape[0], attribute_array_1.shape[1]))
        return attribute_array_1_, attribute_array_2_, outputs

    else:
        print("ndim should be 1 or 2!")
