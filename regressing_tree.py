#you should be lagging behind the stopping criteria condition its needs to be adaptive and user controlable
#the time complexity is also not very good comparition to other popular librarires (no wonder)
#function for predition (foget to make)(will be pending until next version come)
#last thing -> this is shit! (no wonder)

import numpy
import pandas
import copy

class Regression_tree:
    '''criteria = sum of sqaured residual(/squared_error)'''
    def __init__(self):
        self.state_dict = {'feature' : None, 'value' : None, 'inside' : None, 'leaf' : None, 'SSR' : None}
        #self.data_dict = {'l_D' : None, 'r_D' : None, 'SSR' : None, 'Err' : None}
        self._sd  = copy.deepcopy(self.state_dict)
        self.stopping_criteria = {'nof_ex_wrg' : 10, 'SSR' : 0, 'depth' : 5}
    
    def fit(self, data : numpy.array, label : numpy.array) -> None:
        self.data = pandas.DataFrame(data)
        self.label = pandas.DataFrame(label)
        assert self.data.shape[0] == self.label.shape[0]
        assert self.label.shape[1] == 1
        assert len(self.data.shape) == 2
        assert len(self.label.shape) == 2
        self.all_data = pandas.concat((self.data, self.label), axis=1).reset_index(drop=True)
        self.num_ftr = self.data.shape[-1]
        self.node(dict_ = self.state_dict, data = self.all_data.index, depth_ = 0)
        del self.all_data, self.num_ftr, self.data, self.label
        #del self.node, self.stopping_criteria, self.tools
        return
    
    def node(self, dict_ : dict, data : numpy.array, depth_ : int, track : dict = None) -> None:
        if self.tools(call = 'chk_stp', depth = depth_):return
        data = self.all_data.iloc[data, :]
        track = self.parallel(data_ = data)
        in_00 = track['l_s']
        in_01 = track['r_s']
        dict_['feature'] = track['feature']
        dict_['value'] = track['threshold']
        dict_['SSR'] = track['SSR']
        dict_['inside'] = list()
        dict_['leaf'] = False
        #root stopping criteria #SSR_value
        if self.tools(call = 'chk_stp', depth = depth_, SSR = dict_['SSR']) and depth_==0:
            #dict_['feature'] = None
            dict_['value'] = numpy.mean(data.iloc[:, -1].values)
            dict_['leaf'] == True
            #dict_['inside'] = None
            del dict_['feature']
            del dict_['inside']
            del dict_['SSR']
            return
        
        dict_['inside'].append(list())
        dict_['inside'][0] = copy.deepcopy(self._sd)
        if(len(self.all_data.iloc[in_00, -1].unique()) == 1):
            #dict_['inside'][0]['feature'] = None
            dict_['inside'][0]['value'] = numpy.mean(self.all_data.iloc[in_00, -1].values)
            dict_['inside'][0]['leaf'] = True
            #dict_['inside'][0]['inside'] = None
            del dict_['inside'][0]['inside']
            del dict_['inside'][0]['feature']
            del dict_['inside'][0]['SSR']
        else: self.node(dict_['inside'][0], data = in_00, depth_= depth_ + 1)
        dict_['inside'].append(list())
        dict_['inside'][1] = copy.deepcopy(self._sd)
        
        if(len(self.all_data.iloc[in_01, -1].unique()) == 1):
            #dict_['inside'][1]['feature'] = None
            dict_['inside'][1]['value'] = numpy.mean(self.all_data.iloc[in_01, -1].values)
            dict_['inside'][1]['leaf'] = True
            #dict_['inside'][1]['inside'] = None
            del dict_['inside'][1]['feature']
            del dict_['inside'][1]['inside']
            del dict_['inside'][1]['SSR']   
        else: self.node(dict_['inside'][1], data = in_01, depth_= depth_ + 1)
        return
        
    def parallel(self, data_) -> dict:
        SSR_ = list()
        SSR_ftr = list()#contain the bst threshold and value of its SSR of all the feature 
        SSR_top = dict()#contain the bst threshold and bst feature among all (and lr split)
        for _0 in range(self.num_ftr):
            #BELOW CODE WILL BELONG TO FUNCTION FOR FUTHER PARLLEL COMPUTING
            d_tmp = data_.iloc[:, _0]
            threshold = self.tools(call = 'threshold', data = d_tmp)
            if len(threshold) == 0: continue
            for _1 in threshold:
                _, _, val = self.tools(call = 'split_SSR', condition = _1, data = d_tmp)
                SSR_.append(val)
            SSR_ = numpy.array(SSR_)
            in_0, in_1 = threshold[numpy.where(SSR_ == SSR_.min())], SSR_.min()
            SSR_ftr.append({'feature' : _0, 'threshold' : in_0[0], 'SSR' : in_1})
            SSR_ = list()
        tmp = list()
        for _2 in range(len(SSR_ftr)): tmp.append(SSR_ftr[_2]['SSR'])
        in_3 = numpy.where(numpy.array(tmp) == min(tmp))[0][0]
        SSR_top = SSR_ftr[in_3]
        in_4 = SSR_top['threshold']
        d_tmp = data_.iloc[:, SSR_top['feature']]
        in_5, in_6, _ = self.tools(call='split_SSR', condition = in_4, data = d_tmp)
        assert _ == SSR_top['SSR']
        SSR_top.update({'l_s' : in_5, 'r_s' : in_6})
        return SSR_top
    
    def tools(self, call : str = None, **kwargs):
        def threshold(kwargs) -> numpy.array:
            data = kwargs['data']
            arr1 = pandas.Series(data, index=None)
            arr = numpy.sort(arr1.unique()).reshape(-1, 1)
            arr = numpy.concatenate((arr[:-1], arr[1:]), axis=1)
            threshold = arr.mean(axis=1)
            return threshold
        def split_SSR(self, kwargs) -> tuple:
            i = kwargs['condition']
            data = kwargs['data']
            #left and right split of data(index)
            l_D = data.where(data<=i).dropna().index.values
            r_D = data.where(data>i).dropna().index.values
            #prediction and actual value for left and right split
            pred_l = numpy.mean(self.all_data.iloc[l_D, -1].values)
            orig_l = self.all_data.iloc[l_D, -1].values
            pred_r = numpy.mean(self.all_data.iloc[r_D, -1].values)
            orig_r = self.all_data.iloc[r_D, -1].values
            #calculating SSR value
            val = numpy.sum(numpy.square(orig_l - pred_l))
            val = val + numpy.sum(numpy.square(orig_r - pred_r))
            return (l_D, r_D, val)
        def chk_stp(kwargs) -> bool:
            depth_ = kwargs['depth']
            try: 
                SSR = kwargs['SSR']
                if SSR == 0: return True
            except: pass
            if depth_ == self.stopping_criteria['depth']: return True
            return False
        if call == None: return
        elif call == 'chk_stp': return chk_stp(kwargs)
        elif call == 'threshold': return threshold(kwargs)
        elif call == 'split_SSR': return split_SSR(self, kwargs)
    
    def plot_tree(self) -> None:
        def display(data_, depth=0):
            if data_['leaf'] == True: 
                print('|\t'*depth, "|---\033[1;32mpredict : ", data_['value'], sep='', end='\033[0;30m\n')
                return
            print('|\t'*depth, f"|---\033[0;31mfeature_{data_['feature']} <= {data_['value']}", sep='', end='\033[0;30m\n')
            display(data_['inside'][0], depth+1)
            print('|\t'*depth, f"|---\033[0;31mfeature_{data_['feature']} > {data_['value']}", sep='', end='\033[0;30m\n')
            display(data_['inside'][1], depth+1)
        display(self.state_dict)
